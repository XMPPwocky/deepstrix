//! MTP oracle (restored M40, adapted 2026-07): at each position P during
//! normal single-token decode, ask MTP to predict the token at P+1 and
//! compare against the main model's argmax at P (which also predicts P+1).
//! High agreement ⇒ MTP correctly implemented + a good greedy draft-accept.
//!
//! Reports top-1 / top-3 / top-10 hit rate (MTP argmax vs main argmax) and
//! the expected speculative-sampling accept probability (1 - TVD(p, q)).
//!
//! Scope: DRAFTER ONLY — pure single-token decode with MTP shadowing; no
//! verify pair-forward, no accept/reject loop.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test mtp_oracle -- --ignored --nocapture
//!
//! Env: ORACLE_TOKENS (default 30)

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{HC_DIM, N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
    MtpLayerState, MtpScratch, MtpWeights,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const MTP_MODEL_PATH: &str = "/persist/lumi/models/DeepSeek-V4-Flash-MTP-Q4K-Q8_0-F32.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}
fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

struct EmbedCache {
    bytes: Vec<u8>,
    n_embd: u32,
}
impl EmbedCache {
    fn load(gguf: &MappedGguf, n_embd: u32) -> eyre::Result<Self> {
        let t = gguf
            .gguf()
            .tensor("token_embd.weight")
            .ok_or_else(|| eyre!("token_embd.weight missing"))?;
        if t.dtype != GgufType::F16 {
            return Err(eyre!("token_embd dtype != F16"));
        }
        let bytes = gguf.read_tensor(t)?;
        Ok(Self { bytes, n_embd })
    }
    fn lookup(&self, token_id: i32) -> Vec<f32> {
        let row_bytes = (self.n_embd as usize) * 2;
        let off = (token_id as usize) * row_bytes;
        let row = &self.bytes[off..off + row_bytes];
        let mut out = vec![0f32; self.n_embd as usize];
        for i in 0..self.n_embd as usize {
            let bits = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
            out[i] = v4flash_kernels::iq2_xxs_tables::f16_to_f32(bits);
        }
        out
    }
}

fn broadcast_to_hc(embd: &[f32], n_hc: usize) -> Vec<f32> {
    let n = embd.len();
    let mut out = vec![0.0f32; n_hc * n];
    for h in 0..n_hc {
        out[h * n..(h + 1) * n].copy_from_slice(embd);
    }
    out
}

fn argmax(x: &[f32]) -> (i32, f32) {
    let mut best = 0i32;
    let mut bestv = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > bestv {
            bestv = v;
            best = i as i32;
        }
    }
    (best, bestv)
}

fn topk(x: &[f32], k: usize) -> Vec<(i32, f32)> {
    let mut idx: Vec<(i32, f32)> = x
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as i32, v))
        .collect();
    idx.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    idx.truncate(k);
    idx
}

fn softmax_prob_at(x: &[f32], target_idx: i32) -> f32 {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut denom = 0.0f64;
    for &v in x {
        denom += (((v - m) as f64).exp()).min(1e30);
    }
    let num = (((x[target_idx as usize] - m) as f64).exp()).min(1e30);
    (num / denom).max(0.0).min(1.0) as f32
}

fn softmax_full_f64(logits: &[f32]) -> Vec<f64> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits
        .iter()
        .map(|&v| (((v - m) as f64).exp()).min(1e30))
        .collect();
    let denom: f64 = probs.iter().sum();
    if denom > 0.0 {
        for p in probs.iter_mut() {
            *p /= denom;
        }
    }
    probs
}

/// Expected speculative-sampling accept prob for one drafted token:
/// sum_X min(p_target(X), q_draft(X)) = 1 - TVD(p, q).
fn spec_sampling_accept_prob(p_target: &[f64], q_draft: &[f64]) -> f64 {
    let mut acc = 0.0f64;
    for (pt, qd) in p_target.iter().zip(q_draft.iter()) {
        acc += pt.min(*qd);
    }
    acc.max(0.0).min(1.0)
}

#[test]
#[ignore]
fn mtp_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_tokens: usize = std::env::var("ORACLE_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    eprintln!(
        "MTP oracle: n_tokens={n_tokens}, prompt_len={}",
        PROMPT_TOKENS.len()
    );

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let mtp_gguf = MappedGguf::open(MTP_MODEL_PATH)?;
    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };
    let rope = rope_for_layer(0)?;

    eprintln!("loading main weights...");
    let main_weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loading MTP weights...");
    let mtp_weights = MtpWeights::load(&mtp_gguf, dgpu, igpu, rope)?;
    eprintln!("loading token_embd cache...");
    let embed_cache = EmbedCache::load(&main_gguf, N_EMBD)?;

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut mtp_scratch = MtpScratch::alloc(dgpu)?;
    let mut mtp_state = MtpLayerState::alloc(dgpu)?;
    let total_positions = (PROMPT_TOKENS.len() + n_tokens + 2) as u32;
    let mut state = HetModelState::alloc(dgpu, igpu, total_positions)?;

    let mut prev_hc_buf = DeviceBuffer::<f32>::new(dgpu.id, HC_DIM as usize)?;
    prev_hc_buf.fill_zero()?;
    let mut main_logits = vec![0f32; N_VOCAB as usize];
    let mut mtp_logits = vec![0f32; N_VOCAB as usize];

    let mut hits_top1: u32 = 0;
    let mut hits_top3: u32 = 0;
    let mut hits_top10: u32 = 0;
    let mut total: u32 = 0;
    let mut mtp_prob_at_main: Vec<f32> = Vec::new();
    let mut rs_accept_probs: Vec<f64> = Vec::new();
    let mut trace: Vec<(usize, i32, i32, f32, f32, f32, f64)> = Vec::new();

    let mut cur_token: i32 = PROMPT_TOKENS[0];
    for step in 0..(PROMPT_TOKENS.len() + n_tokens) {
        let pos = step as u32;
        let token = cur_token;
        let embd = embed_cache.lookup(token);
        let input_hc = broadcast_to_hc(&embd, N_HC as usize);

        // 1. Main forward.
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights,
            &input_hc,
            pos,
            token,
        )?;
        dgpu_scratch.logits.copy_to_host(&mut main_logits)?;
        let (main_top, main_top_v) = argmax(&main_logits);
        let main_top_prob = softmax_prob_at(&main_logits, main_top);

        // 2. Snap the main model's final-layer HC at pos (residual_next
        //    after forward_token's epilogue swap).
        prev_hc_buf.copy_from_buffer(&dgpu_scratch.residual_next)?;

        // 3. MTP draft from main HC at pos.
        engine.forward_mtp_draft(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut mtp_scratch,
            &mut mtp_state,
            &main_weights.global,
            &mtp_weights,
            &prev_hc_buf,
            &embd,
            pos,
            token,
        )?;
        mtp_scratch.mtp_logits.copy_to_host(&mut mtp_logits)?;
        let (mtp_top, mtp_top_v) = argmax(&mtp_logits);
        let mtp_at_main = softmax_prob_at(&mtp_logits, main_top);
        let mtp_top_prob = softmax_prob_at(&mtp_logits, mtp_top);

        let mtp_top3 = topk(&mtp_logits, 3);
        let mtp_top10 = topk(&mtp_logits, 10);
        if mtp_top == main_top {
            hits_top1 += 1;
        }
        if mtp_top3.iter().any(|(i, _)| *i == main_top) {
            hits_top3 += 1;
        }
        if mtp_top10.iter().any(|(i, _)| *i == main_top) {
            hits_top10 += 1;
        }
        total += 1;
        mtp_prob_at_main.push(mtp_at_main);

        let p_target = softmax_full_f64(&main_logits);
        let q_draft = softmax_full_f64(&mtp_logits);
        let rs_accept = spec_sampling_accept_prob(&p_target, &q_draft);
        rs_accept_probs.push(rs_accept);

        trace.push((
            pos as usize,
            main_top,
            mtp_top,
            mtp_at_main,
            main_top_prob,
            mtp_top_prob,
            rs_accept,
        ));

        cur_token = if step + 1 < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[step + 1]
        } else {
            main_top
        };

        let _ = main_top_v;
        let _ = mtp_top_v;
    }

    eprintln!("\n===== MTP ORACLE RESULTS =====");
    eprintln!("positions: {total}");
    eprintln!(
        "top-1 hit:  {:>4}/{:>4} = {:>5.1}%  (MTP argmax == main argmax → greedy draft-accept)",
        hits_top1,
        total,
        100.0 * hits_top1 as f64 / total as f64
    );
    eprintln!(
        "top-3 hit:  {:>4}/{:>4} = {:>5.1}%",
        hits_top3,
        total,
        100.0 * hits_top3 as f64 / total as f64
    );
    eprintln!(
        "top-10 hit: {:>4}/{:>4} = {:>5.1}%",
        hits_top10,
        total,
        100.0 * hits_top10 as f64 / total as f64
    );
    let avg_prob = mtp_prob_at_main.iter().sum::<f32>() / total as f32;
    let mut sorted = mtp_prob_at_main.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p50 = sorted[sorted.len() / 2];
    eprintln!(
        "MTP softmax-prob @ MAIN's argmax: avg={:.4}  p50={:.4}",
        avg_prob, p50
    );

    let rs_avg = rs_accept_probs.iter().sum::<f64>() / total as f64;
    let mut rs_sorted = rs_accept_probs.clone();
    rs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let rs_min = *rs_sorted.first().unwrap();
    let rs_max = *rs_sorted.last().unwrap();
    let rs_p50 = rs_sorted[total as usize / 2];
    eprintln!("\n===== REJECTION-SAMPLING ACCEPT (expected) =====");
    eprintln!(
        "per-position accept prob: avg={:.4}  min={:.4}  p50={:.4}  max={:.4}",
        rs_avg, rs_min, rs_p50, rs_max
    );
    eprintln!(
        "  → expected acceptance rate across {total} positions = {:.1}%",
        rs_avg * 100.0
    );

    eprintln!("\n===== TRAJECTORY =====");
    eprintln!("pos  main_top  mtp_top  match  main_p   mtp_at_main  mtp_top_p  rs_accept");
    for (pos, main_top, mtp_top, mtp_at_main, main_top_prob, mtp_top_prob, rs_accept) in
        trace.iter().take(40)
    {
        let m = if main_top == mtp_top { '+' } else { '-' };
        eprintln!(
            "{:>3}  {:>8}  {:>7}  {}     {:.4}   {:.4}        {:.4}    {:.4}",
            pos, main_top, mtp_top, m, main_top_prob, mtp_at_main, mtp_top_prob, rs_accept
        );
    }

    engine.shutdown()?;
    Ok(())
}
