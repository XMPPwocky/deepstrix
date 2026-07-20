//! MTP HC-ablation direct test (2026-07).
//!
//! THE DIRECT TEST for the "HC drift limits MTP acceptance" hypothesis.
//!
//! Teacher-forces the main model along the CPU-canonical token sequence
//! (the sequence captured in reference/v4flash-cpu-activations). At each
//! position P it runs the MTP drafter THREE times, each fed a different
//! `prev_hc`, and measures greedy draft-acceptance (drafter argmax ==
//! main-model argmax at P, both predicting P+1):
//!
//!   A. our-HC    — the GPU main model's own final-layer HC (residual_next)
//!   B. dump-HC   — the CPU-canonical final-layer HC (L42 layer_output_residual)
//!   C. our-HC×2.5 — magnitude-scaled our-HC (RMS-norm invariance sanity)
//!
//! If B raises acceptance toward the reference ~0.85, HC alignment is the
//! lever. If A≈B, our GPU HC already matches canonical → the gap is NOT a
//! GPU-vs-canonical HC bug. C should EXACTLY equal A because the drafter
//! per-row RMS-norms prev_hc immediately (scale-invariant) — this refutes
//! any "magnitude alignment" fix.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test mtp_hc_ablation -- --ignored --nocapture

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
const ROPE_ORIG_CTX: u64 = 65536;
/// Last transformer layer index (N_LAYER=43 → layers 0..42).
const LAST_LAYER: i32 = 42;

// Canonical token sequence from reference/v4flash-cpu-activations/tokens.json
// (7 prompt + 50 CPU-greedy generated = 57 tokens, dump tokens 0..56).
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const GEN_TOKENS: [i32; 50] = [
    260, 1017, 9353, 294, 8281, 10192, 39, 940, 15890, 13523, 13973, 14, 418, 270, 5304, 304, 1699,
    103345, 1951, 305, 10559, 3051, 14, 305, 2123, 270, 103345, 4647, 2019, 16, 455, 2645, 14449,
    4346, 890, 304, 223, 7833, 45, 35977, 305, 5238, 890, 304, 223, 26, 45, 35977, 16, 455,
];

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

fn argmax(x: &[f32]) -> i32 {
    let mut best = 0i32;
    let mut bestv = x[0];
    for (i, &v) in x.iter().enumerate().skip(1) {
        if v > bestv {
            bestv = v;
            best = i as i32;
        }
    }
    best
}

/// rel-RMS(a, b) = ||a-b|| / ||b||  (global).
fn rel_rms(a: &[f32], b: &[f32]) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        num += ((x - y) as f64).powi(2);
        den += (y as f64).powi(2);
    }
    (num / den.max(1e-30)).sqrt() as f32
}

fn l2(x: &[f32]) -> f32 {
    (x.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>()).sqrt() as f32
}

/// per-row (N_HC rows of N_EMBD) normalized directional rel-RMS. Removes
/// each row's scale first (what the drafter's hnorm actually sees).
fn rowwise_normalized_rel_rms(a: &[f32], b: &[f32], n_hc: usize, n_embd: usize) -> f32 {
    let mut num = 0f64;
    let mut den = 0f64;
    for r in 0..n_hc {
        let ar = &a[r * n_embd..(r + 1) * n_embd];
        let br = &b[r * n_embd..(r + 1) * n_embd];
        let an = l2(ar).max(1e-20);
        let bn = l2(br).max(1e-20);
        for i in 0..n_embd {
            let x = ar[i] / an;
            let y = br[i] / bn;
            num += ((x - y) as f64).powi(2);
            den += (y as f64).powi(2);
        }
    }
    (num / den.max(1e-30)).sqrt() as f32
}

#[test]
#[ignore]
fn mtp_hc_ablation() -> eyre::Result<()> {
    install_panic_handler()?;

    let mut tokens: Vec<i32> = PROMPT_TOKENS.to_vec();
    tokens.extend_from_slice(&GEN_TOKENS);
    let max_pos_env: usize = std::env::var("ABLATION_POS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(tokens.len() - 1);
    // We compare at pos where main predicts pos+1; need tokens[pos] and a
    // reference "next" — and dump HC at pos. Cap to tokens.len()-1.
    let n_pos = max_pos_env.min(tokens.len() - 1);
    eprintln!("MTP HC ablation: {n_pos} positions, seq_len={}", tokens.len());

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
    // Three independent MTP KV worlds (A=our, B=dump, C=scaled).
    let mut state_a = MtpLayerState::alloc(dgpu)?;
    let mut state_b = MtpLayerState::alloc(dgpu)?;
    let mut state_c = MtpLayerState::alloc(dgpu)?;
    let total_positions = (tokens.len() + 2) as u32;
    let mut state = HetModelState::alloc(dgpu, igpu, total_positions)?;

    let mut our_hc = DeviceBuffer::<f32>::new(dgpu.id, HC_DIM as usize)?;
    let mut dump_hc = DeviceBuffer::<f32>::new(dgpu.id, HC_DIM as usize)?;
    let mut scaled_hc = DeviceBuffer::<f32>::new(dgpu.id, HC_DIM as usize)?;
    let mut main_logits = vec![0f32; N_VOCAB as usize];
    let mut mtp_logits = vec![0f32; N_VOCAB as usize];
    let mut our_hc_host = vec![0f32; HC_DIM as usize];

    let mut hits_a = 0u32;
    let mut hits_b = 0u32;
    let mut hits_c = 0u32;
    let mut total = 0u32;
    let mut sum_relrms = 0f64;
    let mut sum_relrms_norm = 0f64;
    let mut sum_magratio = 0f64;
    let mut main_matches_dump = 0u32;

    let run_draft = |engine: &HeterogeneousEngine,
                     dgpu_scratch: &mut DgpuScratch,
                     igpu_scratch: &mut IgpuScratch,
                     mtp_scratch: &mut MtpScratch,
                     st: &mut MtpLayerState,
                     prev: &DeviceBuffer<f32>,
                     embd: &[f32],
                     pos: u32,
                     token: i32,
                     out: &mut [f32]|
     -> eyre::Result<i32> {
        engine.forward_mtp_draft(
            dgpu_scratch,
            igpu_scratch,
            mtp_scratch,
            st,
            &main_weights.global,
            &mtp_weights,
            prev,
            embd,
            pos,
            token,
        )?;
        mtp_scratch.mtp_logits.copy_to_host(out)?;
        Ok(argmax(out))
    };

    eprintln!("\npos tok  main  A(our) B(dump) C(x2.5) | relRMS relRMS_dir magR");
    for pos in 0..n_pos {
        let p = pos as u32;
        let token = tokens[pos];
        let embd = embed_cache.lookup(token);
        let input_hc = broadcast_to_hc(&embd, N_HC as usize);

        // Main forward (teacher-forced on canonical sequence).
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights,
            &input_hc,
            p,
            token,
        )?;
        dgpu_scratch.logits.copy_to_host(&mut main_logits)?;
        let main_top = argmax(&main_logits);
        // Does GPU main agree with the CPU-canonical greedy? (alignment check)
        if pos + 1 < tokens.len() && main_top == tokens[pos + 1] {
            main_matches_dump += 1;
        }

        // Snapshot our GPU final-layer HC.
        our_hc.copy_from_buffer(&dgpu_scratch.residual_next)?;
        our_hc.copy_to_host(&mut our_hc_host)?;

        // Load canonical dump HC (L42 layer_output_residual @ token=pos).
        let ent = dump
            .tensor("layer_output_residual", LAST_LAYER, pos as i32)
            .ok_or_else(|| eyre!("missing dump HC L{LAST_LAYER} T{pos}"))?;
        let dump_host = dump.read_f32(ent)?;
        if dump_host.len() != HC_DIM as usize {
            return Err(eyre!(
                "dump HC len {} != HC_DIM {HC_DIM}",
                dump_host.len()
            ));
        }
        dump_hc.copy_from_host(&dump_host)?;

        // Scaled copy of our HC.
        let scaled_host: Vec<f32> = our_hc_host.iter().map(|&v| v * 2.5).collect();
        scaled_hc.copy_from_host(&scaled_host)?;

        // Divergence metrics (our vs canonical dump).
        let rr = rel_rms(&our_hc_host, &dump_host);
        let rrn = rowwise_normalized_rel_rms(&our_hc_host, &dump_host, N_HC as usize, N_EMBD as usize);
        let magr = l2(&our_hc_host) / l2(&dump_host).max(1e-20);
        sum_relrms += rr as f64;
        sum_relrms_norm += rrn as f64;
        sum_magratio += magr as f64;

        // Three drafter runs.
        let a = run_draft(&engine, &mut dgpu_scratch, &mut igpu_scratch, &mut mtp_scratch, &mut state_a, &our_hc, &embd, p, token, &mut mtp_logits)?;
        let b = run_draft(&engine, &mut dgpu_scratch, &mut igpu_scratch, &mut mtp_scratch, &mut state_b, &dump_hc, &embd, p, token, &mut mtp_logits)?;
        let c = run_draft(&engine, &mut dgpu_scratch, &mut igpu_scratch, &mut mtp_scratch, &mut state_c, &scaled_hc, &embd, p, token, &mut mtp_logits)?;

        if a == main_top { hits_a += 1; }
        if b == main_top { hits_b += 1; }
        if c == main_top { hits_c += 1; }
        total += 1;

        eprintln!(
            "{:>3} {:>5} {:>6} {:>6} {:>6} {:>6} | {:.3} {:.3} {:.3}",
            pos, token, main_top, a, b, c, rr, rrn, magr
        );
    }

    eprintln!("\n===== HC ABLATION RESULTS ({total} positions) =====");
    eprintln!(
        "A our-HC    accept: {:>3}/{:>3} = {:>5.1}%",
        hits_a, total, 100.0 * hits_a as f64 / total as f64
    );
    eprintln!(
        "B dump-HC   accept: {:>3}/{:>3} = {:>5.1}%",
        hits_b, total, 100.0 * hits_b as f64 / total as f64
    );
    eprintln!(
        "C our-HCx2.5 accept:{:>3}/{:>3} = {:>5.1}%  (must equal A: rms-norm invariance)",
        hits_c, total, 100.0 * hits_c as f64 / total as f64
    );
    eprintln!(
        "\nGPU main vs CPU-canonical greedy agreement: {}/{} = {:.1}%",
        main_matches_dump, total, 100.0 * main_matches_dump as f64 / total as f64
    );
    eprintln!(
        "HC divergence our-vs-dump: rel-RMS avg={:.4}  rowwise-normalized-rel-RMS avg={:.4}  ||our||/||dump|| avg={:.3}",
        sum_relrms / total as f64,
        sum_relrms_norm / total as f64,
        sum_magratio / total as f64
    );

    engine.shutdown()?;
    Ok(())
}
