//! Rejection-sampling (speculative-sampling) MTP spec-decode QUALITY GATE.
//!
//! The production recipe is T=1.0 multinomial sampling (min_p=0), and the user
//! cares about output-DISTRIBUTION quality (KL/TVD), not greedy-exactness.
//! Rejection sampling draws every accepted token from the exact target
//! distribution `p`, so the committed stream is distributed as the target
//! model itself — up to the batch-vs-seq KV drift the B=2 verify introduces.
//!
//! This test, with a FIXED seed:
//!   1. Runs `spec_decode_sample_run` for N tokens, recording per committed
//!      token the target distribution it was drawn from.
//!   2. Teacher-forces that same committed stream through the plain B=1
//!      `forward_token` decode path, recording the plain next-token
//!      distribution at each position. Same prefix ⇒ the two distributions
//!      differ ONLY by batch-vs-seq drift.
//!   3. Reports acceptance rate, mean/max KL(plain‖spec) + TVD (the drift's
//!      effect on quality), and honest e2e tok/s vs a plain multinomial decode.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 \
//!   DGPU_HOT_EXPERTS=8 \
//!   DGPU_HOT_EXPERTS_FILE=/home/claude-code/deepstrix/reference/decode_hot_experts.txt \
//!   SPEC_TOKENS=48 SPEC_SEED=1234 \
//!     nix develop -c cargo test --release -p v4flash-kernels \
//!       --test spec_decode_sampling -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    softmax_dist, BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState,
    HetModelWeights, HeterogeneousEngine, IgpuScratch, MtpLayerState, MtpScratch, MtpWeights,
    SpecSampleConfig,
};
use v4flash_kernels::sampler::SamplerRng;
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

/// Host-side F16 `token_embd` row cache → F32 rows (main model).
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

fn broadcast_hc(embd: &[f32]) -> Vec<f32> {
    let n = embd.len();
    let mut out = vec![0f32; (N_HC as usize) * n];
    for h in 0..N_HC as usize {
        out[h * n..(h + 1) * n].copy_from_slice(embd);
    }
    out
}

fn sample_from_dist(d: &[f32], u: f32) -> i32 {
    let mut r = u as f64;
    for (i, &p) in d.iter().enumerate() {
        r -= p as f64;
        if r <= 0.0 {
            return i as i32;
        }
    }
    for i in (0..d.len()).rev() {
        if d[i] > 0.0 {
            return i as i32;
        }
    }
    0
}

/// KL(a‖b) in nats, guarding b's zeros with a floor.
fn kl(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0f64;
    for i in 0..a.len() {
        let ai = a[i] as f64;
        if ai > 0.0 {
            let bi = (b[i] as f64).max(1e-12);
            s += ai * (ai / bi).ln();
        }
    }
    s.max(0.0)
}

/// Total-variation distance = 0.5 * L1.
fn tvd(a: &[f32], b: &[f32]) -> f64 {
    let mut s = 0f64;
    for i in 0..a.len() {
        s += (a[i] as f64 - b[i] as f64).abs();
    }
    0.5 * s
}

/// Residual `norm(max(0, p-q))` (mirrors the private one in spec_decode.rs).
fn residual(p: &[f32], q: &[f32]) -> Option<Vec<f32>> {
    let mut r = vec![0f32; p.len()];
    let mut z = 0f64;
    for i in 0..p.len() {
        let d = p[i] - q[i];
        if d > 0.0 {
            r[i] = d;
            z += d as f64;
        }
    }
    if z <= 1e-9 {
        return None;
    }
    for x in r.iter_mut() {
        *x = (*x as f64 / z) as f32;
    }
    Some(r)
}

/// CPU-only proof that the accept rule (draw x~q, accept w.p. min(1,p/q),
/// else residual-resample) emits tokens distributed EXACTLY as the target p.
/// This validates `softmax_dist` + the accept/residual math authored here,
/// with no GPU/model. It is the mathematical-correctness half of the gate.
#[test]
fn rejection_sampling_marginal_matches_target() {
    let mut rng = SamplerRng::new(0xC0FFEE);
    // A few random (target, draft) logit pairs over a small vocab.
    for trial in 0..6 {
        let vocab = 12;
        let mut plog = vec![0f32; vocab];
        let mut qlog = vec![0f32; vocab];
        for i in 0..vocab {
            plog[i] = rng.next_f32() * 6.0 - 3.0;
            qlog[i] = rng.next_f32() * 6.0 - 3.0;
        }
        let p = softmax_dist(&plog, 1.0, 0.0);
        let q = softmax_dist(&qlog, 1.0, 0.0);

        let n = 400_000usize;
        let mut hist = vec![0u64; vocab];
        for _ in 0..n {
            let x = sample_from_dist(&q, rng.next_f32()) as usize;
            let ratio = if q[x] > 0.0 { p[x] as f64 / q[x] as f64 } else { f64::INFINITY };
            if (rng.next_f32() as f64) < ratio.min(1.0) {
                hist[x] += 1; // accept
            } else {
                let emit = match residual(&p, &q) {
                    Some(res) => sample_from_dist(&res, rng.next_f32()),
                    None => sample_from_dist(&p, rng.next_f32()),
                } as usize;
                hist[emit] += 1;
            }
        }
        let mut max_err = 0f64;
        for i in 0..vocab {
            let emp = hist[i] as f64 / n as f64;
            max_err = max_err.max((emp - p[i] as f64).abs());
        }
        // Sampling error at n=4e5 is ~1/sqrt(n)≈1.6e-3; allow generous slack.
        assert!(
            max_err < 8e-3,
            "trial {trial}: emitted marginal deviates from target p by {max_err:.4} (> 8e-3) — accept/residual math wrong"
        );
        eprintln!("  [rej-sampling trial {trial}] max|emp - p| = {max_err:.5}");
    }
}

#[test]
#[ignore]
fn spec_decode_sampling_quality() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_tokens: usize = std::env::var("SPEC_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(48);
    let seed: u64 = std::env::var("SPEC_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1234);
    let temperature: f32 = std::env::var("SPEC_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let min_p: f32 = std::env::var("SPEC_MIN_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    eprintln!(
        "spec_decode_sampling: n_tokens={n_tokens} seed={seed} T={temperature} min_p={min_p} prompt_len={}",
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

    eprintln!("loading main weights (~90s)...");
    let weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    eprintln!("loading MTP weights...");
    let mtp_weights = MtpWeights::load(&mtp_gguf, dgpu, igpu, rope)?;
    eprintln!("loading token_embd cache...");
    let embed_cache = EmbedCache::load(&main_gguf, N_EMBD)?;
    eprintln!("weights loaded.");

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;
    let mut bd = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi = BatchIgpuScratch::alloc(igpu)?;
    let mut mtp_scratch = MtpScratch::alloc(dgpu)?;

    let n_kv_max = (PROMPT_TOKENS.len() + n_tokens + 8) as u32;
    let embed = |tok: i32| embed_cache.lookup(tok);
    let vocab = N_VOCAB as usize;

    // ===================== REJECTION-SAMPLING spec decode =====================
    eprintln!("\n########## SPEC decode REJECTION-SAMPLING ##########");
    let (spec_out, spec_wall) = {
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut frontier = state.alloc_frontier(dgpu)?;
        let mut mtp_state = MtpLayerState::alloc(dgpu)?;
        let t0 = Instant::now();
        let out = engine.spec_decode_sample_run(
            &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
            &mut state, &weights, &mut mtp_scratch, &mut mtp_state, &mtp_weights, &mut frontier,
            &PROMPT_TOKENS, &embed,
            SpecSampleConfig { n_tokens, temperature, min_p, seed, collect_dists: true },
        )?;
        (out, t0.elapsed().as_secs_f64())
    };
    let s = &spec_out.stats;
    eprintln!("committed ({}): {:?}", spec_out.tokens.len(), spec_out.tokens);
    eprintln!("\n===== ACCEPTANCE =====");
    eprintln!("  rounds:            {}", s.rounds);
    eprintln!("  committed:         {}", s.committed);
    eprintln!(
        "  x1 accept:         {}/{} = {:.3}",
        s.draft1_accept, s.draft1,
        s.draft1_accept as f64 / s.draft1.max(1) as f64
    );
    eprintln!(
        "  x2 accept:         {}/{} = {:.3}",
        s.draft2_accept, s.draft2,
        s.draft2_accept as f64 / s.draft2.max(1) as f64
    );
    eprintln!("  bonus tokens:      {}", s.bonus);
    eprintln!("  overall accept:    {:.3}  (accepted drafts / total drafts)", s.accept_rate());
    eprintln!("  tokens / round:    {:.3}", s.tokens_per_round());
    eprintln!("  wall:              {spec_wall:.3} s");
    eprintln!("  >>> spec tok/s:    {:.2}", s.committed as f64 / spec_wall);

    // ============ QUALITY GATE: teacher-force spec stream, plain B=1 ============
    // Same committed prefix through the plain single-token decode path; the
    // per-position distribution differs from spec's ONLY by batch-vs-seq drift.
    eprintln!("\n########## QUALITY GATE: plain teacher-forced dists ##########");
    let mut plain_dists: Vec<Vec<f32>> = Vec::with_capacity(n_tokens);
    {
        engine.clear_graphs();
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut logits = vec![0f32; vocab];
        for (pos, &tok) in PROMPT_TOKENS.iter().enumerate() {
            let inp = broadcast_hc(&embed(tok));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos as u32, tok,
            )?;
        }
        dgpu_scratch.logits.copy_to_host(&mut logits)?;
        plain_dists.push(softmax_dist(&logits, temperature, min_p)); // dist for token[0]
        let mut pos = PROMPT_TOKENS.len() as u32;
        for i in 0..n_tokens - 1 {
            let cur = spec_out.tokens[i];
            let inp = broadcast_hc(&embed(cur));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos, cur,
            )?;
            dgpu_scratch.logits.copy_to_host(&mut logits)?;
            plain_dists.push(softmax_dist(&logits, temperature, min_p));
            pos += 1;
        }
    }

    assert_eq!(spec_out.target_dists.len(), n_tokens, "spec dist count");
    assert_eq!(plain_dists.len(), n_tokens, "plain dist count");

    let mut kl_sum = 0f64;
    let mut kl_max = 0f64;
    let mut tvd_sum = 0f64;
    let mut tvd_max = 0f64;
    // argmax-agreement of the two target distributions (sanity).
    let argmax = |d: &[f32]| -> i32 {
        let mut b = 0i32;
        let mut bv = d[0];
        for (i, &v) in d.iter().enumerate().skip(1) {
            if v > bv {
                bv = v;
                b = i as i32;
            }
        }
        b
    };
    let mut argmax_agree = 0usize;
    eprintln!("  per-position KL(plain‖spec) / TVD:");
    for i in 0..n_tokens {
        let sp = &spec_out.target_dists[i];
        let pl = &plain_dists[i];
        let k = kl(pl, sp);
        let tv = tvd(pl, sp);
        kl_sum += k;
        tvd_sum += tv;
        kl_max = kl_max.max(k);
        tvd_max = tvd_max.max(tv);
        if argmax(sp) == argmax(pl) {
            argmax_agree += 1;
        }
        if i < 16 || k > 1e-2 {
            eprintln!("    [{i:>2}] tok={:>6}  KL={k:.5}  TVD={tv:.5}", spec_out.tokens[i]);
        }
    }
    let n = n_tokens as f64;
    eprintln!("\n===== QUALITY (spec target vs plain single-token, teacher-forced) =====");
    eprintln!("  mean KL(plain‖spec): {:.6} nats", kl_sum / n);
    eprintln!("  max  KL:             {:.6} nats", kl_max);
    eprintln!("  mean TVD:            {:.6}", tvd_sum / n);
    eprintln!("  max  TVD:            {:.6}", tvd_max);
    eprintln!(
        "  argmax agreement:    {}/{} = {:.3}",
        argmax_agree, n_tokens, argmax_agree as f64 / n
    );

    // ===================== BASELINE: plain multinomial tok/s =====================
    eprintln!("\n########## BASELINE plain multinomial decode (same seed) ##########");
    let (base_tokens, base_wall) = {
        engine.clear_graphs();
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut logits = vec![0f32; vocab];
        let mut rng = SamplerRng::new(seed);
        for (pos, &tok) in PROMPT_TOKENS.iter().enumerate() {
            let inp = broadcast_hc(&embed(tok));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos as u32, tok,
            )?;
        }
        let t0 = Instant::now();
        let mut toks: Vec<i32> = Vec::with_capacity(n_tokens);
        dgpu_scratch.logits.copy_to_host(&mut logits)?;
        let mut cur = sample_from_dist(&softmax_dist(&logits, temperature, min_p), rng.next_f32());
        let mut pos = PROMPT_TOKENS.len() as u32;
        for _ in 0..n_tokens {
            toks.push(cur);
            let inp = broadcast_hc(&embed(cur));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos, cur,
            )?;
            dgpu_scratch.logits.copy_to_host(&mut logits)?;
            cur = sample_from_dist(&softmax_dist(&logits, temperature, min_p), rng.next_f32());
            pos += 1;
        }
        (toks, t0.elapsed().as_secs_f64())
    };
    eprintln!("  plain committed:   {}", base_tokens.len());
    eprintln!("  wall:              {base_wall:.3} s");
    eprintln!("  >>> plain tok/s:   {:.2}", base_tokens.len() as f64 / base_wall);
    eprintln!(
        "\n  e2e speed ratio spec/plain: {:.2}x (regression expected pre perf re-arch)",
        (s.committed as f64 / spec_wall) / (base_tokens.len() as f64 / base_wall)
    );

    engine.shutdown()?;

    // Quality gate: rejection sampling is distribution-correct, so KL must be
    // small — dominated by the batch-vs-seq drift on the B=2-verified positions,
    // NOT by a systematic sampler bug. Generous ceiling; a broken accept rule
    // (e.g. committing argmax, or wrong residual) blows this up.
    let mean_kl = kl_sum / n;
    assert!(
        mean_kl < 0.05,
        "QUALITY GATE: mean KL(plain‖spec) = {mean_kl:.6} nats too high — spec target distribution diverged from plain decode (not just drift)"
    );
    Ok(())
}
