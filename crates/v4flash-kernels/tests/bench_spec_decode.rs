//! M40-P6.3: bench `spec_decode_step` end-to-end. Loops N rounds of
//! (forward_token, mtp_draft, forward_pair_interleaved) and reports
//! median wall + effective tok/s at the actual acceptance rate.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_spec_decode \
//!     -- --ignored --nocapture
//!
//! Optional env:
//!   BENCH_ROUNDS=N      (default 30) — number of spec_decode rounds
//!   BENCH_WARMUP=N      (default 8)  — single-token forwards to fill KV
//!
//! Perfetto: pair with `perfetto_spec` test to get a trace with
//! dgpu.spec.target_step / dgpu.spec.mtp_draft / dgpu.spec.verify_pair
//! spans visible on the dGPU device track.

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::{
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
    MtpWeights,
};
use v4flash_kernels::{ActivationDump, RopeParams};

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

fn pick_dgpu_device() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}

fn pick_igpu_device() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

/// Cached embedding lookup. token_embd is ~1 GB; mmap once, decode per call.
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
            return Err(eyre!("token_embd dtype {:?} != F16", t.dtype));
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

#[test]
#[ignore]
fn bench_spec_decode() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_rounds: i32 = std::env::var("BENCH_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);
    let warmup_tokens: i32 = std::env::var("BENCH_WARMUP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    eprintln!("spec_decode bench: n_rounds={n_rounds} warmup_tokens={warmup_tokens}");

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let mtp_gguf = MappedGguf::open(MTP_MODEL_PATH)?;
    let dgpu = pick_dgpu_device()?;
    let igpu = pick_igpu_device()?;
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
    eprintln!("loading token_embd cache (~1 GB host)...");
    let embed_cache = EmbedCache::load(&main_gguf, v4flash_kernels::forward::N_EMBD)?;

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let total_positions = warmup_tokens + 2 * n_rounds + PROMPT_TOKENS.len() as i32;
    let mut state = HetModelState::alloc(dgpu, igpu, total_positions as u32)?;
    state.alloc_mtp(dgpu)?;

    use v4flash_kernels::forward::{HC_DIM, N_EMBD};
    let max_inp_pos = dump.n_logit_rows as i32 + PROMPT_TOKENS.len() as i32 - 2;

    eprintln!("preloading {} input_hcs from dump...", total_positions);
    let mut inputs: Vec<Vec<f32>> = Vec::with_capacity(total_positions as usize);
    for pos in 0..total_positions {
        let inp_pos = pos.min(max_inp_pos);
        let inp_entry = dump
            .tensor("layer_input_residual", 0, inp_pos)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{inp_pos}"))?;
        let input_hc = dump.read_f32(inp_entry)?;
        assert_eq!(input_hc.len(), HC_DIM as usize);
        inputs.push(input_hc);
    }

    // Warmup: fill KV cache with N single-token forwards.
    eprintln!("warmup: {warmup_tokens} forward_tokens...");
    for pos in 0..warmup_tokens {
        let token_id = if (pos as usize) < PROMPT_TOKENS.len() {
            PROMPT_TOKENS[pos as usize]
        } else {
            0
        };
        engine.forward_token(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights,
            &inputs[pos as usize],
            pos as u32,
            token_id,
        )?;
    }

    // Initial t_committed = whatever's at warmup_tokens position.
    // Initial t_draft = some token (in real spec decode this'd be MTP's
    // first prediction; for timing we just use the next prompt token).
    eprintln!("bench: {n_rounds} spec_decode rounds...");
    let mut step_us: Vec<u64> = Vec::with_capacity(n_rounds as usize);
    let mut accepts: u32 = 0;
    let mut committed_total: u32 = 0;
    let mut cur_pos: u32 = warmup_tokens as u32;
    let mut cur_t_committed: i32 = if (cur_pos as usize) < PROMPT_TOKENS.len() {
        PROMPT_TOKENS[cur_pos as usize]
    } else {
        0
    };
    let mut cur_t_draft: i32 = if ((cur_pos + 1) as usize) < PROMPT_TOKENS.len() {
        PROMPT_TOKENS[(cur_pos + 1) as usize]
    } else {
        0
    };
    let bench_start = Instant::now();
    for _r in 0..n_rounds {
        if (cur_pos as usize + 3) >= inputs.len() {
            break;
        }
        let input_committed = &inputs[cur_pos as usize];
        let input_draft = &inputs[(cur_pos + 1) as usize];
        let draft_embd = embed_cache.lookup(cur_t_draft);
        let _ = N_EMBD;

        let t = Instant::now();
        let res = engine.spec_decode_step(
            &mut dgpu_scratch,
            &mut igpu_scratch,
            &mut state,
            &main_weights,
            &mtp_weights,
            cur_t_committed,
            cur_t_draft,
            cur_pos,
            input_committed,
            input_draft,
            &draft_embd,
        )?;
        let dt = t.elapsed().as_micros() as u64;
        step_us.push(dt);
        if res.accepted {
            accepts += 1;
        }
        committed_total += res.committed.len() as u32;
        cur_t_committed = res.next_t_committed;
        cur_t_draft = res.next_t_draft;
        cur_pos = res.next_pos;
    }
    let bench_wall = bench_start.elapsed();

    let mut sorted = step_us.clone();
    sorted.sort_unstable();
    let n = sorted.len() as f64;
    let median_us = sorted[sorted.len() / 2];
    let min_us = *sorted.first().unwrap();
    let max_us = *sorted.last().unwrap();
    let avg_us: f64 = step_us.iter().sum::<u64>() as f64 / n;
    let total_secs = bench_wall.as_secs_f64();
    let effective_tps = committed_total as f64 / total_secs;
    let accept_rate = (accepts as f64) / n * 100.0;

    eprintln!(
        "BENCH SPEC_DECODE: n_rounds={} commits={} accept_rate={:.1}% wall={:.2}s",
        step_us.len(),
        committed_total,
        accept_rate,
        total_secs
    );
    eprintln!(
        "  per-step: avg={:.2}ms median={:.2}ms min={:.2}ms max={:.2}ms",
        avg_us / 1000.0,
        median_us as f64 / 1000.0,
        min_us as f64 / 1000.0,
        max_us as f64 / 1000.0
    );
    eprintln!(
        "  effective: {:.2} tok/s (={} commits / {:.2}s)",
        effective_tps, committed_total, total_secs
    );
    eprintln!("BENCH SPEC_DECODE percentiles (ms):");
    for (p, label) in [(10.0, "p10"), (50.0, "p50"), (90.0, "p90"), (99.0, "p99")] {
        let idx = ((p / 100.0) * (n - 1.0)).round() as usize;
        eprintln!("  {label} {:>7.2} ms", sorted[idx] as f64 / 1000.0);
    }

    Ok(())
}
