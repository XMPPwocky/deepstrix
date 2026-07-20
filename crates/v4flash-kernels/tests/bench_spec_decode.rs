//! K=1 MTP speculative-decode e2e harness.
//!
//! Drives `HeterogeneousEngine::spec_decode_run` for ~N tokens and reports:
//!   * effective tok/s (committed tokens / wall)
//!   * free-check pass rate (MTP top-1 vs main argmax @P+1 — the ~0.65 oracle
//!     sanity) and B=2 accept rate
//!   * GREEDY-EXACTNESS gate: the committed stream MUST equal a plain greedy
//!     `forward_token` decode from the same start. This is the correctness
//!     gate for the loop + compressor-frontier rollback.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 \
//!   DGPU_HOT_EXPERTS=8 \
//!   DGPU_HOT_EXPERTS_FILE=/home/claude-code/deepstrix/reference/decode_hot_experts.txt \
//!   SPEC_TOKENS=40 \
//!     nix develop -c cargo test --release -p v4flash-kernels \
//!       --test bench_spec_decode -- --ignored --nocapture

use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch, MtpLayerState, MtpScratch, MtpWeights, SpecDecodeConfig,
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

#[test]
#[ignore]
fn bench_spec_decode() -> eyre::Result<()> {
    install_panic_handler()?;

    let n_tokens: usize = std::env::var("SPEC_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(40);
    eprintln!(
        "spec_decode: n_tokens={n_tokens} prompt_len={}",
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
    let mut mtp_state = MtpLayerState::alloc(dgpu)?;

    let n_kv_max = (PROMPT_TOKENS.len() + n_tokens + 8) as u32;

    let embed = |tok: i32| embed_cache.lookup(tok);

    // ================= REFERENCE: plain greedy forward_token =================
    // Same prime + greedy argmax chain the spec loop must reproduce exactly.
    eprintln!("\n########## REFERENCE greedy (forward_token) ##########");
    let mut ref_tokens: Vec<i32> = Vec::with_capacity(n_tokens);
    {
        engine.clear_graphs();
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut logits = vec![0f32; N_VOCAB as usize];
        // prime prompt
        for (pos, &tok) in PROMPT_TOKENS.iter().enumerate() {
            let inp = broadcast_hc(&embed(tok));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos as u32, tok,
            )?;
        }
        dgpu_scratch.logits.copy_to_host(&mut logits)?;
        let mut cur = argmax(&logits);
        let mut pos = PROMPT_TOKENS.len() as u32;
        for _ in 0..n_tokens {
            ref_tokens.push(cur);
            let inp = broadcast_hc(&embed(cur));
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights, &inp, pos, cur,
            )?;
            dgpu_scratch.logits.copy_to_host(&mut logits)?;
            cur = argmax(&logits);
            pos += 1;
        }
    }
    eprintln!("reference greedy tokens: {ref_tokens:?}");

    // ================= SPECULATIVE DECODE =================
    let report = |tag: &str, out: &v4flash_kernels::het::SpecDecodeOut, wall: f64| {
        let s = &out.stats;
        let free_rate = s.free_check_pass as f64 / s.rounds.max(1) as f64;
        let b2_rate = if s.b2_verifies > 0 {
            s.b2_accepts as f64 / s.b2_verifies as f64
        } else {
            0.0
        };
        let tok_s = s.committed as f64 / wall;
        eprintln!("\n===== STATS [{tag}] =====");
        eprintln!("  rounds:            {}", s.rounds);
        eprintln!("  committed tokens:  {}", s.committed);
        eprintln!(
            "  free-check pass:   {}/{} = {:.3}  (MTP top-1 == main argmax@P+1; oracle sanity ~0.65)",
            s.free_check_pass, s.rounds, free_rate
        );
        eprintln!(
            "  B=2 accept:        {}/{} = {:.3}  (2nd draft accepted)",
            s.b2_accepts, s.b2_verifies, b2_rate
        );
        eprintln!("  tokens / round:    {:.3}", s.committed as f64 / s.rounds.max(1) as f64);
        eprintln!("  wall:              {wall:.3} s");
        eprintln!("  >>> effective:     {tok_s:.2} tok/s");
    };

    let n_matching = |a: &[i32], b: &[i32]| -> usize {
        a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
    };

    // ---- (1) BIT-EXACT mode: the greedy-exactness GATE ----
    eprintln!("\n########## SPEC decode BIT-EXACT (verify B=2 + rollback, commit via exact B=1) ##########");
    let (exact_out, exact_wall) = {
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut frontier = state.alloc_frontier(dgpu)?;
        // fresh MTP cache for this run
        mtp_state = MtpLayerState::alloc(dgpu)?;
        let t0 = Instant::now();
        let out = engine.spec_decode_run(
            &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
            &mut state, &weights, &mut mtp_scratch, &mut mtp_state, &mtp_weights, &mut frontier,
            &PROMPT_TOKENS, &embed, SpecDecodeConfig { n_tokens, bit_exact: true },
        )?;
        (out, t0.elapsed().as_secs_f64())
    };
    eprintln!("committed: {:?}", exact_out.tokens);
    let exact = exact_out.tokens == ref_tokens;
    eprintln!("\n===== GREEDY-EXACTNESS GATE =====");
    if exact {
        eprintln!("  PASS: bit-exact committed stream == plain greedy forward_token ({} tokens)", exact_out.tokens.len());
    } else {
        let m = n_matching(&exact_out.tokens, &ref_tokens);
        eprintln!("  FAIL: diverged at idx {m}: spec={:?} ref={:?}",
            exact_out.tokens.get(m), ref_tokens.get(m));
    }
    report("bit-exact", &exact_out, exact_wall);

    // ---- (2) SPEEDUP mode: bank the batched B=2 accept (NOT bit-exact) ----
    eprintln!("\n########## SPEC decode SPEEDUP (bank batched B=2 accept — not bit-exact) ##########");
    let (fast_out, fast_wall) = {
        let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut frontier = state.alloc_frontier(dgpu)?;
        mtp_state = MtpLayerState::alloc(dgpu)?;
        let t0 = Instant::now();
        let out = engine.spec_decode_run(
            &mut bd, &mut bi, &mut dgpu_scratch, &mut igpu_scratch, &mut head_scratch,
            &mut state, &weights, &mut mtp_scratch, &mut mtp_state, &mtp_weights, &mut frontier,
            &PROMPT_TOKENS, &embed, SpecDecodeConfig { n_tokens, bit_exact: false },
        )?;
        (out, t0.elapsed().as_secs_f64())
    };
    let fast_match = n_matching(&fast_out.tokens, &ref_tokens);
    eprintln!("committed: {:?}", fast_out.tokens);
    eprintln!("  matches plain greedy for first {fast_match}/{} tokens (drift after that)", fast_out.tokens.len());
    report("speedup", &fast_out, fast_wall);

    let _ = HC_DIM;
    engine.shutdown()?;
    assert!(exact, "GREEDY-EXACTNESS GATE FAILED: bit-exact spec-decode stream diverged from plain greedy");
    Ok(())
}
