//! Golden gate: the production V4.1 engine against the frozen CPU
//! reference (DeepSeek's unmodified `inference/model.py`, captured by
//! `scripts/v41_oracle/oracle.py --golden` and exported by `export_golden.py`).
//!
//! Free routing, end to end, teacher-forced over the golden transcript:
//!   * the prompt is everything before the first assistant turn's `<think>`; the
//!     trailing `<think>` is forwarded as the first DECODE step, as production does;
//!   * every later token is fed through a decode path and the logits at each
//!     step are compared with the reference's logits at that position;
//!   * paths: `serial` (forward_prefill_pipelined + forward_token_paged, the
//!     serial worker) and `arena` (PrefillJob + forward_step_arena at one row,
//!     the multistream path);
//!   * every routing decision is read back through the pick sink and compared
//!     with the reference's top-6; each difference is classified by the
//!     reference's own score gap between the expert it lost and the one it gained.
//!     A small gap is a near-tie flip (expected under numeric error); a large gap
//!     is a bug, whatever the KL says.
//!
//! Routing modes (`GOLDEN_ROUTING`, default both):
//!   * `free`: the engine routes itself, as in production;
//!   * `pinned`: every routing decision, prompt and decode, is forced to the
//!     reference's top-6 (`het::routing_tap` PIN), weighted from the engine's own
//!     router scores. Selection is the one discontinuity in the forward pass, so
//!     pinned KL prices the engine's numerics alone; free minus pinned is what
//!     the routing flips cost. A pinned run must show zero differing picks (the
//!     pin is checked, not assumed) and reports how many rows it overrode.
//!
//! Reports KL(ref || engine) mean / p50 / p90 / p99 / max, top-1 agreement and the
//! transcript NLL, over generated positions (the assistant spans) and over all
//! decoded positions, plus the routing-flip table. The first runs establish the
//! baseline; the only hard failures are non-finite logits and gross divergence.
//!
//! Box 1 only by default (V41_REMOTE_ADDR unset: every expert pages from box 1's
//! NVMe). Production also adds box 2's partial sums at the combine, so expect
//! f32 reassociation differences against a two-box run, not errors.
//!
//! Needs the model loaded, i.e. the server DOWN:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c \
//!   cargo test -p v4flash-kernels --release --test v41_golden_gate -- --ignored --nocapture
//! ```
//! Env: GOLDEN_CASE (fixture dir, default ~/.cache/deepstrix/goldens/agentic),
//! GOLDEN_PATHS (serial,arena), GOLDEN_ROUTING (free,pinned), GOLDEN_MAX_STEPS, GOLDEN_REPORT (JSON path),
//! V41_HF_DIR, V41_ENGRAM_DIR, V41_PAGER_POOL_GB (default 40 here).

use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre};
use v4flash_core::{EngramHash, EngramTable, V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{
    COMPRESS_RATIOS, ENGRAM_IN, ENGRAM_LAYERS, HC_DIM, KV_SOURCE_LAYERS, N_EXPERT, N_EXPERT_USED, N_LAYER, N_VOCAB,
};
use v4flash_kernels::embed::embed_lookup;
use v4flash_kernels::het::routing_tap::{
    pick_sink_enable, pick_sink_take, pin_overrides_take, pin_set, PickRecord, PinTable,
};
use v4flash_kernels::het::forward_prefill::{LazyEngramRows, PrefillJob};
use v4flash_kernels::het::kv_arena::{KvArena, RowTablesDev};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode, ExpertPager,
    HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::RopeParams;

const ROPE_ORIG_CTX: u64 = 65536;
const HF_DIR_DEFAULT: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";
/// Reference score gap (lost expert minus gained expert) above which a routing
/// difference cannot be a near-tie flip.
const LARGE_FLIP_GAP: f32 = 0.05;

fn rope_for_layer(layer: i32) -> eyre::Result<RopeParams> {
    let compressed = COMPRESS_RATIOS[layer as usize] != 0;
    let freq_base = if compressed { 160000.0 } else { 10000.0 };
    let freq_scale = if compressed { 1.0 / 16.0 } else { 1.0 };
    let ext_factor = if compressed { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0f32 / freq_scale).ln();
    }
    let n_ctx_orig = if compressed { ROPE_ORIG_CTX } else { 0 };
    RopeParams::from_dump_blob(&[freq_base, freq_scale, ext_factor, attn_factor, 32.0, 1.0], n_ctx_orig)
}

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix} device"))
}

// ------------------------------------------------------------------ fixture

struct Fixture {
    tokens: Vec<i32>,
    /// (think_idx, eos_idx): generated regions; positions think..eos-1 predict generated tokens.
    spans: Vec<(usize, usize)>,
    vocab: usize,
    logits: Vec<f32>,     // [T, V]
    topk: Vec<i32>,       // [L, T, 6]
    sel: Vec<f32>,        // [L, T, 384]
}

fn read_f32(p: &Path) -> eyre::Result<Vec<f32>> {
    let b = std::fs::read(p).map_err(|e| eyre!("{}: {e}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}
fn read_i32(p: &Path) -> eyre::Result<Vec<i32>> {
    let b = std::fs::read(p).map_err(|e| eyre!("{}: {e}", p.display()))?;
    Ok(b.chunks_exact(4).map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
}

impl Fixture {
    fn load(dir: &Path) -> eyre::Result<Self> {
        let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(dir.join("fixture.json"))?)?;
        let tokens: Vec<i32> = serde_json::from_slice(&std::fs::read(dir.join("tokens.json"))?)?;
        let spans: Vec<(usize, usize)> = serde_json::from_slice(&std::fs::read(dir.join("spans.json"))?)?;
        let vocab = fx["vocab"].as_u64().ok_or_else(|| eyre!("fixture.json: vocab"))? as usize;
        let t = tokens.len();
        let logits = read_f32(&dir.join("logits_all.f32"))?;
        let topk = read_i32(&dir.join("topk_ids.i32"))?;
        let sel = read_f32(&dir.join("router_sel.f32"))?;
        let l = N_LAYER as usize;
        if logits.len() != t * vocab || topk.len() != l * t * N_EXPERT_USED || sel.len() != l * t * N_EXPERT as usize {
            return Err(eyre!("fixture shapes do not match T={t} V={vocab} L={l}"));
        }
        if vocab != N_VOCAB as usize {
            return Err(eyre!("fixture vocab {vocab} != engine {N_VOCAB}"));
        }
        Ok(Self { tokens, spans, vocab, logits, topk, sel })
    }
    fn ref_logits(&self, pos: usize) -> &[f32] {
        &self.logits[pos * self.vocab..(pos + 1) * self.vocab]
    }
    fn ref_topk(&self, layer: usize, pos: usize) -> &[i32] {
        let t = self.tokens.len();
        &self.topk[(layer * t + pos) * N_EXPERT_USED..(layer * t + pos + 1) * N_EXPERT_USED]
    }
    fn ref_sel(&self, layer: usize, pos: usize) -> &[f32] {
        let (t, e) = (self.tokens.len(), N_EXPERT as usize);
        &self.sel[(layer * t + pos) * e..(layer * t + pos + 1) * e]
    }
    fn generated_pred(&self, pos: usize) -> bool {
        self.spans.iter().any(|&(a, b)| pos >= a && pos < b)
    }
}

// ------------------------------------------------------------------ Engram

struct Engram {
    hasher: EngramHash,
    tables: Vec<EngramTable>,
}

impl Engram {
    fn rows_for_prompt(&self, st: &v4flash_core::SafetensorsDir, tokens: &[i32]) -> eyre::Result<Vec<Vec<f32>>> {
        let ein = ENGRAM_IN as usize;
        let hashes = self.hasher.hash_sequence(tokens);
        let mut out = vec![vec![0f32; tokens.len() * ein]; self.tables.len()];
        for (li, tbl) in self.tables.iter().enumerate() {
            for t in 0..tokens.len() {
                tbl.gather_position(st, &hashes[t][li], &mut out[li][t * ein..(t + 1) * ein])?;
            }
        }
        Ok(out)
    }
    fn rows_at(&self, st: &v4flash_core::SafetensorsDir, seq: &[i32], pos: usize) -> eyre::Result<Vec<Vec<f32>>> {
        let ein = ENGRAM_IN as usize;
        let compressed: Vec<i32> = seq[..=pos].iter().map(|&t| self.hasher.compress(t)).collect();
        let hashes = self.hasher.hash_ids(&compressed, pos);
        let mut out = Vec::with_capacity(self.tables.len());
        for (li, tbl) in self.tables.iter().enumerate() {
            let mut r = vec![0f32; ein];
            tbl.gather_position(st, &hashes[li], &mut r)?;
            out.push(r);
        }
        Ok(out)
    }
}

// ------------------------------------------------------------------ metrics

fn log_softmax(v: &[f32]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln();
    v.iter().map(|&x| x as f64 - lse).collect()
}
fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0
}

#[derive(Default)]
struct StepStats {
    pos: Vec<usize>,
    kl: Vec<f64>,
    top1: Vec<bool>,
    nll_eng: Vec<f64>,
    nll_ref: Vec<f64>,
    generated: Vec<bool>,
}

impl StepStats {
    fn push(&mut self, fx: &Fixture, pos: usize, eng: &[f32]) -> eyre::Result<()> {
        if eng.iter().any(|x| !x.is_finite()) {
            return Err(eyre!("non-finite engine logits at position {pos}"));
        }
        let r = fx.ref_logits(pos);
        let (lr, le) = (log_softmax(r), log_softmax(eng));
        let kl: f64 = lr.iter().zip(&le).map(|(a, b)| a.exp() * (a - b)).sum();
        let nxt = fx.tokens.get(pos + 1).copied().unwrap_or(fx.tokens[pos]) as usize;
        self.pos.push(pos);
        self.kl.push(kl);
        self.top1.push(argmax(r) == argmax(eng));
        self.nll_eng.push(-le[nxt]);
        self.nll_ref.push(-lr[nxt]);
        self.generated.push(fx.generated_pred(pos));
        Ok(())
    }
    fn summary(&self, only_generated: bool) -> serde_json::Value {
        let idx: Vec<usize> = (0..self.kl.len()).filter(|&i| !only_generated || self.generated[i]).collect();
        if idx.is_empty() {
            return serde_json::json!({"n": 0});
        }
        let mut kl: Vec<f64> = idx.iter().map(|&i| self.kl[i]).collect();
        kl.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| kl[((kl.len() - 1) as f64 * p).round() as usize];
        let n = idx.len() as f64;
        let worst = idx.iter().copied().max_by(|&a, &b| self.kl[a].partial_cmp(&self.kl[b]).unwrap()).unwrap();
        serde_json::json!({
            "n": idx.len(),
            "kl_mean": kl.iter().sum::<f64>() / n, "kl_p50": q(0.5), "kl_p90": q(0.9), "kl_p99": q(0.99), "kl_max": q(1.0),
            "kl_max_pos": self.pos[worst],
            "top1_agree": idx.iter().filter(|&&i| self.top1[i]).count() as f64 / n,
            "nll_engine": idx.iter().map(|&i| self.nll_eng[i]).sum::<f64>() / n,
            "nll_reference": idx.iter().map(|&i| self.nll_ref[i]).sum::<f64>() / n,
        })
    }
}

/// Routing comparison against the reference top-6, per token-layer.
#[derive(Default)]
struct FlipStats {
    token_layers: usize,
    differing: usize,
    /// reference gap (lost - gained selection score) of the worst swap at each differing token-layer
    gaps: Vec<f32>,
    per_layer_diff: Vec<usize>,
    large: Vec<(usize, usize, f32)>,
    /// Pinned runs: rows whose pick set the pin replaced (prompt, decode).
    pin_overrides: (u64, u64),
}

impl FlipStats {
    fn new() -> Self {
        Self { per_layer_diff: vec![0; N_LAYER as usize], ..Default::default() }
    }
    fn observe(&mut self, fx: &Fixture, pos: usize, picks: &[PickRecord]) -> eyre::Result<()> {
        if picks.len() != N_LAYER as usize {
            return Err(eyre!("position {pos}: {} pick records, expected one per layer ({N_LAYER})", picks.len()));
        }
        for p in picks {
            let l = p.layer as usize;
            let want = fx.ref_topk(l, pos);
            let sel = fx.ref_sel(l, pos);
            self.token_layers += 1;
            let lost: Vec<i32> = want.iter().copied().filter(|e| !p.ids.contains(e)).collect();
            let gained: Vec<i32> = p.ids.iter().copied().filter(|e| !want.contains(e)).collect();
            if lost.is_empty() {
                continue;
            }
            self.differing += 1;
            self.per_layer_diff[l] += 1;
            // Worst-case pairing: the best-scoring lost expert against the worst-scoring gained one.
            let lost_s = lost.iter().map(|&e| sel[e as usize]).fold(f32::NEG_INFINITY, f32::max);
            let gained_s = gained.iter().map(|&e| sel[e as usize]).fold(f32::INFINITY, f32::min);
            let gap = lost_s - gained_s;
            self.gaps.push(gap);
            if gap >= LARGE_FLIP_GAP {
                self.large.push((pos, l, gap));
            }
        }
        Ok(())
    }
    fn summary(&self) -> serde_json::Value {
        let n = self.gaps.len().max(1) as f64;
        let share = |lo: f32, hi: f32| self.gaps.iter().filter(|&&g| g >= lo && g < hi).count() as f64 / n;
        serde_json::json!({
            "token_layers": self.token_layers,
            "differing": self.differing,
            "differing_share": self.differing as f64 / self.token_layers.max(1) as f64,
            "gap_lt_1e-3": share(f32::NEG_INFINITY, 1e-3),
            "gap_1e-3_to_1e-2": share(1e-3, 1e-2),
            "gap_1e-2_to_0.05": share(1e-2, LARGE_FLIP_GAP),
            "gap_ge_0.05": share(LARGE_FLIP_GAP, f32::INFINITY),
            "large_flips": self.large.iter().take(50).map(|&(p, l, g)| serde_json::json!({"pos": p, "layer": l, "gap": g})).collect::<Vec<_>>(),
            "per_layer_differing": self.per_layer_diff,
            "pin_overrides_prompt": self.pin_overrides.0,
            "pin_overrides_decode": self.pin_overrides.1,
        })
    }
}

// ------------------------------------------------------------------ test

#[test]
#[ignore]
fn v41_golden_gate() -> eyre::Result<()> {
    install_panic_handler()?;
    // Production code paths (deploy/run-hub.sh); these are cached on first use,
    // so they must be set before the engine exists.
    for (k, v) in [("V41_INDEX_K", "1"), ("V41_CANDIDATE_POOL", "1"), ("V41_PAGER_POOL_GB", "40")] {
        if std::env::var(k).is_err() {
            std::env::set_var(k, v);
        }
    }
    let home = std::env::var("HOME").unwrap_or_default();
    let case = PathBuf::from(std::env::var("GOLDEN_CASE").unwrap_or_else(|_| format!("{home}/.cache/deepstrix/goldens/agentic")));
    let list = |k: &str, d: &str| -> Vec<String> {
        std::env::var(k).unwrap_or_else(|_| d.into()).split(',').map(|s| s.trim().to_string()).collect()
    };
    let paths = list("GOLDEN_PATHS", "serial,arena");
    let routings = list("GOLDEN_ROUTING", "free,pinned");
    if let Some(r) = routings.iter().find(|r| !matches!(r.as_str(), "free" | "pinned")) {
        return Err(eyre!("unknown GOLDEN_ROUTING entry {r}"));
    }
    let runs: Vec<(String, bool)> =
        paths.iter().flat_map(|p| routings.iter().map(move |r| (p.clone(), r == "pinned"))).collect();
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| format!("{home}/.cache/deepstrix/v41/engram"));

    let fx = Fixture::load(&case)?;
    let t_n = fx.tokens.len();
    let (think, _) = *fx.spans.first().ok_or_else(|| eyre!("fixture has no generated spans"))?;
    let prompt = &fx.tokens[..think]; // the trailing <think> is the first decode step
    let max_steps = std::env::var("GOLDEN_MAX_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(usize::MAX);
    let last_pos = (t_n - 2).min(think + max_steps.saturating_sub(1));
    eprintln!(
        "golden gate: case {} T={t_n}, prompt {} tokens, decode positions {think}..={last_pos}, paths {paths:?}",
        case.display(), prompt.len()
    );

    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let darch = dgpu.properties()?.gcn_arch_name;
    let iarch = igpu.properties()?.gcn_arch_name;
    let hf = V41HfWeights::open(&dir, None)?;
    let src = WeightSrc::from(&hf);
    let te = src.tensor("token_embd.weight").ok_or_else(|| eyre!("missing token_embd.weight"))?;
    let te_dtype = te.dtype;
    let te_bytes = src.read_tensor(te)?;
    let embed = |tok: i32| -> eyre::Result<Vec<f32>> {
        let mut r = vec![0f32; HC_DIM as usize];
        embed_lookup(&te_bytes, te_dtype, tok, &mut r)?;
        Ok(r)
    };
    let weights = HetModelWeights::load_all(src, dgpu, igpu, &rope_for_layer)?;
    let engine = HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?;
    let mut ds = DgpuScratch::alloc(dgpu)?;
    let mut is = IgpuScratch::alloc(igpu)?;
    let n_kv_max: u32 = ((t_n + 64).next_power_of_two().max(1024)) as u32;
    let lane_rows = v4flash_kernels::het::batch_scratch::B_MAX.div_ceil(2);
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows_ctx(dgpu, lane_rows, n_kv_max)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
    let hasher = EngramHash::load(Path::new(&engram_dir))?;
    hasher.self_check()?;
    let mut tables = Vec::new();
    for &l in ENGRAM_LAYERS {
        tables.push(EngramTable::open(pg.raw(), l as usize)?);
    }
    let engram = Engram { hasher, tables };
    let hcs_prompt: Vec<Vec<f32>> = prompt.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
    let rows_prompt = engram.rows_for_prompt(pg.raw(), prompt)?;

    let mut report = serde_json::Map::new();
    report.insert("case".into(), case.display().to_string().into());
    report.insert("tokens".into(), t_n.into());
    report.insert("prompt_tokens".into(), prompt.len().into());
    report.insert("remote".into(), std::env::var("V41_REMOTE_ADDR").is_ok().into());

    for (path, pinned) in &runs {
        let name = if *pinned { format!("{path}+pin") } else { path.clone() };
        pin_set(if *pinned { Some(PinTable::new(N_LAYER as usize, t_n, fx.topk.clone())?) } else { None });
        let t0 = std::time::Instant::now();
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let mut steps = StepStats::default();
        let mut flips = FlipStats::new();
        // ---- prefill: last-position logits predict prompt.len()
        let prefill_logits = match path.as_str() {
            "serial" => engine.forward_prefill_pipelined(
                &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
                &hcs_prompt, prompt, 0, true, None, None, None, None, Some(&mut pg), Some(&rows_prompt),
            )?,
            "arena" => {
                let mut job = PrefillJob::new(prompt.to_vec(), hcs_prompt.clone(), Some(rows_prompt.clone()), None, 0, 1024)?;
                while !job.chunks_done() {
                    engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut pg))?;
                }
                engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights, Some(&mut pg))?
            }
            other => return Err(eyre!("unknown GOLDEN_PATHS entry {other}")),
        };
        st.restore_compressor_lending();
        engine.dgpu.compute.synchronize()?;
        let mut prefill_stats = StepStats::default();
        prefill_stats.push(&fx, prompt.len() - 1, &prefill_logits)?;
        flips.pin_overrides.0 = pin_overrides_take();
        let t_prefill = t0.elapsed().as_secs_f64();

        // ---- teacher-forced decode over the rest of the transcript
        let mut arena = None;
        let mut slot = 0u32;
        let mut dev = None;
        if path == "arena" {
            let mut a = KvArena::alloc(dgpu, 1, n_kv_max)?;
            slot = a.admit_from_state(&st, n_kv_max, prompt.len() as u32, &engine.dgpu.compute)?;
            engine.dgpu.compute.synchronize()?;
            arena = Some(a);
            dev = Some(RowTablesDev::alloc(dgpu, 1, KV_SOURCE_LAYERS.len())?);
        }
        let nv = N_VOCAB as usize;
        pick_sink_enable(true);
        let _ = pick_sink_take();
        for pos in think..=last_pos {
            let tok = fx.tokens[pos];
            pg.drain_prefetched()?;
            let rows = engram.rows_at(pg.raw(), &fx.tokens, pos)?;
            let hc = embed(tok)?;
            let logits = if path == "serial" {
                engine.forward_token_paged(&mut ds, &mut is, &mut st, &weights, &hc, pos as u32, tok, &mut pg, Some(&rows))?;
                engine.dgpu.compute.synchronize()?;
                let mut l = vec![0f32; nv];
                ds.logits.slice_view(0, nv).copy_to_host(&mut l)?;
                l
            } else {
                engine.forward_step_arena(
                    &mut bd_a, &mut bi_a, &mut sd, &mut si, arena.as_mut().unwrap(), dev.as_mut().unwrap(), &[slot], &weights,
                    &[hc], &[tok], &mut LazyEngramRows::ready(Some(rows)), Some(&mut pg),
                )?;
                engine.head_rows(&mut ds, &bd_a, 1, &weights)?
            };
            steps.push(&fx, pos, &logits)?;
            flips.observe(&fx, pos, &pick_sink_take())?;
            if (pos - think) % 100 == 0 {
                eprintln!("  [{name}] pos {pos}: KL {:.5} top1 {}", steps.kl.last().unwrap(), steps.top1.last().unwrap());
            }
        }
        pick_sink_enable(false);
        flips.pin_overrides.1 = pin_overrides_take();
        pin_set(None);
        if let Some(a) = arena.as_mut() {
            a.release(slot)?;
        }
        let secs = t0.elapsed().as_secs_f64();
        let r = serde_json::json!({
            "prefill_last": prefill_stats.summary(false),
            "decode_generated": steps.summary(true),
            "decode_all": steps.summary(false),
            "routing": flips.summary(),
            "seconds": {"prefill": t_prefill, "total": secs},
        });
        eprintln!("[{name}] {}", serde_json::to_string_pretty(&r)?);
        report.insert(name, r);
    }

    let out = std::env::var("GOLDEN_REPORT").map(PathBuf::from).unwrap_or_else(|_| {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../target").join(format!(
            "golden_gate_{}.json",
            case.file_name().and_then(|s| s.to_str()).unwrap_or("case")
        ))
    });
    std::fs::write(&out, serde_json::to_string_pretty(&serde_json::Value::Object(report.clone()))?)?;
    eprintln!("report: {}", out.display());
    engine.shutdown()?;

    // Hard failures only for now; thresholds tighten once a baseline exists.
    for (name, r) in &report {
        let Some(d) = r.get("decode_all") else { continue };
        let (mean, top1) = (d["kl_mean"].as_f64().unwrap_or(f64::NAN), d["top1_agree"].as_f64().unwrap_or(0.0));
        if !(mean < 0.5) || top1 < 0.8 {
            return Err(eyre!("[{name}] gross divergence from the reference: KL mean {mean:.4}, top-1 {top1:.3}"));
        }
        let differing = r["routing"]["differing"].as_u64().unwrap_or(u64::MAX);
        if name.ends_with("+pin") && differing != 0 {
            return Err(eyre!("[{name}] the pin did not hold: {differing} token-layers differ from the reference picks"));
        }
    }
    Ok(())
}
