//! Multi-stream decode step harness (docs/v41/MULTISTREAM_DECODE_PLAN.md 3.6 / 3.7,
//! M1a step 3): S prompts of different lengths are prefilled through the ordinary
//! batched prefill into S single-sequence states, admitted into two `KvArena`s,
//! and decoded T teacher-forced tokens three ways:
//!
//!   dec    — today's decode path (`forward_token_paged`), one stream at a time,
//!            on the stream's own state; its greedy tokens are the forced
//!            continuation for all three;
//!   alone  — the arena step (`forward_step_arena`) with ONE row per step;
//!   batch  — the arena step with all S rows co-batched.
//!
//!   pipe   — the arena step with all S rows on the TWO-LANE PIPELINED driver
//!            (`forward_step_arena_pipelined`), which production uses from 4
//!            rows up and which interleaves the lanes' pre-MoE phases.
//!
//! Gates:
//!   G5a  alone == batch bit-exactly (a row's output does not depend on its
//!        co-rows), unless MS_ALLOW_INEXACT=1 (then reported only);
//!   G5b  KLD(dec || batch) per token under MS_KLD_MEAN / MS_KLD_MAX (defaults
//!        0.02 / 0.5 nats; the decode-vs-verify baseline is ~1e-3).
//!        NOTE: with the default tiny prompts (5/40/131/260) and the two-box
//!        split attached, the BASELINE itself sits at mean ~0.08 / max ~0.43,
//!        so G5b fails on an unmodified tree; raise MS_KLD_MEAN for that
//!        configuration. G5a and G5c are the gates that discriminate.
//!   G5c  alone == pipe bit-exactly, same bars as G5b on KLD(dec || pipe). This
//!        is the gate on the LANE INTERLEAVING: shared scratch clobbered across
//!        lanes shows up as exactly one lane's rows differing (a `lane` column
//!        in the per-row table says which). It caught two such hazards when the
//!        phases were first split on 2026-09-22 -- `BatchIgpuShared` scratch and
//!        the per-layer `remap_dev` exclusion mask -- both of which the ordinary
//!        `alone`/`batch` gates are blind to because they are single-lane.
//!
//!   G5d  pipe == stag bit-exactly (same KLD bars): the TWO-LANE STAGGERED
//!        driver (`forward_step_arena_lanes` with 2 lanes), where the lanes are
//!        offset by half a layer so box 2 always has the other lane's request
//!        queued. Unlike the lockstep driver, the two lanes here are on
//!        DIFFERENT layers at once (pager eviction, per-layer remap, shared
//!        scratch all see that), so it needs its own gate before production.
//!
//!   G5e  pipe == ready-first bit-exactly: the staggered lanes with the host
//!        running whichever lane is ready (`forward_step_arena_ready_first`),
//!        whose interleave varies with timing.
//!
//!   G5f  SPECULATIVE BLOCKS (docs/v41/DSPARK_ARENA_PLAN.md 3.1-3.4): each
//!        stream alone runs blocks of 1-8 rows of ITSELF at consecutive
//!        positions (its next token plus "draft" rows) and keeps a prefix
//!        (`KvArena::accept`). Arm `spec` puts WRONG tokens in the rows past the
//!        kept prefix (a rejected tail), arm `spec_true` the right ones; both
//!        run the same block shapes. Gates: spec == spec_true bit-exactly on
//!        every kept row (a rejected tail leaves nothing behind: raw KV, comp
//!        rows, keys and accumulator blocks past the counters are dead, and a
//!        row does not depend on the rows after it), and spec vs alone like
//!        G5a (bit-exact unless MS_ALLOW_INEXACT=1) / G5b (KL bars). The block
//!        schedule starts blocks at both ratio-2 parities; long prompts
//!        (MS_LENS past 1024) cross the indexer's gathered path and MS_STEPS
//!        past ~130 the raw-region compaction.
//!
//!   G5g  ORDERED TWO-LANE VERIFY (docs/v41/DSPARK_SINGLE_STREAM_PERF.md 1): the
//!        G5f schedule through the READY-FIRST driver, every block of r >= 2
//!        rows cut into two lanes of the SAME stream ([ceil(r/2), floor(r/2)]),
//!        the later lane entering each layer only after the earlier one.
//!        Before every block a junk block runs at the same positions and is NOT
//!        accepted, so the dead state the block writes holds garbage (a lane
//!        reading too early reads that, never stale-but-right data). Arms:
//!        `spec2` (ordered), `spec2_hold` (ordered, lane 0 held until lane 1
//!        has posted each layer: the overtaking the ordering exists for, forced;
//!        must record Chain waits), `spec2_unord` (the hold with the ordering
//!        OFF: the negative control, must differ somewhere). Gates: spec2 and
//!        spec2_hold == alone and == spec bit-exactly on every kept row (with
//!        V41_SUB unset; the cache prior reads residency per lane), hold
//!        waits > 0, unord differs.
//!
//!   G5h  TWO STREAMS' BLOCKS IN ONE STEP (docs/v41/MS_DSPARK_STREAMS_DESIGN.md
//!        5): streams paired, both blocks of a pair in the SAME step (one
//!        `StepRows`, wrong tails, a junk step first), block order alternating
//!        so the balanced two-lane cut falls between the blocks, inside the
//!        first and inside the second. Arms: one lane (<= 8 rows), two lanes
//!        ready-first, the forced overtake, the UNORDERED control. Gates: every
//!        kept row == alone bit-exactly (KL bars under MS_ALLOW_INEXACT=1),
//!        every cut kind seen, hold waits > 0, unord differs. Needs
//!        MS_STREAMS >= 2 and prompts on one side of `need_mask` (512
//!        compressed rows) for the bit-exact arms.
//!
//! Needs the model loaded, i.e. the server DOWN. Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 V41_PAGED_EXPERTS=1 V41_INDEX_K=1 V41_CANDIDATE_POOL=1 \
//!   MS_STREAMS=4 MS_STEPS=6 CARGO_TARGET_DIR=target-v41 \
//!   nix develop -c cargo test -p v4flash-kernels --features v41 --release \
//!     --test multistream_step -- --ignored --nocapture
//! ```
//! `V41_HF_DIR` (default: the production snapshot), `V41_ENGRAM_DIR`,
//! `V41_PAGER_POOL_GB` (default 40 here), `MS_LENS` (comma list of prompt lengths).
#![cfg(feature = "v41")]

use std::path::Path;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{EngramHash, EngramTable, V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{
    COMPRESS_RATIOS, ENGRAM_IN, ENGRAM_LAYERS, HC_DIM, KV_SOURCE_LAYERS, N_VOCAB, SWA_WINDOW,
};
use v4flash_kernels::embed::embed_lookup;
use v4flash_kernels::het::kv_arena::{KvArena, RowTablesDev, ARENA_ROWS_PER_STREAM};
use v4flash_kernels::het::step_rows::StepRows;
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    ExpertPager, HetModelState, HetModelWeights, HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::RopeParams;

const ROPE_ORIG_CTX: u64 = 65536;
const HF_DIR_DEFAULT: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";

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

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_f64(k: &str, d: f64) -> f64 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

/// Deterministic pseudo-prompt: ids in [1000, 30000), per-stream seed.
fn synth_prompt(seed: u64, len: usize) -> Vec<i32> {
    let mut x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            1000 + ((x >> 33) % 29000) as i32
        })
        .collect()
}

struct Engram {
    hasher: EngramHash,
    tables: Vec<EngramTable>,
}

impl Engram {
    /// One `[len * ENGRAM_IN]` buffer per Engram layer for a whole prompt.
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
    /// One `[ENGRAM_IN]` row per Engram layer for the token at `pos` of `seq`
    /// (`seq[..=pos]` is the sequence so far).
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

fn half_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s << 31 } else {
            let mut e2 = 127 - 15 + 1;
            let mut m2 = m;
            while m2 & 0x400 == 0 { m2 <<= 1; e2 -= 1; }
            (s << 31) | ((e2 as u32) << 23) | ((m2 & 0x3ff) << 13)
        }
    } else if e == 31 { (s << 31) | 0x7f80_0000 | (m << 13) } else { (s << 31) | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}

fn argmax(v: &[f32]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate() {
        if x > v[best] {
            best = i;
        }
    }
    best
}

/// KL(p || q) in nats over softmaxes of two logit rows (f64).
fn kld(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    fn lse(v: &[f32]) -> f64 {
        let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
        m + v.iter().map(|&x| (x as f64 - m).exp()).sum::<f64>().ln()
    }
    let lp = lse(p_logits);
    let lq = lse(q_logits);
    p_logits
        .iter()
        .zip(q_logits)
        .map(|(&a, &b)| {
            let lpa = a as f64 - lp;
            lpa.exp() * (lpa - (b as f64 - lq))
        })
        .sum()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore]
fn multistream_step_matches_alone_and_decode() -> eyre::Result<()> {
    install_panic_handler()?;
    if std::env::var("V41_PAGED_EXPERTS").is_err() {
        std::env::set_var("V41_PAGED_EXPERTS", "1");
    }
    if std::env::var("V41_PAGER_POOL_GB").is_err() {
        std::env::set_var("V41_PAGER_POOL_GB", "40");
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let engram_dir = std::env::var("V41_ENGRAM_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/v41/engram", std::env::var("HOME").unwrap_or_default())
    });
    let n_streams = env_usize("MS_STREAMS", 4);
    // MS_FORCE_CONT="a,b,c": teacher-force these tokens through the decode
    // oracle (and everything else) instead of decode's own greedy picks;
    // overrides MS_STEPS with its length.
    let force_cont: Option<Vec<i32>> = std::env::var("MS_FORCE_CONT").ok().map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let n_steps = force_cont.as_ref().map(|c| c.len()).unwrap_or_else(|| env_usize("MS_STEPS", 6));
    let lens: Vec<usize> = match std::env::var("MS_LENS") {
        Ok(v) => v.split(',').map(|s| s.trim().parse().unwrap()).collect(),
        // Below / at / past the SWA window, odd and even (ratio-2 parity).
        Err(_) => vec![5, 40, 131, 260],
    };
    let allow_inexact = std::env::var("MS_ALLOW_INEXACT").as_deref() == Ok("1");
    let kld_mean_bar = env_f64("MS_KLD_MEAN", 0.02);
    let kld_max_bar = env_f64("MS_KLD_MAX", 0.5);
    // MS_PROMPT_IDS="a,b,c,...": stream 0 runs THIS sequence — prompt = all but
    // the last id, and the last id is the forced first decode token (so the
    // step-0 logits line up with an oracle dump of the full sequence).
    let forced: Option<Vec<i32>> = std::env::var("MS_PROMPT_IDS").ok().map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let mut prompts: Vec<Vec<i32>> = (0..n_streams).map(|s| synth_prompt(s as u64 + 1, lens[s % lens.len()])).collect();
    if let Some(f) = &forced {
        prompts[0] = f[..f.len() - 1].to_vec();
    }
    let max_len = prompts.iter().map(|p| p.len()).max().unwrap();
    eprintln!(
        "multistream harness: S={n_streams} T={n_steps} lens={:?}",
        prompts.iter().map(|p| p.len()).collect::<Vec<_>>()
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

    eprintln!("loading weights (paged experts)...");
    let t0 = std::time::Instant::now();
    let weights = HetModelWeights::load_all(src, dgpu, igpu, &rope_for_layer)?;
    eprintln!("weights loaded in {:.1} s", t0.elapsed().as_secs_f64());
    let engine = HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?;
    // MS_ENGINE2=1: the decode oracle runs on a SECOND engine instance (own
    // streams, events, graph caches, atomics), everything else shared.
    let engine_dec = if std::env::var("MS_ENGINE2").as_deref() == Ok("1") {
        Some(HeterogeneousEngine::new(dgpu, &darch, igpu, &iarch, ExecMode::HetParallel)?)
    } else {
        None
    };
    let eng_dec: &HeterogeneousEngine = engine_dec.as_ref().unwrap_or(&engine);
    let mut ds = DgpuScratch::alloc(dgpu)?;
    let mut is = IgpuScratch::alloc(igpu)?;
    // MS_DIAG=restore[:P,S] (see the restore block below) needs P + S positions.
    let restore_ps: Option<(usize, usize)> = std::env::var("MS_DIAG").ok().and_then(|d| {
        let r = d.strip_prefix("restore")?.trim_start_matches(':').to_string();
        let mut it = r.split(',').filter(|x| !x.is_empty()).map(|x| x.trim().parse::<usize>().unwrap());
        Some((it.next().unwrap_or(600), it.next().unwrap_or(500)))
    });
    let restore_need = restore_ps.map(|(p, s)| p + s).unwrap_or(0);
    let n_kv_max: u32 = (max_len.max(restore_need) + n_steps + 64).next_power_of_two().max(1024) as u32;
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

    // MS_DIAG=restore[:P,S] -- REGRESSION for KNOWN_BUGS #28 (2026-09-23): a
    // continuation (snapshot restored at P, suffix S) must match an
    // UNINTERRUPTED prefill of P+S. Under CED the replay runs only the last
    // SWA_WINDOW rows; with S > SWA_WINDOW the decoder rings used to keep the
    // previous turn's rows from `S - 128` positions back and attend them as
    // neighbours. Gate: KL(fresh || continued) on the last row below
    // MS_RESTORE_KLD (default 0.05; the only legitimate difference is the chunk
    // split). A short suffix (S = 60, the #25 path: rings kept by design) is
    // reported, not gated -- it is not expected to equal a fresh prefill.
    if let Some((plen, slen)) = restore_ps {
        let bar = env_f64("MS_RESTORE_KLD", 0.05);
        let chunk = env_usize("MS_RESTORE_CHUNK", 1024);
        let ein = ENGRAM_IN as usize;
        let mut worst = 0f64;
        let run = |toks: &[i32], pos0: usize, rows_all: &[Vec<f32>], st: &mut HetModelState,
                   bd_a: &mut BatchDgpuScratch, bi_a: &mut BatchIgpuScratch, bd_b: &mut BatchDgpuScratch, bi_b: &mut BatchIgpuScratch,
                   sd: &mut BatchDgpuShared, si: &mut BatchIgpuShared, ds: &mut DgpuScratch, pg: &mut ExpertPager| -> eyre::Result<Vec<f32>> {
            let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
            let rows: Vec<Vec<f32>> = rows_all.iter().map(|r| r[pos0 * ein..(pos0 + toks.len()) * ein].to_vec()).collect();
            let mut job = v4flash_kernels::het::forward_prefill::PrefillJob::new(toks.to_vec(), hcs, Some(rows), None, pos0 as u32, chunk)?;
            while !job.chunks_done() {
                engine.prefill_job_chunk(&mut job, bd_a, bi_a, bd_b, bi_b, sd, si, ds, st, &weights, Some(pg))?;
            }
            let l = engine.prefill_job_finish(&mut job, bd_a, bi_a, bd_b, bi_b, sd, si, ds, st, &weights, Some(pg))?;
            st.restore_compressor_lending();
            engine.dgpu.compute.synchronize()?;
            Ok(l)
        };
        for (case, (seed, s_len, gated)) in [(1u64, slen, true), (2u64, slen, true), (3u64, 60usize, false)].into_iter().enumerate() {
            let p = synth_prompt(seed * 101, plen);
            let sfx = synth_prompt(seed * 101 + 7, s_len);
            let full: Vec<i32> = p.iter().chain(sfx.iter()).copied().collect();
            let rows_all = engram.rows_for_prompt(pg.raw(), &full)?;
            let mut st_f = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            let l_fresh = run(&full, 0, &rows_all, &mut st_f, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut pg)?;
            drop(st_f);
            let mut st_c = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            let _ = run(&p, 0, &rows_all, &mut st_c, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut pg)?;
            let l_cont = run(&sfx, p.len(), &rows_all, &mut st_c, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut pg)?;
            let kl = kld(&l_fresh, &l_cont);
            eprintln!(
                "restore case {case}: P={} S={} KL(fresh||continued) {kl:.5} argmax {} / {}{}",
                p.len(), sfx.len(), argmax(&l_fresh), argmax(&l_cont), if gated { "" } else { "  (S <= SWA_WINDOW: report only)" }
            );
            if gated {
                worst = worst.max(kl);
            }
        }
        if worst > bar {
            return Err(eyre!("MS_DIAG=restore failed: worst KL(fresh||continued) {worst:.5} > {bar} (KNOWN_BUGS #28 regression?)"));
        }
        eprintln!("MS_DIAG=restore: OK (worst gated KL {worst:.5} <= {bar})");
        return Ok(());
    }

    // Arenas: one slot per stream in both; the store cap covers every stream.
    let comp_rows_cap = (n_streams as u32) * n_kv_max;
    let mut arena_alone = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_batch = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_pipe = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_stag = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_rf = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_spec = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_spec_true = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_spec2 = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_spec2_hold = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    let mut arena_spec2_unord = KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)?;
    // G5h's four arms (one lane, two lanes, forced overtake, unordered).
    let mut arena_h: Vec<KvArena> = (0..4).map(|_| KvArena::alloc(dgpu, n_streams as u32, comp_rows_cap)).collect::<eyre::Result<_>>()?;
    let mut slots_h: Vec<Vec<u32>> = vec![Vec::new(); 4];
    let mut dev_spec = RowTablesDev::alloc(dgpu, ARENA_ROWS_PER_STREAM, KV_SOURCE_LAYERS.len())?;
    let mut dev_spec_b = RowTablesDev::alloc(dgpu, ARENA_ROWS_PER_STREAM, KV_SOURCE_LAYERS.len())?;
    let mut dev_b = RowTablesDev::alloc(dgpu, n_streams as u32, KV_SOURCE_LAYERS.len())?;
    let mut dev = RowTablesDev::alloc(dgpu, n_streams as u32, KV_SOURCE_LAYERS.len())?;

    // 1. Prefill each prompt on its own state; admit into both arenas.
    let mut states: Vec<HetModelState> = Vec::new();
    let mut first_tok: Vec<i32> = Vec::new();
    let mut prefill_logits: Vec<Vec<f32>> = Vec::new();
    let mut slots_alone: Vec<u32> = Vec::new();
    let mut slots_batch: Vec<u32> = Vec::new();
    let mut slots_pipe: Vec<u32> = Vec::new();
    let mut slots_stag: Vec<u32> = Vec::new();
    let mut slots_rf: Vec<u32> = Vec::new();
    let mut slots_spec: Vec<u32> = Vec::new();
    let mut slots_spec_true: Vec<u32> = Vec::new();
    let mut slots_spec2: Vec<u32> = Vec::new();
    let mut slots_spec2_hold: Vec<u32> = Vec::new();
    let mut slots_spec2_unord: Vec<u32> = Vec::new();
    for (s, toks) in prompts.iter().enumerate() {
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let rows = engram.rows_for_prompt(pg.raw(), toks)?;
        let t = std::time::Instant::now();
        let logits = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
            &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
        )?;
        st.restore_compressor_lending();
        eprintln!("stream {s}: prefill {} tokens in {:.2} s", toks.len(), t.elapsed().as_secs_f64());
        let pos = toks.len() as u32;
        let cap = pos + n_steps as u32 + 8;
        slots_alone.push(arena_alone.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        slots_batch.push(arena_batch.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        slots_pipe.push(arena_pipe.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        slots_stag.push(arena_stag.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        slots_rf.push(arena_rf.admit_from_state(&st, cap, pos, &engine.dgpu.compute)?);
        let cap_spec = cap + ARENA_ROWS_PER_STREAM;
        slots_spec.push(arena_spec.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        slots_spec_true.push(arena_spec_true.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        slots_spec2.push(arena_spec2.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        slots_spec2_hold.push(arena_spec2_hold.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        slots_spec2_unord.push(arena_spec2_unord.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        // Admitted HERE, from the freshly prefilled state: the decode oracle
        // advances `states` later (the 2026-10-04 gate run admitted G5h's arenas
        // after it and failed "source windows 17/17 rows at pos 5").
        for (a, sl) in arena_h.iter_mut().zip(slots_h.iter_mut()) {
            sl.push(a.admit_from_state(&st, cap_spec, pos, &engine.dgpu.compute)?);
        }
        engine.dgpu.compute.synchronize()?;
        first_tok.push(if s == 0 { forced.as_ref().map(|f| *f.last().unwrap()) } else { None }.unwrap_or(argmax(&logits) as i32));
        prefill_logits.push(logits);
        states.push(st);
    }
    // MS_DIAG=kv: prefill each prompt a SECOND time and compare the two states'
    // KV bit for bit (raw windows per layer, comp rows / keys / accumulators per
    // store). Says whether two prefills of one prompt even agree before any
    // decode-vs-prefill question is asked.
    // MS_DIAG=job[:rows]: the second prefill runs through `PrefillJob` (chunk by
    // chunk, `rows` per chunk, default 3) — must be bit-identical to the one-shot
    // `forward_prefill_pipelined` in KV and in the last logits.
    let diag = std::env::var("MS_DIAG").unwrap_or_default();
    let job_rows: Option<usize> = diag.strip_prefix("job").map(|r| r.trim_start_matches(':').parse().unwrap_or(3));
    if diag == "kv" || job_rows.is_some() {
        let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
        for (s, toks) in prompts.iter().enumerate() {
            let mut st2 = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
            let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
            let rows = engram.rows_for_prompt(pg.raw(), toks)?;
            let l2 = if let Some(cr) = job_rows {
                let mut job = v4flash_kernels::het::forward_prefill::PrefillJob::new(toks.clone(), hcs.clone(), Some(rows.clone()), None, 0, cr)?;
                let mut n_chunks = 0;
                while !job.chunks_done() {
                    engine.prefill_job_chunk(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights, Some(&mut pg))?;
                    n_chunks += 1;
                }
                let l = engine.prefill_job_finish(&mut job, &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights, Some(&mut pg))?;
                eprintln!("stream {s}: PrefillJob ran {n_chunks} chunks of <= {cr} rows");
                l
            } else {
                engine.forward_prefill_pipelined(
                    &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st2, &weights,
                    &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
                )?
            };
            st2.restore_compressor_lending();
            engine.dgpu.compute.synchronize()?;
            let first_logits = &prefill_logits[s];
            let dl = max_abs_diff(first_logits, &l2);
            eprintln!("stream {s}: second prefill argmax {} (first {}); |logits diff| max {dl:.3e}; KL(first||second) {:.5}; per-layer raw-window compare:", argmax(&l2), argmax(first_logits), kld(first_logits, &l2));
            let a = &states[s];
            for l in 0..v4flash_kernels::config::N_LAYER as usize {
                let (la, lb) = (&a.layers[l], &st2.layers[l]);
                let n = la.n_raw.min(lb.n_raw) as usize;
                let mut x = vec![0u16; n * hd];
                let mut y = vec![0u16; n * hd];
                la.kv_cache.slice_view(la.raw_off as usize * hd, n * hd).copy_to_host(&mut x)?;
                lb.kv_cache.slice_view(lb.raw_off as usize * hd, n * hd).copy_to_host(&mut y)?;
                let rows_diff = (0..n).filter(|&r| x[r * hd..(r + 1) * hd] != y[r * hd..(r + 1) * hd]).count();
                let mut comp = String::new();
                if let (Some(ca), Some(cb)) = (la.compressor.as_ref(), lb.compressor.as_ref()) {
                    let nc = ca.n_comp.min(cb.n_comp) as usize;
                    let w = ca.width as usize;
                    let (Some(fa), Some(fb)) = (ca.comp_kv.f16(), cb.comp_kv.f16()) else { continue };
                    let mut cx = vec![0u16; nc * w];
                    let mut cy = vec![0u16; nc * w];
                    fa.slice_view(0, nc * w).copy_to_host(&mut cx)?;
                    fb.slice_view(0, nc * w).copy_to_host(&mut cy)?;
                    let cd = (0..nc).filter(|&r| cx[r * w..(r + 1) * w] != cy[r * w..(r + 1) * w]).count();
                    let mut sx = vec![0f32; ca.state_kv.len()];
                    let mut sy = vec![0f32; cb.state_kv.len()];
                    ca.state_kv.copy_to_host(&mut sx)?;
                    cb.state_kv.copy_to_host(&mut sy)?;
                    let sd_max = sx.iter().zip(&sy).map(|(p, q)| (p - q).abs()).fold(0f32, f32::max);
                    comp = format!("  comp n={}/{} rows_diff={cd} state max|d|={sd_max:.3e}", ca.n_comp, cb.n_comp);
                }
                if rows_diff > 0 || !comp.is_empty() || l < 3 {
                    eprintln!("  L{l:2}: raw n={}/{} off={}/{} rows_diff={rows_diff}{comp}", la.n_raw, lb.n_raw, la.raw_off, lb.raw_off);
                }
            }
        }
        return Ok(());
    }
    let st_n_raw: Vec<u32> = states.iter().map(|s| s.layers[0].n_raw).collect();
    eprintln!("admitted; raw windows {:?} (W={SWA_WINDOW})", st_n_raw);

    // 2. Decode oracle, greedy, per stream: forced continuation + logits.
    let mut cont: Vec<Vec<i32>> = vec![Vec::new(); n_streams];
    let mut logits_dec: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let nv = N_VOCAB as usize;
    for s in 0..n_streams {
        let mut seq = prompts[s].clone();
        let mut tok = first_tok[s];
        for _ in 0..n_steps {
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            let hc = embed(tok)?;
            eng_dec.forward_token_paged(&mut ds, &mut is, &mut states[s], &weights, &hc, pos as u32, tok, &mut pg, Some(&rows))?;
            eng_dec.dgpu.compute.synchronize()?;
            let mut l = vec![0f32; nv];
            ds.logits.slice_view(0, nv).copy_to_host(&mut l)?;
            cont[s].push(tok);
            tok = match (&force_cont, s) {
                (Some(c), 0) if cont[s].len() < c.len() => c[cont[s].len()],
                _ => argmax(&l) as i32,
            };
            logits_dec[s].push(l);
        }
        eprintln!("stream {s}: decode oracle tokens {:?}", cont[s]);
    }
    if std::env::var("MS_DEVSYNC").as_deref() == Ok("1") {
        dgpu.set_current()?;
        dgpu.synchronize()?;
        igpu.set_current()?;
        igpu.synchronize()?;
        dgpu.set_current()?;
        engine.invalidate_device_cache();
        eprintln!("MS_DEVSYNC: both devices synchronized after the decode oracle");
    }

    // MS_SPEC=1: hold the DSpark verify's `SpeculativeAppend` scope over the
    // continuation steps (pf1 AND the arena). It flips the batched driver's
    // eviction (none), the sparse-residency predicate, and — the suspect — the
    // pager's "count this as prefill + scan window" mode.
    let spec_guard = if std::env::var("MS_SPEC").as_deref() == Ok("1") {
        Some(v4flash_kernels::het::forward_prefill::SpeculativeAppend::begin())
    } else {
        None
    };

    // 2b. The CONTIGUOUS batched path, one token per call on a fresh prefilled
    //     state (what the DSpark verify runs at B=1): separates "arena arm
    //     wrong" (pf1 != alone) from "prefill family vs decode" (pf1 == alone).
    let mut logits_pf1: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let skip_pf1 = std::env::var("MS_SKIP_PF1").as_deref() == Ok("1");
    if skip_pf1 {
        logits_pf1 = vec![vec![vec![0f32; nv]; n_steps]; n_streams];
    }
    for s in 0..n_streams {
        if skip_pf1 { break; }
        let toks = &prompts[s];
        let mut st = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
        let hcs: Vec<Vec<f32>> = toks.iter().map(|&t| embed(t)).collect::<eyre::Result<_>>()?;
        let rows = engram.rows_for_prompt(pg.raw(), toks)?;
        let _ = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
            &hcs, toks, 0, true, None, None, None, None, Some(&mut pg), Some(&rows),
        )?;
        st.restore_compressor_lending();
        let mut seq = toks.clone();
        for t in 0..n_steps {
            let tok = cont[s][t];
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            engine.normalize_raw_windows(&mut ds, &mut st)?;
            // last_only=false: `ced_enabled() && last_only` would otherwise turn
            // CED on and replay the decoder over THIS call's rows only (the
            // one new token), wiping layers 20-39's window. The DSpark verify
            // passes false for the same reason.
            let l = engine.forward_prefill_pipelined(
                &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut ds, &mut st, &weights,
                &[embed(tok)?], &[tok], pos as u32, false, None, None, None, None, Some(&mut pg), Some(&rows),
            )?;
            st.restore_compressor_lending();
            logits_pf1[s].push(l);
            // MS_DIAG=kvrow (after step 0): compare this state's raw windows with the
            // decode state's, row by row, per layer. Rows before the step must be
            // identical (same prefill); the step's own row shows whether the two
            // paths WRITE the same K/V.
            if t == 0 && std::env::var("MS_DIAG").as_deref() == Ok("kvrow") {
                let hd = v4flash_kernels::config::N_HEAD_DIM as usize;
                let f = |u: u16| -> f32 { half_to_f32(u) };
                eprintln!("kvrow: layer  n_raw(dec/pf1)  rows_identical/total  last-row relRMSE(pf1 vs dec)");
                for l in 0..v4flash_kernels::config::N_LAYER as usize {
                    let (la, lb) = (&states[s].layers[l], &st.layers[l]);
                    let n = la.n_raw.min(lb.n_raw) as usize;
                    let mut x = vec![0u16; n * hd];
                    let mut y = vec![0u16; n * hd];
                    la.kv_cache.slice_view(la.raw_off as usize * hd, n * hd).copy_to_host(&mut x)?;
                    lb.kv_cache.slice_view(lb.raw_off as usize * hd, n * hd).copy_to_host(&mut y)?;
                    let same = (0..n).filter(|&r| x[r * hd..(r + 1) * hd] == y[r * hd..(r + 1) * hd]).count();
                    let (mut num, mut den) = (0f64, 0f64);
                    for i in (n - 1) * hd..n * hd {
                        let (a, b) = (f(x[i]) as f64, f(y[i]) as f64);
                        num += (a - b) * (a - b);
                        den += a * a;
                    }
                    if l < 4 || same != n {
                        eprintln!("kvrow: L{l:2}  {}/{}  {same}/{n}  {:.3e}", la.n_raw, lb.n_raw, (num / den.max(1e-30)).sqrt());
                    }
                }
            }
        }
    }

    if let Ok(dir) = std::env::var("MS_SAVE_LOGITS") {
        std::fs::create_dir_all(&dir)?;
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for s in 0..n_streams {
            let mut seq = prompts[s].clone();
            seq.extend_from_slice(&cont[s]);
            std::fs::write(format!("{dir}/s{s}_tokens.json"), serde_json::to_string(&seq)?)?;
            for t in 0..n_steps {
                std::fs::write(format!("{dir}/s{s}_t{t}_dec.bin"), f32s(&logits_dec[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_pf1.bin"), f32s(&logits_pf1[s][t]))?;
            }
        }
        eprintln!("MS_SAVE_LOGITS: wrote {dir}");
    }
    if std::env::var("MS_STOP_AFTER").as_deref() == Ok("pf1") {
        let mut kls = Vec::new();
        for s in 0..n_streams {
            for t in 0..n_steps {
                let kp = kld(&logits_dec[s][t], &logits_pf1[s][t]);
                eprintln!("{s:2} {t:2}  argmax dec/pf1 {:6}/{:6}  KL(dec||pf1) {kp:.5}", argmax(&logits_dec[s][t]), argmax(&logits_pf1[s][t]));
                kls.push(kp);
            }
        }
        eprintln!("MS_STOP_AFTER=pf1: KL(dec||pf1) mean {:.5}", kls.iter().sum::<f64>() / kls.len() as f64);
        return Ok(());
    }

    // MS_FRESH_PAGER=1: throw the pager away and build a new one right before
    // the arena passes, so its pool/remap carry no history from the decode
    // oracle and the contiguous reference.
    if std::env::var("MS_FRESH_PAGER").as_deref() == Ok("1") {
        drop(pg);
        pg = ExpertPager::new(V41HfWeights::open(&dir, None)?, igpu, 0)?;
        eprintln!("MS_FRESH_PAGER: pager rebuilt");
    }

    // 3. Arena, one row per step (alone).
    let mut logits_alone: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    for s in 0..n_streams {
        let mut seq = prompts[s].clone();
        for t in 0..n_steps {
            let tok = cont[s][t];
            let pos = seq.len();
            seq.push(tok);
            let rows = engram.rows_at(pg.raw(), &seq, pos)?;
            let rows_b: Vec<Vec<f32>> = rows;
            engine.forward_step_arena(
                &mut bd_a, &mut bi_a, &mut sd, &mut si, &mut arena_alone, &mut dev, &StepRows::plain(&[slots_alone[s]])?, &weights,
                &[embed(tok)?], &[tok], &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
            )?;
            arena_alone.accept(slots_alone[s], 1, &engine.dgpu.compute)?;
            let l = engine.head_rows(&mut ds, &bd_a, 1, &weights)?;
            logits_alone[s].push(l);
        }
    }

    // 3b. G5f: speculative blocks (see the module doc). (R, keep) per block,
    // cycled; R is the rows of the block, keep its kept prefix.
    let schedule: [(usize, usize); 10] = [(1, 1), (2, 2), (3, 1), (6, 4), (5, 5), (4, 2), (2, 1), (8, 6), (3, 3), (7, 3)];
    let ein_spec = ENGRAM_IN as usize;
    let mut run_spec = |arena: &mut KvArena, slots: &[u32], wrong_tail: bool| -> eyre::Result<(Vec<Vec<Vec<f32>>>, usize)> {
        let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
        let mut blocks = 0usize;
        for s in 0..n_streams {
            let mut seq = prompts[s].clone();
            let (mut t, mut blk) = (0usize, 0usize);
            while t < n_steps {
                let (r_want, k_want) = schedule[blk % schedule.len()];
                blk += 1;
                let keep = k_want.min(n_steps - t);
                let r = r_want.max(keep);
                let toks: Vec<i32> = (0..r)
                    .map(|j| match cont[s].get(t + j) {
                        Some(&c) if j < keep || !wrong_tail => c,
                        _ => ((cont[s][t] as i64 + 7919 * (j as i64 + 1)) % 100_000) as i32,
                    })
                    .collect();
                let pos = seq.len();
                let mut ext = seq.clone();
                ext.extend_from_slice(&toks);
                let mut rows_b = vec![vec![0f32; r * ein_spec]; ENGRAM_LAYERS.len()];
                for j in 0..r {
                    for (li, x) in engram.rows_at(pg.raw(), &ext, pos + j)?.iter().enumerate() {
                        rows_b[li][j * ein_spec..(j + 1) * ein_spec].copy_from_slice(x);
                    }
                }
                let hcs: Vec<Vec<f32>> = toks.iter().map(|&x| embed(x)).collect::<eyre::Result<_>>()?;
                engine.forward_step_arena(
                    &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_spec, &StepRows::chains(&[(slots[s], r - 1)])?, &weights,
                    &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b)), Some(&mut pg),
                )?;
                arena.accept(slots[s], keep as u32, &engine.dgpu.compute)?;
                let l = engine.head_rows(&mut ds, &bd_a, r, &weights)?;
                for j in 0..keep {
                    out[s].push(l[j * nv..(j + 1) * nv].to_vec());
                }
                seq.extend_from_slice(&toks[..keep]);
                t += keep;
                blocks += 1;
            }
        }
        Ok((out, blocks))
    };
    let (logits_spec, spec_blocks) = run_spec(&mut arena_spec, &slots_spec, true)?;
    let (logits_spec_true, _) = run_spec(&mut arena_spec_true, &slots_spec_true, false)?;
    eprintln!("G5f: {spec_blocks} speculative blocks over {n_streams} streams");

    // 3c. G5g: the same blocks (rejected tails included) as ORDERED two-lane
    // cuts through the ready-first driver; see the module doc.
    use std::sync::atomic::Ordering::Relaxed;
    use v4flash_kernels::het::forward_prefill::{
        take_chain_waits, READY_FIRST_TEST_HOLD_LANE0, READY_FIRST_TEST_UNORDERED,
    };
    let mut run_spec2 = |arena: &mut KvArena, slots: &[u32], hold: bool, unordered: bool| -> eyre::Result<(Vec<Vec<Vec<f32>>>, u64)> {
        let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
        let _ = take_chain_waits();
        let mut waits = 0u64;
        for s in 0..n_streams {
            let mut seq = prompts[s].clone();
            let (mut t, mut blk) = (0usize, 0usize);
            while t < n_steps {
                let (r_want, k_want) = schedule[blk % schedule.len()];
                blk += 1;
                let keep = k_want.min(n_steps - t);
                let r = r_want.max(keep);
                let toks: Vec<i32> = (0..r)
                    .map(|j| match cont[s].get(t + j) {
                        Some(&c) if j < keep => c,
                        _ => ((cont[s][t] as i64 + 7919 * (j as i64 + 1)) % 100_000) as i32,
                    })
                    .collect();
                let pos = seq.len();
                // Poison: a junk block at the same positions, NOT accepted
                // (counters do not move; only the dead state it writes stays).
                let junk: Vec<i32> = (0..r).map(|j| ((cont[s][t] as i64 * 31 + 104_729 * (j as i64 + 3)) % 100_000) as i32).collect();
                let junk_hcs: Vec<Vec<f32>> = junk.iter().map(|&x| embed(x)).collect::<eyre::Result<_>>()?;
                engine.forward_step_arena(
                    &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_spec, &StepRows::chains(&[(slots[s], r - 1)])?, &weights,
                    &junk_hcs, &junk, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(vec![vec![0f32; r * ein_spec]; ENGRAM_LAYERS.len()])), Some(&mut pg),
                )?;
                engine.dgpu.compute.synchronize()?;
                let mut ext = seq.clone();
                ext.extend_from_slice(&toks);
                let mut rows_b = vec![vec![0f32; r * ein_spec]; ENGRAM_LAYERS.len()];
                for j in 0..r {
                    for (li, x) in engram.rows_at(pg.raw(), &ext, pos + j)?.iter().enumerate() {
                        rows_b[li][j * ein_spec..(j + 1) * ein_spec].copy_from_slice(x);
                    }
                }
                let hcs: Vec<Vec<f32>> = toks.iter().map(|&x| embed(x)).collect::<eyre::Result<_>>()?;
                let l = if r >= 2 {
                    READY_FIRST_TEST_HOLD_LANE0.store(hold, Relaxed);
                    READY_FIRST_TEST_UNORDERED.store(unordered, Relaxed);
                    let res = {
                        let mut lanes: [(&mut BatchDgpuScratch, &mut BatchIgpuScratch, &mut RowTablesDev); 2] =
                            [(&mut bd_a, &mut bi_a, &mut dev_spec), (&mut bd_b, &mut bi_b, &mut dev_spec_b)];
                        engine.forward_step_arena_ready_first(
                            &mut lanes, &mut sd, &mut si, arena, &StepRows::chains(&[(slots[s], r - 1)])?, &weights,
                            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b)), Some(&mut pg),
                        )
                    };
                    READY_FIRST_TEST_HOLD_LANE0.store(false, Relaxed);
                    READY_FIRST_TEST_UNORDERED.store(false, Relaxed);
                    res?;
                    waits += take_chain_waits().0;
                    let b_a = v4flash_kernels::het::forward_prefill::lane_rows(r, 2)[0];
                    let mut l = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
                    l.extend(engine.head_rows(&mut ds, &bd_b, r - b_a, &weights)?);
                    l
                } else {
                    engine.forward_step_arena(
                        &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_spec, &StepRows::chains(&[(slots[s], r - 1)])?, &weights,
                        &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b)), Some(&mut pg),
                    )?;
                    engine.head_rows(&mut ds, &bd_a, r, &weights)?
                };
                arena.accept(slots[s], keep as u32, &engine.dgpu.compute)?;
                for j in 0..keep {
                    out[s].push(l[j * nv..(j + 1) * nv].to_vec());
                }
                seq.extend_from_slice(&toks[..keep]);
                t += keep;
            }
        }
        Ok((out, waits))
    };
    let (logits_spec2, waits_ord) = run_spec2(&mut arena_spec2, &slots_spec2, false, false)?;
    let (logits_spec2_hold, waits_hold) = run_spec2(&mut arena_spec2_hold, &slots_spec2_hold, true, false)?;
    let (logits_spec2_unord, _) = run_spec2(&mut arena_spec2_unord, &slots_spec2_unord, true, true)?;
    let (mut g5g_ord, mut g5g_hold, mut g5g_spec, mut g5g_unord) = (0usize, 0usize, 0usize, 0usize);
    for s in 0..n_streams {
        for t in 0..n_steps {
            g5g_ord += usize::from(max_abs_diff(&logits_spec2[s][t], &logits_alone[s][t]) != 0.0);
            g5g_hold += usize::from(max_abs_diff(&logits_spec2_hold[s][t], &logits_alone[s][t]) != 0.0);
            g5g_spec += usize::from(max_abs_diff(&logits_spec2[s][t], &logits_spec[s][t]) != 0.0);
            g5g_unord += usize::from(max_abs_diff(&logits_spec2_unord[s][t], &logits_alone[s][t]) != 0.0);
        }
    }
    eprintln!(
        "G5g: ordered two-lane rows differing from alone {g5g_ord} / from single-lane spec {g5g_spec}; forced-overtake {g5g_hold} \
         (Chain waits {waits_hold}, unforced {waits_ord}); UNORDERED control differs on {g5g_unord} of {} rows (want > 0)",
        n_streams * n_steps
    );
    // KL against alone (for runs that cross a lane-wide host decision, e.g. the
    // TOP_K crossing, judged under MS_ALLOW_INEXACT=1 by the same bars as G5f).
    let kls_g5g: Vec<f64> = (0..n_streams)
        .flat_map(|s| (0..n_steps).map(move |t| (s, t)))
        .flat_map(|(s, t)| [kld(&logits_alone[s][t], &logits_spec2[s][t]), kld(&logits_alone[s][t], &logits_spec2_hold[s][t])])
        .collect();
    let mean_g = kls_g5g.iter().sum::<f64>() / kls_g5g.len().max(1) as f64;
    let max_g = kls_g5g.iter().cloned().fold(0.0, f64::max);
    eprintln!("G5g: KL(alone||two-lane) mean {mean_g:.5} max {max_g:.5}");
    // The state the LAST block left (later blocks only test it indirectly): one
    // more one-row step (not accepted) from each two-lane arena must give the
    // logits it gives from the single-lane spec arena -- same blocks, same keeps.
    let mut g5g_state = 0usize;
    for s in 0..n_streams {
        let tok = cont[s][0];
        let mut probe = |arena: &mut KvArena, slot: u32| -> eyre::Result<Vec<f32>> {
            engine.forward_step_arena(
                &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_spec, &StepRows::plain(&[slot])?, &weights,
                &[embed(tok)?], &[tok], &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(vec![vec![0f32; ein_spec]; ENGRAM_LAYERS.len()])), Some(&mut pg),
            )?;
            engine.head_rows(&mut ds, &bd_a, 1, &weights)
        };
        let base = probe(&mut arena_spec, slots_spec[s])?;
        g5g_state += usize::from(max_abs_diff(&probe(&mut arena_spec2, slots_spec2[s])?, &base) != 0.0);
        g5g_state += usize::from(max_abs_diff(&probe(&mut arena_spec2_hold, slots_spec2_hold[s])?, &base) != 0.0);
    }
    eprintln!("G5g: final-state probes differing from the single-lane arena {g5g_state} of {} (want 0)", 2 * n_streams);

    // 3d. G5h: TWO STREAMS' BLOCKS IN ONE STEP (docs/v41/MS_DSPARK_STREAMS_DESIGN.md 5).
    // Streams are paired (0, 1), (2, 3), ...; each step runs both streams'
    // G5f-style blocks (WRONG tokens past the kept prefix) as one `StepRows`
    // after a junk step at the same positions that is NOT accepted. Block order
    // alternates descending / ascending row count, so the balanced two-lane cut
    // falls between the blocks, inside the first, and inside the second. Arms:
    // one lane (each block <= 4 rows: 8 per lane), two lanes through the
    // ready-first driver (ordered wherever the cut crosses a block), the forced
    // overtake (lane 0 held), and the UNORDERED control (must differ).
    #[derive(Clone, Copy, PartialEq)]
    enum H {
        One,
        Two { hold: bool, unordered: bool },
    }
    let mut dev_h = RowTablesDev::alloc(dgpu, 2 * ARENA_ROWS_PER_STREAM, KV_SOURCE_LAYERS.len())?;
    let mut dev_h_b = RowTablesDev::alloc(dgpu, 2 * ARENA_ROWS_PER_STREAM, KV_SOURCE_LAYERS.len())?;
    let mut run_pairs = |arena: &mut KvArena, slots: &[u32], mode: H| -> eyre::Result<(Vec<Vec<Vec<f32>>>, u64, [usize; 3])> {
        let mut out: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
        let _ = take_chain_waits();
        // Two-lane cuts: between the blocks, through the first, through the second.
        let (mut waits, mut cuts) = (0u64, [0usize; 3]);
        for p in (0..n_streams).step_by(2) {
            let pair: Vec<usize> = if p + 1 < n_streams { vec![p, p + 1] } else { vec![p] };
            let mut seqs: Vec<Vec<i32>> = pair.iter().map(|&s| prompts[s].clone()).collect();
            let mut t = vec![0usize; pair.len()];
            let mut blk = 0usize;
            while (0..pair.len()).any(|k| t[k] < n_steps) {
                // (stream, rows, keep) of each block still running; the second
                // stream's schedule is offset by ONE block, which covers every
                // two-lane cut kind from MS_STEPS=4 up (offline walk, review).
                let mut blocks: Vec<(usize, usize, usize)> = Vec::new();
                for (k, &s) in pair.iter().enumerate() {
                    if t[k] >= n_steps {
                        continue;
                    }
                    let (r_want, k_want) = schedule[(blk + k) % schedule.len()];
                    let keep = k_want.min(n_steps - t[k]);
                    let (r, keep) = if mode == H::One { (r_want.max(keep).min(4), keep.min(4)) } else { (r_want.max(keep), keep) };
                    blocks.push((k, r, keep));
                    let _ = s;
                }
                if blk % 2 == 0 {
                    blocks.sort_by_key(|b| std::cmp::Reverse(b.1));
                } else {
                    blocks.sort_by_key(|b| b.1);
                }
                blk += 1;
                let rows = StepRows::chains(&blocks.iter().map(|&(k, r, _)| (slots[pair[k]], r - 1)).collect::<Vec<_>>())?;
                let total: usize = blocks.iter().map(|b| b.1).sum();
                let mut toks: Vec<i32> = Vec::with_capacity(total);
                let mut junk: Vec<i32> = Vec::with_capacity(total);
                let mut rows_b = vec![vec![0f32; total * ein_spec]; ENGRAM_LAYERS.len()];
                for &(k, r, keep) in &blocks {
                    let s = pair[k];
                    let block: Vec<i32> = (0..r)
                        .map(|j| match cont[s].get(t[k] + j) {
                            Some(&c) if j < keep => c,
                            _ => ((cont[s][t[k]] as i64 + 7919 * (j as i64 + 1)) % 100_000) as i32,
                        })
                        .collect();
                    let mut ext = seqs[k].clone();
                    ext.extend_from_slice(&block);
                    let base = toks.len();
                    for j in 0..r {
                        for (li, x) in engram.rows_at(pg.raw(), &ext, seqs[k].len() + j)?.iter().enumerate() {
                            rows_b[li][(base + j) * ein_spec..(base + j + 1) * ein_spec].copy_from_slice(x);
                        }
                    }
                    toks.extend_from_slice(&block);
                    junk.extend((0..r).map(|j| ((cont[s][t[k]] as i64 * 31 + 104_729 * (j as i64 + 3)) % 100_000) as i32));
                }
                // Poison: the same shape with junk tokens, not accepted.
                let junk_hcs: Vec<Vec<f32>> = junk.iter().map(|&x| embed(x)).collect::<eyre::Result<_>>()?;
                engine.forward_step_arena(
                    &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_h, &rows, &weights,
                    &junk_hcs, &junk, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(vec![vec![0f32; total * ein_spec]; ENGRAM_LAYERS.len()])), Some(&mut pg),
                )?;
                engine.dgpu.compute.synchronize()?;
                let hcs: Vec<Vec<f32>> = toks.iter().map(|&x| embed(x)).collect::<eyre::Result<_>>()?;
                let l = match mode {
                    H::Two { hold, unordered } if total >= 2 => {
                        let b_a = v4flash_kernels::het::forward_prefill::lane_rows(total, 2)[0];
                        if blocks.len() == 2 {
                            cuts[if b_a == blocks[0].1 { 0 } else if b_a < blocks[0].1 { 1 } else { 2 }] += 1;
                        }
                        READY_FIRST_TEST_HOLD_LANE0.store(hold, Relaxed);
                        READY_FIRST_TEST_UNORDERED.store(unordered, Relaxed);
                        let res = {
                            let mut lanes: [(&mut BatchDgpuScratch, &mut BatchIgpuScratch, &mut RowTablesDev); 2] =
                                [(&mut bd_a, &mut bi_a, &mut dev_h), (&mut bd_b, &mut bi_b, &mut dev_h_b)];
                            engine.forward_step_arena_ready_first(
                                &mut lanes, &mut sd, &mut si, arena, &rows, &weights,
                                &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b)), Some(&mut pg),
                            )
                        };
                        READY_FIRST_TEST_HOLD_LANE0.store(false, Relaxed);
                        READY_FIRST_TEST_UNORDERED.store(false, Relaxed);
                        res?;
                        waits += take_chain_waits().0;
                        let mut l = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
                        l.extend(engine.head_rows(&mut ds, &bd_b, total - b_a, &weights)?);
                        l
                    }
                    _ => {
                        engine.forward_step_arena(
                            &mut bd_a, &mut bi_a, &mut sd, &mut si, arena, &mut dev_h, &rows, &weights,
                            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b)), Some(&mut pg),
                        )?;
                        engine.head_rows(&mut ds, &bd_a, total, &weights)?
                    }
                };
                for &(k, _, keep) in &blocks {
                    let s = pair[k];
                    let r0 = rows.root_of(slots[s]).ok_or_else(|| eyre!("G5h: no root row for stream {s}"))?;
                    arena.accept(slots[s], keep as u32, &engine.dgpu.compute)?;
                    for j in 0..keep {
                        out[s].push(l[(r0 + j) * nv..(r0 + j + 1) * nv].to_vec());
                    }
                    let start = seqs[k].len() - prompts[s].len();
                    seqs[k].extend_from_slice(&cont[s][start..start + keep]);
                    t[k] += keep;
                }
            }
        }
        Ok((out, waits, cuts))
    };
    let (logits_h1, _, _) = run_pairs(&mut arena_h[0], &slots_h[0], H::One)?;
    let (logits_h2, waits_h2, cuts_h2) = run_pairs(&mut arena_h[1], &slots_h[1], H::Two { hold: false, unordered: false })?;
    let (logits_h2_hold, waits_h2_hold, _) = run_pairs(&mut arena_h[2], &slots_h[2], H::Two { hold: true, unordered: false })?;
    let (logits_h2_unord, _, _) = run_pairs(&mut arena_h[3], &slots_h[3], H::Two { hold: true, unordered: true })?;
    let (mut g5h_one, mut g5h_two, mut g5h_hold, mut g5h_unord) = (0usize, 0usize, 0usize, 0usize);
    for s in 0..n_streams {
        for t in 0..n_steps {
            g5h_one += usize::from(max_abs_diff(&logits_h1[s][t], &logits_alone[s][t]) != 0.0);
            g5h_two += usize::from(max_abs_diff(&logits_h2[s][t], &logits_alone[s][t]) != 0.0);
            g5h_hold += usize::from(max_abs_diff(&logits_h2_hold[s][t], &logits_alone[s][t]) != 0.0);
            g5h_unord += usize::from(max_abs_diff(&logits_h2_unord[s][t], &logits_alone[s][t]) != 0.0);
        }
    }
    let kls_g5h: Vec<f64> = (0..n_streams)
        .flat_map(|s| (0..n_steps).map(move |t| (s, t)))
        .flat_map(|(s, t)| [kld(&logits_alone[s][t], &logits_h1[s][t]), kld(&logits_alone[s][t], &logits_h2[s][t]), kld(&logits_alone[s][t], &logits_h2_hold[s][t])])
        .collect();
    let mean_h = kls_g5h.iter().sum::<f64>() / kls_g5h.len().max(1) as f64;
    let max_h = kls_g5h.iter().cloned().fold(0.0, f64::max);
    if n_streams < 2 {
        eprintln!("G5h NOT RUN: two blocks per step need MS_STREAMS >= 2 (the arms ran one block per step, i.e. G5f / G5g again)");
    }
    eprintln!(
        "G5h: two blocks per step, rows differing from alone: one lane {g5h_one}, two lanes {g5h_two}, forced-overtake {g5h_hold} \
         (Chain waits {waits_h2_hold}, unforced {waits_h2}); UNORDERED control differs on {g5h_unord} of {} rows (want > 0); \
         two-lane cuts between/through first/through second {cuts_h2:?}; KL(alone||G5h) mean {mean_h:.5} max {max_h:.5}",
        n_streams * n_steps
    );

    // 4. Arena, all rows co-batched.
    let mut logits_batch: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs: Vec<Vec<i32>> = prompts.clone();
    let ein = ENGRAM_IN as usize;
    let mut step_ms = Vec::new();
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs[s].len();
            seqs[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        engine.forward_step_arena(
            &mut bd_a, &mut bi_a, &mut sd, &mut si, &mut arena_batch, &mut dev, &StepRows::plain(&slots_batch)?, &weights,
            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
        )?;
        for &sl in &slots_batch { arena_batch.accept(sl, 1, &engine.dgpu.compute)?; }
        let all = engine.head_rows(&mut ds, &bd_a, n_streams, &weights)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_batch[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("batched step wall (S={n_streams}, incl. head): {:?} ms", step_ms.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 4b. Arena, all rows co-batched, TWO-LANE PIPELINED driver (the production
    // step at >= 6 rows). Since 2026-09-22 this driver runs the pre-MoE phases
    // in dependency-graph order (chain A, chain B, route A, route B, prep A,
    // prep B, launch A, launch B) with the cross-lane iGPU drain removed; the
    // invariant it relies on -- no MoE in flight during `ensure` -- is checked
    // inside the driver, and THIS comparison is what says the reorder kept the
    // numerics: same rows, same tokens, must match `alone` like `batch` does.
    let mut logits_pipe: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs_p: Vec<Vec<i32>> = prompts.clone();
    let mut step_ms_p = Vec::new();
    let b_a = n_streams.div_ceil(2);
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs_p[s].len();
            seqs_p[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs_p[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        engine.forward_step_arena_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut arena_pipe, &mut dev, &mut dev_b, &StepRows::plain(&slots_pipe)?, &weights,
            &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
        )?;
        for &sl in &slots_pipe { arena_pipe.accept(sl, 1, &engine.dgpu.compute)?; }
        let mut all = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
        all.extend(engine.head_rows(&mut ds, &bd_b, n_streams - b_a, &weights)?);
        step_ms_p.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_pipe[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("pipelined step wall (S={n_streams}, 2 lanes, incl. head): {:?} ms", step_ms_p.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 4c. Arena, all rows co-batched, TWO-LANE STAGGERED driver
    // (`forward_step_arena_lanes` with 2 lanes): per layer post(A,L) -> pre(A,L+1)
    // -> post(B,L) -> pre(B,L+1), so lane A's next box-2 request is out while
    // box 1 still works on lane B -- the lanes are offset by half a layer instead
    // of in lockstep. The lanes sit on DIFFERENT layers at the same time here,
    // which the lockstep driver never does, so this is its own gate (G5d).
    let mut logits_stag: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs_s: Vec<Vec<i32>> = prompts.clone();
    let mut step_ms_s = Vec::new();
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs_s[s].len();
            seqs_s[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs_s[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        {
            let mut lanes: [(&mut BatchDgpuScratch, &mut BatchIgpuScratch, &mut RowTablesDev); 2] =
                [(&mut bd_a, &mut bi_a, &mut dev), (&mut bd_b, &mut bi_b, &mut dev_b)];
            engine.forward_step_arena_lanes(
                &mut lanes, &mut sd, &mut si, &mut arena_stag, &StepRows::plain(&slots_stag)?, &weights,
                &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
            )?;
        }
        for &sl in &slots_stag { arena_stag.accept(sl, 1, &engine.dgpu.compute)?; }
        let mut all = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
        all.extend(engine.head_rows(&mut ds, &bd_b, n_streams - b_a, &weights)?);
        step_ms_s.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_stag[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("staggered step wall (S={n_streams}, 2 lanes, incl. head): {:?} ms", step_ms_s.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    // 4d. Arena, all rows co-batched, TWO-LANE READY-FIRST driver (G5e): the staggered
    // lanes with the host running whichever lane is ready first; the interleave
    // varies with timing, so this arm is the one that proves it cannot matter.
    // (Original 4c notes:
    // (`forward_step_arena_lanes` with 2 lanes): per layer post(A,L) -> pre(A,L+1)
    // -> post(B,L) -> pre(B,L+1), so lane A's next box-2 request is out while
    // box 1 still works on lane B -- the lanes are offset by half a layer instead
    // of in lockstep. The lanes sit on DIFFERENT layers at the same time here,
    // which the lockstep driver never does, so this is its own gate (G5d).
    let mut logits_rf: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_streams];
    let mut seqs_rf: Vec<Vec<i32>> = prompts.clone();
    let mut step_ms_rf = Vec::new();
    for t in 0..n_steps {
        let toks: Vec<i32> = (0..n_streams).map(|s| cont[s][t]).collect();
        let mut rows_b = vec![vec![0f32; n_streams * ein]; ENGRAM_LAYERS.len()];
        let mut hcs = Vec::with_capacity(n_streams);
        for s in 0..n_streams {
            let pos = seqs_rf[s].len();
            seqs_rf[s].push(toks[s]);
            let rows = engram.rows_at(pg.raw(), &seqs_rf[s], pos)?;
            for (li, r) in rows.iter().enumerate() {
                rows_b[li][s * ein..(s + 1) * ein].copy_from_slice(r);
            }
            hcs.push(embed(toks[s])?);
        }
        let t0 = std::time::Instant::now();
        {
            let mut lanes: [(&mut BatchDgpuScratch, &mut BatchIgpuScratch, &mut RowTablesDev); 2] =
                [(&mut bd_a, &mut bi_a, &mut dev), (&mut bd_b, &mut bi_b, &mut dev_b)];
            engine.forward_step_arena_ready_first(
                &mut lanes, &mut sd, &mut si, &mut arena_rf, &StepRows::plain(&slots_rf)?, &weights,
                &hcs, &toks, &mut v4flash_kernels::het::forward_prefill::LazyEngramRows::ready(Some(rows_b.clone())), Some(&mut pg),
            )?;
        }
        for &sl in &slots_rf { arena_rf.accept(sl, 1, &engine.dgpu.compute)?; }
        let mut all = engine.head_rows(&mut ds, &bd_a, b_a, &weights)?;
        all.extend(engine.head_rows(&mut ds, &bd_b, n_streams - b_a, &weights)?);
        step_ms_rf.push(t0.elapsed().as_secs_f64() * 1e3);
        for s in 0..n_streams {
            logits_rf[s].push(all[s * nv..(s + 1) * nv].to_vec());
        }
    }
    eprintln!("ready-first step wall (S={n_streams}, 2 lanes, incl. head): {:?} ms", step_ms_rf.iter().map(|x| (*x * 10.0).round() / 10.0).collect::<Vec<_>>());

    if let Ok(dir) = std::env::var("MS_SAVE_LOGITS") {
        let f32s = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for s in 0..n_streams {
            for t in 0..n_steps {
                std::fs::write(format!("{dir}/s{s}_t{t}_alone.bin"), f32s(&logits_alone[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_batch.bin"), f32s(&logits_batch[s][t]))?;
                std::fs::write(format!("{dir}/s{s}_t{t}_pipe.bin"), f32s(&logits_pipe[s][t]))?;
            }
        }
    }

    drop(spec_guard);

    // 5. Compare.
    let mut g5a_fail = 0;
    let mut g5c_fail = 0;
    let mut g5d_fail = 0;
    let mut g5e_fail = 0;
    let mut kls_stag = Vec::new();
    let mut kls = Vec::new();
    let mut kls_alone = Vec::new();
    let mut kls_pipe = Vec::new();
    eprintln!(" s  t  lane  |alone-batch|  |alone-pipe|  argmax(dec/alone/batch/pipe)   KL(dec||batch)  KL(dec||pipe)  |alone-stag|");
    for s in 0..n_streams {
        for t in 0..n_steps {
            let d = max_abs_diff(&logits_alone[s][t], &logits_batch[s][t]);
            let dp = max_abs_diff(&logits_pf1[s][t], &logits_alone[s][t]);
            let kb = kld(&logits_dec[s][t], &logits_batch[s][t]);
            let ka = kld(&logits_dec[s][t], &logits_alone[s][t]);
            let kp = kld(&logits_dec[s][t], &logits_pf1[s][t]);
            let (ad, ap, aa, ab) = (argmax(&logits_dec[s][t]), argmax(&logits_pf1[s][t]), argmax(&logits_alone[s][t]), argmax(&logits_batch[s][t]));
            if d != 0.0 {
                g5a_fail += 1;
            }
            let dpipe = max_abs_diff(&logits_alone[s][t], &logits_pipe[s][t]);
            let kpipe = kld(&logits_dec[s][t], &logits_pipe[s][t]);
            if dpipe != 0.0 {
                g5c_fail += 1;
            }
            let dstag = max_abs_diff(&logits_alone[s][t], &logits_stag[s][t]);
            // G5d's discriminating comparison is against the LOCKSTEP pipelined
            // driver (same rows, same two-lane split, same box-2 batched path):
            // `alone` runs 1-row steps, which with the two-box split take box 2's
            // per-token decode kernels and are not bit-identical to any multi-row
            // arm (G5a fails the same way there), so it cannot isolate the lanes.
            let dsp = max_abs_diff(&logits_pipe[s][t], &logits_stag[s][t]);
            let drf = max_abs_diff(&logits_pipe[s][t], &logits_rf[s][t]);
            if drf != 0.0 {
                g5e_fail += 1;
            }
            if dsp != 0.0 {
                g5d_fail += 1;
            }
            eprintln!("   G5d/e row s={s} t={t} lane={}: |pipe-stag| {dsp:.3e}  |batch-stag| {:.3e}  |pipe-rf| {drf:.3e}", if s < b_a { "A" } else { "B" }, max_abs_diff(&logits_batch[s][t], &logits_stag[s][t]));
            kls_stag.push(kld(&logits_dec[s][t], &logits_stag[s][t]));
            kls.push(kb);
            kls_alone.push(ka);
            kls_pipe.push(kpipe);
            let _ = (dp, ap, ka, kp);
            // `lane` is which pipeline lane the row belongs to (rows [0, b_a) =
            // A): a G5c failure that is entirely one lane means shared scratch
            // clobbered across lanes, which is how the 2026-09-22 regression
            // presented (12 of 24 rows = one lane).
            let lane = if s < b_a { "A" } else { "B" };
            eprintln!("{s:2} {t:2}   {lane}    {d:10.3e}    {dpipe:10.3e}   {ad:6}/{aa:6}/{ab:6}/{:6}     {kb:9.5}      {kpipe:9.5}    {dstag:10.3e}", argmax(&logits_pipe[s][t]));
        }
    }
    let mean = kls.iter().sum::<f64>() / kls.len() as f64;
    let max = kls.iter().cloned().fold(0.0, f64::max);
    let mean_a = kls_alone.iter().sum::<f64>() / kls_alone.len() as f64;
    eprintln!(
        "G5a: {} of {} (stream, step) rows differ between alone and batched (want 0)",
        g5a_fail,
        kls.len()
    );
    eprintln!("G5b: KL(dec||batch) mean {mean:.5} max {max:.5} nats (bars {kld_mean_bar} / {kld_max_bar}); KL(dec||alone) mean {mean_a:.5}");
    let mean_p = kls_pipe.iter().sum::<f64>() / kls_pipe.len() as f64;
    let max_p = kls_pipe.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "G5c: {} of {} (stream, step) rows differ between alone and PIPELINED (want 0); KL(dec||pipe) mean {mean_p:.5} max {max_p:.5}",
        g5c_fail,
        kls_pipe.len()
    );
    let mean_s = kls_stag.iter().sum::<f64>() / kls_stag.len() as f64;
    let max_s = kls_stag.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "G5d: {} of {} (stream, step) rows differ between PIPELINED and STAGGERED (want 0); KL(dec||stag) mean {mean_s:.5} max {max_s:.5}",
        g5d_fail,
        kls_stag.len()
    );
    eprintln!("G5e: {} of {} (stream, step) rows differ between PIPELINED and READY-FIRST (want 0)", g5e_fail, kls_stag.len());
    let (mut g5f_leak, mut g5f_alone) = (0usize, 0usize);
    let mut kls_spec = Vec::new();
    for s in 0..n_streams {
        for t in 0..n_steps {
            let dl = max_abs_diff(&logits_spec[s][t], &logits_spec_true[s][t]);
            let da = max_abs_diff(&logits_spec[s][t], &logits_alone[s][t]);
            let k = kld(&logits_alone[s][t], &logits_spec[s][t]);
            g5f_leak += usize::from(dl != 0.0);
            g5f_alone += usize::from(da != 0.0);
            kls_spec.push(k);
            if dl != 0.0 || da != 0.0 {
                eprintln!("   G5f row s={s} t={t}: |spec-spec_true| {dl:.3e}  |spec-alone| {da:.3e}  KL(alone||spec) {k:.5}  argmax spec/alone {}/{}",
                    argmax(&logits_spec[s][t]), argmax(&logits_alone[s][t]));
            }
        }
    }
    let mean_f = kls_spec.iter().sum::<f64>() / kls_spec.len() as f64;
    let max_f = kls_spec.iter().cloned().fold(0.0, f64::max);
    eprintln!(
        "G5f: {g5f_leak} of {} kept rows differ between a rejected tail and a correct one (want 0); {g5f_alone} differ from alone; KL(alone||spec) mean {mean_f:.5} max {max_f:.5}",
        kls_spec.len()
    );
    // Inexact mode (two-box split: the tail's routing changes the kept rows'
    // expert batches on box 2): a leak still shows as a large KL between arms.
    let leak_kl = (0..n_streams).flat_map(|s| (0..n_steps).map(move |t| (s, t)))
        .map(|(s, t)| kld(&logits_spec_true[s][t], &logits_spec[s][t])).fold(0.0, f64::max);
    eprintln!("G5f: max KL(spec_true||spec) {leak_kl:.5}");
    if g5f_leak > 0 && (!allow_inexact || leak_kl > kld_mean_bar) {
        return Err(eyre!("G5f failed: {g5f_leak} kept rows depend on the REJECTED rows of their block or an earlier one (rollback leaks; max KL {leak_kl:.5})"));
    }
    if g5f_alone > 0 && !allow_inexact {
        return Err(eyre!("G5f failed: {g5f_alone} speculative rows differ from one-row steps (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if (g5g_ord > 0 || g5g_hold > 0 || g5g_spec > 0 || g5g_state > 0) && !allow_inexact {
        return Err(eyre!("G5g failed: ordered two-lane verify rows differ (alone {g5g_ord}, forced-overtake {g5g_hold}, single-lane spec {g5g_spec}, final state {g5g_state})"));
    }
    if mean_g > kld_mean_bar || max_g > kld_max_bar {
        return Err(eyre!("G5g failed: two-lane KL(alone||spec2) mean {mean_g:.5} / max {max_g:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if waits_hold == 0 {
        return Err(eyre!("G5g failed: the forced overtake recorded no Chain waits -- the ordering was never exercised"));
    }
    if g5g_unord == 0 {
        return Err(eyre!("G5g failed: the UNORDERED control matched alone -- the gate does not exercise the cross-lane dependency"));
    }
    if (g5h_one > 0 || g5h_two > 0 || g5h_hold > 0) && !allow_inexact {
        return Err(eyre!("G5h failed: rows of two blocks in one step differ from one-row steps (one lane {g5h_one}, two lanes {g5h_two}, forced-overtake {g5h_hold})"));
    }
    if allow_inexact && (mean_h > kld_mean_bar || max_h > kld_max_bar) {
        return Err(eyre!("G5h failed: KL(alone||G5h) mean {mean_h:.5} / max {max_h:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if n_streams >= 2 {
        if cuts_h2.iter().any(|&c| c == 0) {
            return Err(eyre!("G5h failed: the schedule did not cover every two-lane cut (between / through first / through second = {cuts_h2:?})"));
        }
        if waits_h2_hold == 0 {
            return Err(eyre!("G5h failed: the forced overtake recorded no Chain waits -- the ordering was never exercised"));
        }
        if g5h_unord == 0 {
            return Err(eyre!("G5h failed: the UNORDERED control matched alone -- the gate does not exercise the cross-lane dependency"));
        }
    }
    if mean_f > kld_mean_bar || max_f > kld_max_bar {
        return Err(eyre!("G5f failed: speculative KL(alone||spec) mean {mean_f:.5} / max {max_f:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if g5a_fail > 0 && !allow_inexact {
        return Err(eyre!("G5a failed: {g5a_fail} rows not batch-invariant (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean > kld_mean_bar || max > kld_max_bar {
        return Err(eyre!("G5b failed: KL mean {mean:.5} / max {max:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if g5c_fail > 0 && !allow_inexact {
        return Err(eyre!("G5c failed: {g5c_fail} rows not invariant between alone and the pipelined driver (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean_p > kld_mean_bar || max_p > kld_max_bar {
        return Err(eyre!("G5c failed: pipelined KL mean {mean_p:.5} / max {max_p:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    if g5d_fail > 0 && !allow_inexact {
        return Err(eyre!("G5d failed: {g5d_fail} rows differ between the lockstep and STAGGERED drivers (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if g5e_fail > 0 && !allow_inexact {
        return Err(eyre!("G5e failed: {g5e_fail} rows differ between the lockstep and READY-FIRST drivers (MS_ALLOW_INEXACT=1 to report only)"));
    }
    if mean_s > kld_mean_bar || max_s > kld_max_bar {
        return Err(eyre!("G5d failed: staggered KL mean {mean_s:.5} / max {max_s:.5} over bars {kld_mean_bar} / {kld_max_bar}"));
    }
    Ok(())
}
