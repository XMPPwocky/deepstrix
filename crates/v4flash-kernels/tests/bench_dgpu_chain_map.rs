//! dGPU decode chain MAP (fusion-group survey, 2026-10-13): floors, every per-lane-layer dGPU op of
//! the production arena decode in isolation, and a REPLAY of one lane-layer's dGPU chain (same
//! launchers, same order, same launch geometry as `het/forward_prefill.rs::pre_moe_chain` +
//! `issue_shared_expert_prefill` + `forward_layer_post_moe_v2` under the production env: graphs OFF,
//! V41_DEC_FUSE / V41_MHC_FAST / V41_SHARED_FUSED / V41_ATTN_DEC_FUSED / V41_DGPU_ROUTER_X2 /
//! V41_DGPU_COMBINE3 / V41_RB_PACK on, V41_SUB=3 prior, V41_REMOTE_SPLIT=1, k1 look-ahead,
//! V41_INDEX_K=1, V41_CANDIDATE_POOL=1). Synthetic weights, no model load, read-only w.r.t. production.
//!
//! Sections (BENCH_SECTION=floors|ops|chain|all, default all):
//!   floors : streaming read BW (cold = MALL flushed / warm), empty-kernel launch slope (device us per
//!            dependent launch on one stream), host enqueue cost per call, small H2D copies, a cross-
//!            stream event wait.
//!   ops    : every op of the lane-layer chain alone, at BENCH_B rows (default 2,4,8), queued behind a
//!            ~50 us spin (device time only) with the MALL flushed first (cold, production-like: 80
//!            lane-layers x ~180 MB of weights evict the 64 MB infinity cache) and warm (repeated).
//!   chain  : the chain classes back to back -- reuse layer (30 of 40), index-source layer (4: 24..36),
//!            KV+index source layer firing its compressor (layers 2/8/14) -- queued wall (device time
//!            incl. every inter-kernel gap), host-paced wall (idle stream, host enqueues as it goes),
//!            host enqueue time, vs the SUM of the same ops timed alone (warm, same cache state: every
//!            weight aliases one 46 MB slice that stays MALL-resident). wall - sum = launch/gap cost.
//!            An identical second copy of the reuse chain gives the A/A spread; every other class is
//!            reported as a PAIRED difference against the production reuse chain of the same round:
//!              reuse_zc    : box 2's partial and the cache prior read zero-copy (no H2D copies)
//!              reuse_noev  : without the selected_ready event record
//!              reuse_min   : PROXY of the fused design -- only the 12 launches that survive fusion
//!                            (absorbed epilogue work not added back: an upper bound on the saving)
//!              route:*     : only the ops up to selected_ready (the router critical path)
//!              probe_*     : launch floors in the same harness (N empty / 80-WG launches, 8 small
//!                            pinned / pageable H2D, 8 event records, 8 x 80 KB pinned H2D)
//!            BENCH_CLASSES=a,b,... restricts the classes (reuse always runs).
//!
//! Every round runs all items in a RANDOM order (live hub drift hits all alike) and sleeps a few ms
//! between rounds (bench duty on the shared dGPU stays low). Report: p10 / median / p90 in us. Beside a
//! busy hub the isolated-op event timings are inflated by queue contention; the kernel durations come
//! from rocprofv3 (below) and the walls from the paired chain classes.
//!
//! VRAM: ~130 MB (a 100 MB region = weights at its head + the MALL-flush range behind them, plus
//! scratch). The live hub leaves only ~240-450 MB free on the dGPU: do not raise BENCH_POS above 65536.
//!
//!   HIP_VISIBLE_DEVICES=0 (= gfx1201 on box 1) HIP_FORCE_DEV_KERNARG=1 DEBUG_CLR_KERNARG_HDP_FLUSH_WA=1 \
//!   DEBUG_CLR_KERNARG_DEV_ARCHS=gfx1201 BENCH_SECTION=all BENCH_ROUNDS=150 \
//!   CARGO_TARGET_DIR=target-fuse nix develop -c cargo test --release --features v41 \
//!     -p v4flash-kernels --test bench_dgpu_chain_map -- --ignored --nocapture
//!
//! rocprofv3 --kernel-trace --memory-copy-trace on THIS process only (ROCR_VISIBLE_DEVICES=0 = the dGPU;
//! never the hub, never ATT/thread trace on the iGPU) gives the kernel durations and the gaps between
//! them: BENCH_SECTION=chain BENCH_ROUNDS=30 BENCH_LABELS=<file>. Each queued sample starts with ONE
//! spin kernel (`fma_f32_throughput_probe`); the k-th spin-delimited group of the trace is the k-th
//! label in <file> (align from the end: calibration spins come first; drop `ceil_read_v4` = flush).
#![cfg(feature = "v41")]
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, PinnedBuffer, Stream};
use v4flash_kernels::config::{
    BLOCKS_Q8K_GATE_IN, EXPERT_WEIGHT_SCALE, HC_DIM, HC_MIX_DIM, INDEXER_TOP_K, N_EMBD, N_EXPERT,
    N_EXPERT_USED, N_FF_SHARED, N_HC, N_HEAD, N_HEAD_DIM, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_LORA_Q,
    N_ROT, RMS_EPS, SINKHORN_EPS, SINKHORN_ITERS, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::mhc_arena::{FastCollapse, FastMix, MIX_NORMED, MIX_PRE_SCALED};
use v4flash_kernels::readback_pack::{PackPlan, PackSeg};
use v4flash_kernels::router_topk::RouterEx;
use v4flash_kernels::wmma_probe::WmmaProbe;
use v4flash_kernels::{RopeParams, ATTN_MIXED_MAX_KEYS, E2M1_KEY_ROW_BYTES};

const MB: usize = 1 << 20;
const BMAX: usize = 8;
const ROUTER_WEIGHT_EPS: f32 = 6.103515625e-5;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
const GROUP_DIM: u32 = 4096;
const RANK: u32 = 1024;
const N_GROUPS: u32 = 8;
const OUT_LOW: u32 = N_GROUPS * RANK; // 8192
const COMP_W: u32 = 512; // ratio-2 / ratio-1 compressor width
const ENGRAM_IN: u32 = 24 * 256;
/// The Engram wkv is [25600, 6144] Q8_0 = 167 MB: too big beside the hub. Measured at a quarter of
/// its rows (41.8 MB, same K) and scaled x4 (a B-packed GEMV is linear in rows at these sizes).
const ENGRAM_OUT_Q: u32 = (N_HC + 1) * N_EMBD / 4;
/// Region: weights alias [0, W_SPAN); the MALL flush reads [W_SPAN, REGION).
const REGION: usize = 100 * MB;
const W_SPAN: usize = 46 * MB;

fn q8_bytes(rows: u32, k: u32) -> usize {
    rows as usize * (k as usize / 32) * 34
}
fn f16_bytes(rows: u32, k: u32) -> usize {
    rows as usize * k as usize * 2
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn unit(&mut self) -> f32 {
        (self.next() & 0xffffff) as f32 / 16777216.0
    }
    fn sym(&mut self, a: f32) -> f32 {
        (self.unit() * 2.0 - 1.0) * a
    }
}

fn pct(v: &[f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[((s.len() - 1) as f64 * p).round() as usize]
}

fn dev_f32(id: i32, h: &[f32]) -> eyre::Result<DeviceBuffer<f32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn dev_rand(id: i32, n: usize, rng: &mut Lcg, a: f32) -> eyre::Result<DeviceBuffer<f32>> {
    dev_f32(id, &(0..n).map(|_| rng.sym(a)).collect::<Vec<_>>())
}
fn dev_pos(id: i32, n: usize, rng: &mut Lcg) -> eyre::Result<DeviceBuffer<f32>> {
    dev_f32(id, &(0..n).map(|_| 0.2 + rng.unit()).collect::<Vec<_>>())
}
fn dev_i32(id: i32, h: &[i32]) -> eyre::Result<DeviceBuffer<i32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn dev_u32(id: i32, h: &[u32]) -> eyre::Result<DeviceBuffer<u32>> {
    let mut d = DeviceBuffer::new(id, h.len())?;
    d.copy_from_host(h)?;
    Ok(d)
}
fn zeros<T: Copy + Default>(id: i32, n: usize) -> eyre::Result<DeviceBuffer<T>> {
    let mut d = DeviceBuffer::new(id, n)?;
    d.copy_from_host(&vec![T::default(); n])?;
    Ok(d)
}
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x7f_ffff;
    if exp <= 0 {
        return sign;
    }
    sign | ((exp as u32) << 10 | (mant >> 13)) as u16
}

// ---------------------------------------------------------------------------------------------
// Timing rig: one stream, one event pair, a single-wave spin kernel, a MALL flush.
// ---------------------------------------------------------------------------------------------
struct Rig {
    s: Stream,
    side: Stream,
    ea: Event,
    eb: Event,
    probe: WmmaProbe,
    ceil: DeviceCeilings,
    sink: DeviceBuffer<u32>,
    spin_out: DeviceBuffer<f32>,
    spin_a: DeviceBuffer<f32>,
    spin_b: DeviceBuffer<f32>,
    spin_iters: u32,
    spin_long: u32,
}

impl Rig {
    fn new(id: i32, arch: &str) -> eyre::Result<Self> {
        let mut r = Self {
            s: Stream::new(id)?,
            side: Stream::new(id)?,
            ea: Event::new()?,
            eb: Event::new()?,
            probe: WmmaProbe::for_arch(arch)?,
            ceil: DeviceCeilings::for_arch(arch)?,
            sink: zeros(id, 1 << 16)?,
            spin_out: DeviceBuffer::new(id, 32)?,
            spin_a: dev_f32(id, &[1.0])?,
            spin_b: dev_f32(id, &[0.999])?,
            spin_iters: 2000,
            spin_long: 2000,
        };
        for _ in 0..3 {
            r.spin_only(r.spin_iters)?;
        }
        let t = pct(&(0..15).map(|_| r.spin_only(2000)).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        r.spin_iters = ((2000.0 * 50.0 / t.max(1.0)) as u32).clamp(100, 2_000_000);
        r.spin_long = ((2000.0 * 1500.0 / t.max(1.0)) as u32).clamp(100, 20_000_000);
        let t1 = pct(&(0..15).map(|_| r.spin_only(r.spin_iters)).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        let t2 = pct(&(0..5).map(|_| r.spin_only(r.spin_long)).collect::<eyre::Result<Vec<_>>>()?, 0.5);
        eprintln!("spinner: {} iters = {t1:.1} us; long {} iters = {t2:.0} us", r.spin_iters, r.spin_long);
        Ok(r)
    }
    fn spin(&mut self, iters: u32) -> eyre::Result<()> {
        // SAFETY: the stream outlives the call and is only used on this thread.
        let st = unsafe { &*(&self.s as *const Stream) };
        self.probe.launch_fma_f32(st, &mut self.spin_out, &self.spin_a, &self.spin_b, iters, 1, 32)
    }
    fn spin_only(&mut self, iters: u32) -> eyre::Result<f64> {
        let st = unsafe { &*(&self.s as *const Stream) };
        self.ea.record(st)?;
        self.spin(iters)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
    fn flush(&mut self, region: &DeviceBuffer<u8>) -> eyre::Result<()> {
        let st = unsafe { &*(&self.s as *const Stream) };
        let v = region.slice_view(W_SPAN, REGION - W_SPAN);
        self.ceil.read_v4(st, &v, &mut self.sink, 64 * 8, 256)
    }
    /// Device time of `f`: [flush?] [spin] evA f evB. Returns (device us, host enqueue us of f).
    fn time_q(&mut self, region: Option<&DeviceBuffer<u8>>, long: bool, f: &mut dyn FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<(f64, f64)> {
        let st = unsafe { &*(&self.s as *const Stream) };
        if let Some(r) = region {
            self.flush(r)?;
        }
        self.spin(if long { self.spin_long } else { self.spin_iters })?;
        self.ea.record(st)?;
        let t0 = std::time::Instant::now();
        f(st)?;
        let host = t0.elapsed().as_secs_f64() * 1e6;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok((Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3, host))
    }
    /// Host-paced: idle stream, evA f evB (the host enqueues while the device runs).
    fn time_idle(&mut self, f: &mut dyn FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<f64> {
        let st = unsafe { &*(&self.s as *const Stream) };
        st.synchronize()?;
        self.ea.record(st)?;
        f(st)?;
        self.eb.record(st)?;
        st.synchronize()?;
        Ok(Event::elapsed_ms(&self.ea, &self.eb)? as f64 * 1e3)
    }
}


// Sample labels for the rocprofv3 analysis: every queued sample starts with ONE spin kernel
// (`fma_f32_throughput_probe`), so the k-th spin-delimited kernel group is the k-th label.
static LABELS: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());
fn label(s: String) {
    LABELS.lock().unwrap().push(s);
}
/// Host-paced samples have no spin: they ride in the group of the preceding queued sample, so the
/// analysis tags them onto the previous label (never mixed with queued groups).
fn label_idle(s: String) {
    let mut l = LABELS.lock().unwrap();
    if let Some(last) = l.last_mut() {
        last.push_str(" +");
        last.push_str(&s);
    }
}

fn duty_pause() {
    std::thread::sleep(std::time::Duration::from_millis(3));
}

// ---------------------------------------------------------------------------------------------
// The ops of one production lane-layer (file:line of forward_prefill.rs on worktree-dgpu-fusion-prod).
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Op {
    Empty,        // baseline: vec_scale n = 1 (one 1-WG launch)
    VecAdd,       // floor probe: vec_add over 4 x 5120 (80 WGs)
    WaitMoe,      // 12576 compute.wait_event(moe_arrived) [here: a completed side-stream event]
    PostH2d,      // 12773 remote partial: pinned H2D b*5120 f32 (V41_REMOTE_PARTIAL_ASYNC=1)
    PostCombine,  // 12897 hc_post_add2 (moe + shared + remote) -> residual_next
    EgH2d,        // 4642 Engram rows pinned H2D (layers 1, 14)
    EgQuant,      // 5815 q8 quantize_input_batched (6144)
    EgMvQ,        // 5832 q8 matvec_batched [25600 x 6144] at 1/4 rows (x4 = production)
    EgGate,       // 5841 engram_gate
    MhcPreAttn,   // 5932 MhcArena::launch_fast_via (mix PRE_SCALED + collapse + rms_w + carry)
    QQuant,       // 6159 q8 quantize_input_batched(attn_input_norm)
    QA,           // 6176 dispatch -> q8 matvec_batched [1280 x 5120]
    QRmsQuant,    // 6229 rms_w launch_weighted_quant_q8_1280
    QB,           // 6293 q8 matvec_batched [32768 x 1280]
    QRope,        // 6335 rope launch_forward_batched_copy (q_normed := rope(q))
    KvMv,         // 6432 q8 matvec_batched [512 x 5120]
    KvRrf,        // 6465 rope launch_kv_rms_rope_fp8
    KvAppend,     // 6671 kv_append launch_batched_rows
    CompMv,       // 6788 f16 matvec_pair_batched_tiled [512 x 5120] x2 (KV-source layers)
    CompH2d,      // 6829 2x pageable H2D i32[b] (row_per_b, pos_mod_per_b)
    CompSw,       // 6919 compressor_state_write launch_batched_rows
    CompPool,     // 7101 compressor_pool launch_batched_rows (fire)
    CompRms,      // 7133 rms_w (512) over the fired rows
    CompPosH2d,   // 7144 pageable H2D i32[n_bnd]
    IkMv,         // 7160 f16 matvec_batched [128 x 512] (index-K, V41_INDEX_K=1)
    IkRms,        // 7180 rms_w (128)
    IkRope,       // 7192 rope (1 x 128)
    IkAppend,     // 7204 index_kv_e2m1 launch_append_batched_rows
    CompRope,     // 7222 rope (1 x 512)
    CompFp4,      // 7238 fp4kv launch
    CompAppend,   // 7253 comp_kv_append launch_batched_rows
    AttnMeta,     // 7637 attn_meta launch (n_raw_per, n_raw_offset_per, n_comp_per[, n_idx])
    IxQ,          // 7924 f16 matvec_batched_z16 [4096 x 1280] (index-source layers)
    IxRope,       // 7949 rope (32 x 128)
    IxQat,        // 7968 indexer_qat launch_fp4
    IxProj,       // 7977 f16 matvec_batched_z16 (b<=4) / matvec_batched [32 x 5120]
    IxScale,      // 8000 vec_scale
    IxScore,      // 8037 indexer_score_wmma launch_batched_mw_e2m1_rows over n_idx keys
    IxCand,       // 8154 candidate_blocks launch_mask (layers > 20)
    IxTopk,       // 8171 indexer_topk_bitonic launch_batched (threshold select)
    Gather,       // 8201 / 8276 indexer_gather launch_batched_rows (512 rows of 1 KB per row)
    Attn,         // 8348 attn_dec launch_fused_f16s_rows (128 raw + 512 gathered keys)
    ORopeQ,       // 8706 rope launch_inverse_quant_q8 (heads -> Q8_0)
    OA,           // 8766 q8_grouped matvec_grouped_batched [8 x 1024 x 4096]
    OQuant,       // 8806 q8 quantize_input_batched(low, 8192)
    OB,           // 8818 q8 matvec_batched [5120 x 8192]
    HcPost,       // 8860 hc_post launch_from_split_batched (mhc_post_attn)
    MhcPreFfn,    // 8943 MhcArena::launch_fast (collapse + rms_w; mix late)
    RouterMv,     // 9131 f16 matvec_batched_router_x2 [384 x 5120] x2 (l and l+1)
    PriorH2d,     // 9238 pinned H2D 384 f32 (cache prior, V41_SUB=3)
    RouterTopk,   // 9253 router_topk launch_batched_ex (prior, protect 2, orig_sel, range)
    LookTopk,     // 9411 router_topk launch_batched (layer l+1 look-ahead)
    XqQ8k,        // 9438 q8k launch (box-2 / iGPU Q8_K rows)
    RbPack,       // 9502 rb_pack launch (sel, look, look_ew, orig, range, ew, xq) -> pinned
    SelRecord,    // 9507 selected_ready.record
    SelCount,     // 9514 expert_sel_count launch
    ShQuant,      // 3753 q8 quantize_input_batched(ffn_input_norm)
    ShFused,      // 3792 shared_fused launch (gate+up+swiglu+quant, b <= 5)
    ShGate,       // 3811 q8 matvec_batched [2304 x 5120] (b 6..8)
    ShUp,         // 3828
    ShSwiglu,     // 3843
    ShQuantMid,   // 3857
    ShDown,       // 3884 q8 matvec_batched [5120 x 2304]
    MixLate,      // 9549 MhcArena::launch_fast (mix NORMED + carry)
    // --- proxies of the fused design (not production) ---
    PostCombineZc, // hc_post_add2 reading box 2's partial straight from the pinned host buffer (no SDMA copy)
    RouterTopkZc,  // router top-k reading the cache prior straight from pinned host memory (no blit copy)
    EgQuantZc,     // Engram quantize reading the gathered rows straight from pinned host memory
    // --- dGPU fusion group G1 (V41_DGPU_G1_FUSED): MhcPreFfn .. RbPack + SelCount as ONE launch
    G1Fused,
}

use Op::*;

impl Op {
    fn launches(self) -> u32 {
        match self {
            CompH2d => 2,
            _ => 1,
        }
    }
    /// Bytes the op must move at minimum (weights + activations in/out), for its floor.
    fn bytes(self, b: u32, n_idx: u32) -> usize {
        let b = b as usize;
        let f = 4usize;
        let ne = N_EMBD as usize;
        let hcd = HC_DIM as usize;
        let nb = b.div_ceil(2);
        match self {
            Empty | WaitMoe | SelRecord => 0,
            VecAdd => 4 * ne * 3 * f,
            PostH2d => b * ne * f,
            PostCombine | PostCombineZc => b * (3 * ne + 2 * hcd) * f,
            EgH2d => b * ENGRAM_IN as usize * f,
            EgQuant | EgQuantZc => b * ENGRAM_IN as usize * (f + 1),
            G1Fused => 2 * f16_bytes(N_EXPERT, N_EMBD) + b * (hcd + 2 * ne) * f,
            EgMvQ => q8_bytes(ENGRAM_OUT_Q, ENGRAM_IN),
            EgGate => b * (2 * hcd + 5 * ne) * f,
            MhcPreAttn => f16_bytes(HC_MIX_DIM, HC_DIM) + b * (hcd + 2 * ne) * f,
            QQuant => b * ne * (f + 1),
            QA => q8_bytes(N_LORA_Q, N_EMBD),
            QRmsQuant => b * N_LORA_Q as usize * (2 * f + 1),
            QB => q8_bytes(Q_FLAT, N_LORA_Q),
            QRope => b * Q_FLAT as usize * 2 * f,
            KvMv => q8_bytes(N_HEAD_DIM, N_EMBD),
            KvRrf => b * 512 * 2 * f,
            KvAppend => b * 512 * (f + 2),
            CompMv => 2 * f16_bytes(COMP_W, N_EMBD),
            CompH2d => 2 * b * 4,
            CompSw => b * COMP_W as usize * 4 * f,
            CompPool => nb * 2 * COMP_W as usize * 2 * f,
            CompRms | CompRope | CompFp4 => nb * 512 * 2 * f,
            CompPosH2d => nb * 4,
            IkMv => f16_bytes(128, 512),
            IkRms | IkRope => nb * 128 * 2 * f,
            IkAppend => nb * (128 * f + E2M1_KEY_ROW_BYTES),
            CompAppend => nb * 512 * (f + 2),
            AttnMeta => 3 * b * 4,
            IxQ => f16_bytes(N_INDEXER_HEAD * N_INDEXER_HEAD_DIM, N_LORA_Q),
            IxRope | IxQat => b * 4096 * 2 * f,
            IxProj => f16_bytes(N_INDEXER_HEAD, N_EMBD),
            IxScale => b * 32 * 2 * f,
            IxScore => n_idx as usize * E2M1_KEY_ROW_BYTES + b * n_idx as usize * f,
            IxCand => b * n_idx as usize * 2 * f,
            IxTopk => b * n_idx as usize * f,
            Gather => b * 512 * 512 * 2 * 2,
            Attn => b * (640 * 512 * 2 + Q_FLAT as usize * 2 * f),
            ORopeQ => b * Q_FLAT as usize * (2 * f + 1),
            OA => q8_bytes(N_GROUPS * RANK, GROUP_DIM),
            OQuant => b * OUT_LOW as usize * (f + 1),
            OB => q8_bytes(N_EMBD, OUT_LOW),
            HcPost => b * (2 * hcd + ne) * f,
            MhcPreFfn => b * (hcd + 2 * ne) * f,
            RouterMv => 2 * f16_bytes(N_EXPERT, N_EMBD),
            PriorH2d => N_EXPERT as usize * f,
            RouterTopk | RouterTopkZc | LookTopk => b * N_EXPERT as usize * f,
            XqQ8k => b * ne * f + b * 16 * 292,
            RbPack => b * 16 * 292 + b * 6 * 4 * 6,
            SelCount => b * 6 * 4,
            ShQuant => b * ne * (f + 1),
            ShFused => 2 * q8_bytes(N_FF_SHARED, N_EMBD),
            ShGate | ShUp => q8_bytes(N_FF_SHARED, N_EMBD),
            ShSwiglu => b * N_FF_SHARED as usize * 3 * f,
            ShQuantMid => b * N_FF_SHARED as usize * (f + 1),
            ShDown => q8_bytes(N_EMBD, N_FF_SHARED),
            MixLate => f16_bytes(HC_MIX_DIM, HC_DIM) + b * hcd * f,
        }
    }
    fn group(self) -> &'static str {
        match self {
            Empty | VecAdd => "baseline",
            WaitMoe | PostH2d | PostCombine | PostCombineZc => "combine",
            EgH2d | EgQuant | EgQuantZc | EgMvQ | EgGate => "engram",
            MhcPreAttn => "mhc_pre_attn",
            QQuant | QA | QRmsQuant | QB | QRope => "q_chain",
            KvMv | KvRrf => "kv_chain",
            KvAppend | CompMv | CompH2d | CompSw | CompPool | CompRms | CompPosH2d | IkMv | IkRms | IkRope | IkAppend | CompRope
            | CompFp4 | CompAppend => "kv_append_comp",
            AttnMeta | IxQ | IxRope | IxQat | IxProj | IxScale | IxScore | IxCand | IxTopk | Gather | Attn => "attn_compute",
            ORopeQ | OA | OQuant | OB => "output_proj",
            HcPost => "mhc_post_attn",
            MhcPreFfn => "mhc_pre_ffn",
            RouterMv | PriorH2d | RouterTopk | RouterTopkZc | LookTopk | XqQ8k => "router",
            RbPack | SelRecord | SelCount => "rb_pack",
            G1Fused => "g1_router_block",
            ShQuant | ShFused | ShGate | ShUp | ShSwiglu | ShQuantMid | ShDown => "shared_expert",
            MixLate => "mix_ffn_late",
        }
    }
}

/// The chain classes. `reuse` = 30 of the 40 layers (ratio>0, not an index source: s2-reuse gather);
/// `index` = the index sources without a compressor of their own (24, 28, 32, 36; candidate mask);
/// `kvsrc` = KV + index source firing its ratio-2 compressor (2, 8, 14; 20 is ratio 1).
fn chain(class: &str, b: u32) -> Vec<Op> {
    match class {
        // Production reuse chain with the three host->device copies replaced by zero-copy reads.
        "reuse_zc" => {
            return chain("reuse", b)
                .into_iter()
                .filter(|o| !matches!(o, PostH2d | PriorH2d))
                .map(|o| match o {
                    PostCombine => PostCombineZc,
                    RouterTopk => RouterTopkZc,
                    o => o,
                })
                .collect()
        }
        // Production reuse chain without the selected_ready event record.
        // Launch-floor probes (same harness as the chains: paired, interleaved).
        "probe_empty1" => return vec![Empty],
        "probe_empty17" => return vec![Empty; 17],
        "probe_vadd17" => return vec![VecAdd; 17],
        "probe_pinned8" => return [vec![Empty], vec![PriorH2d; 8]].concat(),
        "probe_pageable8" => return [vec![Empty], vec![CompPosH2d; 8]].concat(),
        "probe_event8" => return [vec![Empty], vec![SelRecord; 8]].concat(),
        "probe_remote8" => return [vec![Empty], vec![PostH2d; 8]].concat(),
        "reuse_noev" => return chain("reuse", b).into_iter().filter(|o| *o != SelRecord).collect(),
        // dGPU fusion group G1: the router block as one launch (selected_ready still recorded after it).
        "reuse_g1" => {
            let mut v = Vec::new();
            for o in chain("reuse", b) {
                match o {
                    MhcPreFfn => v.push(G1Fused),
                    RouterMv | PriorH2d | RouterTopk | LookTopk | XqQ8k | RbPack | SelCount => {}
                    o => v.push(o),
                }
            }
            return v;
        }
        // PROXY of the fused design: only the launches that survive fusion (their current kernels;
        // the absorbed epilogue work is not added back, so this is a lower bound on the fused wall).
        "reuse_min" => {
            let mut v = vec![PostCombineZc, MhcPreAttn, QA, QB, Attn, OA, OB, MhcPreFfn, RouterMv, RouterTopkZc];
            v.push(if b <= 5 { ShFused } else { ShGate });
            v.push(ShDown);
            return v;
        }
        _ => {}
    }
    let mut v = vec![WaitMoe, PostH2d, PostCombine];
    if class == "engram" {
        v.extend([EgH2d, EgQuant, EgMvQ, EgGate]);
    }
    v.extend([MhcPreAttn, QQuant, QA, QRmsQuant, QB, QRope, KvMv, KvRrf, KvAppend]);
    if class == "kvsrc" {
        v.extend([CompMv, CompH2d, CompSw, CompPool, CompRms, CompPosH2d, IkMv, IkRms, IkRope, IkAppend, CompRope, CompFp4, CompAppend]);
    }
    v.push(AttnMeta);
    if class == "index" || class == "kvsrc" {
        v.extend([IxQ, IxRope, IxQat, IxProj, IxScale, IxScore]);
        if class == "index" {
            v.push(IxCand);
        }
        v.push(IxTopk);
    }
    v.extend([Gather, Attn, ORopeQ, OA, OQuant, OB, HcPost, MhcPreFfn, RouterMv, PriorH2d, RouterTopk, LookTopk, XqQ8k, RbPack, SelRecord, SelCount, ShQuant]);
    if b <= 5 {
        v.push(ShFused);
    } else {
        v.extend([ShGate, ShUp, ShSwiglu, ShQuantMid]);
    }
    v.extend([ShDown, MixLate]);
    v
}

/// Ops that dispatch a kernel (the rest are copies, event records, waits).
fn is_kernel(op: Op) -> bool {
    // (the *Zc proxies are kernels)
    !matches!(op, WaitMoe | PostH2d | PriorH2d | SelRecord | CompH2d | CompPosH2d | EgH2d)
}

/// Ops of the router critical path: from the MoE's arrival to `selected_ready` (what the host waits
/// for before it can route the next MoE). Everything after it (rb_pack excluded) overlaps the MoE.
fn on_route_path(op: Op) -> bool {
    !matches!(op, SelCount | ShQuant | ShFused | ShGate | ShUp | ShSwiglu | ShQuantMid | ShDown | MixLate)
}

// ---------------------------------------------------------------------------------------------
// Buffers.
// ---------------------------------------------------------------------------------------------
struct Bufs {
    region: DeviceBuffer<u8>,
    rope: RopeParams,
    n_idx: u32,
    moe_ev: Event,
    sel_ev: Event,
    // mHC / residual
    residual: DeviceBuffer<f32>,
    residual_next: DeviceBuffer<f32>,
    after_attn_hc: DeviceBuffer<f32>,
    carry: DeviceBuffer<f32>,
    split: DeviceBuffer<f32>,
    mix: DeviceBuffer<f32>,
    counters: DeviceBuffer<u32>,
    inv_rows: DeviceBuffer<f32>,
    hc_scale: DeviceBuffer<f32>,
    hc_base: DeviceBuffer<f32>,
    attn_cur: DeviceBuffer<f32>,
    attn_in: DeviceBuffer<f32>,
    attn_norm_w: DeviceBuffer<f32>,
    ffn_cur: DeviceBuffer<f32>,
    ffn_in: DeviceBuffer<f32>,
    ffn_norm_w: DeviceBuffer<f32>,
    // combine
    moe_recv: DeviceBuffer<f32>,
    ffn_shared: DeviceBuffer<f32>,
    remote: DeviceBuffer<f32>,
    remote_pin: PinnedBuffer<f32>,
    // engram
    eg_pin: PinnedBuffer<f32>,
    eg_rows: DeviceBuffer<f32>,
    eg_xq: DeviceBuffer<i8>,
    eg_xs: DeviceBuffer<f32>,
    eg_kv: DeviceBuffer<f32>,
    eg_qk: DeviceBuffer<f32>,
    // q / kv
    xq: DeviceBuffer<i8>,
    xs: DeviceBuffer<f32>,
    qr: DeviceBuffer<f32>,
    qr_normed: DeviceBuffer<f32>,
    qr_xq: DeviceBuffer<i8>,
    qr_xs: DeviceBuffer<f32>,
    q_a_norm_w: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    q_normed: DeviceBuffer<f32>,
    pos: DeviceBuffer<i32>,
    kv_raw: DeviceBuffer<f32>,
    kv_normed: DeviceBuffer<f32>,
    kv_norm_w: DeviceBuffer<f32>,
    kv_cache: DeviceBuffer<u16>,
    slot_per: DeviceBuffer<i32>,
    // compressor
    kv_cur: DeviceBuffer<f32>,
    sc_cur: DeviceBuffer<f32>,
    row_per_b: DeviceBuffer<i32>,
    pos_mod_per_b: DeviceBuffer<i32>,
    row_host: Vec<i32>,
    state_kv: DeviceBuffer<f32>,
    state_score: DeviceBuffer<f32>,
    state_base: DeviceBuffer<i32>,
    fire_state_idx: DeviceBuffer<i32>,
    fire_dst_row: DeviceBuffer<i32>,
    comp_pooled: DeviceBuffer<f32>,
    comp_rows: DeviceBuffer<f32>,
    comp_norm_w: DeviceBuffer<f32>,
    comp_pos: DeviceBuffer<i32>,
    ik_rows: DeviceBuffer<f32>,
    ik_normed: DeviceBuffer<f32>,
    ik_norm_w: DeviceBuffer<f32>,
    ik_store: DeviceBuffer<u8>,
    // attention
    n_raw_per: DeviceBuffer<i32>,
    n_raw_off: DeviceBuffer<i32>,
    n_comp_per: DeviceBuffer<i32>,
    n_raw_host: Vec<i32>,
    n_off_host: Vec<i32>,
    n_comp_host: Vec<i32>,
    comp_store: DeviceBuffer<u16>,
    comp_base: DeviceBuffer<i32>,
    sel_saved: DeviceBuffer<i32>,
    active: DeviceBuffer<u16>,
    sinks: DeviceBuffer<f32>,
    heads: DeviceBuffer<f32>,
    // indexer
    iq: DeviceBuffer<f32>,
    ihw: DeviceBuffer<f32>,
    keys: DeviceBuffer<u8>,
    keys_base: DeviceBuffer<u32>,
    n_idx_per: DeviceBuffer<u32>,
    scores: DeviceBuffer<f32>,
    scores_pristine: DeviceBuffer<f32>,
    topk_scratch: DeviceBuffer<u32>,
    topk_done: DeviceBuffer<u32>,
    block_score: DeviceBuffer<f32>,
    threshold: DeviceBuffer<u32>,
    // output proj
    heads_xq: DeviceBuffer<i8>,
    heads_xs: DeviceBuffer<f32>,
    low: DeviceBuffer<f32>,
    low_xq: DeviceBuffer<i8>,
    low_xs: DeviceBuffer<f32>,
    attn_out: DeviceBuffer<f32>,
    // router
    logits: DeviceBuffer<f32>,
    logits2: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    bias2: DeviceBuffer<f32>,
    prior: DeviceBuffer<f32>,
    prior_pin: PinnedBuffer<f32>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    orig_sel: DeviceBuffer<i32>,
    range: DeviceBuffer<f32>,
    look_sel: DeviceBuffer<i32>,
    look_ew: DeviceBuffer<f32>,
    rxq: DeviceBuffer<u8>,
    rb: PinnedBuffer<u32>,
    sel_bank: DeviceBuffer<u32>,
    // shared
    mid_xq: DeviceBuffer<i8>,
    mid_xs: DeviceBuffer<f32>,
    gate_sh: DeviceBuffer<f32>,
    up_sh: DeviceBuffer<f32>,
    mid_sh: DeviceBuffer<f32>,
}

impl Bufs {
    fn new(id: i32, rng: &mut Lcg, n_idx: u32) -> eyre::Result<Self> {
        let b = BMAX;
        let (ne, hcd, m) = (N_EMBD as usize, HC_DIM as usize, HC_MIX_DIM as usize);
        // Weight bytes valid as Q8_0 blocks AND as f16: `& 0xBB` keeps every f16 high byte's exponent
        // field <= 14 (finite, |x| < 1); as int8 they are in [-128, 59]; Q8_0 scales are finite.
        let region = {
            let mut h = vec![0u8; REGION];
            for (i, x) in h.iter_mut().enumerate() {
                *x = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 37) as u8 & 0xBB;
            }
            let mut d = DeviceBuffer::<u8>::new(id, REGION)?;
            d.copy_from_host(&h)?;
            d
        };
        let carry0: Vec<f32> = (0..b * m).map(|i| if i % m < 4 { 0.25 } else { 0.1 }).collect();
        // One stream's rows at consecutive positions (a DSpark verify lane) deep in a long context.
        let pos0 = n_idx as i32 * 2 + 7;
        let pos_h: Vec<i32> = (0..b as i32).map(|i| pos0 + i).collect();
        let slot_h: Vec<i32> = (0..b as i32).map(|i| 300 + i).collect();
        let n_raw_host = vec![128i32; b];
        let n_off_host: Vec<i32> = slot_h.iter().map(|&s| s + 1 - 128).collect();
        let n_comp_host = vec![INDEXER_TOP_K as i32; b];
        let comp_rows_store = 4096usize;
        let sel_h: Vec<i32> = (0..b * INDEXER_TOP_K as usize).map(|_| (rng.next() % comp_rows_store as u64) as i32).collect();
        let keys = {
            let rb = E2M1_KEY_ROW_BYTES;
            let mut hk = vec![0u8; n_idx as usize * rb];
            for r in 0..n_idx as usize {
                let row = &mut hk[r * rb..(r + 1) * rb];
                for x in row[..64].iter_mut() {
                    *x = (rng.next() & 0xff) as u8;
                }
                for x in row[64..rb].iter_mut() {
                    *x = ((rng.next() % 5) as i8 - 3) as u8;
                }
            }
            let mut d = DeviceBuffer::<u8>::new(id, hk.len())?;
            d.copy_from_host(&hk)?;
            d
        };
        let stride = ATTN_MIXED_MAX_KEYS as usize;
        let levels = v4flash_kernels::indexer::topk_merge_levels(n_idx, INDEXER_TOP_K);
        let per_row: usize = levels.iter().map(|&n| n as usize).sum::<usize>().max(1);
        let nb_stride = v4flash_kernels::candidate_blocks::CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS) as usize;
        let mut remote_pin = PinnedBuffer::<f32>::new(b * ne)?;
        for (i, x) in remote_pin.as_mut_slice().iter_mut().enumerate() {
            *x = ((i % 97) as f32 - 48.0) * 0.001;
        }
        let mut eg_pin = PinnedBuffer::<f32>::new(b * ENGRAM_IN as usize)?;
        for (i, x) in eg_pin.as_mut_slice().iter_mut().enumerate() {
            *x = ((i % 89) as f32 - 44.0) * 0.01;
        }
        let mut prior_pin = PinnedBuffer::<f32>::new(N_EXPERT as usize)?;
        for (i, x) in prior_pin.as_mut_slice().iter_mut().enumerate() {
            *x = if i % 3 == 0 { 0.15 } else { 0.0 };
        }
        let rope = RopeParams {
            freq_base: 160000.0,
            freq_scale: 0.0625,
            ext_factor: 1.0,
            attn_factor: 1.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            n_ctx_orig: 65536,
        };
        Ok(Self {
            region,
            rope,
            n_idx,
            moe_ev: Event::new()?,
            sel_ev: Event::new()?,
            residual: dev_rand(id, b * hcd, rng, 1.0)?,
            residual_next: zeros(id, b * hcd)?,
            after_attn_hc: dev_rand(id, b * hcd, rng, 1.0)?,
            carry: dev_f32(id, &carry0)?,
            split: dev_f32(id, &carry0)?,
            mix: zeros(id, b * m)?,
            counters: zeros(id, 16)?,
            inv_rows: zeros(id, 16)?,
            hc_scale: dev_f32(id, &[0.5, 0.5, 0.5])?,
            hc_base: dev_rand(id, m, rng, 0.5)?,
            attn_cur: zeros(id, b * ne)?,
            attn_in: dev_rand(id, b * ne, rng, 1.0)?,
            attn_norm_w: dev_pos(id, ne, rng)?,
            ffn_cur: zeros(id, b * ne)?,
            ffn_in: dev_rand(id, b * ne, rng, 1.0)?,
            ffn_norm_w: dev_pos(id, ne, rng)?,
            moe_recv: dev_rand(id, b * ne, rng, 0.1)?,
            ffn_shared: dev_rand(id, b * ne, rng, 0.1)?,
            remote: dev_rand(id, b * ne, rng, 0.1)?,
            remote_pin,
            eg_pin,
            eg_rows: dev_rand(id, b * ENGRAM_IN as usize, rng, 1.0)?,
            eg_xq: zeros(id, b * ENGRAM_IN as usize)?,
            eg_xs: zeros(id, b * ENGRAM_IN as usize / 32)?,
            eg_kv: dev_rand(id, b * (N_HC as usize + 1) * ne, rng, 0.1)?,
            eg_qk: dev_rand(id, N_HC as usize * ne, rng, 0.1)?,
            xq: zeros(id, b * ne)?,
            xs: zeros(id, b * ne / 32)?,
            qr: dev_rand(id, b * N_LORA_Q as usize, rng, 1.0)?,
            qr_normed: dev_rand(id, b * N_LORA_Q as usize, rng, 1.0)?,
            qr_xq: zeros(id, b * N_LORA_Q as usize)?,
            qr_xs: zeros(id, b * N_LORA_Q as usize / 32)?,
            q_a_norm_w: dev_pos(id, N_LORA_Q as usize, rng)?,
            q: dev_rand(id, b * Q_FLAT as usize, rng, 0.05)?,
            q_normed: dev_rand(id, b * Q_FLAT as usize, rng, 0.05)?,
            pos: dev_i32(id, &pos_h)?,
            kv_raw: dev_rand(id, b * 512, rng, 1.0)?,
            kv_normed: zeros(id, b * 512)?,
            kv_norm_w: dev_pos(id, 512, rng)?,
            kv_cache: {
                let h: Vec<u16> = (0..1024 * 512).map(|_| f16_bits(rng.sym(1.0))).collect();
                let mut d = DeviceBuffer::<u16>::new(id, h.len())?;
                d.copy_from_host(&h)?;
                d
            },
            slot_per: dev_i32(id, &slot_h)?,
            kv_cur: dev_rand(id, b * COMP_W as usize, rng, 1.0)?,
            sc_cur: dev_rand(id, b * COMP_W as usize, rng, 1.0)?,
            row_per_b: dev_i32(id, &(0..b as i32).map(|i| (pos0 + i) % 2).collect::<Vec<_>>())?,
            pos_mod_per_b: dev_i32(id, &(0..b as i32).map(|i| (pos0 + i) % 2).collect::<Vec<_>>())?,
            row_host: (0..b as i32).map(|i| (pos0 + i) % 2).collect(),
            state_kv: dev_rand(id, b * 2 * COMP_W as usize, rng, 1.0)?,
            state_score: dev_rand(id, b * 2 * COMP_W as usize, rng, 1.0)?,
            state_base: zeros(id, b)?,
            fire_state_idx: zeros(id, b)?,
            fire_dst_row: dev_i32(id, &(0..b as i32).map(|i| 100 + i).collect::<Vec<_>>())?,
            comp_pooled: zeros(id, b * 512)?,
            comp_rows: dev_rand(id, b * 512, rng, 1.0)?,
            comp_norm_w: dev_pos(id, 512, rng)?,
            comp_pos: dev_i32(id, &pos_h)?,
            ik_rows: zeros(id, b * 128)?,
            ik_normed: dev_rand(id, b * 128, rng, 1.0)?,
            ik_norm_w: dev_pos(id, 128, rng)?,
            ik_store: zeros(id, 512 * E2M1_KEY_ROW_BYTES)?,
            n_raw_per: dev_i32(id, &n_raw_host)?,
            n_raw_off: dev_i32(id, &n_off_host)?,
            n_comp_per: dev_i32(id, &n_comp_host)?,
            n_raw_host,
            n_off_host,
            n_comp_host,
            comp_store: {
                let h: Vec<u16> = (0..comp_rows_store * 512).map(|_| f16_bits(rng.sym(1.0))).collect();
                let mut d = DeviceBuffer::<u16>::new(id, h.len())?;
                d.copy_from_host(&h)?;
                d
            },
            comp_base: zeros(id, b)?,
            sel_saved: dev_i32(id, &sel_h)?,
            active: zeros(id, b * INDEXER_TOP_K as usize * 512)?,
            sinks: dev_rand(id, N_HEAD as usize, rng, 1.0)?,
            heads: dev_rand(id, b * Q_FLAT as usize, rng, 0.5)?,
            iq: dev_rand(id, b * 4096, rng, 1.0)?,
            ihw: dev_rand(id, b * 32, rng, 1.0)?,
            keys,
            keys_base: zeros(id, b)?,
            n_idx_per: dev_u32(id, &vec![n_idx; b])?,
            scores: zeros(id, b * stride)?,
            scores_pristine: zeros(id, 1)?,
            topk_scratch: zeros(id, b * per_row)?,
            topk_done: zeros(id, b)?,
            block_score: zeros(id, b * nb_stride)?,
            threshold: zeros(id, b)?,
            heads_xq: zeros(id, b * Q_FLAT as usize)?,
            heads_xs: zeros(id, b * Q_FLAT as usize / 32)?,
            low: dev_rand(id, b * OUT_LOW as usize, rng, 1.0)?,
            low_xq: zeros(id, b * OUT_LOW as usize)?,
            low_xs: zeros(id, b * OUT_LOW as usize / 32)?,
            attn_out: zeros(id, b * ne)?,
            logits: dev_rand(id, b * N_EXPERT as usize, rng, 1.0)?,
            logits2: dev_rand(id, b * N_EXPERT as usize, rng, 1.0)?,
            bias: dev_rand(id, N_EXPERT as usize, rng, 0.01)?,
            bias2: dev_rand(id, N_EXPERT as usize, rng, 0.01)?,
            prior: zeros(id, N_EXPERT as usize)?,
            prior_pin,
            sel: zeros(id, b * N_EXPERT_USED)?,
            ew: zeros(id, b * N_EXPERT_USED)?,
            orig_sel: zeros(id, b * N_EXPERT_USED)?,
            range: zeros(id, b)?,
            look_sel: zeros(id, b * N_EXPERT_USED)?,
            look_ew: zeros(id, b * N_EXPERT_USED)?,
            rxq: zeros(id, b * BLOCKS_Q8K_GATE_IN as usize * 292)?,
            rb: PinnedBuffer::<u32>::new(16384)?,
            sel_bank: zeros(id, 64 * N_EXPERT as usize)?,
            mid_xq: zeros(id, b * N_FF_SHARED as usize)?,
            mid_xs: zeros(id, b * N_FF_SHARED as usize / 32)?,
            gate_sh: zeros(id, b * N_FF_SHARED as usize)?,
            up_sh: zeros(id, b * N_FF_SHARED as usize)?,
            mid_sh: dev_rand(id, b * N_FF_SHARED as usize, rng, 1.0)?,
        })
    }
    /// A weight view at the head of the region (every weight aliases [0, W_SPAN)). `slot` staggers
    /// the two weights of one launch so they are distinct bytes.
    fn w(&self, bytes: usize, slot: usize) -> DeviceBuffer<u8> {
        let off = slot * 8 * MB;
        assert!(off + bytes <= W_SPAN, "weight {bytes} B at slot {slot} past W_SPAN");
        self.region.slice_view(off, bytes)
    }
}

/// Enqueue one op on `s` (production launcher + arguments, see the `Op` file:line notes).
fn run(e: &DeviceEngine, s: &Stream, bf: &mut Bufs, op: Op, b: u32) -> eyre::Result<()> {
    let bu = b as usize;
    let ne = N_EMBD as usize;
    match op {
        Empty => e.vec_scale.launch(s, &mut bf.inv_rows, 1.0, 1),
        VecAdd => e.vec_add.launch(s, &mut bf.attn_cur, &bf.ffn_cur, 4 * N_EMBD),
        WaitMoe => s.wait_event(&bf.moe_ev),
        PostH2d => {
            let src = &bf.remote_pin.as_slice()[..bu * ne];
            bf.remote.slice_view_mut(0, bu * ne).copy_from_host_async(src, s)
        }
        PostCombine => e.hc_post.launch_from_split_batched_add2(
            s, &mut bf.residual_next, &bf.moe_recv, &bf.ffn_shared, &bf.remote, &bf.after_attn_hc, &bf.split, N_HC, N_EMBD, N_HC, b,
        ),
        EgH2d => {
            let n = bu * ENGRAM_IN as usize;
            let src = &bf.eg_pin.as_slice()[..n];
            bf.eg_rows.slice_view_mut(0, n).copy_from_host_async(src, s)
        }
        EgQuant => e.q8.quantize_input_batched(s, &mut bf.eg_xq, &mut bf.eg_xs, &bf.eg_rows, ENGRAM_IN, b),
        EgMvQ => {
            let w = bf.w(q8_bytes(ENGRAM_OUT_Q, ENGRAM_IN), 0);
            e.q8.matvec_batched(s, &mut bf.eg_kv, &w, &bf.eg_xq, &bf.eg_xs, ENGRAM_OUT_Q, ENGRAM_IN, b)
        }
        EgGate => {
            let mut h = bf.residual.slice_view_mut(0, bu * HC_DIM as usize);
            e.engram_gate.launch(s, &mut h, &bf.eg_kv, &bf.eg_qk, N_HC, N_EMBD, (N_HC + 1) * N_EMBD, N_HC * N_EMBD, RMS_EPS, b)
        }
        MhcPreAttn => {
            let w = bf.w(f16_bytes(HC_MIX_DIM, HC_DIM), 0);
            e.mhc_arena.launch_fast(
                s,
                Some(FastMix {
                    weight: &w,
                    x: &bf.residual,
                    scale: &bf.hc_scale,
                    base: &bf.hc_base,
                    mode: MIX_PRE_SCALED,
                    split_out: &mut bf.split,
                    mix_out: &mut bf.mix,
                    counters: &mut bf.counters,
                    inv_rows: &mut bf.inv_rows,
                }),
                Some(FastCollapse { x: &bf.residual, cur_out: &mut bf.attn_cur, norm_out: &mut bf.attn_in, norm_w: &bf.attn_norm_w }),
                &mut bf.carry,
                true,
                HC_DIM,
                RMS_EPS,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )
        }
        QQuant => e.q8.quantize_input_batched(s, &mut bf.xq, &mut bf.xs, &bf.attn_in, N_EMBD, b),
        QA => {
            let w = bf.w(q8_bytes(N_LORA_Q, N_EMBD), 0);
            e.q8.matvec_batched(s, &mut bf.qr, &w, &bf.xq, &bf.xs, N_LORA_Q, N_EMBD, b)
        }
        QRmsQuant => e.rms_w.launch_weighted_quant_q8_1280(s, &mut bf.qr_normed, &mut bf.qr_xq, &mut bf.qr_xs, &bf.qr, &bf.q_a_norm_w, RMS_EPS, b),
        QB => {
            let w = bf.w(q8_bytes(Q_FLAT, N_LORA_Q), 0);
            e.q8.matvec_batched(s, &mut bf.q, &w, &bf.qr_xq, &bf.qr_xs, Q_FLAT, N_LORA_Q, b)
        }
        QRope => {
            let pos = bf.pos.slice_view(0, bu);
            e.rope.launch_forward_batched_copy(s, &mut bf.q_normed, &bf.q, &pos, N_HEAD, N_HEAD_DIM, N_ROT, b, &bf.rope)
        }
        KvMv => {
            let w = bf.w(q8_bytes(N_HEAD_DIM, N_EMBD), 0);
            e.q8.matvec_batched(s, &mut bf.kv_raw, &w, &bf.xq, &bf.xs, N_HEAD_DIM, N_EMBD, b)
        }
        KvRrf => {
            let pos = bf.pos.slice_view(0, bu);
            e.rope.launch_kv_rms_rope_fp8(s, &mut bf.kv_normed, &bf.kv_raw, &bf.kv_norm_w, RMS_EPS, &pos, N_ROT, b, &bf.rope)
        }
        KvAppend => e.kv_append.launch_batched_rows(s, &mut bf.kv_cache, &bf.kv_normed, 0, N_HEAD_DIM, b, Some(&bf.slot_per)),
        CompMv => {
            let (w0, w1) = (bf.w(f16_bytes(COMP_W, N_EMBD), 0), bf.w(f16_bytes(COMP_W, N_EMBD), 1));
            e.f16.matvec_pair_batched_tiled(s, &mut bf.kv_cur, &mut bf.sc_cur, &w0, &w1, &bf.attn_in, COMP_W, N_EMBD, b)
        }
        CompH2d => {
            let h = bf.row_host[..bu].to_vec();
            bf.row_per_b.slice_view_mut(0, bu).copy_from_host_async(&h, s)?;
            bf.pos_mod_per_b.slice_view_mut(0, bu).copy_from_host_async(&h, s)
        }
        CompSw => {
            let ape = bf.w(2 * COMP_W as usize * 2, 0);
            let row_v = bf.row_per_b.slice_view(0, bu);
            let pm_v = bf.pos_mod_per_b.slice_view(0, bu);
            e.compressor_state_write.launch_batched_rows(
                s, &mut bf.state_kv, &mut bf.state_score, &bf.kv_cur, &bf.sc_cur, &ape, &row_v, &pm_v, COMP_W, b, Some(&bf.state_base),
            )
        }
        CompPool => {
            let nb = b.div_ceil(2);
            e.compressor_pool.launch_batched_rows(s, &mut bf.comp_pooled, &bf.state_kv, &bf.state_score, N_HEAD_DIM, 2, nb, Some(&bf.fire_state_idx))
        }
        CompRms => e.rms_w.launch_weighted_batched(s, &mut bf.comp_rows, &bf.comp_pooled, &bf.comp_norm_w, N_HEAD_DIM, RMS_EPS, b.div_ceil(2)),
        CompPosH2d => {
            let nb = bu.div_ceil(2);
            let h: Vec<i32> = (0..nb as i32).map(|i| 1000 + 2 * i).collect();
            bf.comp_pos.slice_view_mut(0, nb).copy_from_host_async(&h, s)
        }
        IkMv => {
            let w = bf.w(f16_bytes(N_INDEXER_HEAD_DIM, N_HEAD_DIM), 0);
            e.f16.matvec_batched(s, &mut bf.ik_rows, &w, &bf.comp_rows, N_INDEXER_HEAD_DIM, N_HEAD_DIM, b.div_ceil(2))
        }
        IkRms => e.rms_w.launch_weighted_batched(s, &mut bf.ik_normed, &bf.ik_rows, &bf.ik_norm_w, N_INDEXER_HEAD_DIM, RMS_EPS, b.div_ceil(2)),
        IkRope => {
            let nb = bu.div_ceil(2);
            let pos = bf.comp_pos.slice_view(0, nb);
            e.rope.launch_forward_batched(s, &mut bf.ik_normed, &pos, 1, N_INDEXER_HEAD_DIM, N_ROT, nb as u32, &bf.rope)
        }
        IkAppend => e.index_kv_e2m1.launch_append_batched_rows(s, &mut bf.ik_store, &bf.ik_normed, 0, b.div_ceil(2), Some(&bf.fire_dst_row)),
        CompRope => {
            let nb = bu.div_ceil(2);
            let pos = bf.comp_pos.slice_view(0, nb);
            e.rope.launch_forward_batched(s, &mut bf.comp_rows, &pos, 1, N_HEAD_DIM, N_ROT, nb as u32, &bf.rope)
        }
        CompFp4 => e.fp4kv.launch(s, &mut bf.comp_rows, b.div_ceil(2), N_HEAD_DIM),
        CompAppend => e.comp_kv_append.launch_batched_rows(s, &mut bf.comp_store, &bf.comp_rows, 0, N_HEAD_DIM, b.div_ceil(2), Some(&bf.fire_dst_row)),
        AttnMeta => {
            use v4flash_kernels::attn_meta::MetaDst;
            let (a, c, d) = (bf.n_raw_host[..bu].to_vec(), bf.n_off_host[..bu].to_vec(), bf.n_comp_host[..bu].to_vec());
            e.attn_meta.launch(
                s,
                [
                    Some(MetaDst::new(&mut bf.n_raw_per, &a)?),
                    Some(MetaDst::new(&mut bf.n_raw_off, &c)?),
                    Some(MetaDst::new(&mut bf.n_comp_per, &d)?),
                    None,
                ],
            )
        }
        IxQ => {
            let w = bf.w(f16_bytes(N_INDEXER_HEAD * N_INDEXER_HEAD_DIM, N_LORA_Q), 0);
            e.f16.matvec_batched_z16(s, &mut bf.iq, &w, &bf.qr_normed, N_INDEXER_HEAD * N_INDEXER_HEAD_DIM, N_LORA_Q, b)
        }
        IxRope => {
            let pos = bf.pos.slice_view(0, bu);
            e.rope.launch_forward_batched(s, &mut bf.iq, &pos, N_INDEXER_HEAD, N_INDEXER_HEAD_DIM, N_ROT, b, &bf.rope)
        }
        IxQat => e.indexer_qat.launch_fp4(s, &mut bf.iq, b * N_INDEXER_HEAD),
        IxProj => {
            let w = bf.w(f16_bytes(N_INDEXER_HEAD, N_EMBD), 0);
            if b <= v4flash_kernels::f16::Z16_PROJ_MAX_B {
                e.f16.matvec_batched_z16(s, &mut bf.ihw, &w, &bf.attn_in, N_INDEXER_HEAD, N_EMBD, b)
            } else {
                e.f16.matvec_batched(s, &mut bf.ihw, &w, &bf.attn_in, N_INDEXER_HEAD, N_EMBD, b)
            }
        }
        IxScale => {
            let sc = 1.0f32 / ((N_INDEXER_HEAD_DIM as f32) * (N_INDEXER_HEAD as f32)).sqrt();
            e.vec_scale.launch(s, &mut bf.ihw, sc, b * N_INDEXER_HEAD)
        }
        IxScore => {
            let wm = e.indexer_score_wmma.as_ref().ok_or_else(|| eyre!("no IndexerScoreWmma"))?;
            wm.launch_batched_mw_e2m1_rows(s, &mut bf.scores, &bf.iq, &bf.ihw, &bf.keys, &bf.n_idx_per, bf.n_idx, ATTN_MIXED_MAX_KEYS, b, Some(&bf.keys_base))
        }
        IxCand => {
            let nb = v4flash_kernels::candidate_blocks::CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS);
            e.candidate_blocks.launch_mask(s, &mut bf.scores, &bf.block_score, &bf.threshold, bf.n_idx_per.raw(), ATTN_MIXED_MAX_KEYS, nb, bf.n_idx, b)
        }
        IxTopk => {
            let n_words = ATTN_MIXED_MAX_KEYS.div_ceil(32);
            // The selection lands in a scratch copy, never in `sel_saved` (the gather's in-range rows).
            let mut out = unsafe { bf.active.view_as::<i32>(0, bu * INDEXER_TOP_K as usize) };
            e.indexer_topk_bitonic.launch_batched(
                s, &mut out, None, &mut bf.topk_scratch, &bf.scores, &bf.n_idx_per, bf.n_idx, ATTN_MIXED_MAX_KEYS, n_words, INDEXER_TOP_K, b,
                Some(&mut bf.topk_done),
            )
        }
        Gather => e.indexer_gather.launch_batched_rows(s, &mut bf.active, &bf.comp_store, &bf.sel_saved, INDEXER_TOP_K, N_HEAD_DIM, b, Some(&bf.comp_base)),
        Attn => {
            let (nrp, nro, ncp) = (bf.n_raw_per.slice_view(0, bu), bf.n_raw_off.slice_view(0, bu), bf.n_comp_per.slice_view(0, bu));
            e.attn_dec.launch_fused_f16s_rows(
                s, &mut bf.heads, &bf.sinks, &bf.q_normed, &bf.kv_cache, Some(&bf.active), &nrp, &nro, &ncp, N_HEAD, N_HEAD_DIM, 128 + INDEXER_TOP_K, b,
                INDEXER_TOP_K, None,
            )
        }
        ORopeQ => {
            let pos = bf.pos.slice_view(0, bu);
            e.rope.launch_inverse_quant_q8(s, &mut bf.heads, &mut bf.heads_xq, &mut bf.heads_xs, &pos, N_HEAD, N_HEAD_DIM, N_ROT, b, &bf.rope)
        }
        OA => {
            let w = bf.w(q8_bytes(N_GROUPS * RANK, GROUP_DIM), 0);
            e.q8_grouped.matvec_grouped_batched(s, &mut bf.low, &w, &bf.heads_xq, &bf.heads_xs, GROUP_DIM, RANK, N_GROUPS, b)
        }
        OQuant => e.q8.quantize_input_batched(s, &mut bf.low_xq, &mut bf.low_xs, &bf.low, OUT_LOW, b),
        OB => {
            let w = bf.w(q8_bytes(N_EMBD, OUT_LOW), 0);
            e.q8.matvec_batched(s, &mut bf.attn_out, &w, &bf.low_xq, &bf.low_xs, N_EMBD, OUT_LOW, b)
        }
        HcPost => e.hc_post.launch_from_split_batched(s, &mut bf.after_attn_hc, &bf.attn_out, &bf.residual, &bf.split, N_HC, N_EMBD, N_HC, b),
        MhcPreFfn => e.mhc_arena.launch_fast(
            s,
            None,
            Some(FastCollapse { x: &bf.after_attn_hc, cur_out: &mut bf.ffn_cur, norm_out: &mut bf.ffn_in, norm_w: &bf.ffn_norm_w }),
            &mut bf.carry,
            false,
            HC_DIM,
            RMS_EPS,
            SINKHORN_ITERS,
            SINKHORN_EPS,
            b,
        ),
        RouterMv => {
            let (w0, w1) = (bf.w(f16_bytes(N_EXPERT, N_EMBD), 0), bf.w(f16_bytes(N_EXPERT, N_EMBD), 1));
            let fused = e.f16.matvec_batched_router_x2(s, &mut bf.logits, &mut bf.logits2, &w0, &w1, &bf.ffn_in, N_EXPERT, N_EMBD, b)?;
            if !fused {
                return Err(eyre!("router_x2 declined b = {b}"));
            }
            Ok(())
        }
        PriorH2d => {
            let src = &bf.prior_pin.as_slice()[..N_EXPERT as usize];
            bf.prior.copy_from_host_async(src, s)
        }
        RouterTopk => e.router_topk.launch_batched_ex(
            s,
            &mut bf.sel,
            &mut bf.ew,
            &bf.logits,
            Some(&bf.bias),
            N_EXPERT,
            N_EXPERT_USED as u32,
            EXPERT_WEIGHT_SCALE,
            ROUTER_WEIGHT_EPS,
            b,
            RouterEx {
                alts: None,
                n_alt: 0,
                alt_w: None,
                prior: Some(&bf.prior),
                n_protect: 2,
                prior_dry: false,
                orig_sel: Some(&mut bf.orig_sel),
                range_out: Some(&mut bf.range),
            },
        ),
        LookTopk => e.router_topk.launch_batched(
            s, &mut bf.look_sel, &mut bf.look_ew, &bf.logits2, Some(&bf.bias2), N_EXPERT, N_EXPERT_USED as u32, EXPERT_WEIGHT_SCALE, ROUTER_WEIGHT_EPS, b,
        ),
        XqQ8k => e.q8k.launch(s, &mut bf.rxq, &bf.ffn_in, BLOCKS_Q8K_GATE_IN * b),
        RbPack => {
            let n_sel = bu * N_EXPERT_USED;
            let mut plan = PackPlan::default();
            plan.push(PackSeg::words(&bf.sel, n_sel)?);
            plan.push(PackSeg::words(&bf.look_sel, n_sel)?);
            plan.push(PackSeg::words(&bf.look_ew, n_sel)?);
            plan.push(PackSeg::words(&bf.orig_sel, n_sel)?);
            plan.push(PackSeg::words(&bf.range, bu)?);
            plan.push(PackSeg::words(&bf.ew, n_sel)?);
            plan.push(PackSeg::bytes(&bf.rxq, bu * BLOCKS_Q8K_GATE_IN as usize * 292)?);
            e.rb_pack.launch(s, &mut bf.rb, plan.segs())
        }
        SelRecord => bf.sel_ev.record(s),
        SelCount => e.expert_sel_count.launch(s, &mut bf.sel_bank, &bf.sel, 7, N_EXPERT, b * N_EXPERT_USED as u32),
        G1Fused => {
            use v4flash_kernels::g1_router_block::{G1Collapse, G1Io, G1Look, G1PackOffsets, G1Prior};
            let (w0, w1) = (bf.w(f16_bytes(N_EXPERT, N_EMBD), 0), bf.w(f16_bytes(N_EXPERT, N_EMBD), 1));
            let ns = (bu * N_EXPERT_USED) as u32;
            let offs = G1PackOffsets { sel: 0, look: ns, look_ew: 2 * ns, orig: 3 * ns, range: 4 * ns, ew: 4 * ns + bu as u32, xq: 5 * ns + bu as u32 };
            let io = G1Io {
                collapse: G1Collapse { x: &bf.after_attn_hc, carry: &bf.carry, norm_w: &bf.ffn_norm_w, cur_out: &mut bf.ffn_cur, norm_out: &mut bf.ffn_in, rms_eps: RMS_EPS },
                w0: &w0,
                bias0: Some(&bf.bias),
                logits0: &mut bf.logits,
                look: Some(G1Look { w: &w1, bias: Some(&bf.bias2), logits: &mut bf.logits2, sel: &mut bf.look_sel, ew: &mut bf.look_ew }),
                prior: Some(G1Prior { host: &bf.prior_pin, host_off: 0, dev: &mut bf.prior, n_protect: 2, dry: false }),
                sel: &mut bf.sel,
                ew: &mut bf.ew,
                orig_sel: Some(&mut bf.orig_sel),
                range: Some(&mut bf.range),
                xq: Some(&mut bf.rxq),
                pack: Some((&mut bf.rb, offs)),
                sel_bank: Some((&mut bf.sel_bank, 7 * N_EXPERT)),
                done: None,
                n_used: N_EXPERT_USED as u32,
                scale: EXPERT_WEIGHT_SCALE,
                weight_eps: ROUTER_WEIGHT_EPS,
                b,
            };
            e.g1.launch(s, io)
        }
        ShQuant => e.q8.quantize_input_batched(s, &mut bf.xq, &mut bf.xs, &bf.ffn_in, N_EMBD, b),
        ShFused => {
            let (w0, w1) = (bf.w(q8_bytes(N_FF_SHARED, N_EMBD), 0), bf.w(q8_bytes(N_FF_SHARED, N_EMBD), 2));
            e.shared_fused.launch(s, &mut bf.mid_xq, &mut bf.mid_xs, &w0, &w1, &bf.xq, &bf.xs, N_EMBD, N_FF_SHARED, b, SWIGLU_CLAMP_EXP)
        }
        ShGate => {
            let w = bf.w(q8_bytes(N_FF_SHARED, N_EMBD), 0);
            e.q8.matvec_batched(s, &mut bf.gate_sh, &w, &bf.xq, &bf.xs, N_FF_SHARED, N_EMBD, b)
        }
        ShUp => {
            let w = bf.w(q8_bytes(N_FF_SHARED, N_EMBD), 2);
            e.q8.matvec_batched(s, &mut bf.up_sh, &w, &bf.xq, &bf.xs, N_FF_SHARED, N_EMBD, b)
        }
        ShSwiglu => e.swiglu.launch_clamped(s, &mut bf.mid_sh, &bf.gate_sh, &bf.up_sh, b * N_FF_SHARED, SWIGLU_CLAMP_EXP),
        ShQuantMid => e.q8.quantize_input_batched(s, &mut bf.mid_xq, &mut bf.mid_xs, &bf.mid_sh, N_FF_SHARED, b),
        ShDown => {
            let w = bf.w(q8_bytes(N_EMBD, N_FF_SHARED), 0);
            e.q8.matvec_batched(s, &mut bf.ffn_shared, &w, &bf.mid_xq, &bf.mid_xs, N_EMBD, N_FF_SHARED, b)
        }
        PostCombineZc => {
            // SAFETY: the pinned buffer is hipHostMalloc'd (mapped, UVA): its pointer is valid on the device.
            let remote = unsafe { DeviceBuffer::<f32>::from_raw_parts(bf.remote_pin.device_ptr(), bu * ne, s.device_id()) };
            e.hc_post.launch_from_split_batched_add2(
                s, &mut bf.residual_next, &bf.moe_recv, &bf.ffn_shared, &remote, &bf.after_attn_hc, &bf.split, N_HC, N_EMBD, N_HC, b,
            )
        }
        EgQuantZc => {
            let rows = unsafe { DeviceBuffer::<f32>::from_raw_parts(bf.eg_pin.device_ptr(), bu * ENGRAM_IN as usize, s.device_id()) };
            e.q8.quantize_input_batched(s, &mut bf.eg_xq, &mut bf.eg_xs, &rows, ENGRAM_IN, b)
        }
        RouterTopkZc => {
            let prior = unsafe { DeviceBuffer::<f32>::from_raw_parts(bf.prior_pin.device_ptr(), N_EXPERT as usize, s.device_id()) };
            e.router_topk.launch_batched_ex(
                s,
                &mut bf.sel,
                &mut bf.ew,
                &bf.logits,
                Some(&bf.bias),
                N_EXPERT,
                N_EXPERT_USED as u32,
                EXPERT_WEIGHT_SCALE,
                ROUTER_WEIGHT_EPS,
                b,
                RouterEx {
                    alts: None,
                    n_alt: 0,
                    alt_w: None,
                    prior: Some(&prior),
                    n_protect: 2,
                    prior_dry: false,
                    orig_sel: Some(&mut bf.orig_sel),
                    range_out: Some(&mut bf.range),
                },
            )
        }
        MixLate => {
            let w = bf.w(f16_bytes(HC_MIX_DIM, HC_DIM), 1);
            e.mhc_arena.launch_fast(
                s,
                Some(FastMix {
                    weight: &w,
                    x: &bf.after_attn_hc,
                    scale: &bf.hc_scale,
                    base: &bf.hc_base,
                    mode: MIX_NORMED,
                    split_out: &mut bf.split,
                    mix_out: &mut bf.mix,
                    counters: &mut bf.counters,
                    inv_rows: &mut bf.inv_rows,
                }),
                None,
                &mut bf.carry,
                true,
                HC_DIM,
                RMS_EPS,
                SINKHORN_ITERS,
                SINKHORN_EPS,
                b,
            )
        }
    }
}

/// Production-like indexer scores (the select's running time depends on them): run the real
/// score once and keep it; the candidate-block build needs them too.
fn prime(e: &DeviceEngine, s: &Stream, bf: &mut Bufs) -> eyre::Result<()> {
    for op in [AttnMeta, IxQ, IxRope, IxQat, IxProj, IxScale, IxScore] {
        run(e, s, bf, op, BMAX as u32)?;
    }
    let nb = v4flash_kernels::candidate_blocks::CandidateBlocks::n_blocks(ATTN_MIXED_MAX_KEYS);
    e.candidate_blocks.launch_build(s, &bf.scores, &mut bf.block_score, &mut bf.threshold, bf.n_idx_per.raw(), ATTN_MIXED_MAX_KEYS, nb, bf.n_idx, BMAX as u32)?;
    s.synchronize()
}

fn dgpu() -> eyre::Result<(Device, String)> {
    let dev = Device::all()?
        .into_iter()
        .find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with("gfx1201")).unwrap_or(false))
        .ok_or_else(|| eyre!("no gfx1201 visible (set HIP_VISIBLE_DEVICES to the dGPU)"))?;
    let arch = dev.properties()?.gcn_arch_name;
    Ok((dev, arch))
}

fn env_list(name: &str, def: &[u32]) -> Vec<u32> {
    std::env::var(name).ok().map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect()).unwrap_or_else(|| def.to_vec())
}

fn shuffle<T>(v: &mut [T], rng: &mut Lcg) {
    for i in (1..v.len()).rev() {
        let j = (rng.next() as usize) % (i + 1);
        v.swap(i, j);
    }
}

// ---------------------------------------------------------------------------------------------
// floors
// ---------------------------------------------------------------------------------------------
fn section_floors(e: &DeviceEngine, rig: &mut Rig, bf: &mut Bufs, rounds: usize, rng: &mut Lcg) -> eyre::Result<()> {
    let id = e.device.id;
    eprintln!("\n== floors ==");
    // (1) streaming read bandwidth: cold (MALL flushed) and warm, several sizes / grids.
    for &mb in &[4usize, 16, 45, 54] {
        let mut best = (0.0f64, 0.0f64, 0u32);
        for &g in &[64u32 * 4, 64 * 8, 64 * 16, 64 * 32] {
            let (off, len) = if mb == 54 { (W_SPAN, 54 * MB) } else { (0, mb * MB) };
            let v = bf.region.slice_view(off, len);
            let mut cold = Vec::new();
            let mut warm = Vec::new();
            for _ in 0..rounds.min(60) {
                let sink = unsafe { &mut *(&mut rig.sink as *mut DeviceBuffer<u32>) };
                let ceil = unsafe { &*(&rig.ceil as *const DeviceCeilings) };
                let flush_region = if mb == 54 { None } else { Some(unsafe { &*(&bf.region as *const DeviceBuffer<u8>) }) };
                cold.push(rig.time_q(flush_region, false, &mut |s| ceil.read_v4(s, &v, sink, g, 256))?.0);
                warm.push(rig.time_q(None, false, &mut |s| ceil.read_v4(s, &v, sink, g, 256))?.0);
            }
            let (c, w) = (pct(&cold, 0.5), pct(&warm, 0.5));
            let gb = |us: f64| (len as f64) / 1e9 / (us / 1e6);
            if gb(c) > best.0 {
                best = (gb(c), gb(w), g);
            }
            eprintln!("  read_v4 {mb:>3} MB grid {g:>5}x256: cold p50 {c:8.1} us = {:6.1} GB/s | warm p50 {w:8.1} us = {:6.1} GB/s", gb(c), gb(w));
            duty_pause();
        }
        eprintln!("  -> {mb} MB best cold {:.1} GB/s (warm {:.1}) at grid {}", best.0, best.1, best.2);
    }
    // (2) launch floors: N dependent direct launches on one stream (queued: device time only).
    let mut dummy: DeviceBuffer<f32> = zeros(id, 64)?;
    let x = dev_rand(id, BMAX * N_EMBD as usize, rng, 1.0)?;
    let mut y: DeviceBuffer<f32> = zeros(id, BMAX * N_EMBD as usize)?;
    let ns = [1u32, 2, 4, 8, 16, 32];
    let mut rows: Vec<(String, Vec<(u32, f64, f64)>)> = Vec::new();
    let kinds = ["empty (vec_scale n=1, 1 WG)", "vec_add 4x5120 (80 WGs)", "event record", "pinned H2D 16 B", "pageable H2D 16 B", "wait_event (done)"];
    for (ki, kind) in kinds.iter().enumerate() {
        let mut pts = Vec::new();
        for &n in &ns {
            let mut dev = Vec::new();
            let mut host = Vec::new();
            for _ in 0..rounds.min(150) {
                let ev = unsafe { &*(&bf.moe_ev as *const Event) };
                let rec = unsafe { &*(&bf.sel_ev as *const Event) };
                let pin = unsafe { &*(&bf.prior_pin as *const PinnedBuffer<f32>) };
                let mut f = |s: &Stream| -> eyre::Result<()> {
                    for _ in 0..n {
                        match ki {
                            0 => e.vec_scale.launch(s, &mut dummy, 1.0, 1)?,
                            1 => e.vec_add.launch(s, &mut y, &x, 4 * N_EMBD)?,
                            2 => rec.record(s)?,
                            3 => dummy.slice_view_mut(0, 4).copy_from_host_async(&pin.as_slice()[..4], s)?,
                            4 => dummy.slice_view_mut(0, 4).copy_from_host_async(&[1.0f32, 2.0, 3.0, 4.0], s)?,
                            _ => s.wait_event(ev)?,
                        }
                    }
                    Ok(())
                };
                let (d, h) = rig.time_q(None, true, &mut f)?;
                dev.push(d);
                host.push(h);
            }
            pts.push((n, pct(&dev, 0.5), pct(&host, 0.5)));
        }
        duty_pause();
        rows.push((kind.to_string(), pts));
    }
    for (k, pts) in &rows {
        // Least-squares slope over N.
        let nf: Vec<f64> = pts.iter().map(|p| p.0 as f64).collect();
        let fit = |ys: &[f64]| {
            let n = nf.len() as f64;
            let (sx, sy) = (nf.iter().sum::<f64>(), ys.iter().sum::<f64>());
            let sxx: f64 = nf.iter().map(|x| x * x).sum();
            let sxy: f64 = nf.iter().zip(ys).map(|(x, y)| x * y).sum();
            let slope = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            (slope, (sy - slope * sx) / n)
        };
        let (ds, di) = fit(&pts.iter().map(|p| p.1).collect::<Vec<_>>());
        let (hs, _) = fit(&pts.iter().map(|p| p.2).collect::<Vec<_>>());
        let detail: Vec<String> = pts.iter().map(|p| format!("N{}={:.1}/{:.1}", p.0, p.1, p.2)).collect();
        eprintln!("  {k:<30} device {ds:6.2} us/op (+{di:5.1} fixed) | host enqueue {hs:5.2} us/op   [{}] (dev/host us)", detail.join(" "));
    }
    // (3) How fast does the HOST learn that a chain finished? The ready-first driver polls
    // `selected_ready.query()`; the alternative is a flag the last kernel writes to pinned memory.
    // Both after the same ~50 us spin; host clock from enqueue to observed completion.
    {
        let mut flag_dev: DeviceBuffer<u32> = zeros(id, 64)?;
        flag_dev.copy_from_host(&vec![7u32; 64])?;
        let mut t_ev = Vec::new();
        let mut t_flag = Vec::new();
        let st = unsafe { &*(&rig.s as *const Stream) };
        for r in 0..rounds.min(200) {
            for variant in [r % 2, 1 - r % 2] {
                st.synchronize()?;
                bf.rb.as_mut_slice()[0] = 0;
                std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
                let t0 = std::time::Instant::now();
                rig.spin(rig.spin_iters)?;
                if variant == 0 {
                    bf.sel_ev.record(st)?;
                    while !bf.sel_ev.query()? {
                        std::hint::spin_loop();
                    }
                    t_ev.push(t0.elapsed().as_secs_f64() * 1e6);
                } else {
                    let mut plan = PackPlan::default();
                    plan.push(PackSeg::words(&flag_dev, 1)?);
                    e.rb_pack.launch(st, &mut bf.rb, plan.segs())?;
                    let p = bf.rb.as_slice().as_ptr();
                    while unsafe { std::ptr::read_volatile(p) } == 0 {
                        std::hint::spin_loop();
                    }
                    t_flag.push(t0.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        st.synchronize()?;
        eprintln!(
            "  completion seen by the host after a ~{:.0} us spin: event.query() poll p50 {:.1} p90 {:.1} us | pinned flag written by a kernel p50 {:.1} p90 {:.1} us",
            pct(&(0..5).map(|_| rig.spin_only(rig.spin_iters)).collect::<eyre::Result<Vec<_>>>()?, 0.5),
            pct(&t_ev, 0.5), pct(&t_ev, 0.9), pct(&t_flag, 0.5), pct(&t_flag, 0.9)
        );
    }
    // (4) Step prologue: the per-row BLOCKING pageable upload of input_hcs (80 KB per row).
    {
        let hc: Vec<f32> = (0..HC_DIM as usize).map(|i| (i % 13) as f32).collect();
        let mut t = Vec::new();
        for _ in 0..rounds.min(100) {
            let t0 = std::time::Instant::now();
            for k in 0..BMAX {
                bf.residual_next.slice_view_mut(k * HC_DIM as usize, HC_DIM as usize).copy_from_host(&hc)?;
            }
            t.push(t0.elapsed().as_secs_f64() * 1e6);
            duty_pause();
        }
        eprintln!("  step prologue: 8 blocking pageable 80 KB input_hcs uploads: host p50 {:.1} p90 {:.1} us", pct(&t, 0.5), pct(&t, 0.9));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// ops
// ---------------------------------------------------------------------------------------------
type Stats = std::collections::HashMap<(u32, Op), (f64, f64, f64, f64, f64)>; // cold p50, cold p90, warm p50, warm p90, host

fn all_ops(b: u32) -> Vec<Op> {
    let mut v: Vec<Op> = vec![Empty];
    for c in ["engram", "kvsrc", "index", "reuse", "reuse_zc", "reuse_min"] {
        for op in chain(c, b) {
            if !v.contains(&op) {
                v.push(op);
            }
        }
    }
    v
}

fn section_ops(e: &DeviceEngine, rig: &mut Rig, bf: &mut Bufs, bs: &[u32], rounds: usize, rng: &mut Lcg, stats: &mut Stats) -> eyre::Result<()> {
    for &b in bs {
        let ops = all_ops(b);
        let mut cold: Vec<Vec<f64>> = vec![Vec::new(); ops.len()];
        let mut warm: Vec<Vec<f64>> = vec![Vec::new(); ops.len()];
        let mut host: Vec<Vec<f64>> = vec![Vec::new(); ops.len()];
        for (i, &op) in ops.iter().enumerate() {
            for _ in 0..3 {
                let s2 = unsafe { &*(&rig.s as *const Stream) };
                run(e, s2, bf, op, b)?;
                let _ = i;
            }
        }
        rig.s.synchronize()?;
        let mut order: Vec<usize> = (0..2 * ops.len()).collect();
        for r in 0..rounds {
            shuffle(&mut order, rng);
            for &o in &order {
                let (i, is_cold) = (o % ops.len(), o >= ops.len());
                let op = ops[i];
                let region = unsafe { &*(&bf.region as *const DeviceBuffer<u8>) };
                label(format!("op b{b} {op:?} {}", if is_cold { "cold" } else { "warm" }));
                let (d, h) = rig.time_q(if is_cold { Some(region) } else { None }, false, &mut |s| run(e, s, bf, op, b))?;
                if is_cold {
                    cold[i].push(d);
                } else {
                    warm[i].push(d);
                    host[i].push(h);
                }
            }
            if r % 4 == 3 {
                duty_pause();
            }
        }
        eprintln!("\n== ops, b = {b} (queued behind a spin: device us incl. the op's own launch; event pair included) ==");
        eprintln!("  {:<12} {:<16} {:>9} {:>9} {:>9} {:>9} {:>8} {:>10} {:>9}", "op", "stage", "cold p50", "cold p90", "warm p50", "warm p90", "host", "bytes", "floor@BW");
        let bw = bw_gbs();
        for (i, &op) in ops.iter().enumerate() {
            let bytes = op.bytes(b, bf.n_idx);
            let st = (pct(&cold[i], 0.5), pct(&cold[i], 0.9), pct(&warm[i], 0.5), pct(&warm[i], 0.9), pct(&host[i], 0.5));
            stats.insert((b, op), st);
            eprintln!(
                "  {:<12} {:<16} {:>9.1} {:>9.1} {:>9.1} {:>9.1} {:>8.2} {:>10} {:>9.1}",
                format!("{op:?}"), op.group(), st.0, st.1, st.2, st.3, st.4, bytes, bytes as f64 / (bw * 1e3)
            );
        }
    }
    Ok(())
}

fn bw_gbs() -> f64 {
    std::env::var("BENCH_BW_GBS").ok().and_then(|v| v.parse().ok()).unwrap_or(600.0)
}

// ---------------------------------------------------------------------------------------------
// chain
// ---------------------------------------------------------------------------------------------
fn section_chain(e: &DeviceEngine, rig: &mut Rig, bf: &mut Bufs, bs: &[u32], rounds: usize, rng: &mut Lcg, stats: &Stats) -> eyre::Result<()> {
    let classes_all = ["reuse", "reuse#AA", "index", "kvsrc", "route:reuse", "reuse_zc", "reuse_noev", "reuse_min", "route:reuse_zc",
        "reuse_g1", "route:reuse_g1",
        "probe_empty1", "probe_empty17", "probe_vadd17", "probe_pinned8", "probe_pageable8", "probe_event8", "probe_remote8"];
    let only = std::env::var("BENCH_CLASSES").ok();
    let classes: Vec<&str> = classes_all.iter().copied().filter(|c| *c == "reuse" || only.as_deref().map_or(true, |o| o.split(',').any(|x| x == *c))).collect();
    for &b in bs {
        let chains: Vec<Vec<Op>> = classes
            .iter()
            .map(|c| {
                let base = c.trim_end_matches("#AA").trim_start_matches("route:");
                let v = chain(base, b);
                if c.starts_with("route:") { v.into_iter().filter(|&o| on_route_path(o)).collect() } else { v }
            })
            .collect();
        for v in &chains {
            for &op in v {
                let s2 = unsafe { &*(&rig.s as *const Stream) };
                run(e, s2, bf, op, b)?;
            }
        }
        rig.s.synchronize()?;
        let n = classes.len();
        let mut q: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut idle: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut host: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut aa: Vec<f64> = Vec::new();
        let mut pairs: Vec<Vec<f64>> = Vec::new();
        let mut order: Vec<usize> = (0..2 * n).collect();
        for r in 0..rounds {
            shuffle(&mut order, rng);
            let mut q_this = vec![f64::NAN; n];
            for &o in &order {
                let (i, is_idle) = (o % n, o >= n);
                let ops = chains[i].clone();
                let mut f = |s: &Stream| -> eyre::Result<()> {
                    for &op in &ops {
                        run(e, s, bf, op, b)?;
                    }
                    Ok(())
                };
                if is_idle {
                    idle[i].push(rig.time_idle(&mut f)?);
                    label_idle(format!("chain b{b} {} idle", classes[i]));
                } else {
                    label(format!("chain b{b} {} queued k={}", classes[i], chains[i].iter().filter(|o| is_kernel(**o)).count()));
                    let (d, h) = rig.time_q(None, true, &mut f)?;
                    q[i].push(d);
                    host[i].push(h);
                    q_this[i] = d;
                }
            }
            if let Some(j) = classes.iter().position(|c| *c == "reuse#AA") {
                aa.push(q_this[0] - q_this[j]);
            }
            pairs.push(q_this.clone());
            if r % 2 == 1 {
                duty_pause();
            }
        }
        eprintln!("\n== chain replay, b = {b} (n_idx = {}) ==", bf.n_idx);
        eprintln!("  A/A (reuse vs identical reuse#AA, queued): paired diff p50 {:+.1} us, |diff| p50 {:.1} p90 {:.1} us", pct(&aa, 0.5), pct(&aa.iter().map(|x| x.abs()).collect::<Vec<_>>(), 0.5), pct(&aa.iter().map(|x| x.abs()).collect::<Vec<_>>(), 0.9));
        // Paired differences vs the production reuse chain (same round, queued).
        for (j, c) in classes.iter().enumerate().skip(1) {
            let d: Vec<f64> = pairs.iter().filter_map(|r| if r[0].is_finite() && r[j].is_finite() { Some(r[0] - r[j]) } else { None }).collect();
            eprintln!("    paired reuse - {c:<16} p50 {:+7.1} us (p25 {:+7.1}, p75 {:+7.1})", pct(&d, 0.5), pct(&d, 0.25), pct(&d, 0.75));
        }
        for (i, c) in classes.iter().enumerate() {
            let ops = &chains[i];
            let launches: u32 = ops.iter().map(|o| o.launches()).sum();
            let kernels = ops.iter().filter(|o| is_kernel(**o)).count();
            let _ = kernels;
            let sum_warm: f64 = ops.iter().map(|o| stats.get(&(b, *o)).map(|s| s.2).unwrap_or(f64::NAN)).sum();
            let sum_cold: f64 = ops.iter().map(|o| stats.get(&(b, *o)).map(|s| s.0).unwrap_or(f64::NAN)).sum();
            let wall = pct(&q[i], 0.5);
            eprintln!(
                "  {:<14} ops {:>2} (kernels {:>2}, launches+copies {:>2}) | queued wall p10 {:7.1} p50 {:7.1} p90 {:7.1} | idle wall p10 {:7.1} p50 {:7.1} | host enqueue p50 {:6.1} us ({:.2}/op)",
                c, ops.len(), kernels, launches, pct(&q[i], 0.1), wall, pct(&q[i], 0.9), pct(&idle[i], 0.1), pct(&idle[i], 0.5), pct(&host[i], 0.5), pct(&host[i], 0.5) / ops.len() as f64
            );
            eprintln!(
                "  {:<12}   sum of isolated warm p50 {:7.1} (each incl. one event-pair) | sum cold p50 {:7.1} | wall - sum_warm {:+7.1}",
                "", sum_warm, sum_cold, wall - sum_warm
            );
        }
        eprintln!("  per-group sums (cold p50, isolated) for class reuse:");
        let mut groups: Vec<(&str, f64, f64, usize)> = Vec::new();
        for &op in &chains[0] {
            let st = stats.get(&(b, op)).copied().unwrap_or((f64::NAN, 0.0, f64::NAN, 0.0, 0.0));
            match groups.iter_mut().find(|g| g.0 == op.group()) {
                Some(g) => {
                    g.1 += st.0;
                    g.2 += st.2;
                    g.3 += 1;
                }
                None => groups.push((op.group(), st.0, st.2, 1)),
            }
        }
        for g in groups {
            eprintln!("    {:<16} ops {:>2}  cold {:7.1}  warm {:7.1}", g.0, g.3, g.1, g.2);
        }
    }
    Ok(())
}

#[test]
#[ignore]
fn bench_dgpu_chain_map() -> eyre::Result<()> {
    install_panic_handler()?;
    let rounds: usize = std::env::var("BENCH_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(150);
    let section = std::env::var("BENCH_SECTION").unwrap_or_else(|_| "all".into());
    let bs = env_list("BENCH_B", &[2, 4, 8]);
    let n_idx = env_list("BENCH_POS", &[32768])[0].min(65536);
    let (dev, arch) = dgpu()?;
    dev.set_current()?;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let mut rig = Rig::new(e.device.id, &arch)?;
    let mut rng = Lcg(0x5eed_c4a1);
    let mut bf = Bufs::new(e.device.id, &mut rng, n_idx)?;
    {
        let s = unsafe { &*(&rig.s as *const Stream) };
        bf.moe_ev.record(&rig.side)?;
        prime(&e, s, &mut bf)?;
    }
    eprintln!(
        "gfx1201 {arch}, rounds {rounds}, b {bs:?}, n_idx {n_idx}; kernargs: HIP_FORCE_DEV_KERNARG={:?} HDP_FLUSH_WA={:?}",
        std::env::var("HIP_FORCE_DEV_KERNARG").ok(),
        std::env::var("DEBUG_CLR_KERNARG_HDP_FLUSH_WA").ok()
    );
    if section == "all" || section == "floors" {
        section_floors(&e, &mut rig, &mut bf, rounds, &mut rng)?;
    }
    let mut stats = Stats::new();
    if section == "all" || section == "ops" || section == "chain" {
        section_ops(&e, &mut rig, &mut bf, &bs, if section == "chain" { rounds.min(40) } else { rounds }, &mut rng, &mut stats)?;
    }
    if section == "all" || section == "chain" {
        section_chain(&e, &mut rig, &mut bf, &bs, rounds, &mut rng, &stats)?;
    }
    if let Ok(p) = std::env::var("BENCH_LABELS") {
        std::fs::write(&p, LABELS.lock().unwrap().join("\n"))?;
        eprintln!("labels -> {p}");
    }
    Ok(())
}
