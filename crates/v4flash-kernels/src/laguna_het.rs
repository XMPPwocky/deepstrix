//! Laguna-S-2.1 — heterogeneous (dual-GPU) decode.
//!
//! Splits the single-device [`crate::laguna::LagunaModel`] forward across the
//! two GPUs to exploit their asymmetric bandwidth:
//!
//!   * **dGPU** (gfx1201, ~600 GB/s): attention + ALL non-expert weights —
//!     q/k/v/o/gate F16 projections, norms, QK-norm, RoPE, GQA, the softplus
//!     gate, the dense layer-0 FFN, the output norm, and the Q6_K LM head, plus
//!     the per-layer f16 KV cache and token-embedding dequant. The profile says
//!     58.6% of decode is the F16 attention projections — bandwidth-bound at the
//!     iGPU's ~229 GB/s — so moving them to the 2.6× dGPU is the main lever.
//!   * **iGPU** (gfx1151, unified memory): the 68.6 GB of routed + shared
//!     experts (the router matvec, MoE SwiGLU, shared expert). Only the iGPU's
//!     `no_system_mem_limit` unified memory can hold them.
//!
//! Per MoE layer the two devices hand off tiny activations:
//!   1. dGPU computes attention → `op` (ffn_inp residual) → `fn_in` (ffn_norm).
//!   2. peer-copy `fn_in` (3072 f32, ~12 KB) dGPU→iGPU.
//!   3. iGPU computes router + routed MoE + shared expert → `ffn_out`.
//!   4. peer-copy `ffn_out` (3072 f32, ~12 KB) iGPU→dGPU.
//!   5. dGPU residual add `h = ffn_out + op`.
//!
//! This first landing is the **correctness-first sequential** split: each
//! handoff is a stream sync (no dual-stream overlap yet). Peer copies obey the
//! load-bearing rule (`project_peer_copy_stream_rule`): `hipMemcpyPeerAsync` is
//! queued on the **source** device's stream (via [`crate::het::sync`]).
//!
//! Layer 0 is dense (no experts) so it runs entirely on the dGPU.

use std::fs::File;
use std::os::unix::fs::FileExt;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::{GgufType};
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, DeviceBuffer, Event, Stream};

use crate::het::sync::{peer_push_f32, peer_push_i32};
use crate::laguna::{
    block_bytes, dequant_q4k_superblock, qmatvec, qmatvec_batched, LagunaHparams, LagunaOps,
    QWeight, EPS, FF_DENSE, FF_EXP, FF_SHEXP, HEAD_DIM, HIDDEN, N_EXPERT, N_KV_HEAD, N_LAYER, TOPK,
    VOCAB,
};
use crate::{
    F16Matvec, GqaAttention, LagunaMoeTiled, MoeGroupBuilder, Q4KMatvec, Q4_KDenseMatvec, Q6KMatvec,
    Q6_KDenseMatvec, RmsNorm, RopeParams, RopeTail, Swiglu, Q8KQuantize, VecAddInplace,
};

/// Sliding-window attention window size (spec `sliding_window`=512,
/// LLAMA_SWA_TYPE_STANDARD). SWA layers (il%4 != 0) attend only the previous
/// `SWA_WINDOW` keys inclusive of self.
pub const SWA_WINDOW: usize = 512;

/// Physical KV ring-buffer capacity for SWA layers. Must be >= the largest
/// prefill chunk `B_MAX` (512) + `SWA_WINDOW` so that, within any left-to-right
/// chunk, every in-window key for the whole chunk span
/// `[chunk_start-511, chunk_end]` is still physically resident (not yet
/// overwritten by a later position). 2048 gives headroom over the 1024 minimum
/// and is a multiple of the 32-key attention tile. SWA layers only ever read the
/// last SWA_WINDOW keys, so allocating full context for them is pure waste — the
/// ring drops 36 of 48 layers from `max_kv` rows to `SWA_RING_CAP` rows.
pub const SWA_RING_CAP: usize = 2048;

/// The 5 decode attention projections (wq/wk/wv/wo/wg) route through the
/// bandwidth-driven b128 vector-load matvec (`f16.matvec_wide_vec`, 96% DRAM BW)
/// by DEFAULT; LAGUNA_PROJ_LDS_TILED=0 restores the scalar `f16.matvec` (75% BW)
/// for A/B. e2e decode: +7.8% @4K, +6.1% @32K, token-exact. Cached once (hot path).
fn proj_vec_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("LAGUNA_PROJ_LDS_TILED").ok().as_deref() != Some("0"))
}

/// Full Laguna kernel set for one device. Kernels are HSACO blobs (tiny), so
/// carrying the whole set on both devices is cheap; only the weights split.
struct HetKernels {
    f16: F16Matvec,
    q4d: Q4_KDenseMatvec,
    q6d: Q6_KDenseMatvec,
    gqa: GqaAttention,
    rms: RmsNorm,
    rope: RopeTail,
    q8k: Q8KQuantize,
    q4b: Q4KMatvec,
    q6b: Q6KMatvec,
    ops: LagunaOps,
    swiglu: Swiglu,
    vadd: VecAddInplace,
}

impl HetKernels {
    fn for_arch(arch: &str) -> eyre::Result<Self> {
        Ok(Self {
            f16: F16Matvec::for_arch(arch)?,
            q4d: Q4_KDenseMatvec::for_arch(arch)?,
            q6d: Q6_KDenseMatvec::for_arch(arch)?,
            gqa: GqaAttention::for_arch(arch)?,
            rms: RmsNorm::for_arch(arch)?,
            rope: RopeTail::for_arch(arch)?,
            q8k: Q8KQuantize::for_arch(arch)?,
            q4b: Q4KMatvec::for_arch(arch)?,
            q6b: Q6KMatvec::for_arch(arch)?,
            ops: LagunaOps::for_arch(arch)?,
            swiglu: Swiglu::for_arch(arch)?,
            vadd: VecAddInplace::for_arch(arch)?,
        })
    }
}

/// dGPU-resident per-layer weights: attention + norms + (layer 0) dense FFN.
struct DgpuLayer {
    is_full: bool,
    n_head: usize,
    attn_norm: DeviceBuffer<f32>,
    ffn_norm: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    wq: DeviceBuffer<u8>,
    wk: DeviceBuffer<u8>,
    wv: DeviceBuffer<u8>,
    wo: DeviceBuffer<u8>,
    wg: DeviceBuffer<u8>,
    dense: Option<(QWeight, QWeight, QWeight)>, // gate, up, down (layer 0 only)
    // dGPU-resident copy of the shared expert (gate, up, down) for MoE layers.
    // Populated so the shared expert can run on the dGPU CONCURRENTLY with the
    // iGPU routed experts (the dGPU is otherwise idle during the MoE window).
    // ~6 MB/layer (~288 MB total). See `LAGUNA_SHEXP_DGPU`.
    shexp: Option<(QWeight, QWeight, QWeight)>,
    // dGPU-resident router (matvec weight + score-correction bias) for the
    // het-split path: the router runs on the dGPU so `fn_in` never has to make a
    // round trip to the iGPU before the hot experts can start. MoE layers only.
    router: Option<DeviceBuffer<f32>>,
    router_bias: Option<DeviceBuffer<f32>>,
    // dGPU-resident HOT routed experts (the K globally-most-frequent experts for
    // this layer) + the global->local slot map. Populated when the het-MoE split
    // is enabled (`LAGUNA_HOT_EXPERTS_DGPU=<file>`).
    hot: Option<DgpuHot>,
}

/// dGPU-resident hot routed experts: the K most-frequent experts for a layer,
/// packed exactly like the iGPU's all-expert buffers but only K entries wide.
struct DgpuHot {
    hot_map: DeviceBuffer<i32>, // [N_EXPERT] global id -> local slot 0..K, else -1
    gate_all: DeviceBuffer<u8>, // [K * gate_stride]
    up_all: DeviceBuffer<u8>,   // [K * up_stride]
    down_all: DeviceBuffer<u8>, // [K * down_stride]
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    down_dt: GgufType,
    n_hot: usize, // K — the number of dGPU-resident hot experts for this layer
}

/// iGPU-resident per-layer MoE weights (None for the dense layer 0).
struct IgpuLayer {
    moe: Option<HetMoe>,
    // iGPU-resident copy of the per-layer hot-map (global expert id -> local
    // dGPU slot, or -1). Needed by the COLD by-expert group builder
    // (`launch_hetsplit` mode=0) so the iGPU knows which selections the dGPU
    // took. Populated with the dGPU hot residency (prefill het-split).
    hot_map: Option<DeviceBuffer<i32>>,
}

struct HetMoe {
    router: DeviceBuffer<f32>, // [n_expert, hidden] f32
    bias: DeviceBuffer<f32>,   // [n_expert]
    sh_gate: QWeight,
    sh_up: QWeight,
    sh_down: QWeight,
    gate_all: DeviceBuffer<u8>,
    up_all: DeviceBuffer<u8>,
    down_all: DeviceBuffer<u8>,
    gate_stride: usize,
    up_stride: usize,
    down_stride: usize,
    down_dt: GgufType,
}

/// dGPU scratch — attention path + residual carriers + output head.
struct DgpuScratch {
    h: DeviceBuffer<f32>,
    ain: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    qf: DeviceBuffer<u16>,
    k: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    od: DeviceBuffer<f32>,
    // split-KV decode attention scratch (partials merged by the combine kernel).
    attn_op: DeviceBuffer<f32>,  // [n_head_max * DECODE_KV_SPLITS_MAX * head_dim]
    attn_mp: DeviceBuffer<f32>,  // [n_head_max * DECODE_KV_SPLITS_MAX]
    attn_lp: DeviceBuffer<f32>,  // [n_head_max * DECODE_KV_SPLITS_MAX]
    gate_logits: DeviceBuffer<f32>,
    op: DeviceBuffer<f32>,       // o_proj + residual == ffn_inp
    fn_in: DeviceBuffer<f32>,    // ffn_norm(op) -> pushed to iGPU
    ffn_out: DeviceBuffer<f32>,  // dense (layer 0) FFN output
    gate_big: DeviceBuffer<f32>,
    up_big: DeviceBuffer<f32>,
    sw_big: DeviceBuffer<f32>,
    moe_recv: DeviceBuffer<f32>, // ffn_out received from iGPU
    rn: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    // dGPU shared-expert scratch (overlaps iGPU routed experts).
    sh_gate: DeviceBuffer<f32>, // [FF_SHEXP]
    sh_up: DeviceBuffer<f32>,   // [FF_SHEXP]
    sh_sw: DeviceBuffer<f32>,   // [FF_SHEXP]
    sh_down: DeviceBuffer<f32>, // [HIDDEN] shared-expert output
    // het-MoE-split (router-on-dGPU + hot routed experts on dGPU) scratch.
    sel: DeviceBuffer<i32>,          // [TOPK] router selection (global ids)
    ew: DeviceBuffer<f32>,           // [TOPK] routing weights
    router_probs: DeviceBuffer<f32>, // [N_EXPERT]
    router_scores: DeviceBuffer<f32>,// [N_EXPERT]
    hot_sel: DeviceBuffer<i32>,      // [TOPK] local slot (hot) else -1
    hot_ew: DeviceBuffer<f32>,       // [TOPK]
    cold_sel: DeviceBuffer<i32>,     // [TOPK] global id (cold) else -1
    cold_ew: DeviceBuffer<f32>,      // [TOPK]
    xq_hidden: DeviceBuffer<u8>,     // [(HIDDEN/256)*292] q8k(fn_in)
    mid_hot: DeviceBuffer<f32>,      // [TOPK*FF_EXP]
    xq_mid: DeviceBuffer<u8>,        // [TOPK*(FF_EXP/256)*292]
    acc_hot: DeviceBuffer<f32>,      // [HIDDEN] hot routed partial sum
}

/// iGPU scratch — router + routed MoE + shared expert.
struct IgpuScratch {
    fn_in_recv: DeviceBuffer<f32>, // received from dGPU
    // het-split: cold selection computed on the dGPU and pushed here (the iGPU
    // runs no router in split mode). `[TOPK]`, sentinel -1 in hot slots.
    cold_sel_recv: DeviceBuffer<i32>,
    cold_ew_recv: DeviceBuffer<f32>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    router_probs: DeviceBuffer<f32>,
    router_scores: DeviceBuffer<f32>,
    xq_hidden: DeviceBuffer<u8>,
    mid: DeviceBuffer<f32>,
    xq_mid: DeviceBuffer<u8>,
    acc: DeviceBuffer<f32>,
    gate_s: DeviceBuffer<f32>,
    up_s: DeviceBuffer<f32>,
    sw_s: DeviceBuffer<f32>,
    down_s: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
}

/// Heterogeneous Laguna model. Attention + non-expert on the dGPU, experts on
/// the iGPU.
#[allow(dead_code)]
pub struct LagunaHetModel {
    dgpu: Device,
    igpu: Device,
    dstream: Stream, // dGPU compute + dGPU->iGPU peer copies
    istream: Stream, // iGPU compute + iGPU->dGPU peer copies
    // Cross-device handoff events (created no-timing). `fn_in_evt` is recorded
    // on the dGPU stream after the fn_in push; the iGPU stream waits on it.
    // `moe_evt` is recorded on the iGPU stream after the ffn_out push; the dGPU
    // stream waits on it. This replaces per-layer host `synchronize()`s (94/token)
    // with device-side waits so the host enqueues the whole token without blocking.
    fn_in_evt: Event,
    moe_evt: Event,
    // Per-lane handoff events for the two-lane PIPELINED batched prefill.
    // Index 0 = lane A, 1 = lane B. Kept separate from the decode-path
    // `fn_in_evt`/`moe_evt` so the pipeline can have two independent
    // dGPU↔iGPU handoffs in flight (lane A's iGPU MoE overlaps lane B's
    // dGPU attention).
    pipe_fn_in_evt: [Event; 2],
    pipe_moe_evt: [Event; 2],

    gguf: MappedGguf,
    raw_file: File,
    hp: LagunaHparams,
    rope_full: RopeParams,
    rope_swa: RopeParams,

    dk: HetKernels, // dGPU kernels
    ik: HetKernels, // iGPU kernels

    dlayers: Vec<DgpuLayer>,
    ilayers: Vec<IgpuLayer>,

    output_norm: DeviceBuffer<f32>,
    output_w: QWeight,
    tok_embd_off: u64,
    tok_embd_row_bytes: usize,

    max_kv: usize,
    /// Per-layer physical KV capacity (ring size). Global layers (il%4==0) get
    /// `max_kv`; SWA layers get `min(SWA_RING_CAP, max_kv)`. K/V is indexed by
    /// absolute position `p` at physical slot `p % kv_cap[il]`.
    kv_cap: Vec<usize>,
    kc: Vec<DeviceBuffer<u16>>,
    vc: Vec<DeviceBuffer<u16>>,
    kv_len: usize,

    ds: DgpuScratch,
    is: IgpuScratch,

    // LAGUNA_HET_DIAG=1: accumulate per-token dGPU-attn vs iGPU-MoE wall (µs)
    // by syncing at the handoff points. Sequential design means the syncs only
    // measure, they don't remove overlap.
    diag: bool,
    diag_dgpu_us: u64,
    diag_igpu_us: u64,

    // Run the shared expert on the dGPU concurrently with the iGPU routed
    // experts (default on). `LAGUNA_SHEXP_DGPU=0` restores the all-iGPU MoE.
    shexp_dgpu: bool,

    // LAGUNA_EXPERT_HIST=1: accumulate a per-layer routed-expert selection
    // histogram over decode steps. Used to size the hot-expert set (which
    // globally-frequent experts to make dGPU-resident) and to measure the
    // per-token top-K capture fraction that bounds the het-MoE overlap win.
    hist_enabled: bool,
    expert_hist: Vec<Vec<u32>>, // [N_LAYER][N_EXPERT] selection counts
    hist_sel_host: Vec<i32>,    // reusable [TOPK] host scratch

    // het-MoE split: hot routed experts on the dGPU overlap the cold experts on
    // the iGPU. Enabled by `LAGUNA_HOT_EXPERTS_DGPU=<file>` (per-layer hot-expert
    // id list). `hot_split` is true only when the hot residency actually loaded.
    hetmoe: crate::laguna_het_moe::LagunaHetMoeSplit, // partition kernel (dGPU)
    hot_split: bool,

    // Prefill het-MoE split: when true, `attn_batched` leaves `fn_in` on the
    // dGPU (the router runs on the dGPU in the split path) instead of pushing it
    // to the iGPU. Set for the duration of `prefill_batched_het` only.
    prefill_split: bool,
    // Per-token dGPU-resident-slot cap for the prefill split (hetsplit `cap`).
    // TOPK == no cap (every resident selection goes to the dGPU). Lower values
    // shift work back to the iGPU to balance the two device legs.
    prefill_hot_cap: u32,
}

/// Parse a hot-experts file: `N_LAYER` lines, line `il` holds the space-
/// separated global expert ids to make dGPU-resident for layer `il` (blank for
/// the dense layer 0). Missing trailing lines => empty (all-cold) layers.
fn parse_hot_experts(path: &str) -> eyre::Result<Vec<Vec<usize>>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| eyre!("LAGUNA_HOT_EXPERTS_DGPU={path}: {e}"))?;
    let mut per_layer: Vec<Vec<usize>> = text
        .lines()
        .map(|line| {
            line.split_whitespace()
                .filter_map(|t| t.parse::<usize>().ok())
                .collect()
        })
        .collect();
    per_layer.resize(N_LAYER, Vec::new());
    per_layer.truncate(N_LAYER);
    Ok(per_layer)
}

impl LagunaHetModel {
    /// Load the model split across `dgpu` (attention + non-expert) and `igpu`
    /// (experts). `dgpu_arch`/`igpu_arch` are the gcn arch names.
    pub fn load(
        gguf_path: &str,
        dgpu: Device,
        dgpu_arch: &str,
        igpu: Device,
        igpu_arch: &str,
        max_kv: usize,
    ) -> eyre::Result<Self> {
        // Enable bidirectional peer access (peer copies both directions).
        dgpu.set_current()?;
        if !dgpu.can_access_peer(igpu)? {
            return Err(eyre!("dGPU {} cannot peer-access iGPU {}", dgpu.id, igpu.id));
        }
        let _ = dgpu.enable_peer_access(igpu);
        igpu.set_current()?;
        if !igpu.can_access_peer(dgpu)? {
            return Err(eyre!("iGPU {} cannot peer-access dGPU {}", igpu.id, dgpu.id));
        }
        let _ = igpu.enable_peer_access(dgpu);

        let gguf = MappedGguf::open(gguf_path)?;
        let raw_file = File::open(gguf_path)?;
        let hp = LagunaHparams::from_gguf(gguf.gguf());
        let rope_full = hp.rope_full();
        let rope_swa = hp.rope_swa();

        // --- device-targeted load helpers. hipMalloc allocates on the CURRENT
        //     device, so callers must set_current(dev) before invoking these. ---
        let mk_qweight = |dev: i32, name: &str| -> eyre::Result<QWeight> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            let bytes = gguf.read_tensor(t)?;
            let mut b = DeviceBuffer::<u8>::new(dev, bytes.len())?;
            b.copy_from_host(&bytes)?;
            Ok(QWeight { bytes: b, dtype: t.dtype })
        };
        let mk_u8 = |dev: i32, name: &str| -> eyre::Result<DeviceBuffer<u8>> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            if t.dtype != GgufType::F16 {
                return Err(eyre!("{name} expected F16, got {:?}", t.dtype));
            }
            let bytes = gguf.read_tensor(t)?;
            let mut b = DeviceBuffer::<u8>::new(dev, bytes.len())?;
            b.copy_from_host(&bytes)?;
            Ok(b)
        };
        let mk_f32 = |dev: i32, name: &str| -> eyre::Result<DeviceBuffer<f32>> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            if t.dtype != GgufType::F32 {
                return Err(eyre!("{name} expected F32, got {:?}", t.dtype));
            }
            let bytes = gguf.read_tensor(t)?;
            let v: Vec<f32> = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            let mut b = DeviceBuffer::<f32>::new(dev, v.len())?;
            b.copy_from_host(&v)?;
            Ok(b)
        };

        // het-MoE split: optional per-layer hot-expert residency on the dGPU.
        // `LAGUNA_HOT_EXPERTS_DGPU=<file>` — N_LAYER lines, line `il` = the
        // space-separated global expert ids to make dGPU-resident for layer il
        // (empty for the dense layer 0). Absent/empty => pure-iGPU MoE fallback.
        let hot_experts: Option<Vec<Vec<usize>>> = match std::env::var("LAGUNA_HOT_EXPERTS_DGPU") {
            Ok(path) if !path.is_empty() => Some(parse_hot_experts(&path)?),
            _ => None,
        };
        // Pack the K hot experts' rows out of the full [N_EXPERT*stride] tensor
        // into a K-wide dGPU-resident buffer (local slot j == hot_ids[j]).
        let mk_hot = |dev: i32, name: &str, stride: usize, hot_ids: &[usize]| -> eyre::Result<DeviceBuffer<u8>> {
            let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
            let bytes = gguf.read_tensor(t)?;
            let want = N_EXPERT * stride;
            if bytes.len() != want {
                return Err(eyre!("{name}: expected {want} bytes, got {}", bytes.len()));
            }
            let mut packed = vec![0u8; hot_ids.len() * stride];
            for (j, &g) in hot_ids.iter().enumerate() {
                if g >= N_EXPERT {
                    return Err(eyre!("hot expert id {g} >= N_EXPERT for {name}"));
                }
                packed[j * stride..(j + 1) * stride]
                    .copy_from_slice(&bytes[g * stride..(g + 1) * stride]);
            }
            let mut b = DeviceBuffer::<u8>::new(dev, packed.len())?;
            b.copy_from_host(&packed)?;
            Ok(b)
        };

        let mut dlayers = Vec::with_capacity(N_LAYER);
        let mut ilayers = Vec::with_capacity(N_LAYER);
        for il in 0..N_LAYER {
            let is_full = il % 4 == 0;
            let n_head = if is_full { 48 } else { 72 };
            let p = |s: &str| format!("blk.{il}.{s}");

            // --- dGPU: attention + norms (+ layer-0 dense FFN) ---
            dgpu.set_current()?;
            let dense = if il == 0 {
                Some((
                    mk_qweight(dgpu.id, &p("ffn_gate.weight"))?,
                    mk_qweight(dgpu.id, &p("ffn_up.weight"))?,
                    mk_qweight(dgpu.id, &p("ffn_down.weight"))?,
                ))
            } else {
                None
            };
            // dGPU-resident shared expert (MoE layers only) for the overlap path.
            let shexp = if il == 0 {
                None
            } else {
                Some((
                    mk_qweight(dgpu.id, &p("ffn_gate_shexp.weight"))?,
                    mk_qweight(dgpu.id, &p("ffn_up_shexp.weight"))?,
                    mk_qweight(dgpu.id, &p("ffn_down_shexp.weight"))?,
                ))
            };
            dlayers.push(DgpuLayer {
                is_full,
                n_head,
                attn_norm: mk_f32(dgpu.id, &p("attn_norm.weight"))?,
                ffn_norm: mk_f32(dgpu.id, &p("ffn_norm.weight"))?,
                q_norm: mk_f32(dgpu.id, &p("attn_q_norm.weight"))?,
                k_norm: mk_f32(dgpu.id, &p("attn_k_norm.weight"))?,
                wq: mk_u8(dgpu.id, &p("attn_q.weight"))?,
                wk: mk_u8(dgpu.id, &p("attn_k.weight"))?,
                wv: mk_u8(dgpu.id, &p("attn_v.weight"))?,
                wo: mk_u8(dgpu.id, &p("attn_output.weight"))?,
                wg: mk_u8(dgpu.id, &p("attn_gate.weight"))?,
                dense,
                shexp,
                router: None,
                router_bias: None,
                hot: None,
            });

            // --- iGPU: routed + shared experts + router ---
            igpu.set_current()?;
            let moe = if il == 0 {
                None
            } else {
                let g = gguf.gguf();
                let gate_t = g.tensor(&p("ffn_gate_exps.weight")).unwrap();
                let up_t = g.tensor(&p("ffn_up_exps.weight")).unwrap();
                let down_t = g.tensor(&p("ffn_down_exps.weight")).unwrap();
                let gate_stride = FF_EXP * (HIDDEN / 256) * block_bytes(gate_t.dtype);
                let up_stride = FF_EXP * (HIDDEN / 256) * block_bytes(up_t.dtype);
                let down_stride = HIDDEN * (FF_EXP / 256) * block_bytes(down_t.dtype);
                let down_dt = down_t.dtype;
                let mk_resident = |name: &str, stride: usize| -> eyre::Result<DeviceBuffer<u8>> {
                    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
                    let bytes = gguf.read_tensor(t)?;
                    let want = N_EXPERT * stride;
                    if bytes.len() != want {
                        return Err(eyre!(
                            "{name}: expected {want} bytes ({N_EXPERT}*{stride}), got {}",
                            bytes.len()
                        ));
                    }
                    let mut b = DeviceBuffer::<u8>::new(igpu.id, bytes.len())?;
                    b.copy_from_host(&bytes)?;
                    Ok(b)
                };
                Some(HetMoe {
                    router: mk_f32(igpu.id, &p("ffn_gate_inp.weight"))?,
                    bias: mk_f32(igpu.id, &p("exp_probs_b.bias"))?,
                    sh_gate: mk_qweight(igpu.id, &p("ffn_gate_shexp.weight"))?,
                    sh_up: mk_qweight(igpu.id, &p("ffn_up_shexp.weight"))?,
                    sh_down: mk_qweight(igpu.id, &p("ffn_down_shexp.weight"))?,
                    gate_all: mk_resident(&p("ffn_gate_exps.weight"), gate_stride)?,
                    up_all: mk_resident(&p("ffn_up_exps.weight"), up_stride)?,
                    down_all: mk_resident(&p("ffn_down_exps.weight"), down_stride)?,
                    gate_stride,
                    up_stride,
                    down_stride,
                    down_dt,
                })
            };
            ilayers.push(IgpuLayer { moe, hot_map: None });

            // --- dGPU: router + HOT routed experts (het-split), MoE layers ---
            if il > 0 {
                if let Some(hot_ids) = hot_experts.as_ref().map(|h| h.get(il)).flatten() {
                    if !hot_ids.is_empty() {
                        dgpu.set_current()?;
                        let g = gguf.gguf();
                        let gate_t = g.tensor(&p("ffn_gate_exps.weight")).unwrap();
                        let up_t = g.tensor(&p("ffn_up_exps.weight")).unwrap();
                        let down_t = g.tensor(&p("ffn_down_exps.weight")).unwrap();
                        let gate_stride = FF_EXP * (HIDDEN / 256) * block_bytes(gate_t.dtype);
                        let up_stride = FF_EXP * (HIDDEN / 256) * block_bytes(up_t.dtype);
                        let down_stride = HIDDEN * (FF_EXP / 256) * block_bytes(down_t.dtype);
                        let down_dt = down_t.dtype;
                        // hot_map[global] = local slot, else -1.
                        let mut map = vec![-1i32; N_EXPERT];
                        for (j, &e) in hot_ids.iter().enumerate() {
                            map[e] = j as i32;
                        }
                        let mut hot_map = DeviceBuffer::<i32>::new(dgpu.id, N_EXPERT)?;
                        hot_map.copy_from_host(&map)?;
                        let hot = DgpuHot {
                            hot_map,
                            gate_all: mk_hot(dgpu.id, &p("ffn_gate_exps.weight"), gate_stride, hot_ids)?,
                            up_all: mk_hot(dgpu.id, &p("ffn_up_exps.weight"), up_stride, hot_ids)?,
                            down_all: mk_hot(dgpu.id, &p("ffn_down_exps.weight"), down_stride, hot_ids)?,
                            gate_stride,
                            up_stride,
                            down_stride,
                            down_dt,
                            n_hot: hot_ids.len(),
                        };
                        let dl = dlayers.last_mut().unwrap();
                        dl.router = Some(mk_f32(dgpu.id, &p("ffn_gate_inp.weight"))?);
                        dl.router_bias = Some(mk_f32(dgpu.id, &p("exp_probs_b.bias"))?);
                        dl.hot = Some(hot);
                        // iGPU-resident hot-map for the COLD by-expert group
                        // builder (mode=0 needs the same remap to exclude the
                        // dGPU-taken selections).
                        igpu.set_current()?;
                        let mut hot_map_i = DeviceBuffer::<i32>::new(igpu.id, N_EXPERT)?;
                        hot_map_i.copy_from_host(&map)?;
                        ilayers.last_mut().unwrap().hot_map = Some(hot_map_i);
                        dgpu.set_current()?;
                    }
                }
            }
        }
        let hot_split = hot_experts.is_some();

        // --- output head + KV cache on dGPU ---
        dgpu.set_current()?;
        let output_norm = mk_f32(dgpu.id, "output_norm.weight")?;
        let output_w = mk_qweight(dgpu.id, "output.weight")?;
        let tok_embd_t = gguf.gguf().tensor("token_embd.weight").ok_or_else(|| eyre!("no token_embd"))?;
        let tok_embd_off = tok_embd_t.abs_offset;
        let tok_embd_row_bytes = (HIDDEN / 256) * 144;

        // SWA-aware KV allocation: global layers (il%4==0) hold the full context
        // (max_kv rows); SWA layers only ever read the last SWA_WINDOW keys, so a
        // small ring buffer (SWA_RING_CAP rows, capped at max_kv for short ctx)
        // suffices. At 100K ctx this drops KV from ~19 GB to ~5 GB.
        let swa_cap = SWA_RING_CAP.min(max_kv);
        let mut kv_cap = Vec::with_capacity(N_LAYER);
        let mut kc = Vec::with_capacity(N_LAYER);
        let mut vc = Vec::with_capacity(N_LAYER);
        for il in 0..N_LAYER {
            let cap = if il % 4 == 0 { max_kv } else { swa_cap };
            kv_cap.push(cap);
            kc.push(DeviceBuffer::<u16>::new(dgpu.id, cap * N_KV_HEAD * HEAD_DIM)?);
            vc.push(DeviceBuffer::<u16>::new(dgpu.id, cap * N_KV_HEAD * HEAD_DIM)?);
        }

        // --- dGPU scratch ---
        let n_embd_q_max = 72 * HEAD_DIM;
        let mkd = |n: usize| DeviceBuffer::<f32>::new(dgpu.id, n);
        let ds = DgpuScratch {
            h: mkd(HIDDEN)?,
            ain: mkd(HIDDEN)?,
            q: mkd(n_embd_q_max)?,
            qn: mkd(n_embd_q_max)?,
            qf: DeviceBuffer::<u16>::new(dgpu.id, n_embd_q_max)?,
            k: mkd(N_KV_HEAD * HEAD_DIM)?,
            kn: mkd(N_KV_HEAD * HEAD_DIM)?,
            v: mkd(N_KV_HEAD * HEAD_DIM)?,
            od: mkd(n_embd_q_max)?,
            attn_op: mkd(72 * crate::gqa_attention::DECODE_KV_SPLITS_MAX as usize * HEAD_DIM)?,
            attn_mp: mkd(72 * crate::gqa_attention::DECODE_KV_SPLITS_MAX as usize)?,
            attn_lp: mkd(72 * crate::gqa_attention::DECODE_KV_SPLITS_MAX as usize)?,
            gate_logits: mkd(72)?,
            op: mkd(HIDDEN)?,
            fn_in: mkd(HIDDEN)?,
            ffn_out: mkd(HIDDEN)?,
            gate_big: mkd(FF_DENSE)?,
            up_big: mkd(FF_DENSE)?,
            sw_big: mkd(FF_DENSE)?,
            moe_recv: mkd(HIDDEN)?,
            rn: mkd(HIDDEN)?,
            logits: mkd(VOCAB)?,
            sh_gate: mkd(FF_SHEXP)?,
            sh_up: mkd(FF_SHEXP)?,
            sh_sw: mkd(FF_SHEXP)?,
            sh_down: mkd(HIDDEN)?,
            sel: DeviceBuffer::<i32>::new(dgpu.id, TOPK)?,
            ew: mkd(TOPK)?,
            router_probs: mkd(N_EXPERT)?,
            router_scores: mkd(N_EXPERT)?,
            hot_sel: DeviceBuffer::<i32>::new(dgpu.id, TOPK)?,
            hot_ew: mkd(TOPK)?,
            cold_sel: DeviceBuffer::<i32>::new(dgpu.id, TOPK)?,
            cold_ew: mkd(TOPK)?,
            xq_hidden: DeviceBuffer::<u8>::new(dgpu.id, (HIDDEN / 256) * 292)?,
            mid_hot: mkd(TOPK * FF_EXP)?,
            xq_mid: DeviceBuffer::<u8>::new(dgpu.id, TOPK * (FF_EXP / 256) * 292)?,
            acc_hot: mkd(HIDDEN)?,
        };

        // --- iGPU scratch ---
        igpu.set_current()?;
        let mki = |n: usize| DeviceBuffer::<f32>::new(igpu.id, n);
        let is = IgpuScratch {
            fn_in_recv: mki(HIDDEN)?,
            cold_sel_recv: DeviceBuffer::<i32>::new(igpu.id, TOPK)?,
            cold_ew_recv: mki(TOPK)?,
            sel: DeviceBuffer::<i32>::new(igpu.id, TOPK)?,
            ew: mki(TOPK)?,
            router_probs: mki(N_EXPERT)?,
            router_scores: mki(N_EXPERT)?,
            xq_hidden: DeviceBuffer::<u8>::new(igpu.id, (HIDDEN / 256) * 292)?,
            mid: mki(TOPK * FF_EXP)?,
            xq_mid: DeviceBuffer::<u8>::new(igpu.id, TOPK * (FF_EXP / 256) * 292)?,
            acc: mki(HIDDEN)?,
            gate_s: mki(FF_SHEXP)?,
            up_s: mki(FF_SHEXP)?,
            sw_s: mki(FF_SHEXP)?,
            down_s: mki(HIDDEN)?,
            ffn_out: mki(HIDDEN)?,
        };

        dgpu.set_current()?;
        let dstream = Stream::new(dgpu.id)?;
        let fn_in_evt = Event::new_no_timing()?;
        igpu.set_current()?;
        let istream = Stream::new(igpu.id)?;
        let moe_evt = Event::new_no_timing()?;
        // fn_in events fire on the dGPU (source) stream; moe events on the
        // iGPU (source) stream. HIP events aren't strictly device-bound for
        // cross-device waits (peer access is enabled), but we mirror the
        // decode-path convention: create fn_in events under dgpu-current and
        // moe events under igpu-current.
        let pipe_moe_evt = [Event::new_no_timing()?, Event::new_no_timing()?];
        dgpu.set_current()?;
        let pipe_fn_in_evt = [Event::new_no_timing()?, Event::new_no_timing()?];

        Ok(Self {
            dgpu,
            igpu,
            dstream,
            istream,
            fn_in_evt,
            moe_evt,
            pipe_fn_in_evt,
            pipe_moe_evt,
            gguf,
            raw_file,
            hp,
            rope_full,
            rope_swa,
            hetmoe: {
                dgpu.set_current()?;
                crate::laguna_het_moe::LagunaHetMoeSplit::for_arch(dgpu_arch)?
            },
            hot_split,
            dk: {
                dgpu.set_current()?;
                HetKernels::for_arch(dgpu_arch)?
            },
            ik: {
                igpu.set_current()?;
                let k = HetKernels::for_arch(igpu_arch)?;
                dgpu.set_current()?;
                k
            },
            dlayers,
            ilayers,
            output_norm,
            output_w,
            tok_embd_off,
            tok_embd_row_bytes,
            max_kv,
            kv_cap,
            kc,
            vc,
            kv_len: 0,
            ds,
            is,
            diag: std::env::var("LAGUNA_HET_DIAG").map(|v| v == "1").unwrap_or(false),
            diag_dgpu_us: 0,
            diag_igpu_us: 0,
            shexp_dgpu: std::env::var("LAGUNA_SHEXP_DGPU").map(|v| v != "0").unwrap_or(true),
            hist_enabled: std::env::var("LAGUNA_EXPERT_HIST").map(|v| v == "1").unwrap_or(false),
            expert_hist: vec![vec![0u32; N_EXPERT]; N_LAYER],
            hist_sel_host: vec![0i32; TOPK],
            prefill_split: false,
            prefill_hot_cap: std::env::var("LAGUNA_PREFILL_HOT_CAP")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(TOPK as u32),
        })
    }

    pub fn hparams(&self) -> &LagunaHparams {
        &self.hp
    }

    pub fn reset(&mut self) {
        self.kv_len = 0;
    }

    /// (dGPU-attn µs, iGPU-MoE µs) accumulated across all layers/tokens since
    /// the last [`reset_diag`]. Only populated when `LAGUNA_HET_DIAG=1`.
    pub fn diag_split(&self) -> (u64, u64) {
        (self.diag_dgpu_us, self.diag_igpu_us)
    }
    pub fn reset_diag(&mut self) {
        self.diag_dgpu_us = 0;
        self.diag_igpu_us = 0;
    }

    /// Per-layer routed-expert selection histogram (`[N_LAYER][N_EXPERT]`).
    /// Only populated when `LAGUNA_EXPERT_HIST=1`.
    pub fn expert_hist(&self) -> &[Vec<u32>] {
        &self.expert_hist
    }

    /// For each layer, pick the `k` globally-most-frequent experts (descending
    /// count) and return their ids. This is the static hot set the dGPU would
    /// hold resident.
    pub fn hot_experts_per_layer(&self, k: usize) -> Vec<Vec<usize>> {
        self.expert_hist
            .iter()
            .map(|counts| {
                let mut idx: Vec<usize> = (0..counts.len()).collect();
                idx.sort_by(|&a, &b| counts[b].cmp(&counts[a]).then(a.cmp(&b)));
                idx.truncate(k);
                idx
            })
            .collect()
    }

    /// Fraction of all per-token top-K routing SLOTS that land on the `k`
    /// dGPU-resident hot experts (per-layer hot set = the k most frequent in
    /// that layer). This upper-bounds the iGPU work that can be moved to the
    /// dGPU. Returns (overall_fraction, per_layer_fraction).
    pub fn hot_capture_fraction(&self, k: usize) -> (f64, Vec<f64>) {
        let hot = self.hot_experts_per_layer(k);
        let mut tot_all = 0u64;
        let mut tot_hot = 0u64;
        let per_layer: Vec<f64> = self
            .expert_hist
            .iter()
            .enumerate()
            .map(|(il, counts)| {
                let layer_all: u64 = counts.iter().map(|&c| c as u64).sum();
                if layer_all == 0 {
                    return 0.0;
                }
                let layer_hot: u64 = hot[il].iter().map(|&e| counts[e] as u64).sum();
                tot_all += layer_all;
                tot_hot += layer_hot;
                layer_hot as f64 / layer_all as f64
            })
            .collect();
        let overall = if tot_all == 0 { 0.0 } else { tot_hot as f64 / tot_all as f64 };
        (overall, per_layer)
    }

    /// Host Q4_K dequant of one token-embedding row -> dGPU hidden.
    fn embed(&mut self, tok_id: usize) -> eyre::Result<()> {
        let mut rb = vec![0u8; self.tok_embd_row_bytes];
        self.raw_file
            .read_exact_at(&mut rb, self.tok_embd_off + (tok_id as u64) * self.tok_embd_row_bytes as u64)?;
        let mut row = vec![0f32; HIDDEN];
        for sb in 0..(HIDDEN / 256) {
            dequant_q4k_superblock(&rb[sb * 144..(sb + 1) * 144], &mut row[sb * 256..(sb + 1) * 256]);
        }
        self.dgpu.set_current()?;
        self.ds.h.copy_from_host(&row)?;
        Ok(())
    }

    /// One transformer layer at `pos`. dGPU attention → (MoE layers) iGPU FFN →
    /// dGPU residual. Sequential handoffs (stream syncs).
    fn layer(&mut self, il: usize, pos: usize) -> eyre::Result<()> {
        let n_head = self.dlayers[il].n_head;
        let is_full = self.dlayers[il].is_full;
        let is_dense = self.dlayers[il].dense.is_some();
        let n_embd_q = n_head * HEAD_DIM;
        let n_rot = if is_full { self.hp.n_rot_full as u32 } else { self.hp.n_rot_swa as u32 };
        let n_kv = pos + 1;
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let diag_t0 = if self.diag { Some(std::time::Instant::now()) } else { None };

        // ================= dGPU: attention -> op (ffn_inp), fn_in =================
        self.dgpu.set_current()?;
        {
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let ds = &mut self.ds;
            let st = &self.dstream;
            let rope = if is_full { &self.rope_full } else { &self.rope_swa };
            let cap = self.kv_cap[il]; // KV ring capacity for this layer

            let pv = proj_vec_enabled();
            dk.rms.launch_weighted(st, &mut ds.ain, &ds.h, &dlw.attn_norm, HIDDEN as u32, EPS)?;
            if pv {
                dk.f16.matvec_wide_vec(st, &mut ds.q, &dlw.wq, &ds.ain, n_embd_q as u32, HIDDEN as u32)?;
                dk.f16.matvec_wide_vec(st, &mut ds.k, &dlw.wk, &ds.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
                dk.f16.matvec_wide_vec(st, &mut ds.v, &dlw.wv, &ds.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
            } else {
                dk.f16.matvec(st, &mut ds.q, &dlw.wq, &ds.ain, n_embd_q as u32, HIDDEN as u32)?;
                dk.f16.matvec(st, &mut ds.k, &dlw.wk, &ds.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
                dk.f16.matvec(st, &mut ds.v, &dlw.wv, &ds.ain, (N_KV_HEAD * HEAD_DIM) as u32, HIDDEN as u32)?;
            }
            dk.ops.qk_rmsnorm(st, &mut ds.qn, &ds.q, &dlw.q_norm, n_head as u32, HEAD_DIM as u32, EPS)?;
            dk.ops.qk_rmsnorm(st, &mut ds.kn, &ds.k, &dlw.k_norm, N_KV_HEAD as u32, HEAD_DIM as u32, EPS)?;
            dk.rope.launch_forward(st, &mut ds.qn, n_head as u32, HEAD_DIM as u32, n_rot, pos as u32, rope)?;
            dk.rope.launch_forward(st, &mut ds.kn, N_KV_HEAD as u32, HEAD_DIM as u32, n_rot, pos as u32, rope)?;
            {
                // Ring write: physical slot = pos % cap (single row, never wraps).
                let pslot = (pos % cap) * N_KV_HEAD * HEAD_DIM;
                let mut kslot = self.kc[il].slice_view_mut(pslot, N_KV_HEAD * HEAD_DIM);
                dk.ops.cast_f16(st, &mut kslot, &ds.kn, (N_KV_HEAD * HEAD_DIM) as u32)?;
                let mut vslot = self.vc[il].slice_view_mut(pslot, N_KV_HEAD * HEAD_DIM);
                dk.ops.cast_f16(st, &mut vslot, &ds.v, (N_KV_HEAD * HEAD_DIM) as u32)?;
            }
            dk.ops.cast_f16(st, &mut ds.qf, &ds.qn, n_embd_q as u32)?;
            {
                // SLIDING-WINDOW ATTENTION (SWA). SWA layers (`!is_full`, 3 of every
                // 4 layers) attend only the previous `SWA_WINDOW=512` keys inclusive
                // of self — [pos-511, pos] — per the Laguna spec (sliding_window=512,
                // LLAMA_SWA_TYPE_STANDARD). Full layers attend the whole causal
                // history. K/V are RoPE'd at absolute positions in the ring cache;
                // the window is a key-range restriction [k0, n_kv). We pass the WHOLE
                // physical ring buffer plus `k_base=k0` + `kv_capacity=cap`, and the
                // decode kernels map relative key j -> physical (k0+j) % cap. For
                // global layers cap==max_kv and k0==0, so the modulo is a no-op and
                // the path is byte-identical. Slicing the ring directly would break
                // when the window wraps the ring boundary, so the modulo lives in the
                // kernel. NO-OP at pos < 512 (k0 == 0). WITHOUT SWA, SWA layers
                // over-attended the full history and diverged from the oracle > 512.
                // LAGUNA_SWA_OFF=1 disables windowing (A/B). Note: with the SWA ring
                // (cap < max_kv), SWA_OFF is only meaningful when cap >= n_kv.
                let swa_off = std::env::var("LAGUNA_SWA_OFF").as_deref() == Ok("1");
                let k0 = if is_full || swa_off { 0 } else { n_kv.saturating_sub(SWA_WINDOW) };
                let attn_nkv = n_kv - k0;
                let swa_win = if is_full || swa_off { 0u32 } else { SWA_WINDOW as u32 };
                let qf_v = ds.qf.slice_view(0, n_embd_q);
                let k_v = self.kc[il].slice_view(0, cap * N_KV_HEAD * HEAD_DIM);
                let v_v = self.vc[il].slice_view(0, cap * N_KV_HEAD * HEAD_DIM);
                let mut od_v = ds.od.slice_view_mut(0, n_embd_q);
                // Decode attention kernel selection. Default SPLIT-KV ("flash
                // decoding"): partitions the causal history across n_head*n_splits
                // workgroups so the dGPU is filled and each WG's serial key-tile
                // chain shrinks by n_splits. The batch=1 flash kernel launched only
                // n_head WGs, each marching n_kv/32 tiles serially -> 82.9% of decode
                // wall at 32K (measured). Short contexts (< 512 keys) still use the
                // naive per-key kernel (a single split has no parallelism to gain).
                // `LAGUNA_DECODE_ATTN=naive|flash|splitkv` overrides for A/B.
                use crate::gqa_attention::DecodeAttn;
                let decode_flash_min_kv = crate::gqa_attention::decode_flash_min_kv();
                let variant = crate::gqa_attention::decode_attn_variant();
                // k_base = k0 (absolute start of the windowed key range); the split
                // math stays over the relative count attn_nkv, mapped to physical
                // (k0 + rel) % cap in the kernel.
                let k_base = k0 as u32;
                let capu = cap as u32;
                if variant == DecodeAttn::Naive || attn_nkv < decode_flash_min_kv {
                    dk.gqa.single_query(
                        st, &mut od_v, &qf_v, &k_v, &v_v,
                        n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, attn_nkv as u32, scale,
                        k_base, capu,
                    )?;
                } else if variant == DecodeAttn::Flash {
                    // Flash reuses the prefill kernel's ABSOLUTE causal indexing +
                    // internal sliding-window mask, so it takes the full causal count
                    // n_kv and windows itself (no host-side k_base slice).
                    dk.gqa.single_query_flash(
                        st, &mut od_v, &qf_v, &k_v, &v_v,
                        n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, n_kv as u32, scale,
                        swa_win, capu,
                    )?;
                } else if variant == DecodeAttn::SplitKv {
                    let n_splits = crate::gqa_attention::decode_kv_splits(attn_nkv as u32);
                    dk.gqa.single_query_splitkv(
                        st, &mut od_v, &mut ds.attn_op, &mut ds.attn_mp, &mut ds.attn_lp,
                        &qf_v, &k_v, &v_v,
                        n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, attn_nkv as u32, n_splits, scale,
                        k_base, capu,
                    )?;
                } else {
                    // Default: head-grouped split-KV (K/V staged once per KV head,
                    // reused across all kv_group query heads).
                    let n_splits = crate::gqa_attention::decode_kv_splits_hg(attn_nkv as u32);
                    dk.gqa.single_query_splitkv_hg(
                        st, &mut od_v, &mut ds.attn_op, &mut ds.attn_mp, &mut ds.attn_lp,
                        &qf_v, &k_v, &v_v,
                        n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, attn_nkv as u32, n_splits, scale,
                        k_base, capu,
                    )?;
                }
            }
            if pv {
                dk.f16.matvec_wide_vec(st, &mut ds.gate_logits, &dlw.wg, &ds.ain, n_head as u32, HIDDEN as u32)?;
            } else {
                dk.f16.matvec(st, &mut ds.gate_logits, &dlw.wg, &ds.ain, n_head as u32, HIDDEN as u32)?;
            }
            dk.ops.softplus_gate(st, &mut ds.od, &ds.gate_logits, n_head as u32, HEAD_DIM as u32)?;
            if pv {
                dk.f16.matvec_wide_vec(st, &mut ds.op, &dlw.wo, &ds.od, HIDDEN as u32, n_embd_q as u32)?;
            } else {
                dk.f16.matvec(st, &mut ds.op, &dlw.wo, &ds.od, HIDDEN as u32, n_embd_q as u32)?;
            }
            dk.vadd.launch(st, &mut ds.op, &ds.h, HIDDEN as u32)?;
            dk.rms.launch_weighted(st, &mut ds.fn_in, &ds.op, &dlw.ffn_norm, HIDDEN as u32, EPS)?;
        }

        // ================= FFN =================
        if is_dense {
            // Layer 0 dense FFN — entirely on the dGPU.
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let ds = &mut self.ds;
            let st = &self.dstream;
            let (gw, uw, dw) = dlw.dense.as_ref().unwrap();
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.gate_big, gw, &ds.fn_in, FF_DENSE as u32, HIDDEN as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.up_big, uw, &ds.fn_in, FF_DENSE as u32, HIDDEN as u32)?;
            dk.swiglu.launch(st, &mut ds.sw_big, &ds.gate_big, &ds.up_big, FF_DENSE as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.ffn_out, dw, &ds.sw_big, HIDDEN as u32, FF_DENSE as u32)?;
            ds.h.copy_from_buffer_async(&ds.ffn_out, st)?;
            dk.vadd.launch(st, &mut ds.h, &ds.op, HIDDEN as u32)?;
            return Ok(());
        }

        // ================= het-MoE SPLIT (hot experts on dGPU) =================
        // Router + the K globally-hottest routed experts run on the dGPU (which
        // is otherwise idle during the MoE window), CONCURRENTLY with the cold
        // experts on the iGPU. The two partial routed sums + the dGPU shared
        // expert are recombined by addition after the handoff. Enabled by
        // `LAGUNA_HOT_EXPERTS_DGPU=<file>`; falls through to the pure-iGPU MoE
        // below when a layer has no hot residency.
        if self.hot_split && self.dlayers[il].hot.is_some() {
            self.layer_moe_split(il, diag_t0)?;
            return Ok(());
        }

        // MoE layer: hand fn_in to the iGPU, run experts, take ffn_out back.
        // Handoffs are device-side event waits (no host block): the iGPU stream
        // waits on the dGPU's fn_in push, and the dGPU stream later waits on the
        // iGPU's ffn_out push. Host enqueues the whole layer without stalling.
        // (1) peer-push fn_in dGPU->iGPU on the dGPU (source) stream, record evt.
        peer_push_f32(&self.ds.fn_in, &mut self.is.fn_in_recv, &self.dstream)?;
        self.fn_in_evt.record(&self.dstream)?;

        // (1b) SHARED EXPERT on the dGPU, overlapped with the iGPU routed experts.
        // The dGPU is idle for the whole MoE window; the shared expert only needs
        // `fn_in` (already dGPU-resident), so enqueue it on `dstream` right after
        // the fn_in push. It runs CONCURRENTLY with the iGPU's router+routed path
        // (which waits on `fn_in_evt`). Result lands in `ds.sh_down` and is folded
        // into the residual after the iGPU handoff. `LAGUNA_SHEXP_DGPU=0` disables.
        if self.shexp_dgpu {
            let dk = &self.dk;
            let ds = &mut self.ds;
            let st = &self.dstream;
            let (gw, uw, dw) = self.dlayers[il].shexp.as_ref().unwrap();
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_gate, gw, &ds.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_up, uw, &ds.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
            dk.swiglu.launch(st, &mut ds.sh_sw, &ds.sh_gate, &ds.sh_up, FF_SHEXP as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_down, dw, &ds.sh_sw, HIDDEN as u32, FF_SHEXP as u32)?;
        }
        let diag_t1 = if self.diag {
            self.dstream.synchronize()?;
            let now = std::time::Instant::now();
            self.diag_dgpu_us += now.duration_since(diag_t0.unwrap()).as_micros() as u64;
            Some(now)
        } else {
            None
        };

        // (2) iGPU: wait for the push, then router + routed MoE + shared expert.
        self.igpu.set_current()?;
        self.istream.wait_event(&self.fn_in_evt)?;
        let moe_scale = self.hp.moe_scale;
        let shexp_dgpu = self.shexp_dgpu;
        {
            let ik = &self.ik;
            let is = &mut self.is;
            let ist = &self.istream;
            let moe = self.ilayers[il].moe.as_ref().unwrap();

            ik.ops.router_split(
                ist, &mut is.sel, &mut is.ew, &mut is.router_probs, &mut is.router_scores,
                &moe.router, &is.fn_in_recv, &moe.bias,
                N_EXPERT as u32, HIDDEN as u32, TOPK as u32, moe_scale, 1e-20,
            )?;
            let n_blk_hidden = (HIDDEN / 256) as u32;
            let n_blk_mid = (FF_EXP / 256) as u32;
            ik.q8k.launch(ist, &mut is.xq_hidden, &is.fn_in_recv, n_blk_hidden)?;
            ik.q4b.launch_pair_swiglu_batched(
                ist, &mut is.mid, &moe.gate_all, &moe.up_all, &is.xq_hidden, &is.ew, &is.sel,
                moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, 0.0, FF_EXP as u32, n_blk_hidden,
            )?;
            ik.q8k.launch(ist, &mut is.xq_mid, &is.mid, (TOPK as u32) * n_blk_mid)?;
            let xq_slot_stride = n_blk_mid * 292;
            match moe.down_dt {
                GgufType::Q6_K => ik.q6b.launch_batched(
                    ist, &mut is.acc, &moe.down_all, &is.xq_mid, &is.sel,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                GgufType::Q4_K => ik.q4b.launch_batched(
                    ist, &mut is.acc, &moe.down_all, &is.xq_mid, &is.sel,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                other => return Err(eyre!("moe down dtype {other:?}")),
            }
            if shexp_dgpu {
                // Shared expert runs on the dGPU (overlapped); iGPU emits routed
                // sum only. The dGPU folds in `ds.sh_down` during the residual.
                is.ffn_out.copy_from_buffer_async(&is.acc, ist)?;
            } else {
                // shared expert (dense SwiGLU) added to the routed sum on the iGPU
                qmatvec(&ik.q4d, &ik.q6d, ist, &mut is.gate_s, &moe.sh_gate, &is.fn_in_recv, FF_SHEXP as u32, HIDDEN as u32)?;
                qmatvec(&ik.q4d, &ik.q6d, ist, &mut is.up_s, &moe.sh_up, &is.fn_in_recv, FF_SHEXP as u32, HIDDEN as u32)?;
                ik.swiglu.launch(ist, &mut is.sw_s, &is.gate_s, &is.up_s, FF_SHEXP as u32)?;
                qmatvec(&ik.q4d, &ik.q6d, ist, &mut is.down_s, &moe.sh_down, &is.sw_s, HIDDEN as u32, FF_SHEXP as u32)?;
                is.ffn_out.copy_from_buffer_async(&is.acc, ist)?;
                ik.vadd.launch(ist, &mut is.ffn_out, &is.down_s, HIDDEN as u32)?;
            }
        }

        // Expert-selection histogram (LAGUNA_EXPERT_HIST=1). Blocking DtoH of the
        // TOPK selection after the routed MoE; perturbs timing, so gate off by
        // default. Only meaningful with real (non-garbage-KV) decode content.
        if self.hist_enabled {
            self.istream.synchronize()?;
            self.is.sel.copy_to_host(&mut self.hist_sel_host)?;
            for &e in &self.hist_sel_host {
                if (0..N_EXPERT as i32).contains(&e) {
                    self.expert_hist[il][e as usize] += 1;
                }
            }
        }

        // (3) peer-push ffn_out iGPU->dGPU on the iGPU (source) stream, record evt.
        peer_push_f32(&self.is.ffn_out, &mut self.ds.moe_recv, &self.istream)?;
        self.moe_evt.record(&self.istream)?;
        if self.diag {
            self.istream.synchronize()?;
            self.diag_igpu_us += std::time::Instant::now()
                .duration_since(diag_t1.unwrap())
                .as_micros() as u64;
        }

        // (4) dGPU: wait for ffn_out, residual h = ffn_out + op. No host sync —
        //     the next layer's attention chains on dstream; forward_* syncs once
        //     at the end of the token.
        self.dgpu.set_current()?;
        self.dstream.wait_event(&self.moe_evt)?;
        {
            let ds = &mut self.ds;
            let st = &self.dstream;
            // DtoD copy_from_buffer is a blocking hipMemcpy; use the async
            // variant so it stays ordered on dstream after the event wait.
            ds.h.copy_from_buffer_async(&ds.moe_recv, st)?;
            self.dk.vadd.launch(st, &mut ds.h, &ds.op, HIDDEN as u32)?;
            if self.shexp_dgpu {
                // fold the concurrently-computed dGPU shared expert into h.
                self.dk.vadd.launch(st, &mut ds.h, &ds.sh_down, HIDDEN as u32)?;
            }
        }
        Ok(())
    }

    /// het-MoE split FFN for one MoE layer. Assumes `layer()` has already run
    /// the dGPU attention front (so `ds.fn_in` / `ds.op` hold ffn_norm / the
    /// ffn residual) and that `self.dlayers[il].hot` is populated. Router + hot
    /// routed experts + shared expert run on `dstream`; cold routed experts run
    /// on `istream`; the partial sums are recombined on the dGPU.
    fn layer_moe_split(&mut self, il: usize, diag_t0: Option<std::time::Instant>) -> eyre::Result<()> {
        let moe_scale = self.hp.moe_scale;
        let n_blk_hidden = (HIDDEN / 256) as u32;
        let n_blk_mid = (FF_EXP / 256) as u32;
        let xq_slot_stride = n_blk_mid * 292;

        // (A) dGPU: router(fn_in) -> sel/ew, then partition into hot/cold slots.
        self.dgpu.set_current()?;
        {
            let dk = &self.dk;
            let ds = &mut self.ds;
            let st = &self.dstream;
            let dlw = &self.dlayers[il];
            let rw = dlw.router.as_ref().unwrap();
            let rb = dlw.router_bias.as_ref().unwrap();
            dk.ops.router_split(
                st, &mut ds.sel, &mut ds.ew, &mut ds.router_probs, &mut ds.router_scores,
                rw, &ds.fn_in, rb, N_EXPERT as u32, HIDDEN as u32, TOPK as u32, moe_scale, 1e-20,
            )?;
            let hot = dlw.hot.as_ref().unwrap();
            self.hetmoe.partition(
                st, &ds.sel, &ds.ew, &hot.hot_map,
                &mut ds.hot_sel, &mut ds.hot_ew, &mut ds.cold_sel, &mut ds.cold_ew, TOPK as u32,
            )?;
        }

        // (B) push fn_in + the COLD selection to the iGPU; record fn_in_evt. The
        //     iGPU runs no router in split mode — it consumes cold_sel/cold_ew.
        peer_push_f32(&self.ds.fn_in, &mut self.is.fn_in_recv, &self.dstream)?;
        peer_push_i32(&self.ds.cold_sel, &mut self.is.cold_sel_recv, &self.dstream)?;
        peer_push_f32(&self.ds.cold_ew, &mut self.is.cold_ew_recv, &self.dstream)?;
        self.fn_in_evt.record(&self.dstream)?;

        // (C) dGPU HOT routed experts + shared expert, on dstream (overlaps the
        //     iGPU cold path which is gated on fn_in_evt).
        {
            let dk = &self.dk;
            let ds = &mut self.ds;
            let st = &self.dstream;
            let dlw = &self.dlayers[il];
            let hot = dlw.hot.as_ref().unwrap();
            dk.q8k.launch(st, &mut ds.xq_hidden, &ds.fn_in, n_blk_hidden)?;
            dk.q4b.launch_pair_swiglu_batched(
                st, &mut ds.mid_hot, &hot.gate_all, &hot.up_all, &ds.xq_hidden, &ds.hot_ew, &ds.hot_sel,
                hot.gate_stride as u32, hot.up_stride as u32, TOPK as u32, 0.0, FF_EXP as u32, n_blk_hidden,
            )?;
            dk.q8k.launch(st, &mut ds.xq_mid, &ds.mid_hot, (TOPK as u32) * n_blk_mid)?;
            match hot.down_dt {
                GgufType::Q6_K => dk.q6b.launch_batched(
                    st, &mut ds.acc_hot, &hot.down_all, &ds.xq_mid, &ds.hot_sel,
                    hot.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                GgufType::Q4_K => dk.q4b.launch_batched(
                    st, &mut ds.acc_hot, &hot.down_all, &ds.xq_mid, &ds.hot_sel,
                    hot.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                other => return Err(eyre!("hot moe down dtype {other:?}")),
            }
            // Shared expert (always dGPU in split mode).
            let (gw, uw, dw) = dlw.shexp.as_ref().unwrap();
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_gate, gw, &ds.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_up, uw, &ds.fn_in, FF_SHEXP as u32, HIDDEN as u32)?;
            dk.swiglu.launch(st, &mut ds.sh_sw, &ds.sh_gate, &ds.sh_up, FF_SHEXP as u32)?;
            qmatvec(&dk.q4d, &dk.q6d, st, &mut ds.sh_down, dw, &ds.sh_sw, HIDDEN as u32, FF_SHEXP as u32)?;
        }
        let diag_t1 = if self.diag {
            self.dstream.synchronize()?;
            let now = std::time::Instant::now();
            self.diag_dgpu_us += now.duration_since(diag_t0.unwrap()).as_micros() as u64;
            Some(now)
        } else {
            None
        };

        // (D) iGPU COLD routed experts (no router; cold sel/ew received).
        self.igpu.set_current()?;
        self.istream.wait_event(&self.fn_in_evt)?;
        {
            let ik = &self.ik;
            let is = &mut self.is;
            let ist = &self.istream;
            let moe = self.ilayers[il].moe.as_ref().unwrap();
            ik.q8k.launch(ist, &mut is.xq_hidden, &is.fn_in_recv, n_blk_hidden)?;
            ik.q4b.launch_pair_swiglu_batched(
                ist, &mut is.mid, &moe.gate_all, &moe.up_all, &is.xq_hidden,
                &is.cold_ew_recv, &is.cold_sel_recv,
                moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, 0.0, FF_EXP as u32, n_blk_hidden,
            )?;
            ik.q8k.launch(ist, &mut is.xq_mid, &is.mid, (TOPK as u32) * n_blk_mid)?;
            match moe.down_dt {
                GgufType::Q6_K => ik.q6b.launch_batched(
                    ist, &mut is.acc, &moe.down_all, &is.xq_mid, &is.cold_sel_recv,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                GgufType::Q4_K => ik.q4b.launch_batched(
                    ist, &mut is.acc, &moe.down_all, &is.xq_mid, &is.cold_sel_recv,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, HIDDEN as u32, n_blk_mid,
                )?,
                other => return Err(eyre!("cold moe down dtype {other:?}")),
            }
            is.ffn_out.copy_from_buffer_async(&is.acc, ist)?;
        }

        // Histogram (uses the full dGPU-side selection). Perturbs timing.
        if self.hist_enabled {
            self.dgpu.set_current()?;
            self.dstream.synchronize()?;
            self.ds.sel.copy_to_host(&mut self.hist_sel_host)?;
            for &e in &self.hist_sel_host {
                if (0..N_EXPERT as i32).contains(&e) {
                    self.expert_hist[il][e as usize] += 1;
                }
            }
            self.igpu.set_current()?;
        }

        // (E) push cold sum back; combine on dGPU: h = op + cold + hot + shexp.
        peer_push_f32(&self.is.ffn_out, &mut self.ds.moe_recv, &self.istream)?;
        self.moe_evt.record(&self.istream)?;
        if self.diag {
            self.istream.synchronize()?;
            self.diag_igpu_us += std::time::Instant::now()
                .duration_since(diag_t1.unwrap())
                .as_micros() as u64;
        }

        self.dgpu.set_current()?;
        self.dstream.wait_event(&self.moe_evt)?;
        {
            let ds = &mut self.ds;
            let st = &self.dstream;
            ds.h.copy_from_buffer_async(&ds.moe_recv, st)?;
            self.dk.vadd.launch(st, &mut ds.h, &ds.op, HIDDEN as u32)?;
            self.dk.vadd.launch(st, &mut ds.h, &ds.acc_hot, HIDDEN as u32)?;
            self.dk.vadd.launch(st, &mut ds.h, &ds.sh_down, HIDDEN as u32)?;
        }
        Ok(())
    }

    pub fn forward_no_logits(&mut self, tok_id: usize, pos: usize) -> eyre::Result<()> {
        self.embed(tok_id)?;
        for il in 0..N_LAYER {
            self.layer(il, pos)?;
        }
        // Drain the token before returning: the next token's `embed` does a
        // blocking H2D into `h`, which must not race with this token's still-
        // pending dstream work (the event-driven layers never blocked the host).
        self.dgpu.set_current()?;
        self.dstream.synchronize()?;
        self.kv_len = pos + 1;
        Ok(())
    }

    /// Forward one token and return the full host logit vector (`VOCAB` long).
    /// The device path is identical to [`Self::forward_logits`]; this variant
    /// hands back all logits so the caller can sample (temperature / top-p)
    /// host-side instead of taking the argmax.
    pub fn forward_logits_full(&mut self, tok_id: usize, pos: usize) -> eyre::Result<Vec<f32>> {
        self.embed(tok_id)?;
        for il in 0..N_LAYER {
            self.layer(il, pos)?;
        }
        self.dgpu.set_current()?;
        let st = &self.dstream;
        self.dk.rms.launch_weighted(st, &mut self.ds.rn, &self.ds.h, &self.output_norm, HIDDEN as u32, EPS)?;
        match self.output_w.dtype {
            GgufType::Q6_K => self.dk.q6d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
            GgufType::Q4_K => self.dk.q4d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
            other => return Err(eyre!("LM head dtype {other:?}")),
        }
        st.synchronize()?;
        self.kv_len = pos + 1;

        let mut logits = vec![0f32; VOCAB];
        self.ds.logits.copy_to_host(&mut logits)?;
        Ok(logits)
    }

    pub fn forward_logits(&mut self, tok_id: usize, pos: usize) -> eyre::Result<(usize, f32)> {
        let logits = self.forward_logits_full(tok_id, pos)?;
        let (argmax, maxv) = logits
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) });
        Ok((argmax, maxv))
    }

    pub fn prefill(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        self.reset();
        let last = tokens.len() - 1;
        for (pos, &tok) in tokens.iter().enumerate() {
            if pos == last {
                return self.forward_logits(tok, pos);
            }
            self.forward_no_logits(tok, pos)?;
        }
        unreachable!()
    }

    pub fn decode_step(&mut self, tok_id: usize, pos: usize) -> eyre::Result<(usize, f32)> {
        self.forward_logits(tok_id, pos)
    }

    // ======================================================================
    // BATCHED PREFILL — process B prompt tokens as one batch through all 48
    // layers, then hand off to the existing decode loop. The dGPU runs the
    // batched attention front-half (proj → qk-norm → rope → GQA prefill →
    // gate → o_proj → residual → ffn_norm), the iGPU runs the by-expert TILED
    // MoE (moe_group_builder → LagunaMoeTiled reg kernels) + shared expert.
    //
    // All per-batch activations are TIGHTLY packed `[B, width]` (row pitch =
    // the layer's actual width, not the SWA-max), so every op indexes with a
    // consistent stride. See `prefill_batched` for the parity contract.
    // ======================================================================

    /// Embed `toks` (host Q4_K dequant of each row) into `dst[0 .. n*HIDDEN]`.
    fn embed_batch(&mut self, toks: &[usize], dst: &mut DeviceBuffer<f32>) -> eyre::Result<()> {
        let mut host = vec![0f32; toks.len() * HIDDEN];
        let mut rb = vec![0u8; self.tok_embd_row_bytes];
        for (bi, &tok) in toks.iter().enumerate() {
            self.raw_file.read_exact_at(
                &mut rb,
                self.tok_embd_off + (tok as u64) * self.tok_embd_row_bytes as u64,
            )?;
            let row = &mut host[bi * HIDDEN..(bi + 1) * HIDDEN];
            for sb in 0..(HIDDEN / 256) {
                dequant_q4k_superblock(&rb[sb * 144..(sb + 1) * 144], &mut row[sb * 256..(sb + 1) * 256]);
            }
        }
        self.dgpu.set_current()?;
        let mut view = dst.slice_view_mut(0, host.len());
        view.copy_from_host(&host)?;
        Ok(())
    }

    /// One transformer layer over a `b`-token batch (single-lane). `q_offset`
    /// is the absolute position of batch row 0 (the KV cache already holds keys
    /// `0..q_offset` from prior tiles). Thin wrapper over the three pipeline
    /// phases, run back-to-back on lane 0 — identical work to the pre-pipeline
    /// monolithic layer, kept as the `LAGUNA_PIPELINE=0` reference path.
    #[allow(clippy::too_many_arguments)]
    fn layer_batched(
        &mut self,
        il: usize,
        q_offset: usize,
        b: usize,
        ps: &mut PrefillScratch,
        tiled: &LagunaMoeTiled,
        gb: &MoeGroupBuilder,
    ) -> eyre::Result<()> {
        if self.attn_batched(il, q_offset, b, ps, 0)? {
            return Ok(()); // dense layer-0 fully handled on the dGPU
        }
        self.moe_batched(il, b, ps, tiled, gb, 0)?;
        self.combine_batched(il, b, ps, 0)?;
        Ok(())
    }

    /// PIPELINE PHASE 1 — dGPU attention front-half for one batch lane.
    /// norm→qkv→qk-norm→rope→flash-attn→gate→o_proj→residual→ffn_norm into
    /// `ps.op` (ffn_inp) + `ps.fn_in` (ffn_norm); appends this lane's K/V to the
    /// shared cache. Returns `Ok(true)` for the dense layer-0 (FFN done on the
    /// dGPU, `ps.h` updated, no MoE). Otherwise peer-pushes `fn_in` to the iGPU,
    /// records `pipe_fn_in_evt[lane]`, and returns `Ok(false)` — the iGPU MoE
    /// for `lane` is now unblocked.
    #[allow(clippy::too_many_arguments)]
    fn attn_batched(
        &mut self,
        il: usize,
        q_offset: usize,
        b: usize,
        ps: &mut PrefillScratch,
        lane: usize,
    ) -> eyre::Result<bool> {
        let n_head = self.dlayers[il].n_head;
        let is_full = self.dlayers[il].is_full;
        let is_dense = self.dlayers[il].dense.is_some();
        let n_embd_q = n_head * HEAD_DIM;
        let kv_stride = N_KV_HEAD * HEAD_DIM;
        let n_rot = if is_full { self.hp.n_rot_full as u32 } else { self.hp.n_rot_swa as u32 };
        let scale = 1.0 / (HEAD_DIM as f32).sqrt();
        let bu = b as u32;

        // =============== dGPU: batched attention -> op, fn_in ===============
        self.dgpu.set_current()?;
        {
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let st = &self.dstream;
            let rope = if is_full { &self.rope_full } else { &self.rope_swa };
            let cap = self.kv_cap[il]; // KV ring capacity for this layer
            // Ring-wrap correctness: within a left-to-right chunk, an SWA query reads
            // keys [q-511, q]; the whole chunk's read span is [q_offset-511,
            // q_offset+b-1] ≤ b + SWA_WINDOW positions. Every one must still be
            // physically resident (not overwritten by a later position), which holds
            // iff that span ≤ cap. When cap == max_kv the ring never wraps (whole
            // context resident), so the check only bites when the ring is active.
            debug_assert!(
                cap == self.max_kv || b + SWA_WINDOW <= cap,
                "SWA ring too small: b={b} + SWA_WINDOW={SWA_WINDOW} > cap={cap} (il={il}); \
                 raise SWA_RING_CAP or lower LAGUNA_PREFILL_BMAX"
            );

            dk.rms.launch_weighted_batched(st, &mut ps.ain, &ps.h, &dlw.attn_norm, HIDDEN as u32, EPS, bu)?;
            // WMMA GEMM (each f16 weight read once) replaces matvec_batched
            // (weight-BW-bound, re-reads weights per batch row). K=HIDDEN=3072
            // is a multiple of BK=32; n_rows/batch partial tiles are bounds-guarded.
            dk.f16.gemm_batched_wmma(st, &mut ps.q, &dlw.wq, &ps.ain, n_embd_q as u32, HIDDEN as u32, bu)?;
            dk.f16.gemm_batched_wmma(st, &mut ps.k, &dlw.wk, &ps.ain, kv_stride as u32, HIDDEN as u32, bu)?;
            dk.f16.gemm_batched_wmma(st, &mut ps.v, &dlw.wv, &ps.ain, kv_stride as u32, HIDDEN as u32, bu)?;
            // per-head QK-norm over ALL B rows: contiguous [B*n_head, head_dim].
            dk.ops.qk_rmsnorm(st, &mut ps.qn, &ps.q, &dlw.q_norm, (b * n_head) as u32, HEAD_DIM as u32, EPS)?;
            dk.ops.qk_rmsnorm(st, &mut ps.kn, &ps.k, &dlw.k_norm, (b * N_KV_HEAD) as u32, HEAD_DIM as u32, EPS)?;
            dk.rope.launch_forward_batched(st, &mut ps.qn, &ps.pos, n_head as u32, HEAD_DIM as u32, n_rot, bu, rope)?;
            dk.rope.launch_forward_batched(st, &mut ps.kn, &ps.pos, N_KV_HEAD as u32, HEAD_DIM as u32, n_rot, bu, rope)?;
            // Append the whole batch's K/V into ring slots for absolute positions
            // [q_offset .. q_offset+b). Physical start row = q_offset % cap. When the
            // batch straddles the ring boundary (phys0 + b > cap) the write splits
            // into two contiguous cast_f16 calls (tail then wrap). For global layers
            // cap==max_kv and q_offset+b<=max_kv, so `first==b` — a single write,
            // byte-identical to the pre-ring path.
            {
                let phys0 = q_offset % cap;
                let first = (cap - phys0).min(b); // rows written before the wrap
                let mut kslot = self.kc[il].slice_view_mut(phys0 * kv_stride, first * kv_stride);
                let kn_v = ps.kn.slice_view(0, first * kv_stride);
                dk.ops.cast_f16(st, &mut kslot, &kn_v, (first * kv_stride) as u32)?;
                let mut vslot = self.vc[il].slice_view_mut(phys0 * kv_stride, first * kv_stride);
                let v_v = ps.v.slice_view(0, first * kv_stride);
                dk.ops.cast_f16(st, &mut vslot, &v_v, (first * kv_stride) as u32)?;
                if first < b {
                    let rem = b - first; // wrapped rows at physical row 0
                    let mut kslot2 = self.kc[il].slice_view_mut(0, rem * kv_stride);
                    let kn_v2 = ps.kn.slice_view(first * kv_stride, rem * kv_stride);
                    dk.ops.cast_f16(st, &mut kslot2, &kn_v2, (rem * kv_stride) as u32)?;
                    let mut vslot2 = self.vc[il].slice_view_mut(0, rem * kv_stride);
                    let v_v2 = ps.v.slice_view(first * kv_stride, rem * kv_stride);
                    dk.ops.cast_f16(st, &mut vslot2, &v_v2, (rem * kv_stride) as u32)?;
                }
            }
            {
                let qn_v = ps.qn.slice_view(0, b * n_embd_q);
                let mut qf_v = ps.qf.slice_view_mut(0, b * n_embd_q);
                dk.ops.cast_f16(st, &mut qf_v, &qn_v, (b * n_embd_q) as u32)?;
            }
            {
                let qf_v = ps.qf.slice_view(0, b * n_embd_q);
                let n_kv_total = q_offset + b;
                // Pass the WHOLE physical ring buffer (cap rows). The kernel bounds
                // keys by the logical n_kv_total and maps absolute key -> physical
                // key % cap. For global layers cap==max_kv >= n_kv_total, so this is
                // the same memory the pre-ring path read (byte-identical).
                let k_v = self.kc[il].slice_view(0, cap * kv_stride);
                let v_v = self.vc[il].slice_view(0, cap * kv_stride);
                let mut od_v = ps.od.slice_view_mut(0, b * n_embd_q);
                // Flash-tiled prefill attention (K/V-reuse, one barrier per key
                // tile) by default; LAGUNA_ATTN_NAIVE=1 restores the naive
                // per-query kernel for A/B benchmarking.
                // Prefill attention on the dGPU (gfx1201, real f16 matrix core):
                // WMMA score+AV wins at LONG context (O(L²) global attn dominates the
                // dGPU there — measured 468.7 vs 355 tok/s @4K, +32%) but REGRESSES short
                // context (its LDS-staging overhead inflates the dGPU leg past the
                // iGPU-MoE regime — 279 vs 465 tok/s @512, -40%). Gate on sequence length.
                // Envs (A/B): LAGUNA_ATTN_NAIVE=1 naive; LAGUNA_ATTN_WMMA=1 force WMMA;
                // LAGUNA_ATTN_FLASH=1 force scalar-ILP flash. Threshold is a conservative
                // guess between the measured 512/4096 crossover — TODO tune at 1K/2K/3K.
                const PREFILL_ATTN_WMMA_MIN_KV: usize = 2048;
                // SLIDING-WINDOW ATTENTION: SWA layers (!is_full) attend only the
                // previous SWA_WINDOW=512 keys per query row (spec sliding_window=512,
                // LLAMA_SWA_TYPE_STANDARD). 0 = full causal (global layers). Passing
                // 0 keeps global layers byte-identical to the pre-window behavior.
                const SWA_WINDOW: u32 = 512;
                let swa_off = std::env::var("LAGUNA_SWA_OFF").as_deref() == Ok("1");
                let swa = if is_full || swa_off { 0u32 } else { SWA_WINDOW };
                let force_wmma = std::env::var("LAGUNA_ATTN_WMMA").is_ok();
                let force_flash = std::env::var("LAGUNA_ATTN_FLASH").is_ok();
                if std::env::var("LAGUNA_ATTN_NAIVE").is_ok() {
                    dk.gqa.prefill(
                        st, &mut od_v, &qf_v, &k_v, &v_v,
                        bu, n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, q_offset as u32, scale, swa, cap as u32,
                    )?;
                } else if force_wmma || (!force_flash && n_kv_total >= PREFILL_ATTN_WMMA_MIN_KV) {
                    // Default WMMA path is the FA2 register-resident-O kernel (≥2 WG/CU,
                    // flattens the O(L²) global-attn falloff). LAGUNA_ATTN_WMMA_LEGACY=1
                    // restores the LDS-Os kernel for A/B.
                    if std::env::var("LAGUNA_ATTN_WMMA_LEGACY").is_ok() {
                        dk.gqa.prefill_flash_wmma(
                            st, &mut od_v, &qf_v, &k_v, &v_v,
                            bu, n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, q_offset as u32, scale, swa, cap as u32,
                        )?;
                    } else if is_full
                        && std::env::var("LAGUNA_ATTN_HG").as_deref() != Ok("0")
                    {
                        // Head-grouped WMMA prefill is the DEFAULT on the O(L²) global
                        // layers (kv_group=6 divisible by HG_G=3): block=256 (8 waves)
                        // stages each K/V key-tile into LDS once and runs all 3 grouped
                        // query heads against it, amortising barriers with no wave-
                        // occupancy loss — +13.6% e2e @100K, parity-exact. SWA layers
                        // read only 512 keys so grouping there buys nothing — fa2.
                        // LAGUNA_ATTN_HG=0 restores fa2 on global layers for A/B.
                        dk.gqa.prefill_flash_wmma_fa2_hg(
                            st, &mut od_v, &qf_v, &k_v, &v_v,
                            bu, n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, q_offset as u32, scale, swa, cap as u32,
                        )?;
                    } else {
                        // KV-first grid remap (Infinity-Cache locality) for the O(L²)
                        // global layers; A/B via LAGUNA_ATTN_KVFIRST=1. SWA layers read
                        // only 512 keys so their K/V is already tiny — skip the remap.
                        let kv_first = is_full
                            && std::env::var("LAGUNA_ATTN_KVFIRST").as_deref() == Ok("1");
                        dk.gqa.prefill_flash_wmma_fa2(
                            st, &mut od_v, &qf_v, &k_v, &v_v,
                            bu, n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, q_offset as u32, scale, swa, cap as u32,
                            kv_first,
                        )?;
                    }
                } else {
                    dk.gqa.prefill_flash(
                        st, &mut od_v, &qf_v, &k_v, &v_v,
                        bu, n_head as u32, N_KV_HEAD as u32, HEAD_DIM as u32, q_offset as u32, scale, swa, cap as u32,
                    )?;
                }
            }
            // softplus gate over [B*n_head] logits, applied to od[B*n_head, head_dim].
            dk.f16.gemm_batched_wmma(st, &mut ps.gate_logits, &dlw.wg, &ps.ain, n_head as u32, HIDDEN as u32, bu)?;
            dk.ops.softplus_gate(st, &mut ps.od, &ps.gate_logits, (b * n_head) as u32, HEAD_DIM as u32)?;
            // o-proj: K=n_embd_q (6144/9216) also a multiple of 32.
            dk.f16.gemm_batched_wmma(st, &mut ps.op, &dlw.wo, &ps.od, HIDDEN as u32, n_embd_q as u32, bu)?;
            dk.vadd.launch(st, &mut ps.op, &ps.h, (b * HIDDEN) as u32)?; // op = o_proj + h (ffn_inp)
            dk.rms.launch_weighted_batched(st, &mut ps.fn_in, &ps.op, &dlw.ffn_norm, HIDDEN as u32, EPS, bu)?;
        }

        // ===================== FFN =====================
        if is_dense {
            // Layer 0 dense FFN — batched, entirely on the dGPU.
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let st = &self.dstream;
            let (gw, uw, dw) = dlw.dense.as_ref().unwrap();
            qmatvec_batched(&dk.q4d, &dk.q6d, st, &mut ps.gate_big, gw, &ps.fn_in, FF_DENSE as u32, HIDDEN as u32, bu)?;
            qmatvec_batched(&dk.q4d, &dk.q6d, st, &mut ps.up_big, uw, &ps.fn_in, FF_DENSE as u32, HIDDEN as u32, bu)?;
            dk.swiglu.launch(st, &mut ps.sw_big, &ps.gate_big, &ps.up_big, (b * FF_DENSE) as u32)?;
            qmatvec_batched(&dk.q4d, &dk.q6d, st, &mut ps.ffn_out, dw, &ps.sw_big, HIDDEN as u32, FF_DENSE as u32, bu)?;
            ps.h.copy_from_buffer_async(&ps.ffn_out, st)?;
            dk.vadd.launch(st, &mut ps.h, &ps.op, (b * HIDDEN) as u32)?;
            return Ok(true);
        }

        // Non-dense: hand this lane's fn_in[b,HIDDEN] to the iGPU and record the
        // per-lane handoff event. The MoE itself runs in `moe_batched`, so a
        // second lane's attention can be enqueued on the dGPU stream while this
        // lane's MoE runs on the iGPU stream.
        {
            let fn_in_v = ps.fn_in.slice_view(0, b * HIDDEN);
            let mut recv_v = ps.fn_in_recv.slice_view_mut(0, b * HIDDEN);
            peer_push_f32(&fn_in_v, &mut recv_v, &self.dstream)?;
        }
        self.pipe_fn_in_evt[lane].record(&self.dstream)?;
        Ok(false)
    }

    /// PIPELINE PHASE 2 — iGPU router + routed MoE + shared expert for one lane.
    /// Waits `pipe_fn_in_evt[lane]` (the dGPU→iGPU fn_in push), computes
    /// `ps.ffn_out_i`, peer-pushes it back to the dGPU (`ps.moe_recv`), and
    /// records `pipe_moe_evt[lane]`.
    #[allow(clippy::too_many_arguments)]
    fn moe_batched(
        &mut self,
        il: usize,
        b: usize,
        ps: &mut PrefillScratch,
        tiled: &LagunaMoeTiled,
        gb: &MoeGroupBuilder,
        lane: usize,
    ) -> eyre::Result<()> {
        let bu = b as u32;
        self.igpu.set_current()?;
        self.istream.wait_event(&self.pipe_fn_in_evt[lane])?;
        let moe_scale = self.hp.moe_scale;
        {
            let ik = &self.ik;
            let ist = &self.istream;
            let moe = self.ilayers[il].moe.as_ref().unwrap();
            let n_blk_hidden = (HIDDEN / 256) as u32;
            let n_blk_mid = (FF_EXP / 256) as u32;
            let max_per_expert = bu; // each expert picked ≤ once per token

            // Batched router (grid.z = B): sel[B,TOPK] + ew[B,TOPK], the exact
            // layout the group builder / tiled MoE want. Two launches, not 2*B.
            ik.ops.router_split_batched(
                ist, &mut ps.sel, &mut ps.ew, &mut ps.router_probs, &mut ps.router_scores,
                &moe.router, &ps.fn_in_recv, &moe.bias,
                N_EXPERT as u32, HIDDEN as u32, TOPK as u32, moe_scale, 1e-20, bu,
            )?;

            // Q8_K of the whole batch's fn_in: [B, n_blk_hidden] blocks.
            ik.q8k.launch(ist, &mut ps.xq_hidden, &ps.fn_in_recv, bu * n_blk_hidden)?;

            // by-expert groups.
            ps.group_count.fill_zero_async(ist)?;
            gb.launch(
                ist, &mut ps.group_count, &mut ps.members, &ps.sel,
                bu, TOPK as u32, N_EXPERT as u32, max_per_expert,
            )?;

            // reg-tiled gate×up×swiglu×ew -> mid[B, n_used, FF_EXP].
            // LAGUNA_GATEUP: "reg" (default, register-tiled) | "plain" (non-reg,
            // occ-16 but re-decodes per member) — A/B occupancy vs instrs.
            // LAGUNA_GATEUP: "col" (default, WIN #4 column-tiled: NT_COL members
            // staged per barrier) | "reg" (register-tiled, one barrier/member) |
            // "plain" (non-reg, re-decode per member).
            match std::env::var("LAGUNA_GATEUP").as_deref() {
                Ok("plain") => tiled.gate_up_swiglu(
                    ist, &mut ps.mid, &moe.gate_all, &moe.up_all, &ps.xq_hidden, &ps.ew,
                    &ps.group_count, &ps.members,
                    moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, max_per_expert,
                    0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
                )?,
                Ok("reg") => tiled.gate_up_swiglu_reg(
                    ist, &mut ps.mid, &moe.gate_all, &moe.up_all, &ps.xq_hidden, &ps.ew,
                    &ps.group_count, &ps.members,
                    moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, max_per_expert,
                    0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
                )?,
                Ok("col") => tiled.gate_up_swiglu_reg_col(
                    ist, &mut ps.mid, &moe.gate_all, &moe.up_all, &ps.xq_hidden, &ps.ew,
                    &ps.group_count, &ps.members,
                    moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, max_per_expert,
                    0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
                )?,
                // default: WIN #2 wide-row (32 rows/WG) column-tiled
                _ => tiled.gate_up_swiglu_reg_col_r32(
                    ist, &mut ps.mid, &moe.gate_all, &moe.up_all, &ps.xq_hidden, &ps.ew,
                    &ps.group_count, &ps.members,
                    moe.gate_stride as u32, moe.up_stride as u32, TOPK as u32, max_per_expert,
                    0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
                )?,
            }
            ik.q8k.launch(ist, &mut ps.xq_mid, &ps.mid, bu * TOPK as u32 * n_blk_mid)?;
            let xq_slot_stride = n_blk_mid * 292;
            // LAGUNA_DOWN_PART (default ON): atomic-free down — plain-store each
            // (b,slot) member to a unique partial slice, then streaming-sum the
            // TOPK slots into acc. Removes the cross-expert global-atomic
            // contention that was ~23-35% of the down kernel. "0" restores the
            // atomic-accumulate path.
            let use_part = std::env::var("LAGUNA_DOWN_PART").map(|v| v != "0").unwrap_or(true);
            if use_part && matches!(moe.down_dt, GgufType::Q6_K | GgufType::Q4_K) {
                tiled.down_part(
                    ist, moe.down_dt, &mut ps.down_part, &moe.down_all, &ps.xq_mid,
                    &ps.group_count, &ps.members, moe.down_stride as u32, xq_slot_stride,
                    TOPK as u32, max_per_expert, HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?;
                tiled.down_reduce_slots(
                    ist, &mut ps.acc, &ps.down_part,
                    HIDDEN as u32, TOPK as u32, bu * HIDDEN as u32,
                )?;
            } else {
            ps.acc.fill_zero_async(ist)?;
            // LAGUNA_DOWN: "w32" (default, full-warp win #2, +17% e2e) |
            // "reg"/"plain" (old half-warp variants) | "nodot" (floor, no parity).
            let down_variant = std::env::var("LAGUNA_DOWN").unwrap_or_default();
            match (moe.down_dt, down_variant.as_str()) {
                (GgufType::Q6_K, "reg") => tiled.down_reg_q6k(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (GgufType::Q6_K, "nodot") => tiled.down_reg_q6k_nodot(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (GgufType::Q6_K, "plain") => tiled.down(
                    ist, GgufType::Q6_K, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (GgufType::Q6_K, "w32") => tiled.down_reg_q6k_w32(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                // default: WIN #4 column-tiled full-warp
                (GgufType::Q6_K, _) => tiled.down_reg_q6k_w32_col(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (GgufType::Q4_K, "plain") | (GgufType::Q4_K, "reg") => tiled.down(
                    ist, GgufType::Q4_K, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (GgufType::Q4_K, "w32") => tiled.down_q4k_w32(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                // default: WIN #4 column-tiled full-warp (reg-held weight + NT_COL staged)
                (GgufType::Q4_K, _) => tiled.down_reg_q4k_w32_col(
                    ist, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
                (dt, _) => tiled.down(
                    ist, dt, &mut ps.acc, &moe.down_all, &ps.xq_mid, &ps.group_count, &ps.members,
                    moe.down_stride as u32, xq_slot_stride, TOPK as u32, max_per_expert,
                    HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
                )?,
            }
            }

            // shared expert (batched dense SwiGLU) added to the routed sum.
            // LAGUNA_SHEXP: "dp4a" (default, WIN #1 weight-read-once dp4a GEMM —
            // gate/up reuse xq_hidden = Q8_K(fn_in_recv), down needs Q8_K(sw_s))
            // | "old" (BxN dense gemv, weight re-read per token).
            match std::env::var("LAGUNA_SHEXP").as_deref() {
                Ok("old") => {
                    qmatvec_batched(&ik.q4d, &ik.q6d, ist, &mut ps.gate_s, &moe.sh_gate, &ps.fn_in_recv, FF_SHEXP as u32, HIDDEN as u32, bu)?;
                    qmatvec_batched(&ik.q4d, &ik.q6d, ist, &mut ps.up_s, &moe.sh_up, &ps.fn_in_recv, FF_SHEXP as u32, HIDDEN as u32, bu)?;
                    ik.swiglu.launch(ist, &mut ps.sw_s, &ps.gate_s, &ps.up_s, (b * FF_SHEXP) as u32)?;
                    qmatvec_batched(&ik.q4d, &ik.q6d, ist, &mut ps.down_s, &moe.sh_down, &ps.sw_s, HIDDEN as u32, FF_SHEXP as u32, bu)?;
                }
                _ => {
                    // gate/up: input is fn_in_recv, already Q8_K'd into xq_hidden above.
                    tiled.dense_gemm_dp4a(ist, moe.sh_gate.dtype, &mut ps.gate_s, &moe.sh_gate.bytes, &ps.xq_hidden, bu, FF_SHEXP as u32, n_blk_hidden)?;
                    tiled.dense_gemm_dp4a(ist, moe.sh_up.dtype, &mut ps.up_s, &moe.sh_up.bytes, &ps.xq_hidden, bu, FF_SHEXP as u32, n_blk_hidden)?;
                    ik.swiglu.launch(ist, &mut ps.sw_s, &ps.gate_s, &ps.up_s, (b * FF_SHEXP) as u32)?;
                    // down: input is sw_s (FF_SHEXP wide) — Q8_K it (FF_SHEXP/256 = n_blk_mid blocks).
                    ik.q8k.launch(ist, &mut ps.xq_sw, &ps.sw_s, bu * n_blk_mid)?;
                    tiled.dense_gemm_dp4a(ist, moe.sh_down.dtype, &mut ps.down_s, &moe.sh_down.bytes, &ps.xq_sw, bu, HIDDEN as u32, n_blk_mid)?;
                }
            }
            ps.ffn_out_i.copy_from_buffer_async(&ps.acc, ist)?;
            ik.vadd.launch(ist, &mut ps.ffn_out_i, &ps.down_s, (b * HIDDEN) as u32)?;
        }

        // Push this lane's ffn_out iGPU -> dGPU; record the per-lane event.
        {
            let out_v = ps.ffn_out_i.slice_view(0, b * HIDDEN);
            let mut recv_v = ps.moe_recv.slice_view_mut(0, b * HIDDEN);
            peer_push_f32(&out_v, &mut recv_v, &self.istream)?;
        }
        self.pipe_moe_evt[lane].record(&self.istream)?;
        Ok(())
    }

    /// PIPELINE PHASE 3 — dGPU residual `h = ffn_out + op` for one lane. Waits
    /// `pipe_moe_evt[lane]` so this lane's MoE result has landed on the dGPU.
    fn combine_batched(
        &mut self,
        _il: usize,
        b: usize,
        ps: &mut PrefillScratch,
        lane: usize,
    ) -> eyre::Result<()> {
        self.dgpu.set_current()?;
        self.dstream.wait_event(&self.pipe_moe_evt[lane])?;
        {
            let dk = &self.dk;
            let st = &self.dstream;
            ps.h.copy_from_buffer_async(&ps.moe_recv, st)?;
            dk.vadd.launch(st, &mut ps.h, &ps.op, (b * HIDDEN) as u32)?;
        }
        Ok(())
    }

    // ======================================================================
    // BATCHED PREFILL HET-MoE SPLIT — the K globally-hottest routed experts
    // run on the dGPU (which is otherwise idle during the iGPU MoE window),
    // CONCURRENTLY with the COLD experts on the iGPU. Router runs on the dGPU
    // (fn_in is already dGPU-resident); the two partial routed sums + the iGPU
    // shared expert are recombined by addition on the dGPU. This spans the MoE
    // itself across both devices (on top of the attention∥MoE pipeline).
    //
    // Reuses the DECODE het residency (`DgpuLayer.hot`, `LAGUNA_HOT_EXPERTS_DGPU`)
    // and the generic residency-aware group builder
    // (`MoeGroupBuilder::launch_hetsplit`): mode=1 yields dense-local hot groups
    // (grid.y = K, K-wide packed weights), mode=0 yields the original-id cold
    // groups (grid.y = N_EXPERT). No dedicated batched partition kernel needed.
    // ======================================================================

    /// het-split MoE for one batch lane. Router + hot experts (+ q8k) run on the
    /// dGPU (`dstream`); cold experts + shared expert run on the iGPU (`istream`).
    /// `attn_batched` already pushed `fn_in` to the iGPU and recorded
    /// `pipe_fn_in_evt[lane]`; here we push the freshly-routed `sel`/`ew` and
    /// re-record that event so the iGPU cold path also waits on them. The dGPU
    /// hot partial lands in `hs.acc`; the iGPU cold+shexp sum lands in
    /// `ps.moe_recv` after `pipe_moe_evt[lane]`.
    #[allow(clippy::too_many_arguments)]
    fn moe_batched_split(
        &mut self,
        il: usize,
        b: usize,
        ps: &mut PrefillScratch,
        hs: &mut HotScratch,
        tiled_d: &LagunaMoeTiled,
        gb_d: &MoeGroupBuilder,
        tiled_i: &LagunaMoeTiled,
        gb_i: &MoeGroupBuilder,
        lane: usize,
    ) -> eyre::Result<()> {
        let bu = b as u32;
        let moe_scale = self.hp.moe_scale;
        let n_blk_hidden = (HIDDEN / 256) as u32;
        let n_blk_mid = (FF_EXP / 256) as u32;
        let xq_slot_stride = n_blk_mid * 292;
        let cap = self.prefill_hot_cap;
        let max_per_expert = bu; // each expert picked ≤ once per token

        // (A) dGPU: router(fn_in) -> sel/ew on dstream (fn_in already resident).
        self.dgpu.set_current()?;
        {
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let st = &self.dstream;
            let rw = dlw.router.as_ref().unwrap();
            let rb = dlw.router_bias.as_ref().unwrap();
            dk.ops.router_split_batched(
                st, &mut hs.sel, &mut hs.ew, &mut hs.router_probs, &mut hs.router_scores,
                rw, &ps.fn_in, rb, N_EXPERT as u32, HIDDEN as u32, TOPK as u32, moe_scale, 1e-20, bu,
            )?;
        }

        // (B) push the routed sel/ew to the iGPU (fn_in was already pushed by
        //     attn_batched); re-record fn_in_evt so the cold path waits on these.
        {
            let sel_v = hs.sel.slice_view(0, b * TOPK);
            let mut sel_recv = ps.sel.slice_view_mut(0, b * TOPK);
            peer_push_i32(&sel_v, &mut sel_recv, &self.dstream)?;
            let ew_v = hs.ew.slice_view(0, b * TOPK);
            let mut ew_recv = ps.ew.slice_view_mut(0, b * TOPK);
            peer_push_f32(&ew_v, &mut ew_recv, &self.dstream)?;
        }
        self.pipe_fn_in_evt[lane].record(&self.dstream)?;

        // (C) dGPU HOT path on dstream (overlaps the iGPU cold path).
        {
            let dk = &self.dk;
            let dlw = &self.dlayers[il];
            let hot = dlw.hot.as_ref().unwrap();
            let st = &self.dstream;
            let n_hot = hot.n_hot as u32;
            dk.q8k.launch(st, &mut hs.xq_hidden, &ps.fn_in, bu * n_blk_hidden)?;
            hs.group_count.fill_zero_async(st)?; // K_MAX ints — tiny
            gb_d.launch_hetsplit(
                st, &mut hs.group_count, &mut hs.members, &hs.sel, &hot.hot_map,
                1, cap, bu, TOPK as u32, n_hot, max_per_expert,
            )?;
            tiled_d.gate_up_swiglu_reg_col_r32(
                st, &mut hs.mid, &hot.gate_all, &hot.up_all, &hs.xq_hidden, &hs.ew,
                &hs.group_count, &hs.members, hot.gate_stride as u32, hot.up_stride as u32,
                TOPK as u32, max_per_expert, 0.0, FF_EXP as u32, n_blk_hidden, n_hot,
            )?;
            dk.q8k.launch(st, &mut hs.xq_mid, &hs.mid, bu * TOPK as u32 * n_blk_mid)?;
            // In the split, only this token's HOT slots get a down_part member;
            // the cold slots stay unwritten. down_reduce_slots sums ALL TOPK
            // slots, so zero the unwritten ones first (the non-split path writes
            // every slot, hence needs no zeroing).
            {
                let mut dp = hs.down_part.slice_view_mut(0, b * TOPK * HIDDEN);
                dp.fill_zero_async(st)?;
            }
            tiled_d.down_part(
                st, hot.down_dt, &mut hs.down_part, &hot.down_all, &hs.xq_mid,
                &hs.group_count, &hs.members, hot.down_stride as u32, xq_slot_stride,
                TOPK as u32, max_per_expert, HIDDEN as u32, n_blk_mid, n_hot,
            )?;
            tiled_d.down_reduce_slots(
                st, &mut hs.acc, &hs.down_part, HIDDEN as u32, TOPK as u32, bu * HIDDEN as u32,
            )?;
        }

        // (D) iGPU COLD path on istream (waits fn_in_evt). Router NOT re-run.
        self.igpu.set_current()?;
        self.istream.wait_event(&self.pipe_fn_in_evt[lane])?;
        {
            let ik = &self.ik;
            let ist = &self.istream;
            let moe = self.ilayers[il].moe.as_ref().unwrap();
            let hot_map_i = self.ilayers[il].hot_map.as_ref().unwrap();
            ik.q8k.launch(ist, &mut ps.xq_hidden, &ps.fn_in_recv, bu * n_blk_hidden)?;
            ps.group_count.fill_zero_async(ist)?;
            gb_i.launch_hetsplit(
                ist, &mut ps.group_count, &mut ps.members, &ps.sel, hot_map_i,
                0, cap, bu, TOPK as u32, N_EXPERT as u32, max_per_expert,
            )?;
            tiled_i.gate_up_swiglu_reg_col_r32(
                ist, &mut ps.mid, &moe.gate_all, &moe.up_all, &ps.xq_hidden, &ps.ew,
                &ps.group_count, &ps.members, moe.gate_stride as u32, moe.up_stride as u32,
                TOPK as u32, max_per_expert, 0.0, FF_EXP as u32, n_blk_hidden, N_EXPERT as u32,
            )?;
            ik.q8k.launch(ist, &mut ps.xq_mid, &ps.mid, bu * TOPK as u32 * n_blk_mid)?;
            // Split: only COLD slots get written here — zero the rest so the
            // TOPK-slot reduce doesn't fold in garbage (see the hot path).
            {
                let mut dp = ps.down_part.slice_view_mut(0, b * TOPK * HIDDEN);
                dp.fill_zero_async(ist)?;
            }
            tiled_i.down_part(
                ist, moe.down_dt, &mut ps.down_part, &moe.down_all, &ps.xq_mid,
                &ps.group_count, &ps.members, moe.down_stride as u32, xq_slot_stride,
                TOPK as u32, max_per_expert, HIDDEN as u32, n_blk_mid, N_EXPERT as u32,
            )?;
            tiled_i.down_reduce_slots(
                ist, &mut ps.acc, &ps.down_part, HIDDEN as u32, TOPK as u32, bu * HIDDEN as u32,
            )?;
            // shared expert (iGPU, read-once dp4a) folded into the cold sum.
            tiled_i.dense_gemm_dp4a(ist, moe.sh_gate.dtype, &mut ps.gate_s, &moe.sh_gate.bytes, &ps.xq_hidden, bu, FF_SHEXP as u32, n_blk_hidden)?;
            tiled_i.dense_gemm_dp4a(ist, moe.sh_up.dtype, &mut ps.up_s, &moe.sh_up.bytes, &ps.xq_hidden, bu, FF_SHEXP as u32, n_blk_hidden)?;
            ik.swiglu.launch(ist, &mut ps.sw_s, &ps.gate_s, &ps.up_s, (b * FF_SHEXP) as u32)?;
            ik.q8k.launch(ist, &mut ps.xq_sw, &ps.sw_s, bu * n_blk_mid)?;
            tiled_i.dense_gemm_dp4a(ist, moe.sh_down.dtype, &mut ps.down_s, &moe.sh_down.bytes, &ps.xq_sw, bu, HIDDEN as u32, n_blk_mid)?;
            ps.ffn_out_i.copy_from_buffer_async(&ps.acc, ist)?;
            ik.vadd.launch(ist, &mut ps.ffn_out_i, &ps.down_s, (b * HIDDEN) as u32)?;
        }

        // Push the cold+shexp sum iGPU -> dGPU; record the per-lane event.
        {
            let out_v = ps.ffn_out_i.slice_view(0, b * HIDDEN);
            let mut recv_v = ps.moe_recv.slice_view_mut(0, b * HIDDEN);
            peer_push_f32(&out_v, &mut recv_v, &self.istream)?;
        }
        self.pipe_moe_evt[lane].record(&self.istream)?;
        Ok(())
    }

    /// het-split residual: `h = op + (cold+shexp) + hot`. Waits `pipe_moe_evt`
    /// (cold sum landed on the dGPU); the hot partial `hs.acc` is already on
    /// `dstream` (FIFO-ordered after step C).
    fn combine_batched_split(
        &mut self,
        _il: usize,
        b: usize,
        ps: &mut PrefillScratch,
        hs: &mut HotScratch,
        lane: usize,
    ) -> eyre::Result<()> {
        self.dgpu.set_current()?;
        self.dstream.wait_event(&self.pipe_moe_evt[lane])?;
        {
            let dk = &self.dk;
            let st = &self.dstream;
            ps.h.copy_from_buffer_async(&ps.moe_recv, st)?;        // cold + shexp
            dk.vadd.launch(st, &mut ps.h, &ps.op, (b * HIDDEN) as u32)?;   // + residual
            dk.vadd.launch(st, &mut ps.h, &hs.acc, (b * HIDDEN) as u32)?;  // + hot partial
        }
        Ok(())
    }

    /// BATCHED prefill: process `tokens` in B_MAX-sized tiles, building the KV
    /// cache across the whole prompt, then return the greedy next token after
    /// the last prompt token. Logits/greedy token match the sequential
    /// [`prefill`] (attention + routing are numerically identical; the MoE GEMM
    /// differs only by atomic-accumulation reorder ~1e-4, greedy-stable).
    ///
    /// After this the model is positioned exactly as after `prefill`, so the
    /// caller drives generation with the existing [`decode_step`].
    pub fn prefill_batched(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        if tokens.is_empty() {
            return Err(eyre!("prefill_batched: empty prompt"));
        }
        // Prefill het-MoE split (hot experts on the dGPU) when the decode hot
        // residency is loaded (`LAGUNA_HOT_EXPERTS_DGPU=<file>`) AND
        // `LAGUNA_PREFILL_HET=1`. Single-lane, env-gated; falls through to the
        // pure-iGPU pipeline otherwise. Keeps a pure-iGPU fallback intact.
        let prefill_het = std::env::var("LAGUNA_PREFILL_HET").map(|v| v == "1").unwrap_or(false);
        if prefill_het && self.hot_split {
            return self.prefill_batched_het(tokens);
        }
        // Two-lane cross-device PIPELINE is the default: it hides the (now
        // small) dGPU attention under the iGPU MoE by keeping two per-lane
        // handoffs in flight. `LAGUNA_PIPELINE=0` restores the sequential
        // dGPU-then-iGPU tiles for A/B benchmarking.
        let pipeline = std::env::var("LAGUNA_PIPELINE").map(|v| v != "0").unwrap_or(true);
        if pipeline {
            return self.prefill_batched_pipelined(tokens);
        }
        self.reset();
        let b_max: usize = std::env::var("LAGUNA_PREFILL_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
            .min(self.max_kv)
            .max(1);

        // Build the tiled-MoE + group-builder kernels on the iGPU.
        self.igpu.set_current()?;
        let igpu_arch = self.igpu.properties()?.gcn_arch_name;
        let tiled = LagunaMoeTiled::for_arch(&igpu_arch)?;
        let gb = MoeGroupBuilder::for_arch(&igpu_arch)?;
        self.dgpu.set_current()?;

        let mut ps = PrefillScratch::new(&self.dgpu, &self.igpu, b_max)?;
        self.dgpu.set_current()?;

        let n = tokens.len();
        let mut tile_start = 0usize;
        let mut last_result: Option<(usize, f32)> = None;
        while tile_start < n {
            let b = (n - tile_start).min(b_max);
            let toks = &tokens[tile_start..tile_start + b];
            let is_last_tile = tile_start + b == n;

            self.embed_batch(toks, &mut ps.h)?;
            // per-row absolute positions for rope.
            let pos_h: Vec<i32> = (0..b).map(|i| (tile_start + i) as i32).collect();
            self.dgpu.set_current()?;
            ps.pos.slice_view_mut(0, b).copy_from_host(&pos_h)?;

            for il in 0..N_LAYER {
                self.layer_batched(il, tile_start, b, &mut ps, &tiled, &gb)?;
            }

            if is_last_tile {
                // output norm + LM head on the LAST token's hidden row.
                self.dgpu.set_current()?;
                let st = &self.dstream;
                let last = b - 1;
                let h_last = ps.h.slice_view(last * HIDDEN, HIDDEN);
                self.dk.rms.launch_weighted(st, &mut self.ds.rn, &h_last, &self.output_norm, HIDDEN as u32, EPS)?;
                match self.output_w.dtype {
                    GgufType::Q6_K => self.dk.q6d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
                    GgufType::Q4_K => self.dk.q4d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
                    other => return Err(eyre!("LM head dtype {other:?}")),
                }
                st.synchronize()?;
                let mut logits = vec![0f32; VOCAB];
                self.ds.logits.copy_to_host(&mut logits)?;
                let (argmax, maxv) = logits.iter().enumerate().fold(
                    (0usize, f32::NEG_INFINITY),
                    |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                );
                last_result = Some((argmax, maxv));
            } else {
                self.dgpu.set_current()?;
                self.dstream.synchronize()?;
            }

            tile_start += b;
            self.kv_len = tile_start;
        }
        last_result.ok_or_else(|| eyre!("prefill_batched: no tiles processed"))
    }

    /// TWO-LANE PIPELINED batched prefill (default path). Splits each B_MAX
    /// tile into lane A (first ceil(b/2) tokens) and lane B (the rest), each
    /// with its own [`PrefillScratch`]. Per layer the enqueue order is
    ///
    ///   dstream:  attn_A  attn_B  [wait moe_A] combine_A  [wait moe_B] combine_B
    ///   istream:  [wait fn_in_A] moe_A  [wait fn_in_B] moe_B
    ///
    /// so lane A's iGPU MoE overlaps lane B's dGPU attention (and lane B's MoE
    /// overlaps lane A's combine). Steady-state wall collapses from
    /// Σ(dGPU)+Σ(iGPU) toward ≈Σ(iGPU) since the dGPU attention is now small.
    ///
    /// Cross-lane correctness: both lanes share the dGPU stream (FIFO) and the
    /// single KV cache. Lane A's `attn_batched` (which appends KV rows
    /// `[q_off_a .. q_off_a+b_a)` and reads only `[0 .. q_off_a+b_a)`) is
    /// enqueued before lane B's, so lane B's causal attention over
    /// `[0 .. q_off_b+i]` sees lane A's freshly-written keys — no event needed.
    /// Lane A never reads lane B's (future) keys. Parity token stays 22718.
    pub fn prefill_batched_pipelined(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        if tokens.is_empty() {
            return Err(eyre!("prefill_batched_pipelined: empty prompt"));
        }
        self.reset();
        let b_max: usize = std::env::var("LAGUNA_PREFILL_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
            .min(self.max_kv)
            .max(1);

        self.igpu.set_current()?;
        let igpu_arch = self.igpu.properties()?.gcn_arch_name;
        let tiled = LagunaMoeTiled::for_arch(&igpu_arch)?;
        let gb = MoeGroupBuilder::for_arch(&igpu_arch)?;
        self.dgpu.set_current()?;

        // Per-lane scratch sized to the max lane width (ceil(b_max/2)). Two of
        // them total ≈ one b_max scratch, so no extra peak memory vs single-lane.
        let lane_max = b_max.div_ceil(2).max(1);
        let mut ps_a = PrefillScratch::new(&self.dgpu, &self.igpu, lane_max)?;
        let mut ps_b = PrefillScratch::new(&self.dgpu, &self.igpu, lane_max)?;
        self.dgpu.set_current()?;

        let n = tokens.len();
        let mut tile_start = 0usize;
        let mut last_result: Option<(usize, f32)> = None;
        while tile_start < n {
            let b = (n - tile_start).min(b_max);
            let toks = &tokens[tile_start..tile_start + b];
            let is_last_tile = tile_start + b == n;

            if b < 2 {
                // Tiny tail: single lane (lane 0) — no overlap to be had.
                self.embed_batch(toks, &mut ps_a.h)?;
                let pos_h: Vec<i32> = (0..b).map(|i| (tile_start + i) as i32).collect();
                self.dgpu.set_current()?;
                ps_a.pos.slice_view_mut(0, b).copy_from_host(&pos_h)?;
                for il in 0..N_LAYER {
                    self.layer_batched(il, tile_start, b, &mut ps_a, &tiled, &gb)?;
                }
                if is_last_tile {
                    last_result = Some(self.prefill_head(&ps_a, b - 1)?);
                } else {
                    self.dgpu.set_current()?;
                    self.dstream.synchronize()?;
                }
                tile_start += b;
                self.kv_len = tile_start;
                continue;
            }

            let b_a = b.div_ceil(2);
            let b_b = b - b_a; // ≥ 1 for b ≥ 2
            let off_a = tile_start;
            let off_b = tile_start + b_a;
            let toks_a = &toks[..b_a];
            let toks_b = &toks[b_a..];

            self.embed_batch(toks_a, &mut ps_a.h)?;
            self.embed_batch(toks_b, &mut ps_b.h)?;
            let pos_a: Vec<i32> = (0..b_a).map(|i| (off_a + i) as i32).collect();
            let pos_b: Vec<i32> = (0..b_b).map(|i| (off_b + i) as i32).collect();
            self.dgpu.set_current()?;
            ps_a.pos.slice_view_mut(0, b_a).copy_from_host(&pos_a)?;
            ps_b.pos.slice_view_mut(0, b_b).copy_from_host(&pos_b)?;

            // DEEP two-lane pipeline. `pre(lane,L)` = dGPU attention + iGPU MoE
            // submit (moe_batched only ENQUEUES on the iGPU stream behind an
            // event — it does not block the host); `post(lane,L)` = dGPU
            // residual. The schedule queues, per lane, `post(L)` immediately
            // followed by `pre(L+1)` BEFORE switching lanes, so lane A's next-
            // layer attention is enqueued right after its own combine (not
            // behind lane B's combine). The iGPU stream then sees a continuous
            // moe_A(L), moe_B(L), moe_A(L+1), … with no per-layer bubble waiting
            // for the dGPU to produce the next lane's fn_in.
            //
            // Layer 0 is dense (no MoE) — run it for both lanes fully on the
            // dGPU, then pipeline the MoE layers 1..N_LAYER.
            self.attn_batched(0, off_a, b_a, &mut ps_a, 0)?;
            self.attn_batched(0, off_b, b_b, &mut ps_b, 1)?;

            // Warmup: pre() for the first MoE layer, both lanes.
            self.attn_batched(1, off_a, b_a, &mut ps_a, 0)?;
            self.moe_batched(1, b_a, &mut ps_a, &tiled, &gb, 0)?;
            self.attn_batched(1, off_b, b_b, &mut ps_b, 1)?;
            self.moe_batched(1, b_b, &mut ps_b, &tiled, &gb, 1)?;

            // Steady state: finish layer L then start L+1, per lane.
            for il in 1..(N_LAYER - 1) {
                self.combine_batched(il, b_a, &mut ps_a, 0)?;
                self.attn_batched(il + 1, off_a, b_a, &mut ps_a, 0)?;
                self.moe_batched(il + 1, b_a, &mut ps_a, &tiled, &gb, 0)?;

                self.combine_batched(il, b_b, &mut ps_b, 1)?;
                self.attn_batched(il + 1, off_b, b_b, &mut ps_b, 1)?;
                self.moe_batched(il + 1, b_b, &mut ps_b, &tiled, &gb, 1)?;
            }

            // Cooldown: post() for the last MoE layer, both lanes.
            self.combine_batched(N_LAYER - 1, b_a, &mut ps_a, 0)?;
            self.combine_batched(N_LAYER - 1, b_b, &mut ps_b, 1)?;

            if is_last_tile {
                // Last prompt token is lane B's last row (b_b ≥ 1).
                last_result = Some(self.prefill_head(&ps_b, b_b - 1)?);
            } else {
                // Drain the dGPU stream before the next tile's blocking embed
                // H2D into ps.h. combine_B (last dstream op) waited on moe_B, so
                // syncing dstream also drains the iGPU work feeding it.
                self.dgpu.set_current()?;
                self.dstream.synchronize()?;
            }

            tile_start += b;
            self.kv_len = tile_start;
        }
        last_result.ok_or_else(|| eyre!("prefill_batched_pipelined: no tiles processed"))
    }

    /// Output norm + LM head on batch row `row` of `ps.h`, returning
    /// (argmax, max-logit). Used by the pipelined prefill's last tile.
    fn prefill_head(&mut self, ps: &PrefillScratch, row: usize) -> eyre::Result<(usize, f32)> {
        self.dgpu.set_current()?;
        let st = &self.dstream;
        let h_row = ps.h.slice_view(row * HIDDEN, HIDDEN);
        self.dk.rms.launch_weighted(st, &mut self.ds.rn, &h_row, &self.output_norm, HIDDEN as u32, EPS)?;
        match self.output_w.dtype {
            GgufType::Q6_K => self.dk.q6d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
            GgufType::Q4_K => self.dk.q4d.matvec(st, &mut self.ds.logits, &self.output_w.bytes, &self.ds.rn, VOCAB as u32, HIDDEN as u32)?,
            other => return Err(eyre!("LM head dtype {other:?}")),
        }
        st.synchronize()?;
        let mut logits = vec![0f32; VOCAB];
        self.ds.logits.copy_to_host(&mut logits)?;
        Ok(logits.iter().enumerate().fold(
            (0usize, f32::NEG_INFINITY),
            |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
        ))
    }

    /// SINGLE-LANE batched prefill with the het-MoE split (hot experts on the
    /// dGPU ∥ cold experts on the iGPU). Attention and the LM head are identical
    /// to `prefill_batched`; only the MoE layers span both devices. Router runs
    /// on the dGPU. Sequential per layer (attn → split-MoE → combine); the split
    /// overlaps the dGPU hot experts with the iGPU cold experts. Parity: the
    /// routed sum is reordered (hot+cold vs all-iGPU), greedy-exact + in-tol.
    pub fn prefill_batched_het(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        if tokens.is_empty() {
            return Err(eyre!("prefill_batched_het: empty prompt"));
        }
        // Two-lane cross-device pipeline is the default (hides the dGPU
        // attention + hot-MoE leg under the iGPU cold-MoE leg across lanes).
        // `LAGUNA_PIPELINE=0` runs the sequential single-lane split.
        let pipeline = std::env::var("LAGUNA_PIPELINE").map(|v| v != "0").unwrap_or(true);
        if pipeline {
            return self.prefill_batched_het_pipelined(tokens);
        }
        self.reset();
        let b_max: usize = std::env::var("LAGUNA_PREFILL_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
            .min(self.max_kv)
            .max(1);

        // Tiled MoE + group builder for BOTH devices (hot on dGPU, cold on iGPU).
        self.igpu.set_current()?;
        let igpu_arch = self.igpu.properties()?.gcn_arch_name;
        let tiled_i = LagunaMoeTiled::for_arch(&igpu_arch)?;
        let gb_i = MoeGroupBuilder::for_arch(&igpu_arch)?;
        self.dgpu.set_current()?;
        let dgpu_arch = self.dgpu.properties()?.gcn_arch_name;
        let tiled_d = LagunaMoeTiled::for_arch(&dgpu_arch)?;
        let gb_d = MoeGroupBuilder::for_arch(&dgpu_arch)?;

        let mut ps = PrefillScratch::new(&self.dgpu, &self.igpu, b_max)?;
        let mut hs = HotScratch::new(&self.dgpu, b_max)?;
        self.dgpu.set_current()?;
        self.prefill_split = true;

        let n = tokens.len();
        let mut tile_start = 0usize;
        let mut last_result: Option<(usize, f32)> = None;
        while tile_start < n {
            let b = (n - tile_start).min(b_max);
            let toks = &tokens[tile_start..tile_start + b];
            let is_last_tile = tile_start + b == n;

            self.embed_batch(toks, &mut ps.h)?;
            let pos_h: Vec<i32> = (0..b).map(|i| (tile_start + i) as i32).collect();
            self.dgpu.set_current()?;
            ps.pos.slice_view_mut(0, b).copy_from_host(&pos_h)?;

            for il in 0..N_LAYER {
                // dGPU attention front-half (pushes fn_in + records fn_in_evt).
                if self.attn_batched(il, tile_start, b, &mut ps, 0)? {
                    continue; // dense layer-0 fully handled on the dGPU
                }
                if self.dlayers[il].hot.is_some() {
                    self.moe_batched_split(il, b, &mut ps, &mut hs, &tiled_d, &gb_d, &tiled_i, &gb_i, 0)?;
                    self.combine_batched_split(il, b, &mut ps, &mut hs, 0)?;
                } else {
                    // No hot residency for this layer: pure-iGPU MoE fallback
                    // (attn_batched already pushed fn_in + recorded fn_in_evt).
                    self.moe_batched(il, b, &mut ps, &tiled_i, &gb_i, 0)?;
                    self.combine_batched(il, b, &mut ps, 0)?;
                }
            }

            if is_last_tile {
                last_result = Some(self.prefill_head(&ps, b - 1)?);
            } else {
                self.dgpu.set_current()?;
                self.dstream.synchronize()?;
            }
            tile_start += b;
            self.kv_len = tile_start;
        }
        self.prefill_split = false;
        last_result.ok_or_else(|| eyre!("prefill_batched_het: no tiles processed"))
    }

    /// TWO-LANE PIPELINED het-split prefill (default when het is enabled). Same
    /// lane schedule as [`Self::prefill_batched_pipelined`] but each MoE layer
    /// runs the hot experts on the dGPU (`dstream`) ∥ cold experts on the iGPU
    /// (`istream`). The dGPU leg is now attention + hot MoE; the iGPU leg is the
    /// cold MoE + shared expert. Two lanes keep both legs busy: lane A's iGPU
    /// cold MoE overlaps lane B's dGPU attention + hot MoE.
    pub fn prefill_batched_het_pipelined(&mut self, tokens: &[usize]) -> eyre::Result<(usize, f32)> {
        if tokens.is_empty() {
            return Err(eyre!("prefill_batched_het_pipelined: empty prompt"));
        }
        self.reset();
        let b_max: usize = std::env::var("LAGUNA_PREFILL_BMAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
            .min(self.max_kv)
            .max(1);

        self.igpu.set_current()?;
        let igpu_arch = self.igpu.properties()?.gcn_arch_name;
        let tiled_i = LagunaMoeTiled::for_arch(&igpu_arch)?;
        let gb_i = MoeGroupBuilder::for_arch(&igpu_arch)?;
        self.dgpu.set_current()?;
        let dgpu_arch = self.dgpu.properties()?.gcn_arch_name;
        let tiled_d = LagunaMoeTiled::for_arch(&dgpu_arch)?;
        let gb_d = MoeGroupBuilder::for_arch(&dgpu_arch)?;

        let lane_max = b_max.div_ceil(2).max(1);
        let mut ps_a = PrefillScratch::new(&self.dgpu, &self.igpu, lane_max)?;
        let mut ps_b = PrefillScratch::new(&self.dgpu, &self.igpu, lane_max)?;
        let mut hs_a = HotScratch::new(&self.dgpu, lane_max)?;
        let mut hs_b = HotScratch::new(&self.dgpu, lane_max)?;
        self.dgpu.set_current()?;
        self.prefill_split = true;

        let n = tokens.len();
        let mut tile_start = 0usize;
        let mut last_result: Option<(usize, f32)> = None;
        while tile_start < n {
            let b = (n - tile_start).min(b_max);
            let toks = &tokens[tile_start..tile_start + b];
            let is_last_tile = tile_start + b == n;

            if b < 2 {
                // Tiny tail: single lane, no overlap to be had.
                self.embed_batch(toks, &mut ps_a.h)?;
                let pos_h: Vec<i32> = (0..b).map(|i| (tile_start + i) as i32).collect();
                self.dgpu.set_current()?;
                ps_a.pos.slice_view_mut(0, b).copy_from_host(&pos_h)?;
                for il in 0..N_LAYER {
                    if self.attn_batched(il, tile_start, b, &mut ps_a, 0)? {
                        continue;
                    }
                    if self.dlayers[il].hot.is_some() {
                        self.moe_batched_split(il, b, &mut ps_a, &mut hs_a, &tiled_d, &gb_d, &tiled_i, &gb_i, 0)?;
                        self.combine_batched_split(il, b, &mut ps_a, &mut hs_a, 0)?;
                    } else {
                        self.moe_batched(il, b, &mut ps_a, &tiled_i, &gb_i, 0)?;
                        self.combine_batched(il, b, &mut ps_a, 0)?;
                    }
                }
                if is_last_tile {
                    last_result = Some(self.prefill_head(&ps_a, b - 1)?);
                } else {
                    self.dgpu.set_current()?;
                    self.dstream.synchronize()?;
                }
                tile_start += b;
                self.kv_len = tile_start;
                continue;
            }

            let b_a = b.div_ceil(2);
            let b_b = b - b_a;
            let off_a = tile_start;
            let off_b = tile_start + b_a;
            let toks_a = &toks[..b_a];
            let toks_b = &toks[b_a..];

            self.embed_batch(toks_a, &mut ps_a.h)?;
            self.embed_batch(toks_b, &mut ps_b.h)?;
            let pos_a: Vec<i32> = (0..b_a).map(|i| (off_a + i) as i32).collect();
            let pos_b: Vec<i32> = (0..b_b).map(|i| (off_b + i) as i32).collect();
            self.dgpu.set_current()?;
            ps_a.pos.slice_view_mut(0, b_a).copy_from_host(&pos_a)?;
            ps_b.pos.slice_view_mut(0, b_b).copy_from_host(&pos_b)?;

            // Helper closures aren't ergonomic with &mut self; inline the split
            // MoE call chain. Layer 0 is dense (both lanes on the dGPU).
            self.attn_batched(0, off_a, b_a, &mut ps_a, 0)?;
            self.attn_batched(0, off_b, b_b, &mut ps_b, 1)?;

            // Warmup: attn+MoE-submit for the first MoE layer, both lanes.
            self.attn_batched(1, off_a, b_a, &mut ps_a, 0)?;
            self.moe_batched_split(1, b_a, &mut ps_a, &mut hs_a, &tiled_d, &gb_d, &tiled_i, &gb_i, 0)?;
            self.attn_batched(1, off_b, b_b, &mut ps_b, 1)?;
            self.moe_batched_split(1, b_b, &mut ps_b, &mut hs_b, &tiled_d, &gb_d, &tiled_i, &gb_i, 1)?;

            // Steady state: finish layer L then start L+1, per lane.
            for il in 1..(N_LAYER - 1) {
                self.combine_batched_split(il, b_a, &mut ps_a, &mut hs_a, 0)?;
                self.attn_batched(il + 1, off_a, b_a, &mut ps_a, 0)?;
                self.moe_batched_split(il + 1, b_a, &mut ps_a, &mut hs_a, &tiled_d, &gb_d, &tiled_i, &gb_i, 0)?;

                self.combine_batched_split(il, b_b, &mut ps_b, &mut hs_b, 1)?;
                self.attn_batched(il + 1, off_b, b_b, &mut ps_b, 1)?;
                self.moe_batched_split(il + 1, b_b, &mut ps_b, &mut hs_b, &tiled_d, &gb_d, &tiled_i, &gb_i, 1)?;
            }

            // Cooldown: combine the last MoE layer, both lanes.
            self.combine_batched_split(N_LAYER - 1, b_a, &mut ps_a, &mut hs_a, 0)?;
            self.combine_batched_split(N_LAYER - 1, b_b, &mut ps_b, &mut hs_b, 1)?;

            if is_last_tile {
                last_result = Some(self.prefill_head(&ps_b, b_b - 1)?);
            } else {
                self.dgpu.set_current()?;
                self.dstream.synchronize()?;
            }
            tile_start += b;
            self.kv_len = tile_start;
        }
        self.prefill_split = false;
        last_result.ok_or_else(|| eyre!("prefill_batched_het_pipelined: no tiles processed"))
    }
}

/// Reusable device scratch for batched prefill, sized to `b_max`. dGPU carries
/// the attention front-half + residuals; iGPU carries the tiled MoE + shared
/// expert. Every per-batch buffer is TIGHTLY packed `[b_max, width]`.
struct PrefillScratch {
    // dGPU
    h: DeviceBuffer<f32>,
    ain: DeviceBuffer<f32>,
    q: DeviceBuffer<f32>,
    qn: DeviceBuffer<f32>,
    qf: DeviceBuffer<u16>,
    k: DeviceBuffer<f32>,
    kn: DeviceBuffer<f32>,
    v: DeviceBuffer<f32>,
    od: DeviceBuffer<f32>,
    gate_logits: DeviceBuffer<f32>,
    op: DeviceBuffer<f32>,
    fn_in: DeviceBuffer<f32>,
    ffn_out: DeviceBuffer<f32>,
    gate_big: DeviceBuffer<f32>,
    up_big: DeviceBuffer<f32>,
    sw_big: DeviceBuffer<f32>,
    moe_recv: DeviceBuffer<f32>,
    pos: DeviceBuffer<i32>,
    // iGPU
    fn_in_recv: DeviceBuffer<f32>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    router_probs: DeviceBuffer<f32>,
    router_scores: DeviceBuffer<f32>,
    xq_hidden: DeviceBuffer<u8>,
    mid: DeviceBuffer<f32>,
    xq_mid: DeviceBuffer<u8>,
    acc: DeviceBuffer<f32>,
    down_part: DeviceBuffer<f32>, // [b, TOPK, HIDDEN] atomic-free down partials
    gate_s: DeviceBuffer<f32>,
    up_s: DeviceBuffer<f32>,
    sw_s: DeviceBuffer<f32>,
    xq_sw: DeviceBuffer<u8>, // Q8_K of sw_s for the read-once shared-expert down GEMM
    down_s: DeviceBuffer<f32>,
    ffn_out_i: DeviceBuffer<f32>,
    group_count: DeviceBuffer<i32>,
    members: DeviceBuffer<i32>,
}

impl PrefillScratch {
    fn new(dgpu_dev: &Device, igpu_dev: &Device, b: usize) -> eyre::Result<Self> {
        let n_embd_q_max = 72 * HEAD_DIM;
        let n_blk_hidden = HIDDEN / 256;
        let n_blk_mid = FF_EXP / 256;
        let dgpu = dgpu_dev.id;
        let igpu = igpu_dev.id;
        // dGPU allocations — hipMalloc uses the CURRENT device.
        dgpu_dev.set_current()?;
        let mkd = |n: usize| DeviceBuffer::<f32>::new(dgpu, n);
        let h = mkd(b * HIDDEN)?;
        let ain = mkd(b * HIDDEN)?;
        let q = mkd(b * n_embd_q_max)?;
        let qn = mkd(b * n_embd_q_max)?;
        let qf = DeviceBuffer::<u16>::new(dgpu, b * n_embd_q_max)?;
        let k = mkd(b * N_KV_HEAD * HEAD_DIM)?;
        let kn = mkd(b * N_KV_HEAD * HEAD_DIM)?;
        let v = mkd(b * N_KV_HEAD * HEAD_DIM)?;
        let od = mkd(b * n_embd_q_max)?;
        let gate_logits = mkd(b * 72)?;
        let op = mkd(b * HIDDEN)?;
        let fn_in = mkd(b * HIDDEN)?;
        let ffn_out = mkd(b * HIDDEN)?;
        let gate_big = mkd(b * FF_DENSE)?;
        let up_big = mkd(b * FF_DENSE)?;
        let sw_big = mkd(b * FF_DENSE)?;
        let moe_recv = mkd(b * HIDDEN)?;
        let pos = DeviceBuffer::<i32>::new(dgpu, b)?;
        // iGPU allocations.
        igpu_dev.set_current()?;
        let mki = |n: usize| DeviceBuffer::<f32>::new(igpu, n);
        let fn_in_recv = mki(b * HIDDEN)?;
        let sel = DeviceBuffer::<i32>::new(igpu, b * TOPK)?;
        let ew = mki(b * TOPK)?;
        let router_probs = mki(b * N_EXPERT)?;
        let router_scores = mki(b * N_EXPERT)?;
        let xq_hidden = DeviceBuffer::<u8>::new(igpu, b * n_blk_hidden * 292)?;
        let mid = mki(b * TOPK * FF_EXP)?;
        let xq_mid = DeviceBuffer::<u8>::new(igpu, b * TOPK * n_blk_mid * 292)?;
        let acc = mki(b * HIDDEN)?;
        let down_part = mki(b * TOPK * HIDDEN)?;
        let gate_s = mki(b * FF_SHEXP)?;
        let up_s = mki(b * FF_SHEXP)?;
        let sw_s = mki(b * FF_SHEXP)?;
        let xq_sw = DeviceBuffer::<u8>::new(igpu, b * (FF_SHEXP / 256) * 292)?;
        let down_s = mki(b * HIDDEN)?;
        let ffn_out_i = mki(b * HIDDEN)?;
        let group_count = DeviceBuffer::<i32>::new(igpu, N_EXPERT)?;
        let members = DeviceBuffer::<i32>::new(igpu, N_EXPERT * b)?;
        Ok(Self {
            h, ain, q, qn, qf, k, kn, v, od, gate_logits, op, fn_in, ffn_out,
            gate_big, up_big, sw_big, moe_recv, pos,
            fn_in_recv, sel, ew, router_probs, router_scores, xq_hidden, mid, xq_mid,
            acc, down_part, gate_s, up_s, sw_s, xq_sw, down_s, ffn_out_i, group_count, members,
        })
    }
}

/// dGPU-resident scratch for the prefill het-split HOT path (router + hot
/// experts). Sized to `b_max`. The K-wide hot WEIGHTS live in `DgpuLayer.hot`;
/// this holds only the per-batch activations + the K-space by-expert groups.
/// `K_MAX = 32` bounds the hot-expert count (matches the largest hot-set file).
struct HotScratch {
    xq_hidden: DeviceBuffer<u8>,      // Q8_K(fn_in) [b, n_blk_hidden, 292]
    sel: DeviceBuffer<i32>,           // router selection [b, TOPK] (global ids)
    ew: DeviceBuffer<f32>,            // routing weights [b, TOPK]
    router_probs: DeviceBuffer<f32>,  // [b, N_EXPERT]
    router_scores: DeviceBuffer<f32>, // [b, N_EXPERT]
    group_count: DeviceBuffer<i32>,   // [K_MAX] dense-local hot groups
    members: DeviceBuffer<i32>,       // [K_MAX * b]
    mid: DeviceBuffer<f32>,           // [b, TOPK, FF_EXP]
    xq_mid: DeviceBuffer<u8>,         // Q8_K(mid) [b, TOPK, n_blk_mid, 292]
    down_part: DeviceBuffer<f32>,     // [b, TOPK, HIDDEN] atomic-free down partials
    acc: DeviceBuffer<f32>,           // [b, HIDDEN] hot routed partial sum
}

impl HotScratch {
    const K_MAX: usize = 32;

    fn new(dgpu_dev: &Device, b: usize) -> eyre::Result<Self> {
        let n_blk_hidden = HIDDEN / 256;
        let n_blk_mid = FF_EXP / 256;
        let dgpu = dgpu_dev.id;
        dgpu_dev.set_current()?;
        let mkd = |n: usize| DeviceBuffer::<f32>::new(dgpu, n);
        Ok(Self {
            xq_hidden: DeviceBuffer::<u8>::new(dgpu, b * n_blk_hidden * 292)?,
            sel: DeviceBuffer::<i32>::new(dgpu, b * TOPK)?,
            ew: mkd(b * TOPK)?,
            router_probs: mkd(b * N_EXPERT)?,
            router_scores: mkd(b * N_EXPERT)?,
            group_count: DeviceBuffer::<i32>::new(dgpu, Self::K_MAX)?,
            members: DeviceBuffer::<i32>::new(dgpu, Self::K_MAX * b)?,
            mid: mkd(b * TOPK * FF_EXP)?,
            xq_mid: DeviceBuffer::<u8>::new(dgpu, b * TOPK * n_blk_mid * 292)?,
            down_part: mkd(b * TOPK * HIDDEN)?,
            acc: mkd(b * HIDDEN)?,
        })
    }
}
