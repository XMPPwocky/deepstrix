//! Launchers for `kernels/mxfp4_pair_matvec.hip` — the fused gate+up SwiGLU
//! pair kernels for MXFP4 routed experts (DeepSeek-V4.1-Flash's native
//! expert format). Same contracts as [`crate::iq3_s::Iq3SPairMatvec`], so
//! `het::dispatch` routes `GgufType::MXFP4` gate/up here.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const MXFP4_PAIR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1201"));
const MXFP4_PAIR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_MATVEC_GFX1151"));
/// `kernels/mxfp4_moe_wmma.hip` (the int8-WMMA gate+up / down arm, 2026-09-26
/// sweep B_moe_prefill). gfx1151 only: the RDNA3 WMMA intrinsic; the gfx12 build
/// of that file traps, so it is never loaded there.
pub(crate) const MXFP4_MOE_WMMA_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MOE_WMMA_GFX1151"));
/// `kernels/mxfp4_moe_wmma_f16.hip` (the f16-WMMA arm, 2026-10-10): same entry
/// names / args / chunk as the int8 arm, selected per launch (`V41_MOE_WMMA_F16`).
pub(crate) const MXFP4_MOE_WMMA_F16_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_MOE_WMMA_F16_GFX1151"));
/// `kernels/mxfp4_pair_decode.hip` (the decode-regime gate+up, `V41_MOE_PAIR_DEC`,
/// 2026-10-13). Loaded on gfx1151 only (the iGPU runs the MXFP4 MoE).
pub(crate) const MXFP4_PAIR_DECODE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_MXFP4_PAIR_DECODE_GFX1151"));

/// Q8_K super-blocks per row the decode kernel is compiled for (`MXDEC_NB`:
/// V4.1 gate/up, 5120 / 256).
pub const MXFP4_DEC_N_BLOCKS: u32 = 20;
/// Most token rows (`rows`) the decode kernel stages (`MXDEC_MAXTOK`).
pub const MXFP4_DEC_MAX_TOK: u32 = 8;
/// Default waves per work-group of the decode kernel (one work-group per CU,
/// `__launch_bounds__(512)` caps it at 16). `V41_MOE_PAIR_DEC_WAVES` overrides it.
pub const MXFP4_DEC_WAVES_DEFAULT: u32 = 16;
/// Work-item buffers that can hold a dynamic-order ticket of the decode kernel
/// at once (a launch past them runs the static unit order).
pub const MXFP4_DEC_TICKET_SLOTS: usize = 32;

/// Members per work item the WMMA kernels stage (`WM_MAX_CHUNK`; the kernel
/// silently clamps a larger chunk, which would DROP members, so the launchers
/// refuse it).
pub const MXFP4_WMMA_MAX_CHUNK: u32 = 32;
/// Weight rows per work-group of the WMMA kernels (`WM_ROWS_PER_WG`).
pub const MXFP4_WMMA_ROWS_PER_WG: u32 = 128;

/// Q8_K superblocks the kernels stage in LDS (`MXFP4_PAIR_MAX_BLOCKS`).
pub const MXFP4_PAIR_MAX_BLOCKS: u32 = 32;
/// Upper bound on the prefill kwide kernel's `chunk_size` (`MXFP4_KW_MAX_CHUNK`
/// in the kernel: sizes its LDS staging and per-lane accumulators).
pub const MXFP4_KW_MAX_CHUNK: u32 = 32;
/// Members one pass of the grouped drafter gate+up holds in registers
/// (`MXFP4_GROUPED_MAX_MEMBERS` in the kernel). A group never has more members
/// than tokens (a token's top-k picks are distinct), so `n_tok <= 8` is one pass.
pub const MXFP4_GROUPED_MAX_MEMBERS: u32 = 8;
/// Most picks (`n_tok * n_used`) the grouped drafter kernels take
/// (`MXFP4_GROUPED_MAX_PICKS`, kernels/mxfp4_matvec.hip: the down kernel's
/// per-lane `inner[pick]` registers).
pub const MXFP4_GROUPED_MAX_PICKS: u32 = 16;

/// The MoE launch-arg canary for a gate+up launch (`crate::moe_canary`): every pointer argument live, the weights on
/// their pool's slots, the scalars and grid consistent. `extra` = the decode kernel's ticket.
#[allow(clippy::too_many_arguments)]
fn canary_gate_up(
    kernel: &'static str,
    stream: &Stream,
    mid: &DeviceBuffer<f32>,
    gate_w_base: &DeviceBuffer<u8>,
    up_w_base: &DeviceBuffer<u8>,
    xq: &DeviceBuffer<u8>,
    expert_w: &DeviceBuffer<f32>,
    group_count: &DeviceBuffer<i32>,
    expert_members: &DeviceBuffer<i32>,
    work_items: &DeviceBuffer<i32>,
    n_work_items_dev: Option<&DeviceBuffer<i32>>,
    extra: Option<crate::moe_canary::Ptr>,
    gate_bpe: u32,
    up_bpe: u32,
    n_rows: u32,
    n_blocks: u32,
    n_work_items: u32,
    chunk_size: u32,
    cfg: &LaunchConfig,
    grid_kind: crate::moe_canary::Grid,
    loc: &'static std::panic::Location<'static>,
) -> eyre::Result<()> {
    use crate::moe_canary::{check, MoeLaunch, Ptr, Role, Weight};
    let mut ptrs = [Ptr::of("mid", mid); 8];
    let mut n = 1;
    for p in [
        Some(Ptr::of("xq", xq)),
        Some(Ptr::of("expert_w", expert_w)),
        Some(Ptr::of("group_count", group_count)),
        Some(Ptr::of("expert_members", expert_members)),
        Some(Ptr::of("work_items", work_items)),
        n_work_items_dev.map(|c| Ptr::of("n_work_items_dev", c)),
        extra,
    ]
    .into_iter()
    .flatten()
    {
        ptrs[n] = p;
        n += 1;
    }
    let weights = [
        Weight { arg: Ptr::of("gate_w_base", gate_w_base), role: Role::Gate, bpe: gate_bpe },
        Weight { arg: Ptr::of("up_w_base", up_w_base), role: Role::Up, bpe: up_bpe },
    ];
    check(
        &MoeLaunch {
            kernel,
            device: stream.device_id(),
            ptrs: &ptrs[..n],
            weights: &weights,
            n_rows,
            n_blocks,
            n_work_items,
            work_items_len: work_items.len(),
            chunk_size,
            grid: cfg.grid,
            grid_kind,
        },
        loc,
    )
}

pub struct Mxfp4PairMatvec {
    module: Module,
    /// The int8-WMMA module (gfx1151 only), see [`Self::launch_fused_swiglu_wmma_ex`].
    wmma: Option<Module>,
    /// The f16-WMMA module (gfx1151 only), see [`Self::launch_fused_swiglu_wmma_arm`].
    wmma_f16: Option<Module>,
    /// The decode-regime module (gfx1151 only), see [`Self::launch_fused_swiglu_dec_ex`].
    dec: Option<Module>,
    /// CUs of the device the decode kernel launches on (its persistent grid).
    dec_cus: std::sync::OnceLock<u32>,
    /// The decode kernel's self-resetting ticket counters, 2 u32 per slot, zeroed
    /// at load (never inside a launch path: no memset can land in a captured
    /// graph), and the `work_items` buffer each slot belongs to. Two launches
    /// that share a work-item buffer are never in flight together (the builder
    /// rewrites it for every launch, so the kernels' own outputs already depend
    /// on that), hence neither are two launches that share a ticket -- captured
    /// graphs included, whose replays carry the captured buffer.
    dec_tickets: Option<DeviceBuffer<u32>>,
    dec_ticket_keys: std::sync::Mutex<Vec<usize>>,
}

/// `V41_MOE_PAIR_DEC_WAVES` (read once, 1..=16): waves per work-group of the
/// decode kernel; default [`MXFP4_DEC_WAVES_DEFAULT`].
pub fn dec_waves() -> u32 {
    static W: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_MOE_PAIR_DEC_WAVES")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| (1..=16).contains(n))
            .unwrap_or(MXFP4_DEC_WAVES_DEFAULT)
    });
    *W
}

/// `V41_MOE_PAIR_DEC_DYN` (read once; default ON, `0` = off): the decode kernel
/// takes its units after the first from a ticket counter (dynamic order, one
/// ticket per `work_items` buffer, see `dec_tickets`) instead of a fixed stride.
/// Same bits either way; measured 2% faster at D = 16 (the static stride leaves a
/// one-unit tail and waits for the slowest CU), neutral at D <= 8.
pub fn dec_dynamic() -> bool {
    static D: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_MOE_PAIR_DEC_DYN").as_deref() != Ok("0"));
    *D
}

/// `V41_MOE_PAIR_DEC_WGS` (read once): work-groups of the decode kernel's
/// persistent grid; 0 / unset = one per CU of the launching device.
fn dec_wgs_override() -> u32 {
    static G: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_MOE_PAIR_DEC_WGS").ok().and_then(|v| v.parse::<u32>().ok()).unwrap_or(0)
    });
    *G
}

/// Warps per workgroup (= rows per workgroup, one row each) for the decode
/// pair kernels. `V41_MXFP4_PAIR_WARPS` sweeps it; default 8 = the historical
/// hardcoded geometry.
fn pair_warps() -> u32 {
    static W: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_MXFP4_PAIR_WARPS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|w| (1..=32).contains(w))
            .unwrap_or(8)
    });
    *W
}

impl Mxfp4PairMatvec {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            MXFP4_PAIR_GFX1201
        } else if arch.starts_with("gfx1151") {
            MXFP4_PAIR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for mxfp4_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        let (wmma, wmma_f16, dec) = if arch.starts_with("gfx1151") {
            (
                Some(Module::load_data(MXFP4_MOE_WMMA_GFX1151)?),
                Some(Module::load_data(MXFP4_MOE_WMMA_F16_GFX1151)?),
                Some(Module::load_data(MXFP4_PAIR_DECODE_GFX1151)?),
            )
        } else {
            (None, None, None)
        };
        // The decode kernel's per-device state, set up here (never in a launch
        // path, which may be inside a graph capture): the CU count of its
        // persistent grid and the zeroed tickets of the dynamic order.
        let dec_cus = std::sync::OnceLock::new();
        let dec_tickets = if dec.is_some() {
            let mut dev = 0i32;
            // SAFETY: plain query of the calling thread's current device.
            unsafe { v4flash_hip::sys::hipGetDevice(&mut dev) };
            // HIP reports WGPs on RDNA: CUs = 2x.
            let wgps = v4flash_hip::Device::new(dev).properties()?.multi_processor_count.max(1) as u32;
            let _ = dec_cus.set(2 * wgps);
            let mut t = DeviceBuffer::<u32>::new(dev, 2 * MXFP4_DEC_TICKET_SLOTS)?;
            t.fill_zero()?;
            Some(t)
        } else {
            None
        };
        Ok(Self {
            module, wmma, wmma_f16, dec, dec_cus, dec_tickets,
            dec_ticket_keys: std::sync::Mutex::new(Vec::new()),
        })
    }

    /// Whether the int8-WMMA gate+up kernel exists on this device (gfx1151).
    pub fn has_wmma(&self) -> bool { self.wmma.is_some() }
    /// Whether the f16-WMMA gate+up kernel exists on this device (gfx1151).
    pub fn has_wmma_f16(&self) -> bool { self.wmma_f16.is_some() }

    /// Whether the decode-regime gate+up kernel exists on this device (gfx1151).
    pub fn has_dec(&self) -> bool { self.dec.is_some() }

    /// The loaded modules (tests: explicit-symbol launches).
    pub fn module(&self) -> &Module { &self.module }
    pub fn dec_module(&self) -> Option<&Module> { self.dec.as_ref() }

    /// The decode kernel's ticket for launches on `work_items` (a 2-u32 view),
    /// or None when every slot belongs to another work-item buffer.
    pub fn dec_ticket(&self, work_items: &DeviceBuffer<i32>) -> Option<DeviceBuffer<u32>> {
        let t = self.dec_tickets.as_ref()?;
        let key = work_items.raw() as usize;
        let mut m = self.dec_ticket_keys.lock().ok()?;
        let slot = match m.iter().position(|&k| k == key) {
            Some(i) => i,
            None if m.len() < MXFP4_DEC_TICKET_SLOTS => {
                m.push(key);
                m.len() - 1
            }
            None => return None,
        };
        Some(t.slice_view(2 * slot, 2))
    }

    /// The decode kernel's default persistent grid on `stream`'s device: one
    /// work-group per CU (`V41_MOE_PAIR_DEC_WGS` overrides).
    pub fn dec_wgs(&self, _stream: &Stream) -> eyre::Result<u32> {
        if dec_wgs_override() > 0 {
            return Ok(dec_wgs_override());
        }
        self.dec_cus.get().copied().ok_or_else(|| eyre!("mxfp4 pair dec: no decode module on this arch (gfx1151 only)"))
    }
    pub fn wmma_module(&self) -> Option<&Module> { self.wmma.as_ref() }
    pub fn wmma_f16_module(&self) -> Option<&Module> { self.wmma_f16.as_ref() }

    fn check(mid: &DeviceBuffer<f32>, n_used: u32, n_rows: u32, n_blocks: u32) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!(
                "mxfp4 pair: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}] (LDS staging)"
            ));
        }
        if mid.len() < (n_used as usize) * (n_rows as usize) {
            return Err(eyre!("mxfp4 pair mid: len {} < n_used*n_rows", mid.len()));
        }
        Ok(())
    }

    /// Decode: `mid[slot, row] = swiglu(gate·x, up·x) * expert_w[slot]` for
    /// the `n_used` selected experts. `n_blocks` counts Q8_K superblocks.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        Self::check(mid, n_used, n_rows, n_blocks)?;
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_batch")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(pair_warps()), n_used, 1),
            block: (pair_warps() * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(),
                gate_w_base.raw(),
                up_w_base.raw(),
                xq.raw(),
                expert_w.raw(),
                selected.raw(),
                gate_bpe,
                up_bpe,
                clamp,
                n_rows,
                n_blocks
            ]
        )
    }

    /// Decode het-split (see the kernel comment for the `remap` / `mode` /
    /// `dgpu_cap` contract).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch_hetsplit(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        mode: u32,
        dgpu_cap: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        Self::check(mid, n_used, n_rows, n_blocks)?;
        let function = self
            .module
            .get_function("mxfp4_pair_matvec_fused_swiglu_batch_hetsplit")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(pair_warps()), n_used, 1),
            block: (pair_warps() * 32, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(),
                gate_w_base.raw(),
                up_w_base.raw(),
                xq.raw(),
                expert_w.raw(),
                selected.raw(),
                remap.raw(),
                mode,
                dgpu_cap,
                gate_bpe,
                up_bpe,
                clamp,
                n_rows,
                n_blocks
            ]
        )
    }

    /// Drafter block gate+up GROUPED BY EXPERT
    /// (`mxfp4_pair_matvec_fused_swiglu_grouped`, `V41_MTP_MOE_GROUPED`): all
    /// `n_tok` tokens' `n_used` picks in ONE launch, each DISTINCT resident
    /// expert read once and applied to every token that picked it. Picks are
    /// token-major (`p = token * n_used + slot`, the drafter's `[B, TOPK]`
    /// `d_selected` / `d_ew`); `mid` is `[n_picks, n_rows]` and `xq` is
    /// `[n_tok, n_blocks * 292]`.
    ///
    /// BIT-IDENTICAL per (pick, row) to `n_tok` launches of
    /// [`Self::launch_fused_swiglu_batch_hetsplit`] with mode 0 and
    /// `dgpu_cap >= n_used` (the only het-split contract it implements: a pick
    /// is ours iff `remap[sel] < 0`, at expert `-remap[sel] - 1`; others get
    /// `mid = 0`). Graph-capturable: no host-side grouping.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_grouped(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        selected: &DeviceBuffer<i32>,
        remap: &DeviceBuffer<i32>,
        gate_bpe: u32,
        up_bpe: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_tok: u32,
        n_used: u32,
    ) -> eyre::Result<()> {
        let n_picks = n_tok * n_used;
        if n_tok == 0 || n_used == 0 || n_picks > MXFP4_GROUPED_MAX_PICKS {
            return Err(eyre!("mxfp4 pair grouped: n_tok={n_tok} x n_used={n_used} outside 1..={MXFP4_GROUPED_MAX_PICKS} picks"));
        }
        Self::check(mid, n_picks, n_rows, n_blocks)?;
        let xq_bytes = n_blocks as usize * crate::q8_k::BLOCK_Q8_K_BYTES;
        if xq.len() < n_tok as usize * xq_bytes {
            return Err(eyre!("mxfp4 pair grouped xq: {} bytes < n_tok * {xq_bytes}", xq.len()));
        }
        if (expert_w.len() as u32) < n_picks || (selected.len() as u32) < n_picks {
            return Err(eyre!(
                "mxfp4 pair grouped: expert_w {} / selected {} < n_picks {n_picks}",
                expert_w.len(),
                selected.len()
            ));
        }
        // One LDS slot per member of a pass; n_tok slots = one pass per group.
        let member_slots = n_tok.min(MXFP4_GROUPED_MAX_MEMBERS);
        let lds = member_slots as usize * xq_bytes;
        if lds > 64 * 1024 {
            return Err(eyre!("mxfp4 pair grouped: {lds} B of LDS staging > 64 KiB"));
        }
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_grouped")?;
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_picks, 1),
            block: (256, 1, 1),
            shared_mem_bytes: lds as u32,
        };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                selected.raw(), remap.raw(), gate_bpe, up_bpe, clamp, n_rows, n_blocks,
                n_used, n_picks, member_slots
            ]
        )
    }

    /// Prefill chunked by-expert (work-items contract of
    /// [`crate::iq3_s::Iq3SPairMatvec::launch_fused_swiglu_chunked`]): the
    /// per-member re-dequant fallback (`PAIR_VARIANT=chunked`).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair chunked: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!("mxfp4 pair chunked: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}]"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_chunked")?;
        let cfg = LaunchConfig { grid: (n_rows / 8, n_work_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks
            ]
        )
    }

    /// Prefill kwide (M51 structure: each lane's 16 gate+up weights dequantised
    /// ONCE per super-block pair and amortised across the chunk's members).
    /// Same work-items contract as `launch_fused_swiglu_chunked`; additionally
    /// requires `n_blocks % 2 == 0` and `chunk_size <= MXFP4_KW_MAX_CHUNK`.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_kwide(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        self.launch_fused_swiglu_kwide_ex(
            stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members,
            work_items, n_work_items, gate_bpe, up_bpe, n_used, max_per_expert, chunk_size,
            clamp, n_rows, n_blocks, None,
        )
    }

    /// As `launch_fused_swiglu_kwide`. With `n_work_items_dev` (the builder's
    /// device-side count) `n_work_items` is only an UPPER BOUND for grid.y
    /// (<= `work_items.len()`): work-groups past the device count exit at once,
    /// so the host never reads the count back (`V41_MOE_WI_DEVCOUNT`).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn launch_fused_swiglu_kwide_ex(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 pair kwide: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if n_rows % 8 != 0 {
            return Err(eyre!("mxfp4 pair kwide: n_rows={n_rows} not %8"));
        }
        if n_blocks == 0 || n_blocks % 2 != 0 {
            return Err(eyre!("mxfp4 pair kwide: n_blocks={n_blocks} must be even and > 0"));
        }
        if chunk_size > MXFP4_KW_MAX_CHUNK {
            return Err(eyre!("mxfp4 pair kwide: chunk_size={chunk_size} exceeds MXFP4_KW_MAX_CHUNK={MXFP4_KW_MAX_CHUNK}"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self.module.get_function("mxfp4_pair_matvec_fused_swiglu_kwide")?;
        let cfg = LaunchConfig { grid: (n_rows / 8, n_work_items, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        canary_gate_up(
            function.name(), stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, work_items,
            n_work_items_dev, None, gate_bpe, up_bpe, n_rows, n_blocks, n_work_items, chunk_size, &cfg,
            crate::moe_canary::Grid::RowsByItems { rows_per_wg: 8 }, std::panic::Location::caller(),
        )?;
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks,
                n_wi_dev
            ]
        )
    }

    /// int8-WMMA twin of [`Self::launch_fused_swiglu_kwide_ex`]
    /// (`mxfp4_pair_matvec_fused_swiglu_wmma`, kernels/mxfp4_moe_wmma.hip): the
    /// SAME argument list and work-item / device-count contract, grid
    /// (ceil(n_rows/128), n_work_items) x 256. NOT bit-exact vs kwide (f32
    /// re-association of identical int32 dots, rel_rmse ~5e-7); needs Q8_K `xq`
    /// with bsums (production q8_k_quantize writes them). gfx1151 only.
    /// Selected by `dispatch::moe_gate_up_chunked_rows` (`V41_MOE_WMMA_GATEUP`).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn launch_fused_swiglu_wmma_ex(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        self.launch_fused_swiglu_wmma_arm(
            false, stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, work_items,
            n_work_items, gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks, n_work_items_dev,
        )
    }

    /// [`Self::launch_fused_swiglu_wmma_ex`] on the int8 (`f16 = false`) or the f16-WMMA arm
    /// (kernels/mxfp4_moe_wmma_f16.hip: same symbol, args, grid and 32-member chunk; NOT bit-exact vs the int8 arm,
    /// f32 re-association). Selected by `dispatch::moe_gate_up_chunked_rows` (`V41_MOE_WMMA_F16`).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn launch_fused_swiglu_wmma_arm(
        &self,
        f16: bool,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
    ) -> eyre::Result<()> {
        let module = if f16 { self.wmma_f16.as_ref() } else { self.wmma.as_ref() }
            .ok_or_else(|| eyre!("mxfp4 pair wmma (f16={f16}): no WMMA module on this arch (gfx1151 only)"))?;
        if n_work_items_dev.is_some() && n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 pair wmma: grid bound {n_work_items} > work_items {}", work_items.len()));
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if n_blocks == 0 || n_blocks > MXFP4_PAIR_MAX_BLOCKS {
            return Err(eyre!("mxfp4 pair wmma: n_blocks={n_blocks} outside [1, {MXFP4_PAIR_MAX_BLOCKS}]"));
        }
        if chunk_size == 0 || chunk_size > MXFP4_WMMA_MAX_CHUNK {
            return Err(eyre!("mxfp4 pair wmma: chunk_size={chunk_size} not in 1..={MXFP4_WMMA_MAX_CHUNK} (the kernel would drop members)"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let function = module.get_function("mxfp4_pair_matvec_fused_swiglu_wmma")?;
        let cfg = LaunchConfig {
            grid: (n_rows.div_ceil(MXFP4_WMMA_ROWS_PER_WG), n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        canary_gate_up(
            function.name(), stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, work_items,
            n_work_items_dev, None, gate_bpe, up_bpe, n_rows, n_blocks, n_work_items, chunk_size, &cfg,
            crate::moe_canary::Grid::RowsByItems { rows_per_wg: MXFP4_WMMA_ROWS_PER_WG }, std::panic::Location::caller(),
        )?;
        launch_kernel!(
            function,
            cfg,
            stream,
            [
                mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                group_count.raw(), expert_members.raw(), work_items.raw(),
                gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows, n_blocks,
                n_wi_dev
            ]
        )
    }

    /// Decode-regime twin of [`Self::launch_fused_swiglu_kwide_ex`]
    /// (`mxfp4_pair_matvec_fused_swiglu_dec`, kernels/mxfp4_pair_decode.hip): the
    /// SAME argument list, work-item / device-count contract and outputs,
    /// BIT-IDENTICAL to kwide (tests/mxfp4_pair_decode.rs), plus `rows` = the
    /// pass's token rows (`xq` holds `rows` Q8_K rows; 1..=8). A PERSISTENT grid:
    /// one work-group of 16 waves per CU whatever the work-item count (no dead
    /// work-groups for the device-count bound), its waves taking every (work item,
    /// row) unit -- the units after the first from `work_items`' ticket
    /// (`V41_MOE_PAIR_DEC_DYN`, default on). Needs `n_blocks == 20` and even
    /// `n_rows`; gfx1151 only. 1 launch, like kwide.
    /// Selected by `dispatch::moe_gate_up_chunked_rows` (`V41_MOE_PAIR_DEC`).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn launch_fused_swiglu_dec_ex(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
        rows: u32,
    ) -> eyre::Result<()> {
        let wgs = self.dec_wgs(stream)?;
        let ticket = if dec_dynamic() { self.dec_ticket(work_items) } else { None };
        self.launch_fused_swiglu_dec_raw(
            "mxfp4_pair_matvec_fused_swiglu_dec", wgs, dec_waves(), ticket.as_ref(), None, stream, mid, gate_w_base, up_w_base, xq,
            expert_w, group_count, expert_members, work_items, n_work_items, gate_bpe, up_bpe, n_used,
            max_per_expert, chunk_size, clamp, n_rows, n_blocks, n_work_items_dev, rows,
        )
    }

    /// [`Self::launch_fused_swiglu_dec_ex`] with an explicit entry point, grid,
    /// work-group size and unit order (`ticket`: 2 zeroed u32, the dynamic order;
    /// None = static) (tests / bench: every geometry and order must give the same
    /// bits; `sink` is the extra argument of the bench-only load-only entries).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn launch_fused_swiglu_dec_raw(
        &self,
        symbol: &str,
        wgs: u32,
        waves: u32,
        ticket: Option<&DeviceBuffer<u32>>,
        sink: Option<&mut DeviceBuffer<u32>>,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        group_count: &DeviceBuffer<i32>,
        expert_members: &DeviceBuffer<i32>,
        work_items: &DeviceBuffer<i32>,
        n_work_items: u32,
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items_dev: Option<&DeviceBuffer<i32>>,
        rows: u32,
    ) -> eyre::Result<()> {
        let module = self.dec.as_ref().ok_or_else(|| eyre!("mxfp4 pair dec: no decode module on this arch (gfx1151 only)"))?;
        if n_work_items as usize > work_items.len() {
            return Err(eyre!("mxfp4 pair dec: n_work_items {n_work_items} > work_items {}", work_items.len()));
        }
        if n_blocks != MXFP4_DEC_N_BLOCKS {
            return Err(eyre!("mxfp4 pair dec: n_blocks={n_blocks}, the kernel is compiled for {MXFP4_DEC_N_BLOCKS}"));
        }
        if n_rows == 0 || n_rows % 2 != 0 {
            return Err(eyre!("mxfp4 pair dec: n_rows={n_rows} must be even and > 0"));
        }
        if rows == 0 || rows > MXFP4_DEC_MAX_TOK {
            return Err(eyre!("mxfp4 pair dec: rows={rows} outside 1..={MXFP4_DEC_MAX_TOK} (LDS staging)"));
        }
        let xq_row = n_blocks as usize * crate::q8_k::BLOCK_Q8_K_BYTES;
        if xq.len() < rows as usize * xq_row {
            return Err(eyre!("mxfp4 pair dec: xq {} bytes < rows x {xq_row}", xq.len()));
        }
        if chunk_size == 0 || wgs == 0 || waves == 0 || waves > 16 {
            return Err(eyre!("mxfp4 pair dec: chunk_size={chunk_size} wgs={wgs} waves={waves}"));
        }
        // Per-lane row offsets are 32-bit and stay inside one expert.
        let row_bytes = n_blocks as u64 * 136;
        if n_rows as u64 * row_bytes > gate_bpe.min(up_bpe) as u64 {
            return Err(eyre!("mxfp4 pair dec: n_rows x row bytes exceeds bytes per expert ({gate_bpe}, {up_bpe})"));
        }
        if n_work_items == 0 {
            return Ok(());
        }
        let n_wi_dev = n_work_items_dev.map_or(std::ptr::null_mut(), |c| c.raw());
        if ticket.is_some_and(|t| t.len() < 2) {
            return Err(eyre!("mxfp4 pair dec: the ticket needs 2 u32"));
        }
        let tk = ticket.map_or(std::ptr::null_mut(), |t| t.raw());
        let function = module.get_function(symbol)?;
        let cfg = LaunchConfig { grid: (wgs, 1, 1), block: (waves * 32, 1, 1), shared_mem_bytes: 0 };
        canary_gate_up(
            function.name(), stream, mid, gate_w_base, up_w_base, xq, expert_w, group_count, expert_members, work_items,
            n_work_items_dev, ticket.map(|t| crate::moe_canary::Ptr::of("ticket", t)), gate_bpe, up_bpe, n_rows, n_blocks,
            n_work_items, chunk_size, &cfg, crate::moe_canary::Grid::Persistent, std::panic::Location::caller(),
        )?;
        match sink {
            None => launch_kernel!(
                function,
                cfg,
                stream,
                [
                    mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                    group_count.raw(), expert_members.raw(), work_items.raw(), n_work_items,
                    gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows,
                    n_wi_dev, rows, tk
                ]
            ),
            Some(k) => launch_kernel!(
                function,
                cfg,
                stream,
                [
                    mid.raw(), gate_w_base.raw(), up_w_base.raw(), xq.raw(), expert_w.raw(),
                    group_count.raw(), expert_members.raw(), work_items.raw(), n_work_items,
                    gate_bpe, up_bpe, n_used, max_per_expert, chunk_size, clamp, n_rows,
                    n_wi_dev, rows, tk, k.raw()
                ]
            ),
        }
    }
}
