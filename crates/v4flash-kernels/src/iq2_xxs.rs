//! IQ2_XXS paired matvec: gate[r] and up[r] from two IQ2_XXS weight rows
//! sharing one Q8_K-quantized activation. Mirrors ds4
//! `matvec_iq2_xxs_experts_mid_prequant` inner kernel (ds4.c:3879).

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const IQ2_XXS_PAIR_GFX1201: &[u8] = include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_GFX1201"));
const IQ2_XXS_PAIR_GFX1151: &[u8] = include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_GFX1151"));
const IQ2_XXS_PAIR_PAR_GFX1201: &[u8] =
    include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_PAR_GFX1201"));
const IQ2_XXS_PAIR_PAR_GFX1151: &[u8] =
    include_bytes!(env!("KERNEL_IQ2_XXS_PAIR_MATVEC_PAR_GFX1151"));

pub const BLOCK_IQ2_XXS_BYTES: usize = 66;

pub struct Iq2XxsPairMatvec {
    module: Module,
}

impl Iq2XxsPairMatvec {
    /// Use the M14d parallel variant by default — 2 lanes per super-block,
    /// all 32 warp lanes active (vs 16 in the serial kernel for n_blocks=16).
    /// Numerics match the serial kernel within f32-ULP.
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            IQ2_XXS_PAIR_PAR_GFX1201
        } else if arch.starts_with("gfx1151") {
            IQ2_XXS_PAIR_PAR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for iq2_xxs_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// Original serial kernel — kept for the serial-vs-parallel regression
    /// test in `tests/iq2_xxs_pair.rs`.
    pub fn for_arch_serial(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            IQ2_XXS_PAIR_GFX1201
        } else if arch.starts_with("gfx1151") {
            IQ2_XXS_PAIR_GFX1151
        } else {
            return Err(eyre!("unsupported arch for iq2_xxs_pair_matvec: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    /// `gate[r] = sum_b dot_iq2xxs(gate_w[r,b], xq[b])` and analogous for `up`,
    /// for `r in 0..n_rows`. Each weight row is `n_blocks` × 66 B blocks; the
    /// activation `xq` is `n_blocks` × 292 B Q8_K blocks. `n_rows` must be a
    /// multiple of 8.
    pub fn launch(
        &self,
        stream: &Stream,
        gate: &mut DeviceBuffer<f32>,
        up: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        up_w: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        self.launch_with_offsets(stream, gate, up, gate_w, 0, up_w, 0, xq, n_rows, n_blocks)
    }

    /// Same as [`launch`] but reads the gate/up weights starting at a
    /// per-buffer byte offset. Used by the forward orchestrator to point
    /// into a single resident routed-expert tensor (256 experts of
    /// `n_rows * n_blocks * BLOCK_IQ2_XXS_BYTES` bytes each) without
    /// allocating per-slot scratch buffers.
    pub fn launch_with_offsets(
        &self,
        stream: &Stream,
        gate: &mut DeviceBuffer<f32>,
        up: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        gate_w_offset: usize,
        up_w: &DeviceBuffer<u8>,
        up_w_offset: usize,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_xxs_pair: n_rows={n_rows} must be multiple of 8"));
        }
        let row_bytes = (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
        let need = (n_rows as usize) * row_bytes;
        if gate_w.byte_len() < gate_w_offset + need {
            return Err(eyre!(
                "gate_w bytes: have {}, need offset {} + {} = {}",
                gate_w.byte_len(),
                gate_w_offset,
                need,
                gate_w_offset + need
            ));
        }
        if up_w.byte_len() < up_w_offset + need {
            return Err(eyre!(
                "up_w bytes: have {}, need offset {} + {} = {}",
                up_w.byte_len(),
                up_w_offset,
                need,
                up_w_offset + need
            ));
        }

        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_par")
            .or_else(|_| self.module.get_function("iq2_xxs_pair_matvec"))?;
        let mut g_ptr = gate.raw();
        let mut u_ptr = up.raw();
        // SAFETY: bounds-checked above; pointer math within the allocation.
        let mut gw_ptr = unsafe { (gate_w.raw() as *mut u8).add(gate_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut uw_ptr = unsafe { (up_w.raw() as *mut u8).add(up_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut xq_ptr = xq.raw();
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 7] = [
            &mut g_ptr as *mut _ as *mut c_void,
            &mut u_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Full-offset variant: also offsets the gate/up output buffers.
    /// Lets the MoE pipeline write each per-slot result directly into a
    /// concatenated `[N_USED, N_FF_EXP]` buffer with zero host sync —
    /// eliminating the per-slot copy_to_host / copy_from_host roundtrip
    /// in the routed-MoE inner loop.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_with_full_offsets(
        &self,
        stream: &Stream,
        gate: &mut DeviceBuffer<f32>,
        gate_offset_elems: usize,
        up: &mut DeviceBuffer<f32>,
        up_offset_elems: usize,
        gate_w: &DeviceBuffer<u8>,
        gate_w_offset: usize,
        up_w: &DeviceBuffer<u8>,
        up_w_offset: usize,
        xq: &DeviceBuffer<u8>,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_xxs_pair: n_rows={n_rows} must be multiple of 8"));
        }
        let row_bytes = (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
        let need = (n_rows as usize) * row_bytes;
        if gate_w.byte_len() < gate_w_offset + need {
            return Err(eyre!(
                "gate_w bytes: have {}, need offset {} + {} = {}",
                gate_w.byte_len(),
                gate_w_offset,
                need,
                gate_w_offset + need
            ));
        }
        if up_w.byte_len() < up_w_offset + need {
            return Err(eyre!(
                "up_w bytes: have {}, need offset {} + {} = {}",
                up_w.byte_len(),
                up_w_offset,
                need,
                up_w_offset + need
            ));
        }
        if gate.len() < gate_offset_elems + n_rows as usize {
            return Err(eyre!(
                "gate out: len {} < offset {} + n_rows {}",
                gate.len(),
                gate_offset_elems,
                n_rows
            ));
        }
        if up.len() < up_offset_elems + n_rows as usize {
            return Err(eyre!(
                "up out: len {} < offset {} + n_rows {}",
                up.len(),
                up_offset_elems,
                n_rows
            ));
        }

        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_par")
            .or_else(|_| self.module.get_function("iq2_xxs_pair_matvec"))?;
        let mut g_ptr = unsafe { (gate.raw() as *mut f32).add(gate_offset_elems) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut u_ptr = unsafe { (up.raw() as *mut f32).add(up_offset_elems) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut gw_ptr = unsafe { (gate_w.raw() as *mut u8).add(gate_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut uw_ptr = unsafe { (up_w.raw() as *mut u8).add(up_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut xq_ptr = xq.raw();
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 7] = [
            &mut g_ptr as *mut _ as *mut c_void,
            &mut u_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Fused variant: iq2_pair matvec → silu_clamp(g) * clamp(u) * expert_w[slot]
    /// → writes directly to `mid[slot * n_rows + r]`. Skips the gate_cat /
    /// up_cat staging buffers and the standalone `swiglu_clamp_weighted` launch
    /// (M14e). One call per slot; expect `clamp <= 1e-6` to disable clamping.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,
        gate_w: &DeviceBuffer<u8>,
        gate_w_offset: usize,
        up_w: &DeviceBuffer<u8>,
        up_w_offset: usize,
        xq: &DeviceBuffer<u8>,
        expert_w: &DeviceBuffer<f32>,
        slot: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!("iq2_xxs_pair_fused: n_rows={n_rows} must be multiple of 8"));
        }
        let row_bytes = (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
        let need = (n_rows as usize) * row_bytes;
        if gate_w.byte_len() < gate_w_offset + need {
            return Err(eyre!(
                "gate_w bytes: have {}, need offset {} + {} = {}",
                gate_w.byte_len(),
                gate_w_offset,
                need,
                gate_w_offset + need
            ));
        }
        if up_w.byte_len() < up_w_offset + need {
            return Err(eyre!(
                "up_w bytes: have {}, need offset {} + {} = {}",
                up_w.byte_len(),
                up_w_offset,
                need,
                up_w_offset + need
            ));
        }
        let mid_off = (slot as usize) * (n_rows as usize);
        if mid.len() < mid_off + n_rows as usize {
            return Err(eyre!(
                "mid out: len {} < offset {} + n_rows {}",
                mid.len(),
                mid_off,
                n_rows
            ));
        }
        if (expert_w.len() as u32) <= slot {
            return Err(eyre!(
                "expert_w len {} <= slot {slot}",
                expert_w.len()
            ));
        }

        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = unsafe { (gate_w.raw() as *mut u8).add(gate_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut uw_ptr = unsafe { (up_w.raw() as *mut u8).add(up_w_offset) }
            as v4flash_hip::sys::hipDeviceptr_t;
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut slot_v = slot;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 9] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut slot_v as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, 1, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Batched fused variant: single launch handles all `n_used` slots
    /// using `selected` (device-side i32 buffer of expert indices) for
    /// routing. `gate_bpe` and `up_bpe` are byte strides per expert in
    /// the full weight tensors. Output written to
    /// `mid[slot * n_rows + r]`. (M14j — eliminates 5 launches/layer.)
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
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "iq2_xxs_pair_fused_batch: n_rows={n_rows} must be multiple of 8"
            ));
        }
        if mid.len() < (n_used as usize) * (n_rows as usize) {
            return Err(eyre!(
                "mid out: len {} < n_used {} * n_rows {} = {}",
                mid.len(),
                n_used,
                n_rows,
                (n_used as usize) * (n_rows as usize)
            ));
        }
        if (selected.len() as u32) < n_used {
            return Err(eyre!(
                "selected len {} < n_used {n_used}",
                selected.len()
            ));
        }
        if (expert_w.len() as u32) < n_used {
            return Err(eyre!(
                "expert_w len {} < n_used {n_used}",
                expert_w.len()
            ));
        }

        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_batch")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut sel_ptr = selected.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 11] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut sel_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_used, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 3 v0: batched variant — grid.z = B. Per-batch xq, selected,
    /// expert_w, mid. Weight shared across batch.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_batch_bxn(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,        // [B, n_used, n_rows]
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,              // [B, n_blocks * 292]
        expert_w: &DeviceBuffer<f32>,       // [B, n_used]
        selected: &DeviceBuffer<i32>,       // [B, n_used]
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        batch: u32,
    ) -> eyre::Result<()> {
        if batch == 0 {
            return Ok(());
        }
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "iq2_xxs_pair_fused_batch_bxn: n_rows={n_rows} must be multiple of 8"
            ));
        }
        let needed = (batch as usize) * (n_used as usize) * (n_rows as usize);
        if mid.len() < needed {
            return Err(eyre!(
                "mid out len {} < B*n_used*n_rows = {needed}",
                mid.len()
            ));
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_batch_BxN")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut sel_ptr = selected.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 12] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut sel_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_used, batch),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 7 by-expert iq2 fused-swiglu. Grid (n_rows/8, n_expert, 1).
    /// Each WG handles one (row_block, expert) pair; iterates over all
    /// (token, slot) members that picked this expert (built by the
    /// `moe_group_builder` pre-pass kernel). Expected ~6× weight-BW
    /// amortization at B=64 vs the by-token batched variant, based on
    /// measured per-expert reuse stats.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_by_expert(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,         // [B, n_used, n_rows]
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,               // [B, n_blocks*292]
        expert_w: &DeviceBuffer<f32>,        // [B, n_used]
        group_count: &DeviceBuffer<i32>,     // [n_expert]
        expert_members: &DeviceBuffer<i32>,  // [n_expert * max_per_expert]
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        n_expert: u32,
        max_per_expert: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "iq2_xxs by_expert: n_rows={n_rows} must be multiple of 8"
            ));
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_by_expert")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 14] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_expert, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// M50 Phase 7.2 chunked-static by-expert iq2 dispatch. Solves the
    /// popular-expert tail latency by splitting popular groups into
    /// CHUNK_SIZE-bounded work items, each handled by its own WG.
    /// Combined with the per-expert weight reuse from by-expert dispatch,
    /// this captures the ~6× BW savings vs by-token at large B without
    /// the wave-quantization tail.
    ///
    /// Caller must populate group_count + expert_members (via
    /// moe_group_builder) and work_items + n_work_items (via
    /// moe_work_items_builder), then sync to read n_work_items[0] for
    /// the grid.y dimension.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked(
        &self,
        stream: &Stream,
        mid: &mut DeviceBuffer<f32>,           // [B, n_used, n_rows]
        gate_w_base: &DeviceBuffer<u8>,
        up_w_base: &DeviceBuffer<u8>,
        xq: &DeviceBuffer<u8>,                 // [B, n_blocks*292]
        expert_w: &DeviceBuffer<f32>,          // [B, n_used]
        group_count: &DeviceBuffer<i32>,       // [n_expert]
        expert_members: &DeviceBuffer<i32>,    // [n_expert * max_per_expert]
        work_items: &DeviceBuffer<i32>,        // [n_work_items]
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32, // grid.y — host-known after sync on n_work_items_buf
    ) -> eyre::Result<()> {
        if n_rows % 8 != 0 {
            return Err(eyre!(
                "iq2_xxs chunked: n_rows={n_rows} must be multiple of 8"
            ));
        }
        if n_work_items == 0 {
            return Ok(()); // no expert was picked? nothing to do
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Phase 7.4: chunked + inline-sign-math (replaces s_sign_pair LDS lookup
    /// with VALU bit ops: `i ^ ((popcount(i) & 1) << 7) → expand_signs`).
    /// Same signature as `launch_fused_swiglu_chunked`. Goal: cut ~half of
    /// the data-dependent LDS reads to reduce LDSBankConflict + MemUnitBusy.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked_inline(
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
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32,
    ) -> eyre::Result<()> {
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked_inline")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Phase 7.3: chunked + per-WG LDS-staged weights. Identical signature
    /// to `launch_fused_swiglu_chunked`. The per-WG cooperative LDS-stage
    /// of iq2 weights cuts global weight reads by a factor of ~B (one
    /// expert's weights read once per chunk instead of once per member).
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked_lds(
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
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32,
    ) -> eyre::Result<()> {
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked_lds")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Phase 7.5: padded-LDS variant. Pads s_grid and s_sign_pair from
    /// 8-byte stride to 12-byte (8 useful + 4 padding) so the stride is
    /// coprime with the 32-bank LDS, doubling distinct bank classes
    /// (16 → 32) and roughly halving expected bank-conflict serialization.
    /// Expected ~14% iq2 kernel wall improvement.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked_padded(
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
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32,
    ) -> eyre::Result<()> {
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked_padded")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Phase 7.5 DIAGNOSTIC: same as chunked but all LDS lookup indices
    /// forced to 0. Same LDS read count, but broadcast-access → no bank
    /// conflicts. Result is numerically wrong; ONLY for timing diagnostic.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked_zeroidx(
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
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32,
    ) -> eyre::Result<()> {
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked_zeroidx")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }

    /// Phase 7.2 diagnostic: chunked kernel with dot-product loop stubbed.
    /// Same signature & args as `launch_fused_swiglu_chunked`. Use to
    /// isolate dot-product wall vs LDS-init + xq-load + reduce overhead.
    #[allow(clippy::too_many_arguments)]
    pub fn launch_fused_swiglu_chunked_nodot(
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
        gate_bpe: u32,
        up_bpe: u32,
        n_used: u32,
        max_per_expert: u32,
        chunk_size: u32,
        clamp: f32,
        n_rows: u32,
        n_blocks: u32,
        n_work_items: u32,
    ) -> eyre::Result<()> {
        if n_work_items == 0 {
            return Ok(());
        }
        let function = self
            .module
            .get_function("iq2_xxs_pair_matvec_fused_swiglu_chunked_NODOT")?;
        let mut mid_ptr = mid.raw();
        let mut gw_ptr = gate_w_base.raw();
        let mut uw_ptr = up_w_base.raw();
        let mut xq_ptr = xq.raw();
        let mut ew_ptr = expert_w.raw();
        let mut gc_ptr = group_count.raw();
        let mut em_ptr = expert_members.raw();
        let mut wi_ptr = work_items.raw();
        let mut gbpe = gate_bpe;
        let mut ubpe = up_bpe;
        let mut nu = n_used;
        let mut mpe = max_per_expert;
        let mut cs = chunk_size;
        let mut clamp_v = clamp;
        let mut nr = n_rows;
        let mut nb = n_blocks;
        let mut args: [*mut c_void; 16] = [
            &mut mid_ptr as *mut _ as *mut c_void,
            &mut gw_ptr as *mut _ as *mut c_void,
            &mut uw_ptr as *mut _ as *mut c_void,
            &mut xq_ptr as *mut _ as *mut c_void,
            &mut ew_ptr as *mut _ as *mut c_void,
            &mut gc_ptr as *mut _ as *mut c_void,
            &mut em_ptr as *mut _ as *mut c_void,
            &mut wi_ptr as *mut _ as *mut c_void,
            &mut gbpe as *mut _ as *mut c_void,
            &mut ubpe as *mut _ as *mut c_void,
            &mut nu as *mut _ as *mut c_void,
            &mut mpe as *mut _ as *mut c_void,
            &mut cs as *mut _ as *mut c_void,
            &mut clamp_v as *mut _ as *mut c_void,
            &mut nr as *mut _ as *mut c_void,
            &mut nb as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (n_rows / 8, n_work_items, 1),
            block: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
