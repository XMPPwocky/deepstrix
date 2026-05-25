//! Per-layer state for the het orchestrator (KV cache + compressor state).
//!
//! In M13.1 everything lives on the dGPU (the device that runs attention
//! and reads the KV cache). M13.5 will migrate compressor state to the
//! iGPU and stream `comp_row` back to the dGPU on boundary tokens via a
//! tiny peer push.

use color_eyre::eyre;
use v4flash_hip::{Device, DeviceBuffer, Stream};

use crate::forward::{COMPRESS_RATIOS, N_HEAD_DIM, N_INDEXER_HEAD_DIM, N_LAYER, NEG_INF, SWA_WINDOW};

/// Per-layer compressor state. After M13.5, `state_kv` and `state_score`
/// live on the **iGPU** (where the compressor kernels run), while
/// `comp_kv` stays on the **dGPU** (where `attn_mixed` reads it). On
/// each boundary the iGPU compressor emits a `comp_row` that's
/// peer-pushed to the dGPU's `comp_kv`.
pub struct HetCompressorState {
    /// iGPU-resident: compressor sliding state.
    pub state_kv: DeviceBuffer<f32>,
    pub state_score: DeviceBuffer<f32>,
    /// dGPU-resident: cumulative pooled comp-KV cache consumed by
    /// `attn_mixed`.
    pub comp_kv: DeviceBuffer<f32>,
    pub n_comp: u32,
    pub width: u32,
    pub head_dim: u32,
    /// M40-P1: device-side backup buffers for snapshot/restore on the
    /// spec-decode verify path. Same device + size as state_kv /
    /// state_score / comp_kv. Lazy-allocated on first snapshot call to
    /// avoid paying the memory cost when spec decode is disabled.
    pub state_kv_backup: Option<DeviceBuffer<f32>>,
    pub state_score_backup: Option<DeviceBuffer<f32>>,
    pub comp_kv_backup: Option<DeviceBuffer<f32>>,
}

impl HetCompressorState {
    pub fn alloc(
        igpu_device: Device,
        dgpu_device: Device,
        ratio: u32,
        head_dim: u32,
        n_kv_max: u32,
    ) -> eyre::Result<Self> {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim;
        let state_rows = ratio * coff;
        let n_state = (state_rows as usize) * (width as usize);
        let zeros = vec![0f32; n_state];
        let neg_inf = vec![NEG_INF; n_state];

        // State on iGPU.
        igpu_device.set_current()?;
        let mut state_kv: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        let mut state_score: DeviceBuffer<f32> = DeviceBuffer::new(igpu_device.id, n_state)?;
        state_kv.copy_from_host(&zeros)?;
        state_score.copy_from_host(&neg_inf)?;

        // comp_kv on dGPU.
        dgpu_device.set_current()?;
        let max_n_comp = (n_kv_max + ratio - 1) / ratio;
        let comp_kv_capacity = (max_n_comp as usize) * (head_dim as usize);
        let comp_kv: DeviceBuffer<f32> = DeviceBuffer::new(dgpu_device.id, comp_kv_capacity)?;
        Ok(Self {
            state_kv,
            state_score,
            comp_kv,
            n_comp: 0,
            width,
            head_dim,
            state_kv_backup: None,
            state_score_backup: None,
            comp_kv_backup: None,
        })
    }

    /// M40-P1: snapshot state_kv + state_score to device-side backup
    /// buffers (lazy-allocates on first call). Used by spec_decode to
    /// save state after token0 in a verify pair so that on partial-
    /// reject of token1 we can restore. Caller tracks the counter
    /// (n_comp) separately.
    /// Snapshot state_kv + state_score + comp_kv to device-side backup
    /// buffers, queued ASYNC on the provided streams. Lazy-allocates
    /// backup buffers on first call. `state_stream` must be on the
    /// same device as state_kv/state_score; `comp_kv_stream` on the
    /// same device as comp_kv.
    pub fn snapshot_async(
        &mut self,
        state_stream: &Stream,
        comp_kv_stream: &Stream,
    ) -> eyre::Result<()> {
        let state_device_id = self.state_kv.device_id();
        let comp_kv_device_id = self.comp_kv.device_id();
        let n_state = self.state_kv.len();
        let n_comp_kv = self.comp_kv.len();
        if self.state_kv_backup.is_none() {
            Device::new(state_device_id).set_current()?;
            self.state_kv_backup = Some(DeviceBuffer::new(state_device_id, n_state)?);
            self.state_score_backup = Some(DeviceBuffer::new(state_device_id, n_state)?);
            Device::new(comp_kv_device_id).set_current()?;
            self.comp_kv_backup = Some(DeviceBuffer::new(comp_kv_device_id, n_comp_kv)?);
        }
        let kv_b = self.state_kv_backup.as_mut().unwrap();
        kv_b.copy_from_buffer_async(&self.state_kv, state_stream)?;
        let sc_b = self.state_score_backup.as_mut().unwrap();
        sc_b.copy_from_buffer_async(&self.state_score, state_stream)?;
        let comp_b = self.comp_kv_backup.as_mut().unwrap();
        comp_b.copy_from_buffer_async(&self.comp_kv, comp_kv_stream)?;
        Ok(())
    }

    /// Restore state_kv + state_score + comp_kv from backup buffers,
    /// queued ASYNC on the provided streams. Caller is responsible for
    /// also restoring `n_comp` from its own saved value.
    pub fn restore_async(
        &mut self,
        state_stream: &Stream,
        comp_kv_stream: &Stream,
    ) -> eyre::Result<()> {
        let kv_b = self
            .state_kv_backup
            .take()
            .ok_or_else(|| eyre::eyre!("compressor restore_async: no snapshot taken"))?;
        let sc_b = self
            .state_score_backup
            .take()
            .ok_or_else(|| eyre::eyre!("compressor restore_async: no snapshot taken"))?;
        let comp_b = self
            .comp_kv_backup
            .take()
            .ok_or_else(|| eyre::eyre!("compressor restore_async: no snapshot taken"))?;
        self.state_kv.copy_from_buffer_async(&kv_b, state_stream)?;
        self.state_score
            .copy_from_buffer_async(&sc_b, state_stream)?;
        self.comp_kv.copy_from_buffer_async(&comp_b, comp_kv_stream)?;
        self.state_kv_backup = Some(kv_b);
        self.state_score_backup = Some(sc_b);
        self.comp_kv_backup = Some(comp_b);
        Ok(())
    }
}

pub struct HetLayerState {
    pub kv_cache: DeviceBuffer<f32>,
    pub kv_cache_host: Vec<f32>,
    pub n_raw: u32,
    pub compressor: Option<HetCompressorState>,
    pub indexer_compressor: Option<HetCompressorState>,
    /// M40-P1: per-layer snapshot of (n_raw, n_comp, n_index_comp) at
    /// the moment of `snapshot()`. Compressor state arrays + kv_cache
    /// are snapshotted in their own device-side backup buffers; this
    /// just captures the counters. `None` if no snapshot taken.
    pub snapshot_state: Option<HetLayerSnapshot>,
    /// M40-P1: device-side backup of `kv_cache`. Lazy-allocated.
    pub kv_cache_backup: Option<DeviceBuffer<f32>>,
}

/// M40-P1: per-layer snapshot of mutable counter state. Compressor
/// state arrays are snapshotted in-place via
/// `HetCompressorState::snapshot()` (device-to-device copy to backup
/// buffers).
#[derive(Clone, Copy, Default)]
pub struct HetLayerSnapshot {
    pub n_raw: u32,
    pub n_comp: u32,
    pub n_index_comp: u32,
}

impl HetLayerState {
    /// Snapshot this layer's mutable state (kv_cache, compressor +
    /// indexer state arrays, all counters), queued ASYNC on the
    /// provided streams. Used between token0's and token1's pair-mode
    /// forward_layer calls per layer in spec decode.
    /// `dgpu_stream` must be on dGPU (kv_cache + attn compressor +
    /// comp_kv); `igpu_stream` must be on iGPU (indexer state).
    pub fn snapshot_async(
        &mut self,
        dgpu_stream: &Stream,
        igpu_stream: &Stream,
    ) -> eyre::Result<()> {
        let n_comp = self.compressor.as_ref().map(|c| c.n_comp).unwrap_or(0);
        let n_index_comp = self
            .indexer_compressor
            .as_ref()
            .map(|c| c.n_comp)
            .unwrap_or(0);
        let kv_device_id = self.kv_cache.device_id();
        let kv_len = self.kv_cache.len();
        if self.kv_cache_backup.is_none() {
            Device::new(kv_device_id).set_current()?;
            self.kv_cache_backup = Some(DeviceBuffer::new(kv_device_id, kv_len)?);
        }
        let kv_b = self.kv_cache_backup.as_mut().unwrap();
        kv_b.copy_from_buffer_async(&self.kv_cache, dgpu_stream)?;
        // M14L: attn compressor state lives on dGPU; indexer compressor on iGPU.
        if let Some(comp) = self.compressor.as_mut() {
            comp.snapshot_async(dgpu_stream, dgpu_stream)?;
        }
        if let Some(idx) = self.indexer_compressor.as_mut() {
            idx.snapshot_async(igpu_stream, dgpu_stream)?;
        }
        self.snapshot_state = Some(HetLayerSnapshot {
            n_raw: self.n_raw,
            n_comp,
            n_index_comp,
        });
        Ok(())
    }

    /// Restore this layer's state to the snapshot, queued ASYNC on the
    /// provided streams. Caller must have called `snapshot_async()` first.
    pub fn restore_async(
        &mut self,
        dgpu_stream: &Stream,
        igpu_stream: &Stream,
    ) -> eyre::Result<()> {
        let s = self
            .snapshot_state
            .ok_or_else(|| eyre::eyre!("HetLayerState::restore_async: no snapshot taken"))?;
        self.n_raw = s.n_raw;
        let kv_b = self
            .kv_cache_backup
            .take()
            .ok_or_else(|| eyre::eyre!("HetLayerState::restore_async: no kv_cache backup"))?;
        self.kv_cache.copy_from_buffer_async(&kv_b, dgpu_stream)?;
        self.kv_cache_backup = Some(kv_b);
        if let Some(comp) = self.compressor.as_mut() {
            comp.n_comp = s.n_comp;
            comp.restore_async(dgpu_stream, dgpu_stream)?;
        }
        if let Some(idx) = self.indexer_compressor.as_mut() {
            idx.n_comp = s.n_index_comp;
            idx.restore_async(igpu_stream, dgpu_stream)?;
        }
        Ok(())
    }
}

pub struct HetModelState {
    pub layers: Vec<HetLayerState>,
    pub n_kv_max: u32,
    /// M40-P2: MTP layer's own KV cache + counter. Separate from the
    /// main model's KV (positions advance independently — MTP processes
    /// speculative future tokens that may be rejected). `None` if MTP
    /// is disabled for this session.
    pub mtp: Option<MtpLayerState>,
}

/// M40-P2: per-session MTP layer state. SWA-windowed raw KV cache on
/// dGPU (same shape as a main-model layer's). Spec decode tracks
/// `mtp_n_raw` separately so that on rejection of speculative future
/// tokens the counter can be rolled back without copying cache bytes.
pub struct MtpLayerState {
    pub kv_cache: DeviceBuffer<f32>,
    pub n_raw: u32,
}

impl HetModelState {
    pub fn alloc(dgpu_device: Device, igpu_device: Device, n_kv_max: u32) -> eyre::Result<Self> {
        let mut layers = Vec::with_capacity(N_LAYER as usize);
        for layer in 0..N_LAYER {
            let ratio = COMPRESS_RATIOS[layer as usize];
            let compressor = if ratio > 0 {
                // M14L: attn compressor state moved to dGPU so the compressor
                // can run there alongside attn_input_norm with no peer push.
                Some(HetCompressorState::alloc(
                    dgpu_device,
                    dgpu_device,
                    ratio,
                    N_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            let indexer_compressor = if ratio == 4 {
                Some(HetCompressorState::alloc(
                    igpu_device,
                    dgpu_device,
                    ratio,
                    N_INDEXER_HEAD_DIM,
                    n_kv_max,
                )?)
            } else {
                None
            };
            // SWA caps the raw KV cache at SWA_WINDOW rows regardless
            // of total context length.
            dgpu_device.set_current()?;
            let raw_rows = SWA_WINDOW.max(n_kv_max);
            layers.push(HetLayerState {
                kv_cache: DeviceBuffer::new(
                    dgpu_device.id,
                    (raw_rows as usize) * (N_HEAD_DIM as usize),
                )?,
                kv_cache_host: vec![0f32; (raw_rows as usize) * (N_HEAD_DIM as usize)],
                n_raw: 0,
                compressor,
                indexer_compressor,
                snapshot_state: None,
                kv_cache_backup: None,
            });
        }
        Ok(Self {
            layers,
            n_kv_max,
            mtp: None,
        })
    }

    /// M40-P2: lazily allocate the MTP layer's KV cache on dGPU. Called
    /// by spec-decode setup once MTP is loaded; safe to call multiple
    /// times (no-op if already allocated).
    pub fn alloc_mtp(&mut self, dgpu_device: Device) -> eyre::Result<()> {
        if self.mtp.is_some() {
            return Ok(());
        }
        dgpu_device.set_current()?;
        let raw_rows = SWA_WINDOW.max(self.n_kv_max);
        let kv_cache = DeviceBuffer::new(
            dgpu_device.id,
            (raw_rows as usize) * (N_HEAD_DIM as usize),
        )?;
        self.mtp = Some(MtpLayerState { kv_cache, n_raw: 0 });
        Ok(())
    }

    /// Bulk restore_async — convenience for restoring all per-layer
    /// snapshots at once (e.g., on partial-reject of token1 in a
    /// verify pair). Each layer must have a snapshot taken via
    /// `HetLayerState::snapshot_async()`. Queued on the provided streams.
    pub fn restore_all_async(
        &mut self,
        dgpu_stream: &Stream,
        igpu_stream: &Stream,
    ) -> eyre::Result<()> {
        for layer in self.layers.iter_mut() {
            layer.restore_async(dgpu_stream, igpu_stream)?;
        }
        Ok(())
    }
}
