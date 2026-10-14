//! dGPU fusion group G4 (kernels/g4_shared.hip, live knob `V41_DGPU_G4_SHARED`, default off): the
//! decode shared expert's input quantize + the late FFN mixes (`mhc_fast_batched`, mix NORMED +
//! carry := split) as ONE latency-bound launch, `g4_qmix`, every output bit-identical
//! (tests/g4_shared_bitexact.rs). The row RMS the 24 mix workgroups each recomputed runs once per row
//! (ticketed: the mix workgroups wait for it); the shared expert's GEMVs (at the byte roof) keep their
//! launches. See the kernel's header for the roles, the forward-progress argument and the two
//! measured dead ends.
//!
//! The kernel keeps a small self-resetting scratch (`sync`, zero between launches): launches that
//! share one [`G4Shared`] must be STREAM-ORDERED (production: `de.compute` only).

use crate::config::{HC_DIM, HC_MIX_DIM, N_HC};
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, Stream};

const G4_GFX1201: &[u8] = include_bytes!(env!("KERNEL_G4_SHARED_GFX1201"));
const G4_GFX1151: &[u8] = include_bytes!(env!("KERNEL_G4_SHARED_GFX1151"));

/// Rows per launch (the kernel's `G4_MAXB`).
pub const G4_MAX_B: u32 = 16;
const S_WORDS: usize = 64 + 32 * G4_MAX_B as usize + 32;
const S_ERR: usize = 64 + 32 * G4_MAX_B as usize;

/// The kernel's `G4Args`, field for field.
#[repr(C)]
#[derive(Clone, Copy)]
struct G4ArgsRaw {
    xq: u64,
    xscale: u64,
    xin: u64,
    split_out: u64,
    mix_out: u64,
    counters: u64,
    weight: u64,
    x: u64,
    scale: u64,
    base: u64,
    carry: u64,
    inv_scr: u64,
    sync: u64,
    ts: u64,
    b: u32,
    qblocks: u32,
    nq: u32,
    write_carry: u32,
    rms_eps: f32,
    n_hc: u32,
    iters: u32,
    eps: f32,
}

/// The shared expert's input quantize (`Q8_0Matvec::quantize_input_batched` of `[b, n]`).
pub struct G4Quant<'a> {
    pub xq: &'a mut DeviceBuffer<i8>,
    pub xscale: &'a mut DeviceBuffer<f32>,
    /// [b, n] f32 (`ffn_input_norm`).
    pub x: &'a DeviceBuffer<f32>,
    pub n: u32,
}

/// The late FFN mixes' operands (`MhcArena::launch_fast` with `FastMix { mode: MIX_NORMED }`, no
/// collapse).
pub struct G4Mix<'a> {
    /// [24, HC_DIM] f16 hc_ffn_fn.
    pub weight: &'a DeviceBuffer<u8>,
    /// [b, HC_DIM] after_attn_hc.
    pub x: &'a DeviceBuffer<f32>,
    pub scale: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    pub split_out: &'a mut DeviceBuffer<f32>,
    pub mix_out: &'a mut DeviceBuffer<f32>,
    pub counters: &'a mut DeviceBuffer<u32>,
    pub carry: &'a mut DeviceBuffer<f32>,
    pub write_carry: bool,
    pub rms_eps: f32,
    pub sinkhorn_iters: u32,
    pub sinkhorn_eps: f32,
}

pub struct G4Shared {
    module: Module,
    sync: DeviceBuffer<u32>,
    inv_scr: DeviceBuffer<f32>,
    ts: Option<DeviceBuffer<u64>>,
}

fn need<T>(name: &str, buf: &DeviceBuffer<T>, n: usize) -> eyre::Result<u64> {
    if buf.len() < n {
        return Err(eyre!("g4_shared: {name} has {} elements, needs {n}", buf.len()));
    }
    Ok(buf.raw() as u64)
}

impl G4Shared {
    pub fn for_arch(arch: &str, device_id: i32) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            G4_GFX1201
        } else if arch.starts_with("gfx1151") {
            G4_GFX1151
        } else {
            return Err(eyre!("unsupported arch for g4_shared: {arch}"));
        };
        let mut sync = DeviceBuffer::<u32>::new(device_id, S_WORDS)?;
        sync.copy_from_host(&[0u32; S_WORDS])?;
        let inv_scr = DeviceBuffer::<f32>::new(device_id, G4_MAX_B as usize)?;
        let ts = if std::env::var_os("V41_DGPU_G4_TS").is_some() {
            Some(DeviceBuffer::<u64>::new(device_id, 4096 * 8)?)
        } else {
            None
        };
        Ok(Self { module: Module::load_data(image)?, sync, inv_scr, ts })
    }

    /// The sticky error word (a bounded poll expired; 0 = never). Synchronizes.
    pub fn error_word(&self) -> eyre::Result<u32> {
        let mut v = vec![0u32; S_WORDS];
        self.sync.copy_to_host(&mut v)?;
        Ok(v[S_ERR])
    }

    /// The whole sync scratch (tests: every word but S_ERR is zero between launches). Synchronizes.
    pub fn sync_words(&self) -> eyre::Result<Vec<u32>> {
        let mut v = vec![0u32; S_WORDS];
        self.sync.copy_to_host(&mut v)?;
        Ok(v)
    }

    /// Debug phase timestamps of the last launch (`V41_DGPU_G4_TS`): `[ticket][8]` (0 entry, 1 ticket,
    /// 2 RMS released / quant done / mix wait over, 3 mix done, 4 finish, 5 the row's tail end on the
    /// WG that ran it, 6 role 0 RMS / 1 quant / 2 mix). Synchronizes.
    pub fn ts_dump(&self) -> eyre::Result<Option<Vec<u64>>> {
        match self.ts.as_ref() {
            None => Ok(None),
            Some(t) => {
                let mut v = vec![0u64; t.len()];
                t.copy_to_host(&mut v)?;
                Ok(Some(v))
            }
        }
    }

    /// Quantize workgroups of a launch (8 Q8_0 blocks each, as `quantize_input_batched`).
    pub fn quant_wgs(b: u32, n: u32) -> u32 {
        (b * n / 32).div_ceil(8)
    }

    /// One `g4_qmix` launch for `b` rows (1..=16): the quantize of `q.x` and the late mixes `m`.
    /// Grid `b + ceil(b n / 256) + 24 b` x 256.
    pub fn launch_qmix(&self, stream: &Stream, q: G4Quant<'_>, m: G4Mix<'_>, b: u32) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if b > G4_MAX_B {
            return Err(eyre!("g4_qmix: b = {b} > {G4_MAX_B}"));
        }
        if N_HC != 4 || HC_MIX_DIM != 24 || q.n % 32 != 0 {
            return Err(eyre!("g4_qmix: compiled for n_hc 4; n = {} must be a multiple of 32", q.n));
        }
        let (bu, kd, mm) = (b as usize, HC_DIM as usize, HC_MIX_DIM as usize);
        if m.weight.byte_len() != mm * kd * 2 || m.scale.len() < 3 || m.base.len() < mm {
            return Err(eyre!("g4_qmix: mix weight / scale / base shape"));
        }
        let qblocks = b * q.n / 32;
        let nq = Self::quant_wgs(b, q.n);
        let a = G4ArgsRaw {
            xq: need("xq", q.xq, bu * q.n as usize)?,
            xscale: need("xscale", q.xscale, qblocks as usize)?,
            xin: need("x", q.x, bu * q.n as usize)?,
            split_out: need("split", m.split_out, bu * mm)?,
            mix_out: need("mix", m.mix_out, bu * mm)?,
            counters: need("counters", m.counters, bu)?,
            weight: m.weight.raw() as u64,
            x: need("mix x", m.x, bu * kd)?,
            scale: m.scale.raw() as u64,
            base: m.base.raw() as u64,
            carry: need("carry", m.carry, bu * mm)?,
            inv_scr: self.inv_scr.raw() as u64,
            sync: self.sync.raw() as u64,
            ts: self.ts.as_ref().map_or(0, |t| t.raw() as u64),
            b,
            qblocks,
            nq,
            write_carry: u32::from(m.write_carry),
            rms_eps: m.rms_eps,
            n_hc: N_HC,
            iters: m.sinkhorn_iters,
            eps: m.sinkhorn_eps,
        };
        let cfg = LaunchConfig { grid: (b * (1 + HC_MIX_DIM) + nq, 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        let function = self.module.get_function("g4_qmix")?;
        launch_kernel!(function, cfg, stream, [a])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g4_grid_and_knob() {
        // production's quantize grid: blocks.div_ceil(8) WGs of 8 blocks
        assert_eq!(G4Shared::quant_wgs(1, 5120), 20);
        assert_eq!(G4Shared::quant_wgs(4, 5120), 80);
        assert_eq!(G4Shared::quant_wgs(3, 64), 1);
        assert!(crate::knobs::DGPU_G4_SHARED.live);
        assert_eq!(crate::knobs::DGPU_G4_SHARED.kind.parse("1"), Some(1));
    }
}
