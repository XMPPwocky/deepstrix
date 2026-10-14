//! dGPU fusion group G6 (kernels/g6_boundary.hip, live knob `V41_DGPU_G6_BOUNDARY`, default off): the
//! decode layer BOUNDARY -- box 2's partial upload (pinned H2D), the FFN combine
//! (`hc_post_from_split_batched_add2` / `_add`) and the next layer's `mhc_fast_batched` pre-attn
//! sub-block (mix PRE_SCALED + collapse + RMS + Sinkhorn + carry) -- as ONE launch, bit-identical
//! (tests/g6_boundary_bitexact.rs). Box 2's partial is read ZERO-COPY from its pinned staging, once
//! per element. See the kernel's header for the roles and the forward-progress argument (ticketed
//! workgroups, bounded polls).
//!
//! The kernel keeps a small self-resetting scratch (`sync`, zero between launches): launches that
//! share one [`G6Boundary`] must be STREAM-ORDERED (production: `de.compute` only).

use crate::config::{HC_DIM, HC_MIX_DIM, N_EMBD, N_HC};
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{launch_kernel, DeviceBuffer, LaunchConfig, Module, PinnedBuffer, Stream};

const G6_GFX1201: &[u8] = include_bytes!(env!("KERNEL_G6_BOUNDARY_GFX1201"));
const G6_GFX1151: &[u8] = include_bytes!(env!("KERNEL_G6_BOUNDARY_GFX1151"));

/// Rows per launch (the kernel's `G6_MAXB`).
pub const G6_MAX_B: u32 = 16;
/// Mix rows per mix workgroup of the kernel variant (`g6_boundary_m{MR}`, 1 / 2 / 3; default 2:
/// in-kernel trace 10-14, cold, b = 1 / 2 / 4 / 8 / 16 finish at 8.3 / 8.4 / 9.4 / 12.6 / 19.4 us vs
/// 9.9 / 10.0 / 11.4 / 15.6 / 24.4 at MR 3; MR 1 slower still: 24 x-row reads per row and more WGs).
/// `V41_DGPU_G6_MR` overrides (bench, startup).
pub fn g6_mr() -> u32 {
    static MR: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_DGPU_G6_MR").ok().and_then(|v| v.parse().ok()).filter(|m| [1u32, 2, 3].contains(m)).unwrap_or(2)
    });
    *MR
}
/// Workgroups per row: the lead (combine + collapse) and the 24 / MR mix workgroups (which also
/// carry the 16 RMS slices).
pub fn g6_wgs_per_row(mr: u32) -> u32 {
    1 + HC_MIX_DIM / mr
}
const RMS_SLICES: usize = 16;
const S_WORDS: usize = 64 + 32 * G6_MAX_B as usize + 32; // the kernel's S_WORDS
const S_ERR: usize = 64 + 32 * G6_MAX_B as usize;

/// The kernel's `G6Args`, field for field (pointers first, then 32-bit scalars).
#[repr(C)]
#[derive(Clone, Copy)]
struct G6ArgsRaw {
    resid_out: u64,
    moe: u64,
    shared: u64,
    remote: u64,
    after_attn: u64,
    split_ffn: u64,
    weight: u64,
    scale: u64,
    base: u64,
    split_out: u64,
    mix_out: u64,
    counters: u64,
    inv_rows: u64,
    rms_part: u64,
    carry: u64,
    cur_out: u64,
    norm_out: u64,
    norm_w: u64,
    sync: u64,
    ts: u64,
    b: u32,
    write_carry: u32,
    rms_eps: f32,
    n_hc: u32,
    iters: u32,
    eps: f32,
    dbg: u32,
    remote_sys: u32,
}

/// Where box 2's partial (the combine's second addend) is read from.
pub enum G6Remote<'a> {
    /// ZERO-COPY from its pinned host staging (`&staging`, element offset of row 0): the lead
    /// workgroups read it over PCIe with system-scope loads.
    Pinned(&'a PinnedBuffer<f32>, usize),
    /// Already uploaded to the device (`remote_ffn_moe`, e.g. by the early side-stream copy).
    Device(&'a DeviceBuffer<f32>),
}

/// The FFN combine's operands (`forward_layer_post_moe_v2`'s `hc_post_from_split_batched_add2`).
pub struct G6Combine<'a> {
    /// [b, n_embd] the iGPU MoE result as landed (`ffn_moe_recv`).
    pub moe: &'a DeviceBuffer<f32>,
    /// [b, n_embd] the shared expert (`ffn_shared`, the combine's first addend).
    pub shared: &'a DeviceBuffer<f32>,
    /// Box 2's partial (the second addend); `None` = no box-2 partial this lane-layer (the `_add`
    /// form).
    pub remote: Option<G6Remote<'a>>,
    /// [b, 4 n_embd] after_attn_hc.
    pub after_attn: &'a DeviceBuffer<f32>,
    /// [b, 4 n_embd] the combine's output = the next layer's residual.
    pub resid_out: &'a mut DeviceBuffer<f32>,
}

/// The next layer's pre-attn mHC operands (`MhcArena::launch_fast`'s mix PRE_SCALED + collapse).
pub struct G6Mhc<'a> {
    pub weight: &'a DeviceBuffer<u8>,
    pub scale: &'a DeviceBuffer<f32>,
    pub base: &'a DeviceBuffer<f32>,
    /// [b, 24] `split`: READ by the combine as the FFN split (post / comb), then OVERWRITTEN by the
    /// Sinkhorn tail with the pre-attn split (production: both are `bd.split`; the lead reads it
    /// before its counter increment, the tail writes after the last one).
    pub split: &'a mut DeviceBuffer<f32>,
    pub mix_out: &'a mut DeviceBuffer<f32>,
    pub counters: &'a mut DeviceBuffer<u32>,
    pub inv_rows: &'a mut DeviceBuffer<f32>,
    pub carry: &'a mut DeviceBuffer<f32>,
    pub write_carry: bool,
    pub cur_out: &'a mut DeviceBuffer<f32>,
    pub norm_out: &'a mut DeviceBuffer<f32>,
    pub norm_w: &'a DeviceBuffer<f32>,
    pub rms_eps: f32,
    pub sinkhorn_iters: u32,
    pub sinkhorn_eps: f32,
}

pub struct G6Boundary {
    module: Module,
    sync: DeviceBuffer<u32>,
    /// [16 rows, 16] the RMS slice sums (mix workgroups -> the row's tail), launch-internal.
    rms_part: DeviceBuffer<f32>,
    ts: Option<DeviceBuffer<u64>>,
}

fn need<T>(name: &str, buf: &DeviceBuffer<T>, n: usize) -> eyre::Result<u64> {
    if buf.len() < n {
        return Err(eyre!("g6_boundary: {name} has {} elements, needs {n}", buf.len()));
    }
    Ok(buf.raw() as u64)
}

/// `V41_DGPU_G6_DBG` (bench ablations only; 0 = the real kernel): see the kernel's `G6Args::dbg`.
fn g6_dbg() -> u32 {
    static D: std::sync::LazyLock<u32> =
        std::sync::LazyLock::new(|| std::env::var("V41_DGPU_G6_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
    *D
}

impl G6Boundary {
    pub fn for_arch(arch: &str, device_id: i32) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            G6_GFX1201
        } else if arch.starts_with("gfx1151") {
            G6_GFX1151
        } else {
            return Err(eyre!("unsupported arch for g6_boundary: {arch}"));
        };
        let mut sync = DeviceBuffer::<u32>::new(device_id, S_WORDS)?;
        sync.copy_from_host(&[0u32; S_WORDS])?;
        let ts = if std::env::var_os("V41_DGPU_G6_TS").is_some() {
            Some(DeviceBuffer::<u64>::new(device_id, (G6_MAX_B * g6_wgs_per_row(1)) as usize * 16)?)
        } else {
            None
        };
        let rms_part = DeviceBuffer::<f32>::new(device_id, G6_MAX_B as usize * RMS_SLICES)?;
        Ok(Self { module: Module::load_data(image)?, sync, rms_part, ts })
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

    /// Debug phase timestamps of the last launch (`V41_DGPU_G6_TS`, 100 MHz steady counter):
    /// `[ticket][16]` (0 entry, 1 ticket, 2 lead: ready released / mix: wait over, 3 role done,
    /// 4 finish, 5 row, 6 role 0 lead / 2 mix, 7 the row's tail end and 10 its start (on the WG that
    /// ran it), 8 mix: dots done, 9 mix: first barrier, 11 lead: combine stored).
    /// None when the knob is off. Synchronizes.
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

    /// One boundary launch for `b` rows (1..=16). Grid `(1 + 24 / MR) b` x 256.
    pub fn launch(&self, stream: &Stream, c: G6Combine<'_>, m: G6Mhc<'_>, b: u32) -> eyre::Result<()> {
        if b == 0 {
            return Ok(());
        }
        if b > G6_MAX_B {
            return Err(eyre!("g6_boundary: b = {b} > {G6_MAX_B}"));
        }
        let (bu, ne, k, mix) = (b as usize, N_EMBD as usize, HC_DIM as usize, HC_MIX_DIM as usize);
        if N_HC != 4 || k != 4 * ne {
            return Err(eyre!("g6_boundary: compiled for n_hc 4, K = 4 n_embd"));
        }
        if m.weight.byte_len() != mix * k * 2 {
            return Err(eyre!("g6_boundary: hc weight bytes {} != {}", m.weight.byte_len(), mix * k * 2));
        }
        if m.scale.len() < 3 || m.base.len() < mix || m.norm_w.len() != ne {
            return Err(eyre!("g6_boundary: scale / base / norm_w shape"));
        }
        let (remote, remote_sys) = match c.remote {
            Some(G6Remote::Pinned(pin, off)) => {
                if pin.len() < off + bu * ne {
                    return Err(eyre!("g6_boundary: pinned partial has {} elements, needs {}", pin.len(), off + bu * ne));
                }
                // The address: hipHostMalloc memory is device-accessible (UVA); the caller keeps the
                // staging alive and unwritten until the launch has completed.
                ((pin.device_ptr() as u64) + (off * 4) as u64, 1u32)
            }
            Some(G6Remote::Device(d)) => (need("remote", d, bu * ne)?, 0u32),
            None => (0, 0),
        };
        let a = G6ArgsRaw {
            resid_out: need("resid_out", c.resid_out, bu * k)?,
            moe: need("moe", c.moe, bu * ne)?,
            shared: need("shared", c.shared, bu * ne)?,
            remote,
            after_attn: need("after_attn", c.after_attn, bu * k)?,
            split_ffn: need("split", m.split, bu * mix)?,
            weight: m.weight.raw() as u64,
            scale: m.scale.raw() as u64,
            base: m.base.raw() as u64,
            split_out: m.split.raw() as u64,
            mix_out: need("mix_out", m.mix_out, bu * mix)?,
            counters: need("counters", m.counters, bu)?,
            inv_rows: need("inv_rows", m.inv_rows, bu)?,
            rms_part: self.rms_part.raw() as u64,
            carry: need("carry", m.carry, bu * mix)?,
            cur_out: need("cur_out", m.cur_out, bu * ne)?,
            norm_out: need("norm_out", m.norm_out, bu * ne)?,
            norm_w: m.norm_w.raw() as u64,
            sync: self.sync.raw() as u64,
            ts: self.ts.as_ref().map_or(0, |t| t.raw() as u64),
            b,
            write_carry: u32::from(m.write_carry),
            rms_eps: m.rms_eps,
            n_hc: N_HC,
            iters: m.sinkhorn_iters,
            eps: m.sinkhorn_eps,
            dbg: g6_dbg(),
            remote_sys,
        };
        let mr = g6_mr();
        let cfg = LaunchConfig { grid: (b * g6_wgs_per_row(mr), 1, 1), block: (256, 1, 1), shared_mem_bytes: 0 };
        let function = self.module.get_function(&format!("g6_boundary_m{mr}"))?;
        launch_kernel!(function, cfg, stream, [a])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g6_grid_and_knob() {
        assert_eq!(g6_wgs_per_row(1), 25);
        assert_eq!(g6_wgs_per_row(2), 13);
        assert_eq!(g6_wgs_per_row(3), 9);
        let k = &crate::knobs::DGPU_G6_BOUNDARY;
        assert_eq!(k.kind.parse("off"), Some(0));
        assert_eq!(k.kind.parse("0"), Some(0));
        assert_eq!(k.kind.parse("early"), Some(1));
        assert_eq!(k.kind.parse("1"), Some(1));
        assert_eq!(k.kind.parse("zc"), Some(2));
        assert_eq!(k.kind.parse("bogus"), None);
        assert!(k.live);
    }
}
