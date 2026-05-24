//! RoPE oracle test — validates `rope_tail` against ds4's
//! `rope_tail_layer_inplace` (the Q + KV RoPE in the attention setup path).
//!
//! Two stripes per (L, T):
//!   - KV stripe: n_head=1, head_dim=512, n_rot=64. Input `kv_normed`,
//!     expected `kv_post_rope`. (Simplest case; one head.)
//!   - Q stripe: n_head=64, head_dim=512, n_rot=64. Input `q_head_normed`,
//!     expected `q_post_rope`. (Exercises YaRN ramp on compressed layers.)
//!
//! Per-layer RoPE parameters come from the dump's `weight:rope_params`
//! (6 f32). `n_ctx_orig` is layer-invariant — `DS4_ROPE_ORIG_CTX=65536` for
//! compressed layers, 0 otherwise — and ext_factor==0 disambiguates here.
//!
//! Pass: `max_abs_diff < 5e-5` on both stripes. The KV-vs-Q split gives
//! a discriminator: any divergence in only one stripe points at the
//! n_head dimension handling rather than the rope math.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test rope_tail -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{ActivationDump, RopeParams, RopeTail};

const N_LAYER: i32 = 43;
const N_HEAD: u32 = 64;
const N_HEAD_KV: u32 = 1;
const N_HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM;
const KV_FLAT: u32 = N_HEAD_KV * N_HEAD_DIM;
const ROPE_ORIG_CTX: u64 = 65536; // DS4_ROPE_ORIG_CTX
const THRESHOLD: f32 = 5.0e-5;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

#[derive(Default)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}
impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > self.max_abs {
                self.max_abs = d;
            }
            self.sum_abs += d as f64;
            self.count += 1;
        }
    }
    fn mean_abs(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_abs / self.count as f64
        }
    }
}

#[test]
#[ignore]
fn rope_tail_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;
    eprintln!(
        "dump: n_tensors={}, n_tokens={}, n_layers={}",
        dump.len(),
        n_tokens,
        N_LAYER,
    );

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = RopeTail::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_kv: DeviceBuffer<f32> = DeviceBuffer::new(device.id, KV_FLAT as usize)?;
    let mut got_q = vec![0f32; Q_FLAT as usize];
    let mut got_kv = vec![0f32; KV_FLAT as usize];

    let mut q_stats = DiffStats::default();
    let mut kv_stats = DiffStats::default();
    let mut worst_q = (-1i32, -1i32);
    let mut worst_kv = (-1i32, -1i32);
    let mut compressed_count = 0;

    for layer in 0..N_LAYER {
        // Per-layer RoPE params from the dump.
        let rp_entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
        let rp_floats = dump.read_f32(rp_entry)?;
        let ext_factor = rp_floats[2];
        let n_ctx_orig = if ext_factor != 0.0 { ROPE_ORIG_CTX } else { 0 };
        let params = RopeParams::from_dump_blob(&rp_floats, n_ctx_orig)?;
        if params.ext_factor != 0.0 {
            compressed_count += 1;
        }

        for token in 0..n_tokens {
            // KV stripe (n_head=1).
            let kv_in = match dump.tensor("kv_normed", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let kv_exp = dump
                .tensor("kv_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing kv_post_rope at L{layer} T{token}"))?;
            let kv_x = dump.read_f32(kv_in)?;
            let kv_e = dump.read_f32(kv_exp)?;
            assert_eq!(kv_x.len(), KV_FLAT as usize);
            assert_eq!(kv_e.len(), KV_FLAT as usize);

            d_kv.copy_from_host(&kv_x)?;
            kernel.launch_forward(
                &stream,
                &mut d_kv,
                N_HEAD_KV,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &params,
            )?;
            stream.synchronize()?;
            d_kv.copy_to_host(&mut got_kv)?;

            let prev = kv_stats.max_abs;
            kv_stats.update(&got_kv, &kv_e);
            if kv_stats.max_abs > prev {
                worst_kv = (layer, token);
            }

            // Q stripe (n_head=64).
            let q_in = dump
                .tensor("q_head_normed", layer, token)
                .ok_or_else(|| eyre!("missing q_head_normed at L{layer} T{token}"))?;
            let q_exp = dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            let q_x = dump.read_f32(q_in)?;
            let q_e = dump.read_f32(q_exp)?;
            assert_eq!(q_x.len(), Q_FLAT as usize);
            assert_eq!(q_e.len(), Q_FLAT as usize);

            d_q.copy_from_host(&q_x)?;
            kernel.launch_forward(
                &stream,
                &mut d_q,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &params,
            )?;
            stream.synchronize()?;
            d_q.copy_to_host(&mut got_q)?;

            let prev = q_stats.max_abs;
            q_stats.update(&got_q, &q_e);
            if q_stats.max_abs > prev {
                worst_q = (layer, token);
            }
        }
    }

    eprintln!("compressed layers: {compressed_count}/{N_LAYER}");
    eprintln!(
        "KV stripe: max_abs={:.3e}, mean_abs={:.3e}, n={}, worst at L{} T{}",
        kv_stats.max_abs,
        kv_stats.mean_abs(),
        kv_stats.count,
        worst_kv.0,
        worst_kv.1,
    );
    eprintln!(
        "Q  stripe: max_abs={:.3e}, mean_abs={:.3e}, n={}, worst at L{} T{}",
        q_stats.max_abs,
        q_stats.mean_abs(),
        q_stats.count,
        worst_q.0,
        worst_q.1,
    );

    assert!(
        kv_stats.max_abs < THRESHOLD,
        "KV max_abs_diff {:.3e} exceeds threshold {:.3e}",
        kv_stats.max_abs,
        THRESHOLD
    );
    assert!(
        q_stats.max_abs < THRESHOLD,
        "Q max_abs_diff {:.3e} exceeds threshold {:.3e}",
        q_stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
