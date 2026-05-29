//! Grouped Q8_0 matvec oracle — validates `q8_0_grouped_gemv` against
//! ds4's `matvec_q8_0_grouped_rows_decode_scratch` (ds4.c:3618).
//!
//! Input  tag: `attn_heads_inv_rope` (32768 f32 = n_groups=8 × group_dim=4096)
//! Weight:    `blk.{il}.attn_output_a.weight` Q8_0 [4096, 8192]
//! Output tag: `attn_out_low`        (8192 f32 = n_groups=8 × rank=1024)
//!
//! Layer-agnostic — runs for all 43 layers since the grouped projection
//! operates the same regardless of attention variant.
//!
//! Pass: max_abs_diff < 2e-2, mean<1e-4 as regression signal.
//!
//! Threshold rationale: attn_heads_inv_rope values are O(1)–O(5) with
//! occasional spikes (post-softmax + post-inverse-RoPE has wide dynamic
//! range). Q8_0 input quantisation at high-spike (L, T) positions produces
//! per-output noise dominated by the absmax-per-block scale. Mean stays
//! at f32-ULP (~1e-5) across all 18M comparisons — the bulk is correct.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test q8_0_grouped_matvec -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{weights, oracle::ActivationDump, Q8_0GroupedMatvec, Q8_0Matvec};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_LAYER: i32 = 43;
const N_GROUPS: u32 = 8;
const GROUP_DIM: u32 = 4096; // n_head * n_head_dim / n_groups = 64 * 512 / 8
const RANK: u32 = 1024; // DS4_N_LORA_O
const IN_FLAT: u32 = N_GROUPS * GROUP_DIM; // 32768
const OUT_DIM: u32 = N_GROUPS * RANK; // 8192
const BLOCKS_PER_GROUP: u32 = GROUP_DIM / 32; // 128
const BLOCKS_TOTAL: u32 = N_GROUPS * BLOCKS_PER_GROUP; // 1024
const THRESHOLD: f32 = 2.0e-2;

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
fn q8_0_grouped_matvec_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let grouped = Q8_0GroupedMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, IN_FLAT as usize)?;
    let mut d_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, IN_FLAT as usize)?;
    let mut d_xscale: DeviceBuffer<f32> = DeviceBuffer::new(device.id, BLOCKS_TOTAL as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, OUT_DIM as usize)?;
    let mut got = vec![0f32; OUT_DIM as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in 0..N_LAYER {
        let w_out_a = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_output_a.weight"),
            device.id,
        )?;
        // Sanity: shape should be [group_dim=4096, n_groups*rank=8192].
        assert_eq!(w_out_a.shape, vec![GROUP_DIM as u64, OUT_DIM as u64]);

        for token in 0..n_tokens {
            let in_entry = dump
                .tensor("attn_heads_inv_rope", layer, token)
                .ok_or_else(|| eyre!("missing attn_heads_inv_rope at L{layer} T{token}"))?;
            let exp_entry = dump
                .tensor("attn_out_low", layer, token)
                .ok_or_else(|| eyre!("missing attn_out_low at L{layer} T{token}"))?;

            let x_host = dump.read_f32(in_entry)?;
            let expected = dump.read_f32(exp_entry)?;
            assert_eq!(x_host.len(), IN_FLAT as usize);
            assert_eq!(expected.len(), OUT_DIM as usize);

            d_x.copy_from_host(&x_host)?;
            // Per-block quantisation of the flat 32768-element input. Block
            // boundaries align with group boundaries (group_dim % 32 == 0),
            // so this is bit-identical to ds4's per-group quantise loop.
            q8.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_x, IN_FLAT)?;
            grouped.matvec_grouped(
                &stream,
                &mut d_out,
                &w_out_a.buffer,
                &d_xq,
                &d_xscale,
                GROUP_DIM,
                RANK,
                N_GROUPS,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let prev = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }
        }

        drop(w_out_a);
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n={}, worst at L{} T{}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count,
        worst.0,
        worst.1,
    );

    assert!(
        stats.max_abs < THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
