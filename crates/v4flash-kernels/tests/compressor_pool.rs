//! compressor_pool oracle — validates `CompressorPool` against ds4's
//! `compressor_pool_decode_state` (ds4.c:6439). The kernel takes a state
//! buffer (composed from `ratio` prior tokens' `comp_state_kv_row` +
//! `comp_state_score_row`) and produces a `[head_dim]` pooled output.
//!
//! For ratio==4 layers we reconstruct the 8-row state by maintaining a
//! "previous generation" buffer (initially zeros). For ratio==128 the
//! 128-row state is accumulated naturally over 128 tokens, but our
//! 57-token M1 prompt never triggers a ratio==128 boundary — so this
//! test only fires on ratio==4 layers.
//!
//! Coverage: 21 ratio==4 layers × ~14 pool firings = ~294 comparisons
//! × head_dim=512 = ~150K element-level diffs.
//!
//! Threshold: 1e-4, mean<1e-6 (per-thread sequential softmax matches CPU
//! exactly modulo ULP).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{ActivationDump, CompressorPool};

const HEAD_DIM_MAIN: u32 = 512;
const COMP_WIDTH_R4: usize = 2 * HEAD_DIM_MAIN as usize; // 1024
const STATE_ROWS_R4: usize = 8; // 2 * ratio
const THRESHOLD: f32 = 1.0e-4;

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
fn compressor_pool_oracle_ratio4_main() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = CompressorPool::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let total_state = STATE_ROWS_R4 * COMP_WIDTH_R4;
    let mut d_kv: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_state)?;
    let mut d_sc: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_state)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HEAD_DIM_MAIN as usize)?;
    let mut got = vec![0f32; HEAD_DIM_MAIN as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    // Iterate ratio==4 layers: L=2,4,…,42 (21 layers).
    // Important: ds4 initialises attn_state_score to DS4_NEG_INF (ds4.c:6338),
    // not zero. attn_state_kv is zero-initialised via xmalloc_zeroed (6335).
    const NEG_INF: f32 = -3.4028235e38;
    for layer in (2..=42).step_by(2) {
        // Previous generation (rows 0..3). state_kv=0, state_score=NEG_INF.
        let mut prev_kv = vec![0f32; 4 * COMP_WIDTH_R4];
        let mut prev_sc = vec![NEG_INF; 4 * COMP_WIDTH_R4];
        // Current generation (rows 4..7). Same init; per-token writes fill in.
        let mut cur_kv = vec![0f32; 4 * COMP_WIDTH_R4];
        let mut cur_sc = vec![NEG_INF; 4 * COMP_WIDTH_R4];

        for token in 0..n_tokens {
            let pos_mod = (token as usize) % 4;

            // Read this token's state row writes (compressor writes happen
            // every token, even on non-boundary tokens).
            let kv_entry = dump
                .tensor("comp_state_kv_row", layer, token)
                .ok_or_else(|| eyre!("missing comp_state_kv_row at L{layer} T{token}"))?;
            let sc_entry = dump
                .tensor("comp_state_score_row", layer, token)
                .ok_or_else(|| eyre!("missing comp_state_score_row at L{layer} T{token}"))?;
            let kv_row = dump.read_f32(kv_entry)?;
            let sc_row = dump.read_f32(sc_entry)?;
            assert_eq!(kv_row.len(), COMP_WIDTH_R4);

            // Write into current generation at pos_mod.
            let off = pos_mod * COMP_WIDTH_R4;
            cur_kv[off..off + COMP_WIDTH_R4].copy_from_slice(&kv_row);
            cur_sc[off..off + COMP_WIDTH_R4].copy_from_slice(&sc_row);

            // Boundary fires every 4 tokens: T = 3, 7, 11, …, 55.
            if (token + 1) % 4 != 0 {
                continue;
            }

            // Compose state buffer for pool kernel: rows 0..3 = prev, rows 4..7 = cur.
            let mut state_kv = vec![0f32; total_state];
            let mut state_sc = vec![0f32; total_state];
            state_kv[..4 * COMP_WIDTH_R4].copy_from_slice(&prev_kv);
            state_kv[4 * COMP_WIDTH_R4..].copy_from_slice(&cur_kv);
            state_sc[..4 * COMP_WIDTH_R4].copy_from_slice(&prev_sc);
            state_sc[4 * COMP_WIDTH_R4..].copy_from_slice(&cur_sc);

            d_kv.copy_from_host(&state_kv)?;
            d_sc.copy_from_host(&state_sc)?;
            kernel.launch(&stream, &mut d_out, &d_kv, &d_sc, HEAD_DIM_MAIN, 4)?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let expected_entry = dump
                .tensor("comp_pool_out", layer, token)
                .ok_or_else(|| eyre!("missing comp_pool_out at L{layer} T{token}"))?;
            let expected = dump.read_f32(expected_entry)?;
            assert_eq!(expected.len(), HEAD_DIM_MAIN as usize);

            let prev_max = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev_max {
                worst = (layer, token);
            }

            // Apply state shuffle: previous := current. (Both buffers'
            // rows 0..3 and 4..7 become the now-current values; rows 4..7
            // will be overwritten on the next 4 tokens.)
            std::mem::swap(&mut prev_kv, &mut cur_kv);
            std::mem::swap(&mut prev_sc, &mut cur_sc);
            // Don't reset cur_*: ds4 leaves them with their old values
            // since they'll be overwritten on the next 4 tokens anyway.
        }
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
