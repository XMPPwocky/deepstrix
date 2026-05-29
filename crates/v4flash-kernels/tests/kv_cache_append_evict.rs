//! `kv_cache_append_evict_gather` oracle — validates the one-shot eviction
//! gather against a host model of the serial per-token slide loop.
//!
//! The serial loop (`kv_cache_append`, append-or-evict-oldest) is the ground
//! truth: prefill used to call it B times per chunk. The gather reproduces its
//! final ring state in a single launch for the eviction regime
//! (`r0 + b > swa_window`), INCLUDING partial eviction (`b < swa_window`),
//! which is the case the old `..batched_evict` kernel underflowed on.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test kv_cache_append_evict -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::KvCacheAppend;

fn pick_device() -> eyre::Result<Device> {
    Device::all()?
        .first()
        .copied()
        .ok_or_else(|| eyre!("no HIP devices"))
}

/// Host model of the serial slide loop: start with `r0` rows already in a
/// `w`-row ring, then append `b` new rows one at a time, evicting the oldest
/// when full. Returns the final `w * head_dim` ring contents.
fn serial_reference(r0: usize, b: usize, w: usize, head_dim: usize, new: &[f32]) -> Vec<f32> {
    let mut cache = vec![0f32; w * head_dim];
    // Seed old rows with their global row id (matches the device-side seed).
    for row in 0..r0 {
        for d in 0..head_dim {
            cache[row * head_dim + d] = row as f32 + (d as f32) * 0.001;
        }
    }
    let mut fill = r0;
    for i in 0..b {
        if fill < w {
            for d in 0..head_dim {
                cache[fill * head_dim + d] = new[i * head_dim + d];
            }
            fill += 1;
        } else {
            for s in 0..w - 1 {
                for d in 0..head_dim {
                    cache[s * head_dim + d] = cache[(s + 1) * head_dim + d];
                }
            }
            for d in 0..head_dim {
                cache[(w - 1) * head_dim + d] = new[i * head_dim + d];
            }
        }
    }
    cache
}

#[test]
#[ignore]
fn kv_cache_append_evict_gather_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = KvCacheAppend::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    let head_dim = 16usize;
    // (r0, b, w) — all eviction-regime (r0 + b > w). The middle group is the
    // partial-eviction case (b < w, surviving prior rows) that underflowed.
    let cases = [
        (10usize, 20usize, 16usize), // b > w, some old rows survive logically
        (0, 20, 16),                 // r0 = 0, b > w
        (16, 16, 16),                // b == w
        (12, 8, 16),                 // b < w, partial eviction (underflow case)
        (16, 4, 16),                 // full window, tiny chunk, survivors
        (15, 2, 16),                 // crosses the window by exactly one row
        (100, 50, 128),              // b < w at the real SWA_WINDOW
    ];

    for (r0, b, w) in cases {
        assert!(r0 + b > w, "case ({r0},{b},{w}) is not in the eviction regime");

        // New rows: global ids continue after the r0 old rows.
        let mut new = vec![0f32; b * head_dim];
        for i in 0..b {
            for d in 0..head_dim {
                new[i * head_dim + d] = (r0 + i) as f32 + (d as f32) * 0.001;
            }
        }

        let expected = serial_reference(r0, b, w, head_dim, &new);

        // Seed device cache old rows the same way serial_reference does.
        let mut cache_host = vec![0f32; w * head_dim];
        for row in 0..r0.min(w) {
            for d in 0..head_dim {
                cache_host[row * head_dim + d] = row as f32 + (d as f32) * 0.001;
            }
        }

        let mut d_cache: DeviceBuffer<f32> = DeviceBuffer::new(device.id, w * head_dim)?;
        let mut d_new: DeviceBuffer<f32> = DeviceBuffer::new(device.id, b * head_dim)?;
        let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, w * head_dim)?;
        d_cache.copy_from_host(&cache_host)?;
        d_new.copy_from_host(&new)?;

        kernel.launch_evict_gather(
            &stream,
            &d_cache,
            &d_new,
            &mut d_out,
            r0 as u32,
            b as u32,
            w as u32,
            head_dim as u32,
        )?;
        stream.synchronize()?;

        let mut got = vec![0f32; w * head_dim];
        d_out.copy_to_host(&mut got)?;

        let mut max_abs = 0f32;
        let mut worst = (0usize, 0usize);
        for s in 0..w {
            for d in 0..head_dim {
                let diff = (got[s * head_dim + d] - expected[s * head_dim + d]).abs();
                if diff > max_abs {
                    max_abs = diff;
                    worst = (s, d);
                }
            }
        }
        eprintln!(
            "case (r0={r0}, b={b}, w={w}): max_abs_diff={max_abs:.3e} (worst slot {}, d {})",
            worst.0, worst.1
        );
        assert!(
            max_abs == 0.0,
            "case (r0={r0}, b={b}, w={w}): gather diverged from serial loop (max_abs={max_abs:.3e})"
        );
    }

    eprintln!("ALL CASES MATCH serial slide loop exactly");
    Ok(())
}
