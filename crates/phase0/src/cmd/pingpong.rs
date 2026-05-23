//! `phase0 pingpong` — characterize the cross-device round-trip floor
//! using the host-bounce path (which we expect to work even if Gate C's
//! peer-direct path doesn't).
//!
//! For each ordered device pair (a, b): allocate small buffers on both,
//! plus a host scratch buffer; time N iterations of
//!   1. write pattern on a (kernel-less: hipMemcpy HtoD a)
//!   2. memcpy a → host
//!   3. memcpy host → b
//!   4. memcpy b → host (validate pattern survived)
//! We use sync hipMemcpy for the floor measurement; Gate C will compare
//! against async + events later.

use std::time::Instant;

use color_eyre::eyre;
use serde::Serialize;
use v4flash_hip::{Device, DeviceBuffer};

use crate::results;

#[derive(Serialize)]
pub struct PingpongReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub iterations: u32,
    pub payload_bytes: usize,
    pub pairs: Vec<PairResult>,
}

#[derive(Serialize)]
pub struct PairResult {
    pub src_device: i32,
    pub src_arch: String,
    pub dst_device: i32,
    pub dst_arch: String,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub correctness_ok: bool,
}

pub fn run(iterations: u32, payload_bytes: usize) -> eyre::Result<()> {
    let devices = Device::all()?;
    if devices.len() < 2 {
        println!("pingpong needs at least 2 devices, found {}", devices.len());
        return Ok(());
    }

    let payload_u32 = payload_bytes / 4;
    let pattern: Vec<u32> = (0..payload_u32 as u32).map(|i| i.wrapping_mul(2654435761)).collect();

    let mut pair_results = Vec::new();
    for src in &devices {
        for dst in &devices {
            if src.id == dst.id {
                continue;
            }

            let src_props = src.properties()?;
            let dst_props = dst.properties()?;

            println!(
                "pingpong {} ({}) -> {} ({}): {} iter × {} B",
                src.id, src_props.gcn_arch_name, dst.id, dst_props.gcn_arch_name,
                iterations, payload_bytes,
            );

            // Allocate src + dst buffers on their respective devices.
            src.set_current()?;
            let mut src_buf: DeviceBuffer<u32> = DeviceBuffer::new(src.id, payload_u32)?;
            src_buf.copy_from_host(&pattern)?;

            dst.set_current()?;
            let mut dst_buf: DeviceBuffer<u32> = DeviceBuffer::new(dst.id, payload_u32)?;
            dst_buf.fill_zero()?;

            let mut scratch = vec![0u32; payload_u32];
            let mut samples: Vec<f64> = Vec::with_capacity(iterations as usize);

            // Warm: do one round before timing.
            do_round(*src, &src_buf, *dst, &mut dst_buf, &mut scratch)?;

            for _ in 0..iterations {
                let t0 = Instant::now();
                do_round(*src, &src_buf, *dst, &mut dst_buf, &mut scratch)?;
                samples.push(t0.elapsed().as_nanos() as f64 / 1000.0);
            }

            // Validate one final round wrote the pattern back through both
            // hops by reading dst → host and comparing.
            dst.set_current()?;
            dst_buf.copy_to_host(&mut scratch)?;
            let correct = scratch == pattern;
            if !correct {
                println!(
                    "    correctness: FAIL (first mismatch at {})",
                    scratch.iter().zip(&pattern).position(|(a, b)| a != b).unwrap_or(0)
                );
            }

            samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mean = samples.iter().sum::<f64>() / samples.len() as f64;
            let p50 = samples[samples.len() / 2];
            let p99 = samples[(samples.len() * 99) / 100];
            let min = *samples.first().unwrap();
            let max = *samples.last().unwrap();

            println!(
                "    mean {:.1} us, p50 {:.1} us, p99 {:.1} us (min {:.1} / max {:.1})",
                mean, p50, p99, min, max
            );

            pair_results.push(PairResult {
                src_device: src.id,
                src_arch: src_props.gcn_arch_name.clone(),
                dst_device: dst.id,
                dst_arch: dst_props.gcn_arch_name.clone(),
                mean_us: mean,
                p50_us: p50,
                p99_us: p99,
                min_us: min,
                max_us: max,
                correctness_ok: correct,
            });
        }
    }

    let report = PingpongReport {
        gate: "pingpong",
        timestamp: results::now_unix(),
        iterations,
        payload_bytes,
        pairs: pair_results,
    };
    let path = results::write("pingpong", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn do_round(
    src: Device,
    src_buf: &DeviceBuffer<u32>,
    dst: Device,
    dst_buf: &mut DeviceBuffer<u32>,
    host_scratch: &mut [u32],
) -> eyre::Result<()> {
    // Pull src → host (must be on src's current device).
    src.set_current()?;
    src_buf.copy_to_host(host_scratch)?;

    // Push host → dst (must be on dst's current device).
    dst.set_current()?;
    dst_buf.copy_from_host(host_scratch)?;
    Ok(())
}
