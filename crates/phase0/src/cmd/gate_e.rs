//! Gate E — effective fused-dequant-GEMV bandwidth.
//!
//! Measures actual sustained throughput of a representative Q8_0 GEMV
//! kernel (ds4-style: pre-quantized int8 input, __sudot4 inner loop) and
//! compares against theoretical memory bandwidth on each device. Phase
//! 1+ perf claims are scaled by this efficiency ratio.
//!
//! Shape M=4096 K=2048: weight memory = 4096 * 2048 / 32 * 34 = 8.5 MiB.
//! At 215 GB/s (Strix LPDDR5X theoretical) → 41 us / iter minimum.
//! At 644 GB/s (9070 XT GDDR6 theoretical) → 14 us / iter minimum.

use std::ffi::c_void;

use color_eyre::eyre;
use serde::Serialize;
use v4flash_hip::{Device, DeviceBuffer, Event, LaunchConfig, Module, Stream};

use crate::results;

const GEMV_GFX1201: &[u8] = include_bytes!(env!("KERNEL_Q8_0_GEMV_GFX1201"));
const GEMV_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_0_GEMV_GFX1151"));

const Q8_0_BLOCK_BYTES: u32 = 34;

/// Shapes to sweep. Critical: working set must exceed the largest cache
/// on the device to measure DRAM bandwidth (Strix MALL 32 MB, 9070 XT
/// Infinity Cache 64 MB). Smaller shapes are still useful as cache
/// bandwidth measurements.
const SHAPES: &[(u32, u32, &str)] = &[
    (4096, 2048, "cache_resident"),    // 8.5 MiB — fits in both caches
    (8192, 4096, "between_caches"),    // 34 MiB — defeats MALL, fits IC
    (16384, 8192, "dram_bound_dgpu"),  // 136 MiB — defeats both caches
    (32768, 8192, "dram_bound_large"), // 272 MiB — far past both
];

// Theoretical per the design doc §2 (matches LPDDR5X-8000 256-bit / GDDR6 20Gbps 256-bit).
const STRIX_THEORETICAL_GB_S: f64 = 215.0;
const NAVI48_THEORETICAL_GB_S: f64 = 644.0;

#[derive(Serialize)]
pub struct GateEReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub samples: Vec<DeviceSample>,
    pub decision: Decision,
}

#[derive(Serialize)]
pub struct DeviceSample {
    pub device_id: i32,
    pub gcn_arch: String,
    pub shape_label: String,
    pub m: u32,
    pub k: u32,
    pub weight_bytes: u64,
    pub iterations: u32,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p99_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub stddev_us: f64,
    pub theoretical_gb_per_s: f64,
    pub mean_gb_per_s: f64,
    pub p50_gb_per_s: f64,
    pub min_gb_per_s: f64,
    pub efficiency_p50: f64,
    pub efficiency_min: f64,
}

#[derive(Serialize)]
pub struct Decision {
    pub min_efficiency: f64,
    pub recommendation: String,
    pub rationale: String,
}

pub fn run(iterations: u32) -> eyre::Result<()> {
    let devices = Device::all()?;
    let mut all_samples = Vec::new();

    for dev in &devices {
        let props = dev.properties()?;
        let arch = props.gcn_arch_name.clone();
        let image = match arch.as_str() {
            s if s.starts_with("gfx1201") => Some(GEMV_GFX1201),
            s if s.starts_with("gfx1151") => Some(GEMV_GFX1151),
            _ => None,
        };
        if image.is_none() {
            println!("device {} ({arch}): SKIP — no hsaco built", dev.id);
            continue;
        }

        let theoretical = if arch.starts_with("gfx1201") {
            NAVI48_THEORETICAL_GB_S
        } else {
            STRIX_THEORETICAL_GB_S
        };

        println!("\n== device {} ({arch}), theoretical {:.0} GB/s ==", dev.id, theoretical);

        for &(m, k, label) in SHAPES {
            let blocks = k / 32;
            let weight_bytes = (m as u64) * (blocks as u64) * (Q8_0_BLOCK_BYTES as u64);
            // 9070 XT only has 16 GB; skip shapes > 8 GB to leave headroom
            if weight_bytes > 8 * (1u64 << 30) {
                println!("  {label} M={m} K={k} ({} MiB): SKIP — too large", weight_bytes >> 20);
                continue;
            }
            match measure(*dev, image.unwrap(), m, k, iterations) {
                Ok(stats) => {
                    let mean_gb = (weight_bytes as f64) / (stats.mean * 1e-6) / 1e9;
                    let p50_gb = (weight_bytes as f64) / (stats.p50 * 1e-6) / 1e9;
                    let min_gb = (weight_bytes as f64) / (stats.min * 1e-6) / 1e9;
                    let eff_p50 = p50_gb / theoretical;
                    let eff_min = min_gb / theoretical;

                    println!(
                        "  {label} M={m} K={k} ({:>5} MiB): mean {:.1} us ({:.1} GB/s), p50 {:.1} us ({:.1} GB/s, {:.0}%), min {:.1} us ({:.1} GB/s, {:.0}%), stddev {:.2}",
                        weight_bytes >> 20,
                        stats.mean, mean_gb, stats.p50, p50_gb, eff_p50 * 100.0,
                        stats.min, min_gb, eff_min * 100.0, stats.stddev,
                    );

                    all_samples.push(DeviceSample {
                        device_id: dev.id,
                        gcn_arch: arch.clone(),
                        shape_label: label.into(),
                        m,
                        k,
                        weight_bytes,
                        iterations,
                        mean_us: stats.mean,
                        p50_us: stats.p50,
                        p99_us: stats.p99,
                        min_us: stats.min,
                        max_us: stats.max,
                        stddev_us: stats.stddev,
                        theoretical_gb_per_s: theoretical,
                        mean_gb_per_s: mean_gb,
                        p50_gb_per_s: p50_gb,
                        min_gb_per_s: min_gb,
                        efficiency_p50: eff_p50,
                        efficiency_min: eff_min,
                    });
                }
                Err(e) => println!("  {label} M={m} K={k}: FAILED ({e:#})", ),
            }
        }
    }

    let decision = decide(&all_samples);
    println!("\n== decision ==");
    println!("min DRAM-bound efficiency:  {:.0}%", decision.min_efficiency * 100.0);
    println!("recommendation: {}", decision.recommendation);
    println!("rationale:      {}", decision.rationale);

    let report = GateEReport {
        gate: "gate_e",
        timestamp: results::now_unix(),
        samples: all_samples,
        decision,
    };
    let path = results::write("gate_e", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

struct Stats {
    mean: f64,
    p50: f64,
    p99: f64,
    min: f64,
    max: f64,
    stddev: f64,
}

fn measure(dev: Device, image: &[u8], m: u32, k: u32, iterations: u32) -> eyre::Result<Stats> {
    dev.set_current()?;
    let module = Module::load_data(image)?;
    let gemv = module.get_function("q8_0_gemv_warp8")?;

    let stream = Stream::new(dev.id)?;

    let blocks = k / 32;
    let weight_bytes = (m as usize) * (blocks as usize) * (Q8_0_BLOCK_BYTES as usize);
    let mut weights: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, weight_bytes)?;
    weights.fill_zero()?;

    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(dev.id, k as usize)?;
    xq.fill_zero()?;
    let mut xscale: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, blocks as usize)?;
    xscale.fill_zero()?;
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, m as usize)?;
    out.fill_zero()?;

    let mut out_ptr = out.raw();
    let mut w_ptr = weights.raw();
    let mut xq_ptr = xq.raw();
    let mut xs_ptr = xscale.raw();
    let in_dim: u32 = k;
    let out_dim: u32 = m;
    let n_blocks: u32 = blocks;
    let mut args: [*mut c_void; 7] = [
        &mut out_ptr as *mut _ as *mut c_void,
        &mut w_ptr as *mut _ as *mut c_void,
        &mut xq_ptr as *mut _ as *mut c_void,
        &mut xs_ptr as *mut _ as *mut c_void,
        &in_dim as *const _ as *mut c_void,
        &out_dim as *const _ as *mut c_void,
        &n_blocks as *const _ as *mut c_void,
    ];

    let cfg = LaunchConfig {
        grid: ((m + 7) / 8, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm.
    for _ in 0..20 {
        unsafe { gemv.launch_raw(cfg, &stream, &mut args)? };
    }
    stream.synchronize()?;

    // Per-iteration event timing.
    let mut events: Vec<(Event, Event)> = Vec::with_capacity(iterations as usize);
    for _ in 0..iterations {
        events.push((Event::new()?, Event::new()?));
    }

    for (start, end) in &events {
        start.record(&stream)?;
        unsafe { gemv.launch_raw(cfg, &stream, &mut args)? };
        end.record(&stream)?;
    }
    stream.synchronize()?;

    let mut samples: Vec<f64> = events
        .iter()
        .map(|(s, e)| Event::elapsed_ms(s, e).map(|ms| ms as f64 * 1000.0))
        .collect::<eyre::Result<_>>()?;

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let p50 = samples[samples.len() / 2];
    let p99 = samples[(samples.len() * 99) / 100];
    let min = *samples.first().unwrap();
    let max = *samples.last().unwrap();
    let variance =
        samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / samples.len() as f64;
    Ok(Stats { mean, p50, p99, min, max, stddev: variance.sqrt() })
}

fn decide(samples: &[DeviceSample]) -> Decision {
    // Use only the DRAM-bound shapes for the decision — cache-resident
    // shapes give misleadingly high efficiency.
    let min_eff = samples
        .iter()
        .filter(|s| s.shape_label.starts_with("dram_bound"))
        .map(|s| s.efficiency_p50)
        .fold(f64::INFINITY, f64::min);

    if min_eff.is_infinite() {
        return Decision {
            min_efficiency: 0.0,
            recommendation: "no DRAM-bound samples collected".into(),
            rationale: "All sample shapes fit in cache; cannot estimate DRAM bandwidth".into(),
        };
    }

    let (recommendation, rationale) = if min_eff >= 0.85 {
        (
            "doc's bandwidth-derived perf targets are tight".into(),
            format!(
                "Min device efficiency p50 = {:.0}% of theoretical. The 55 tok/s default \
                 target should hold; the 65 tok/s stretch may be reachable.",
                min_eff * 100.0
            ),
        )
    } else if min_eff >= 0.60 {
        (
            "doc's perf targets need scaling down".into(),
            format!(
                "Min efficiency p50 = {:.0}% — below 85% but above 60%. Scale per-token \
                 throughput estimates proportionally: ~{:.0} tok/s expected vs doc's 55.",
                min_eff * 100.0,
                55.0 * min_eff / 0.85
            ),
        )
    } else {
        (
            "doc's perf targets are unreachable; investigate".into(),
            format!(
                "Min efficiency p50 = {:.0}% — far below 60% threshold. Either the kernel \
                 needs more work (LDS staging, prefetch, dual-issue) or the hardware is \
                 not delivering theoretical bandwidth (frequency scaling, MALL pollution).",
                min_eff * 100.0
            ),
        )
    };

    Decision { min_efficiency: min_eff, recommendation, rationale }
}
