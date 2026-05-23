//! Gate D — MALL / Infinity Cache characterization.
//!
//! Three measurements per device, sweeping buffer size:
//!   1. baseline:    single pass read → effective bandwidth at this size
//!   2. polluted:    write a 32 MiB unrelated buffer between two reads,
//!                   measure the SECOND read's bandwidth (post-eviction)
//!   3. nontemporal: re-read with __builtin_nontemporal_load hint —
//!                   verifies whether the hint actually bypasses cache
//!
//! What the curves tell us:
//!   - Strix MALL is 32 MB. Below 32 MiB working set we should see
//!     "cached" bandwidth (multiple × DRAM rate). Above 32 MiB, DRAM rate.
//!   - 9070 XT IC is 64 MB. Same logic.
//!   - If `polluted` looks identical to `baseline` second-read, MALL
//!     evicts under interleaved traffic (doc's §4.3 worry).
//!   - If `nontemporal` is similar to `cached` re-read, the bypass hint
//!     is ignored on this hardware (doc's §4.3 worry).

use std::ffi::c_void;

use color_eyre::eyre;
use serde::Serialize;
use v4flash_hip::{Device, DeviceBuffer, Event, LaunchConfig, Module, Stream};

use crate::results;

const PROBE_GFX1201: &[u8] = include_bytes!(env!("KERNEL_CACHE_PROBE_GFX1201"));
const PROBE_GFX1151: &[u8] = include_bytes!(env!("KERNEL_CACHE_PROBE_GFX1151"));

// Buffer sizes in u32 elements (× 4 = bytes). Log-spaced, spanning well
// below and above both cache sizes.
const SIZES_WORDS: &[usize] = &[
    256 * 1024,        // 1 MiB
    1024 * 1024,       // 4 MiB
    2 * 1024 * 1024,   // 8 MiB
    4 * 1024 * 1024,   // 16 MiB
    6 * 1024 * 1024,   // 24 MiB — just under Strix MALL
    8 * 1024 * 1024,   // 32 MiB — at Strix MALL boundary
    12 * 1024 * 1024,  // 48 MiB — over MALL, under IC
    16 * 1024 * 1024,  // 64 MiB — at IC boundary
    24 * 1024 * 1024,  // 96 MiB — over both
    32 * 1024 * 1024,  // 128 MiB — clearly DRAM-bound
    64 * 1024 * 1024,  // 256 MiB
];

const POLLUTION_BYTES: usize = 32 * 1024 * 1024; // 32 MiB write between reads

#[derive(Serialize)]
pub struct GateDReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub samples: Vec<DeviceSample>,
    pub decision: Decision,
}

#[derive(Serialize)]
pub struct DeviceSample {
    pub device_id: i32,
    pub gcn_arch: String,
    pub variant: String,
    pub size_bytes: usize,
    pub mean_us: f64,
    pub p50_us: f64,
    pub gb_per_s_p50: f64,
}

#[derive(Serialize)]
pub struct Decision {
    pub mall_effective_capacity_mib: u32,
    pub ic_effective_capacity_mib: u32,
    pub mall_evicts_under_pollution: bool,
    pub nontemporal_bypass_works_strix: bool,
    pub nontemporal_bypass_works_dgpu: bool,
    pub recommendation: String,
}

pub fn run(iterations: u32) -> eyre::Result<()> {
    let devices = Device::all()?;
    let mut all_samples = Vec::new();

    for dev in &devices {
        let props = dev.properties()?;
        let arch = props.gcn_arch_name.clone();
        let image = match arch.as_str() {
            s if s.starts_with("gfx1201") => Some(PROBE_GFX1201),
            s if s.starts_with("gfx1151") => Some(PROBE_GFX1151),
            _ => None,
        };
        if image.is_none() {
            continue;
        }
        let image = image.unwrap();
        println!("\n== device {} ({arch}) ==", dev.id);
        println!(
            "  {:>7}  {:>15}  {:>15}  {:>15}",
            "MiB", "baseline GB/s", "polluted GB/s", "nontemp GB/s"
        );

        dev.set_current()?;
        let module = Module::load_data(image)?;
        let read_cached = module.get_function("read_cached")?;
        let read_nontemporal = module.get_function("read_nontemporal")?;
        let pollute = module.get_function("pollute_write")?;
        let stream = Stream::new(dev.id)?;

        // Pollution buffer (32 MiB) — used to evict between reads.
        let pollution_buf: DeviceBuffer<u32> =
            DeviceBuffer::new(dev.id, POLLUTION_BYTES / 4)?;

        for &n_words in SIZES_WORDS {
            let size_bytes = n_words * 4;
            // Skip absurd sizes for the 16 GB 9070 XT
            if size_bytes > 4 * (1usize << 30) {
                continue;
            }
            let mut buf: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, n_words)?;
            buf.fill_zero()?;

            // Output: one u32 per thread. Total threads = grid × block.
            let block: u32 = 256;
            let grid: u32 = 512;
            let total_threads = (block * grid) as usize;
            let mut out: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, total_threads)?;
            out.fill_zero()?;

            let mut buf_ptr = buf.raw();
            let mut out_ptr = out.raw();
            let mut pol_ptr = pollution_buf.raw();
            let n_arg: u32 = n_words as u32;
            let pol_n: u32 = (POLLUTION_BYTES / 4) as u32;
            let pol_seed: u32 = 0xDEAD_BEEF;

            let cfg = LaunchConfig {
                grid: (grid, 1, 1),
                block: (block, 1, 1),
                shared_mem_bytes: 0,
            };

            // Helper to time N kernel launches.
            let time_kernel = |func: &v4flash_hip::Function,
                               args: &mut [*mut c_void],
                               iters: u32,
                               warm: u32|
             -> eyre::Result<f64> {
                for _ in 0..warm {
                    unsafe { func.launch_raw(cfg, &stream, args)? };
                }
                stream.synchronize()?;
                let mut samples = Vec::with_capacity(iters as usize);
                for _ in 0..iters {
                    let start = Event::new()?;
                    let end = Event::new()?;
                    start.record(&stream)?;
                    unsafe { func.launch_raw(cfg, &stream, args)? };
                    end.record(&stream)?;
                    end.synchronize()?;
                    samples.push(Event::elapsed_ms(&start, &end)? as f64 * 1000.0);
                }
                samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
                Ok(samples[samples.len() / 2])
            };

            // baseline: time the (warm) cached read
            let mut read_args: [*mut c_void; 3] = [
                &mut buf_ptr as *mut _ as *mut c_void,
                &mut out_ptr as *mut _ as *mut c_void,
                &n_arg as *const _ as *mut c_void,
            ];
            let p50_baseline_us = time_kernel(&read_cached, &mut read_args, iterations, 5)?;
            let gb_baseline = size_bytes as f64 / (p50_baseline_us * 1e-6) / 1e9;

            // polluted: write to pollution buf (large enough to evict),
            // then read the small buf. Measure JUST the read.
            let mut pol_args: [*mut c_void; 3] = [
                &mut pol_ptr as *mut _ as *mut c_void,
                &pol_n as *const _ as *mut c_void,
                &pol_seed as *const _ as *mut c_void,
            ];
            // Mix-then-time loop: for each timed iteration, pollute first.
            let mut polluted_samples = Vec::with_capacity(iterations as usize);
            for _ in 0..5 {
                unsafe { pollute.launch_raw(cfg, &stream, &mut pol_args)? };
                unsafe { read_cached.launch_raw(cfg, &stream, &mut read_args)? };
            }
            stream.synchronize()?;
            for _ in 0..iterations {
                unsafe { pollute.launch_raw(cfg, &stream, &mut pol_args)? };
                let start = Event::new()?;
                let end = Event::new()?;
                start.record(&stream)?;
                unsafe { read_cached.launch_raw(cfg, &stream, &mut read_args)? };
                end.record(&stream)?;
                end.synchronize()?;
                polluted_samples.push(Event::elapsed_ms(&start, &end)? as f64 * 1000.0);
            }
            polluted_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p50_polluted_us = polluted_samples[polluted_samples.len() / 2];
            let gb_polluted = size_bytes as f64 / (p50_polluted_us * 1e-6) / 1e9;

            // nontemporal: read same buf with __builtin_nontemporal_load hint
            let p50_nt_us = time_kernel(&read_nontemporal, &mut read_args, iterations, 5)?;
            let gb_nt = size_bytes as f64 / (p50_nt_us * 1e-6) / 1e9;

            println!(
                "  {:>7}  {:>15.1}  {:>15.1}  {:>15.1}",
                size_bytes >> 20,
                gb_baseline,
                gb_polluted,
                gb_nt,
            );

            all_samples.push(DeviceSample {
                device_id: dev.id,
                gcn_arch: arch.clone(),
                variant: "baseline_cached".into(),
                size_bytes,
                mean_us: p50_baseline_us,
                p50_us: p50_baseline_us,
                gb_per_s_p50: gb_baseline,
            });
            all_samples.push(DeviceSample {
                device_id: dev.id,
                gcn_arch: arch.clone(),
                variant: "polluted_then_cached".into(),
                size_bytes,
                mean_us: p50_polluted_us,
                p50_us: p50_polluted_us,
                gb_per_s_p50: gb_polluted,
            });
            all_samples.push(DeviceSample {
                device_id: dev.id,
                gcn_arch: arch.clone(),
                variant: "nontemporal".into(),
                size_bytes,
                mean_us: p50_nt_us,
                p50_us: p50_nt_us,
                gb_per_s_p50: gb_nt,
            });
        }
    }

    let decision = decide(&all_samples);
    println!("\n== decision ==");
    println!("Strix MALL effective capacity ≈ {} MiB", decision.mall_effective_capacity_mib);
    println!("9070 XT IC effective capacity  ≈ {} MiB", decision.ic_effective_capacity_mib);
    println!("MALL evicts under pollution:        {}", decision.mall_evicts_under_pollution);
    println!("nontemporal bypass works (Strix):   {}", decision.nontemporal_bypass_works_strix);
    println!("nontemporal bypass works (9070 XT): {}", decision.nontemporal_bypass_works_dgpu);
    println!("recommendation: {}", decision.recommendation);

    let report = GateDReport {
        gate: "gate_d",
        timestamp: results::now_unix(),
        samples: all_samples,
        decision,
    };
    let path = results::write("gate_d", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn decide(samples: &[DeviceSample]) -> Decision {
    // Effective cache capacity ≈ largest size where baseline > 1.5× DRAM rate.
    // DRAM rate is approximated by the LARGEST tested size on that device.
    let dram_rate = |device_id: i32| -> f64 {
        samples
            .iter()
            .filter(|s| s.device_id == device_id && s.variant == "baseline_cached")
            .map(|s| (s.size_bytes, s.gb_per_s_p50))
            .max_by_key(|(sz, _)| *sz)
            .map(|(_, gbs)| gbs)
            .unwrap_or(0.0)
    };

    let cap_for = |device_id: i32| -> u32 {
        let d = dram_rate(device_id);
        if d == 0.0 { return 0; }
        let cap_bytes = samples
            .iter()
            .filter(|s| s.device_id == device_id && s.variant == "baseline_cached")
            .filter(|s| s.gb_per_s_p50 > d * 1.5)
            .map(|s| s.size_bytes)
            .max()
            .unwrap_or(0);
        (cap_bytes >> 20) as u32
    };

    let strix = samples
        .iter()
        .find(|s| s.gcn_arch.starts_with("gfx1151"))
        .map(|s| s.device_id);
    let dgpu = samples
        .iter()
        .find(|s| s.gcn_arch.starts_with("gfx1201"))
        .map(|s| s.device_id);

    let mall_mib = strix.map(cap_for).unwrap_or(0);
    let ic_mib = dgpu.map(cap_for).unwrap_or(0);

    // MALL eviction: do polluted samples at a cache-resident size on Strix
    // match the DRAM rate (= MALL evicted)?
    let mall_evicts = if let Some(id) = strix {
        let dram = dram_rate(id);
        samples
            .iter()
            .filter(|s| s.device_id == id && s.variant == "polluted_then_cached")
            .filter(|s| s.size_bytes <= 8 << 20)
            .any(|s| s.gb_per_s_p50 < dram * 1.5)
    } else {
        false
    };

    // Non-temporal bypass works iff at a small-buffer size, the nontemp
    // bandwidth is significantly LOWER than the cached bandwidth.
    let bypass_works = |device_id: i32| -> bool {
        let small: Vec<&DeviceSample> = samples
            .iter()
            .filter(|s| s.device_id == device_id && s.size_bytes <= 8 << 20)
            .collect();
        let baseline_max = small
            .iter()
            .filter(|s| s.variant == "baseline_cached")
            .map(|s| s.gb_per_s_p50)
            .fold(0.0_f64, f64::max);
        let nt_max = small
            .iter()
            .filter(|s| s.variant == "nontemporal")
            .map(|s| s.gb_per_s_p50)
            .fold(0.0_f64, f64::max);
        // Bypass "works" if nontemporal is < 60% of cached (i.e. closer
        // to DRAM rate than cache rate).
        baseline_max > 0.0 && nt_max < baseline_max * 0.6
    };

    let nt_strix = strix.map(bypass_works).unwrap_or(false);
    let nt_dgpu = dgpu.map(bypass_works).unwrap_or(false);

    let recommendation = if !nt_strix {
        "non-temporal hint is IGNORED on Strix iGPU — design doc's §4.3 W_down \
         bypass strategy will NOT work. Plan for W_down to pollute MALL alongside \
         the cached W_up/W_gate. Need a different overlap strategy."
            .into()
    } else if mall_evicts {
        "MALL evicts under interleaved write traffic — top-N expert residency \
         is fragile. Either size N down to fit, or accept partial hit rate."
            .into()
    } else {
        format!(
            "MALL holds ~{} MiB resilient to pollution; non-temporal bypass works. \
             Doc's §4.3 cache strategy is viable.",
            mall_mib
        )
    };

    Decision {
        mall_effective_capacity_mib: mall_mib,
        ic_effective_capacity_mib: ic_mib,
        mall_evicts_under_pollution: mall_evicts,
        nontemporal_bypass_works_strix: nt_strix,
        nontemporal_bypass_works_dgpu: nt_dgpu,
        recommendation,
    }
}
