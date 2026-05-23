//! Gate B — gfx1100 binary vs gfx1151 native on Strix iGPU.
//!
//! HSA_OVERRIDE_GFX_VERSION is process-global; we spawn one child per
//! config and compare their reported Q8_0 GEMV throughput on the Strix
//! iGPU. The DRAM-bound shape (M=16384, K=8192, ~136 MiB weights) gives
//! the most trustworthy bandwidth number.
//!
//! Decision: if gfx1100 binary is >30% faster than gfx1151 native, the
//! design doc's "gfx1100-via-override" path is worth the two-process
//! complexity (since under override the 9070 XT also reports gfx1100 —
//! see Gate A). Otherwise stay native.

use std::env;
use std::process::Command;

use color_eyre::eyre::{self, Context};
use serde::{Deserialize, Serialize};

use crate::results;

#[derive(Serialize, Deserialize)]
pub struct GateBReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub probes: Vec<ProbeOutcome>,
    pub decision: Decision,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProbeOutcome {
    pub label: String,
    pub hsa_override: Option<String>,
    pub exit_ok: bool,
    pub stderr_excerpt: String,
    pub probed_arch: String,
    pub p50_us: f64,
    pub gb_per_s_p50: f64,
}

#[derive(Serialize, Deserialize)]
pub struct ProbeJson {
    pub probed_arch: String,
    pub p50_us: f64,
    pub gb_per_s_p50: f64,
}

#[derive(Serialize, Deserialize)]
pub struct Decision {
    pub gfx1100_speedup: f64,
    pub use_gfx1100_binary: bool,
    pub recommendation: String,
}

pub fn run() -> eyre::Result<()> {
    let self_exe = env::current_exe().wrap_err("current_exe")?;

    let configs: &[(&str, Option<&str>)] = &[
        ("native_gfx1151", None),
        ("override_gfx1100", Some("11.0.0")),
    ];

    let mut probes = Vec::new();
    for (label, override_val) in configs {
        println!("\n== probing {label} (HSA_OVERRIDE_GFX_VERSION={override_val:?}) ==");
        let outcome = run_probe(&self_exe, label, *override_val)?;
        if outcome.exit_ok {
            println!(
                "    arch={}, p50 {:.1} us, {:.1} GB/s",
                outcome.probed_arch, outcome.p50_us, outcome.gb_per_s_p50
            );
        } else {
            println!("    FAILED. stderr excerpt:");
            for line in outcome.stderr_excerpt.lines() {
                println!("    | {line}");
            }
        }
        probes.push(outcome);
    }

    let decision = decide(&probes);
    println!("\n== decision ==");
    println!("gfx1100 speedup vs gfx1151:  {:.2}×", decision.gfx1100_speedup);
    println!("recommend gfx1100 binary:    {}", decision.use_gfx1100_binary);
    println!("recommendation:              {}", decision.recommendation);

    let report = GateBReport {
        gate: "gate_b",
        timestamp: results::now_unix(),
        probes,
        decision,
    };
    let path = results::write("gate_b", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn run_probe(
    self_exe: &std::path::Path,
    label: &str,
    override_val: Option<&str>,
) -> eyre::Result<ProbeOutcome> {
    let mut cmd = Command::new(self_exe);
    cmd.arg("gate-b-probe");
    cmd.env_remove("HSA_OVERRIDE_GFX_VERSION");
    if let Some(v) = override_val {
        cmd.env("HSA_OVERRIDE_GFX_VERSION", v);
    }
    let output = cmd.output().wrap_err_with(|| format!("spawn probe {label}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let stderr_excerpt: String = stderr.lines().take(8).collect::<Vec<_>>().join("\n");

    if !output.status.success() {
        return Ok(ProbeOutcome {
            label: label.into(),
            hsa_override: override_val.map(String::from),
            exit_ok: false,
            stderr_excerpt,
            probed_arch: String::new(),
            p50_us: 0.0,
            gb_per_s_p50: 0.0,
        });
    }

    let pj: ProbeJson = serde_json::from_slice(&output.stdout)
        .wrap_err_with(|| format!("parse probe stdout for {label}"))?;

    Ok(ProbeOutcome {
        label: label.into(),
        hsa_override: override_val.map(String::from),
        exit_ok: true,
        stderr_excerpt,
        probed_arch: pj.probed_arch,
        p50_us: pj.p50_us,
        gb_per_s_p50: pj.gb_per_s_p50,
    })
}

fn decide(probes: &[ProbeOutcome]) -> Decision {
    let native = probes.iter().find(|p| p.label == "native_gfx1151");
    let overridden = probes.iter().find(|p| p.label == "override_gfx1100");

    match (native, overridden) {
        (Some(n), Some(o)) if n.exit_ok && o.exit_ok => {
            let speedup = o.gb_per_s_p50 / n.gb_per_s_p50.max(1e-9);
            let use_1100 = speedup >= 1.30;
            let recommendation = if use_1100 {
                format!(
                    "gfx1100 binary is {:.0}% faster on Strix iGPU — worth the \
                     two-process complexity to enable it (override is process-global \
                     so dGPU would also be aliased to gfx1100; need a separate \
                     process for the iGPU side).",
                    (speedup - 1.0) * 100.0
                )
            } else {
                format!(
                    "gfx1100 binary is only {:.0}% faster (or slower); not worth the \
                     two-process complexity. Stay with native gfx1151.",
                    (speedup - 1.0) * 100.0
                )
            };
            Decision {
                gfx1100_speedup: speedup,
                use_gfx1100_binary: use_1100,
                recommendation,
            }
        }
        _ => Decision {
            gfx1100_speedup: 0.0,
            use_gfx1100_binary: false,
            recommendation: "at least one probe failed; cannot compare. Stay native gfx1151.".into(),
        },
    }
}

/// Child probe: find the integrated GPU, load the GEMV image matching
/// its CURRENT reported arch (gfx1151 native or gfx1100 under override),
/// run a DRAM-bound shape, print JSON to stdout.
pub fn probe_to_stdout() -> eyre::Result<()> {
    use std::ffi::c_void;
    use v4flash_hip::{Device, DeviceBuffer, Event, LaunchConfig, Module, Stream};

    const GEMV_GFX1151: &[u8] = include_bytes!(env!("KERNEL_Q8_0_GEMV_GFX1151"));
    const GEMV_GFX1100: &[u8] = include_bytes!(env!("KERNEL_Q8_0_GEMV_GFX1100"));
    const Q8_0_BLOCK_BYTES: u32 = 34;
    const M: u32 = 16384;
    const K: u32 = 8192;
    const BLOCKS: u32 = K / 32;
    const ITERATIONS: u32 = 50;

    let devices = Device::all()?;
    let igpu = devices
        .iter()
        .find(|d| d.properties().map(|p| p.integrated).unwrap_or(false))
        .copied()
        .ok_or_else(|| color_eyre::eyre::eyre!("no integrated GPU found"))?;

    let props = igpu.properties()?;
    let arch = props.gcn_arch_name.clone();
    let image = match arch.as_str() {
        s if s.starts_with("gfx1151") => GEMV_GFX1151,
        s if s.starts_with("gfx1100") => GEMV_GFX1100,
        other => return Err(color_eyre::eyre::eyre!("no GEMV image for reported arch {other}")),
    };

    igpu.set_current()?;
    let module = Module::load_data(image)?;
    let gemv = module.get_function("q8_0_gemv_warp8")?;
    let stream = Stream::new(igpu.id)?;

    let weight_bytes = (M as usize) * (BLOCKS as usize) * (Q8_0_BLOCK_BYTES as usize);
    let mut weights: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, weight_bytes)?;
    weights.fill_zero()?;
    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(igpu.id, K as usize)?;
    xq.fill_zero()?;
    let mut xs: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, BLOCKS as usize)?;
    xs.fill_zero()?;
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, M as usize)?;
    out.fill_zero()?;

    let mut op = out.raw();
    let mut wp = weights.raw();
    let mut xp = xq.raw();
    let mut sp = xs.raw();
    let in_dim: u32 = K;
    let out_dim: u32 = M;
    let n_blocks: u32 = BLOCKS;
    let mut args: [*mut c_void; 7] = [
        &mut op as *mut _ as *mut c_void,
        &mut wp as *mut _ as *mut c_void,
        &mut xp as *mut _ as *mut c_void,
        &mut sp as *mut _ as *mut c_void,
        &in_dim as *const _ as *mut c_void,
        &out_dim as *const _ as *mut c_void,
        &n_blocks as *const _ as *mut c_void,
    ];
    let cfg = LaunchConfig {
        grid: ((M + 7) / 8, 1, 1),
        block: (256, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm.
    for _ in 0..10 {
        unsafe { gemv.launch_raw(cfg, &stream, &mut args)? };
    }
    stream.synchronize()?;

    let mut samples = Vec::with_capacity(ITERATIONS as usize);
    for _ in 0..ITERATIONS {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        unsafe { gemv.launch_raw(cfg, &stream, &mut args)? };
        end.record(&stream)?;
        end.synchronize()?;
        samples.push(Event::elapsed_ms(&start, &end)? as f64 * 1000.0);
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p50_us = samples[samples.len() / 2];
    let gb_s = weight_bytes as f64 / (p50_us * 1e-6) / 1e9;

    let pj = ProbeJson {
        probed_arch: arch,
        p50_us,
        gb_per_s_p50: gb_s,
    };
    serde_json::to_writer(std::io::stdout(), &pj).wrap_err("serialize probe stdout")?;
    Ok(())
}
