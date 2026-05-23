//! Gate A — HSA override compatibility on dual-device process.
//!
//! `HSA_OVERRIDE_GFX_VERSION` is a process-global env var consulted by HSA
//! at runtime initialization. We can't toggle it after the fact, so the
//! gate spawns one child process per (override-state) and aggregates.
//!
//! What we want to know:
//!  - **No override (native):** does ROCm 7.2.3 enumerate both devices
//!    with their true gfx archs AND launch arch-matched kernels on each?
//!  - **Override to 11.5.1 (Strix):** what does the runtime report for
//!    each device? Does the dGPU now claim gfx1151? Does its kernel still
//!    launch?
//!  - **Override to 11.0.0 (gfx1100 fallback):** same questions; this is
//!    the configuration Gate B wants if the gfx1100-binary-on-gfx1151 path
//!    is meaningfully faster.
//!
//! Each child uses the existing `phase0 toolchain` machinery (via an
//! internal `gate-a-probe` subcommand) and writes JSON to stdout, which
//! the parent parses.

use std::env;
use std::process::Command;

use color_eyre::eyre::{self, Context};
use serde::{Deserialize, Serialize};

use crate::cmd::toolchain::{DeviceReport, ToolchainReport};
use crate::results;

#[derive(Serialize, Deserialize)]
pub struct GateAReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub probes: Vec<ProbeOutcome>,
    pub decision: Decision,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProbeOutcome {
    pub config_label: String,
    pub hsa_override: Option<String>,
    pub exit_ok: bool,
    pub stderr_excerpt: String,
    pub devices: Vec<ProbeDevice>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct ProbeDevice {
    pub id: i32,
    pub name: String,
    pub gcn_arch_name: String,
    pub kernel_loaded: bool,
    pub kernel_result: Option<i32>,
    pub kernel_error: Option<String>,
}

#[derive(Serialize, Deserialize)]
pub struct Decision {
    pub native_works: bool,
    pub override_safe_on_dgpu: bool,
    pub override_safe_on_igpu: bool,
    pub recommendation: String,
    pub rationale: String,
}

const CONFIGS: &[(&str, Option<&str>)] = &[
    ("native_no_override", None),
    ("override_11_5_1_strix", Some("11.5.1")),
    ("override_11_0_0_gfx1100", Some("11.0.0")),
];

pub fn run() -> eyre::Result<()> {
    // Locate our own binary path so children re-invoke this binary.
    let self_exe = env::current_exe().wrap_err("current_exe")?;

    let mut probes = Vec::new();
    for (label, override_val) in CONFIGS {
        println!("\n== probing {label} (HSA_OVERRIDE_GFX_VERSION={override_val:?}) ==");
        let outcome = run_probe(&self_exe, label, *override_val)?;
        summarize(&outcome);
        probes.push(outcome);
    }

    let decision = decide(&probes);
    println!("\n== decision ==");
    println!("native_works:          {}", decision.native_works);
    println!("override_safe_on_dGPU: {}", decision.override_safe_on_dgpu);
    println!("override_safe_on_iGPU: {}", decision.override_safe_on_igpu);
    println!("recommendation:        {}", decision.recommendation);
    println!("rationale:             {}", decision.rationale);

    let report = GateAReport {
        gate: "gate_a",
        timestamp: results::now_unix(),
        probes,
        decision,
    };
    let path = results::write("gate_a", &report)?;
    println!("wrote {}", path.display());
    Ok(())
}

fn run_probe(
    self_exe: &std::path::Path,
    label: &str,
    override_val: Option<&str>,
) -> eyre::Result<ProbeOutcome> {
    let mut cmd = Command::new(self_exe);
    cmd.arg("gate-a-probe");
    cmd.env_remove("HSA_OVERRIDE_GFX_VERSION");
    if let Some(v) = override_val {
        cmd.env("HSA_OVERRIDE_GFX_VERSION", v);
    }
    let output = cmd.output().wrap_err_with(|| format!("spawn probe {label}"))?;

    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let stderr_excerpt: String = stderr.lines().take(10).collect::<Vec<_>>().join("\n");

    if !output.status.success() {
        return Ok(ProbeOutcome {
            config_label: label.into(),
            hsa_override: override_val.map(String::from),
            exit_ok: false,
            stderr_excerpt,
            devices: Vec::new(),
        });
    }

    // Stdout is a JSON ToolchainReport.
    let report: ToolchainReport = serde_json::from_slice(&output.stdout)
        .wrap_err_with(|| format!("parse probe stdout for {label}"))?;

    let devices = report
        .devices
        .into_iter()
        .map(|d: DeviceReport| ProbeDevice {
            id: d.id,
            name: d.name,
            gcn_arch_name: d.gcn_arch_name,
            kernel_loaded: d.hello_kernel_loaded,
            kernel_result: d.hello_kernel_result,
            kernel_error: d.hello_kernel_error,
        })
        .collect();

    Ok(ProbeOutcome {
        config_label: label.into(),
        hsa_override: override_val.map(String::from),
        exit_ok: true,
        stderr_excerpt,
        devices,
    })
}

fn summarize(probe: &ProbeOutcome) {
    if !probe.exit_ok {
        println!("    probe exited non-zero. stderr excerpt:");
        for line in probe.stderr_excerpt.lines() {
            println!("    | {line}");
        }
        return;
    }
    for d in &probe.devices {
        let kernel = if d.kernel_loaded {
            format!("OK ({})", d.kernel_result.unwrap_or(-1))
        } else if let Some(e) = &d.kernel_error {
            format!("FAIL ({e})")
        } else {
            "FAIL (no detail)".into()
        };
        println!("    [{}] {} → {} | kernel: {}", d.id, d.name, d.gcn_arch_name, kernel);
    }
}

fn decide(probes: &[ProbeOutcome]) -> Decision {
    let native = probes.iter().find(|p| p.config_label == "native_no_override");
    let override_strix = probes
        .iter()
        .find(|p| p.config_label == "override_11_5_1_strix");

    let native_works = native
        .map(|p| {
            p.exit_ok
                && !p.devices.is_empty()
                && p.devices.iter().all(|d| d.kernel_loaded)
        })
        .unwrap_or(false);

    // Under override, check each device class. We assume one dGPU
    // (gfx12xx) and one iGPU (gfx11xx). Devices identified by their
    // *native* gcnArchName (from the native probe).
    let dgpu_native_arch = native
        .and_then(|p| {
            p.devices
                .iter()
                .find(|d| d.gcn_arch_name.starts_with("gfx12"))
                .map(|d| (d.id, d.gcn_arch_name.clone()))
        });
    let igpu_native_arch = native
        .and_then(|p| {
            p.devices
                .iter()
                .find(|d| d.gcn_arch_name.starts_with("gfx11"))
                .map(|d| (d.id, d.gcn_arch_name.clone()))
        });

    let override_safe_on_dgpu = match (override_strix, &dgpu_native_arch) {
        (Some(p), Some((id, _))) if p.exit_ok => p
            .devices
            .iter()
            .find(|d| d.id == *id)
            .map(|d| d.kernel_loaded)
            .unwrap_or(false),
        _ => false,
    };

    let override_safe_on_igpu = match (override_strix, &igpu_native_arch) {
        (Some(p), Some((id, _))) if p.exit_ok => p
            .devices
            .iter()
            .find(|d| d.id == *id)
            .map(|d| d.kernel_loaded)
            .unwrap_or(false),
        _ => false,
    };

    let (recommendation, rationale) = match (
        native_works,
        override_safe_on_dgpu,
        override_safe_on_igpu,
    ) {
        (true, _, _) => (
            "single-process, no override needed".to_string(),
            "ROCm 7.2.3 natively enumerates and launches kernels on both gfx1201 (dGPU) \
             and gfx1151 (iGPU) without HSA_OVERRIDE_GFX_VERSION. Override is unnecessary \
             for baseline; Gate B may want it selectively for gfx1100-on-gfx1151 perf."
                .into(),
        ),
        (false, true, true) => (
            "single-process with override=11.5.1".to_string(),
            "Native enumeration failed; override is compatible with both devices."
                .into(),
        ),
        (false, false, true) => (
            "two-process or drop override".to_string(),
            "Override poisons the dGPU. Single-process design with override won't work; \
             either drop override (and accept whatever native gfx1151 perf gives us) or \
             split into two processes with different envs."
                .into(),
        ),
        (false, true, false) => (
            "drop override, native only".to_string(),
            "Native works for dGPU but override breaks iGPU. Suspicious; investigate."
                .into(),
        ),
        (false, false, false) => (
            "stuck — investigate".to_string(),
            "Neither native nor override produces working kernels on both devices."
                .into(),
        ),
    };

    Decision {
        native_works,
        override_safe_on_dgpu,
        override_safe_on_igpu,
        recommendation,
        rationale,
    }
}

/// Internal subcommand: probe the current process's env, write JSON to stdout.
/// Called by gate-a's spawned children. Output schema matches
/// `cmd::toolchain::ToolchainReport`.
pub fn probe_to_stdout() -> eyre::Result<()> {
    let report = crate::cmd::toolchain::collect_report(false)?;
    serde_json::to_writer(std::io::stdout(), &report).wrap_err("serialize probe stdout")?;
    Ok(())
}

