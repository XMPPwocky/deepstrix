//! phase0 — Hardware viability gate dispatcher. See docs/DESIGN.md §8
//! for what each gate measures.

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use tracing_subscriber::EnvFilter;

mod cmd;
mod results;

#[derive(Parser)]
#[command(name = "phase0", about = "deepstrix Phase 0 hardware viability gates")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Print hipcc/ROCm/device info; launch hello kernel on each device.
    Toolchain,
    /// Decompose the cross-device sync floor: empty kernel launch,
    /// same-device double memcpy, host-bounce with pinned/unpinned and
    /// payload sweep. See cmd/pingpong.rs for the layered measurements.
    Pingpong {
        /// Number of timed iterations per (test × payload × variant) cell.
        #[arg(long, default_value_t = 1000)]
        iterations: u32,
    },
    /// Gate A: HSA_OVERRIDE_GFX_VERSION compatibility with dual-device
    /// process. Spawns child probes per-config.
    GateA,
    /// Internal: emit a single ToolchainReport to stdout as JSON. Used by
    /// `gate-a` to read env-dependent state from a fresh process.
    #[command(hide = true)]
    GateAProbe,
    /// Gate C: peer access, peer bandwidth, cross-device event sync RTT,
    /// cache coherency. The decisive test for whether peer-direct beats
    /// host-bounce.
    GateC {
        /// Iterations for RTT measurement.
        #[arg(long, default_value_t = 1000)]
        iterations: u32,
    },
    /// Gate E: effective Q8_0 GEMV bandwidth vs theoretical. Output
    /// efficiency ratio scales all bandwidth-derived perf targets.
    GateE {
        #[arg(long, default_value_t = 200)]
        iterations: u32,
    },
}

fn main() -> eyre::Result<()> {
    v4flash_hip::install_panic_handler()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Toolchain => cmd::toolchain::run(),
        Cmd::Pingpong { iterations } => cmd::pingpong::run(iterations),
        Cmd::GateA => cmd::gate_a::run(),
        Cmd::GateAProbe => cmd::gate_a::probe_to_stdout(),
        Cmd::GateC { iterations } => cmd::gate_c::run(iterations),
        Cmd::GateE { iterations } => cmd::gate_e::run(iterations),
    }
}
