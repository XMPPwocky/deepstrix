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
    /// Measure cross-device round-trip floor via host-bounce memcpy.
    Pingpong {
        /// Number of timed iterations per device pair.
        #[arg(long, default_value_t = 1000)]
        iterations: u32,
        /// Payload size in bytes (must be multiple of 4).
        #[arg(long, default_value_t = 4096)]
        payload_bytes: usize,
    },
    // Gate-A/B/C/D/E land in later commits.
}

fn main() -> eyre::Result<()> {
    v4flash_hip::install_panic_handler()?;
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Toolchain => cmd::toolchain::run(),
        Cmd::Pingpong { iterations, payload_bytes } => {
            cmd::pingpong::run(iterations, payload_bytes)
        }
    }
}
