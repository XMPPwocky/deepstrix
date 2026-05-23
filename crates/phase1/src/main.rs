//! phase1 — validation harness for Phase 1 milestones.
//!
//! M1: drives the captured-reference workflow against ds4-rocm. Later
//! milestones will use this binary to byte-compare ported kernels
//! against the M1 reference output.

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(name = "phase1", about = "deepstrix Phase 1 validation harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Stub. Subcommands land as later milestones need them
    /// (capture-reference, compare-logits, run-kernel-vs-reference, ...).
    Stub,
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
        Cmd::Stub => {
            println!("phase1: harness skeleton — subcommands land in later milestones");
            Ok(())
        }
    }
}
