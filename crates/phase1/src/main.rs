//! phase1 — Phase 1 validation harness binary.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use color_eyre::eyre;
use tracing_subscriber::EnvFilter;

use phase1::{ComparisonReport, compare_logits, load_dump};

#[derive(Parser)]
#[command(name = "phase1", about = "deepstrix Phase 1 validation harness")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Compare two captured (logits.f32, tokens.json) dumps and emit a
    /// per-row + aggregate report.
    Compare {
        /// Reference dump directory (the M1 baseline).
        reference: PathBuf,
        /// Candidate dump directory to validate.
        candidate: PathBuf,
        /// Print per-row stats too. Otherwise only aggregates.
        #[arg(long)]
        verbose: bool,
        /// Write JSON report here.
        #[arg(long)]
        json_out: Option<PathBuf>,
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
        Cmd::Compare { reference, candidate, verbose, json_out } => {
            let ref_dump = load_dump(&reference)?;
            let cand_dump = load_dump(&candidate)?;
            let report = compare_logits(&ref_dump, &cand_dump)?;
            print_report(&report, verbose);
            if let Some(out) = json_out {
                std::fs::write(&out, serde_json::to_string_pretty(&report)?)?;
                println!("wrote {}", out.display());
            }
            // Exit non-zero if thresholds fail
            if !report.passes_design_doc_thresholds {
                std::process::exit(1);
            }
            Ok(())
        }
    }
}

fn print_report(r: &ComparisonReport, verbose: bool) {
    println!("== comparison ==");
    println!("  backend ref:   {}", r.backend_ref);
    println!("  backend cand:  {}", r.backend_cand);
    println!("  rows:          {}", r.n_rows);
    println!("  vocab:         {}", r.vocab_size);
    println!("  tokens match:  {}", r.tokens_match);

    let pct = |n: usize| -> f64 {
        100.0 * n as f64 / r.n_rows.max(1) as f64
    };
    println!("  top-1 match:   {} / {} ({:.1}%)", r.n_top1_match, r.n_rows, pct(r.n_top1_match));
    println!("  top-5 match:   {} / {} ({:.1}%)", r.n_top5_match, r.n_rows, pct(r.n_top5_match));
    println!("  mean KL:       {:.6} nats  (doc target: <0.01)", r.mean_kl_ref_to_cand);
    println!("  max  KL:       {:.6} nats", r.max_kl_ref_to_cand);
    println!("  mean max-|Δ|:  {:.6}", r.mean_max_abs_logit_diff);
    println!("  max  max-|Δ|:  {:.6}", r.max_max_abs_logit_diff);
    println!();

    if r.passes_design_doc_thresholds {
        println!("PASS — design doc §9 thresholds met");
    } else {
        println!("FAIL — at least one threshold missed:");
        if (r.n_top1_match as f64) / (r.n_rows as f64) < 0.90 {
            println!("  - top-1 < 90%");
        }
        if (r.n_top5_match as f64) / (r.n_rows as f64) < 1.00 {
            println!("  - top-5 set match < 100%");
        }
        if r.mean_kl_ref_to_cand >= 0.01 {
            println!("  - mean KL ≥ 0.01 nats");
        }
    }

    if verbose {
        println!("\n== per row ==");
        for s in &r.per_row {
            let mark = if s.top1_match { "✓" } else { "✗" };
            println!(
                "  [{}] row {:>3}: argmax_r={} argmax_c={} top5={} KL_rc={:.4} max|Δ|={:.4}",
                mark, s.row, s.argmax_ref, s.argmax_cand,
                if s.top5_match { "✓" } else { "✗" },
                s.kl_ref_to_cand, s.max_abs_logit_diff,
            );
        }
    }
}
