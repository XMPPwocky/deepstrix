//! phase1 — Phase 1 validation harness library.
//!
//! `compare` is the load-bearing module: reads two `(logits.f32,
//! tokens.json)` pairs and produces a comparison report. This is what
//! every Phase 1 milestone M2+ uses to validate ported kernels against
//! the M1 reference output.

pub mod compare;

pub use compare::{ComparisonReport, LogitDump, PerRowStats, compare_logits, load_dump};
