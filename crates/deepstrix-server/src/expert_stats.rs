//! M62 — on-disk expert-selection aggregate.
//!
//! One sidecar `expert_stats.json` per deepstrix cache root (sibling of
//! `snapshots/`). Deliberately NOT part of the KV snapshot files: expert
//! stats are a global workload aggregate consumed once at model load to
//! rank hot experts, while snapshots are per-prefix, content-addressed,
//! LRU-evicted state — counts stored there would fragment and need a
//! cross-snapshot merge at load.
//!
//! Layout: two banks (prefill / decode), each `N_LAYER × N_EXPERT` u64
//! counts plus the token total they accumulated over. Banks decay by
//! halving once they exceed [`DECAY_TOKENS`] so recent workloads keep
//! steering placement (EWMA over runs); raw counts are stored — placement
//! is computed at load, keeping a future adaptive swapper free to re-rank
//! anytime.

use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre};
use serde::{Deserialize, Serialize};
use v4flash_kernels::config::{N_EXPERT, N_LAYER};

use crate::snapshot::ModelFingerprint;

pub const SCHEMA_VERSION: u32 = 1;
/// Per-bank halving threshold (tokens). ~10M tokens ≈ days of typical use.
pub const DECAY_TOKENS: u64 = 10_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bank {
    pub tokens: u64,
    /// `[N_LAYER × N_EXPERT]` flat counts.
    pub counts: Vec<u64>,
}

impl Bank {
    fn fresh() -> Self {
        Self {
            tokens: 0,
            counts: vec![0; (N_LAYER as usize) * (N_EXPERT as usize)],
        }
    }

    fn merge(&mut self, counts: &[u32], tokens: u64) {
        if tokens == 0 {
            return;
        }
        for (dst, &src) in self.counts.iter_mut().zip(counts) {
            *dst += src as u64;
        }
        self.tokens += tokens;
        if self.tokens > DECAY_TOKENS {
            for c in self.counts.iter_mut() {
                *c /= 2;
            }
            self.tokens /= 2;
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpertStatsAgg {
    pub schema_version: u32,
    pub fingerprint: ModelFingerprint,
    pub prefill: Bank,
    pub decode: Bank,
}

impl ExpertStatsAgg {
    pub fn fresh(fingerprint: ModelFingerprint) -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            fingerprint,
            prefill: Bank::fresh(),
            decode: Bank::fresh(),
        }
    }

    /// Sidecar path: sibling of the snapshots dir (the deepstrix cache root).
    pub fn path_for(snapshot_root: &Path) -> PathBuf {
        snapshot_root
            .parent()
            .unwrap_or(snapshot_root)
            .join("expert_stats.json")
    }

    /// Load, or start fresh on missing file / schema / fingerprint /
    /// shape mismatch (never fails the server over a stats file).
    pub fn load_or_fresh(path: &Path, fingerprint: &ModelFingerprint) -> Self {
        let fresh = || Self::fresh(fingerprint.clone());
        let Ok(bytes) = std::fs::read(path) else { return fresh() };
        match serde_json::from_slice::<Self>(&bytes) {
            Ok(s)
                if s.schema_version == SCHEMA_VERSION
                    && &s.fingerprint == fingerprint
                    && s.prefill.counts.len() == (N_LAYER as usize) * (N_EXPERT as usize)
                    && s.decode.counts.len() == (N_LAYER as usize) * (N_EXPERT as usize) =>
            {
                s
            }
            _ => {
                tracing::warn!(?path, "expert_stats: stale/mismatched file, starting fresh");
                fresh()
            }
        }
    }

    pub fn merge_harvest(
        &mut self,
        prefill_counts: &[u32],
        prefill_tokens: u64,
        decode_counts: &[u32],
        decode_tokens: u64,
    ) {
        self.prefill.merge(prefill_counts, prefill_tokens);
        self.decode.merge(decode_counts, decode_tokens);
    }

    /// Atomic write (tmp + rename).
    pub fn save(&self, path: &Path) -> eyre::Result<()> {
        let tmp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(self)?;
        std::fs::write(&tmp, &bytes).map_err(|e| eyre!("write {:?}: {e}", tmp))?;
        std::fs::rename(&tmp, path).map_err(|e| eyre!("rename {:?}: {e}", path))?;
        Ok(())
    }

    /// Derived placement file path (same dir as the aggregate).
    pub fn placement_path(stats_path: &Path) -> PathBuf {
        stats_path
            .parent()
            .unwrap_or(Path::new("."))
            .join("hot_experts.txt")
    }

    /// Write the placement input the weights loader already understands
    /// (one line per layer, descending `id:count`, top 64 nonzero).
    /// Score = `(1-α)·prefill_freq + α·decode_freq` per token, scaled to
    /// u64 — the loader's global-greedy budget then ranks (layer, expert)
    /// pairs by these. Banks with zero tokens drop out of the mix (their
    /// weight renormalizes onto the other). The kernels loader stays
    /// dependency-free: raw banks live in the JSON, placement is derived
    /// here.
    pub fn write_placement(&self, path: &Path, alpha: f64) -> eyre::Result<()> {
        let n_l = N_LAYER as usize;
        let n_e = N_EXPERT as usize;
        let (wp, wd) = match (self.prefill.tokens, self.decode.tokens) {
            (0, 0) => return Err(eyre!("no tokens accumulated")),
            (_, 0) => (1.0, 0.0),
            (0, _) => (0.0, 1.0),
            _ => (1.0 - alpha, alpha),
        };
        let mut out = String::with_capacity(n_l * 64 * 10);
        for l in 0..n_l {
            let mut row: Vec<(u64, usize)> = (0..n_e)
                .map(|e| {
                    let i = l * n_e + e;
                    let pf = if self.prefill.tokens > 0 {
                        self.prefill.counts[i] as f64 / self.prefill.tokens as f64
                    } else {
                        0.0
                    };
                    let df = if self.decode.tokens > 0 {
                        self.decode.counts[i] as f64 / self.decode.tokens as f64
                    } else {
                        0.0
                    };
                    (((wp * pf + wd * df) * 1e9) as u64, e)
                })
                .filter(|&(s, _)| s > 0)
                .collect();
            row.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            row.truncate(64);
            let line: Vec<String> =
                row.iter().map(|&(s, e)| format!("{e}:{s}")).collect();
            out.push_str(&line.join(","));
            out.push('\n');
        }
        let tmp = path.with_extension("txt.tmp");
        std::fs::write(&tmp, &out).map_err(|e| eyre!("write {:?}: {e}", tmp))?;
        std::fs::rename(&tmp, path).map_err(|e| eyre!("rename {:?}: {e}", path))?;
        Ok(())
    }
}
