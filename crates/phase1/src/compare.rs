//! Compare two captured `(logits.f32, tokens.json)` dumps and produce
//! per-token + aggregate statistics.
//!
//! `logits.f32` layout: row-major float32, shape (n_rows, vocab_size).
//! `tokens.json` schema: matches what `external/ds4-dump/dump_logits.c`
//! writes — see that file for the field set.

use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use color_eyre::eyre::{self, Context, eyre};
use serde::{Deserialize, Serialize};

/// Parsed `tokens.json` companion file. Tolerant about extra fields so
/// future schema additions don't break loading.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TokensJson {
    pub prompt_tokens: Vec<i32>,
    pub generated_tokens: Vec<i32>,
    pub vocab_size: u32,
    #[serde(default)]
    pub backend: String,
    pub n_logit_rows: u32,
}

/// In-memory representation of one logit dump.
#[derive(Debug, Clone)]
pub struct LogitDump {
    pub tokens: TokensJson,
    /// Flat (n_rows × vocab_size) row-major float32.
    pub logits: Vec<f32>,
    pub n_rows: usize,
    pub vocab_size: usize,
}

impl LogitDump {
    pub fn row(&self, i: usize) -> &[f32] {
        let start = i * self.vocab_size;
        &self.logits[start..start + self.vocab_size]
    }
}

/// Read a (tokens.json, logits.f32) pair from a directory.
pub fn load_dump(dir: impl AsRef<Path>) -> eyre::Result<LogitDump> {
    let dir = dir.as_ref();
    let tokens_path = dir.join("tokens.json");
    let logits_path = dir.join("logits.f32");

    let tokens_str = std::fs::read_to_string(&tokens_path)
        .wrap_err_with(|| format!("read {}", tokens_path.display()))?;
    let tokens: TokensJson = serde_json::from_str(&tokens_str)
        .wrap_err_with(|| format!("parse {}", tokens_path.display()))?;

    let logits = read_f32_file(&logits_path)
        .wrap_err_with(|| format!("read {}", logits_path.display()))?;

    let vocab_size = tokens.vocab_size as usize;
    let n_rows = tokens.n_logit_rows as usize;
    let expected = n_rows.checked_mul(vocab_size).ok_or_else(|| {
        eyre!("n_rows * vocab_size overflow ({} * {})", n_rows, vocab_size)
    })?;
    if logits.len() != expected {
        return Err(eyre!(
            "logits.f32 size mismatch in {}: expected {} floats ({}×{}), got {}",
            dir.display(), expected, n_rows, vocab_size, logits.len()
        ));
    }

    Ok(LogitDump { tokens, logits, n_rows, vocab_size })
}

fn read_f32_file(path: &Path) -> eyre::Result<Vec<f32>> {
    let f = File::open(path)?;
    let mut r = BufReader::with_capacity(1 << 20, f);
    let mut buf = Vec::new();
    r.read_to_end(&mut buf)?;
    if buf.len() % 4 != 0 {
        return Err(eyre!(
            "logits file is not a whole number of f32s: {} bytes",
            buf.len()
        ));
    }
    let n_floats = buf.len() / 4;
    let mut out = Vec::with_capacity(n_floats);
    for chunk in buf.chunks_exact(4) {
        let bytes: [u8; 4] = chunk.try_into().unwrap();
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

/// Per-row comparison statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerRowStats {
    pub row: usize,
    pub argmax_ref: i32,
    pub argmax_cand: i32,
    pub top1_match: bool,
    pub top5_match: bool,
    /// KL(P_ref || P_cand) in nats.
    pub kl_ref_to_cand: f64,
    /// KL(P_cand || P_ref) in nats — symmetric reporting.
    pub kl_cand_to_ref: f64,
    pub max_abs_logit_diff: f32,
    /// The top-5 ids from the reference, in descending logit order.
    pub ref_top5: Vec<i32>,
}

/// Aggregate report across all rows.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub n_rows: usize,
    pub vocab_size: usize,
    pub backend_ref: String,
    pub backend_cand: String,
    /// Tokens match exactly (greedy decode produced identical tokens).
    pub tokens_match: bool,
    pub n_top1_match: usize,
    pub n_top5_match: usize,
    pub mean_kl_ref_to_cand: f64,
    pub max_kl_ref_to_cand: f64,
    pub mean_max_abs_logit_diff: f64,
    pub max_max_abs_logit_diff: f32,
    /// Pass/fail against the doc's §9 thresholds.
    pub passes_design_doc_thresholds: bool,
    pub per_row: Vec<PerRowStats>,
}

impl ComparisonReport {
    /// Doc §9 floor metrics: greedy top-1 ≥ 90%, top-5 set match 100%,
    /// KL < 0.01 nats/token.
    pub fn check_thresholds(&self) -> bool {
        let top1_pct = self.n_top1_match as f64 / self.n_rows.max(1) as f64;
        let top5_pct = self.n_top5_match as f64 / self.n_rows.max(1) as f64;
        top1_pct >= 0.90 && top5_pct >= 1.00 && self.mean_kl_ref_to_cand < 0.01
    }
}

/// Run the comparison. Caller decides what to do with the report.
pub fn compare_logits(reference: &LogitDump, candidate: &LogitDump) -> eyre::Result<ComparisonReport> {
    if reference.vocab_size != candidate.vocab_size {
        return Err(eyre!(
            "vocab size mismatch: ref={} cand={}",
            reference.vocab_size, candidate.vocab_size
        ));
    }
    if reference.n_rows != candidate.n_rows {
        return Err(eyre!(
            "row count mismatch: ref={} cand={}",
            reference.n_rows, candidate.n_rows
        ));
    }

    let tokens_match = reference.tokens.generated_tokens == candidate.tokens.generated_tokens;

    let mut per_row = Vec::with_capacity(reference.n_rows);
    let mut n_top1 = 0;
    let mut n_top5 = 0;
    let mut sum_kl_rc = 0.0_f64;
    let mut max_kl_rc = 0.0_f64;
    let mut sum_max_diff = 0.0_f64;
    let mut max_max_diff = 0.0_f32;

    for row_idx in 0..reference.n_rows {
        let r = reference.row(row_idx);
        let c = candidate.row(row_idx);

        let (argmax_r, ref_top5_ids) = top_k_ids(r, 5);
        let (argmax_c, _) = top_k_ids(c, 5);
        let (_, cand_top5_ids) = top_k_ids(c, 5);

        let top1_match = argmax_r == argmax_c;
        let ref_top5_set: std::collections::BTreeSet<i32> = ref_top5_ids.iter().copied().collect();
        let cand_top5_set: std::collections::BTreeSet<i32> = cand_top5_ids.iter().copied().collect();
        let top5_match = ref_top5_set == cand_top5_set;

        if top1_match { n_top1 += 1; }
        if top5_match { n_top5 += 1; }

        let kl_rc = kl_divergence(r, c);
        let kl_cr = kl_divergence(c, r);
        sum_kl_rc += kl_rc;
        if kl_rc > max_kl_rc { max_kl_rc = kl_rc; }

        let max_abs = r.iter().zip(c.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        sum_max_diff += max_abs as f64;
        if max_abs > max_max_diff { max_max_diff = max_abs; }

        per_row.push(PerRowStats {
            row: row_idx,
            argmax_ref: argmax_r,
            argmax_cand: argmax_c,
            top1_match,
            top5_match,
            kl_ref_to_cand: kl_rc,
            kl_cand_to_ref: kl_cr,
            max_abs_logit_diff: max_abs,
            ref_top5: ref_top5_ids,
        });
    }

    let n = reference.n_rows.max(1) as f64;
    let mut report = ComparisonReport {
        n_rows: reference.n_rows,
        vocab_size: reference.vocab_size,
        backend_ref: reference.tokens.backend.clone(),
        backend_cand: candidate.tokens.backend.clone(),
        tokens_match,
        n_top1_match: n_top1,
        n_top5_match: n_top5,
        mean_kl_ref_to_cand: sum_kl_rc / n,
        max_kl_ref_to_cand: max_kl_rc,
        mean_max_abs_logit_diff: sum_max_diff / n,
        max_max_abs_logit_diff: max_max_diff,
        passes_design_doc_thresholds: false, // filled below
        per_row,
    };
    report.passes_design_doc_thresholds = report.check_thresholds();
    Ok(report)
}

/// Return (argmax_id, top_k_ids) where `top_k_ids` is sorted by
/// descending logit. argmax_id is the first element of top_k_ids.
fn top_k_ids(logits: &[f32], k: usize) -> (i32, Vec<i32>) {
    let k = k.min(logits.len());
    let mut top: Vec<(f32, i32)> = Vec::with_capacity(k);
    for (i, &v) in logits.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        if top.len() < k {
            top.push((v, i as i32));
            // keep sorted descending after each push
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        } else if v > top[k - 1].0 {
            top[k - 1] = (v, i as i32);
            top.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        }
    }
    let ids: Vec<i32> = top.iter().map(|(_, i)| *i).collect();
    let argmax = ids.first().copied().unwrap_or(-1);
    (argmax, ids)
}

/// Numerically stable KL divergence between two logit vectors, in nats.
/// Computes KL(softmax(p_logits) || softmax(q_logits)).
fn kl_divergence(p_logits: &[f32], q_logits: &[f32]) -> f64 {
    debug_assert_eq!(p_logits.len(), q_logits.len());

    // Find max for each (stability).
    let p_max = p_logits.iter().filter(|x| x.is_finite()).copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let q_max = q_logits.iter().filter(|x| x.is_finite()).copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if !p_max.is_finite() || !q_max.is_finite() {
        return 0.0;
    }

    // logsumexp for normalization
    let p_sum: f64 = p_logits.iter()
        .filter(|x| x.is_finite())
        .map(|&x| ((x - p_max) as f64).exp())
        .sum();
    let q_sum: f64 = q_logits.iter()
        .filter(|x| x.is_finite())
        .map(|&x| ((x - q_max) as f64).exp())
        .sum();
    let p_log_z = (p_max as f64) + p_sum.ln();
    let q_log_z = (q_max as f64) + q_sum.ln();

    // KL = sum_i P(i) * (log P(i) - log Q(i))
    //    = sum_i P(i) * ((p_logits[i] - p_log_z) - (q_logits[i] - q_log_z))
    let mut kl = 0.0_f64;
    for (&pl, &ql) in p_logits.iter().zip(q_logits.iter()) {
        if !pl.is_finite() {
            continue;
        }
        let p_log = (pl as f64) - p_log_z;
        let p = p_log.exp();
        if p == 0.0 { continue; }
        let q_log = if ql.is_finite() { (ql as f64) - q_log_z } else { f64::NEG_INFINITY };
        kl += p * (p_log - q_log);
    }
    kl
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dump(logits: Vec<f32>, vocab: usize, n_rows: usize, backend: &str) -> LogitDump {
        LogitDump {
            tokens: TokensJson {
                prompt_tokens: vec![],
                generated_tokens: vec![0; n_rows],
                vocab_size: vocab as u32,
                backend: backend.into(),
                n_logit_rows: n_rows as u32,
            },
            logits,
            n_rows,
            vocab_size: vocab,
        }
    }

    #[test]
    fn identical_logits_have_zero_kl() {
        let logits = vec![0.0, 1.0, 2.0, 3.0];
        let a = make_dump(logits.clone(), 4, 1, "ref");
        let b = make_dump(logits, 4, 1, "cand");
        let report = compare_logits(&a, &b).unwrap();
        assert_eq!(report.n_top1_match, 1);
        assert_eq!(report.n_top5_match, 1);
        assert!(report.mean_kl_ref_to_cand < 1e-9);
        assert!(report.max_max_abs_logit_diff < 1e-9);
        assert!(report.passes_design_doc_thresholds);
    }

    #[test]
    fn divergent_logits_fail_thresholds() {
        // Two rows, completely different argmaxes
        let a = make_dump(vec![10.0, 0.0, 0.0, 0.0, 0.0,
                              10.0, 0.0, 0.0, 0.0, 0.0], 5, 2, "ref");
        let b = make_dump(vec![0.0, 0.0, 0.0, 0.0, 10.0,
                              0.0, 0.0, 0.0, 0.0, 10.0], 5, 2, "cand");
        let report = compare_logits(&a, &b).unwrap();
        assert_eq!(report.n_top1_match, 0);
        // top5 set is whole vocab in this case (k=min(5, 5)) so they match
        assert_eq!(report.n_top5_match, 2);
        assert!(!report.passes_design_doc_thresholds);
        assert!(report.mean_kl_ref_to_cand > 5.0); // strong divergence
    }

    #[test]
    fn small_perturbation_passes_thresholds() {
        // Sharp 5-vocab distribution, identical argmax + same top-5 set,
        // small float noise on every logit.
        let mut a_l = Vec::new();
        let mut b_l = Vec::new();
        for row in 0..50 {
            // Each row: argmax at (row % 5)
            for i in 0..5 {
                let base = if i == row % 5 { 10.0 } else { 0.0 };
                a_l.push(base);
                b_l.push(base + 0.001 * (i as f32 + 1.0));
            }
        }
        let a = make_dump(a_l, 5, 50, "ref");
        let b = make_dump(b_l, 5, 50, "cand");
        let report = compare_logits(&a, &b).unwrap();
        assert_eq!(report.n_top1_match, 50);
        assert_eq!(report.n_top5_match, 50);
        assert!(report.mean_kl_ref_to_cand < 0.001);
        assert!(report.passes_design_doc_thresholds);
    }

    #[test]
    fn top5_set_match_is_set_not_order() {
        // Same top-5 ids but in different order — set match should hold.
        let a_l = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.0]; // top5: [0,1,2,3,4]
        let b_l = vec![1.0, 5.0, 4.0, 3.0, 2.0, 0.0]; // top5: [1,2,3,4,0] = same set
        let a = make_dump(a_l, 6, 1, "ref");
        let b = make_dump(b_l, 6, 1, "cand");
        let report = compare_logits(&a, &b).unwrap();
        assert_eq!(report.n_top5_match, 1);
    }

    #[test]
    fn top_k_ids_returns_descending() {
        let logits = vec![1.0, 5.0, 3.0, 4.0, 2.0];
        let (argmax, top5) = top_k_ids(&logits, 5);
        assert_eq!(argmax, 1);
        assert_eq!(top5, vec![1, 3, 2, 4, 0]);
    }

    #[test]
    fn ignores_non_finite_logits() {
        let logits = vec![f32::NEG_INFINITY, 1.0, f32::NAN, 2.0, 0.0];
        let (argmax, _) = top_k_ids(&logits, 3);
        assert_eq!(argmax, 3);
    }
}
