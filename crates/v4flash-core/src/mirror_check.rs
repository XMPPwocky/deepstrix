//! Does the EXPERT MIRROR (`V41_EXPERT_MIRROR_DIR`) hold the SAME checkpoint as the primary (`V41_MODEL_DIR`)?
//!
//! Every mirror read (box 1's striped layout reads, `V41_EXPERT_MIRROR_LAYOUT` / `_FRAC`; the hedged expert scale and
//! Engram row reads, `V41_EXPERT_HEDGE_MS` / `V41_ENGRAM_HEDGE_MS`; box 2's split / striped / urgency routes) answers
//! for a byte range of the PRIMARY's shard. Until 10-18 a mirror shard was accepted on FILE SIZE alone (the hedge
//! handles also compared the header and 17 blocks). DeepSeek-V4.1-Flash-1017 has the same tensor names, the same
//! safetensors headers and the same shard sizes as the model it replaces, with different bytes in every shard: a
//! primary repointed at it beside an old (or half-copied) mirror would have served OLD-model experts on ~40% of reads,
//! silently.
//!
//! A mirror shard is accepted only when, against the same-named primary shard,
//!   1. both exist and have the same size,
//!   2. the safetensors header (8-byte length + JSON: names, dtypes, shapes, data offsets) is byte-identical, and
//!   3. `SAMPLES` windows of `SAMPLE_BYTES` at fixed, 4 KiB-aligned offsets spread over the data section (the first
//!      window starts at the data, the last ends at EOF) are byte-identical (O_DIRECT reads where the filesystem
//!      takes them, so the probe neither fills nor drops the page cache).
//! And the whole mirror is refused when the two DIRECTORIES disagree on identity: the HF repo + revision (from
//! `.hf_fetch_rev` "repo rev", or an HF cache path `models--ORG--NAME/snapshots/REV`) when both sides carry one, and
//! `config.json` / `model.safetensors.index.json` bytes when both sides have them.
//!
//! A refusal is never silent: one `MIRROR REJECTED` line naming the shard and the broken invariant (first verification
//! in the process), the per-open summary in `SafetensorsDir::open`, and the counters below (the hub logs them at
//! startup, on `/readyz` and in the `b1 hedged reads` line). Reads of a refused shard go to the primary only.
//!
//! Cost per shard: two header reads (page-cache warm on the primary: it was just parsed) plus `2 * SAMPLES` 64 KiB
//! reads, i.e. 48 shards ~ 768 random 64 KiB reads + 96 header reads, well under a second cold; repeated opens in one
//! process (the hub opens the checkpoint 2-3 times) reuse the verdict while neither file's size or mtime has changed.
//!
//! What it cannot see: a mirror that differs only OUTSIDE the sampled windows (a partial in-place rewrite that left
//! every sample, the header and the size intact), and a mirror rewritten AFTER the hub verified it.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use crate::direct_io::{align_down, align_up, AlignedBuf, DIRECT_ALIGN};

/// Data windows compared per shard.
pub const SAMPLES: usize = 8;
/// Bytes per window (a multiple of `DIRECT_ALIGN`).
pub const SAMPLE_BYTES: usize = 64 << 10;

/// Mirror shards accepted (distinct primary/mirror pairs verified in this process).
pub static SHARDS_ACCEPTED: AtomicU64 = AtomicU64::new(0);
/// Mirror shards present but NOT a copy of the primary's (size, header, sampled data, unreadable, or the whole mirror
/// dir refused on identity): the hazard counter -- nonzero means a mirror is stale or half-copied.
pub static SHARDS_MISMATCHED: AtomicU64 = AtomicU64::new(0);
/// Mirror shards absent from the mirror dir.
pub static SHARDS_MISSING: AtomicU64 = AtomicU64::new(0);

/// `(accepted, mismatched, missing)`.
pub fn counters() -> (u64, u64, u64) {
    (SHARDS_ACCEPTED.load(Relaxed), SHARDS_MISMATCHED.load(Relaxed), SHARDS_MISSING.load(Relaxed))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Same size, header and sampled data.
    Same,
    /// No such file in the mirror dir.
    Missing,
    /// Present, but not a copy of the primary's shard: the broken invariant.
    Differs(String),
}

/// The HF identity `"repo@rev"` of a checkpoint dir: `.hf_fetch_rev` ("<repo> <rev>"), else an HF cache snapshot
/// path `.../models--ORG--NAME/snapshots/REV`. `None` when the dir carries neither.
pub fn dir_identity(dir: &Path) -> Option<String> {
    if let Ok(s) = std::fs::read_to_string(dir.join(".hf_fetch_rev")) {
        let mut it = s.split_whitespace();
        if let (Some(repo), Some(rev)) = (it.next(), it.next()) {
            return Some(format!("{repo}@{rev}"));
        }
    }
    let rev = dir.file_name()?.to_str()?;
    let snaps = dir.parent()?;
    if snaps.file_name()? != "snapshots" {
        return None;
    }
    let repo = snaps.parent()?.file_name()?.to_str()?.strip_prefix("models--")?.replacen("--", "/", 1);
    Some(format!("{repo}@{rev}"))
}

/// Directory-level identity: `Err(reason)` refuses the whole mirror.
pub fn check_dirs(primary: &Path, mirror: &Path) -> Result<(), String> {
    if let (Some(a), Some(b)) = (dir_identity(primary), dir_identity(mirror)) {
        if a != b {
            return Err(format!("HF identity differs: primary {a}, mirror {b}"));
        }
    }
    for f in ["config.json", "model.safetensors.index.json"] {
        if let (Ok(a), Ok(b)) = (std::fs::read(primary.join(f)), std::fs::read(mirror.join(f))) {
            if a != b {
                return Err(format!("{f} differs ({} vs {} bytes{})", a.len(), b.len(), first_diff(&a, &b).map_or(String::new(), |i| format!(", first at byte {i}"))));
            }
        }
    }
    Ok(())
}

/// Offsets of the sampled windows of a `len`-byte shard whose data starts at `data_start`: 4 KiB-aligned, each
/// `SAMPLE_BYTES` long (the last one may run past EOF; only the bytes before EOF are compared). One window when the
/// data section is shorter than `SAMPLES` windows' spread allows.
pub fn sample_windows(data_start: u64, len: u64) -> Vec<u64> {
    let base = align_down(data_start.min(len));
    let top = align_up(len);
    let w = SAMPLE_BYTES as u64;
    if top <= base + w {
        return vec![base];
    }
    let room = top - base - w; // aligned
    let mut v: Vec<u64> = (0..SAMPLES as u64).map(|k| base + align_down(room * k / (SAMPLES as u64 - 1))).collect();
    v.dedup();
    v
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b).position(|(x, y)| x != y).or((a.len() != b.len()).then(|| a.len().min(b.len())))
}

/// The 8-byte length plus the JSON header.
fn header(path: &Path, len: u64) -> Result<Vec<u8>, String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    if len < 8 {
        return Err(format!("{} is {len} bytes: shorter than a safetensors header length", path.display()));
    }
    let mut n = [0u8; 8];
    f.read_exact(&mut n).map_err(|e| format!("{}: header length: {e}", path.display()))?;
    let n = u64::from_le_bytes(n);
    if n > 256 << 20 || 8 + n > len {
        return Err(format!("{}: header length {n} is implausible for a {len}-byte file", path.display()));
    }
    let mut h = vec![0u8; 8 + n as usize];
    h[..8].copy_from_slice(&n.to_le_bytes());
    f.read_exact(&mut h[8..]).map_err(|e| format!("{}: header bytes: {e}", path.display()))?;
    Ok(h)
}

/// Compare one mirror shard with its primary (no caching, no logging, no counters).
pub fn check_shard(primary: &Path, mirror: &Path) -> Verdict {
    let mlen = match std::fs::metadata(mirror) {
        Ok(m) => m.len(),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Verdict::Missing,
        Err(e) => return Verdict::Differs(format!("stat: {e}")),
    };
    let plen = match std::fs::metadata(primary) {
        Ok(m) => m.len(),
        Err(e) => return Verdict::Differs(format!("primary stat: {e}")),
    };
    if plen != mlen {
        return Verdict::Differs(format!("size {mlen} B, primary {plen} B (short / half-copied?)"));
    }
    let (hp, hm) = match (header(primary, plen), header(mirror, mlen)) {
        (Ok(a), Ok(b)) => (a, b),
        (Err(e), _) => return Verdict::Differs(format!("primary unreadable: {e}")),
        (_, Err(e)) => return Verdict::Differs(e),
    };
    if hp != hm {
        return Verdict::Differs(format!("safetensors header differs ({} vs {} bytes, first at byte {})", hm.len(), hp.len(), first_diff(&hp, &hm).unwrap_or(0)));
    }
    let (fp, fm) = match (crate::direct_io::open_direct(primary), crate::direct_io::open_direct(mirror)) {
        (Ok((a, _)), Ok((b, _))) => (a, b),
        (Err(e), _) => return Verdict::Differs(format!("primary open: {e:#}")),
        (_, Err(e)) => return Verdict::Differs(format!("open: {e:#}")),
    };
    let (mut a, mut b) = (AlignedBuf::new(SAMPLE_BYTES), AlignedBuf::new(SAMPLE_BYTES));
    debug_assert_eq!(SAMPLE_BYTES % DIRECT_ALIGN, 0);
    let wins = sample_windows(hp.len() as u64, plen);
    for (k, &off) in wins.iter().enumerate() {
        let n = (SAMPLE_BYTES as u64).min(plen - off) as usize;
        a.as_mut_slice().fill(0);
        b.as_mut_slice().fill(0);
        if let Err(e) = crate::direct_io::read_full(&fp, off, a.as_mut_slice()) {
            return Verdict::Differs(format!("primary sample read at {off}: {e:#}"));
        }
        if let Err(e) = crate::direct_io::read_full(&fm, off, b.as_mut_slice()) {
            return Verdict::Differs(format!("sample read at {off}: {e:#}"));
        }
        if let Some(i) = first_diff(&a.as_slice()[..n], &b.as_slice()[..n]) {
            return Verdict::Differs(format!("data differs at byte {} (sample {} of {}, same size + same header: a DIFFERENT checkpoint)", off + i as u64, k + 1, wins.len()));
        }
    }
    Verdict::Same
}

type Stamp = Option<(u64, std::time::SystemTime)>;
fn stamp(p: &Path) -> Stamp {
    std::fs::metadata(p).ok().map(|m| (m.len(), m.modified().unwrap_or(std::time::UNIX_EPOCH)))
}

static SHARD_MEMO: Mutex<Option<HashMap<(PathBuf, PathBuf, Stamp, Stamp), Verdict>>> = Mutex::new(None);
static DIR_MEMO: Mutex<Option<HashMap<(PathBuf, PathBuf), Result<(), String>>>> = Mutex::new(None);

/// `check_shard`, once per (primary, mirror, both sizes + mtimes) in this process; the first verdict is logged
/// (`MIRROR REJECTED` for a mismatch) and counted.
pub fn verify_shard(primary: &Path, mirror: &Path) -> Verdict {
    let key = (primary.to_path_buf(), mirror.to_path_buf(), stamp(primary), stamp(mirror));
    let mut g = SHARD_MEMO.lock().unwrap_or_else(|p| p.into_inner());
    let memo = g.get_or_insert_with(HashMap::new);
    if let Some(v) = memo.get(&key) {
        return v.clone();
    }
    let v = check_shard(primary, mirror);
    match &v {
        Verdict::Same => SHARDS_ACCEPTED.fetch_add(1, Relaxed),
        Verdict::Missing => {
            // A PARTIAL mirror is legitimate (box 2's /weights2 holds the expert shards, not the hub-only Engram ones).
            eprintln!("safetensors: expert mirror has no {} -- every read of this shard uses the primary only", mirror.display());
            SHARDS_MISSING.fetch_add(1, Relaxed)
        }
        Verdict::Differs(why) => {
            eprintln!(
                "safetensors: MIRROR REJECTED {}: {why} -- it is not a copy of {}; every read of this shard uses the primary only",
                mirror.display(),
                primary.display()
            );
            SHARDS_MISMATCHED.fetch_add(1, Relaxed)
        }
    };
    memo.insert(key, v.clone());
    v
}

/// `check_dirs`, once per (primary dir, mirror dir) in this process; a refusal is logged and counts all `n_shards`
/// as mismatched.
pub fn verify_dirs(primary: &Path, mirror: &Path, n_shards: usize) -> Result<(), String> {
    let key = (primary.to_path_buf(), mirror.to_path_buf());
    let mut g = DIR_MEMO.lock().unwrap_or_else(|p| p.into_inner());
    let memo = g.get_or_insert_with(HashMap::new);
    if let Some(v) = memo.get(&key) {
        return v.clone();
    }
    let v = check_dirs(primary, mirror);
    if let Err(why) = &v {
        eprintln!(
            "safetensors: MIRROR REJECTED (whole dir) {}: {why} -- it is not a copy of {}; all {n_shards} shards read the primary only",
            mirror.display(),
            primary.display()
        );
        SHARDS_MISMATCHED.fetch_add(n_shards as u64, Relaxed);
    }
    let (pi, mi) = (dir_identity(primary), dir_identity(mirror));
    if pi.is_none() || mi.is_none() {
        eprintln!(
            "safetensors: mirror identity: primary {} / mirror {} (unknown on a side: acceptance rests on the per-shard header + sampled data)",
            pi.as_deref().unwrap_or("?"),
            mi.as_deref().unwrap_or("?")
        );
    }
    memo.insert(key, v.clone());
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn windows_are_aligned_cover_both_ends_and_stay_in_range() {
        for (ds, len) in [(8u64 + 1000, 10u64 << 30), (8 + 4088, 3 << 20), (300_000, 7 << 30 | 12345), (24, 5000), (100, 100)] {
            let w = sample_windows(ds, len);
            assert!(!w.is_empty());
            assert!(w.iter().all(|o| o % DIRECT_ALIGN as u64 == 0 && *o < len.max(1)), "{ds} {len}: {w:?}");
            assert!(w[0] <= ds && ds < w[0] + DIRECT_ALIGN as u64 || len <= ds, "{ds} {len}: first window at the data");
            assert!(*w.last().unwrap() + SAMPLE_BYTES as u64 >= len, "{ds} {len}: last window reaches EOF");
            if len - ds > (SAMPLES * SAMPLE_BYTES) as u64 {
                assert_eq!(w.len(), SAMPLES);
            }
        }
    }

    #[test]
    fn hf_identity_from_cache_path_and_fetch_rev() {
        let p = Path::new("/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a");
        assert_eq!(dir_identity(p).as_deref(), Some("deepseek-ai/DeepSeek-V4.1-Flash@dba1be0a"));
        assert_eq!(dir_identity(Path::new("/weights2/nope-not-hf")), None);
        let d = std::env::temp_dir().join(format!("mirror-ident-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join(".hf_fetch_rev"), "deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e10\n").unwrap();
        assert_eq!(dir_identity(&d).as_deref(), Some("deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10"));
        std::fs::remove_dir_all(&d).unwrap();
    }
}
