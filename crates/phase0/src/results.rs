//! Helpers for writing per-gate measurement results as JSON under
//! `results/`. Files are gitignored — final summary lands in
//! `docs/PHASE0.md` at commit 10.
//!
//! Each gate produces one top-level JSON object; we don't enforce a
//! shared schema (gates measure different things) but they all carry a
//! `gate` and `timestamp` field so the report tool can sort them.

use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

use color_eyre::eyre::{self, Context};
use serde::Serialize;

/// Write a gate's result blob to `results/<name>.json`. Returns the
/// absolute path. Pretty-prints so the file is human-diffable.
pub fn write<T: Serialize>(name: &str, value: &T) -> eyre::Result<PathBuf> {
    let dir = PathBuf::from("results");
    fs::create_dir_all(&dir).wrap_err("create results/ dir")?;
    let path = dir.join(format!("{name}.json"));
    let json = serde_json::to_string_pretty(value).wrap_err("serialize gate result")?;
    fs::write(&path, json).wrap_err_with(|| format!("write {}", path.display()))?;
    Ok(path)
}

/// Unix timestamp at write time. We don't use a richer datetime crate;
/// the report tool formats this if needed.
pub fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
