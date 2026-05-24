//! Loader for the M2 activation dump produced by
//! `external/ds4-dump/ds4-dump-activations`. The dump tree contains one
//! binary blob per tensor plus a `manifest.json` indexing them:
//!
//! ```text
//! reference/v4flash-cpu-activations/
//!   manifest.json
//!   logits.f32, tokens.json
//!   L00/T0000/attn_cur.bin, attn_input_norm.bin, ...
//!   L00/weight/attn_norm.bin, ffn_norm.bin
//!   ...
//! ```
//!
//! The oracle parses the manifest once, builds a `(tag, layer, token)`
//! lookup table, and lets tests fetch raw bytes for each tensor on
//! demand. We do not mmap — tensors are small (~16 KB) and tests don't
//! re-read them; simple `fs::read` is fine.
//!
//! See `docs/PHASE1_REFERENCE.md` for the canonical reference SHAs.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre, WrapErr};
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F16,
    Fp8,
    I32,
}

impl Dtype {
    pub fn bytes(self) -> usize {
        match self {
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 => 2,
            Dtype::Fp8 => 1,
        }
    }
    fn parse(s: &str) -> eyre::Result<Self> {
        match s {
            "f32" => Ok(Dtype::F32),
            "f16" => Ok(Dtype::F16),
            "fp8" => Ok(Dtype::Fp8),
            "i32" => Ok(Dtype::I32),
            other => Err(eyre!("unknown dtype in manifest: {other}")),
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawTensor {
    tag: String,
    layer: i32,
    /// `-1` for weight tensors (deduped, not per-token); otherwise the
    /// token position.
    token: i32,
    dtype: String,
    shape: Vec<i64>,
    bytes: usize,
    path: String,
    is_weight: bool,
}

#[derive(Debug, Deserialize)]
struct RawManifest {
    #[serde(default)]
    meta: serde_json::Value,
    tensors: Vec<RawTensor>,
    #[serde(default)]
    n_tensors: usize,
    #[serde(default)]
    n_logit_rows: usize,
    #[serde(default)]
    vocab_size: usize,
    #[serde(default)]
    prompt_len: usize,
}

#[derive(Debug, Clone)]
pub struct TensorEntry {
    pub tag: String,
    pub layer: i32,
    /// `-1` for weight tensors.
    pub token: i32,
    pub dtype: Dtype,
    pub shape: Vec<i64>,
    pub bytes: usize,
    /// Filesystem path *relative to the dump root*.
    pub rel_path: String,
    pub is_weight: bool,
}

impl TensorEntry {
    pub fn n_elements(&self) -> i64 {
        self.shape.iter().product()
    }
}

/// A loaded activation dump tree. Use [`ActivationDump::open`] then
/// [`ActivationDump::tensor`] / [`ActivationDump::read_f32`] to fetch.
pub struct ActivationDump {
    root: PathBuf,
    /// (layer, token, tag) → index into `entries`. token=-1 for weights.
    index: HashMap<(i32, i32, String), usize>,
    entries: Vec<TensorEntry>,
    pub n_logit_rows: usize,
    pub vocab_size: usize,
    pub prompt_len: usize,
}

impl ActivationDump {
    pub fn open<P: AsRef<Path>>(root: P) -> eyre::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let manifest_path = root.join("manifest.json");
        let bytes = fs::read(&manifest_path)
            .wrap_err_with(|| format!("read manifest at {}", manifest_path.display()))?;
        let raw: RawManifest = serde_json::from_slice(&bytes)
            .wrap_err_with(|| format!("parse manifest at {}", manifest_path.display()))?;

        let mut entries = Vec::with_capacity(raw.tensors.len());
        let mut index = HashMap::with_capacity(raw.tensors.len());
        for r in raw.tensors {
            let dtype = Dtype::parse(&r.dtype)?;
            let idx = entries.len();
            index.insert((r.layer, r.token, r.tag.clone()), idx);
            entries.push(TensorEntry {
                tag: r.tag,
                layer: r.layer,
                token: r.token,
                dtype,
                shape: r.shape,
                bytes: r.bytes,
                rel_path: r.path,
                is_weight: r.is_weight,
            });
        }

        let _ = raw.meta; // unused, but read for future extensions
        let _ = raw.n_tensors;
        Ok(ActivationDump {
            root,
            index,
            entries,
            n_logit_rows: raw.n_logit_rows,
            vocab_size: raw.vocab_size,
            prompt_len: raw.prompt_len,
        })
    }

    /// Find an activation tensor by (tag, layer, token).
    pub fn tensor(&self, tag: &str, layer: i32, token: i32) -> Option<&TensorEntry> {
        self.index
            .get(&(layer, token, tag.to_string()))
            .map(|&i| &self.entries[i])
    }

    /// Find a deduped weight tensor by (tag, layer). Equivalent to
    /// `tensor(tag, layer, -1)`.
    pub fn weight(&self, tag: &str, layer: i32) -> Option<&TensorEntry> {
        self.tensor(tag, layer, -1)
    }

    /// Read a tensor's raw bytes. Caller decodes per dtype.
    pub fn read_bytes(&self, entry: &TensorEntry) -> eyre::Result<Vec<u8>> {
        let path = self.root.join(&entry.rel_path);
        let bytes = fs::read(&path)
            .wrap_err_with(|| format!("read tensor blob at {}", path.display()))?;
        if bytes.len() != entry.bytes {
            return Err(eyre!(
                "tensor {} (L{} T{}) size mismatch: manifest says {} bytes, file has {}",
                entry.tag,
                entry.layer,
                entry.token,
                entry.bytes,
                bytes.len()
            ));
        }
        Ok(bytes)
    }

    /// Read a tensor as f32. Errors if dtype != F32 or byte count mismatch.
    pub fn read_f32(&self, entry: &TensorEntry) -> eyre::Result<Vec<f32>> {
        if entry.dtype != Dtype::F32 {
            return Err(eyre!(
                "read_f32 on non-f32 tensor {} (dtype={:?})",
                entry.tag,
                entry.dtype
            ));
        }
        let bytes = self.read_bytes(entry)?;
        let n = bytes.len() / 4;
        let mut out = vec![0f32; n];
        // Native little-endian; ds4 writes raw native bytes.
        for (i, chunk) in bytes.chunks_exact(4).enumerate() {
            out[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(out)
    }

    /// Iterate all entries (in manifest order). Useful for tests that
    /// want to validate one kernel across all (layer, token) positions.
    pub fn entries(&self) -> impl Iterator<Item = &TensorEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
    pub fn root(&self) -> &Path {
        &self.root
    }
}
