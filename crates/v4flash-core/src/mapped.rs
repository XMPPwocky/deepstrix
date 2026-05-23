//! MappedGguf — Gguf + a read-only mmap of the same file, giving
//! zero-copy access to tensor data as `&[u8]` slices.
//!
//! Use this when actually reading tensor bytes (e.g. for dequant kernels).
//! For metadata-only inspection (gguf-inspect / gguf-compare-llama),
//! `Gguf::open` is enough.

use std::fs::File;
use std::path::Path;

use color_eyre::eyre::{self, Context, eyre};
use memmap2::Mmap;

use crate::gguf::{Gguf, GgufTensor};

/// Owning handle to a parsed GGUF + its memory-mapped file. Tensor bytes
/// are returned as borrows into the mmap (no copy).
pub struct MappedGguf {
    gguf: Gguf,
    mmap: Mmap,
}

impl MappedGguf {
    /// Open + parse + mmap a GGUF file.
    pub fn open(path: impl AsRef<Path>) -> eyre::Result<Self> {
        let path = path.as_ref();
        let gguf = Gguf::open(path).map_err(|e| eyre!("parse {}: {e}", path.display()))?;
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        // SAFETY: we hold the file handle for the lifetime of Self; the
        // mapping is read-only.
        let mmap = unsafe { Mmap::map(&file) }
            .wrap_err_with(|| format!("mmap {}", path.display()))?;
        if mmap.len() as u64 != gguf.file_size {
            return Err(eyre!(
                "mmap size {} doesn't match parsed file_size {}",
                mmap.len(),
                gguf.file_size
            ));
        }
        Ok(MappedGguf { gguf, mmap })
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    pub fn raw_bytes(&self) -> &[u8] {
        &self.mmap
    }

    /// Zero-copy bytes for a tensor. Returns None if the tensor is empty
    /// or its dtype is `Unknown` (no size known).
    pub fn tensor_bytes(&self, t: &GgufTensor) -> Option<&[u8]> {
        if t.byte_size == 0 {
            return None;
        }
        let start = t.abs_offset as usize;
        let end = start.checked_add(t.byte_size as usize)?;
        if end > self.mmap.len() {
            return None;
        }
        Some(&self.mmap[start..end])
    }

    /// Convenience — look up by name then return bytes.
    pub fn tensor_bytes_by_name(&self, name: &str) -> Option<&[u8]> {
        self.gguf.tensor(name).and_then(|t| self.tensor_bytes(t))
    }

    /// Hint the kernel to start prefetching the whole file into the page
    /// cache. Safe to ignore failures — purely advisory.
    pub fn advise_willneed(&self) -> eyre::Result<()> {
        self.mmap
            .advise(memmap2::Advice::WillNeed)
            .wrap_err("madvise WILLNEED")?;
        Ok(())
    }

    pub fn advise_random(&self) -> eyre::Result<()> {
        self.mmap
            .advise(memmap2::Advice::Random)
            .wrap_err("madvise RANDOM")?;
        Ok(())
    }
}
