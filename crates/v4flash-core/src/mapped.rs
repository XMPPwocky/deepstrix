//! `MappedGguf` — parsed GGUF header + persistent `File` handle for
//! pread-based tensor loading. Name is historical; this no longer
//! mmaps the file (see commit log — mmap of the whole 86 GiB model
//! anchored the OS page cache and triggered OOMs on the 96 GiB Strix
//! Halo box).
//!
//! Usage pattern:
//! ```ignore
//! let gguf = MappedGguf::open(path)?;
//! let mut staging: Vec<u8> = Vec::new();
//! for tensor in &interesting_tensors {
//!     gguf.read_tensor_into(tensor, &mut staging)?;
//!     // staging now holds the bytes; copy to GPU, then iterate
//! }
//! ```
//!
//! Callers that already own a pre-sized buffer (e.g. a reusable
//! per-thread scratch slab) should use `read_tensor_into_slice` to
//! avoid the resize + zero-fill.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, Context, eyre};

use crate::gguf::{Gguf, GgufTensor};

/// Owning handle to a parsed GGUF + open `File`s for pread.
///
/// Holds NO mmap — every tensor byte read goes through `pread`. The
/// only resident state is the parsed metadata + tensor directory
/// (Gguf, a few MB) plus the file descriptors.
///
/// llama.cpp split GGUFs (`-00001-of-00003.gguf`) are handled
/// transparently: pass any shard's path and the siblings are derived
/// from the name, parsed, and merged (`Gguf::merge_shards`); each
/// tensor preads from its own shard via `GgufTensor::shard`.
pub struct MappedGguf {
    gguf: Gguf,
    /// One file per shard, indexed by `GgufTensor::shard`. Single-file
    /// GGUFs have exactly one entry.
    files: Vec<File>,
    path: PathBuf,
}

/// If `path` matches the llama.cpp shard convention
/// `<stem>-NNNNN-of-MMMMM.gguf`, return every sibling path in shard
/// order. Otherwise return just `path`.
fn shard_paths(path: &Path) -> eyre::Result<Vec<PathBuf>> {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n,
        None => return Ok(vec![path.to_path_buf()]),
    };
    // "<stem>-NNNNN-of-MMMMM.gguf" — fixed 5-digit fields.
    let Some(base) = name.strip_suffix(".gguf") else {
        return Ok(vec![path.to_path_buf()]);
    };
    let bytes = base.as_bytes();
    // "-NNNNN-of-MMMMM" is 15 bytes.
    if bytes.len() < 16 {
        return Ok(vec![path.to_path_buf()]);
    }
    let tail = &base[base.len() - 15..];
    let ok = tail.starts_with('-')
        && tail[1..6].bytes().all(|b| b.is_ascii_digit())
        && &tail[6..10] == "-of-"
        && tail[10..].bytes().all(|b| b.is_ascii_digit());
    if !ok {
        return Ok(vec![path.to_path_buf()]);
    }
    let total: usize = tail[10..].parse().expect("digits checked");
    let stem = &base[..base.len() - 15];
    let dir = path.parent().unwrap_or_else(|| Path::new(""));
    let mut out = Vec::with_capacity(total);
    for i in 1..=total {
        let p = dir.join(format!("{stem}-{i:05}-of-{total:05}.gguf"));
        if !p.exists() {
            return Err(eyre!(
                "split GGUF: shard {} of {} missing: {}",
                i,
                total,
                p.display()
            ));
        }
        out.push(p);
    }
    Ok(out)
}

impl MappedGguf {
    /// Open + parse a GGUF file (or any shard of a split GGUF). Reads
    /// only the header + metadata + tensor directory (a few MB at
    /// most). No bulk data is read or mapped.
    pub fn open(path: impl AsRef<Path>) -> eyre::Result<Self> {
        let path = path.as_ref();
        let paths = shard_paths(path)?;
        let mut shards = Vec::with_capacity(paths.len());
        let mut files = Vec::with_capacity(paths.len());
        for p in &paths {
            shards.push(Gguf::open(p).map_err(|e| eyre!("parse {}: {e}", p.display()))?);
            files.push(File::open(p).wrap_err_with(|| format!("open {}", p.display()))?);
        }
        let gguf = if shards.len() == 1 {
            shards.pop().expect("one shard")
        } else {
            Gguf::merge_shards(shards)
                .map_err(|e| eyre!("merge split GGUF {}: {e}", path.display()))?
        };
        Ok(MappedGguf { gguf, files, path: path.to_path_buf() })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn gguf(&self) -> &Gguf {
        &self.gguf
    }

    /// pread tensor bytes into `dst`, sized to exactly `t.byte_size`.
    /// Grows capacity if needed and zero-fills the new range (safe;
    /// the pread then overwrites). When the caller threads one
    /// staging buffer through a loader loop, only the first call
    /// (or growth to a larger tensor) allocates.
    pub fn read_tensor_into(&self, t: &GgufTensor, dst: &mut Vec<u8>) -> eyre::Result<()> {
        if t.byte_size == 0 {
            return Err(eyre!("tensor {:?} has zero byte_size", t.name));
        }
        let n = t.byte_size as usize;
        if dst.capacity() < n {
            dst.reserve_exact(n - dst.len());
        }
        dst.resize(n, 0);
        self.read_tensor_into_slice(t, &mut dst[..n])
    }

    /// Allocate a Vec<u8> sized to exactly `t.byte_size` and pread
    /// into it. Uses `Vec::with_capacity` so the allocation is the
    /// right size up front; `shrink_to_fit` at the end is defensive
    /// against the resize path inside `read_tensor_into` ever leaving
    /// excess capacity (currently it doesn't, but cheap to guarantee).
    pub fn read_tensor(&self, t: &GgufTensor) -> eyre::Result<Vec<u8>> {
        let n = t.byte_size as usize;
        let mut v = Vec::with_capacity(n);
        self.read_tensor_into(t, &mut v)?;
        v.shrink_to_fit();
        Ok(v)
    }

    /// pread tensor bytes into a pre-sized slice. `dst.len()` must
    /// equal `t.byte_size`. Use this when the caller owns the
    /// staging buffer and doesn't want resize churn.
    pub fn read_tensor_into_slice(&self, t: &GgufTensor, dst: &mut [u8]) -> eyre::Result<()> {
        let n = t.byte_size as usize;
        if dst.len() != n {
            return Err(eyre!(
                "read_tensor_into_slice: dst len {} != tensor {} byte_size {}",
                dst.len(),
                t.name,
                n
            ));
        }
        let file = self.files.get(t.shard).ok_or_else(|| {
            eyre!(
                "tensor {} references shard {} but only {} shard(s) are open",
                t.name,
                t.shard,
                self.files.len()
            )
        })?;
        file.read_exact_at(dst, t.abs_offset).wrap_err_with(|| {
            format!(
                "pread {} bytes at offset {} (shard {}) for {}",
                n, t.abs_offset, t.shard, t.name
            )
        })?;
        // Release the page cache range we just copied — about to
        // memcpy it to a DeviceBuffer and never look at the file
        // bytes again, so leaving them in cache costs kswapd cycles
        // for the rest of the process lifetime.
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                t.abs_offset as i64,
                n as i64,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        Ok(())
    }

    /// Number of shard files backing this GGUF (1 for single-file).
    pub fn n_shards(&self) -> usize {
        self.files.len()
    }

    /// Tell the kernel it can drop the page cache of every shard
    /// (whole files). Cheap insurance after a bulk weight-load pass
    /// to guarantee no stray pages are anchored.
    pub fn drop_page_cache(&self) -> eyre::Result<()> {
        use std::os::unix::io::AsRawFd;
        for file in &self.files {
            let rc = unsafe {
                libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED)
            };
            if rc != 0 {
                return Err(eyre!(
                    "posix_fadvise(DONTNEED) on {} failed: errno {rc}",
                    self.path.display()
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_paths_non_split_passthrough() {
        let p = Path::new("/models/foo.gguf");
        assert_eq!(shard_paths(p).unwrap(), vec![p.to_path_buf()]);
        // Suffix shape but non-digit fields must not match.
        let p = Path::new("/models/foo-abcde-of-00003.gguf");
        assert_eq!(shard_paths(p).unwrap(), vec![p.to_path_buf()]);
    }

    #[test]
    fn shard_paths_derives_siblings_and_errors_on_missing() {
        let dir = std::env::temp_dir().join(format!("shard_paths_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mk = |n: u32| dir.join(format!("m-{n:05}-of-00003.gguf"));
        for n in 1..=2 {
            std::fs::write(mk(n), b"x").unwrap();
        }
        // Shard 3 missing -> error naming it.
        let err = shard_paths(&mk(2)).unwrap_err().to_string();
        assert!(err.contains("00003-of-00003"), "{err}");
        std::fs::write(mk(3), b"x").unwrap();
        // Any shard's path yields all three in order.
        let got = shard_paths(&mk(2)).unwrap();
        assert_eq!(got, vec![mk(1), mk(2), mk(3)]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
