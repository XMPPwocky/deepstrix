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

/// Owning handle to a parsed GGUF + an open `File` for pread.
///
/// Holds NO mmap — every tensor byte read goes through `pread`. The
/// only resident state is the parsed metadata + tensor directory
/// (Gguf, a few MB) plus the file descriptor.
pub struct MappedGguf {
    gguf: Gguf,
    file: File,
    path: PathBuf,
}

impl MappedGguf {
    /// Open + parse a GGUF file. Reads only the header + metadata +
    /// tensor directory (a few MB at most). No bulk data is read or
    /// mapped.
    pub fn open(path: impl AsRef<Path>) -> eyre::Result<Self> {
        let path = path.as_ref();
        let gguf = Gguf::open(path).map_err(|e| eyre!("parse {}: {e}", path.display()))?;
        let file = File::open(path).wrap_err_with(|| format!("open {}", path.display()))?;
        Ok(MappedGguf { gguf, file, path: path.to_path_buf() })
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
        self.file
            .read_exact_at(dst, t.abs_offset)
            .wrap_err_with(|| {
                format!("pread {} bytes at offset {} for {}", n, t.abs_offset, t.name)
            })?;
        // Release the page cache range we just copied — about to
        // memcpy it to a DeviceBuffer and never look at the file
        // bytes again, so leaving them in cache costs kswapd cycles
        // for the rest of the process lifetime.
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                t.abs_offset as i64,
                n as i64,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        Ok(())
    }

    /// Tell the kernel it can drop the file's page cache (whole
    /// file). Cheap insurance after a bulk weight-load pass to
    /// guarantee no stray pages are anchored.
    pub fn drop_page_cache(&self) -> eyre::Result<()> {
        use std::os::unix::io::AsRawFd;
        let rc = unsafe {
            libc::posix_fadvise(
                self.file.as_raw_fd(),
                0,
                0,
                libc::POSIX_FADV_DONTNEED,
            )
        };
        if rc != 0 {
            return Err(eyre!(
                "posix_fadvise(DONTNEED) on {} failed: errno {rc}",
                self.path.display()
            ));
        }
        Ok(())
    }
}
