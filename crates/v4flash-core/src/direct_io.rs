//! Parallel O_DIRECT reads of stream-once data, spread over identical
//! REPLICAS of a file on different drives (the hub's embed phase: the
//! embedding GGUF and the loan image, docs/v41/EMBED_PHASE_DESIGN.md §5.3).
//!
//! O_DIRECT bypasses the page cache, so a phase's reads neither evict the
//! hub's hot page-cache rows nor need a DONTNEED afterwards, and concurrent
//! readers cannot evict each other's readahead (what made 4 buffered readers
//! with per-read DONTNEED 42% slower than one in the V4.1 weight loader,
//! `het/weights.rs` `expert_read_threads`). A span is read as 4 KiB-aligned
//! pieces handed out round-robin over the replicas, one thread per in-flight
//! piece.
//!
//! Where O_DIRECT is refused (tmpfs, some test setups) the file is opened
//! buffered instead and the result is the same bytes.

use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use color_eyre::eyre::{self, eyre, WrapErr};

/// O_DIRECT offset / length / buffer alignment.
pub const DIRECT_ALIGN: usize = 4096;

/// Smallest piece one reader thread reads.
pub const DIRECT_PIECE: usize = 4 << 20;

pub fn align_down(x: u64) -> u64 {
    x / DIRECT_ALIGN as u64 * DIRECT_ALIGN as u64
}

pub fn align_up(x: u64) -> u64 {
    x.div_ceil(DIRECT_ALIGN as u64) * DIRECT_ALIGN as u64
}

/// Identical copies of one file, opened for direct reads.
pub struct DirectFiles {
    files: Vec<File>,
    paths: Vec<PathBuf>,
    /// Per replica: opened with O_DIRECT (else buffered).
    direct: Vec<bool>,
    size: u64,
}

impl DirectFiles {
    /// Open every replica (at least one). They must all have the same size;
    /// a sample of 4 KiB blocks is compared too (a full compare would read
    /// the whole file).
    pub fn open(paths: &[PathBuf]) -> eyre::Result<Self> {
        if paths.is_empty() {
            return Err(eyre!("DirectFiles: no paths"));
        }
        let mut files = Vec::new();
        let mut direct = Vec::new();
        let mut size = None;
        for p in paths {
            let (f, d) = open_direct(p)?;
            let s = f.metadata()?.len();
            match size {
                None => size = Some(s),
                Some(s0) if s0 != s => return Err(eyre!("replica {} is {s} B, {} is {s0} B", p.display(), paths[0].display())),
                _ => {}
            }
            files.push(f);
            direct.push(d);
        }
        let df = DirectFiles { files, paths: paths.to_vec(), direct, size: size.expect("one path") };
        df.check_replicas_agree()?;
        Ok(df)
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn replicas(&self) -> usize {
        self.files.len()
    }

    /// `(path, O_DIRECT?)` per replica, for logs.
    pub fn describe(&self) -> Vec<(String, bool)> {
        self.paths.iter().zip(&self.direct).map(|(p, d)| (p.display().to_string(), *d)).collect()
    }

    /// Compare 16 sampled blocks of every replica with the first one.
    fn check_replicas_agree(&self) -> eyre::Result<()> {
        if self.files.len() < 2 || self.size < DIRECT_ALIGN as u64 {
            return Ok(());
        }
        let mut a = AlignedBuf::new(DIRECT_ALIGN);
        let mut b = AlignedBuf::new(DIRECT_ALIGN);
        let blocks = self.size / DIRECT_ALIGN as u64;
        for k in 0..16u64 {
            let off = (blocks - 1) * k / 15 * DIRECT_ALIGN as u64;
            read_full(&self.files[0], off, a.as_mut_slice())?;
            for (i, f) in self.files.iter().enumerate().skip(1) {
                read_full(f, off, b.as_mut_slice())?;
                if a.as_slice() != b.as_slice() {
                    return Err(eyre!("replica {} differs from {} at byte {off}", self.paths[i].display(), self.paths[0].display()));
                }
            }
        }
        Ok(())
    }

    /// Read `[offset, offset + len)` into `dst` with `threads` readers over
    /// the replicas. `dst` must start `DIRECT_ALIGN`-aligned and hold the
    /// aligned span: `align_up(offset + len) - align_down(offset)` bytes (see
    /// [`Self::span_bytes`]). Returns where the requested bytes start in `dst`.
    pub fn read_span(&self, offset: u64, len: usize, dst: &mut [u8], threads: usize) -> eyre::Result<usize> {
        let start = align_down(offset);
        let end = align_up(offset + len as u64);
        let need = (end - start) as usize;
        if dst.len() < need || (dst.as_ptr() as usize) % DIRECT_ALIGN != 0 {
            return Err(eyre!("read_span: dst {} B at {:p} must be {DIRECT_ALIGN}-aligned and hold {need} B", dst.len(), dst.as_ptr()));
        }
        if offset + len as u64 > self.size {
            return Err(eyre!("read_span: [{offset}, +{len}) past the end ({} B)", self.size));
        }
        let pieces = need.div_ceil(DIRECT_PIECE);
        let threads = threads.clamp(1, pieces.max(1));
        let next = AtomicUsize::new(0);
        let err: Mutex<Option<eyre::Report>> = Mutex::new(None);
        let base = dst.as_mut_ptr() as usize;
        std::thread::scope(|s| {
            for _ in 0..threads {
                s.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= pieces {
                        break;
                    }
                    let at = i * DIRECT_PIECE;
                    let n = DIRECT_PIECE.min(need - at);
                    // SAFETY: piece i is [at, at + n) of `dst`, disjoint from
                    // every other piece, taken by exactly one thread; `dst`
                    // outlives the scope.
                    let d = unsafe { std::slice::from_raw_parts_mut((base + at) as *mut u8, n) };
                    let f = &self.files[i % self.files.len()];
                    if let Err(e) = read_full(f, start + at as u64, d) {
                        *err.lock().unwrap_or_else(|p| p.into_inner()) = Some(e);
                        break;
                    }
                });
            }
        });
        if let Some(e) = err.into_inner().unwrap_or_else(|p| p.into_inner()) {
            return Err(e);
        }
        Ok((offset - start) as usize)
    }

    /// Bytes a buffer needs for [`Self::read_span`] of `[offset, offset + len)`.
    pub fn span_bytes(offset: u64, len: usize) -> usize {
        (align_up(offset + len as u64) - align_down(offset)) as usize
    }
}

/// Open `p` with O_DIRECT, or buffered when the filesystem refuses it.
fn open_direct(p: &Path) -> eyre::Result<(File, bool)> {
    match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(p) {
        Ok(f) => Ok((f, true)),
        Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {
            let f = File::open(p).wrap_err_with(|| format!("open {}", p.display()))?;
            Ok((f, false))
        }
        Err(e) => Err(e).wrap_err_with(|| format!("open {} (O_DIRECT)", p.display())),
    }
}

/// pread until `dst` is full or EOF (an aligned span may end past EOF; the
/// tail stays as it was).
fn read_full(f: &File, mut off: u64, dst: &mut [u8]) -> eyre::Result<()> {
    let mut at = 0;
    while at < dst.len() {
        match f.read_at(&mut dst[at..], off) {
            Ok(0) => break,
            Ok(n) => {
                at += n;
                off += n as u64;
            }
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e).wrap_err_with(|| format!("pread {} B at {off}", dst.len() - at)),
        }
    }
    Ok(())
}

/// A heap buffer aligned for O_DIRECT (tests, benches, tools; the hub reads
/// into page-aligned pinned host memory instead).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(1), DIRECT_ALIGN).expect("layout");
        // SAFETY: non-zero size, valid alignment; zeroed so slices are initialized.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "AlignedBuf: out of memory ({len} B)");
        AlignedBuf { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` holds `len` initialized bytes for the life of self.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len.max(1), DIRECT_ALIGN).expect("layout");
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr, layout) };
    }
}

// SAFETY: plain owned bytes.
unsafe impl Send for AlignedBuf {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_read_back_exactly_over_replicas() {
        let dir = std::env::temp_dir().join(format!("direct-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data: Vec<u8> = (0..(13 << 20) + 777u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        let (a, b) = (dir.join("a.bin"), dir.join("b.bin"));
        std::fs::write(&a, &data).unwrap();
        std::fs::write(&b, &data).unwrap();
        let df = DirectFiles::open(&[a.clone(), b.clone()]).unwrap();
        for (off, len) in [(0u64, 4096usize), (123, 5 << 20), (4096 * 3 + 17, 9 << 20), (data.len() as u64 - 999, 999)] {
            let mut buf = AlignedBuf::new(DirectFiles::span_bytes(off, len));
            let head = df.read_span(off, len, buf.as_mut_slice(), 3).unwrap();
            assert_eq!(&buf.as_slice()[head..head + len], &data[off as usize..off as usize + len], "off {off} len {len}");
        }
        // A replica that differs in a sampled block (block 0 always is) is refused.
        let mut bad = data.clone();
        bad[100] ^= 1;
        std::fs::write(&b, &bad).unwrap();
        assert!(DirectFiles::open(&[a, b]).is_err());
    }
}
