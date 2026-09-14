//! Sharded HF safetensors checkpoint reader.
//!
//! Same discipline as [`crate::mapped::MappedGguf`]: no mmap, every read is an
//! explicit `pread` followed by `POSIX_FADV_DONTNEED`, so a 475 GiB checkpoint
//! streams through a small page cache and never competes with the GPUs for
//! host memory.
//!
//! Format (per `*.safetensors` file): `u64 LE header_len`, then `header_len`
//! bytes of JSON `{ "<name>": { "dtype", "shape", "data_offsets": [begin, end] },
//! "__metadata__": {...} }`, then the byte blob that the offsets index.
//! `model.safetensors.index.json` maps tensor name → shard file; a single
//! unsharded `model.safetensors` is accepted too.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::os::unix::fs::FileExt;
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

use color_eyre::eyre::{self, eyre, Context};

/// Element type as spelled in the safetensors header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StDtype {
    BF16,
    F16,
    F32,
    F64,
    F8E4M3,
    F8E5M2,
    F8E8M0,
    I8,
    U8,
    I16,
    I32,
    I64,
    Bool,
}

impl StDtype {
    fn parse(s: &str) -> eyre::Result<Self> {
        Ok(match s {
            "BF16" => Self::BF16,
            "F16" => Self::F16,
            "F32" => Self::F32,
            "F64" => Self::F64,
            "F8_E4M3" => Self::F8E4M3,
            "F8_E5M2" => Self::F8E5M2,
            "F8_E8M0" => Self::F8E8M0,
            "I8" => Self::I8,
            "U8" => Self::U8,
            "I16" => Self::I16,
            "I32" => Self::I32,
            "I64" => Self::I64,
            "BOOL" => Self::Bool,
            other => return Err(eyre!("unsupported safetensors dtype {other:?}")),
        })
    }

    pub fn elem_bytes(self) -> u64 {
        match self {
            Self::BF16 | Self::F16 | Self::I16 => 2,
            Self::F32 | Self::I32 => 4,
            Self::F64 | Self::I64 => 8,
            Self::F8E4M3 | Self::F8E5M2 | Self::F8E8M0 | Self::I8 | Self::U8 | Self::Bool => 1,
        }
    }
}

/// One tensor's location: `offset` is absolute within shard `shard`.
#[derive(Debug, Clone)]
pub struct StTensor {
    pub name: String,
    pub dtype: StDtype,
    /// Row-major, outermost first (PyTorch order).
    pub shape: Vec<u64>,
    pub shard: usize,
    pub offset: u64,
    pub len: u64,
}

impl StTensor {
    pub fn elements(&self) -> u64 {
        self.shape.iter().product()
    }
}

pub struct SafetensorsDir {
    dir: PathBuf,
    files: Vec<File>,
    /// Second handle per shard opened `O_DIRECT`, for [`Self::read_range_into_direct`].
    /// `None` where the filesystem refused the flag. See that method for why.
    direct_files: Vec<Option<File>>,
    shard_names: Vec<String>,
    tensors: HashMap<String, StTensor>,
}

impl SafetensorsDir {
    /// Open every shard named by `model.safetensors.index.json` (or the single
    /// `model.safetensors`) and parse all headers. Cheap: headers only.
    pub fn open(dir: impl AsRef<Path>) -> eyre::Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let index_path = dir.join("model.safetensors.index.json");
        let shard_names: Vec<String> = if index_path.exists() {
            let f = File::open(&index_path)
                .wrap_err_with(|| format!("open {}", index_path.display()))?;
            let v: serde_json::Value = serde_json::from_reader(f)
                .wrap_err_with(|| format!("parse {}", index_path.display()))?;
            let map = v
                .get("weight_map")
                .and_then(|m| m.as_object())
                .ok_or_else(|| eyre!("{}: no weight_map object", index_path.display()))?;
            let mut names: Vec<String> = map
                .values()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            names.sort();
            names.dedup();
            names
        } else if dir.join("model.safetensors").exists() {
            vec!["model.safetensors".to_owned()]
        } else {
            return Err(eyre!(
                "{}: neither model.safetensors.index.json nor model.safetensors",
                dir.display()
            ));
        };

        let mut files = Vec::with_capacity(shard_names.len());
        let mut direct_files: Vec<Option<File>> = Vec::with_capacity(shard_names.len());
        let mut tensors = HashMap::new();
        for (shard, sname) in shard_names.iter().enumerate() {
            let path = dir.join(sname);
            let mut f = File::open(&path).wrap_err_with(|| format!("open {}", path.display()))?;
            let mut n = [0u8; 8];
            f.read_exact(&mut n)
                .wrap_err_with(|| format!("{}: header length", path.display()))?;
            let n = u64::from_le_bytes(n);
            if n > 256 << 20 {
                return Err(eyre!("{}: header length {n} is implausible", path.display()));
            }
            let mut hdr = vec![0u8; n as usize];
            f.read_exact(&mut hdr)
                .wrap_err_with(|| format!("{}: header bytes", path.display()))?;
            let hdr: serde_json::Value = serde_json::from_slice(&hdr)
                .wrap_err_with(|| format!("{}: header JSON", path.display()))?;
            let obj = hdr
                .as_object()
                .ok_or_else(|| eyre!("{}: header is not a JSON object", path.display()))?;
            let data_start = 8 + n;
            for (tname, ent) in obj {
                if tname == "__metadata__" {
                    continue;
                }
                let ctx = || format!("{}: tensor {tname}", path.display());
                let dtype = ent
                    .get("dtype")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| eyre!("{}: no dtype", ctx()))?;
                let dtype = StDtype::parse(dtype).wrap_err_with(ctx)?;
                let shape = ent
                    .get("shape")
                    .and_then(|s| s.as_array())
                    .ok_or_else(|| eyre!("{}: no shape", ctx()))?
                    .iter()
                    .map(|x| x.as_u64().ok_or_else(|| eyre!("{}: bad shape entry", ctx())))
                    .collect::<eyre::Result<Vec<u64>>>()?;
                let offs = ent
                    .get("data_offsets")
                    .and_then(|o| o.as_array())
                    .filter(|o| o.len() == 2)
                    .ok_or_else(|| eyre!("{}: no data_offsets pair", ctx()))?;
                let (a, b) = match (offs[0].as_u64(), offs[1].as_u64()) {
                    (Some(a), Some(b)) => (a, b),
                    _ => return Err(eyre!("{}: bad data_offsets", ctx())),
                };
                let elems: u64 = shape.iter().product();
                if b < a || b - a != elems * dtype.elem_bytes() {
                    return Err(eyre!(
                        "{}: data_offsets [{a},{b}) inconsistent with shape {shape:?} {dtype:?}",
                        ctx()
                    ));
                }
                let t = StTensor {
                    name: tname.clone(),
                    dtype,
                    shape,
                    shard,
                    offset: data_start + a,
                    len: b - a,
                };
                if tensors.insert(tname.clone(), t).is_some() {
                    return Err(eyre!("{}: duplicated across shards", ctx()));
                }
            }
            direct_files.push(
                std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(libc::O_DIRECT)
                    .open(&path)
                    .ok(),
            );
            files.push(f);
        }
        Ok(Self { dir, files, direct_files, shard_names, tensors })
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn n_shards(&self) -> usize {
        self.files.len()
    }

    pub fn shard_name(&self, shard: usize) -> &str {
        &self.shard_names[shard]
    }

    pub fn n_tensors(&self) -> usize {
        self.tensors.len()
    }

    pub fn tensor(&self, name: &str) -> Option<&StTensor> {
        self.tensors.get(name)
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn get(&self, name: &str) -> eyre::Result<&StTensor> {
        self.tensor(name)
            .ok_or_else(|| eyre!("tensor {name:?} not in checkpoint {}", self.dir.display()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(String::as_str)
    }

    /// `pread` `dst.len()` bytes starting `byte_off` into the tensor's blob,
    /// then drop those pages from the cache (read-once discipline).
    pub fn read_range_into(&self, t: &StTensor, byte_off: u64, dst: &mut [u8]) -> eyre::Result<()> {
        let end = byte_off
            .checked_add(dst.len() as u64)
            .ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!(
                "{}: range [{byte_off},{end}) exceeds tensor length {}",
                t.name,
                t.len
            ));
        }
        let file = self
            .files
            .get(t.shard)
            .ok_or_else(|| eyre!("{}: shard {} not open", t.name, t.shard))?;
        let off = t.offset + byte_off;
        file.read_exact_at(dst, off).wrap_err_with(|| {
            format!(
                "pread {} bytes at {off} in {} for {}",
                dst.len(),
                self.shard_names[t.shard],
                t.name
            )
        })?;
        // SAFETY: plain advisory syscall on an fd we own; no memory is touched.
        unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                off as i64,
                dst.len() as i64,
                libc::POSIX_FADV_DONTNEED,
            );
        }
        Ok(())
    }

    /// `read_range_into` without the trailing `POSIX_FADV_DONTNEED`: for
    /// small random reads whose pages are worth keeping (Engram rows).
    pub fn read_range_into_cached(&self, t: &StTensor, byte_off: u64, dst: &mut [u8]) -> eyre::Result<()> {
        let end = byte_off
            .checked_add(dst.len() as u64)
            .ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds tensor length {}", t.name, t.len));
        }
        let file = self
            .files
            .get(t.shard)
            .ok_or_else(|| eyre!("{}: shard {} not open", t.name, t.shard))?;
        file.read_exact_at(dst, t.offset + byte_off)
            .wrap_err_with(|| format!("pread {} bytes at {} in {} for {}", dst.len(), t.offset + byte_off, self.shard_names[t.shard], t.name))?;
        Ok(())
    }

    /// `read_range_into_cached` through an `O_DIRECT` handle.
    ///
    /// WHY. The buffered path was chosen so the pager's LRU refills could hit the
    /// page cache (see the note in `hf_v41::read_expert_raw`). That reasoning does
    /// not survive box 2's memory pressure: the expert file is 101 GB and the box
    /// has ~5 GB of page cache, so under 5% of it can ever be cached, and the
    /// copy through the cache costs more than the hits save. Measured on box 2,
    /// 18.80 MB (one expert) at random offsets, daemon idle:
    ///
    ///     O_DIRECT   median  4.70 ms = 4.00 GB/s
    ///     buffered   median 12.55 ms = 1.50 GB/s
    ///
    /// 4.00 GB/s single-threaded also matches this drive's 32-thread buffered
    /// depth figure (4.31), so one direct pread replaces the threaded split.
    ///
    /// `O_DIRECT` requires the file offset, the length AND the buffer address to
    /// be block-aligned, so this reads an outward-rounded extent into an aligned
    /// bounce buffer and copies the requested subrange out. The extra copy is
    /// ~0.6 ms for 6 MB against ~8 ms saved.
    ///
    /// Returns `Ok(false)` if this shard has no direct handle (filesystem refused
    /// the flag), so callers can fall back rather than fail.
    pub fn read_range_into_direct(
        &self,
        t: &StTensor,
        byte_off: u64,
        dst: &mut [u8],
    ) -> eyre::Result<bool> {
        const A: u64 = 4096;
        let end = byte_off
            .checked_add(dst.len() as u64)
            .ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds tensor length {}", t.name, t.len));
        }
        let Some(Some(file)) = self.direct_files.get(t.shard) else {
            return Ok(false);
        };
        if dst.is_empty() {
            return Ok(true);
        }
        let abs = t.offset + byte_off;
        let lo = abs & !(A - 1);
        let hi = (abs + dst.len() as u64 + A - 1) & !(A - 1);
        let span = (hi - lo) as usize;

        // Aligned bounce buffer; freed on drop even if the pread fails.
        struct Aligned(*mut u8, std::alloc::Layout);
        impl Drop for Aligned {
            fn drop(&mut self) {
                // SAFETY: allocated with this exact layout in the constructor below.
                unsafe { std::alloc::dealloc(self.0, self.1) }
            }
        }
        let layout = std::alloc::Layout::from_size_align(span, A as usize)
            .map_err(|e| eyre!("{}: bad direct layout: {e}", t.name))?;
        // SAFETY: non-zero size (dst non-empty => span >= A), valid layout.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(eyre!("{}: direct bounce alloc of {span} failed", t.name));
        }
        let guard = Aligned(ptr, layout);
        // SAFETY: `ptr` owns `span` bytes for the lifetime of `guard`.
        let buf = unsafe { std::slice::from_raw_parts_mut(guard.0, span) };

        // The extent is rounded UP to a block boundary, which can run past EOF on
        // the last tensor of a shard; the kernel then returns a short read. That is
        // fine as long as the bytes the caller asked for arrived, so read until the
        // requested subrange is covered and treat Ok(0) as EOF rather than failure.
        let head = (abs - lo) as usize;
        let need = head + dst.len();
        let mut got = 0usize;
        while got < need {
            let n = file.read_at(&mut buf[got..], lo + got as u64).wrap_err_with(|| {
                format!(
                    "O_DIRECT pread at {} in {} for {}",
                    lo + got as u64,
                    self.shard_names[t.shard],
                    t.name
                )
            })?;
            if n == 0 {
                return Err(eyre!(
                    "{}: O_DIRECT short read at {}: got {got} of {need} (span {span})",
                    t.name,
                    lo + got as u64
                ));
            }
            got += n;
        }
        dst.copy_from_slice(&buf[head..head + dst.len()]);
        Ok(true)
    }

    /// Same as [`Self::read_range_into_cached`] but splits the range across
    /// `threads` concurrent preads.
    ///
    /// WHY: a decode miss is a single ~5.9 MB read, and at a ~90% hit rate a
    /// layer averages 0.39 misses — so there is almost never a second miss to
    /// issue alongside it. One pread at a time measured 2.17 GB/s, which is
    /// ~80% of this drive's SINGLE-THREAD figure (2.70) and half its depth
    /// figure (4.31 at 32 threads). The only way to reach depth with one miss
    /// is to split that miss's own read.
    ///
    /// `pread` is positional and thread-safe (no shared file offset), so the
    /// chunks are independent; each thread gets a disjoint `dst` slice.
    pub fn read_range_into_cached_par(
        &self,
        t: &StTensor,
        byte_off: u64,
        dst: &mut [u8],
        threads: usize,
    ) -> eyre::Result<()> {
        let n = dst.len();
        // Below ~1 MB the per-thread overhead dominates the transfer.
        if threads <= 1 || n < (1 << 20) {
            return self.read_range_into_cached(t, byte_off, dst);
        }
        let end = byte_off
            .checked_add(n as u64)
            .ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds tensor length {}", t.name, t.len));
        }
        let file = self
            .files
            .get(t.shard)
            .ok_or_else(|| eyre!("{}: shard {} not open", t.name, t.shard))?;
        let base = t.offset + byte_off;
        // Page-aligned chunks so two threads never share a 4 KiB page.
        let chunk = ((n + threads - 1) / threads + 4095) & !4095;
        let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
        std::thread::scope(|sc| {
            for (i, part) in dst.chunks_mut(chunk).enumerate() {
                let err = &err;
                let off = base + (i * chunk) as u64;
                sc.spawn(move || {
                    if let Err(e) = file.read_exact_at(part, off) {
                        *err.lock().unwrap() = Some(format!("{e}"));
                    }
                });
            }
        });
        if let Some(e) = err.lock().unwrap().take() {
            return Err(eyre!("parallel pread {} at {}: {e}", t.name, base));
        }
        Ok(())
    }

    /// Whole tensor into a caller-sized buffer (`dst.len()` must equal `t.len`).
    pub fn read_into(&self, t: &StTensor, dst: &mut [u8]) -> eyre::Result<()> {
        if dst.len() as u64 != t.len {
            return Err(eyre!(
                "{}: dst len {} != tensor bytes {}",
                t.name,
                dst.len(),
                t.len
            ));
        }
        self.read_range_into(t, 0, dst)
    }

    pub fn read(&self, t: &StTensor) -> eyre::Result<Vec<u8>> {
        let mut v = vec![0u8; t.len as usize];
        self.read_into(t, &mut v)?;
        Ok(v)
    }
}
