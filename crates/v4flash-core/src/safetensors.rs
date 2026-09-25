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
    /// `V41_EXPERT_MIRROR_DIR`: a second copy of the checkpoint on another
    /// drive; per shard an `O_DIRECT` handle to the same-named file there
    /// (None when absent). `read_range_into_direct_split` reads the tail of a
    /// range from it concurrently with the head from `direct_files`, so ONE
    /// expert miss is served by two drives (box 2, 2026-09-22).
    mirror_files: Vec<Option<File>>,
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
        let mirror_dir: Option<std::path::PathBuf> = std::env::var_os("V41_EXPERT_MIRROR_DIR").map(std::path::PathBuf::from);
        let mut mirror_files: Vec<Option<File>> = Vec::with_capacity(shard_names.len());
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
            mirror_files.push(mirror_dir.as_ref().and_then(|d| {
                let mp = d.join(path.file_name()?);
                // Same size, or it is not a copy of this shard.
                let same = std::fs::metadata(&mp).ok()?.len() == std::fs::metadata(&path).ok()?.len();
                same.then(|| std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(&mp).ok()).flatten()
            }));
            files.push(f);
        }
        if mirror_dir.is_some() {
            eprintln!("safetensors: expert mirror {} shards of {} usable from {}", mirror_files.iter().filter(|m| m.is_some()).count(), shard_names.len(), mirror_dir.as_ref().unwrap().display());
        }
        Ok(Self { dir, files, direct_files, mirror_files, shard_names, tensors })
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

    /// Ask the kernel to pull this tensor's bytes into the page cache, without
    /// waiting for them. A hint only: no correctness depends on it, every error
    /// is swallowed, and the pages may be reclaimed before anyone reads them.
    ///
    /// This is the whole mechanism behind prefill read-ahead. It is safe in the
    /// way a device-side prefetch is not: it touches no pool slot, no remap and
    /// no LRU, so it cannot race an MoE kernel that is still queued on the
    /// layer we are running ahead of, and it cannot evict a resident expert.
    /// The only thing it can cost is page cache.
    ///
    /// Useless under `O_DIRECT` (which bypasses the page cache) -- the caller is
    /// responsible for not bothering when `expert_odirect()` is on.
    pub fn willneed(&self, t: &StTensor) -> bool {
        let Some(file) = self.files.get(t.shard) else { return false };
        // SAFETY: fd is owned by `self` and outlives the call; fadvise only
        // advises the page cache and never writes through the pointer-free API.
        let rc = unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                t.offset as libc::off_t,
                t.len as libc::off_t,
                libc::POSIX_FADV_WILLNEED,
            )
        };
        rc == 0
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

    /// Tell the kernel this tensor's byte range is read RANDOMLY, so it stops
    /// issuing readahead for it.
    ///
    /// **MEASURED INERT on the Engram table (2026-09-14) — do not retry there.**
    /// A/B on novel (uncacheable) text, `ENGRAM_FADV_RANDOM` on vs off:
    /// 287.45 vs 301.55 MB of disk per generated token. No effect.
    ///
    /// I reached for this after measuring "41.65 MB/token" of process disk reads
    /// and attributing it to Engram's 96 tiny random rows/token. That attribution
    /// was WRONG: it was whole-request I/O divided by generated tokens, and it is
    /// dominated by a FIXED per-request cost (prefill expert paging plus the CED
    /// replay, documented at ~41 GB/request). Isolating the slope — same prompt,
    /// 64 vs 512 generated tokens on a warm server — gives 7292.6 MB and 1063.7 MB
    /// respectively: 8x the tokens, 7x LESS disk. The per-generated-token slope is
    /// ~zero, i.e. **Engram does almost no disk I/O in steady-state decode.** Its
    /// row cache plus the page cache absorb it, which is exactly what
    /// `engram_table.rs` intends by not fadvise-dropping rows. Engram being
    /// SSD-backed is the design working, not a cost to remove.
    ///
    /// The helper is kept because it is correct and per-range (so a sequential
    /// reader of the same shard keeps its readahead), and because the negative
    /// result is worth not repeating. `FADV_RANDOM` only disables readahead; it
    /// does not drop cached pages.
    pub fn advise_random(&self, t: &StTensor) -> eyre::Result<()> {
        use std::os::unix::io::AsRawFd;
        let Some(file) = self.files.get(t.shard) else {
            return Err(eyre!("{}: shard {} out of range", t.name, t.shard));
        };
        // SAFETY: `file` is an open fd owned by self; offset/len are in range by
        // construction. posix_fadvise is advisory and cannot corrupt data.
        let rc = unsafe {
            libc::posix_fadvise(
                file.as_raw_fd(),
                t.offset as libc::off_t,
                t.len as libc::off_t,
                libc::POSIX_FADV_RANDOM,
            )
        };
        if rc != 0 {
            return Err(eyre!("{}: posix_fadvise(RANDOM) failed: {rc}", t.name));
        }
        Ok(())
    }

    /// O_DIRECT read with **no bounce buffer**: the bytes land straight in
    /// `dst`, and the caller is told where in `dst` they start.
    ///
    /// [`Self::read_range_into_direct`] allocates an aligned bounce of the whole
    /// extent per call and memcpys out of it. At the expert size that is an
    /// 18.8 MB allocation plus an 18.8 MB copy **per miss**, which is why
    /// O_DIRECT measured SLOWER than buffered twice (box 1 -16% decode, box 2
    /// +6%) despite the drive preferring it 5.89 -> 4.84 ms.
    ///
    /// The copy is avoidable because O_DIRECT only needs the file offset, the
    /// memory address and the length to share an alignment — not to be
    /// aligned to zero. Every expert tensor in a shard has a length that is a
    /// multiple of 4096, so they all share ONE residue `pad = offset % 4096`.
    /// Reading `pad + len` bytes from `offset - pad` into a 4096-aligned `dst`
    /// therefore satisfies O_DIRECT and puts the requested bytes at `dst[pad]`.
    ///
    /// `dst` must be 4096-aligned and hold `pad + len` rounded up to 4096;
    /// `capacity_for` gives the size to allocate. Returns `Ok(None)` when this
    /// shard has no O_DIRECT handle (caller falls back to the cached path).
    ///
    /// Alignment of `dst` is CHECKED, not assumed — a misaligned buffer makes
    /// `pread` fail with EINVAL, which is a confusing way to learn this.
    /// [`Self::read_range_into_direct_padded`] over an ABSOLUTE file span rather
    /// than one tensor, so a caller that has PROVEN several tensors adjacent can
    /// fetch them in a single pread.
    ///
    /// The per-tensor entry point refuses `end > t.len` on purpose; this one
    /// takes the shard and byte range directly and is only safe when the caller
    /// has checked contiguity itself (see `hf_v41::expert_runs`).
    ///
    /// Same alignment contract: `dst` must be 4096-aligned and hold
    /// `pad + len` rounded up to 4096. Returns the offset within `dst` at which
    /// the requested bytes start, or `Ok(None)` if this shard has no O_DIRECT
    /// handle.
    pub fn read_span_into_direct_padded(
        &self,
        shard: usize,
        abs: u64,
        len: usize,
        dst: &mut [u8],
    ) -> eyre::Result<Option<usize>> {
        const A: u64 = 4096;
        let Some(Some(file)) = self.direct_files.get(shard) else {
            return Ok(None);
        };
        if len == 0 {
            return Ok(Some(0));
        }
        let pad = (abs & (A - 1)) as usize;
        let span = ((pad + len) as u64).div_ceil(A) as usize * A as usize;
        if dst.len() < span {
            return Err(eyre!("direct span dst {} < span {span} (pad {pad}, len {len})", dst.len()));
        }
        if dst.as_ptr() as usize & (A as usize - 1) != 0 {
            return Err(eyre!("direct span dst is not {A}-aligned"));
        }
        let need = pad + len;
        let mut got = 0usize;
        while got < need {
            let n = file
                .read_at(&mut dst[got..span], abs - pad as u64 + got as u64)
                .wrap_err_with(|| format!("O_DIRECT span pread at {} in shard {shard}", abs - pad as u64 + got as u64))?;
            if n == 0 {
                break;
            }
            got += n;
        }
        if got < need {
            return Err(eyre!("O_DIRECT span short read: got {got} of {need}"));
        }
        Ok(Some(pad))
    }

    /// Whether shard `shard` has a mirror handle.
    pub fn has_mirror(&self, shard: usize) -> bool {
        matches!(self.mirror_files.get(shard), Some(Some(_)))
    }

    /// Every shard has a usable mirror handle (same name, same size, O_DIRECT).
    pub fn mirror_complete(&self) -> bool {
        !self.mirror_files.is_empty() && self.mirror_files.iter().all(|m| m.is_some())
    }

    /// `read_range_into_direct_padded`, with the range split at a 4096-aligned
    /// point: the head `[0, p)` read from the primary handle and the tail
    /// `[p, len)` from the mirror handle CONCURRENTLY (`mirror_frac` = share of
    /// the bytes on the mirror, e.g. 0.6 for a faster mirror drive). EXACTLY
    /// 1.0 / 0.0 reads the whole range from the mirror / primary alone. Falls
    /// back to the single primary read when there is no mirror. Same padded
    /// layout and return value as the unsplit read.
    pub fn read_range_into_direct_split(
        &self,
        t: &StTensor,
        byte_off: u64,
        len: usize,
        dst: &mut [u8],
        mirror_frac: f32,
    ) -> eyre::Result<Option<usize>> {
        const A: u64 = 4096;
        let (Some(Some(file)), Some(Some(mirror))) = (self.direct_files.get(t.shard), self.mirror_files.get(t.shard)) else {
            return self.read_range_into_direct_padded(t, byte_off, len, dst);
        };
        let end = byte_off.checked_add(len as u64).ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds tensor length {}", t.name, t.len));
        }
        let mf = mirror_frac.clamp(0.0, 1.0);
        if len < 2 * A as usize {
            return self.read_range_into_direct_padded_on(t, byte_off, len, dst, mf >= 1.0);
        }
        let abs = t.offset + byte_off;
        let pad = (abs & (A - 1)) as usize;
        let span = ((pad + len) as u64).div_ceil(A) as usize * A as usize;
        if dst.len() < span {
            return Err(eyre!("{}: direct dst {} < span {span} (pad {pad}, len {len})", t.name, dst.len()));
        }
        if dst.as_ptr() as usize & (A as usize - 1) != 0 {
            return Err(eyre!("{}: direct dst is not {A}-aligned", t.name));
        }
        // Split point in the padded span, 4096-aligned: the primary reads
        // [0, cut) of the span, the mirror [cut, span). EXACTLY 0 or 1 routes
        // the whole read to one drive (box 2's urgency routing: demand reads
        // all from the mirror, background ones all from the primary); anything
        // between splits it with at least one block on each side.
        let cut = if mf >= 1.0 {
            0
        } else if mf <= 0.0 {
            span
        } else {
            let frac = (1.0 - mf) as f64;
            (((span as f64 * frac) as usize / A as usize) * A as usize).clamp(A as usize, span - A as usize)
        };
        let base = abs - pad as u64;
        let (head, tail) = dst[..span].split_at_mut(cut);
        // A BACKGROUND read (`io_throttle`) goes in page-aligned chunks with the
        // pause hook before each, so an urgent read never queues behind more
        // than the chunks already at the drive. Read the mark HERE: the mirror
        // half runs on a thread spawned below, which would not inherit it.
        // Chunk size 0 means chunking is OFF: then no pause either, so
        // `V41_B2_SPEC_CHUNK_KB=0` is exactly the unchunked read. Loaded once
        // and clamped to a page, so it can never be 0 inside the loop.
        let (bg, chunk) = match (crate::io_throttle::background(), crate::io_throttle::chunk_bytes()) {
            (Some(token), c) if c > 0 => (Some(token), c.max(A as usize)),
            _ => (None, usize::MAX),
        };
        // Bytes each half must land: only [pad, pad + len) of the span matters
        // (the file's last block may be short), so the head needs its part of
        // that range and the tail the rest. A half that stops early is an
        // error, never stale staging returned as data.
        let need_head = cut.min(pad + len);
        let need_tail = (pad + len).saturating_sub(cut);
        let read_all = move |f: &File, buf: &mut [u8], off: u64, need: usize| -> eyre::Result<()> {
            let mut got = 0usize;
            while got < buf.len() {
                if let Some(token) = bg {
                    crate::io_throttle::pause(token);
                }
                let end = got.saturating_add(chunk).min(buf.len());
                let n = f.read_at(&mut buf[got..end], off + got as u64).wrap_err_with(|| format!("O_DIRECT split pread at {} for {}", off + got as u64, t.name))?;
                if n == 0 { break; }
                got += n;
            }
            if got < need {
                return Err(eyre!("{}: O_DIRECT split read at {off} got {got} of {need} needed bytes (short file?)", t.name));
            }
            Ok(())
        };
        // One side empty (an exact 0 / 1 fraction): read on this thread only.
        let (ra, rb) = if cut == 0 {
            (Ok(()), read_all(mirror, tail, base, need_tail))
        } else if cut == span {
            (read_all(file, head, base, need_head), Ok(()))
        } else {
            std::thread::scope(|sc| {
                let hb = sc.spawn(move || read_all(mirror, tail, base + cut as u64, need_tail));
                let ra = read_all(file, head, base, need_head);
                (ra, hb.join().unwrap_or_else(|_| Err(eyre!("mirror reader panicked"))))
            })
        };
        ra?;
        rb?;
        Ok(Some(pad))
    }

    /// [`Self::read_range_into_direct_padded`] from the PRIMARY drive.
    pub fn read_range_into_direct_padded(
        &self,
        t: &StTensor,
        byte_off: u64,
        len: usize,
        dst: &mut [u8],
    ) -> eyre::Result<Option<usize>> {
        self.read_range_into_direct_padded_on(t, byte_off, len, dst, false)
    }

    /// One unsplit O_DIRECT read of `[byte_off, byte_off + len)` into padded,
    /// 4096-aligned `dst`, from the primary drive, or from the mirror
    /// (`V41_EXPERT_MIRROR_DIR`) when `from_mirror` and the shard has one
    /// (else the primary). Returns the pad before the first requested byte.
    pub fn read_range_into_direct_padded_on(
        &self,
        t: &StTensor,
        byte_off: u64,
        len: usize,
        dst: &mut [u8],
        from_mirror: bool,
    ) -> eyre::Result<Option<usize>> {
        const A: u64 = 4096;
        let end = byte_off
            .checked_add(len as u64)
            .ok_or_else(|| eyre!("{}: range overflow", t.name))?;
        if end > t.len {
            return Err(eyre!("{}: range [{byte_off},{end}) exceeds tensor length {}", t.name, t.len));
        }
        let Some(Some(primary)) = self.direct_files.get(t.shard) else {
            return Ok(None);
        };
        let file = match (from_mirror, self.mirror_files.get(t.shard)) {
            (true, Some(Some(m))) => m,
            _ => primary,
        };
        if len == 0 {
            return Ok(Some(0));
        }
        let abs = t.offset + byte_off;
        let pad = (abs & (A - 1)) as usize;
        let span = ((pad + len) as u64).div_ceil(A) as usize * A as usize;
        if dst.len() < span {
            return Err(eyre!(
                "{}: direct dst {} < span {span} (pad {pad}, len {len})",
                t.name,
                dst.len()
            ));
        }
        if dst.as_ptr() as usize & (A as usize - 1) != 0 {
            return Err(eyre!("{}: direct dst is not {A}-aligned", t.name));
        }
        // The extent is rounded UP to a block boundary, which can run past EOF on
        // the last tensor of a shard; the kernel then returns a short read. That is
        // fine as long as the bytes the caller asked for arrived, so read until the
        // requested subrange is covered and treat Ok(0) as EOF rather than failure.
        let need = pad + len;
        let mut got = 0usize;
        while got < need {
            let n = file.read_at(&mut dst[got..span], abs - pad as u64 + got as u64).wrap_err_with(|| {
                format!(
                    "O_DIRECT padded pread at {} in {} for {}",
                    abs - pad as u64 + got as u64,
                    self.shard_names[t.shard],
                    t.name
                )
            })?;
            if n == 0 {
                return Err(eyre!(
                    "{}: O_DIRECT short read: got {got} of {need} (span {span})",
                    t.name
                ));
            }
            got += n;
        }
        Ok(Some(pad))
    }

    /// Bytes to allocate so [`Self::read_range_into_direct_padded`] can place
    /// `len` bytes at any alignment: one extra block for the head, then the
    /// length rounded up.
    pub fn direct_capacity_for(len: usize) -> usize {
        4096 + len.div_ceil(4096) * 4096
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
