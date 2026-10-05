//! A LOAN of dGPU memory: in-place views over immutable device buffers (the
//! donors), handed to a short exclusive job and put back byte-exact afterwards
//! from an on-disk image written once (docs/v41/EMBED_PHASE_DESIGN.md §4).
//!
//! Nothing is allocated or freed on the device: every view aliases a donor's
//! existing allocation, so any pointer baked into a captured graph stays
//! valid. The donors' CONTENTS are clobbered while the loan is out; the caller
//! guarantees nothing reads them until [`Loan::give_back`] has returned (the
//! hub: an exclusive embed phase between scheduler ticks).
//!
//! The job's buffers are PLANNED once ([`Loan::new`]) with CANARIES around
//! them: a band before the first buffer of each donor, a gap between
//! consecutive buffers, and a guard band after the last. Canaries are imaged
//! but never lent; every return first checks that none changed (a kernel wrote
//! outside its buffer).
//!
//! Generic over the donors: it knows device ranges, not what they hold.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::PathBuf;
use std::time::Instant;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::direct_io::{align_up, DirectFiles, DIRECT_ALIGN};
use v4flash_hip::{DeviceBuffer, Event, PinnedBuffer, Stream};

/// Alignment of every lent buffer.
pub const LOAN_ALIGN: usize = 256;

/// Canary before each donor's first buffer and between consecutive buffers.
pub const CANARY_BYTES: usize = 64 << 10;

/// One donor: a non-owning view of an immutable device buffer.
pub struct Donor {
    pub name: String,
    pub view: DeviceBuffer<u8>,
}

/// The image's content check: one 64-bit hash per `chunk` bytes of each
/// donor range, and one per canary. Accidental corruption is what it guards
/// against, so a fast non-cryptographic mix is enough.
pub fn chunk_hash(bytes: &[u8]) -> u64 {
    const P1: u64 = 0x9E37_79B9_7F4A_7C15;
    const P2: u64 = 0xC2B2_AE3D_27D4_EB4F;
    let mut h: u64 = (bytes.len() as u64).wrapping_mul(P1);
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        let v = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
        h = (h ^ v.wrapping_mul(P2)).rotate_left(29).wrapping_mul(P1);
    }
    for &b in words.remainder() {
        h = (h ^ b as u64).wrapping_mul(P1);
    }
    h ^ (h >> 31)
}

/// What a return cost.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReturnStats {
    pub bytes: usize,
    pub read_ms: f64,
    pub total_ms: f64,
    pub verify_ms: f64,
    /// Chunks whose read-back hash differed on the first attempt (retried).
    pub retried: u32,
    /// Canaries that changed while lent: a kernel wrote outside its buffer.
    /// The return repairs them, but what else it hit is unknown.
    pub guard_violations: u32,
}

/// Where one lent buffer lives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Placement {
    donor: usize,
    offset: usize,
    bytes: usize,
}

/// Plan `sizes` (in order) first-fit into donors of capacity `caps` (taken
/// in order as needed): `canary` before each donor's first buffer and after
/// every buffer, then `guard` after the last (capped at the donor's size).
/// Returns the placements and each donor's imaged bytes.
fn plan(caps: &[usize], sizes: &[usize], canary: usize, guard: usize) -> eyre::Result<(Vec<Placement>, Vec<usize>)> {
    let mut cursor: Vec<usize> = Vec::new();
    let mut out = Vec::with_capacity(sizes.len());
    for &s in sizes {
        let need = s.div_ceil(LOAN_ALIGN) * LOAN_ALIGN;
        loop {
            if let Some(d) = (0..cursor.len()).find(|&d| cursor[d] + need + canary <= caps[d]) {
                out.push(Placement { donor: d, offset: cursor[d], bytes: need });
                cursor[d] += need + canary;
                break;
            }
            if cursor.len() == caps.len() {
                return Err(eyre!("dGPU loan: donors exhausted placing a {need} B buffer (sizes {sizes:?})"));
            }
            cursor.push(canary);
        }
    }
    // Imaged: through the last buffer's trailing canary, plus the guard.
    let used = cursor.iter().zip(caps).map(|(c, cap)| (c + guard.saturating_sub(canary)).min(*cap)).collect();
    Ok((out, used))
}

pub struct Loan {
    donors: Vec<Donor>,
    plan: Vec<Placement>,
    /// Bytes of each donor in the image (buffers + canaries + guard).
    used: Vec<usize>,
    /// Canary regions: (donor, offset, len), everything imaged but not lent.
    canaries: Vec<(usize, usize, usize)>,
    canary_hash: Vec<u64>,
    /// Identical copies of the image, one per drive: a return reads its
    /// chunks O_DIRECT over all of them in parallel.
    images: Vec<PathBuf>,
    /// Image layout: donor `d`'s imaged prefix starts at `image_off[d]`
    /// (`DIRECT_ALIGN`-aligned).
    image_off: Vec<u64>,
    /// `hashes[d][c]` = hash of donor `d`'s chunk `c` (`chunk` bytes, the last
    /// one shorter).
    hashes: Vec<Vec<u64>>,
    chunk: usize,
    /// Every image replica, open and `flock`ed for the life of the process: a
    /// second hub on the same path fails at startup instead of overwriting it.
    locks: Vec<File>,
    /// O_DIRECT readers over the replicas (after `write_image`).
    reader: Option<DirectFiles>,
}

impl Loan {
    /// Plan the job's buffers (`sizes`, in the order it will `take` them)
    /// over the donors, taken in the given order as needed, with
    /// [`CANARY_BYTES`] canaries and a `guard`-byte band after each donor's
    /// last buffer. Errors when the donors run out. `chunk` (the image's I/O
    /// and hash unit, a multiple of `DIRECT_ALIGN`) must hold the largest
    /// canary. `images`: one path per drive (at least one).
    pub fn new(candidates: Vec<Donor>, sizes: &[usize], images: Vec<PathBuf>, chunk: usize, guard: usize) -> eyre::Result<Self> {
        if chunk == 0 || chunk % DIRECT_ALIGN != 0 {
            return Err(eyre!("loan chunk {chunk} must be a non-zero multiple of {DIRECT_ALIGN}"));
        }
        if images.is_empty() {
            return Err(eyre!("dGPU loan: no image path"));
        }
        let canary = CANARY_BYTES.min(chunk);
        let guard = guard.max(canary).min(chunk) / LOAN_ALIGN * LOAN_ALIGN;
        let caps: Vec<usize> = candidates.iter().map(|d| d.view.byte_len() / LOAN_ALIGN * LOAN_ALIGN).collect();
        let (mut plan, used) = plan(&caps, sizes, canary, guard)?;
        // Keep only donors that hold a buffer (one taken while a big buffer
        // looked for room may end up holding nothing), renumbering the plan.
        let mut keep: Vec<Option<usize>> = vec![None; used.len()];
        let mut donors = Vec::new();
        let mut kept_used = Vec::new();
        for (d, cand) in candidates.into_iter().take(used.len()).enumerate() {
            if plan.iter().any(|p| p.donor == d) {
                keep[d] = Some(donors.len());
                donors.push(cand);
                kept_used.push(used[d]);
            }
        }
        for p in plan.iter_mut() {
            p.donor = keep[p.donor].expect("a placement's donor is kept");
        }
        let used = kept_used;
        let mut canaries = Vec::new();
        for (d, &u) in used.iter().enumerate() {
            let mut at = 0;
            for p in plan.iter().filter(|p| p.donor == d) {
                if p.offset > at {
                    canaries.push((d, at, p.offset - at));
                }
                at = p.offset + p.bytes;
            }
            if u > at {
                canaries.push((d, at, u - at));
            }
        }
        let mut image_off = Vec::with_capacity(donors.len());
        let mut off = 0u64;
        for u in &used {
            image_off.push(off);
            off = align_up(off + *u as u64);
        }
        Ok(Loan {
            donors, plan, used, canaries, canary_hash: Vec::new(), images, image_off, hashes: Vec::new(), chunk,
            locks: Vec::new(), reader: None,
        })
    }

    /// The image file's size: every donor region, each starting aligned.
    fn image_bytes(&self) -> u64 {
        self.image_off.last().map_or(0, |o| align_up(o + *self.used.last().expect("donors") as u64))
    }

    /// Bytes lent (the job's buffers).
    pub fn lent_bytes(&self) -> usize {
        self.plan.iter().map(|p| p.bytes).sum()
    }

    /// Bytes imaged and returned per phase (buffers + canaries + guard).
    pub fn total_bytes(&self) -> usize {
        self.used.iter().sum()
    }

    /// `(donor name, bytes imaged)` for the startup log.
    pub fn donor_summary(&self) -> Vec<(String, usize)> {
        self.donors.iter().zip(&self.used).map(|(d, u)| (d.name.clone(), *u)).collect()
    }

    pub fn image_paths(&self) -> &[PathBuf] {
        &self.images
    }

    /// `(path, O_DIRECT?)` per image replica, for the startup log.
    pub fn image_readers(&self) -> Vec<(String, bool)> {
        self.reader.as_ref().map(|r| r.describe()).unwrap_or_default()
    }

    /// Copy every donor's imaged bytes into every image replica and record
    /// their hashes (chunks and canaries). Call once, before the first loan,
    /// while the donors hold their loaded contents. `stream` = a stream on
    /// the donors' device. Takes an exclusive `flock` on each replica first
    /// (held until the process exits). The files' page cache is dropped
    /// after; returns read them O_DIRECT.
    pub fn write_image(&mut self, stream: &Stream) -> eyre::Result<()> {
        let mut files = Vec::with_capacity(self.images.len());
        for p in &self.images {
            if let Some(dir) = p.parent() {
                std::fs::create_dir_all(dir).wrap_err_with(|| format!("create {}", dir.display()))?;
            }
            // Open WITHOUT truncating: the lock must be ours before a byte changes.
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(p)
                .wrap_err_with(|| format!("open loan image {}", p.display()))?;
            if !try_lock_exclusive(&f) {
                return Err(eyre!(
                    "loan image {} is locked by another process (another hub?); set V41_EMBED_LOAN_IMAGE to private paths",
                    p.display()
                ));
            }
            f.set_len(self.image_bytes()).wrap_err("size loan image")?;
            files.push(f);
        }
        let mut host = PinnedBuffer::<u8>::new(self.chunk)?;
        self.hashes.clear();
        for (d, donor) in self.donors.iter().enumerate() {
            let mut hs = Vec::new();
            let mut at = 0usize;
            while at < self.used[d] {
                let n = self.chunk.min(self.used[d] - at);
                donor.view.slice_view(at, n).copy_to_pinned_async(&mut host, 0, stream)?;
                stream.synchronize()?;
                let bytes = &host.as_slice()[..n];
                hs.push(chunk_hash(bytes));
                for (f, p) in files.iter().zip(&self.images) {
                    f.write_all_at(bytes, self.image_off[d] + at as u64)
                        .wrap_err_with(|| format!("write loan image {}", p.display()))?;
                }
                at += n;
            }
            self.hashes.push(hs);
        }
        self.canary_hash.clear();
        for &(d, off, len) in &self.canaries {
            self.donors[d].view.slice_view(off, len).copy_to_pinned_async(&mut host, 0, stream)?;
            stream.synchronize()?;
            self.canary_hash.push(chunk_hash(&host.as_slice()[..len]));
        }
        for f in &files {
            f.sync_data().wrap_err("sync loan image")?;
            drop_page_cache(f);
        }
        self.locks = files;
        self.reader = Some(DirectFiles::open(&self.images)?);
        Ok(())
    }

    /// The planned buffers, in plan order. Errors until the image exists:
    /// lending before it would lose the bytes.
    pub fn allocator(&self) -> eyre::Result<LoanAlloc> {
        if self.reader.is_none() {
            return Err(eyre!("dGPU loan: the image was never written; refusing to lend"));
        }
        Ok(LoanAlloc {
            ranges: self.plan.iter().map(|p| self.donors[p.donor].view.slice_view(p.offset, p.bytes)).collect(),
            next: 0,
        })
    }

    /// Canaries that no longer match the image (`scratch`: a pinned buffer
    /// of at least `chunk` bytes).
    fn canary_violations(&self, stream: &Stream, scratch: &mut PinnedBuffer<u8>) -> eyre::Result<u32> {
        let mut bad = 0;
        for (i, &(d, off, len)) in self.canaries.iter().enumerate() {
            self.donors[d].view.slice_view(off, len).copy_to_pinned_async(scratch, 0, stream)?;
            stream.synchronize()?;
            if chunk_hash(&scratch.as_slice()[..len]) != self.canary_hash[i] {
                bad += 1;
            }
        }
        Ok(bad)
    }

    /// Put every donor back from the image: read a chunk into one of the two
    /// pinned `host` buffers (each at least `chunk` bytes) with `readers`
    /// O_DIRECT threads over the image replicas, check it against its startup
    /// hash (disk corruption; always), and H2D it on `stream`; the next
    /// chunk's read overlaps that copy. With `verify`, each chunk is also
    /// read back into the other buffer and hashed, and a mismatched chunk is
    /// copied once more. Canaries are checked before anything is
    /// overwritten. Returns only when the device holds the image's bytes
    /// (`stream` synchronized).
    pub fn give_back(&self, stream: &Stream, host: &mut [PinnedBuffer<u8>; 2], verify: bool, readers: usize) -> eyre::Result<ReturnStats> {
        let t0 = Instant::now();
        if host.iter().any(|h| h.len() < self.chunk) {
            return Err(eyre!("give_back: need two pinned buffers of >= {} B", self.chunk));
        }
        let rd = self.reader.as_ref().ok_or_else(|| eyre!("give_back: the image was never written"))?;
        let mut st = ReturnStats::default();
        st.guard_violations = self.canary_violations(stream, &mut host[1])?;
        let copied = [Event::new_no_timing()?, Event::new_no_timing()?];
        let mut pending = [false, false];
        let mut k = 0usize;
        for (d, donor) in self.donors.iter().enumerate() {
            let mut at = 0usize;
            let mut c = 0usize;
            while at < self.used[d] {
                let n = self.chunk.min(self.used[d] - at);
                let b = k % 2;
                // This buffer's previous H2D must be done before it is refilled.
                if pending[b] {
                    copied[b].synchronize()?;
                    pending[b] = false;
                }
                let tr = Instant::now();
                // Chunks start aligned (aligned donor regions, chunk a
                // multiple of DIRECT_ALIGN): the bytes land at offset 0.
                let head = rd.read_span(self.image_off[d] + at as u64, n, host[b].as_mut_slice(), readers)?;
                if head != 0 {
                    return Err(eyre!("loan image chunk at {} is not aligned", self.image_off[d] + at as u64));
                }
                st.read_ms += tr.elapsed().as_secs_f64() * 1e3;
                if chunk_hash(&host[b].as_slice()[..n]) != self.hashes[d][c] {
                    return Err(eyre!(
                        "loan image {:?} bytes for donor {} [{at}, {}) do not match their startup hash (disk corruption?)",
                        self.images,
                        donor.name,
                        at + n
                    ));
                }
                let mut dst = donor.view.slice_view(at, n);
                dst.copy_from_host_async(&host[b].as_slice()[..n], stream)?;
                if verify {
                    let tv = Instant::now();
                    // Both buffers' copies are done after this, so the other
                    // one is free as read-back scratch.
                    stream.synchronize()?;
                    pending = [false, false];
                    let [h0, h1] = &mut *host;
                    let (src, scratch) = if b == 0 { (&*h0, h1) } else { (&*h1, h0) };
                    if !chunk_matches(&dst, stream, scratch, n, self.hashes[d][c])? {
                        st.retried += 1;
                        dst.copy_from_host_async(&src.as_slice()[..n], stream)?;
                        stream.synchronize()?;
                        if !chunk_matches(&dst, stream, scratch, n, self.hashes[d][c])? {
                            return Err(eyre!(
                                "dGPU loan: donor {} bytes [{at}, {}) differ from the image after two returns",
                                donor.name,
                                at + n
                            ));
                        }
                    }
                    st.verify_ms += tv.elapsed().as_secs_f64() * 1e3;
                } else {
                    copied[b].record(stream)?;
                    pending[b] = true;
                }
                st.bytes += n;
                at += n;
                c += 1;
                k += 1;
            }
        }
        stream.synchronize()?;
        // O_DIRECT reads leave nothing cached; a buffered fallback replica
        // (a filesystem without O_DIRECT) would.
        for f in &self.locks {
            drop_page_cache(f);
        }
        st.total_ms = t0.elapsed().as_secs_f64() * 1e3;
        Ok(st)
    }
}

/// D2H `dst` into `scratch` and compare its hash with `want`.
fn chunk_matches(dst: &DeviceBuffer<u8>, stream: &Stream, scratch: &mut PinnedBuffer<u8>, n: usize, want: u64) -> eyre::Result<bool> {
    dst.copy_to_pinned_async(scratch, 0, stream)?;
    stream.synchronize()?;
    Ok(chunk_hash(&scratch.as_slice()[..n]) == want)
}

fn drop_page_cache(f: &File) {
    unsafe {
        libc_fadvise_dontneed(f.as_raw_fd());
    }
}

unsafe fn libc_fadvise_dontneed(fd: i32) {
    // POSIX_FADV_DONTNEED = 4 on Linux.
    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
    }
    let _ = posix_fadvise(fd, 0, 0, 4);
}

/// `flock(LOCK_EX | LOCK_NB)`: false when another open file holds it.
fn try_lock_exclusive(f: &File) -> bool {
    extern "C" {
        fn flock(fd: i32, operation: i32) -> i32;
    }
    // LOCK_EX = 2, LOCK_NB = 4.
    unsafe { flock(f.as_raw_fd(), 2 | 4) == 0 }
}

/// The job's buffers, handed out in plan order.
pub struct LoanAlloc {
    ranges: Vec<DeviceBuffer<u8>>,
    next: usize,
}

impl LoanAlloc {
    /// Plain device buffers as the plan (tests, and the standalone GPU gate
    /// that runs without the V4.1 model): one buffer per `take`, in order.
    pub fn over(bufs: Vec<DeviceBuffer<u8>>) -> Self {
        LoanAlloc { ranges: bufs, next: 0 }
    }

    /// The next planned buffer, typed (`len` elements of `T`; any T here is a
    /// plain numeric type, and every buffer starts `LOAN_ALIGN`-aligned).
    pub fn take<T>(&mut self, len: usize) -> eyre::Result<DeviceBuffer<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let buf = self.ranges.get(self.next).ok_or_else(|| eyre!("dGPU loan: the plan has only {} buffers", self.ranges.len()))?;
        if bytes > buf.byte_len() {
            return Err(eyre!("dGPU loan: buffer {} wants {bytes} B, planned {}", self.next, buf.byte_len()));
        }
        self.next += 1;
        // SAFETY: [0, bytes) lies inside the planned range, which no other
        // `take` overlaps; offset 0 is aligned for T.
        Ok(unsafe { buf.view_as::<T>(0, len) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_hash_sees_one_flipped_bit() {
        let mut v: Vec<u8> = (0..4099u32).map(|i| (i * 7 + 3) as u8).collect();
        let h = chunk_hash(&v);
        for i in [0usize, 7, 8, 4095, 4098] {
            v[i] ^= 0x10;
            assert_ne!(chunk_hash(&v), h, "byte {i}");
            v[i] ^= 0x10;
        }
        assert_eq!(chunk_hash(&v), h);
        assert_ne!(chunk_hash(&v[..4098]), h);
    }

    #[test]
    fn plan_surrounds_every_buffer_with_canaries() {
        let c = 1024;
        let (p, used) = plan(&[10_000, 1 << 20], &[3000, 500, 9000, 100], c, 4096).unwrap();
        // 3000 -> 3072 at 1024 in donor 0; 500 -> 512 after a canary; 9000
        // does not fit donor 0 -> donor 1; 100 -> 256 back in donor 0.
        assert_eq!(p[0], Placement { donor: 0, offset: c, bytes: 3072 });
        assert_eq!(p[1], Placement { donor: 0, offset: c + 3072 + c, bytes: 512 });
        assert_eq!(p[2], Placement { donor: 1, offset: c, bytes: 9216 });
        assert_eq!(p[3].donor, 0);
        assert_eq!(p[3].offset, p[1].offset + 512 + c);
        // Imaged through the guard, capped at the donor.
        assert_eq!(used[0], 10_000);
        assert_eq!(used[1], c + 9216 + 4096);
        assert!(plan(&[1000], &[2000], c, 0).is_err());
    }
}
