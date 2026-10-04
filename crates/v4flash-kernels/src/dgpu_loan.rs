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
//! Generic over the donors: it knows device ranges, not what they hold.

use std::fs::File;
use std::os::unix::fs::FileExt;
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_hip::{DeviceBuffer, Event, PinnedBuffer, Stream};

/// Alignment of every sub-allocation.
pub const LOAN_ALIGN: usize = 256;

/// One donor: a non-owning view of an immutable device buffer.
pub struct Donor {
    pub name: String,
    pub view: DeviceBuffer<u8>,
}

/// The image's content check: one 64-bit hash per `chunk` bytes of each
/// donor range. Accidental corruption is what it guards against, so a fast
/// non-cryptographic mix is enough.
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
}

pub struct Loan {
    donors: Vec<Donor>,
    /// Bytes of each donor that the loan uses (a prefix; the last donor may be
    /// used only partly).
    used: Vec<usize>,
    image: PathBuf,
    /// Image layout: donor `d`'s used prefix starts at `image_off[d]`.
    image_off: Vec<u64>,
    /// `hashes[d][c]` = hash of donor `d`'s chunk `c` (`chunk` bytes, the last
    /// one shorter).
    hashes: Vec<Vec<u64>>,
    chunk: usize,
    image_written: bool,
}

impl Loan {
    /// Take donors in the given order until every buffer of `sizes` fits,
    /// placed first-fit in that order (`LOAN_ALIGN`-rounded): the placement
    /// [`LoanAlloc::take`] reproduces when the job asks for the same sizes in
    /// the same order. A donor is used up to its high-water mark only. Errors
    /// when the donors run out.
    pub fn new(candidates: Vec<Donor>, sizes: &[usize], image: PathBuf, chunk: usize) -> eyre::Result<Self> {
        if chunk == 0 || chunk % LOAN_ALIGN != 0 {
            return Err(eyre!("loan chunk {chunk} must be a non-zero multiple of {LOAN_ALIGN}"));
        }
        let cap = |d: &Donor| d.view.byte_len() / LOAN_ALIGN * LOAN_ALIGN;
        let mut cands = candidates.into_iter();
        let mut donors: Vec<Donor> = Vec::new();
        let mut used: Vec<usize> = Vec::new();
        for &s in sizes {
            let need = s.div_ceil(LOAN_ALIGN) * LOAN_ALIGN;
            loop {
                if let Some(i) = (0..donors.len()).find(|&i| cap(&donors[i]) - used[i] >= need) {
                    used[i] += need;
                    break;
                }
                let d = cands.next().ok_or_else(|| {
                    eyre!("dGPU loan: donors exhausted placing a {need} B buffer (sizes {sizes:?})")
                })?;
                donors.push(d);
                used.push(0);
            }
        }
        // A donor taken but never filled (everything after it fit earlier).
        let mut keep = used.iter().map(|u| *u > 0).collect::<Vec<_>>().into_iter();
        donors.retain(|_| keep.next().expect("same length"));
        used.retain(|u| *u > 0);
        let mut image_off = Vec::with_capacity(donors.len());
        let mut off = 0u64;
        for u in &used {
            image_off.push(off);
            off += *u as u64;
        }
        Ok(Loan { donors, used, image, image_off, hashes: Vec::new(), chunk, image_written: false })
    }

    pub fn total_bytes(&self) -> usize {
        self.used.iter().sum()
    }

    /// `(donor name, bytes used)` for the startup log.
    pub fn donor_summary(&self) -> Vec<(String, usize)> {
        self.donors.iter().zip(&self.used).map(|(d, u)| (d.name.clone(), *u)).collect()
    }

    pub fn image_path(&self) -> &Path {
        &self.image
    }

    /// Copy every donor's used bytes into the image file and record their
    /// hashes. Call once, before the first loan, while the donors hold their
    /// loaded contents. `stream` = a stream on the donors' device. The file's
    /// page cache is dropped after the write.
    pub fn write_image(&mut self, stream: &Stream) -> eyre::Result<()> {
        if let Some(dir) = self.image.parent() {
            std::fs::create_dir_all(dir).wrap_err_with(|| format!("create {}", dir.display()))?;
        }
        let f = File::create(&self.image).wrap_err_with(|| format!("create loan image {}", self.image.display()))?;
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
                f.write_all_at(bytes, self.image_off[d] + at as u64)
                    .wrap_err_with(|| format!("write loan image {}", self.image.display()))?;
                at += n;
            }
            self.hashes.push(hs);
        }
        f.sync_data().wrap_err("sync loan image")?;
        drop_page_cache(&f);
        self.image_written = true;
        Ok(())
    }

    /// The loan's memory as a first-fit allocator over the donor ranges.
    /// Errors until the image exists: lending before it would lose the bytes.
    pub fn allocator(&self) -> eyre::Result<LoanAlloc> {
        if !self.image_written {
            return Err(eyre!("dGPU loan: the image was never written; refusing to lend"));
        }
        Ok(LoanAlloc {
            ranges: self.donors.iter().zip(&self.used).map(|(d, u)| (d.view.slice_view(0, *u), 0usize)).collect(),
        })
    }

    /// Put every donor back from the image: pread a chunk into one of the two
    /// pinned `host` buffers (each at least `chunk` bytes) and H2D it on
    /// `stream`; the next chunk's read overlaps that copy. With `verify`, each
    /// chunk is read back into the other buffer and its hash compared with
    /// the image's, and a mismatched chunk is copied once more. Returns only
    /// when the device holds the image's bytes (`stream` synchronized).
    pub fn give_back(&self, stream: &Stream, host: &mut [PinnedBuffer<u8>; 2], verify: bool) -> eyre::Result<ReturnStats> {
        let t0 = Instant::now();
        if host.iter().any(|h| h.len() < self.chunk) {
            return Err(eyre!("give_back: need two pinned buffers of >= {} B", self.chunk));
        }
        let f = File::open(&self.image).wrap_err_with(|| format!("open loan image {}", self.image.display()))?;
        let copied = [Event::new_no_timing()?, Event::new_no_timing()?];
        let mut pending = [false, false];
        let mut st = ReturnStats::default();
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
                f.read_exact_at(&mut host[b].as_mut_slice()[..n], self.image_off[d] + at as u64)
                    .wrap_err_with(|| format!("read loan image {}", self.image.display()))?;
                st.read_ms += tr.elapsed().as_secs_f64() * 1e3;
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
        drop_page_cache(&f);
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

/// First-fit sub-allocation over the loan's ranges. Each allocation lies in
/// one range (never spans two donors) and starts `LOAN_ALIGN`-aligned.
pub struct LoanAlloc {
    ranges: Vec<(DeviceBuffer<u8>, usize)>,
}

impl LoanAlloc {
    /// An allocator over plain device buffers (tests, and the standalone GPU
    /// gate that runs without the V4.1 model).
    pub fn over(bufs: Vec<DeviceBuffer<u8>>) -> Self {
        LoanAlloc { ranges: bufs.into_iter().map(|b| (b, 0usize)).collect() }
    }

    /// `bytes` of loan memory, typed. Never frees: the whole allocator is
    /// dropped at the end of the job.
    pub fn take<T>(&mut self, len: usize) -> eyre::Result<DeviceBuffer<T>> {
        let bytes = len * std::mem::size_of::<T>();
        let need = bytes.div_ceil(LOAN_ALIGN) * LOAN_ALIGN;
        for (buf, at) in self.ranges.iter_mut() {
            if buf.byte_len() - *at >= need {
                // SAFETY: [at, at + bytes) lies inside `buf`, `at` is 256-B
                // aligned (any T here is a plain numeric type), and no other
                // allocation from this allocator overlaps it.
                let v = unsafe { buf.view_as::<T>(*at, len) };
                *at += need;
                return Ok(v);
            }
        }
        Err(eyre!(
            "dGPU loan: no range has {need} B free (ranges free: {:?})",
            self.ranges.iter().map(|(b, at)| b.byte_len() - at).collect::<Vec<_>>()
        ))
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
}
