//! HEDGED READS (2026-10-14): a critical-path read whose PRIMARY copy (box 1's E100, `/weights2`) has not answered within
//! a deadline is re-issued against the MIRROR copy (the YMTC snapshot, `V41_EXPERT_MIRROR_DIR`) and the first answer
//! wins.
//!
//! WHY. The E100 (Crucial CT2000E100, DRAM-less) PAUSES reads for ~100 ms about once a second (latency 1 -> 11.5 ms,
//! in-flight 4 -> 13-17; the YMTC drive is unaffected), and the pause stalls whichever decode-critical read is
//! outstanding: the Engram table join at layer 0 -> 1 (~50% of the 105 ms decode spikes) and the synchronous box-1
//! expert miss reads (~48%). Both copies hold the same bytes, so the result -- and every output -- is bit-identical
//! whichever copy answers.
//!
//! SAFETY OF THE LOSER. A `pread` cannot be cancelled. Every job therefore runs on its OWN detached thread and reads
//! into a buffer it OWNS (`T`), handing it over through a channel: the winner's buffer is copied by the caller, the
//! loser finishes into its own buffer later and drops it (its send fails or is ignored). Nothing ever writes into a
//! caller's destination, a pager stage or a pinned buffer from a hedge thread: no torn writes, no use-after-free.
//!
//! Primary stays primary (owner rule: "E100 primary"): the mirror is read only after the deadline, and at most
//! `MAX_HEDGES_IN_FLIGHT` hedges are outstanding (beyond that a slow read just waits, as without hedging), so a
//! drive that is merely busy cannot double the mirror's load.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre};

/// Which copy answered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Src {
    Primary,
    Mirror,
}

/// What one hedged call did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Report {
    /// The deadline passed and mirror reads were issued.
    pub fired: bool,
    /// Chunks re-issued on the mirror.
    pub hedged: usize,
    /// Of those, the ones the primary still answered first / the mirror answered first.
    pub won_primary: usize,
    pub won_mirror: usize,
    /// Chunks whose primary read FAILED and that the mirror then served.
    pub rescued: usize,
}

/// Hedges outstanding at once, process-wide.
pub const MAX_HEDGES_IN_FLIGHT: usize = 8;
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Per-site counters (cumulative since start): calls, fired, chunks hedged, won by primary, won by mirror, rescued
/// (primary failed), skipped (the in-flight cap was full).
pub struct Stats {
    pub calls: AtomicU64,
    pub fired: AtomicU64,
    pub hedged: AtomicU64,
    pub won_primary: AtomicU64,
    pub won_mirror: AtomicU64,
    pub rescued: AtomicU64,
    pub skipped_cap: AtomicU64,
}

impl Stats {
    pub const fn new() -> Self {
        Stats {
            calls: AtomicU64::new(0),
            fired: AtomicU64::new(0),
            hedged: AtomicU64::new(0),
            won_primary: AtomicU64::new(0),
            won_mirror: AtomicU64::new(0),
            rescued: AtomicU64::new(0),
            skipped_cap: AtomicU64::new(0),
        }
    }

    fn note(&self, r: &Report) {
        self.calls.fetch_add(1, Relaxed);
        if r.fired {
            self.fired.fetch_add(1, Relaxed);
        }
        self.hedged.fetch_add(r.hedged as u64, Relaxed);
        self.won_primary.fetch_add(r.won_primary as u64, Relaxed);
        self.won_mirror.fetch_add(r.won_mirror as u64, Relaxed);
        self.rescued.fetch_add(r.rescued as u64, Relaxed);
    }

    /// (calls, fired, hedged chunks, won by primary, won by mirror, rescued, skipped at the cap).
    pub fn snapshot(&self) -> [u64; 7] {
        [
            self.calls.load(Relaxed),
            self.fired.load(Relaxed),
            self.hedged.load(Relaxed),
            self.won_primary.load(Relaxed),
            self.won_mirror.load(Relaxed),
            self.rescued.load(Relaxed),
            self.skipped_cap.load(Relaxed),
        ]
    }
}

impl Default for Stats {
    fn default() -> Self {
        Self::new()
    }
}

/// Engram row gathers of decode steps (`EngramTable::gather_hedged`).
pub static ENGRAM: Stats = Stats::new();
/// Expert weight-plane reads (`SafetensorsDir::read_range_into_direct_mirrored`).
pub static EXPERT: Stats = Stats::new();

/// `V41_EXPERT_HEDGE_MS` as pushed by the kernels' knob hook (0 = off): the expert read's deadline.
static EXPERT_HEDGE_US: AtomicU64 = AtomicU64::new(0);

pub fn set_expert_hedge_ms(ms: u64) {
    EXPERT_HEDGE_US.store(ms.saturating_mul(1000), Relaxed);
}

/// The expert read's hedge deadline, `None` = off.
pub fn expert_hedge() -> Option<Duration> {
    let us = EXPERT_HEDGE_US.load(Relaxed);
    (us > 0).then(|| Duration::from_micros(us))
}

/// GATE / TEST ONLY: every HEDGED primary read sleeps this long first (a forced E100 stall). Never set in production;
/// it affects only reads that already go through the hedged path (a hedge knob on).
static TEST_PRIMARY_DELAY_US: AtomicU64 = AtomicU64::new(0);

pub fn set_test_primary_delay_ms(ms: u64) {
    TEST_PRIMARY_DELAY_US.store(ms.saturating_mul(1000), Relaxed);
}

/// The injected stall, applied by the primary jobs of the hedged paths.
pub fn test_primary_delay() {
    let us = TEST_PRIMARY_DELAY_US.load(Relaxed);
    if us > 0 {
        std::thread::sleep(Duration::from_micros(us));
    }
}

struct Slot(#[allow(dead_code)] ());
impl Drop for Slot {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Relaxed);
    }
}

fn take_slot() -> Option<Slot> {
    let mut cur = IN_FLIGHT.load(Relaxed);
    loop {
        if cur >= MAX_HEDGES_IN_FLIGHT {
            return None;
        }
        match IN_FLIGHT.compare_exchange_weak(cur, cur + 1, Relaxed, Relaxed) {
            Ok(_) => return Some(Slot(())),
            Err(v) => cur = v,
        }
    }
}

/// Read `n` chunks, chunk `i` by `primary(i)` on its own detached thread; chunks not answered within `deadline` are
/// re-issued as `mirror(i)` (also detached); per chunk the FIRST successful answer is kept. A chunk whose primary
/// FAILS is re-issued on the mirror at once (if a hedge slot is free); an error is returned only when every issued
/// read of some chunk failed. Results in chunk order. `stats` is the site's counter set.
pub fn hedged<T, P, M>(n: usize, deadline: Duration, primary: Arc<P>, mirror: Arc<M>, stats: &Stats) -> eyre::Result<(Vec<T>, Report)>
where
    T: Send + 'static,
    P: Fn(usize) -> eyre::Result<T> + Send + Sync + 'static,
    M: Fn(usize) -> eyre::Result<T> + Send + Sync + 'static,
{
    let mut rep = Report::default();
    if n == 0 {
        return Ok((Vec::new(), rep));
    }
    let (tx, rx) = std::sync::mpsc::channel::<(usize, Src, eyre::Result<T>)>();
    for i in 0..n {
        let (tx, p) = (tx.clone(), primary.clone());
        std::thread::Builder::new()
            .name("hedge-primary".into())
            .spawn(move || {
                // A panicking reader still answers (or the caller would wait forever).
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p(i))).unwrap_or_else(|_| Err(eyre!("hedged read: primary reader panicked")));
                let _ = tx.send((i, Src::Primary, r));
            })
            .map_err(|e| eyre!("hedged read: spawn: {e}"))?;
    }
    // The mirror half: issued at most once per chunk. One hedge slot per call, held (shared) by every mirror thread
    // of the call until the last of them finishes -- losers included.
    let mut slot: Option<Arc<Slot>> = None;
    let mut out: Vec<Option<T>> = (0..n).map(|_| None).collect();
    let mut pend = vec![[true, false]; n]; // [primary, mirror] outstanding
    let mut hedged_chunk = vec![false; n];
    let mut errs: Vec<Option<eyre::Report>> = (0..n).map(|_| None).collect();
    let mut left = n;
    let t_dead = Instant::now() + deadline;
    let issue_mirror = |i: usize, slot: &mut Option<Arc<Slot>>, pend: &mut Vec<[bool; 2]>, hedged_chunk: &mut Vec<bool>| -> bool {
        if slot.is_none() {
            match take_slot() {
                Some(s) => *slot = Some(Arc::new(s)),
                None => {
                    stats.skipped_cap.fetch_add(1, Relaxed);
                    return false;
                }
            }
        }
        let (tx, m, held) = (tx.clone(), mirror.clone(), slot.clone());
        let ok = std::thread::Builder::new()
            .name("hedge-mirror".into())
            .spawn(move || {
                let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| m(i))).unwrap_or_else(|_| Err(eyre!("hedged read: mirror reader panicked")));
                let _ = tx.send((i, Src::Mirror, r));
                drop(held);
            })
            .is_ok();
        if ok {
            pend[i][1] = true;
            hedged_chunk[i] = true;
        }
        ok
    };
    while left > 0 {
        let msg = if !rep.fired {
            match rx.recv_timeout(t_dead.saturating_duration_since(Instant::now())) {
                Ok(m) => m,
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    rep.fired = true;
                    for i in 0..n {
                        if out[i].is_none() && !pend[i][1] && pend[i][0] {
                            issue_mirror(i, &mut slot, &mut pend, &mut hedged_chunk);
                        }
                    }
                    continue;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return Err(eyre!("hedged read: every reader gone")),
            }
        } else {
            rx.recv().map_err(|_| eyre!("hedged read: every reader gone"))?
        };
        let (i, src, r) = msg;
        let k = if src == Src::Primary { 0 } else { 1 };
        pend[i][k] = false;
        if out[i].is_some() {
            continue; // the loser of a chunk already answered: its buffer drops here
        }
        match r {
            Ok(v) => {
                out[i] = Some(v);
                left -= 1;
                if hedged_chunk[i] {
                    match src {
                        Src::Primary => rep.won_primary += 1,
                        Src::Mirror if errs[i].is_some() => rep.rescued += 1,
                        Src::Mirror => rep.won_mirror += 1,
                    }
                }
            }
            Err(e) => {
                if src == Src::Primary && !pend[i][1] && !hedged_chunk[i] {
                    // Primary failed before (or without) a hedge: try the mirror once.
                    errs[i] = Some(e);
                    if issue_mirror(i, &mut slot, &mut pend, &mut hedged_chunk) {
                        continue;
                    }
                    return Err(errs[i].take().expect("set above"));
                }
                if pend[i][0] || pend[i][1] {
                    errs[i] = Some(e); // the other copy may still answer
                    continue;
                }
                return Err(e);
            }
        }
    }
    drop(slot);
    rep.hedged = hedged_chunk.iter().filter(|&&h| h).count();
    stats.note(&rep);
    Ok((out.into_iter().map(|o| o.expect("every chunk answered")).collect(), rep))
}

/// A 4096-aligned heap buffer an O_DIRECT read can target, OWNED (movable between threads).
pub struct AlignedBuf {
    ptr: *mut u8,
    layout: std::alloc::Layout,
}

// SAFETY: the buffer is uniquely owned (no aliasing pointer escapes except through `&self` / `&mut self`).
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub const ALIGN: usize = 4096;

    pub fn new(len: usize) -> eyre::Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len.max(Self::ALIGN), Self::ALIGN).map_err(|e| eyre!("aligned buffer: {e}"))?;
        // SAFETY: non-zero size, valid layout.
        let ptr = unsafe { std::alloc::alloc(layout) };
        if ptr.is_null() {
            return Err(eyre!("aligned buffer: alloc of {len} failed"));
        }
        Ok(AlignedBuf { ptr, layout })
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `ptr` owns `layout.size()` bytes for the lifetime of `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.layout.size()) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.layout.size()) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, self.layout) }
    }
}

/// O_DIRECT read of `want` bytes at the 4096-aligned file offset `at` into `buf` (a short read past EOF is fine once
/// `want` bytes are in, as in `SafetensorsDir::read_range_into_direct`).
pub fn read_direct_exact(f: &std::fs::File, buf: &mut [u8], at: u64, want: usize) -> eyre::Result<()> {
    use std::os::unix::fs::FileExt;
    let mut got = 0usize;
    while got < want {
        let n = f.read_at(&mut buf[got..], at + got as u64).map_err(|e| eyre!("O_DIRECT pread at {}: {e}", at + got as u64))?;
        if n == 0 {
            return Err(eyre!("O_DIRECT short read at {}: got {got} of {want}", at + got as u64));
        }
        got += n;
    }
    Ok(())
}

/// O_DIRECT read of the small range `[abs, abs + dst.len())` (any alignment) through `scratch` (>= the aligned span).
pub fn read_direct_small(f: &std::fs::File, abs: u64, dst: &mut [u8], scratch: &mut AlignedBuf) -> eyre::Result<()> {
    const A: u64 = AlignedBuf::ALIGN as u64;
    let lo = abs & !(A - 1);
    let hi = (abs + dst.len() as u64 + A - 1) & !(A - 1);
    let span = (hi - lo) as usize;
    if scratch.as_slice().len() < span {
        *scratch = AlignedBuf::new(span)?;
    }
    let head = (abs - lo) as usize;
    let buf = scratch.as_mut_slice();
    read_direct_exact(f, &mut buf[..span], lo, head + dst.len())?;
    dst.copy_from_slice(&buf[head..head + dst.len()]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sleepy(ms: u64, tag: u8) -> impl Fn(usize) -> eyre::Result<Vec<u8>> + Send + Sync + 'static {
        move |i| {
            std::thread::sleep(Duration::from_millis(ms));
            Ok(vec![tag, i as u8])
        }
    }

    #[test]
    fn fast_primary_never_hedges() {
        let s = Stats::new();
        let (v, r) = hedged(4, Duration::from_millis(500), Arc::new(sleepy(1, 1)), Arc::new(sleepy(1, 2)), &s).unwrap();
        assert_eq!(v, vec![vec![1, 0], vec![1, 1], vec![1, 2], vec![1, 3]]);
        assert!(!r.fired);
        assert_eq!((r.hedged, r.won_mirror), (0, 0));
        assert_eq!(s.snapshot()[0..2], [1, 0]);
    }

    #[test]
    fn stalled_primary_is_won_by_the_mirror_and_the_loser_is_harmless() {
        let s = Stats::new();
        let t = Instant::now();
        let (v, r) = hedged(3, Duration::from_millis(10), Arc::new(sleepy(400, 1)), Arc::new(sleepy(5, 2)), &s).unwrap();
        assert!(t.elapsed() < Duration::from_millis(300), "waited for the stalled primary: {:?}", t.elapsed());
        assert_eq!(v, vec![vec![2, 0], vec![2, 1], vec![2, 2]], "the mirror's bytes, in chunk order");
        assert!(r.fired);
        assert_eq!((r.hedged, r.won_mirror, r.won_primary), (3, 3, 0));
        // The losers finish later into their own buffers; nothing to observe but that this does not crash.
        std::thread::sleep(Duration::from_millis(450));
    }

    #[test]
    fn only_outstanding_chunks_are_hedged() {
        let s = Stats::new();
        // Chunk 0 answers at once, chunks 1-2 stall on the primary.
        let p = |i: usize| -> eyre::Result<Vec<u8>> {
            std::thread::sleep(Duration::from_millis(if i == 0 { 1 } else { 300 }));
            Ok(vec![1, i as u8])
        };
        let (v, r) = hedged(3, Duration::from_millis(30), Arc::new(p), Arc::new(sleepy(2, 2)), &s).unwrap();
        assert_eq!(v, vec![vec![1, 0], vec![2, 1], vec![2, 2]]);
        assert_eq!((r.hedged, r.won_mirror), (2, 2));
    }

    #[test]
    fn a_primary_that_answers_first_after_the_hedge_still_wins() {
        let s = Stats::new();
        let (v, r) = hedged(1, Duration::from_millis(5), Arc::new(sleepy(30, 1)), Arc::new(sleepy(300, 2)), &s).unwrap();
        assert_eq!(v, vec![vec![1, 0]]);
        assert!(r.fired);
        assert_eq!((r.won_primary, r.won_mirror), (1, 0));
    }

    #[test]
    fn primary_error_is_rescued_by_the_mirror_and_double_failure_errors() {
        let s = Stats::new();
        let bad = |_: usize| -> eyre::Result<Vec<u8>> { Err(eyre!("E100 said no")) };
        let (v, r) = hedged(2, Duration::from_millis(500), Arc::new(bad), Arc::new(sleepy(1, 2)), &s).unwrap();
        assert_eq!(v, vec![vec![2, 0], vec![2, 1]]);
        assert_eq!(r.rescued, 2);
        let bad2 = |_: usize| -> eyre::Result<Vec<u8>> { Err(eyre!("mirror said no")) };
        assert!(hedged(1, Duration::from_millis(500), Arc::new(bad), Arc::new(bad2), &s).is_err());
    }

    #[test]
    fn small_direct_read_handles_unaligned_ranges() {
        // A plain (buffered) temp file stands in for the O_DIRECT handle: the alignment logic is what is tested.
        let p = std::env::temp_dir().join(format!("hedge-small-{}", std::process::id()));
        let data: Vec<u8> = (0..20000u32).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::write(&p, &data).unwrap();
        let f = std::fs::File::open(&p).unwrap();
        let mut sc = AlignedBuf::new(4096).unwrap();
        for (abs, len) in [(0u64, 256usize), (4000, 256), (4095, 8), (8190, 264), (19000, 1000)] {
            let mut d = vec![0u8; len];
            read_direct_small(&f, abs, &mut d, &mut sc).unwrap();
            assert_eq!(&d[..], &data[abs as usize..abs as usize + len], "at {abs}+{len}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
