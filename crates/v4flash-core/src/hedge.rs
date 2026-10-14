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
//! Primary stays primary (owner rule: "E100 primary"): the mirror is read only after the deadline, and each SITE has
//! its own cap on outstanding hedges (`Stats::cap`: hedged CALLS, not chunks -- one call's mirror reads share one
//! slot, held until its last mirror read, losers included, finishes); beyond it a slow read just waits, as without
//! hedging. Expert reads are hedged only on DECODE-critical threads (`critical_scope`, set by the pager's decode
//! `ensure` readers): prefill, late and prefetch reads never hedge, so a busy E100 under prefill cannot fill the cap.

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

/// Default per-site cap on hedged calls outstanding at once.
pub const MAX_HEDGES_IN_FLIGHT: usize = 8;

/// Per-site counters (cumulative since start): calls, fired, chunks hedged, won by primary, won by mirror, rescued
/// (primary failed), skipped (the site's in-flight cap was full). Plus the site's cap and its live in-flight count.
pub struct Stats {
    /// Hedged CALLS this site may have outstanding (a call = all its chunks' mirror reads).
    pub cap: usize,
    in_flight: AtomicUsize,
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
        Self::with_cap(MAX_HEDGES_IN_FLIGHT)
    }

    pub const fn with_cap(cap: usize) -> Self {
        Stats {
            cap,
            in_flight: AtomicUsize::new(0),
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

/// Engram row gathers of decode steps (`EngramTable::gather_hedged`). Its own cap: expert hedges never starve it.
pub static ENGRAM: Stats = Stats::new();
/// Decode-critical expert reads: the weight plane's primary stripe (`read_range_into_direct_mirrored`) and the scale
/// plane (`read_range_into_cached_hedged`).
pub static EXPERT: Stats = Stats::new();

thread_local! {
    /// This thread's reads are DECODE-critical (`critical_scope`): only then do expert reads hedge.
    static CRITICAL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Marks this thread's reads decode-critical (`on`) until the guard drops (restores the previous value). Set by the
/// pager's miss readers when the `ensure` serves DECODE rows; nothing else sets it.
pub struct CriticalScope(bool);

pub fn critical_scope(on: bool) -> CriticalScope {
    CriticalScope(CRITICAL.with(|c| c.replace(on)))
}

impl Drop for CriticalScope {
    fn drop(&mut self) {
        CRITICAL.with(|c| c.set(self.0));
    }
}

/// This thread is inside a decode-critical scope (to pass the mark on to helper threads).
pub fn is_critical() -> bool {
    CRITICAL.with(|c| c.get())
}

/// The expert hedge deadline for a read on THIS thread: the knob, and only inside a decode-critical scope.
pub fn expert_hedge_here() -> Option<Duration> {
    expert_hedge().filter(|_| CRITICAL.with(|c| c.get()))
}

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

struct Slot(&'static AtomicUsize);
impl Drop for Slot {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Relaxed);
    }
}

fn take_slot(stats: &'static Stats) -> Option<Slot> {
    let mut cur = stats.in_flight.load(Relaxed);
    loop {
        if cur >= stats.cap {
            return None;
        }
        match stats.in_flight.compare_exchange_weak(cur, cur + 1, Relaxed, Relaxed) {
            Ok(_) => return Some(Slot(&stats.in_flight)),
            Err(v) => cur = v,
        }
    }
}

/// Read `n` chunks, chunk `i` by `primary(i)` on its own detached thread; chunks not answered within `deadline` are
/// re-issued as `mirror(i)` (also detached); per chunk the FIRST successful answer is kept. A chunk whose primary
/// FAILS is re-issued on the mirror at once (if a hedge slot is free); an error is returned only when every issued
/// read of some chunk failed. Results in chunk order. `stats` is the site's counter set.
pub fn hedged<T, P, M>(n: usize, deadline: Duration, primary: Arc<P>, mirror: Arc<M>, stats: &'static Stats) -> eyre::Result<(Vec<T>, Report)>
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
        let (tx2, p) = (tx.clone(), primary.clone());
        let spawned = std::thread::Builder::new().name("hedge-primary".into()).spawn(move || {
            // A panicking reader still answers (or the caller would wait forever).
            let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| p(i))).unwrap_or_else(|_| Err(eyre!("hedged read: primary reader panicked")));
            let _ = tx2.send((i, Src::Primary, r));
        });
        if spawned.is_err() {
            // No thread: this chunk's primary read runs here, synchronously (as without hedging).
            let _ = tx.send((i, Src::Primary, primary(i)));
        }
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
            match take_slot(stats) {
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

    /// ZEROED (never exposes uninitialised memory); pooled callers pay the zeroing once (`BufPool`).
    pub fn new(len: usize) -> eyre::Result<Self> {
        let layout = std::alloc::Layout::from_size_align(len.max(Self::ALIGN), Self::ALIGN).map_err(|e| eyre!("aligned buffer: {e}"))?;
        // SAFETY: non-zero size, valid layout.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
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

/// A small pool of aligned buffers for the hedged expert reads (multi-MB: a fresh one per read is an mmap, page faults
/// and a TLB shootdown on free). `take(len)` reuses a pooled buffer of at least `len` bytes; a `Pooled` goes back on
/// drop (the winner after its copy, a loser whenever it finishes) unless the pool already holds `POOL_MAX`.
pub struct BufPool {
    free: std::sync::Mutex<Vec<AlignedBuf>>,
}

/// Buffers a pool keeps (~3 roles x a few concurrent misses x primary + mirror).
pub const POOL_MAX: usize = 16;

impl BufPool {
    pub const fn new() -> Self {
        BufPool { free: std::sync::Mutex::new(Vec::new()) }
    }

    pub fn take(&'static self, len: usize) -> eyre::Result<Pooled> {
        let got = {
            let mut f = self.free.lock().unwrap_or_else(|p| p.into_inner());
            // Smallest buffer that fits.
            let best = f.iter().enumerate().filter(|(_, b)| b.as_slice().len() >= len).min_by_key(|(_, b)| b.as_slice().len()).map(|(i, _)| i);
            best.map(|i| f.swap_remove(i))
        };
        let buf = match got {
            Some(b) => b,
            None => AlignedBuf::new(len)?,
        };
        Ok(Pooled { buf: Some(buf), pool: self })
    }

    /// Buffers held now (tests).
    pub fn held(&self) -> usize {
        self.free.lock().map(|f| f.len()).unwrap_or(0)
    }
}

impl Default for BufPool {
    fn default() -> Self {
        Self::new()
    }
}

/// The expert hedge's buffers.
pub static EXPERT_BUFS: BufPool = BufPool::new();

/// A pooled `AlignedBuf` (returned to its pool on drop). Its contents are whatever the last user left: callers read
/// only the bytes they just read into it.
pub struct Pooled {
    buf: Option<AlignedBuf>,
    pool: &'static BufPool,
}

impl Pooled {
    pub fn as_slice(&self) -> &[u8] {
        self.buf.as_ref().expect("held until drop").as_slice()
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.buf.as_mut().expect("held until drop").as_mut_slice()
    }
}

impl Drop for Pooled {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            let mut f = self.pool.free.lock().unwrap_or_else(|p| p.into_inner());
            if f.len() < POOL_MAX {
                f.push(b);
            }
        }
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
        static S: Stats = Stats::new();
        let s = &S;
        let (v, r) = hedged(4, Duration::from_millis(500), Arc::new(sleepy(1, 1)), Arc::new(sleepy(1, 2)), s).unwrap();
        assert_eq!(v, vec![vec![1, 0], vec![1, 1], vec![1, 2], vec![1, 3]]);
        assert!(!r.fired);
        assert_eq!((r.hedged, r.won_mirror), (0, 0));
        assert_eq!(s.snapshot()[0..2], [1, 0]);
    }

    #[test]
    fn stalled_primary_is_won_by_the_mirror_and_the_loser_is_harmless() {
        static S: Stats = Stats::new();
        let s = &S;
        let t = Instant::now();
        let (v, r) = hedged(3, Duration::from_millis(10), Arc::new(sleepy(400, 1)), Arc::new(sleepy(5, 2)), s).unwrap();
        assert!(t.elapsed() < Duration::from_millis(300), "waited for the stalled primary: {:?}", t.elapsed());
        assert_eq!(v, vec![vec![2, 0], vec![2, 1], vec![2, 2]], "the mirror's bytes, in chunk order");
        assert!(r.fired);
        assert_eq!((r.hedged, r.won_mirror, r.won_primary), (3, 3, 0));
        // The losers finish later into their own buffers; nothing to observe but that this does not crash.
        std::thread::sleep(Duration::from_millis(450));
    }

    #[test]
    fn only_outstanding_chunks_are_hedged() {
        static S: Stats = Stats::new();
        let s = &S;
        // Chunk 0 answers at once, chunks 1-2 stall on the primary.
        let p = |i: usize| -> eyre::Result<Vec<u8>> {
            std::thread::sleep(Duration::from_millis(if i == 0 { 1 } else { 300 }));
            Ok(vec![1, i as u8])
        };
        let (v, r) = hedged(3, Duration::from_millis(30), Arc::new(p), Arc::new(sleepy(2, 2)), s).unwrap();
        assert_eq!(v, vec![vec![1, 0], vec![2, 1], vec![2, 2]]);
        assert_eq!((r.hedged, r.won_mirror), (2, 2));
    }

    #[test]
    fn a_primary_that_answers_first_after_the_hedge_still_wins() {
        static S: Stats = Stats::new();
        let s = &S;
        let (v, r) = hedged(1, Duration::from_millis(5), Arc::new(sleepy(30, 1)), Arc::new(sleepy(300, 2)), s).unwrap();
        assert_eq!(v, vec![vec![1, 0]]);
        assert!(r.fired);
        assert_eq!((r.won_primary, r.won_mirror), (1, 0));
    }

    #[test]
    fn primary_error_is_rescued_by_the_mirror_and_double_failure_errors() {
        static S: Stats = Stats::new();
        let s = &S;
        let bad = |_: usize| -> eyre::Result<Vec<u8>> { Err(eyre!("E100 said no")) };
        let (v, r) = hedged(2, Duration::from_millis(500), Arc::new(bad), Arc::new(sleepy(1, 2)), s).unwrap();
        assert_eq!(v, vec![vec![2, 0], vec![2, 1]]);
        assert_eq!(r.rescued, 2);
        let bad2 = |_: usize| -> eyre::Result<Vec<u8>> { Err(eyre!("mirror said no")) };
        assert!(hedged(1, Duration::from_millis(500), Arc::new(bad), Arc::new(bad2), s).is_err());
    }

    #[test]
    fn the_cap_is_per_site_and_counts_calls() {
        static A: Stats = Stats::with_cap(1);
        static B: Stats = Stats::with_cap(1);
        let slow_m = || Arc::new(sleepy(200, 2));
        // A's one slot is taken by a hedged call whose mirror read is still running ...
        let h = std::thread::spawn(|| hedged(3, Duration::from_millis(1), Arc::new(sleepy(300, 1)), Arc::new(sleepy(200, 2)), &A).unwrap());
        std::thread::sleep(Duration::from_millis(50));
        // ... so a second A call cannot hedge (waits for its primary), while B (another site) still can.
        let (_, ra) = hedged(1, Duration::from_millis(1), Arc::new(sleepy(60, 1)), slow_m(), &A).unwrap();
        assert!(ra.fired && ra.hedged == 0, "{ra:?}");
        assert_eq!(A.snapshot()[6], 1, "one cap skip");
        let (_, rb) = hedged(1, Duration::from_millis(1), Arc::new(sleepy(300, 1)), Arc::new(sleepy(2, 2)), &B).unwrap();
        assert_eq!(rb.won_mirror, 1);
        let (_, r1) = h.join().unwrap();
        assert_eq!(r1.hedged, 3, "three chunks, one slot");
        std::thread::sleep(Duration::from_millis(350));
    }

    #[test]
    fn critical_scope_gates_the_expert_hedge_and_restores() {
        set_expert_hedge_ms(15);
        assert_eq!(expert_hedge_here(), None, "not a decode-critical thread");
        {
            let _g = critical_scope(true);
            assert_eq!(expert_hedge_here(), Some(Duration::from_millis(15)));
            {
                let _h = critical_scope(false);
                assert_eq!(expert_hedge_here(), None);
            }
            assert!(expert_hedge_here().is_some(), "restored");
        }
        assert_eq!(expert_hedge_here(), None);
        set_expert_hedge_ms(0);
    }

    #[test]
    fn pool_reuses_and_bounds() {
        static P: BufPool = BufPool::new();
        let a = P.take(10_000).unwrap();
        let pa = a.as_slice().as_ptr();
        assert!(a.as_slice().iter().all(|&b| b == 0), "fresh buffers are zeroed");
        drop(a);
        assert_eq!(P.held(), 1);
        let b = P.take(5_000).unwrap();
        assert_eq!(b.as_slice().as_ptr(), pa, "reused");
        let c = P.take(5_000).unwrap();
        assert_ne!(c.as_slice().as_ptr(), pa);
        drop((b, c));
        assert_eq!(P.held(), 2);
        let many: Vec<Pooled> = (0..POOL_MAX + 4).map(|_| P.take(64).unwrap()).collect();
        drop(many);
        assert_eq!(P.held(), POOL_MAX, "bounded");
    }

    /// Per-read OVERHEAD of the hedged path when the deadline is never reached (host only, no disk), shaped like
    /// `read_range_into_direct_mirrored`: the unhedged read lands in a bounce buffer then is copied to the destination;
    /// the hedged one lands in a pooled buffer on a detached thread (spawn + channel) and is copied from there (no
    /// extra copy). The "read" is a 6 MiB memcpy from RAM. Run: `cargo test -p v4flash-core --release --lib
    /// hedge::tests::overhead -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn overhead_when_the_deadline_is_never_reached() {
        static S: Stats = Stats::new();
        static P: BufPool = BufPool::new();
        const LEN: usize = 6 << 20;
        let src: Arc<Vec<u8>> = Arc::new((0..LEN).map(|i| (i * 7) as u8).collect());
        let mut dst = vec![0u8; LEN];
        let mut bounce = AlignedBuf::new(LEN).unwrap();
        let iters = 300;
        let bench = |f: &mut dyn FnMut()| -> f64 {
            for _ in 0..20 {
                f();
            }
            let t = Instant::now();
            for _ in 0..iters {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / iters as f64
        };
        let inline = bench(&mut || {
            bounce.as_mut_slice()[..LEN].copy_from_slice(&src);
            dst.copy_from_slice(&bounce.as_slice()[..LEN]);
        });
        // The same with the "read" landing from ANOTHER core (as DMA'd O_DIRECT bytes are cache-cold to the copying
        // core in both paths): a scoped thread fills the bounce, this thread copies out.
        let cross = bench(&mut || {
            let b = &mut bounce;
            std::thread::scope(|sc| {
                sc.spawn(|| b.as_mut_slice()[..LEN].copy_from_slice(&src));
            });
            dst.copy_from_slice(&bounce.as_slice()[..LEN]);
        });
        let (s1, s2) = (src.clone(), src.clone());
        let mut hedged_us = || {
            let (s1, s2) = (s1.clone(), s2.clone());
            let p = move |_: usize| -> eyre::Result<Pooled> {
                let mut b = P.take(LEN)?;
                b.as_mut_slice()[..LEN].copy_from_slice(&s1);
                Ok(b)
            };
            let m = move |_: usize| -> eyre::Result<Pooled> {
                let mut b = P.take(LEN)?;
                b.as_mut_slice()[..LEN].copy_from_slice(&s2);
                Ok(b)
            };
            let (mut v, r) = hedged(1, Duration::from_secs(5), Arc::new(p), Arc::new(m), &S).unwrap();
            assert!(!r.fired);
            dst.copy_from_slice(&v.pop().unwrap().as_slice()[..LEN]);
        };
        let hedged_t = bench(&mut hedged_us);
        let machinery = bench(&mut || {
            let (v, _) = hedged(1, Duration::from_secs(5), Arc::new(|_: usize| -> eyre::Result<u8> { Ok(1) }), Arc::new(|_: usize| -> eyre::Result<u8> { Ok(2) }), &S).unwrap();
            std::hint::black_box(v);
        });
        let fresh = bench(&mut || {
            let mut b = AlignedBuf::new(LEN).unwrap();
            b.as_mut_slice()[..LEN].copy_from_slice(&src);
            std::hint::black_box(&b);
        });
        eprintln!("6 MiB read stand-in: same-core {inline:.0} us; read on another core (scoped thread) {cross:.0} us; hedged (pooled, \
            deadline never reached) {hedged_t:.0} us => hedge overhead vs the cross-core read {:.0} us/read; machinery alone (spawn + \
            channel, empty read) {machinery:.0} us; a FRESH zeroed AlignedBuf + fill (no pool) {fresh:.0} us", hedged_t - cross);
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
