//! LAUNCH + COPY RING (HIP 700 instrumentation, owner-approved 2026-10-19): the last [`RING_LEN`] kernel launches,
//! copies, memsets, graph launches and stream value ops PER DEVICE, so a sticky HIP error can be traced back to the
//! operations queued before it. [`dump`] prints them (the sticky-error exit in [`crate::fatal`], or on demand through
//! the `V41_HIP_RING_DUMP` knob).
//!
//! Lock-free: a writer takes a global index with one `fetch_add`, then CLAIMS its slot with a compare-exchange of the
//! slot's sequence number from a complete OLDER entry (`2j + 2`, `j < i`) to "entry `i` being written" (`2i + 1`), and
//! releases it as `2i + 2`. A writer that cannot claim (another writer still owns the slot, or a newer entry already
//! landed there) DROPS its entry (`dropped`): no two writers ever store into one slot at once (review r1, 10-19). A
//! reader keeps an entry only when the sequence was `2i + 2` before and after it read the fields. Every field is one
//! atomic word and the kind is an interned index ([`intern`]), so even a discarded read never forms an invalid
//! value. ~17 relaxed stores + a TSC read per entry (`bench_record_ns`). `V41_HIP_RING=0` (STATIC) turns recording off.
//!
//! Each entry: kind (kernel name or copy kind), grid / size, the first 8 argument words (copies: dst, src, bytes, ...),
//! the thread's (layer, lane) context ([`scope`]), the call site's `Location`, a monotonic timestamp and the thread id.
use std::cell::Cell;
use std::collections::HashMap;
use std::panic::Location;
use std::sync::atomic::{fence, AtomicPtr, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex, OnceLock};
use std::time::Instant;

/// Entries kept per device.
pub const RING_LEN: usize = 64;
/// Argument words kept per entry.
pub const RING_ARGS: usize = 8;
/// Rings: devices `0..RING_DEVICES - 1`, the last one for any other id (`-1` = no device).
const RING_DEVICES: usize = 9;

/// `V41_HIP_RING` (STATIC, default ON): `0|off|false|no` disables recording.
pub fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        !matches!(
            std::env::var("V41_HIP_RING").map(|v| v.trim().to_ascii_lowercase()).as_deref(),
            Ok("0") | Ok("off") | Ok("false") | Ok("no")
        )
    });
    *ON
}

// ---- interned kinds --------------------------------------------------------------------------------------------

/// An entry's kind: an index into the builtin kinds or the interned kernel names.
pub type KindId = u32;

const BUILTIN: [&str; 14] = [
    "?",
    "HtoD",
    "DtoH",
    "HtoD async",
    "DtoH async",
    "DtoD",
    "DtoD async",
    "peer async",
    "memset",
    "memset async",
    "graph launch",
    "graph set kernel params",
    "wait value32",
    "write value32",
];
pub const K_UNKNOWN: KindId = 0;
pub const K_HTOD: KindId = 1;
pub const K_DTOH: KindId = 2;
pub const K_HTOD_ASYNC: KindId = 3;
pub const K_DTOH_ASYNC: KindId = 4;
pub const K_DTOD: KindId = 5;
pub const K_DTOD_ASYNC: KindId = 6;
pub const K_PEER_ASYNC: KindId = 7;
pub const K_MEMSET: KindId = 8;
pub const K_MEMSET_ASYNC: KindId = 9;
pub const K_GRAPH_LAUNCH: KindId = 10;
pub const K_GRAPH_SET_PARAMS: KindId = 11;
pub const K_WAIT_VALUE32: KindId = 12;
pub const K_WRITE_VALUE32: KindId = 13;

/// Interned names beyond the builtins (kernel symbols: a few hundred per process).
const DYN_CAP: usize = 8192;
static DYN: [OnceLock<&'static str>; DYN_CAP] = [const { OnceLock::new() }; DYN_CAP];
static DYN_INDEX: Mutex<Option<HashMap<&'static str, KindId>>> = Mutex::new(None);

/// The kind id of `name` (once per distinct name; a slow path: `Module::get_function`'s cache miss). Past the table's
/// capacity the kind is recorded as `?`.
pub fn intern(name: &'static str) -> KindId {
    if let Some(i) = BUILTIN.iter().position(|b| *b == name) {
        return i as KindId;
    }
    let mut g = DYN_INDEX.lock().unwrap_or_else(|p| p.into_inner());
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(&id) = m.get(name) {
        return id;
    }
    let slot = m.len();
    if slot >= DYN_CAP {
        return K_UNKNOWN;
    }
    // Under the lock: one setter per slot, set BEFORE the id is published.
    let _ = DYN[slot].set(name);
    let id = (BUILTIN.len() + slot) as KindId;
    m.insert(name, id);
    id
}

/// As [`intern`] for a borrowed name: copied (leaked) only the first time it is seen.
pub fn intern_str(name: &str) -> KindId {
    if let Some(i) = BUILTIN.iter().position(|b| *b == name) {
        return i as KindId;
    }
    {
        let g = DYN_INDEX.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(&id) = g.as_ref().and_then(|m| m.get(name)) {
            return id;
        }
    }
    intern(Box::leak(name.to_string().into_boxed_str()))
}

/// The name of a kind id (`?` for an id that was never interned).
pub fn kind_name(id: KindId) -> &'static str {
    let i = id as usize;
    if i < BUILTIN.len() {
        BUILTIN[i]
    } else {
        DYN.get(i - BUILTIN.len()).and_then(|c| c.get().copied()).unwrap_or("?")
    }
}

// ---- clock -----------------------------------------------------------------------------------------------------

/// The ring's clock origin: an `Instant` and the raw clock ([`ticks`]) at the same moment.
struct Epoch {
    t0: Instant,
    ticks0: u64,
}

static EPOCH: LazyLock<Epoch> = LazyLock::new(|| Epoch { t0: Instant::now(), ticks0: ticks() });

/// The entry timestamp's raw clock: the TSC on x86-64 (invariant on this box's Zen parts; ~7 ns vs ~20 ns for
/// `Instant::now`), nanoseconds since first use elsewhere. Converted at dump time ([`ticks_per_ns`]).
#[inline(always)]
fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        // SAFETY: RDTSC has no preconditions on x86-64.
        unsafe { core::arch::x86_64::_rdtsc() }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        static T0: LazyLock<Instant> = LazyLock::new(Instant::now);
        T0.elapsed().as_nanos() as u64
    }
}

/// Raw clock ticks per nanosecond, measured against `Instant` since the epoch.
fn ticks_per_ns() -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let ns = EPOCH.t0.elapsed().as_nanos() as f64;
        let dt = ticks().saturating_sub(EPOCH.ticks0) as f64;
        if ns > 0.0 && dt > 0.0 { dt / ns } else { 1.0 }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        1.0
    }
}

// ---- the rings -------------------------------------------------------------------------------------------------

struct Slot {
    /// `2i + 1` while entry `i` is being written, `2i + 2` once it is complete; 0 = never written.
    seq: AtomicU64,
    /// Raw clock (`ticks`), converted to ns since the epoch by `snapshot`.
    t_raw: AtomicU64,
    kind: AtomicU32,
    loc: AtomicPtr<Location<'static>>,
    stream: AtomicUsize,
    dims: [AtomicU64; 2],
    args: [AtomicU64; RING_ARGS],
    /// `nargs | layer << 16 | lane << 40` (layer / lane biased by 1: 0 = unknown).
    meta: AtomicU64,
    tid: AtomicU64,
}

impl Slot {
    const fn new() -> Self {
        Slot {
            seq: AtomicU64::new(0),
            t_raw: AtomicU64::new(0),
            kind: AtomicU32::new(0),
            loc: AtomicPtr::new(std::ptr::null_mut()),
            stream: AtomicUsize::new(0),
            dims: [const { AtomicU64::new(0) }; 2],
            args: [const { AtomicU64::new(0) }; RING_ARGS],
            meta: AtomicU64::new(0),
            tid: AtomicU64::new(0),
        }
    }
}

struct Ring {
    head: AtomicU64,
    /// Entries a writer dropped because it could not claim its slot.
    dropped: AtomicU64,
    slots: [Slot; RING_LEN],
}

static RINGS: [Ring; RING_DEVICES] =
    [const { Ring { head: AtomicU64::new(0), dropped: AtomicU64::new(0), slots: [const { Slot::new() }; RING_LEN] } }; RING_DEVICES];

fn ring_of(device: i32) -> usize {
    if (0..RING_DEVICES as i32 - 1).contains(&device) {
        device as usize
    } else {
        RING_DEVICES - 1
    }
}

thread_local! {
    /// (layer, lane) of the work this thread is queueing; -1 = unknown.
    static CTX: Cell<(i32, i32)> = const { Cell::new((-1, -1)) };
    static TID: u64 = {
        extern "C" {
            fn gettid() -> i32;
        }
        // SAFETY: glibc's gettid has no preconditions.
        unsafe { gettid() as u64 }
    };
}

/// Restores the previous (layer, lane) context on drop.
pub struct CtxGuard {
    prev: (i32, i32),
}

impl Drop for CtxGuard {
    fn drop(&mut self) {
        CTX.with(|c| c.set(self.prev));
    }
}

/// Tag this thread's ring entries (and the MoE launch canary's messages) with `(layer, lane)` until the guard drops.
/// `lane = -1` when the work has no lane.
pub fn scope(layer: i32, lane: i32) -> CtxGuard {
    CtxGuard { prev: CTX.with(|c| c.replace((layer, lane))) }
}

/// This thread's (layer, lane) context, `-1` = unknown.
pub fn ctx() -> (i32, i32) {
    CTX.with(|c| c.get())
}

/// One argument's first 8 bytes as a word, for [`crate::launch_kernel!`]: scalars and pointers (sizes 1/2/4/8, no
/// padding) by value; any other type is recorded as `0xA16_0000_0000_0000 | size` (its bytes may hold padding).
#[inline(always)]
pub fn arg_word<T>(v: &T) -> u64 {
    let n = std::mem::size_of::<T>();
    match n {
        1 | 2 | 4 | 8 => {
            let mut w = 0u64;
            // SAFETY: `v` is a live `T` of exactly `n <= 8` bytes with no padding (a 1/2/4/8-byte kernel argument is a
            // scalar or a pointer); the copy writes the low `n` bytes of `w` (little-endian).
            unsafe { std::ptr::copy_nonoverlapping(v as *const T as *const u8, &mut w as *mut u64 as *mut u8, n) };
            w
        }
        _ => 0x0A16_0000_0000_0000 | n as u64,
    }
}

/// Record one operation on `device`'s ring.
#[inline]
pub fn record(device: i32, kind: KindId, loc: Option<&'static Location<'static>>, stream: usize, dims: [u64; 2], args: &[u64]) {
    if !enabled() {
        return;
    }
    let r = &RINGS[ring_of(device)];
    let i = r.head.fetch_add(1, Ordering::Relaxed);
    let s = &r.slots[(i % RING_LEN as u64) as usize];
    // Claim: the slot must hold a COMPLETE older entry (even, < 2i + 1); anything else means another writer owns it
    // or a newer entry already landed -- drop this one rather than share the slot.
    let prev = s.seq.load(Ordering::Relaxed);
    if prev % 2 == 1 || prev > 2 * i || s.seq.compare_exchange(prev, 2 * i + 1, Ordering::Acquire, Ordering::Relaxed).is_err() {
        r.dropped.fetch_add(1, Ordering::Relaxed);
        return;
    }
    fence(Ordering::Release);
    let (layer, lane) = ctx();
    let n = args.len().min(RING_ARGS);
    LazyLock::force(&EPOCH);
    s.t_raw.store(ticks(), Ordering::Relaxed);
    s.kind.store(kind, Ordering::Relaxed);
    s.loc.store(loc.map_or(std::ptr::null_mut(), |l| l as *const Location<'static> as *mut Location<'static>), Ordering::Relaxed);
    s.stream.store(stream, Ordering::Relaxed);
    s.dims[0].store(dims[0], Ordering::Relaxed);
    s.dims[1].store(dims[1], Ordering::Relaxed);
    for (k, w) in args[..n].iter().enumerate() {
        s.args[k].store(*w, Ordering::Relaxed);
    }
    let bias = |v: i32| (v.max(-1) + 1) as u64 & 0xff_ffff;
    s.meta.store(n as u64 | bias(layer) << 16 | bias(lane) << 40, Ordering::Relaxed);
    s.tid.store(TID.with(|t| *t), Ordering::Relaxed);
    s.seq.store(2 * i + 2, Ordering::Release);
}

/// A consistent copy of one ring entry.
#[derive(Clone, Debug)]
pub struct Entry {
    /// Global index on its ring (0 = the device's first operation).
    pub index: u64,
    pub t_ns: u64,
    pub kind: &'static str,
    pub loc: Option<&'static Location<'static>>,
    pub stream: usize,
    pub dims: [u64; 2],
    pub args: Vec<u64>,
    pub layer: i32,
    pub lane: i32,
    pub tid: u64,
}

/// What [`snapshot`] saw on one ring.
pub struct Snapshot {
    /// Entries, oldest first.
    pub entries: Vec<Entry>,
    /// Operations the ring has seen in total.
    pub head: u64,
    /// Entries skipped because they were being (re)written while read.
    pub torn: usize,
    /// Entries writers dropped (slot contention).
    pub dropped: u64,
}

/// The entries of `device`'s ring, oldest first.
pub fn snapshot(device: i32) -> Snapshot {
    let r = &RINGS[ring_of(device)];
    let (ticks0, tpn) = (EPOCH.ticks0, ticks_per_ns());
    let head = r.head.load(Ordering::Acquire);
    let first = head.saturating_sub(RING_LEN as u64);
    let mut out = Vec::with_capacity((head - first) as usize);
    let mut torn = 0usize;
    for i in first..head {
        let s = &r.slots[(i % RING_LEN as u64) as usize];
        let s1 = s.seq.load(Ordering::Acquire);
        if s1 != 2 * i + 2 {
            torn += 1;
            continue;
        }
        let t_raw = s.t_raw.load(Ordering::Relaxed);
        let kind = s.kind.load(Ordering::Relaxed);
        let loc = s.loc.load(Ordering::Relaxed);
        let stream = s.stream.load(Ordering::Relaxed);
        let dims = [s.dims[0].load(Ordering::Relaxed), s.dims[1].load(Ordering::Relaxed)];
        let meta = s.meta.load(Ordering::Relaxed);
        let n = (meta & 0xffff) as usize;
        let args: Vec<u64> = s.args[..n.min(RING_ARGS)].iter().map(|a| a.load(Ordering::Relaxed)).collect();
        let tid = s.tid.load(Ordering::Relaxed);
        fence(Ordering::Acquire);
        if s.seq.load(Ordering::Relaxed) != s1 {
            torn += 1;
            continue;
        }
        // SAFETY: `loc` is null or a `&'static Location` some writer stored (every stored value is one).
        let loc = unsafe { loc.as_ref() };
        let unbias = |v: u64| ((v & 0xff_ffff) as i32) - 1;
        let t_ns = (t_raw.saturating_sub(ticks0) as f64 / tpn) as u64;
        out.push(Entry {
            index: i,
            t_ns,
            kind: kind_name(kind),
            loc,
            stream,
            dims,
            args,
            layer: unbias(meta >> 16),
            lane: unbias(meta >> 40),
            tid,
        });
    }
    Snapshot { entries: out, head, torn, dropped: r.dropped.load(Ordering::Relaxed) }
}

fn fmt_entry(e: &Entry, now_ns: u64) -> String {
    let ago = now_ns.saturating_sub(e.t_ns) as f64 / 1e6;
    let ctx = match (e.layer, e.lane) {
        (-1, _) => String::new(),
        (l, -1) => format!(" L{l}"),
        (l, n) => format!(" L{l}/lane{n}"),
    };
    let loc = e.loc.map_or_else(String::new, |l| format!(" @ {}:{}", l.file(), l.line()));
    let args: Vec<String> = e.args.iter().map(|a| format!("{a:#x}")).collect();
    format!(
        "#{} -{ago:.3} ms tid {}{ctx} {} dims [{:#x}, {:#x}] stream {:#x} args [{}]{loc}",
        e.index,
        e.tid,
        e.kind,
        e.dims[0],
        e.dims[1],
        e.stream,
        args.join(" ")
    )
}

fn ring_name(d: i32) -> String {
    if d == RING_DEVICES as i32 - 1 { "other".to_string() } else { format!("dev{d}") }
}

/// The last `n` entries of `device`'s ring (the pointer registry's warn-mode reports). Takes no registry lock.
pub fn tail_string(device: i32, n: usize) -> String {
    let now = EPOCH.t0.elapsed().as_nanos() as u64;
    let snap = snapshot(device);
    let es = &snap.entries[snap.entries.len().saturating_sub(n)..];
    let mut s = format!("ring {} (last {} of {} ops):", ring_name(ring_of(device) as i32), es.len(), snap.head);
    for e in es {
        s.push_str("\n    ");
        s.push_str(&fmt_entry(e, now));
    }
    s
}

/// The dump text (every non-empty ring), oldest entry first per ring.
pub fn dump_string(reason: &str) -> String {
    let now = EPOCH.t0.elapsed().as_nanos() as u64;
    let mut s = format!(
        "HIP RING DUMP ({reason}): last {RING_LEN} ops per device; ptr registry: {} live allocations\n",
        crate::ptr_registry::live_count()
    );
    for d in 0..RING_DEVICES as i32 {
        let snap = snapshot(d);
        if snap.head == 0 {
            continue;
        }
        s.push_str(&format!(
            "  ring {}: {} ops total, {} shown, {} torn (being written), {} dropped (slot contention)\n",
            ring_name(d),
            snap.head,
            snap.entries.len(),
            snap.torn,
            snap.dropped
        ));
        for e in &snap.entries {
            s.push_str("    ");
            s.push_str(&fmt_entry(e, now));
            s.push('\n');
        }
    }
    s
}

/// Print the dump to stderr (unbuffered: survives an immediate `_exit`).
pub fn dump(reason: &str) {
    eprint!("{}", dump_string(reason));
}

/// Pack a launch's grid / block for [`record`]'s `dims`.
pub fn launch_dims(grid: (u32, u32, u32), block: (u32, u32, u32)) -> [u64; 2] {
    [grid.0 as u64 | (grid.1 as u64) << 32, grid.2 as u64 | (block.0 as u64 * block.1 as u64 * block.2 as u64) << 32]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tests own ring 7 (wraparound) and ring 5 (stress); nothing else records there in this process.
    #[test]
    fn wraparound_keeps_the_last_entries_in_order() {
        let dev = 7;
        let _g = scope(12, 1);
        let here = Location::caller();
        let (even, odd) = (intern("even_kernel"), intern("odd copy"));
        let total = RING_LEN as u64 * 3 + 5;
        for i in 0..total {
            record(dev, if i % 2 == 0 { even } else { odd }, Some(here), 0x1234, [i, 0], &[i, i + 1, 0xdead]);
        }
        let snap = snapshot(dev);
        assert_eq!((snap.head, snap.torn, snap.dropped), (total, 0, 0));
        assert_eq!(snap.entries.len(), RING_LEN);
        for (k, e) in snap.entries.iter().enumerate() {
            let i = total - RING_LEN as u64 + k as u64;
            assert_eq!(e.index, i);
            assert_eq!(e.dims[0], i);
            assert_eq!(e.args, vec![i, i + 1, 0xdead]);
            assert_eq!(e.kind, if i % 2 == 0 { "even_kernel" } else { "odd copy" });
            assert_eq!((e.layer, e.lane), (12, 1));
            assert_eq!(e.loc.unwrap().line(), here.line());
        }
        assert!(snap.entries.windows(2).all(|w| w[0].t_ns <= w[1].t_ns), "monotonic timestamps");
        let text = dump_string("test");
        assert!(text.contains("ring dev7") && text.contains("L12/lane1 odd copy") && text.contains("args [0x"), "{text}");
        assert!(tail_string(dev, 2).contains("last 2 of"), "{}", tail_string(dev, 2));
    }

    #[test]
    fn interning() {
        assert_eq!(kind_name(K_HTOD_ASYNC), "HtoD async");
        assert_eq!(intern("memset"), K_MEMSET, "builtins resolve to their constants");
        let a = intern("ring_test_kernel_a");
        assert_eq!(intern("ring_test_kernel_a"), a, "one id per name");
        assert_ne!(intern("ring_test_kernel_b"), a);
        assert_eq!(kind_name(a), "ring_test_kernel_a");
        assert_eq!(kind_name(u32::MAX), "?");
    }

    /// Many writers wrap the ring many times while a reader snapshots it: every entry the reader keeps is internally
    /// consistent (its kind, dims, args and location all come from ONE `record` call), whatever slot contention did.
    #[test]
    fn concurrent_writers_never_mix_entries() {
        let dev = 5;
        let kinds: Vec<KindId> = (0..8).map(|k| intern(Box::leak(format!("stress_kind_{k}").into_boxed_str()))).collect();
        let stop = std::sync::atomic::AtomicBool::new(false);
        let checked = AtomicU64::new(0);
        std::thread::scope(|sc| {
            let reader = sc.spawn(|| {
                while !stop.load(Ordering::Relaxed) {
                    for e in snapshot(dev).entries {
                        let (t, n, k) = (e.args[0], e.args[1], e.args[2]);
                        assert_eq!(e.args.len(), 4, "{e:?}");
                        assert_eq!(e.kind, kind_name(k as KindId), "{e:?}");
                        assert_eq!(e.dims, [n, t], "{e:?}");
                        assert_eq!(e.args[3], t * 1_000_003 + n, "{e:?}");
                        assert_eq!(e.stream as u64, t, "{e:?}");
                        assert_eq!((e.layer, e.lane), (t as i32, (n % 7) as i32), "{e:?}");
                        checked.fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            let writers: Vec<_> = (0..8u64)
                .map(|t| {
                    let kinds = &kinds;
                    sc.spawn(move || {
                        for n in 0..200_000u64 {
                            let k = kinds[((t + n) % 8) as usize];
                            let _g = scope(t as i32, (n % 7) as i32);
                            record(dev, k, Some(Location::caller()), t as usize, [n, t], &[t, n, k as u64, t * 1_000_003 + n]);
                        }
                    })
                })
                .collect();
            for w in writers {
                w.join().unwrap();
            }
            stop.store(true, Ordering::Relaxed);
            reader.join().unwrap();
        });
        let snap = snapshot(dev);
        assert_eq!(snap.head, 8 * 200_000);
        assert!(checked.load(Ordering::Relaxed) > 0, "the reader saw entries");
        eprintln!("ring stress: {} entries checked, {} dropped by contention", checked.load(Ordering::Relaxed), snap.dropped);
    }

    #[test]
    fn ctx_scopes_nest_and_restore() {
        assert_eq!(ctx(), (-1, -1));
        {
            let _a = scope(3, 0);
            {
                let _b = scope(4, -1);
                assert_eq!(ctx(), (4, -1));
            }
            assert_eq!(ctx(), (3, 0));
        }
        assert_eq!(ctx(), (-1, -1));
    }

    #[test]
    fn arg_words() {
        assert_eq!(arg_word(&7u32), 7);
        assert_eq!(arg_word(&-1i32), 0xffff_ffff);
        assert_eq!(arg_word(&1.0f32), 0x3f80_0000);
        assert_eq!(arg_word(&(0x7e00_1234_5678usize as *mut u8)), 0x7e00_1234_5678);
        assert_eq!(arg_word(&[0u8; 12]), 0x0A16_0000_0000_000c);
    }

    /// Cost of one entry, single thread. Run:
    /// `cargo test --release -p v4flash-hip --lib ring::tests::bench_record_ns -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_record_ns() {
        let here = Location::caller();
        let k = intern("bench");
        let iters = 2_000_000u64;
        let t = Instant::now();
        for i in 0..iters {
            record(6, k, Some(here), 1, [i, 2], &[1, 2, 3, 4, 5, 6, 7, 8]);
        }
        eprintln!("ring: record {:.1} ns/entry", t.elapsed().as_nanos() as f64 / iters as f64);
    }
}
