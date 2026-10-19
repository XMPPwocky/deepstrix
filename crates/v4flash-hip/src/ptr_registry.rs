//! LIVE ALLOCATION RANGE REGISTRY (HIP 700 instrumentation, owner-approved 2026-10-19).
//!
//! Two production faults were a GARBAGE POINTER reaching the GPU: 10-13 the dGPU's SDMA wrote to a host-heap VA
//! (`MAPPING_ERROR`), 10-18 an iGPU MoE kernel took an aperture violation. Every hub allocation goes through
//! [`DeviceBuffer::new`](crate::DeviceBuffer::new) / [`PinnedBuffer::new_with_flags`](crate::PinnedBuffer::new_with_flags)
//! (the prefill VRAM loan and the embed loan only VIEW those allocations), so each owning allocation registers its
//! `[base, end)` here and deregisters in `Drop`. Every copy / memset in `buffer.rs` checks its device side (and an async
//! host copy's pinned side) against the registry BEFORE calling HIP: a pointer outside every live allocation -- a garbage
//! pointer, or a view that outlived its parent -- is a MISS naming the address, the nearest live ranges and the caller's
//! `#[track_caller]` location.
//!
//! Views (`slice_view` / `view_as` / `from_raw_parts`) do not register; they must lie inside a live range.
//!
//! MODE ([`mode`]), `V41_PTR_REGISTRY = off | warn | enforce`, default **warn** (review r1: soak before enforcing):
//! * `warn`: a miss logs a rate-limited (per call site) WARN with the full detail and the device's last ring entries,
//!   then the operation proceeds EXACTLY as without the registry. Nothing on the hot path changes behaviour.
//! * `enforce`: a miss is an `Err` before the HIP call (and the MoE canary refuses the launch).
//! * `off`: no checks.
//! Registration is decided ONCE from the env at the first allocation (on unless the env says `off`); the mode itself
//! is LIVE ([`set_mode`], pushed by the hub's / box 2's `V41_PTR_REGISTRY` knob hook), so warn -> enforce needs no
//! restart. A process that started `off` stays off (nothing registered).
//! [`advisory`] scopes (the dGPU loan return) never refuse: refusing there can only end in an abort.
//!
//! Cost: one `RwLock` read + one `BTreeMap` predecessor lookup per checked side (see `bench_lookup_ns`).
use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::panic::Location;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{LazyLock, Mutex, RwLock};
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre};

/// What owns a registered range.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `hipMalloc` (`DeviceBuffer::new`): device memory of `Range::device`.
    Device,
    /// `hipHostMalloc` (`PinnedBuffer`): page-locked host memory, device-accessible from every device.
    Pinned,
}

/// One live allocation.
#[derive(Clone, Copy, Debug)]
pub struct Range {
    pub base: usize,
    /// One past the last byte.
    pub end: usize,
    /// The allocating device (`-1` for pinned host memory).
    pub device: i32,
    pub kind: Kind,
    /// Where it was allocated.
    pub loc: &'static Location<'static>,
}

impl Range {
    fn contains(&self, addr: usize, bytes: usize) -> bool {
        addr >= self.base && addr.checked_add(bytes).is_some_and(|e| e <= self.end)
    }
}

impl fmt::Display for Range {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let who = match self.kind {
            Kind::Device => format!("dev{} device", self.device),
            Kind::Pinned => "pinned host".to_string(),
        };
        write!(
            f,
            "[{:#x}, {:#x}) {} B {who} alloc at {}:{}",
            self.base,
            self.end,
            self.end - self.base,
            self.loc.file(),
            self.loc.line()
        )
    }
}

// ---- mode ------------------------------------------------------------------------------------------------------

/// What a miss does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Off,
    Warn,
    Enforce,
}

impl Mode {
    /// `V41_PTR_REGISTRY`'s spelling (the knob table's `Choice` accepts the same words): `off|0|false|no`,
    /// `enforce|1|on|true|yes`, anything else (unset included) = `warn`.
    pub fn parse(s: Option<&str>) -> Mode {
        match s.map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Some("off" | "0" | "false" | "no") => Mode::Off,
            Some("enforce" | "1" | "on" | "true" | "yes") => Mode::Enforce,
            _ => Mode::Warn,
        }
    }
}

const MODE_UNSET: u8 = 255;
static MODE: AtomicU8 = AtomicU8::new(MODE_UNSET);

fn env_mode() -> Mode {
    Mode::parse(std::env::var("V41_PTR_REGISTRY").ok().as_deref())
}

/// Registration on (STATIC: the env at first use is not `off`).
pub fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| env_mode() != Mode::Off);
    *ON
}

/// The current mode (LIVE).
pub fn mode() -> Mode {
    match MODE.load(Ordering::Relaxed) {
        0 => Mode::Off,
        1 => Mode::Warn,
        2 => Mode::Enforce,
        _ => {
            let m = if enabled() { env_mode() } else { Mode::Off };
            let _ = MODE.compare_exchange(MODE_UNSET, m as u8, Ordering::Relaxed, Ordering::Relaxed);
            mode()
        }
    }
}

/// Switch the mode live. A process whose registration is off (started with `off`) cannot turn checks on: nothing is
/// registered, so every operand would miss. Returns the mode now in force.
pub fn set_mode(m: Mode) -> Mode {
    if m != Mode::Off && !enabled() {
        tracing::error!(wanted = ?m, "V41_PTR_REGISTRY: registration was off at startup; checks stay off until a restart");
        MODE.store(Mode::Off as u8, Ordering::Relaxed);
        return Mode::Off;
    }
    let old = MODE.swap(m as u8, Ordering::Relaxed);
    if old != m as u8 {
        tracing::warn!(mode = ?m, "V41_PTR_REGISTRY: mode changed");
    }
    m
}

thread_local! {
    static ADVISORY: Cell<Option<&'static str>> = const { Cell::new(None) };
}

/// Restores the previous advisory state on drop.
pub struct AdvisoryGuard {
    prev: Option<&'static str>,
}

impl Drop for AdvisoryGuard {
    fn drop(&mut self) {
        ADVISORY.with(|c| c.set(self.prev));
    }
}

/// Until the guard drops, this thread's misses never refuse (even in `enforce`): each is logged as a named ERROR
/// (`why` says why refusing is worse) and the operation proceeds.
pub fn advisory(why: &'static str) -> AdvisoryGuard {
    AdvisoryGuard { prev: ADVISORY.with(|c| c.replace(Some(why))) }
}

/// Misses seen (all modes but off), for logs and tests.
static MISSES: AtomicU64 = AtomicU64::new(0);

pub fn misses() -> u64 {
    MISSES.load(Ordering::Relaxed)
}

/// Per call site: (misses since the last log line, when it was logged).
static REPORTED: Mutex<Option<HashMap<usize, (u64, Instant)>>> = Mutex::new(None);
const REPORT_EVERY: Duration = Duration::from_secs(60);

/// A miss in warn mode (or an advisory scope): one log line per call site per minute, with the count suppressed in
/// between and `device`'s last ring entries. Never fails.
pub fn report(msg: &str, device: i32, loc: &'static Location<'static>) {
    MISSES.fetch_add(1, Ordering::Relaxed);
    let key = loc as *const Location<'static> as usize;
    let suppressed = {
        let mut g = REPORTED.lock().unwrap_or_else(|p| p.into_inner());
        let m = g.get_or_insert_with(HashMap::new);
        let now = Instant::now();
        match m.get_mut(&key) {
            Some((n, last)) if now.duration_since(*last) < REPORT_EVERY => {
                *n += 1;
                return;
            }
            Some((n, last)) => {
                let s = *n;
                *n = 0;
                *last = now;
                s
            }
            None => {
                m.insert(key, (0, now));
                0
            }
        }
    };
    let tail = crate::ring::tail_string(device, 8);
    match ADVISORY.with(|c| c.get()) {
        Some(why) => tracing::error!(
            suppressed,
            advisory = why,
            "PTR-REGISTRY MISS (advisory, NOT refused): {msg}\n  {tail}"
        ),
        None => tracing::warn!(suppressed, mode = ?mode(), "PTR-REGISTRY MISS (warn, NOT refused): {msg}\n  {tail}"),
    }
}

/// A miss: `Err` in enforce mode outside an advisory scope, else [`report`] + `Ok`.
pub fn verdict(e: eyre::Report, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    if mode() == Mode::Enforce && ADVISORY.with(|c| c.get()).is_none() {
        MISSES.fetch_add(1, Ordering::Relaxed);
        return Err(e);
    }
    report(&e.to_string(), device, loc);
    Ok(())
}

// ---- peers -----------------------------------------------------------------------------------------------------

/// Bit `a * 8 + b`: device `a` may access device `b`'s memory (`Device::enable_peer_access` succeeded or was already
/// enabled). Devices 0..8.
static PEERS: AtomicU64 = AtomicU64::new(0);

pub(crate) fn note_peer(accessor: i32, owner: i32) {
    if (0..8).contains(&accessor) && (0..8).contains(&owner) {
        PEERS.fetch_or(1 << (accessor * 8 + owner), Ordering::Relaxed);
    }
}

/// May work queued on `accessor` touch `owner`'s device memory (same device, or peer access enabled)?
pub fn peer_enabled(accessor: i32, owner: i32) -> bool {
    accessor == owner
        || ((0..8).contains(&accessor) && (0..8).contains(&owner) && PEERS.load(Ordering::Relaxed) & (1 << (accessor * 8 + owner)) != 0)
}

// ---- the registry ----------------------------------------------------------------------------------------------

static REG: RwLock<BTreeMap<usize, Range>> = RwLock::new(BTreeMap::new());

fn read() -> std::sync::RwLockReadGuard<'static, BTreeMap<usize, Range>> {
    REG.read().unwrap_or_else(|p| p.into_inner())
}

fn write() -> std::sync::RwLockWriteGuard<'static, BTreeMap<usize, Range>> {
    REG.write().unwrap_or_else(|p| p.into_inner())
}

/// Register an owning allocation. An allocation that overlaps a LIVE range means the bookkeeping is broken (a free
/// that skipped `Drop`, or HIP handed out live memory): `enforce` fails the allocation naming both ranges; `warn` /
/// `off` report it and replace the stale entries (the allocation itself never fails). Zero-byte allocations are not
/// registered: nothing may be read or written through them.
pub(crate) fn register(base: usize, bytes: usize, device: i32, kind: Kind, loc: &'static Location<'static>) -> eyre::Result<()> {
    if !enabled() || bytes == 0 {
        return Ok(());
    }
    let r = Range { base, end: base.checked_add(bytes).ok_or_else(|| eyre!("ptr registry: [{base:#x} + {bytes}) overflows"))?, device, kind, loc };
    let stale = {
        let mut m = write();
        let mut stale = Vec::new();
        while let Some(o) = overlapping(&m, r.base, r.end) {
            if mode() == Mode::Enforce {
                return Err(eyre!("ptr registry: new allocation {r} overlaps the LIVE allocation {o} (a free skipped Drop, or the registry is stale)"));
            }
            m.remove(&o.base);
            stale.push(o);
        }
        m.insert(base, r);
        stale
    };
    for o in stale {
        report(&format!("ptr registry: new allocation {r} overlapped the LIVE allocation {o}; the stale entry was replaced"), device, loc);
    }
    Ok(())
}

/// Deregister an owning allocation (its `Drop`). A missing entry is logged loudly (Drop cannot fail).
pub(crate) fn deregister(base: usize, bytes: usize) {
    if !enabled() || bytes == 0 {
        return;
    }
    if write().remove(&base).is_none() {
        tracing::error!(base = format_args!("{base:#x}"), bytes, "ptr registry: freeing an allocation that was never registered (registry out of sync)");
    }
}

fn overlapping(m: &BTreeMap<usize, Range>, base: usize, end: usize) -> Option<Range> {
    if let Some((_, r)) = m.range(..end).next_back() {
        if r.end > base {
            return Some(*r);
        }
    }
    None
}

/// The live range containing byte `addr`, if any.
pub fn lookup(addr: usize) -> Option<Range> {
    let m = read();
    m.range(..=addr).next_back().map(|(_, r)| *r).filter(|r| addr < r.end)
}

/// Live ranges registered (tests, the ring dump's header).
pub fn live_count() -> usize {
    read().len()
}

/// The live ranges just below and just above `addr` (error messages).
fn neighbours(m: &BTreeMap<usize, Range>, addr: usize) -> String {
    let below = m.range(..=addr).next_back().map(|(_, r)| format!("below {r}"));
    let above = m.range(addr.saturating_add(1)..).next().map(|(_, r)| format!("above {r}"));
    match (below, above) {
        (None, None) => "no live allocations".to_string(),
        (Some(b), None) => b,
        (None, Some(a)) => a,
        (Some(b), Some(a)) => format!("{b}; {a}"),
    }
}

fn err(what: &str, addr: usize, bytes: usize, problem: String, m: &BTreeMap<usize, Range>, loc: &'static Location<'static>) -> eyre::Report {
    eyre!(
        "ptr registry: {what}: [{addr:#x}, +{bytes} B) {problem}; nearest: {}; call site {}:{}",
        neighbours(m, addr),
        loc.file(),
        loc.line()
    )
}

/// The miss (if any) of a DEVICE-side operand: `[addr, addr + bytes)` inside ONE live allocation that is device
/// memory of `device` (the operand's own device id), or pinned host memory.
fn device_miss(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> Option<eyre::Report> {
    let m = read();
    match m.range(..=addr).next_back().map(|(_, r)| r) {
        Some(r) if r.contains(addr, bytes) => {
            if r.kind == Kind::Device && r.device != device {
                Some(err(what, addr, bytes, format!("lies in a dev{} allocation but the operand claims dev{device}", r.device), &m, loc))
            } else {
                None
            }
        }
        Some(r) if addr < r.end => Some(err(
            what,
            addr,
            bytes,
            format!("runs {} B past the end of its live allocation", addr + bytes - r.end),
            &m,
            loc,
        )),
        _ => Some(err(what, addr, bytes, "is not inside any live allocation (garbage pointer, or a view that outlived its parent)".into(), &m, loc)),
    }
}

/// Is a DEVICE-side operand of a copy / memset / kernel live (see [`device_miss`])? `bytes == 0` touches nothing.
/// The miss follows the [`mode`] ([`verdict`]).
pub fn check_device(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    if bytes == 0 || mode() == Mode::Off {
        return Ok(());
    }
    match device_miss(what, addr, bytes, device, loc) {
        None => Ok(()),
        Some(e) => verdict(e, device, loc),
    }
}

/// [`check_device`] as a plain answer, independent of the mode (the MoE canary decides itself).
pub fn device_operand_ok(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> Result<(), String> {
    if bytes == 0 {
        return Ok(());
    }
    device_miss(what, addr, bytes, device, loc).map_or(Ok(()), |e| Err(e.to_string()))
}

/// The HOST side of an async copy that must be pinned (`copy_to_pinned_async`): inside one live PINNED range.
/// `device` = the copy's device (its ring goes into a warn-mode report).
pub fn check_pinned(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    if bytes == 0 || mode() == Mode::Off {
        return Ok(());
    }
    let e = {
        let m = read();
        match m.range(..=addr).next_back().map(|(_, r)| r) {
            Some(r) if r.kind == Kind::Pinned && r.contains(addr, bytes) => return Ok(()),
            Some(r) if r.kind == Kind::Pinned && addr < r.end => err(
                what,
                addr,
                bytes,
                format!("runs {} B past the end of its pinned buffer", addr + bytes - r.end),
                &m,
                loc,
            ),
            _ => err(what, addr, bytes, "is not inside any live pinned buffer".into(), &m, loc),
        }
    };
    verdict(e, device, loc)
}

/// A HOST slice handed to a copy (sync, or async from pageable memory): pageable memory is allowed (a Rust slice is
/// valid by construction), but a slice that STARTS inside a registered allocation must END inside it -- the
/// `from_raw_parts` over a pinned pack with a wrong length (forward_prefill's zero-copy pull) is exactly that case.
/// (A host slice of DEVICE memory is legal on the APU: `DeviceBuffer::as_host_slice_mut`.)
pub fn check_host(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    if bytes == 0 || mode() == Mode::Off {
        return Ok(());
    }
    let e = {
        let m = read();
        match m.range(..=addr).next_back().map(|(_, r)| r) {
            Some(r) if addr < r.end && !r.contains(addr, bytes) => err(
                what,
                addr,
                bytes,
                format!("starts in a live allocation and runs {} B past its end", addr + bytes - r.end),
                &m,
                loc,
            ),
            _ => return Ok(()),
        }
    };
    verdict(e, device, loc)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Fake address space for the tests: non-canonical (bit 62), so it can never collide with a real allocation of a
    /// test that runs in the same process; each test takes its own 1 TiB slice.
    pub(crate) fn fake_base(slot: usize) -> usize {
        0x4800_0000_0000_0000 + (slot << 40)
    }

    /// The lib tests run `enforce` (the env default is `warn`; `tests/ptr_registry_warn.rs` covers warn).
    pub(crate) fn enforce() {
        assert!(enabled(), "these tests need registration (V41_PTR_REGISTRY not off)");
        set_mode(Mode::Enforce);
    }

    #[track_caller]
    pub(crate) fn reg(base: usize, bytes: usize, device: i32, kind: Kind) {
        register(base, bytes, device, kind, Location::caller()).unwrap();
    }

    #[test]
    fn modes_parse() {
        assert_eq!(Mode::parse(None), Mode::Warn);
        assert_eq!(Mode::parse(Some("warn")), Mode::Warn);
        assert_eq!(Mode::parse(Some(" OFF ")), Mode::Off);
        assert_eq!(Mode::parse(Some("0")), Mode::Off);
        assert_eq!(Mode::parse(Some("enforce")), Mode::Enforce);
        assert_eq!(Mode::parse(Some("1")), Mode::Enforce);
        assert_eq!(Mode::parse(Some("garbage")), Mode::Warn);
    }

    #[test]
    fn peers() {
        assert!(peer_enabled(3, 3));
        assert!(!peer_enabled(6, 7));
        note_peer(6, 7);
        assert!(peer_enabled(6, 7) && !peer_enabled(7, 6), "directional");
        assert!(!peer_enabled(-1, 7) && !peer_enabled(6, 9));
    }

    #[test]
    fn insert_lookup_remove_adjacent_and_overlap() {
        enforce();
        let b = fake_base(1);
        let here = Location::caller();
        reg(b, 0x1000, 1, Kind::Device);
        reg(b + 0x1000, 0x1000, 1, Kind::Device); // adjacent: fine
        reg(b + 0x3000, 0x100, -1, Kind::Pinned);
        assert_eq!(lookup(b).unwrap().base, b);
        assert_eq!(lookup(b + 0xfff).unwrap().base, b, "last byte of the first range");
        assert_eq!(lookup(b + 0x1000).unwrap().base, b + 0x1000, "first byte of the adjacent range");
        assert!(lookup(b + 0x2000).is_none(), "the gap");
        assert!(lookup(b + 0x3100).is_none(), "one past the pinned end");
        // overlaps: inside, straddling the start, straddling the end, exactly equal
        for (base, bytes) in [(b + 0x10, 0x10), (b - 0x10, 0x20), (b + 0x1ff0, 0x20), (b + 0x3000, 0x100), (b + 0x2fff, 2)] {
            let e = register(base, bytes, 1, Kind::Device, here).unwrap_err().to_string();
            assert!(e.contains("overlaps the LIVE allocation"), "{e}");
        }
        register(b + 0x2000, 0x1000, 1, Kind::Device, here).unwrap(); // fills the gap exactly
        // views: anything inside a range passes, a straddle of two adjacent ranges does not
        assert!(check_device("t", b + 0x800, 0x800, 1, here).is_ok());
        let e = check_device("t", b + 0x800, 0x1000, 1, here).unwrap_err().to_string();
        assert!(e.contains("runs 2048 B past the end"), "{e}");
        let e = check_device("t", b + 0x10, 4, 0, here).unwrap_err().to_string();
        assert!(e.contains("dev1 allocation but the operand claims dev0"), "{e}");
        assert!(check_device("t", b + 0x3000, 0x100, 0, here).is_ok(), "pinned memory is device-accessible from any device");
        let e = check_device("t", b + 0x5000, 4, 1, here).unwrap_err().to_string();
        assert!(e.contains("not inside any live allocation") && e.contains("below [") && e.contains(&format!("{}:", file!())), "{e}");
        assert!(device_operand_ok("t", b + 0x5000, 4, 1, here).unwrap_err().contains("not inside any live allocation"));
        // advisory: even in enforce, a miss is reported and passes
        {
            let _a = advisory("test: refusing would be worse");
            let before = misses();
            assert!(check_device("t", b + 0x5000, 4, 1, here).is_ok());
            assert!(misses() > before);
        }
        assert!(check_device("t", b + 0x5000, 4, 1, here).is_err(), "the advisory scope ended");
        // remove: the range is gone, its neighbours are not
        deregister(b + 0x1000, 0x1000);
        assert!(lookup(b + 0x1000).is_none());
        assert!(lookup(b).is_some() && lookup(b + 0x2000).is_some());
        assert!(check_device("t", b + 0x1000, 4, 1, here).is_err(), "a view of a freed allocation (use after free)");
        deregister(b, 0x1000);
        deregister(b + 0x2000, 0x1000);
        deregister(b + 0x3000, 0x100);
        assert!(lookup(b).is_none() && lookup(b + 0x3000).is_none());
    }

    #[test]
    fn host_side_checks() {
        enforce();
        let b = fake_base(2);
        let here = Location::caller();
        reg(b, 0x100, -1, Kind::Pinned);
        reg(b + 0x1000, 0x100, 0, Kind::Device);
        assert!(check_pinned("t", b + 0x10, 0x10, 0, here).is_ok());
        assert!(check_pinned("t", b + 0x80, 0x100, 0, here).unwrap_err().to_string().contains("past the end of its pinned buffer"));
        assert!(check_pinned("t", b + 0x1000, 4, 0, here).unwrap_err().to_string().contains("not inside any live pinned buffer"));
        assert!(check_host("t", b + 0x10, 0xf0, 0, here).is_ok());
        assert!(check_host("t", b + 0xf0, 0x20, 0, here).unwrap_err().to_string().contains("starts in a live allocation and runs 16 B past"));
        assert!(check_host("t", b + 0x1000, 4, 0, here).is_ok(), "a host slice of APU device memory");
        assert!(check_host("t", b + 0x10f0, 0x20, 0, here).unwrap_err().to_string().contains("runs 16 B past"));
        assert!(check_host("t", b + 0x5000, 1 << 20, 0, here).is_ok(), "pageable memory is not the registry's business");
        deregister(b, 0x100);
        deregister(b + 0x1000, 0x100);
    }

    /// Lookup cost at a production-like registry size (~5000 live allocations), single thread. Run:
    /// `cargo test --release -p v4flash-hip --lib ptr_registry::tests::bench_lookup_ns -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_lookup_ns() {
        enforce();
        let b = fake_base(3);
        let n = 5000usize;
        for i in 0..n {
            reg(b + i * 0x10_0000, 0x8_0000, (i % 2) as i32, Kind::Device);
        }
        let here = Location::caller();
        let iters = 2_000_000usize;
        let t = std::time::Instant::now();
        let mut ok = 0usize;
        for i in 0..iters {
            let j = (i * 7919) % n;
            ok += check_device("bench", b + j * 0x10_0000 + 64, 256, (j % 2) as i32, here).is_ok() as usize;
        }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        assert_eq!(ok, iters);
        eprintln!("ptr registry: check_device {ns:.1} ns/lookup over {n} live ranges");
        for i in 0..n {
            deregister(b + i * 0x10_0000, 0x8_0000);
        }
    }
}
