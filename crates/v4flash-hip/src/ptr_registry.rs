//! LIVE ALLOCATION RANGE REGISTRY (HIP 700 instrumentation, owner-approved 2026-10-19).
//!
//! Two production faults were a GARBAGE POINTER reaching the GPU: 10-13 the dGPU's SDMA wrote to a host-heap VA
//! (`MAPPING_ERROR`), 10-18 an iGPU MoE kernel took an aperture violation. Every hub allocation goes through
//! [`DeviceBuffer::new`](crate::DeviceBuffer::new) / [`PinnedBuffer::new_with_flags`](crate::PinnedBuffer::new_with_flags)
//! (the prefill VRAM loan and the embed loan only VIEW those allocations), so each owning allocation registers its
//! `[base, end)` here and deregisters in `Drop`. Every copy / memset in `buffer.rs` checks its device side (and an async
//! host copy's pinned side) against the registry BEFORE calling HIP: a pointer outside every live allocation -- a garbage
//! pointer, or a view that outlived its parent -- fails with an error naming the address, the nearest live ranges and
//! the caller's `#[track_caller]` location, instead of reaching the DMA engine.
//!
//! Views (`slice_view` / `view_as` / `from_raw_parts`) do not register; they must lie inside a live range.
//!
//! `V41_PTR_REGISTRY=0` (also `off|false|no`) disables registration AND the checks; read once (STATIC), default ON.
//! Cost: one `RwLock` read + one `BTreeMap` predecessor lookup per checked side (see `bench_lookup_ns`).
use std::collections::BTreeMap;
use std::fmt;
use std::panic::Location;
use std::sync::{LazyLock, RwLock};

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

/// `V41_PTR_REGISTRY` (STATIC, default ON): `0|off|false|no` disables registration and every check.
pub fn enabled() -> bool {
    static ON: LazyLock<bool> = LazyLock::new(|| {
        !matches!(
            std::env::var("V41_PTR_REGISTRY").map(|v| v.trim().to_ascii_lowercase()).as_deref(),
            Ok("0") | Ok("off") | Ok("false") | Ok("no")
        )
    });
    *ON
}

static REG: RwLock<BTreeMap<usize, Range>> = RwLock::new(BTreeMap::new());

fn read() -> std::sync::RwLockReadGuard<'static, BTreeMap<usize, Range>> {
    REG.read().unwrap_or_else(|p| p.into_inner())
}

fn write() -> std::sync::RwLockWriteGuard<'static, BTreeMap<usize, Range>> {
    REG.write().unwrap_or_else(|p| p.into_inner())
}

/// Register an owning allocation. An allocation that overlaps a LIVE range means the bookkeeping is broken (a free
/// that skipped `Drop`, or HIP handed out live memory): an error naming both ranges, never a silent overwrite.
/// Zero-byte allocations (null or not) are not registered: nothing may be read or written through them.
pub(crate) fn register(base: usize, bytes: usize, device: i32, kind: Kind, loc: &'static Location<'static>) -> eyre::Result<()> {
    if !enabled() || bytes == 0 {
        return Ok(());
    }
    let r = Range { base, end: base.checked_add(bytes).ok_or_else(|| eyre!("ptr registry: [{base:#x} + {bytes}) overflows"))?, device, kind, loc };
    let mut m = write();
    if let Some(o) = overlapping(&m, r.base, r.end) {
        return Err(eyre!("ptr registry: new allocation {r} overlaps the LIVE allocation {o} (a free skipped Drop, or the registry is stale)"));
    }
    m.insert(base, r);
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

/// A DEVICE-side operand of a copy / memset / kernel: `[addr, addr + bytes)` must lie inside ONE live allocation that is
/// device memory of `device`, or pinned host memory (device-accessible from every device). `bytes == 0` touches nothing.
pub fn check_device(what: &str, addr: usize, bytes: usize, device: i32, loc: &'static Location<'static>) -> eyre::Result<()> {
    if !enabled() || bytes == 0 {
        return Ok(());
    }
    let m = read();
    match m.range(..=addr).next_back().map(|(_, r)| r) {
        Some(r) if r.contains(addr, bytes) => {
            if r.kind == Kind::Device && r.device != device {
                Err(err(what, addr, bytes, format!("lies in a dev{} allocation but the operand claims dev{device}", r.device), &m, loc))
            } else {
                Ok(())
            }
        }
        Some(r) if addr < r.end => Err(err(
            what,
            addr,
            bytes,
            format!("runs {} B past the end of its live allocation", addr + bytes - r.end),
            &m,
            loc,
        )),
        _ => Err(err(what, addr, bytes, "is not inside any live allocation (garbage pointer, or a view that outlived its parent)".into(), &m, loc)),
    }
}

/// The HOST side of an async copy that must be pinned (`copy_to_pinned_async`): inside one live PINNED range.
pub fn check_pinned(what: &str, addr: usize, bytes: usize, loc: &'static Location<'static>) -> eyre::Result<()> {
    if !enabled() || bytes == 0 {
        return Ok(());
    }
    let m = read();
    match m.range(..=addr).next_back().map(|(_, r)| r) {
        Some(r) if r.kind == Kind::Pinned && r.contains(addr, bytes) => Ok(()),
        Some(r) if r.kind == Kind::Pinned && addr < r.end => Err(err(
            what,
            addr,
            bytes,
            format!("runs {} B past the end of its pinned buffer", addr + bytes - r.end),
            &m,
            loc,
        )),
        _ => Err(err(what, addr, bytes, "is not inside any live pinned buffer".into(), &m, loc)),
    }
}

/// A HOST slice handed to a copy (sync, or async from pageable memory): pageable memory is allowed (a Rust slice is
/// valid by construction), but a slice that STARTS inside a registered allocation must END inside it -- the
/// `from_raw_parts` over a pinned pack with a wrong length (forward_prefill's zero-copy pull) is exactly that case.
/// (A host slice of DEVICE memory is legal on the APU: `DeviceBuffer::as_host_slice_mut`.)
pub fn check_host(what: &str, addr: usize, bytes: usize, loc: &'static Location<'static>) -> eyre::Result<()> {
    if !enabled() || bytes == 0 {
        return Ok(());
    }
    let m = read();
    match m.range(..=addr).next_back().map(|(_, r)| r) {
        Some(r) if addr < r.end && !r.contains(addr, bytes) => Err(err(
            what,
            addr,
            bytes,
            format!("starts in a live allocation and runs {} B past its end", addr + bytes - r.end),
            &m,
            loc,
        )),
        _ => Ok(()),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Fake address space for the tests: non-canonical (bit 62), so it can never collide with a real allocation of a
    /// test that runs in the same process; each test takes its own 1 TiB slice.
    pub(crate) fn fake_base(slot: usize) -> usize {
        0x4800_0000_0000_0000 + (slot << 40)
    }

    #[track_caller]
    pub(crate) fn reg(base: usize, bytes: usize, device: i32, kind: Kind) {
        register(base, bytes, device, kind, Location::caller()).unwrap();
    }

    #[test]
    fn insert_lookup_remove_adjacent_and_overlap() {
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
        let b = fake_base(2);
        let here = Location::caller();
        reg(b, 0x100, -1, Kind::Pinned);
        reg(b + 0x1000, 0x100, 0, Kind::Device);
        assert!(check_pinned("t", b + 0x10, 0x10, here).is_ok());
        assert!(check_pinned("t", b + 0x80, 0x100, here).unwrap_err().to_string().contains("past the end of its pinned buffer"));
        assert!(check_pinned("t", b + 0x1000, 4, here).unwrap_err().to_string().contains("not inside any live pinned buffer"));
        assert!(check_host("t", b + 0x10, 0xf0, here).is_ok());
        assert!(check_host("t", b + 0xf0, 0x20, here).unwrap_err().to_string().contains("starts in a live allocation and runs 16 B past"));
        assert!(check_host("t", b + 0x1000, 4, here).is_ok(), "a host slice of APU device memory");
        assert!(check_host("t", b + 0x10f0, 0x20, here).unwrap_err().to_string().contains("runs 16 B past"));
        assert!(check_host("t", b + 0x5000, 1 << 20, here).is_ok(), "pageable memory is not the registry's business");
        deregister(b, 0x100);
        deregister(b + 0x1000, 0x100);
    }

    /// Lookup cost at a production-like registry size (~5000 live allocations), single thread. Run:
    /// `cargo test --release -p v4flash-hip bench_lookup_ns -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn bench_lookup_ns() {
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
