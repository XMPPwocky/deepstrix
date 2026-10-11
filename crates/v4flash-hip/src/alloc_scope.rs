//! Scoped allocation PLACEMENT for [`DeviceBuffer::new`](crate::DeviceBuffer::new) (the prefill VRAM loan,
//! docs/v41/PREFILL_VRAM_LOAN_DESIGN.md).
//!
//! An existing allocation routine (e.g. a scratch struct's `alloc`) runs TWICE under a thread-local scope for one device:
//! 1. [`record`]: every `DeviceBuffer::new(device, len)` returns a PHANTOM view (a fake, never-dereferenced address in a
//!    reserved range) and its byte size is recorded in call order; `copy_from_host` / `fill_zero` on a phantom are no-ops
//!    that mark the allocation WRITTEN (its contents must persist, so a planner keeps it out of memory that is lent only
//!    part of the time). No device memory is touched; a kernel launch on a phantom would fault, so a recorded routine must
//!    not launch kernels (the batch scratch allocators only copy and zero).
//! 2. [`place`]: every `DeviceBuffer::new(device, len)` returns the NEXT planned view (non-owning, `is_view`); the requested
//!    size must equal the planned size (else the call fails: the routine changed since it was recorded).
//!
//! Allocations on OTHER devices inside a scope are untouched. Scopes do not nest.
use std::cell::RefCell;

use color_eyre::eyre::{self, eyre};

use crate::sys;

/// Phantom addresses live in `[FAKE_BASE, FAKE_BASE + FAKE_SPAN)`: NON-CANONICAL on x86-64 (bits 47..63 not a sign
/// extension), so no real host or device pointer can ever be in the range (review r3 finding 4: HIP device VAs sit in
/// 0x7exx-0x7fxx, where the first choice of range was). Never dereferenced.
const FAKE_BASE: usize = 0x4000_0000_0000_0000;
const FAKE_SPAN: usize = 1 << 44;
/// Phantom allocations are spaced at this alignment (matches what kernels assume of `hipMalloc`).
const FAKE_ALIGN: usize = 256;

/// One recorded allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Recorded {
    pub bytes: usize,
    /// `copy_from_host` / `fill_zero` ran on it during the recorded routine.
    pub written: bool,
}

enum Mode {
    Record { allocs: Vec<(usize, Recorded)>, next: usize },
    Place { ptrs: Vec<(sys::hipDeviceptr_t, usize)>, next: usize },
}

struct Scope {
    device_id: i32,
    mode: Mode,
}

thread_local! {
    static SCOPE: RefCell<Option<Scope>> = const { RefCell::new(None) };
}

/// Ends the scope on drop (also on unwind).
pub struct ScopeGuard {
    _private: (),
}

impl Drop for ScopeGuard {
    fn drop(&mut self) {
        SCOPE.with(|s| *s.borrow_mut() = None);
    }
}

fn open(device_id: i32, mode: Mode) -> eyre::Result<ScopeGuard> {
    SCOPE.with(|s| {
        let mut s = s.borrow_mut();
        if s.is_some() {
            return Err(eyre!("alloc_scope: scopes do not nest"));
        }
        *s = Some(Scope { device_id, mode });
        Ok(ScopeGuard { _private: () })
    })
}

/// Start recording allocations on `device_id` (see the module doc).
pub fn record(device_id: i32) -> eyre::Result<ScopeGuard> {
    open(device_id, Mode::Record { allocs: Vec::new(), next: FAKE_BASE })
}

/// The allocations recorded so far in the open RECORD scope, in call order.
pub fn recorded() -> eyre::Result<Vec<Recorded>> {
    SCOPE.with(|s| match s.borrow().as_ref().map(|sc| &sc.mode) {
        Some(Mode::Record { allocs, .. }) => Ok(allocs.iter().map(|(_, r)| r.clone()).collect()),
        _ => Err(eyre!("alloc_scope: no record scope open")),
    })
}

/// Start placing allocations on `device_id` at `ptrs` (`(address, bytes)` in call order).
pub fn place(device_id: i32, ptrs: Vec<(sys::hipDeviceptr_t, usize)>) -> eyre::Result<ScopeGuard> {
    open(device_id, Mode::Place { ptrs, next: 0 })
}

/// How many planned views the open PLACE scope has handed out, and how many it holds.
pub fn placed() -> eyre::Result<(usize, usize)> {
    SCOPE.with(|s| match s.borrow().as_ref().map(|sc| &sc.mode) {
        Some(Mode::Place { ptrs, next }) => Ok((*next, ptrs.len())),
        _ => Err(eyre!("alloc_scope: no place scope open")),
    })
}

/// `DeviceBuffer::new` hook: `Some(address)` if a scope for `device_id` takes this allocation.
pub(crate) fn take(device_id: i32, bytes: usize) -> Option<eyre::Result<sys::hipDeviceptr_t>> {
    SCOPE.with(|s| {
        let mut s = s.borrow_mut();
        let sc = s.as_mut()?;
        if sc.device_id != device_id {
            return None;
        }
        Some(match &mut sc.mode {
            Mode::Record { allocs, next } => {
                let addr = *next;
                let span = bytes.max(1).div_ceil(FAKE_ALIGN) * FAKE_ALIGN;
                if addr + span > FAKE_BASE + FAKE_SPAN {
                    Err(eyre!("alloc_scope: phantom range exhausted"))
                } else {
                    *next += span;
                    allocs.push((addr, Recorded { bytes, written: false }));
                    Ok(addr as sys::hipDeviceptr_t)
                }
            }
            Mode::Place { ptrs, next } => match ptrs.get(*next) {
                Some(&(p, b)) if b == bytes => {
                    *next += 1;
                    Ok(p)
                }
                Some(&(_, b)) => Err(eyre!(
                    "alloc_scope: placement drift at allocation #{next}: requested {bytes} B, planned {b} B \
                     (the allocation routine changed since its plan was recorded)"
                )),
                None => Err(eyre!("alloc_scope: more allocations than planned ({})", ptrs.len())),
            },
        })
    })
}

/// Is `raw` a phantom address (any record scope, any thread)? Debug guard for the device-side paths.
pub fn is_phantom(raw: sys::hipDeviceptr_t) -> bool {
    (FAKE_BASE..FAKE_BASE + FAKE_SPAN).contains(&(raw as usize))
}

/// Write hook: `true` if `raw` is a PHANTOM of the open record scope (the caller skips the device write; the allocation
/// is marked written). Views into a phantom (carved sub-buffers) count for the allocation that contains them.
pub(crate) fn phantom_write(raw: sys::hipDeviceptr_t) -> bool {
    let a = raw as usize;
    if !(FAKE_BASE..FAKE_BASE + FAKE_SPAN).contains(&a) {
        return false;
    }
    SCOPE.with(|s| {
        let mut s = s.borrow_mut();
        if let Some(Scope { mode: Mode::Record { allocs, .. }, .. }) = s.as_mut() {
            // the allocation with the largest base <= a
            let i = allocs.partition_point(|(base, _)| *base <= a);
            if i > 0 {
                allocs[i - 1].1.written = true;
            }
        }
        true
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_place_round_trip_and_drift_detection() {
        {
            let _g = record(3).unwrap();
            let a = take(3, 1000).unwrap().unwrap();
            let b = take(3, 10).unwrap().unwrap();
            assert!(take(2, 10).is_none(), "other devices untouched");
            assert!(phantom_write(b));
            let rec = recorded().unwrap();
            assert_eq!(rec, vec![Recorded { bytes: 1000, written: false }, Recorded { bytes: 10, written: true }]);
            assert_eq!(b as usize - a as usize, 1024, "256-aligned spacing");
            assert!(record(3).is_err(), "no nesting");
        }
        assert!(recorded().is_err(), "scope closed on drop");
        {
            let _g = place(3, vec![(0x1000 as sys::hipDeviceptr_t, 1000), (0x2000 as sys::hipDeviceptr_t, 10)]).unwrap();
            assert_eq!(take(3, 1000).unwrap().unwrap() as usize, 0x1000);
            assert!(take(3, 11).unwrap().is_err(), "size drift is an error");
            assert_eq!(placed().unwrap(), (1, 2));
        }
        assert!(!phantom_write(0x1000 as sys::hipDeviceptr_t));
    }
}
