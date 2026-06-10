use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr;

use color_eyre::eyre::{self, eyre};

use crate::error::check_eyre;
use crate::stream::Stream;
use crate::sys;

/// Typed device-side buffer. Owned; freed on drop. The `device_id` field
/// is informational — HIP doesn't tag allocations with a device, so we
/// track it ourselves so peer-copy calls can pass the right source/dst
/// device IDs.
///
/// `is_view = true` means this struct refers to a sub-range of another
/// allocation (created via [`DeviceBuffer::slice_view`] /
/// [`DeviceBuffer::slice_view_mut`]); Drop skips `hipFree` for views.
pub struct DeviceBuffer<T> {
    raw: sys::hipDeviceptr_t,
    len: usize,
    device_id: i32,
    is_view: bool,
    _marker: PhantomData<T>,
}

impl<T> DeviceBuffer<T> {
    /// Allocate `len` elements of T on the current device. Caller must
    /// `Device::set_current` first, and pass that device's id so the
    /// buffer knows where it lives.
    #[track_caller]
    pub fn new(device_id: i32, len: usize) -> eyre::Result<Self> {
        let mut raw: sys::hipDeviceptr_t = ptr::null_mut();
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            eyre!("DeviceBuffer size overflow: {} * {}", len, std::mem::size_of::<T>())
        })?;
        // DEEPSTRIX_ALLOC_TRACE=1: print every allocation ≥ 8 MB with a
        // running per-device tally (audit tool; map sizes back to fields
        // by reading the alloc order in scratch/state/weights).
        static TRACE: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
            std::env::var_os("DEEPSTRIX_ALLOC_TRACE").is_some()
        });
        if *TRACE && bytes >= 8 << 20 {
            static TALLY: [std::sync::atomic::AtomicU64; 8] = [
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
                const { std::sync::atomic::AtomicU64::new(0) },
            ];
            let d = (device_id as usize).min(7);
            let tot = TALLY[d].fetch_add(bytes as u64, std::sync::atomic::Ordering::Relaxed)
                + bytes as u64;
            let loc = std::panic::Location::caller();
            eprintln!(
                "ALLOC_TRACE dev{} {:>8.1} MB  (≥8MB tally {:>8.1} MB)  {}:{}",
                device_id,
                bytes as f64 / 1e6,
                tot as f64 / 1e6,
                loc.file(),
                loc.line()
            );
        }
        check_eyre(unsafe { sys::hipMalloc(&mut raw, bytes) }, "hipMalloc")?;
        Ok(DeviceBuffer {
            raw,
            len,
            device_id,
            is_view: false,
            _marker: PhantomData,
        })
    }

    /// Return a non-owning sub-range view starting at `offset` elements
    /// in, with `len` elements. The returned `DeviceBuffer` is a view —
    /// its `Drop` does NOT free the underlying allocation. Caller is
    /// responsible for ensuring the parent allocation outlives the view.
    ///
    /// Used for per-batch operations in M50 batched prefill: kernel
    /// wrappers take `&DeviceBuffer<T>`, so a view lets us point at
    /// `parent[offset..offset+len]` without restructuring every wrapper.
    pub fn slice_view(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset.checked_add(len).map(|e| e <= self.len).unwrap_or(false),
            "slice_view out of range: offset={offset} len={len} parent_len={}",
            self.len
        );
        let byte_off = offset.checked_mul(std::mem::size_of::<T>()).expect("byte offset overflow");
        let raw = unsafe { (self.raw as *mut u8).add(byte_off) as sys::hipDeviceptr_t };
        DeviceBuffer {
            raw,
            len,
            device_id: self.device_id,
            is_view: true,
            _marker: PhantomData,
        }
    }

    /// Mutable variant of [`Self::slice_view`]. Same semantics —
    /// returned view does not own its memory.
    pub fn slice_view_mut(&mut self, offset: usize, len: usize) -> Self {
        self.slice_view(offset, len)
    }

    pub fn raw(&self) -> sys::hipDeviceptr_t {
        self.raw
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn copy_from_host(&mut self, src: &[T]) -> eyre::Result<()> {
        if src.len() != self.len {
            return Err(eyre!(
                "copy_from_host length mismatch: src={} dst={}",
                src.len(),
                self.len
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpy(
                    self.raw,
                    src.as_ptr() as *const c_void,
                    self.byte_len(),
                    sys::HIP_MEMCPY_HOST_TO_DEVICE,
                )
            },
            "hipMemcpy(HtoD)",
        )
    }

    pub fn copy_to_host(&self, dst: &mut [T]) -> eyre::Result<()> {
        if dst.len() != self.len {
            return Err(eyre!(
                "copy_to_host length mismatch: src={} dst={}",
                self.len,
                dst.len()
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpy(
                    dst.as_mut_ptr() as sys::hipDeviceptr_t,
                    self.raw,
                    self.byte_len(),
                    sys::HIP_MEMCPY_DEVICE_TO_HOST,
                )
            },
            "hipMemcpy(DtoH)",
        )
    }

    pub fn copy_from_host_async(&mut self, src: &[T], stream: &Stream) -> eyre::Result<()> {
        if src.len() != self.len {
            return Err(eyre!(
                "copy_from_host_async length mismatch: src={} dst={}",
                src.len(),
                self.len
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpyAsync(
                    self.raw,
                    src.as_ptr() as *const c_void,
                    self.byte_len(),
                    sys::HIP_MEMCPY_HOST_TO_DEVICE,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync(HtoD)",
        )
    }

    /// Async device-to-device copy on the SAME device, queued on
    /// `stream`. Returns immediately; the copy completes when prior
    /// work on `stream` completes. Used by the spec-decode snapshot
    /// path (one snapshot per layer per pair = 100s of copies; sync
    /// version would stall the pipeline).
    pub fn copy_from_buffer_async(
        &mut self,
        src: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> eyre::Result<()> {
        if src.len != self.len {
            return Err(eyre!(
                "copy_from_buffer_async length mismatch: src={} dst={}",
                src.len,
                self.len
            ));
        }
        if src.device_id != self.device_id {
            return Err(eyre!(
                "copy_from_buffer_async cross-device not supported (src dev {}, dst dev {})",
                src.device_id,
                self.device_id
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpyAsync(
                    self.raw,
                    src.raw,
                    self.byte_len(),
                    sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync(DtoD)",
        )
    }

    /// Synchronous device-to-device copy on the SAME device. Returns
    /// when the copy is complete (host-blocking). Used by the spec-
    /// decode snapshot/restore path where we want a strict happens-
    /// before relationship without managing stream events.
    pub fn copy_from_buffer(&mut self, src: &DeviceBuffer<T>) -> eyre::Result<()> {
        if src.len != self.len {
            return Err(eyre!(
                "copy_from_buffer length mismatch: src={} dst={}",
                src.len,
                self.len
            ));
        }
        if src.device_id != self.device_id {
            return Err(eyre!(
                "copy_from_buffer cross-device not supported (src dev {}, dst dev {}); use copy_to_peer_async",
                src.device_id,
                self.device_id
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpy(
                    self.raw,
                    src.raw,
                    self.byte_len(),
                    sys::HIP_MEMCPY_DEVICE_TO_DEVICE,
                )
            },
            "hipMemcpy(DtoD)",
        )
    }

    /// Direct peer-to-peer async copy. `dst` and `self` must live on
    /// different devices; the copy is queued on `stream` (which itself
    /// belongs to *some* device; per HIP docs, peer copy uses the stream's
    /// device's queue).
    pub fn copy_to_peer_async(
        &self,
        dst: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> eyre::Result<()> {
        if dst.len != self.len {
            return Err(eyre!(
                "copy_to_peer_async length mismatch: src={} dst={}",
                self.len,
                dst.len
            ));
        }
        check_eyre(
            unsafe {
                sys::hipMemcpyPeerAsync(
                    dst.raw,
                    dst.device_id,
                    self.raw,
                    self.device_id,
                    self.byte_len(),
                    stream.raw(),
                )
            },
            "hipMemcpyPeerAsync",
        )
    }

    pub fn fill_zero(&mut self) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipMemset(self.raw, 0, self.byte_len()) },
            "hipMemset",
        )
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() && !self.is_view {
            let code = unsafe { sys::hipFree(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipFree failed during drop");
            }
        }
    }
}

/// Pinned host buffer (page-locked). Required for fast async DMA.
pub struct PinnedBuffer<T> {
    raw: *mut T,
    len: usize,
}

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> eyre::Result<Self> {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            eyre!("PinnedBuffer size overflow: {} * {}", len, std::mem::size_of::<T>())
        })?;
        let mut raw: *mut c_void = ptr::null_mut();
        check_eyre(
            unsafe { sys::hipHostMalloc(&mut raw, bytes, 0) },
            "hipHostMalloc",
        )?;
        // hipHostMalloc does not zero, so the bytes are uninitialized.
        // Zero them once here (allocations are rare, off the hot path) so
        // `as_slice`/`as_mut_slice` can hand out `&[T]` over initialized
        // memory. PinnedBuffer is only used with plain-data numeric types,
        // for which the all-zero bit pattern is valid.
        unsafe { ptr::write_bytes(raw as *mut u8, 0, bytes) };
        Ok(PinnedBuffer {
            raw: raw as *mut T,
            len,
        })
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `raw` points at `len * size_of::<T>()` bytes from
        // hipHostMalloc, zero-initialized in `new`, properly aligned, and
        // owned for `&self`'s lifetime. Sound for the plain-data types
        // PinnedBuffer is used with (all-zero is a valid bit pattern).
        unsafe { std::slice::from_raw_parts(self.raw, self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: same invariants as `as_slice`; `&mut self` guarantees
        // exclusive access for the returned slice's lifetime.
        unsafe { std::slice::from_raw_parts_mut(self.raw, self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let code = unsafe { sys::hipHostFree(self.raw as *mut c_void) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipHostFree failed during drop");
            }
        }
    }
}
