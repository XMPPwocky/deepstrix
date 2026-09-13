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
    /// Allocate `len` elements of T on `device_id`.
    ///
    /// `device_id` is AUTHORITATIVE: this enters a `DeviceGuard` so the allocation physically
    /// lands on that device regardless of ambient `hipSetDevice` state, and restores the previous
    /// current device afterwards. The recorded id therefore always matches reality, which the
    /// copy paths rely on (they compare device ids to route local vs peer transfers).
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
        // `device_id` is authoritative — see DeviceGuard. hipMalloc binds to the CURRENT
        // device, so without this the buffer silently lands wherever ambient state points.
        let _guard = crate::device::Device::scoped(device_id)?;
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
    /// Non-owning `DeviceBuffer` view over an existing device-accessible
    /// pointer — e.g. the pointer from `hipHostMalloc`, which on an APU is
    /// system RAM mapped into the GPU's address space.
    ///
    /// # Safety
    /// `raw` must be device-accessible from `device_id` and stay alive for
    /// the view's lifetime. The view never frees.
    pub unsafe fn from_raw_parts(raw: sys::hipDeviceptr_t, len: usize, device_id: i32) -> Self {
        Self { raw, len, device_id, is_view: true, _marker: PhantomData }
    }

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

    /// Reinterpret a byte sub-range of this buffer as a typed non-owning
    /// view (`DeviceBuffer<U>` with `is_view = true`). Used to pack
    /// heterogeneously-typed per-layer transfer payloads into one
    /// allocation so a single peer copy moves them together.
    ///
    /// # Safety
    /// Caller must ensure `byte_offset` is aligned for `U` and that the
    /// producers/consumers agree on the layout. The parent allocation must
    /// outlive the view.
    pub unsafe fn view_as<U>(&self, byte_offset: usize, len: usize) -> DeviceBuffer<U> {
        let need = byte_offset + len * std::mem::size_of::<U>();
        assert!(
            need <= self.byte_len(),
            "view_as out of range: need {need} bytes, have {}",
            self.byte_len()
        );
        let raw = unsafe { (self.raw as *mut u8).add(byte_offset) as sys::hipDeviceptr_t };
        DeviceBuffer {
            raw,
            len,
            device_id: self.device_id,
            is_view: true,
            _marker: PhantomData,
        }
    }

    /// The allocation as a host-writable byte slice.
    ///
    /// Only sound on a unified-memory APU, where `hipMalloc` returns a
    /// CPU-addressable pointer (verified on gfx1151 by writing a sentinel
    /// through `raw()` and reading it back). This is the ONE unsafe step in
    /// the fast model-load path — callers then use ordinary safe slice APIs.
    ///
    /// # Safety
    /// The caller asserts this device's allocations are CPU-addressable, and
    /// that no GPU work touches the buffer concurrently. After writing, issue
    /// a `SeqCst` fence before the GPU reads (see `host_write_barrier`).
    pub unsafe fn as_host_slice_mut(&mut self) -> &mut [u8] {
        std::slice::from_raw_parts_mut(self.raw as *mut u8, self.byte_len())
    }

    /// Publish CPU writes made through [`as_host_slice_mut`] so the GPU sees
    /// them. On x86 a `SeqCst` fence lowers to a locked op / `mfence`, which
    /// also drains write-combining buffers — the case that would otherwise
    /// leave stores sitting in the core when a kernel launches.
    pub fn host_write_barrier() {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
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

    /// Stream-ordered zero fill. Unlike [`fill_zero`](Self::fill_zero)
    /// (synchronous `hipMemset`), this queues on `stream` and returns
    /// immediately — safe to interleave with kernels on the same stream.
    pub fn fill_zero_async(&mut self, stream: &Stream) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipMemsetAsync(self.raw, 0, self.byte_len(), stream.raw()) },
            "hipMemsetAsync",
        )
    }
}

impl<T> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if !self.raw.is_null() && !self.is_view {
            // `hipFree` resolves the pointer against the CURRENT device; freeing
            // a dGPU allocation while the iGPU is current corrupts the runtime's
            // bookkeeping. See the note on `Stream`'s Drop.
            let _guard = crate::device::DeviceGuard::enter(self.device_id);
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

/// `hipHostMalloc` flags. Non-coherent host memory is COARSE grained, so the
/// GPU may cache it — which is what makes GPU reads out of it fast on an APU.
/// Coherent (the default) is fine grained and typically uncached on the GPU.
pub const HIP_HOST_MALLOC_COHERENT: u32 = 0x4000_0000;
pub const HIP_HOST_MALLOC_NON_COHERENT: u32 = 0x8000_0000;

impl<T> PinnedBuffer<T> {
    pub fn new(len: usize) -> eyre::Result<Self> {
        Self::new_with_flags(len, 0)
    }

    /// As `new`, with explicit `hipHostMalloc` flags.
    pub fn new_with_flags(len: usize, flags: u32) -> eyre::Result<Self> {
        let bytes = len.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            eyre!("PinnedBuffer size overflow: {} * {}", len, std::mem::size_of::<T>())
        })?;
        let mut raw: *mut c_void = ptr::null_mut();
        check_eyre(
            unsafe { sys::hipHostMalloc(&mut raw, bytes, flags) },
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

    /// The same allocation viewed as a device pointer. `hipHostMalloc`
    /// memory is device-accessible under ROCm's unified addressing, and on an
    /// APU it is the very same physical RAM the iGPU reads through GTT.
    pub fn device_ptr(&self) -> crate::sys::hipDeviceptr_t {
        self.raw as crate::sys::hipDeviceptr_t
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
