use std::marker::PhantomData;
use std::os::raw::c_void;
use std::ptr;

use color_eyre::eyre::{self, eyre};

use crate::error::check_eyre;
use crate::ptr_registry::{self, Kind};
use crate::ring;
use crate::stream::Stream;
use crate::sys;

type Loc = &'static std::panic::Location<'static>;

/// A device-side operand must lie inside one live allocation of `device` (or pinned memory): `ptr_registry`.
#[inline]
fn on_device(what: &'static str, raw: sys::hipDeviceptr_t, bytes: usize, device: i32, loc: Loc) -> eyre::Result<()> {
    ptr_registry::check_device(what, raw as usize, bytes, device, loc)
}

/// Ring entry for a copy / memset: args = (dst, src, bytes).
#[inline]
fn ring_copy(device: i32, kind: &'static str, loc: Loc, stream: &Stream, dst: usize, src: usize, bytes: usize) {
    ring::record(device, kind, Some(loc), stream.raw() as usize, [bytes as u64, 0], &[dst as u64, src as u64, bytes as u64]);
}

/// As [`ring_copy`] for the blocking calls (the null stream).
#[inline]
fn ring_copy_sync(device: i32, kind: &'static str, loc: Loc, dst: usize, src: usize, bytes: usize) {
    ring::record(device, kind, Some(loc), 0, [bytes as u64, 0], &[dst as u64, src as u64, bytes as u64]);
}

/// Typed device-side buffer. Owned; freed on drop. The `device_id` field
/// is informational — HIP doesn't tag allocations with a device, so we
/// track it ourselves so peer-copy calls can pass the right source/dst
/// device IDs.
///
/// `is_view = true` means this struct refers to a sub-range of another
/// allocation (created via [`DeviceBuffer::slice_view`] /
/// [`DeviceBuffer::slice_view_mut`]); Drop skips `hipFree` for views.
///
/// Owning buffers are registered in [`ptr_registry`] (`[raw, raw + bytes)`, device,
/// allocation site) and deregistered in Drop; every copy / memset checks its device
/// side against the registry before calling HIP (`V41_PTR_REGISTRY=0` disables).
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
        // An open `alloc_scope` for this device records a phantom or hands out a planned view (prefill VRAM loan).
        if let Some(planned) = crate::alloc_scope::take(device_id, bytes) {
            return Ok(DeviceBuffer { raw: planned?, len, device_id, is_view: true, _marker: PhantomData });
        }
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
        if let Err(e) = ptr_registry::register(raw as usize, bytes, device_id, Kind::Device, std::panic::Location::caller()) {
            unsafe { sys::hipFree(raw) };
            return Err(e);
        }
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
    /// LIFETIME HAZARD (shared with `slice_view` / `slice_view_mut` / `view_as`):
    /// a view carries no borrow of its parent, so nothing stops it outliving
    /// the allocation. Not redesigned (2026-10-19); instead every copy / memset
    /// on a view checks it still lies inside a LIVE registered allocation
    /// (`ptr_registry`), which catches a view of a freed parent unless the
    /// address range was re-allocated since. Kernel launches are checked only
    /// where a launcher validates its operands (the MXFP4 MoE launch canary).
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

    #[track_caller]
    pub fn copy_from_host(&mut self, src: &[T]) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if src.len() != self.len {
            return Err(eyre!(
                "copy_from_host length mismatch: src={} dst={}",
                src.len(),
                self.len
            ));
        }
        if crate::alloc_scope::phantom_write(self.raw) {
            return Ok(());
        }
        on_device("copy_from_host dst", self.raw, self.byte_len(), self.device_id, loc)?;
        ptr_registry::check_host("copy_from_host src", src.as_ptr() as usize, self.byte_len(), loc)?;
        ring_copy_sync(self.device_id, "HtoD", loc, self.raw as usize, src.as_ptr() as usize, self.byte_len());
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

    #[track_caller]
    pub fn copy_to_host(&self, dst: &mut [T]) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        debug_assert!(!crate::alloc_scope::is_phantom(self.raw), "copy_to_host from a phantom (alloc_scope record pass)");
        if dst.len() != self.len {
            return Err(eyre!(
                "copy_to_host length mismatch: src={} dst={}",
                self.len,
                dst.len()
            ));
        }
        on_device("copy_to_host src", self.raw, self.byte_len(), self.device_id, loc)?;
        ptr_registry::check_host("copy_to_host dst", dst.as_ptr() as usize, self.byte_len(), loc)?;
        ring_copy_sync(self.device_id, "DtoH", loc, dst.as_ptr() as usize, self.raw as usize, self.byte_len());
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

    /// `src` may be pageable (HIP stages it before returning) or a slice of a
    /// `PinnedBuffer`; a slice that starts in a pinned buffer must end in it
    /// (`ptr_registry::check_host`).
    #[track_caller]
    pub fn copy_from_host_async(&mut self, src: &[T], stream: &Stream) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if src.len() != self.len {
            return Err(eyre!(
                "copy_from_host_async length mismatch: src={} dst={}",
                src.len(),
                self.len
            ));
        }
        if crate::alloc_scope::phantom_write(self.raw) {
            return Ok(());
        }
        on_device("copy_from_host_async dst", self.raw, self.byte_len(), self.device_id, loc)?;
        ptr_registry::check_host("copy_from_host_async src", src.as_ptr() as usize, self.byte_len(), loc)?;
        ring_copy(self.device_id, "HtoD async", loc, stream, self.raw as usize, src.as_ptr() as usize, self.byte_len());
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

    /// Async device-to-host copy of the whole view into `dst[dst_offset..]`,
    /// queued on `stream`. `dst` holds the bytes only once `stream` has been
    /// synchronized (or an event recorded after this copy has); reading it
    /// earlier races the DMA.
    #[track_caller]
    pub fn copy_to_pinned_async(
        &self,
        dst: &mut PinnedBuffer<T>,
        dst_offset: usize,
        stream: &Stream,
    ) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if dst_offset.checked_add(self.len).is_none_or(|end| end > dst.len()) {
            return Err(eyre!(
                "copy_to_pinned_async out of range: offset={} len={} dst={}",
                dst_offset,
                self.len,
                dst.len()
            ));
        }
        // SAFETY: `dst_offset + len <= dst.len()` was checked above.
        let dst_ptr = unsafe { dst.as_mut_slice().as_mut_ptr().add(dst_offset) };
        on_device("copy_to_pinned_async src", self.raw, self.byte_len(), self.device_id, loc)?;
        ptr_registry::check_pinned("copy_to_pinned_async dst", dst_ptr as usize, self.byte_len(), loc)?;
        ring_copy(self.device_id, "DtoH async", loc, stream, dst_ptr as usize, self.raw as usize, self.byte_len());
        check_eyre(
            unsafe {
                sys::hipMemcpyAsync(
                    dst_ptr as sys::hipDeviceptr_t,
                    self.raw,
                    self.byte_len(),
                    sys::HIP_MEMCPY_DEVICE_TO_HOST,
                    stream.raw(),
                )
            },
            "hipMemcpyAsync(DtoH)",
        )
    }

    /// Async device-to-device copy on the SAME device, queued on
    /// `stream`. Returns immediately; the copy completes when prior
    /// work on `stream` completes. Used by the spec-decode snapshot
    /// path (one snapshot per layer per pair = 100s of copies; sync
    /// version would stall the pipeline).
    #[track_caller]
    pub fn copy_from_buffer_async(
        &mut self,
        src: &DeviceBuffer<T>,
        stream: &Stream,
    ) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        debug_assert!(
            !crate::alloc_scope::is_phantom(self.raw) && !crate::alloc_scope::is_phantom(src.raw),
            "device copy on a phantom (alloc_scope record pass)"
        );
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
        on_device("copy_from_buffer_async dst", self.raw, self.byte_len(), self.device_id, loc)?;
        on_device("copy_from_buffer_async src", src.raw, self.byte_len(), src.device_id, loc)?;
        ring_copy(self.device_id, "DtoD async", loc, stream, self.raw as usize, src.raw as usize, self.byte_len());
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
    #[track_caller]
    pub fn copy_from_buffer(&mut self, src: &DeviceBuffer<T>) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        debug_assert!(
            !crate::alloc_scope::is_phantom(self.raw) && !crate::alloc_scope::is_phantom(src.raw),
            "device copy on a phantom (alloc_scope record pass)"
        );
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
        on_device("copy_from_buffer dst", self.raw, self.byte_len(), self.device_id, loc)?;
        on_device("copy_from_buffer src", src.raw, self.byte_len(), src.device_id, loc)?;
        ring_copy_sync(self.device_id, "DtoD", loc, self.raw as usize, src.raw as usize, self.byte_len());
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
    #[track_caller]
    pub fn copy_to_peer_async(
        &self,
        dst: &mut DeviceBuffer<T>,
        stream: &Stream,
    ) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if dst.len != self.len {
            return Err(eyre!(
                "copy_to_peer_async length mismatch: src={} dst={}",
                self.len,
                dst.len
            ));
        }
        on_device("copy_to_peer_async dst", dst.raw, self.byte_len(), dst.device_id, loc)?;
        on_device("copy_to_peer_async src", self.raw, self.byte_len(), self.device_id, loc)?;
        ring_copy(stream.device_id(), "peer async", loc, stream, dst.raw as usize, self.raw as usize, self.byte_len());
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

    #[track_caller]
    pub fn fill_zero(&mut self) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if crate::alloc_scope::phantom_write(self.raw) {
            return Ok(());
        }
        on_device("fill_zero", self.raw, self.byte_len(), self.device_id, loc)?;
        ring_copy_sync(self.device_id, "memset", loc, self.raw as usize, 0, self.byte_len());
        check_eyre(
            unsafe { sys::hipMemset(self.raw, 0, self.byte_len()) },
            "hipMemset",
        )
    }

    /// Stream-ordered zero fill. Unlike [`fill_zero`](Self::fill_zero)
    /// (synchronous `hipMemset`), this queues on `stream` and returns
    /// immediately — safe to interleave with kernels on the same stream.
    #[track_caller]
    pub fn fill_zero_async(&mut self, stream: &Stream) -> eyre::Result<()> {
        let loc = std::panic::Location::caller();
        if crate::alloc_scope::phantom_write(self.raw) {
            return Ok(());
        }
        on_device("fill_zero_async", self.raw, self.byte_len(), self.device_id, loc)?;
        ring_copy(self.device_id, "memset async", loc, stream, self.raw as usize, 0, self.byte_len());
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
            ptr_registry::deregister(self.raw as usize, self.byte_len());
            let _guard = crate::device::DeviceGuard::enter(self.device_id);
            let code = unsafe { sys::hipFree(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipFree failed during drop");
            }
        }
    }
}

/// Pinned host buffer (page-locked). Required for fast async DMA.
/// Registered in [`ptr_registry`] as pinned host memory while it lives.
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
    #[track_caller]
    pub fn new(len: usize) -> eyre::Result<Self> {
        Self::new_with_flags(len, 0)
    }

    /// As `new`, with explicit `hipHostMalloc` flags.
    #[track_caller]
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
        if let Err(e) = ptr_registry::register(raw as usize, bytes, -1, Kind::Pinned, std::panic::Location::caller()) {
            unsafe { sys::hipHostFree(raw) };
            return Err(e);
        }
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
            ptr_registry::deregister(self.raw as usize, self.len * std::mem::size_of::<T>());
            let code = unsafe { sys::hipHostFree(self.raw as *mut c_void) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipHostFree failed during drop");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ptr_registry::tests::{fake_base, reg};

    /// Every copy / memset path checks its pointers BEFORE calling HIP, so the failing cases run on a CPU: fake
    /// registered allocations, views into and around them, and a null-handle stream that never reaches HIP.
    #[test]
    fn copies_with_unregistered_pointers_fail_before_hip() {
        if !ptr_registry::enabled() {
            return; // V41_PTR_REGISTRY=0 in the test environment
        }
        let base = fake_base(10);
        reg(base, 4096, 1, Kind::Device);
        reg(base + 0x10_000, 4096, 2, Kind::Device);
        let stream = Stream::null_for_test(1);
        let view = |addr: usize, n: usize, dev: i32| unsafe { DeviceBuffer::<u32>::from_raw_parts(addr as sys::hipDeviceptr_t, n, dev) };
        let host = vec![0u32; 16];
        let mut back = vec![0u32; 16];
        let named = |e: eyre::Report, what: &str, problem: &str| {
            let s = e.to_string();
            assert!(s.contains(what) && s.contains(problem) && s.contains(file!()), "{s}");
        };

        let mut garbage = view(base + 0x8000, 16, 1);
        named(garbage.copy_from_host_async(&host, &stream).unwrap_err(), "copy_from_host_async dst", "not inside any live allocation");
        named(garbage.copy_from_host(&host).unwrap_err(), "copy_from_host dst", "not inside any live allocation");
        named(garbage.copy_to_host(&mut back).unwrap_err(), "copy_to_host src", "not inside any live allocation");
        named(garbage.fill_zero().unwrap_err(), "fill_zero", "not inside any live allocation");
        named(garbage.fill_zero_async(&stream).unwrap_err(), "fill_zero_async", "not inside any live allocation");

        let mut straddle = view(base + 4096 - 32, 16, 1);
        named(straddle.copy_from_host_async(&host, &stream).unwrap_err(), "copy_from_host_async dst", "runs 32 B past the end");

        let mut wrong_dev = view(base, 16, 0);
        named(wrong_dev.copy_from_host_async(&host, &stream).unwrap_err(), "dst", "dev1 allocation but the operand claims dev0");

        // DtoD: a good dst with a garbage src fails on the src
        let mut good = view(base, 16, 1);
        named(good.copy_from_buffer_async(&garbage, &stream).unwrap_err(), "copy_from_buffer_async src", "not inside any live allocation");
        named(good.copy_from_buffer(&garbage).unwrap_err(), "copy_from_buffer src", "not inside any live allocation");
        // peer: the dst on dev 2 is live, the src is not
        let mut peer_dst = view(base + 0x10_000, 16, 2);
        named(garbage.copy_to_peer_async(&mut peer_dst, &stream).unwrap_err(), "copy_to_peer_async src", "not inside any live allocation");
        // a view of a FREED allocation (`slice_view`'s lifetime hazard) fails once its parent deregisters
        let mut child = good.slice_view(4, 8);
        ptr_registry::deregister(base, 4096);
        named(child.copy_from_host_async(&host[..8], &stream).unwrap_err(), "copy_from_host_async dst", "not inside any live allocation");
        // stream value ops
        let s2 = Stream::null_for_test(2);
        let e = unsafe { s2.write_value32((base + 0x20_000) as *mut u32, 1) }.unwrap_err().to_string();
        assert!(e.contains("write_value32 ptr") && e.contains("not inside any live allocation"), "{e}");
        ptr_registry::deregister(base + 0x10_000, 4096);
    }
}
