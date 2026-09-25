use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::event::Event;
use crate::graph::Graph;
use crate::sys;

/// A HIP stream. `device_id` is AUTHORITATIVE: the constructors enter a `DeviceGuard` so the
/// stream is created on that device regardless of ambient `hipSetDevice` state (hipStreamCreate
/// binds to the CURRENT device), restoring the previous device afterwards. This matters because a
/// stream's device decides peer-copy direction. Streams may be prioritized; smaller priority value = higher
/// priority, per hipDeviceGetStreamPriorityRange.
pub struct Stream {
    raw: sys::hipStream_t,
    device_id: i32,
}

impl Stream {
    pub fn new(device_id: i32) -> eyre::Result<Self> {
        let mut raw: sys::hipStream_t = ptr::null_mut();
        let _guard = crate::device::Device::scoped(device_id)?;
        check_eyre(unsafe { sys::hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Stream { raw, device_id })
    }

    /// A stream that does NOT synchronize with the null stream
    /// (`hipStreamNonBlocking`). A blocking `hipMemcpy` runs on the null stream,
    /// which waits for every blocking stream on the device; work queued here is
    /// ordered only by this stream and the events it waits on.
    pub fn new_non_blocking(device_id: i32) -> eyre::Result<Self> {
        let mut raw: sys::hipStream_t = ptr::null_mut();
        let _guard = crate::device::Device::scoped(device_id)?;
        check_eyre(
            unsafe { sys::hipStreamCreateWithFlags(&mut raw, sys::HIP_STREAM_NON_BLOCKING) },
            "hipStreamCreateWithFlags",
        )?;
        Ok(Stream { raw, device_id })
    }

    pub fn new_with_priority(device_id: i32, priority: i32) -> eyre::Result<Self> {
        let mut raw: sys::hipStream_t = ptr::null_mut();
        let _guard = crate::device::Device::scoped(device_id)?;
        check_eyre(
            unsafe {
                sys::hipStreamCreateWithPriority(&mut raw, sys::HIP_STREAM_DEFAULT, priority)
            },
            "hipStreamCreateWithPriority",
        )?;
        Ok(Stream { raw, device_id })
    }

    pub fn raw(&self) -> sys::hipStream_t {
        self.raw
    }

    pub fn device_id(&self) -> i32 {
        self.device_id
    }

    pub fn synchronize(&self) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipStreamSynchronize(self.raw) },
            "hipStreamSynchronize",
        )
    }

    /// Non-blocking: `Ok(true)` when every operation queued on this stream has
    /// completed, `Ok(false)` while any is still pending (hipErrorNotReady).
    ///
    /// EXPOSED WAIT vs HOST OVERHEAD. A blocking readback costs one of two very
    /// different things, and they need opposite fixes: either it waits on real
    /// device work we depend on (exposed compute — the fix is to OVERLAP it), or
    /// it waits on nothing (pure host overhead — the fix is to DELETE it). The
    /// stage timer alone cannot tell them apart. Calling this immediately before
    /// the blocking call does: busy = exposed compute, idle = overhead.
    pub fn query(&self) -> eyre::Result<bool> {
        let r = unsafe { sys::hipStreamQuery(self.raw) };
        if r == sys::HIP_SUCCESS {
            return Ok(true);
        }
        if r == 600 {
            return Ok(false);
        }
        check_eyre(r, "hipStreamQuery").map(|_| true)
    }

    pub fn wait_event(&self, event: &Event) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipStreamWaitEvent(self.raw, event.raw(), 0) },
            "hipStreamWaitEvent",
        )
    }

    /// Enqueue a wait until the 32-bit value at `ptr` is >= `value`
    /// (unsigned). The comparison happens at stream-EXECUTION time, so the
    /// wait may be enqueued before the producer's write call — unlike
    /// `wait_event`, which snapshots the event's last record at call time
    /// and degenerates to a no-op when enqueued ahead of the record.
    ///
    /// # Safety
    /// `ptr` must remain valid (pinned host memory or device memory
    /// accessible to this stream's device) until the wait completes.
    pub unsafe fn wait_value32_gte(&self, ptr: *mut u32, value: u32) -> eyre::Result<()> {
        check_eyre(
            unsafe {
                sys::hipStreamWaitValue32(
                    self.raw,
                    ptr as *mut std::ffi::c_void,
                    value,
                    0, // hipStreamWaitValueGte
                    0xFFFF_FFFF,
                )
            },
            "hipStreamWaitValue32",
        )
    }

    /// Enqueue a 32-bit write of `value` to `ptr`, ordered after all prior
    /// work on this stream. Companion to [`wait_value32_gte`].
    ///
    /// # Safety
    /// `ptr` must remain valid until the write completes.
    pub unsafe fn write_value32(&self, ptr: *mut u32, value: u32) -> eyre::Result<()> {
        check_eyre(
            unsafe {
                sys::hipStreamWriteValue32(self.raw, ptr as *mut std::ffi::c_void, value, 0)
            },
            "hipStreamWriteValue32",
        )
    }

    /// Begin recording all subsequent async operations on this stream
    /// into a HIP graph. Call [`Stream::end_capture`] to finalize.
    ///
    /// `mode` should normally be
    /// [`sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL`] — `Global` is the
    /// HIP default but enforces synchronization with other threads,
    /// which the orchestrator doesn't need.
    pub fn begin_capture(&self, mode: u32) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipStreamBeginCapture(self.raw, mode) },
            "hipStreamBeginCapture",
        )
    }

    /// End an active capture begun with [`begin_capture`] and return
    /// the recorded [`Graph`].
    pub fn end_capture(&self) -> eyre::Result<Graph> {
        let mut raw: sys::hipGraph_t = ptr::null_mut();
        check_eyre(
            unsafe { sys::hipStreamEndCapture(self.raw, &mut raw) },
            "hipStreamEndCapture",
        )?;
        // SAFETY: we own the returned graph; wrap in our RAII type.
        Ok(crate::graph::Graph::from_raw(raw))
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Bind the stream's own device first: `hipStreamDestroy` reaches
            // `hip::Device::RemoveStream`, which erases the stream from the
            // CURRENT device's stream table, not the one the stream belongs to.
            // Same recorded-but-unapplied `device_id` footgun that DeviceGuard
            // was introduced to close elsewhere; the Drop impls were missed.
            //
            // NOTE: this is a latent correctness fix, NOT the cause of the
            // teardown crashes seen on 2026-09-13. Those were a detached engine
            // thread still dropping HIP objects after `main` returned and libc
            // began running libamdhip64's static destructors — fixed by joining
            // the worker in `EngineHandle::shutdown`. Guarding here did not stop
            // them; joining did.
            let _guard = crate::device::DeviceGuard::enter(self.device_id);
            // Errors during drop are logged but not propagated.
            let code = unsafe { sys::hipStreamDestroy(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipStreamDestroy failed during drop");
            }
        }
    }
}
