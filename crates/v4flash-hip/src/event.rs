use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::stream::Stream;
use crate::sys;

/// A HIP event. By default events carry timing data — pass
/// `Event::new_no_timing()` for low-overhead sync-only events.
pub struct Event {
    raw: sys::hipEvent_t,
    /// Device current at creation. `hipEventCreate` binds the event to it, and
    /// `hipEventDestroy` resolves against the CURRENT device, so Drop must
    /// restore it — see the note on `Stream`'s Drop.
    device_id: i32,
}


impl Event {
    pub fn new() -> eyre::Result<Self> {
        let mut raw: sys::hipEvent_t = ptr::null_mut();
        check_eyre(unsafe { sys::hipEventCreate(&mut raw) }, "hipEventCreate")?;
        Ok(Event { raw, device_id: crate::device::current_device() })
    }

    pub fn new_no_timing() -> eyre::Result<Self> {
        let mut raw: sys::hipEvent_t = ptr::null_mut();
        check_eyre(
            unsafe {
                sys::hipEventCreateWithFlags(&mut raw, sys::HIP_EVENT_DISABLE_TIMING)
            },
            "hipEventCreateWithFlags(DISABLE_TIMING)",
        )?;
        Ok(Event { raw, device_id: crate::device::current_device() })
    }

    pub fn raw(&self) -> sys::hipEvent_t {
        self.raw
    }

    pub fn record(&self, stream: &Stream) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipEventRecord(self.raw, stream.raw()) },
            "hipEventRecord",
        )
    }

    pub fn synchronize(&self) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipEventSynchronize(self.raw) },
            "hipEventSynchronize",
        )
    }

    /// Non-blocking: `Ok(true)` once every operation before the record has
    /// completed (`hipEventQuery` == hipSuccess), `Ok(false)` while pending
    /// (hipErrorNotReady = 600).
    pub fn query(&self) -> eyre::Result<bool> {
        let r = unsafe { sys::hipEventQuery(self.raw) };
        if r == sys::HIP_SUCCESS { return Ok(true); }
        if r == 600 { return Ok(false); }
        check_eyre(r, "hipEventQuery").map(|_| true)
    }

    /// Elapsed milliseconds between two events. Both events must have been
    /// recorded with timing enabled.
    pub fn elapsed_ms(start: &Event, end: &Event) -> eyre::Result<f32> {
        let mut ms: f32 = 0.0;
        check_eyre(
            unsafe { sys::hipEventElapsedTime(&mut ms, start.raw, end.raw) },
            "hipEventElapsedTime",
        )?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _guard = crate::device::DeviceGuard::enter(self.device_id);
            let code = unsafe { sys::hipEventDestroy(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipEventDestroy failed during drop");
            }
        }
    }
}
