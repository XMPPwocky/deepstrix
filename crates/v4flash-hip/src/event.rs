use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::stream::Stream;
use crate::sys;

/// A HIP event. By default events carry timing data — pass
/// `Event::new_no_timing()` for low-overhead sync-only events.
pub struct Event {
    raw: sys::hipEvent_t,
}

impl Event {
    pub fn new() -> eyre::Result<Self> {
        let mut raw: sys::hipEvent_t = ptr::null_mut();
        check_eyre(unsafe { sys::hipEventCreate(&mut raw) }, "hipEventCreate")?;
        Ok(Event { raw })
    }

    pub fn new_no_timing() -> eyre::Result<Self> {
        let mut raw: sys::hipEvent_t = ptr::null_mut();
        check_eyre(
            unsafe {
                sys::hipEventCreateWithFlags(&mut raw, sys::HIP_EVENT_DISABLE_TIMING)
            },
            "hipEventCreateWithFlags(DISABLE_TIMING)",
        )?;
        Ok(Event { raw })
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
            let code = unsafe { sys::hipEventDestroy(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipEventDestroy failed during drop");
            }
        }
    }
}
