use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::event::Event;
use crate::sys;

/// A HIP stream. Created on the current device (caller must `Device::set_current`
/// first). Streams may be prioritized; smaller priority value = higher
/// priority, per hipDeviceGetStreamPriorityRange.
pub struct Stream {
    raw: sys::hipStream_t,
    device_id: i32,
}

impl Stream {
    pub fn new(device_id: i32) -> eyre::Result<Self> {
        let mut raw: sys::hipStream_t = ptr::null_mut();
        check_eyre(unsafe { sys::hipStreamCreate(&mut raw) }, "hipStreamCreate")?;
        Ok(Stream { raw, device_id })
    }

    pub fn new_with_priority(device_id: i32, priority: i32) -> eyre::Result<Self> {
        let mut raw: sys::hipStream_t = ptr::null_mut();
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

    pub fn wait_event(&self, event: &Event) -> eyre::Result<()> {
        check_eyre(
            unsafe { sys::hipStreamWaitEvent(self.raw, event.raw(), 0) },
            "hipStreamWaitEvent",
        )
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            // Errors during drop are logged but not propagated.
            let code = unsafe { sys::hipStreamDestroy(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipStreamDestroy failed during drop");
            }
        }
    }
}
