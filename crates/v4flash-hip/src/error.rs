use std::ffi::CStr;
use std::fmt;

use color_eyre::eyre;

use crate::sys;

/// A HIP runtime error: the numeric code plus the strings the runtime
/// gives us at the call site.
#[derive(Debug, Clone)]
pub struct HipError {
    pub code: i32,
    pub name: String,
    pub message: String,
    pub context: &'static str,
}

impl fmt::Display for HipError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "HIP error {} ({}) in {}: {}",
            self.code, self.name, self.context, self.message
        )
    }
}

impl std::error::Error for HipError {}

impl HipError {
    pub fn from_code(code: sys::hipError_t, context: &'static str) -> Self {
        // SAFETY: hipGetErrorName/String return static strings owned by the
        // runtime; they're always safe to read for any valid error code,
        // including ones we don't recognize.
        let name = unsafe {
            let p = sys::hipGetErrorName(code);
            if p.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        let message = unsafe {
            let p = sys::hipGetErrorString(code);
            if p.is_null() {
                "unknown".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };
        HipError {
            code: code as i32,
            name,
            message,
            context,
        }
    }
}

/// Convert a raw HIP return code to a Result. The `context` literal is
/// included in the error so we can pinpoint the failing call without a
/// stack trace.
#[inline]
pub fn check(code: sys::hipError_t, context: &'static str) -> Result<(), HipError> {
    if code == sys::HIP_SUCCESS {
        Ok(())
    } else {
        Err(HipError::from_code(code, context))
    }
}

/// Convenience: lift a HipError into an eyre report. Library functions
/// return `eyre::Result<T>`; binaries install `color_eyre::install()` to
/// pretty-print.
#[inline]
pub fn check_eyre(code: sys::hipError_t, context: &'static str) -> eyre::Result<()> {
    check(code, context).map_err(eyre::Report::new)
}
