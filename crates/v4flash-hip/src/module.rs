use std::ffi::{CString, c_void};
use std::ptr;

use color_eyre::eyre;

use crate::error::check_eyre;
use crate::stream::Stream;
use crate::sys;

/// A loaded HIP code object (hsaco / clang offload bundle). `load_data`
/// expects the *current* device to be the one we want the module on.
pub struct Module {
    raw: sys::hipModule_t,
}

impl Module {
    /// Load a code object from an in-memory blob. The image is what
    /// `hipcc --genco --offload-arch=gfxNNNN` produces; if hipcc wraps
    /// the result in a clang offload bundle, `hipModuleLoadData` will
    /// transparently unbundle for the current device.
    pub fn load_data(image: &[u8]) -> eyre::Result<Self> {
        let mut raw: sys::hipModule_t = ptr::null_mut();
        check_eyre(
            unsafe { sys::hipModuleLoadData(&mut raw, image.as_ptr() as *const c_void) },
            "hipModuleLoadData",
        )?;
        Ok(Module { raw })
    }

    pub fn get_function(&self, name: &str) -> eyre::Result<Function<'_>> {
        let c_name = CString::new(name).expect("kernel name has null byte");
        let mut raw: sys::hipFunction_t = ptr::null_mut();
        check_eyre(
            unsafe { sys::hipModuleGetFunction(&mut raw, self.raw, c_name.as_ptr()) },
            "hipModuleGetFunction",
        )?;
        Ok(Function {
            raw,
            _marker: std::marker::PhantomData,
        })
    }
}

impl Drop for Module {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let code = unsafe { sys::hipModuleUnload(self.raw) };
            if code != sys::HIP_SUCCESS {
                tracing::warn!(code, "hipModuleUnload failed during drop");
            }
        }
    }
}

/// A kernel handle. Borrowed from its parent `Module`.
pub struct Function<'m> {
    raw: sys::hipFunction_t,
    _marker: std::marker::PhantomData<&'m Module>,
}

/// Grid + block dimensions for a kernel launch.
#[derive(Debug, Clone, Copy)]
pub struct LaunchConfig {
    pub grid: (u32, u32, u32),
    pub block: (u32, u32, u32),
    pub shared_mem_bytes: u32,
}

impl LaunchConfig {
    pub fn simple(grid_x: u32, block_x: u32) -> Self {
        LaunchConfig {
            grid: (grid_x, 1, 1),
            block: (block_x, 1, 1),
            shared_mem_bytes: 0,
        }
    }
}

impl<'m> Function<'m> {
    /// Launch with raw kernel-argument pointers. Caller supplies a slice
    /// of `*mut c_void` where each entry points to a single argument
    /// (HIP/CUDA "kernelParams" ABI).
    ///
    /// # Safety
    /// `args` must contain pointers to live values of the correct type
    /// and lifetime; HIP reads them during the launch (which is async).
    pub unsafe fn launch_raw(
        &self,
        cfg: LaunchConfig,
        stream: &Stream,
        args: &mut [*mut c_void],
    ) -> eyre::Result<()> {
        let extra: *mut *mut c_void = ptr::null_mut();
        check_eyre(
            unsafe {
                sys::hipModuleLaunchKernel(
                    self.raw,
                    cfg.grid.0,
                    cfg.grid.1,
                    cfg.grid.2,
                    cfg.block.0,
                    cfg.block.1,
                    cfg.block.2,
                    cfg.shared_mem_bytes,
                    stream.raw(),
                    args.as_mut_ptr(),
                    extra,
                )
            },
            "hipModuleLaunchKernel",
        )
    }
}
