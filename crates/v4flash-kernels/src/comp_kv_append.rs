//! Append one compressor row into the per-layer `comp_kv` cache
//! (M13.5). Sibling of `kv_cache_append` but without SWA eviction.

use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{DeviceBuffer, LaunchConfig, Module, Stream};

const COMP_KV_APPEND_GFX1201: &[u8] = include_bytes!(env!("KERNEL_COMP_KV_APPEND_GFX1201"));
const COMP_KV_APPEND_GFX1151: &[u8] = include_bytes!(env!("KERNEL_COMP_KV_APPEND_GFX1151"));

pub struct CompKvAppend {
    module: Module,
}

impl CompKvAppend {
    pub fn for_arch(arch: &str) -> eyre::Result<Self> {
        let image: &[u8] = if arch.starts_with("gfx1201") {
            COMP_KV_APPEND_GFX1201
        } else if arch.starts_with("gfx1151") {
            COMP_KV_APPEND_GFX1151
        } else {
            return Err(eyre!("unsupported arch for comp_kv_append: {arch}"));
        };
        let module = Module::load_data(image)?;
        Ok(Self { module })
    }

    pub fn launch(
        &self,
        stream: &Stream,
        comp_kv: &mut DeviceBuffer<f32>,
        row: &DeviceBuffer<f32>,
        n_comp: u32,
        head_dim: u32,
    ) -> eyre::Result<()> {
        if head_dim == 0 || head_dim > 1024 {
            return Err(eyre!(
                "comp_kv_append: head_dim must be in [1, 1024], got {head_dim}"
            ));
        }
        let need = (n_comp as usize + 1) * (head_dim as usize);
        if comp_kv.len() < need {
            return Err(eyre!(
                "comp_kv_append: comp_kv len {} < (n_comp+1)*head_dim {}",
                comp_kv.len(),
                need
            ));
        }
        if row.len() < head_dim as usize {
            return Err(eyre!(
                "comp_kv_append: row len {} < head_dim {}",
                row.len(),
                head_dim
            ));
        }

        let function = self.module.get_function("comp_kv_append")?;
        let mut ck_ptr = comp_kv.raw();
        let mut row_ptr = row.raw();
        let mut nc = n_comp;
        let mut hd = head_dim;
        let mut args: [*mut c_void; 4] = [
            &mut ck_ptr as *mut _ as *mut c_void,
            &mut row_ptr as *mut _ as *mut c_void,
            &mut nc as *mut _ as *mut c_void,
            &mut hd as *mut _ as *mut c_void,
        ];
        let cfg = LaunchConfig {
            grid: (1, 1, 1),
            block: (head_dim, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { function.launch_raw(cfg, stream, &mut args) }
    }
}
