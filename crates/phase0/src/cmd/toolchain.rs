//! `phase0 toolchain` — sanity sweep across the build/runtime toolchain.
//!
//! - HIP runtime version, hipcc path, ROCm path
//! - Per-device: id, name, gcnArchName, total mem, MPC, integrated, pci, compute cap
//! - Environment vars that affect HSA/HIP (HSA_OVERRIDE_GFX_VERSION, HIP_VISIBLE_DEVICES, etc.)
//! - Embedded hsaco load test: load `hello` for each device's reported arch,
//!   launch it, verify it writes 42. This is the "does --genco produce blobs
//!   hipModuleLoadData accepts" check from the design doc's Phase 0 list.

use std::env;
use std::ffi::c_void;

use color_eyre::eyre::{self, eyre};
use serde::Serialize;
use v4flash_hip::{Device, DeviceBuffer, Event, LaunchConfig, Module, Stream};

use crate::results;

const HELLO_GFX1201: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1201"));
const HELLO_GFX1151: &[u8] = include_bytes!(env!("KERNEL_HELLO_GFX1151"));

#[derive(Serialize)]
pub struct ToolchainReport {
    pub gate: &'static str,
    pub timestamp: u64,
    pub env: Env,
    pub devices: Vec<DeviceReport>,
}

#[derive(Serialize)]
pub struct Env {
    pub hsa_override_gfx_version: Option<String>,
    pub hip_visible_devices: Option<String>,
    pub rocr_visible_devices: Option<String>,
    pub gpu_device_ordinal: Option<String>,
    pub rocm_path: Option<String>,
    pub hip_path: Option<String>,
    pub hipcc: Option<String>,
}

#[derive(Serialize)]
pub struct DeviceReport {
    pub id: i32,
    pub name: String,
    pub gcn_arch_name: String,
    pub total_global_mem_mib: u64,
    pub multi_processor_count: i32,
    pub integrated: bool,
    pub pci_bus: i32,
    pub pci_device: i32,
    pub pci_domain: i32,
    pub major: i32,
    pub minor: i32,
    pub clock_rate_mhz: f64,
    pub memory_clock_rate_mhz: f64,
    pub memory_bus_width_bits: i32,
    pub l2_cache_size_bytes: i32,
    pub stream_priorities_supported: bool,
    pub hello_kernel_loaded: bool,
    pub hello_kernel_result: Option<i32>,
    pub hello_kernel_error: Option<String>,
}

pub fn run() -> eyre::Result<()> {
    let env = Env {
        hsa_override_gfx_version: env::var("HSA_OVERRIDE_GFX_VERSION").ok(),
        hip_visible_devices: env::var("HIP_VISIBLE_DEVICES").ok(),
        rocr_visible_devices: env::var("ROCR_VISIBLE_DEVICES").ok(),
        gpu_device_ordinal: env::var("GPU_DEVICE_ORDINAL").ok(),
        rocm_path: env::var("ROCM_PATH").ok(),
        hip_path: env::var("HIP_PATH").ok(),
        hipcc: env::var("HIPCC").ok(),
    };

    println!("== environment ==");
    println!("HSA_OVERRIDE_GFX_VERSION = {:?}", env.hsa_override_gfx_version);
    println!("HIP_VISIBLE_DEVICES      = {:?}", env.hip_visible_devices);
    println!("ROCR_VISIBLE_DEVICES     = {:?}", env.rocr_visible_devices);
    println!("GPU_DEVICE_ORDINAL       = {:?}", env.gpu_device_ordinal);
    println!("ROCM_PATH                = {:?}", env.rocm_path);
    println!("HIPCC                    = {:?}", env.hipcc);

    let devices = Device::all()?;
    println!("\n== devices ({}) ==", devices.len());

    let mut device_reports = Vec::new();
    for dev in &devices {
        let props = dev.properties()?;
        let mut report = DeviceReport {
            id: dev.id,
            name: props.name.clone(),
            gcn_arch_name: props.gcn_arch_name.clone(),
            total_global_mem_mib: (props.total_global_mem >> 20) as u64,
            multi_processor_count: props.multi_processor_count,
            integrated: props.integrated,
            pci_bus: props.pci_bus_id,
            pci_device: props.pci_device_id,
            pci_domain: props.pci_domain_id,
            major: props.major,
            minor: props.minor,
            clock_rate_mhz: props.clock_rate_khz as f64 / 1000.0,
            memory_clock_rate_mhz: props.memory_clock_rate_khz as f64 / 1000.0,
            memory_bus_width_bits: props.memory_bus_width_bits,
            l2_cache_size_bytes: props.l2_cache_size,
            stream_priorities_supported: props.stream_priorities_supported,
            hello_kernel_loaded: false,
            hello_kernel_result: None,
            hello_kernel_error: None,
        };

        println!(
            "[{}] {:?} ({}) — {} MiB, {} MPC, integrated={}, pci={:04x}:{:02x}:{:02x}, cc={}.{}, clock={:.0}MHz",
            dev.id,
            props.name,
            props.gcn_arch_name,
            report.total_global_mem_mib,
            props.multi_processor_count,
            props.integrated,
            props.pci_domain_id,
            props.pci_bus_id,
            props.pci_device_id,
            props.major,
            props.minor,
            report.clock_rate_mhz,
        );

        let image = match props.gcn_arch_name.as_str() {
            s if s.starts_with("gfx1201") => Some(HELLO_GFX1201),
            s if s.starts_with("gfx1151") => Some(HELLO_GFX1151),
            _ => None,
        };

        match image {
            None => {
                report.hello_kernel_error =
                    Some(format!("no embedded hsaco for {}", props.gcn_arch_name));
                println!(
                    "    hello kernel: SKIP (no hsaco built for {})",
                    props.gcn_arch_name
                );
            }
            Some(blob) => match try_hello(*dev, blob) {
                Ok(v) => {
                    report.hello_kernel_loaded = true;
                    report.hello_kernel_result = Some(v);
                    println!("    hello kernel: OK (wrote {})", v);
                }
                Err(e) => {
                    report.hello_kernel_error = Some(format!("{e:#}"));
                    println!("    hello kernel: FAIL ({e:#})");
                }
            },
        }

        device_reports.push(report);
    }

    let report = ToolchainReport {
        gate: "toolchain",
        timestamp: results::now_unix(),
        env,
        devices: device_reports,
    };

    let path = results::write("toolchain", &report)?;
    println!("\nwrote {}", path.display());
    Ok(())
}

/// Try to load `hello` for one device's arch, run it, return the int it
/// wrote. Errors are returned, not panicked, so the report still records
/// other devices even if one fails.
fn try_hello(dev: Device, image: &[u8]) -> eyre::Result<i32> {
    dev.set_current()?;
    let module = Module::load_data(image)?;
    let function = module.get_function("hello")?;

    let stream = Stream::new(dev.id)?;
    let mut buf: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, 1)?;
    buf.fill_zero()?;

    // Argument is a single i32 pointer; HIP wants &mut *mut c_void in the
    // kernelParams slice. We pass &mut raw_ptr.
    let mut raw_ptr = buf.raw();
    let mut args: [*mut c_void; 1] = [&mut raw_ptr as *mut _ as *mut c_void];

    unsafe { function.launch_raw(LaunchConfig::simple(1, 1), &stream, &mut args)? };

    let done = Event::new_no_timing()?;
    done.record(&stream)?;
    done.synchronize()?;

    let mut out = vec![0i32; 1];
    buf.copy_to_host(&mut out)?;
    if out[0] != 42 {
        return Err(eyre!("hello kernel wrote {}, expected 42", out[0]));
    }
    Ok(out[0])
}
