//! WMMA throughput probe. Measures realized f16 WMMA TFLOPs on
//! gfx1151 (Strix Halo iGPU). The result decides whether the iq2
//! WMMA rewrite is worth the engineering cost — theoretical peak says
//! ~32× the dp4a rate, but if realized throughput is much lower the
//! rewrite isn't worth it.

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::wmma_probe::WmmaProbe;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 iGPU"))
}

#[test]
#[ignore]
fn bench_wmma_throughput_igpu() -> eyre::Result<()> {
    install_panic_handler()?;

    let igpu = pick_igpu()?;
    let arch = igpu.properties()?.gcn_arch_name;
    eprintln!("WMMA probe on {arch}");
    igpu.set_current()?;

    let probe = WmmaProbe::for_arch(&arch)?;
    let stream = Stream::new(igpu.id)?;

    let props = igpu.properties()?;
    let n_cus = props.multi_processor_count as u32;
    let max_clock_khz = props.clock_rate_khz;
    let max_clock_ghz = max_clock_khz as f64 / 1.0e6;
    eprintln!(
        "iGPU props: {} CUs, max clock {:.2} GHz",
        n_cus, max_clock_ghz
    );

    // Each warp does N_ITERS sequential WMMA accumulates into one
    // accumulator. To measure peak ALU throughput, we want enough
    // concurrent warps to fill the GPU.
    //
    // gfx1151: 4 SIMDs per CU, each SIMD holds 1 warp at a time when
    // running w32 wavefronts. So max ~4 warps in flight per CU.
    // With 40 CUs that's 160 concurrent warps for full ALU.
    // We oversubscribe a bit so the scheduler has work.
    let n_iters: u32 = std::env::var("WMMA_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10000);
    let warps_per_block: u32 = std::env::var("WMMA_WARPS_PER_BLOCK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);
    let n_blocks: u32 = std::env::var("WMMA_BLOCKS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(n_cus * 2);
    let block_threads = warps_per_block * 32;
    let n_warps_total = n_blocks * warps_per_block;
    eprintln!(
        "config: n_iters={n_iters}, warps_per_block={warps_per_block}, n_blocks={n_blocks}, total warps={n_warps_total}"
    );

    // Per-WMMA: 16×16×16 = 4096 muls + 4096 adds = 8192 ops.
    const OPS_PER_WMMA: u64 = 8192;
    let total_ops: u64 = (n_warps_total as u64) * (n_iters as u64) * OPS_PER_WMMA;
    eprintln!(
        "total ops to emit: {} ({:.2} GOps)",
        total_ops,
        total_ops as f64 / 1.0e9
    );

    // Inputs: just need values. Use ones.
    let a_host: Vec<u16> = vec![0x3c00u16; 16]; // 1.0 in f16
    let b_host: Vec<u16> = vec![0x3c00u16; 16];
    let mut a_in: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, 16)?;
    let mut b_in: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, 16)?;
    a_in.copy_from_host(&a_host)?;
    b_in.copy_from_host(&b_host)?;
    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, (n_warps_total as usize) * 8)?;

    // Warm-up both variants.
    probe.launch(&stream, &mut out, &a_in, &b_in, 100, n_blocks, block_threads)?;
    probe.launch_parallel(&stream, &mut out, &a_in, &b_in, 100, n_blocks, block_threads)?;
    stream.synchronize()?;

    let n_runs = 5;
    // Sequential (1 dep chain per warp).
    let mut seq_ms: Vec<f32> = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        probe.launch(&stream, &mut out, &a_in, &b_in, n_iters, n_blocks, block_threads)?;
        end.record(&stream)?;
        stream.synchronize()?;
        seq_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    seq_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let seq_min = seq_ms[0];
    let seq_total_ops: u64 = (n_warps_total as u64) * (n_iters as u64) * OPS_PER_WMMA;
    let seq_tops = (seq_total_ops as f64 / 1.0e12) / (seq_min as f64 / 1000.0);

    // Parallel (8 indep accumulators per warp).
    let mut par_ms: Vec<f32> = Vec::with_capacity(n_runs);
    for _ in 0..n_runs {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        probe.launch_parallel(&stream, &mut out, &a_in, &b_in, n_iters, n_blocks, block_threads)?;
        end.record(&stream)?;
        stream.synchronize()?;
        par_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    par_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let par_min = par_ms[0];
    let par_total_ops: u64 = (n_warps_total as u64) * (n_iters as u64) * OPS_PER_WMMA * 8;
    let par_tops = (par_total_ops as f64 / 1.0e12) / (par_min as f64 / 1000.0);

    eprintln!("\n=== WMMA f16 throughput on {arch} ===");
    eprintln!("Sequential (1 dep chain):  min {seq_min:.3} ms  realized {seq_tops:.2} TFLOPs");
    eprintln!("Parallel (8 acc/warp):     min {par_min:.3} ms  realized {par_tops:.2} TFLOPs");

    // Per-warp-cycle math:
    // total ops = n_warps_total × n_iters × OPS_PER_WMMA × (1 or 8)
    // ops/cycle/warp = total ops / (n_warps_total × wall_seconds × clock_hz)
    let n_warps_f64 = n_warps_total as f64;
    let clock_hz = max_clock_ghz * 1.0e9;
    let seq_ops_per_cycle_per_warp = (seq_total_ops as f64) / (n_warps_f64 * (seq_min as f64 / 1000.0) * clock_hz);
    let par_ops_per_cycle_per_warp = (par_total_ops as f64) / (n_warps_f64 * (par_min as f64 / 1000.0) * clock_hz);
    eprintln!("\nOps/cycle/warp (assumes warps fully filled, may overstate if oversub):");
    eprintln!("  Sequential: {seq_ops_per_cycle_per_warp:.1} ops/cycle/warp");
    eprintln!("  Parallel:   {par_ops_per_cycle_per_warp:.1} ops/cycle/warp");
    eprintln!("  Ratio (par/seq): {:.2}x (= how much pipelining helps)", par_ops_per_cycle_per_warp / seq_ops_per_cycle_per_warp);

    // f16 peak on RDNA3.5 iGPU is determined by ALU lanes × 2 (fma).
    // Strix Halo: 40 CUs × 64 lanes/CU × 2 fma × clock = peak f16 FMA TFLOPs.
    // (multi_processor_count from HIP reports WGPs = 20, but each WGP has 2 CUs = 40 CUs).
    let cus_actual = n_cus * 2; // HIP reports WGPs; CU count is 2× that
    let lanes_per_cu = 64u32;
    let peak_f16_fma_tops = (cus_actual as f64) * (lanes_per_cu as f64) * 2.0 * clock_hz / 1.0e12;
    eprintln!("\nRDNA3.5 theoretical peak f16 (FMA): {peak_f16_fma_tops:.2} TFLOPs");
    eprintln!("  ({} CUs × {} lanes × 2 fma × {:.2} GHz)", cus_actual, lanes_per_cu, max_clock_ghz);
    eprintln!("Parallel realized vs peak: {:.1}%", par_tops / peak_f16_fma_tops * 100.0);
    Ok(())
}
