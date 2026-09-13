//! Does the 9070 XT's 64 MB infinity cache (MALL) retain a weight matrix
//! across back-to-back matvec launches? If a 32–48 MB Q8_0 weight re-reads
//! at well above the ~640 GB/s DRAM rate, then prefetching the next layer's
//! attention weights during the expert wait (PLAN.md §7e) is a real lever.
//!   cargo test -p v4flash-kernels --release --test bench_mall_retention -- --ignored --nocapture
use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::Q8_0Matvec;

fn pick(prefix: &str) -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with(prefix) {
            return Ok(d);
        }
    }
    Err(eyre!("no {prefix}"))
}

#[test]
#[ignore]
fn mall_retention_vs_weight_size() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick("gfx1201")?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let stream = Stream::new(dev.id)?;
    let k = Q8_0Matvec::for_arch(&arch)?;
    let kdim = 5120u32;
    let row_bytes = (kdim / 32 * 34) as usize;
    let mut xq: DeviceBuffer<i8> = DeviceBuffer::new(dev.id, kdim as usize)?;
    xq.copy_from_host(&vec![3i8; kdim as usize])?;
    let mut xs: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, (kdim / 32) as usize)?;
    xs.copy_from_host(&vec![0.01f32; (kdim / 32) as usize])?;
    // Evict buffer: 256 MB read between measurements to flush the MALL.
    let evict_rows = (256usize << 20) / row_bytes;
    let mut evict: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, evict_rows * row_bytes)?;
    evict.copy_from_host(&vec![1u8; evict_rows * row_bytes])?;
    let mut out_evict: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, evict_rows)?;
    for mb in [8usize, 16, 32, 48, 56, 64, 96, 160] {
        let rows = (mb << 20) / row_bytes;
        let mut w: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, rows * row_bytes)?;
        w.copy_from_host(&vec![2u8; rows * row_bytes])?;
        let mut out: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, rows)?;
        // cold: evict, then one launch
        let mut cold = 0f64;
        let mut warm = 0f64;
        let reps = 10;
        for _ in 0..reps {
            k.matvec(&stream, &mut out_evict, &evict, &xq, &xs, evict_rows as u32, kdim)?;
            stream.synchronize()?;
            let t0 = std::time::Instant::now();
            k.matvec(&stream, &mut out, &w, &xq, &xs, rows as u32, kdim)?;
            stream.synchronize()?;
            cold += t0.elapsed().as_secs_f64();
            // warm: same weight again immediately (5 launches, average)
            let t1 = std::time::Instant::now();
            for _ in 0..5 {
                k.matvec(&stream, &mut out, &w, &xq, &xs, rows as u32, kdim)?;
            }
            stream.synchronize()?;
            warm += t1.elapsed().as_secs_f64() / 5.0;
        }
        let bytes = (rows * row_bytes) as f64;
        eprintln!(
            "{mb:>4} MB weight: cold {:.3} ms = {:>5.0} GB/s | warm {:.3} ms = {:>5.0} GB/s | ratio {:.2}x",
            cold / reps as f64 * 1e3, bytes / (cold / reps as f64) / 1e9,
            warm / reps as f64 * 1e3, bytes / (warm / reps as f64) / 1e9,
            cold / warm
        );
    }
    Ok(())
}
