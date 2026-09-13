//! Measured device ceilings for the V4.1 kernel roofline
//! (docs/v41/KERNEL_ROOFLINE.md). No model load. For each device present
//! (gfx1201 dGPU, gfx1151 iGPU) it sweeps grid sizes and buffer sizes for
//! every access pattern in kernels/device_ceilings.hip and reports the best
//! sustained GB/s together with the pattern + geometry that achieved it,
//! then the compute peaks (dp4a, f32 FMA, packed f16 FMA, f16 WMMA, IU8 WMMA).
//!
//!   HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c \
//!     cargo test --release --features v41 -p v4flash-kernels \
//!     --test bench_device_ceilings -- --ignored --nocapture
//!
//! Env: CEIL_DEVICES=dgpu,igpu (default both), CEIL_MB=256,1024 (buffer
//! sizes, default 256 and 1024 MB on dGPU, 256 and 1024 on iGPU),
//! CEIL_ITERS (default 8), CEIL_JSON=<path> to also emit JSON lines.

use std::io::Write;

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::device_ceilings::DeviceCeilings;
use v4flash_kernels::wmma_probe::WmmaProbe;

fn pick(prefix: &str) -> Option<Device> {
    Device::all().ok()?.into_iter().find(|d| {
        d.properties().map(|p| p.gcn_arch_name.starts_with(prefix)).unwrap_or(false)
    })
}

fn time_ms(stream: &Stream, warmup: usize, iters: usize, mut f: impl FnMut(&Stream) -> eyre::Result<()>) -> eyre::Result<(f32, f32)> {
    for _ in 0..warmup {
        f(stream)?;
    }
    stream.synchronize()?;
    let mut v = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Event::new()?;
        let e = Event::new()?;
        s.record(stream)?;
        f(stream)?;
        e.record(stream)?;
        stream.synchronize()?;
        v.push(Event::elapsed_ms(&s, &e)?);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((v[0], v[v.len() / 2]))
}

struct Out {
    json: Option<std::fs::File>,
}

impl Out {
    fn row(&mut self, dev: &str, metric: &str, pattern: &str, value: f64, unit: &str, note: &str) {
        eprintln!("{dev:<5} {metric:<14} {value:>9.1} {unit:<8} {pattern:<44} {note}");
        if let Some(f) = self.json.as_mut() {
            let _ = writeln!(
                f,
                "{{\"dev\":\"{dev}\",\"metric\":\"{metric}\",\"pattern\":\"{pattern}\",\"value\":{value},\"unit\":\"{unit}\",\"note\":\"{note}\"}}"
            );
        }
    }
}

fn run_device(dev: Device, label: &str, mbs: &[usize], iters: usize, out: &mut Out) -> eyre::Result<()> {
    dev.set_current()?;
    let p = dev.properties()?;
    let arch = p.gcn_arch_name.clone();
    // HIP reports WGPs in multi_processor_count on RDNA; CUs = 2×.
    let wgps = p.multi_processor_count as u32;
    let cus = wgps * 2;
    let clk_ghz = p.clock_rate_khz as f64 / 1e6;
    let mclk_mhz = p.memory_clock_rate_khz as f64 / 1e3;
    let bus = p.memory_bus_width_bits as f64;
    eprintln!(
        "\n=== {label}: {} ({arch}) {wgps} WGPs = {cus} CUs, sclk {clk_ghz:.2} GHz, mclk {mclk_mhz:.0} MHz, bus {bus:.0} b, L2 {} MiB, vram {:.1} GiB, integrated={} ===",
        p.name,
        p.l2_cache_size / (1024 * 1024),
        p.total_global_mem as f64 / (1u64 << 30) as f64,
        p.integrated
    );
    // Nominal DRAM bandwidth from the reported clocks (GDDR6: 16 bits/clk/pin
    // at the reported memory clock ×? HIP reports the *effective* rate/2 on
    // some stacks; print raw so the reader can judge).
    eprintln!("nominal (HIP mclk × bus / 8 × 2): {:.0} GB/s  [only a sanity number]", mclk_mhz * 1e6 * bus / 8.0 * 2.0 / 1e9);

    let stream = Stream::new(dev.id)?;
    let ceil = DeviceCeilings::for_arch(&arch)?;
    let mut sink: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, 64)?;
    sink.fill_zero()?;

    let block = 256u32;
    let grid_mults = [1u32, 2, 4, 8, 16, 32];

    let mut best_read = (0.0f64, String::new());
    let mut best_read_b32 = (0.0f64, String::new());
    let mut best_rows = (0.0f64, String::new());
    let mut best_write = (0.0f64, String::new());
    let mut best_copy = (0.0f64, String::new());

    for &mb in mbs {
        let bytes = mb * 1024 * 1024;
        let mut a: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, bytes)?;
        let mut b: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, bytes)?;
        a.fill_zero()?;
        b.fill_zero()?;
        for &m in &grid_mults {
            let grid = cus * m;
            // read v4
            let (mn, med) = time_ms(&stream, 2, iters, |s| ceil.read_v4(s, &a, &mut sink, grid, block))?;
            let gbs = bytes as f64 / 1e9 / (mn as f64 / 1e3);
            let gbs_med = bytes as f64 / 1e9 / (med as f64 / 1e3);
            let pat = format!("read_v4 {mb}MB grid={grid}x{block}");
            eprintln!("  {pat:<40} min {gbs:7.1} GB/s  p50 {gbs_med:7.1} GB/s");
            if gbs > best_read.0 { best_read = (gbs, pat); }
            // read b32
            let (mn, _) = time_ms(&stream, 2, iters, |s| ceil.read_b32(s, &a, &mut sink, grid, block))?;
            let gbs = bytes as f64 / 1e9 / (mn as f64 / 1e3);
            let pat = format!("read_b32 {mb}MB grid={grid}x{block}");
            eprintln!("  {pat:<40} min {gbs:7.1} GB/s");
            if gbs > best_read_b32.0 { best_read_b32 = (gbs, pat); }
            // write
            let (mn, _) = time_ms(&stream, 2, iters, |s| ceil.write_v4(s, &mut b, grid, block, 7))?;
            let gbs = bytes as f64 / 1e9 / (mn as f64 / 1e3);
            let pat = format!("write_v4 {mb}MB grid={grid}x{block}");
            eprintln!("  {pat:<40} min {gbs:7.1} GB/s");
            if gbs > best_write.0 { best_write = (gbs, pat); }
            // copy (count both directions)
            let (mn, _) = time_ms(&stream, 2, iters, |s| ceil.copy_v4(s, &a, &mut b, grid, block))?;
            let gbs = 2.0 * bytes as f64 / 1e9 / (mn as f64 / 1e3);
            let pat = format!("copy_v4 {mb}MB grid={grid}x{block}");
            eprintln!("  {pat:<40} min {gbs:7.1} GB/s (r+w)");
            if gbs > best_copy.0 { best_copy = (gbs, pat); }
        }
        // wave-per-row: row widths that match our matvecs (Q8_0 rows of
        // K=5120 → 5440 B; K=1280 → 1360 B; K=8192 → 8704 B; MXFP4 K=5120 → 2720 B).
        for &row_bytes in &[1360usize, 2720, 5440, 8704, 65536] {
            let n_rows = (bytes / row_bytes) as u32;
            let (mn, _) = time_ms(&stream, 2, iters, |s| ceil.read_rows(s, &a, &mut sink, row_bytes, n_rows))?;
            let used = n_rows as f64 * row_bytes as f64;
            let gbs = used / 1e9 / (mn as f64 / 1e3);
            let pat = format!("read_rows {mb}MB row={row_bytes}B ({n_rows} rows)");
            eprintln!("  {pat:<40} min {gbs:7.1} GB/s");
            if gbs > best_rows.0 { best_rows = (gbs, pat); }
        }
    }
    out.row(label, "bw_read", &best_read.1, best_read.0, "GB/s", "streaming read, 16B/lane, 4 in flight");
    out.row(label, "bw_read_b32", &best_read_b32.1, best_read_b32.0, "GB/s", "streaming read, 4B/lane");
    out.row(label, "bw_read_rows", &best_rows.1, best_rows.0, "GB/s", "wave-per-row (matvec pattern)");
    out.row(label, "bw_write", &best_write.1, best_write.0, "GB/s", "streaming write, 16B/lane");
    out.row(label, "bw_copy", &best_copy.1, best_copy.0, "GB/s", "read+write counted both ways");

    // ---- compute ceilings ----
    let n_iters: u32 = 4000;
    let mut best_dp4a = (0.0f64, String::new());
    {
        let mut a_in: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, 8)?;
        let mut b_in: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, 8)?;
        a_in.copy_from_host(&[0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10, 1, 2, 3, 4])?;
        b_in.copy_from_host(&[0x11121314, 0x15161718, 0x191a1b1c, 0x1d1e1f20, 5, 6, 7, 8])?;
        for &(wpb, m) in &[(4u32, 2u32), (4, 4), (4, 8), (8, 4), (8, 8), (16, 4)] {
            let grid = cus * m;
            let bt = wpb * 32;
            let mut o: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, (grid * bt) as usize)?;
            let (mn, _) = time_ms(&stream, 1, 5, |s| ceil.dp4a(s, &mut o, &a_in, &b_in, n_iters, grid, bt))?;
            let ops = grid as f64 * bt as f64 * n_iters as f64 * 64.0;
            let tops = ops / 1e12 / (mn as f64 / 1e3);
            let pat = format!("dp4a grid={grid}x{bt}");
            eprintln!("  {pat:<40} {tops:7.2} TOPS");
            if tops > best_dp4a.0 { best_dp4a = (tops, pat); }
        }
    }
    out.row(label, "dp4a", &best_dp4a.1, best_dp4a.0, "TOPS", "v_dot4_i32_i8, 8 acc/lane, reg operands");

    // WMMA / FMA peaks via the existing probe.
    let probe = WmmaProbe::for_arch(&arch)?;
    let a_h: Vec<u16> = vec![0x3c00u16; 16];
    let mut a16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, 16)?;
    let mut b16: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, 16)?;
    a16.copy_from_host(&a_h)?;
    b16.copy_from_host(&a_h)?;
    let mut a8: DeviceBuffer<i8> = DeviceBuffer::new(dev.id, 16)?;
    let mut b8: DeviceBuffer<i8> = DeviceBuffer::new(dev.id, 16)?;
    a8.copy_from_host(&[1i8; 16])?;
    b8.copy_from_host(&[1i8; 16])?;
    let mut a32: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 1)?;
    let mut b32: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, 1)?;
    a32.copy_from_host(&[1.0001f32])?;
    b32.copy_from_host(&[0.9999f32])?;
    let mut best_f16 = (0.0f64, String::new());
    let mut best_iu8 = (0.0f64, String::new());
    let mut best_f32 = (0.0f64, String::new());
    let mut best_f16x2 = (0.0f64, String::new());
    for &(wpb, m) in &[(4u32, 2u32), (4, 4), (8, 2), (8, 4), (16, 2)] {
        let grid = cus * m;
        let bt = wpb * 32;
        let warps = (grid * wpb) as f64;
        let mut o: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, (grid * wpb * 8) as usize)?;
        let mut oi: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, (grid * wpb * 8) as usize)?;
        let mut of: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, (grid * wpb) as usize)?;
        let (mn, _) = time_ms(&stream, 1, 5, |s| probe.launch_parallel(s, &mut o, &a16, &b16, n_iters, grid, bt))?;
        let t = warps * n_iters as f64 * 8192.0 * 8.0 / 1e12 / (mn as f64 / 1e3);
        if t > best_f16.0 { best_f16 = (t, format!("wmma_f16 grid={grid}x{bt}")); }
        let (mn, _) = time_ms(&stream, 1, 5, |s| probe.launch_iu8_parallel(s, &mut oi, &a8, &b8, n_iters, grid, bt))?;
        let t = warps * n_iters as f64 * 8192.0 * 8.0 / 1e12 / (mn as f64 / 1e3);
        if t > best_iu8.0 { best_iu8 = (t, format!("wmma_iu8 grid={grid}x{bt}")); }
        let (mn, _) = time_ms(&stream, 1, 5, |s| probe.launch_fma_f32(s, &mut of, &a32, &b32, n_iters, grid, bt))?;
        let t = warps * 32.0 * n_iters as f64 * 32.0 * 2.0 / 1e12 / (mn as f64 / 1e3);
        if t > best_f32.0 { best_f32 = (t, format!("fma_f32 grid={grid}x{bt}")); }
        let (mn, _) = time_ms(&stream, 1, 5, |s| probe.launch_fma_f16x2(s, &mut of, &a16, &b16, n_iters, grid, bt))?;
        let t = warps * 32.0 * n_iters as f64 * 32.0 * 2.0 * 2.0 / 1e12 / (mn as f64 / 1e3);
        if t > best_f16x2.0 { best_f16x2 = (t, format!("fma_f16x2 grid={grid}x{bt}")); }
    }
    out.row(label, "wmma_f16", &best_f16.1, best_f16.0, "TFLOPS", "16x16x16 f16->f32, 8 independent acc/wave");
    out.row(label, "wmma_iu8", &best_iu8.1, best_iu8.0, "TOPS", "16x16x16 i8->i32, 8 independent acc/wave");
    out.row(label, "fma_f32", &best_f32.1, best_f32.0, "TFLOPS", "v_fma_f32, 32 acc/lane");
    out.row(label, "fma_f16x2", &best_f16x2.1, best_f16x2.0, "TFLOPS", "v_pk_fma_f16, 32 acc/lane");
    Ok(())
}

#[test]
#[ignore]
fn bench_device_ceilings() -> eyre::Result<()> {
    install_panic_handler()?;
    let which = std::env::var("CEIL_DEVICES").unwrap_or_else(|_| "dgpu,igpu".into());
    let iters: usize = std::env::var("CEIL_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let mbs: Vec<usize> = std::env::var("CEIL_MB")
        .ok()
        .map(|s| s.split(',').filter_map(|t| t.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![256, 1024]);
    let json = std::env::var("CEIL_JSON").ok().map(|p| std::fs::File::create(p)).transpose()?;
    let mut out = Out { json };
    eprintln!("{:<5} {:<14} {:>9} {:<8} {:<44} note", "dev", "metric", "value", "unit", "pattern");
    if which.contains("dgpu") {
        match pick("gfx1201") {
            Some(d) => run_device(d, "dgpu", &mbs, iters, &mut out)?,
            None => eprintln!("no gfx1201 dGPU visible"),
        }
    }
    if which.contains("igpu") {
        match pick("gfx1151") {
            Some(d) => run_device(d, "igpu", &mbs, iters, &mut out)?,
            None => eprintln!("no gfx1151 iGPU visible"),
        }
    }
    let _ = eyre!("unused");
    Ok(())
}
