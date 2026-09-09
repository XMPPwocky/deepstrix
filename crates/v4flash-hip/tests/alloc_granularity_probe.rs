//! Allocation-granularity probe: how much device memory does each hipMalloc
//! of size S actually consume (per the driver's own accounting)? Answers
//! whether packing the ~1,300 per-tensor dGPU buffers into per-layer
//! arenas would recover VRAM (every dGPU byte is a host byte via hot-expert
//! dedup). Allocates a few tens of MiB; safe beside a live server.
//! Run: cargo test --release -p v4flash-hip --test alloc_granularity_probe -- --ignored --nocapture
use v4flash_hip::{Device, DeviceBuffer};

fn used(card: &str, file: &str) -> u64 {
    std::fs::read_to_string(format!("/sys/class/drm/{card}/device/{file}")).unwrap().trim().parse().unwrap()
}

#[test]
#[ignore]
fn alloc_granularity_probe() -> color_eyre::eyre::Result<()> {
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        let (card, file) = if arch.starts_with("gfx1201") { ("card1", "mem_info_vram_used") } else { ("card2", "mem_info_gtt_used") };
        d.set_current()?;
        eprintln!("== {arch} ({card}/{file})");
        for &sz in &[64usize << 10, 256 << 10, 1 << 20, (1 << 20) + 4096, 3 << 20, (4 << 20) + 4096, 17 << 20] {
            let n = 32;
            let before = used(card, file);
            let mut keep: Vec<DeviceBuffer<u8>> = Vec::new();
            for _ in 0..n { keep.push(DeviceBuffer::new(d.id, sz)?); }
            let after = used(card, file);
            let per = (after.saturating_sub(before)) as f64 / n as f64;
            eprintln!("  size {:>9} B x{n}: driver charged {:>9.0} B each  (overhead {:+.1}%)", sz, per, (per / sz as f64 - 1.0) * 100.0);
            drop(keep);
        }
    }
    Ok(())
}
