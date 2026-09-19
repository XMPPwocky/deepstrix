//! Isolated B=1 decode MoE bench: `mxfp4_pair_matvec_fused_swiglu_batch`.
//!
//! WHY THIS EXISTS. Decode is iGPU-bound and `igpu.routed_moe` measured
//! 160-390 ms/token, against a 11.4 ms bandwidth roofline for box 1's share.
//! But a *stage* timing folds in stream idle (the host's SSD paging between
//! layers), so it cannot answer "how fast is the kernel". `rocprofv3` would,
//! but this nix build aborts on a gflags conflict. So: launch the kernel alone,
//! on synthetic weights, and time it with HIP events.
//!
//! At B=1 this kernel is PURELY bandwidth-bound — it reads gate+up for
//! `n_used` experts (12.5 MB each) and does ~2.8 MAC/byte. The only figure of
//! merit is achieved GB/s against the iGPU's ~256 GB/s peak.
//!
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release --features v41 \
//!     -p v4flash-kernels --test bench_mxfp4_moe_b1 -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{
    BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::mxfp4_pair::Mxfp4PairMatvec;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

/// ggml MXFP4: 32 elements per 17-byte block (16 nibbles + 1 E8M0 scale).
const MXFP4_BLOCK_ELEMS: usize = 32;
const MXFP4_BLOCK_BYTES: usize = 17;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (iGPU) visible"))
}

#[test]
#[ignore]
fn bench_mxfp4_moe_b1() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_igpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let stream = Stream::new(dev.id)?;
    let k = Mxfp4PairMatvec::for_arch(&arch)?;

    // BENCH_N_USED: slots per launch. Production box 2 serves ~3.6 of the 6 picks
    // under the hash partition, so 3-4 is the representative shape, not 6.
    let n_used: u32 = std::env::var("BENCH_N_USED").ok().and_then(|v| v.parse().ok()).unwrap_or(N_EXPERT_USED as u32);
    let n_rows = N_FF_EXP;            // 2304
    let n_blocks = BLOCKS_Q8K_GATE_IN; // N_EMBD/256 = 20

    // One expert's gate (or up) matrix: n_rows x N_EMBD at MXFP4.
    let bpe = n_rows as usize * (N_EMBD as usize / MXFP4_BLOCK_ELEMS) * MXFP4_BLOCK_BYTES;
    // Distinct experts so nothing is served from cache across slots. And
    // BENCH_N_ALLOC (default 24) allocates MORE experts than a launch uses so
    // the selection can rotate every iteration: a fixed 3-slot selection is
    // 37.6 MB re-read 200 times, and the 32 MB last-level cache serves part of
    // it (measured 238 GB/s at 3 slots vs 198 at 6 with a fixed selection).
    let n_expert_alloc: usize = std::env::var("BENCH_N_ALLOC").ok().and_then(|v| v.parse().ok()).unwrap_or(24).max(n_used as usize);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|v| v.parse().ok()).unwrap_or(20);

    let gate = DeviceBuffer::<u8>::new(dev.id, n_expert_alloc * bpe)?;
    let up = DeviceBuffer::<u8>::new(dev.id, n_expert_alloc * bpe)?;
    let xq = DeviceBuffer::<u8>::new(dev.id, n_blocks as usize * BLOCK_Q8_K_BYTES)?;
    let mut mid = DeviceBuffer::<f32>::new(dev.id, (n_used * n_rows) as usize)?;
    let mut ew = DeviceBuffer::<f32>::new(dev.id, n_used as usize)?;
    let mut sel = DeviceBuffer::<i32>::new(dev.id, n_used as usize)?;
    ew.copy_from_host(&vec![1.0f32; n_used as usize])?;
    sel.copy_from_host(&(0..n_used as i32).collect::<Vec<_>>())?;

    let bytes_read = 2usize * n_used as usize * bpe; // gate + up, per launch
    println!(
        "B=1 MoE: n_used={n_used} n_rows={n_rows} n_blocks={n_blocks} \
         bpe={:.2} MB  bytes/launch={:.2} MB",
        bpe as f64 / 1e6,
        bytes_read as f64 / 1e6
    );
    println!("grid = ({}, {}), block = 256  -> {} workgroups, 1 warp per output row",
             n_rows / 8, n_used, (n_rows / 8) * n_used);

    for _ in 0..warmup {
        k.launch_fused_swiglu_batch(&stream, &mut mid, &gate, &up, &xq, &ew, &sel,
            bpe as u32, bpe as u32, n_used, SWIGLU_CLAMP_EXP, n_rows, n_blocks)?;
    }
    stream.synchronize()?;

    let mut us: Vec<f32> = Vec::with_capacity(iters);
    for it in 0..iters {
        // Rotate the selection so consecutive launches touch different experts.
        let sel_host: Vec<i32> = (0..n_used as usize).map(|i| ((it * n_used as usize + i) % n_expert_alloc) as i32).collect();
        sel.copy_from_host(&sel_host)?;
        let a = Event::new()?;
        let b = Event::new()?;
        a.record(&stream)?;
        k.launch_fused_swiglu_batch(&stream, &mut mid, &gate, &up, &xq, &ew, &sel,
            bpe as u32, bpe as u32, n_used, SWIGLU_CLAMP_EXP, n_rows, n_blocks)?;
        b.record(&stream)?;
        stream.synchronize()?;
        us.push(Event::elapsed_ms(&a, &b)? * 1000.0);
    }
    us.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let p50 = us[us.len() / 2];
    let p10 = us[us.len() / 10];
    let gbps = |t_us: f32| bytes_read as f64 / (t_us as f64 * 1e-6) / 1e9;

    println!();
    println!("  p10 {p10:8.1} us   {:6.1} GB/s", gbps(p10));
    println!("  p50 {p50:8.1} us   {:6.1} GB/s   <-- headline", gbps(p50));
    println!();
    println!("  iGPU peak ~256 GB/s  =>  roofline {:.1} us for this launch",
             bytes_read as f64 / 256e9 * 1e6);
    println!("  measured is {:.1}x off roofline", p50 as f64 / (bytes_read as f64 / 256e9 * 1e6));
    println!();
    println!("  per DECODE TOKEN (40 layers): {:.1} ms", p50 * 40.0 / 1000.0);
    Ok(())
}
