//! Standalone iq2 kernel probe — no GGUF, no model load, no mmap.
//! Designed to run cleanly under rocprofv3 PMC.
//!
//! Allocates synthetic dummy buffers (zero-filled iq2 + q8_k blocks
//! produce zero outputs but don't crash the kernel), runs the
//! chunked iq2 kernel N times.
//!
//! Run normally:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_iq2_isolated \
//!     bench_iq2_isolated -- --ignored --nocapture
//!
//! Run under rocprofv3:
//!   nix develop -c bash -c '
//!     export PATH=/nix/store/c9874ja4w6hkfbrv2fsx0r6zplrplwni-rocprofiler-sdk-7.2.3/bin:$PATH
//!     cargo build --release -p v4flash-kernels --test bench_iq2_isolated
//!     TEST_BIN=$(find target/release/deps -name "bench_iq2_isolated-*" -executable -not -name "*.d" | head -1)
//!     rocprofv3 -i /tmp/iq2_counters.txt -d /tmp/iq2_prof -o run -- \
//!       "$TEST_BIN" bench_iq2_isolated --ignored --nocapture
//!   '

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::forward::{
    BLOCKS_Q8K_GATE_IN, N_EXPERT, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::iq2_xxs::Iq2XxsPairMatvec;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

const BLOCK_IQ2_XXS_BYTES: usize = 66;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

#[test]
#[ignore]
fn bench_iq2_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let b: u32 = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let n_work_items_target: u32 = std::env::var("BENCH_WI")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60); // typical for B=64
    let chunk_size: u32 = 16;
    eprintln!("isolated iq2 probe: B={b}, iters={iters}, n_work_items={n_work_items_target}");

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;

    // === SYNTHETIC BUFFERS ===
    // Weights: one expert's worth × N_EXPERT to be safe with addressing.
    // Each expert: N_FF_EXP rows × BLOCKS_Q8K_GATE_IN blocks × 66 bytes.
    let gate_bpe = (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_IQ2_XXS_BYTES;
    let up_bpe = gate_bpe;
    let total_gate_bytes = gate_bpe * (N_EXPERT as usize);
    let total_up_bytes = up_bpe * (N_EXPERT as usize);
    eprintln!(
        "allocating: gate {} MB, up {} MB",
        total_gate_bytes / 1_000_000,
        total_up_bytes / 1_000_000
    );
    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate_bytes)?;
    let mut up_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_up_bytes)?;
    gate_w.fill_zero()?;
    up_w.fill_zero()?;

    // xq: B tokens × q8_k blocks. Zero-filled.
    let xq_bytes_per_token = (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
    let total_xq_bytes = xq_bytes_per_token * (b as usize);
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_xq_bytes)?;
    xq.fill_zero()?;

    // expert_w, group_count, expert_members, work_items, mid.
    let cs_n_used = N_EXPERT_USED as u32;
    let mut expert_w: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (cs_n_used as usize))?;
    expert_w.fill_zero()?;

    let max_per_expert = b; // B_MAX in production. Synthetic just needs >= chunk_size.
    let mut group_count: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items_target as usize)?;

    // Build group_count + expert_members + work_items so n_work_items_target
    // chunks all hit DISTINCT experts (default — exercises real dispatch
    // pattern without favoring L2 reuse). Each expert has chunk_size members.
    let n_distinct = n_work_items_target.min(N_EXPERT) as usize;
    let mut gc_host = vec![0i32; N_EXPERT as usize];
    let mut em_host = vec![0i32; (N_EXPERT as usize) * (max_per_expert as usize)];
    let mut wi_host = vec![0i32; n_work_items_target as usize];
    for i in 0..n_work_items_target as usize {
        let e = i % n_distinct;
        wi_host[i] = ((e as i32) << 16) | 0;
    }
    for e in 0..n_distinct {
        gc_host[e] = chunk_size as i32;
        for i in 0..(chunk_size as usize) {
            let b_idx = i % (b as usize);
            let slot = i % (cs_n_used as usize);
            em_host[e * (max_per_expert as usize) + i] = ((b_idx as i32) << 16) | (slot as i32);
        }
    }
    group_count.copy_from_host(&gc_host)?;
    expert_members.copy_from_host(&em_host)?;
    work_items.copy_from_host(&wi_host)?;

    let mut mid: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id,
        (b as usize) * (cs_n_used as usize) * (N_FF_EXP as usize),
    )?;

    // Warm up the kernel + caches.
    iq2.launch_fused_swiglu_chunked(
        &stream,
        &mut mid,
        &gate_w,
        &up_w,
        &xq,
        &expert_w,
        &group_count,
        &expert_members,
        &work_items,
        gate_bpe as u32,
        up_bpe as u32,
        cs_n_used,
        max_per_expert,
        chunk_size,
        SWIGLU_CLAMP_EXP,
        N_FF_EXP,
        BLOCKS_Q8K_GATE_IN,
        n_work_items_target,
    )?;
    stream.synchronize()?;

    eprintln!("running {iters} iters under timing...");
    let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        iq2.launch_fused_swiglu_chunked(
            &stream,
            &mut mid,
            &gate_w,
            &up_w,
            &xq,
            &expert_w,
            &group_count,
            &expert_members,
            &work_items,
            gate_bpe as u32,
            up_bpe as u32,
            cs_n_used,
            max_per_expert,
            chunk_size,
            SWIGLU_CLAMP_EXP,
            N_FF_EXP,
            BLOCKS_Q8K_GATE_IN,
            n_work_items_target,
        )?;
        end.record(&stream)?;
        stream.synchronize()?;
        walls_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "iq2 isolated: min={:.3} ms  median={:.3} ms  max={:.3} ms",
        walls_ms[0],
        walls_ms[walls_ms.len() / 2],
        walls_ms[walls_ms.len() - 1]
    );
    Ok(())
}
