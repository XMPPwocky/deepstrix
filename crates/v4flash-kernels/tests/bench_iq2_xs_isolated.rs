//! Standalone IQ2_XS / IQ3_XXS-pair gate+up kernel probe — no GGUF, no
//! model load. Clone of bench_iq2_isolated for the unsloth UD-Q2_K_XL
//! pair kernels (chunked vs kwide).
//!
//! Env:
//!   BENCH_FMT     = iq2_xs (default) | iq3
//!   BENCH_VARIANT = 0 chunked (serial per-member) | 6 kwide
//!   BENCH_B, BENCH_ITERS, BENCH_WI, BENCH_CHUNK as in bench_iq2_isolated.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_iq2_xs_isolated \
//!     bench_iq2_xs_isolated -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{
    BLOCKS_Q8K_GATE_IN, N_EXPERT, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::iq2_xs::{Iq2XsPairMatvec, BLOCK_IQ2_XS_BYTES};
use v4flash_kernels::iq3_xxs_pair::Iq3XxsPairMatvec;
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

const BLOCK_IQ3_XXS_BYTES: usize = 98;

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
fn bench_iq2_xs_isolated() -> eyre::Result<()> {
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
        .unwrap_or(60);
    let chunk_size: u32 = std::env::var("BENCH_CHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let variant: u32 = std::env::var("BENCH_VARIANT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let fmt = std::env::var("BENCH_FMT").unwrap_or_else(|_| "iq2_xs".into());
    let use_kwide = variant == 6;
    let block_bytes = if fmt == "iq3" { BLOCK_IQ3_XXS_BYTES } else { BLOCK_IQ2_XS_BYTES };
    eprintln!("fmt={fmt} variant={variant} (0=chunked, 6=kwide)");
    eprintln!("isolated probe: B={b}, iters={iters}, n_work_items={n_work_items_target}, chunk={chunk_size}");

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2xs = Iq2XsPairMatvec::for_arch(&arch)?;
    let iq3p = Iq3XxsPairMatvec::for_arch(&arch)?;

    let gate_bpe = (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * block_bytes;
    let up_bpe = gate_bpe;
    let total_gate_bytes = gate_bpe * (N_EXPERT as usize);
    eprintln!("allocating: gate+up 2 x {} MB", total_gate_bytes / 1_000_000);
    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate_bytes)?;
    let mut up_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate_bytes)?;
    gate_w.fill_zero()?;
    up_w.fill_zero()?;

    let xq_bytes_per_token = (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
    let mut xq: DeviceBuffer<u8> =
        DeviceBuffer::new(igpu.id, xq_bytes_per_token * (b as usize))?;
    xq.fill_zero()?;

    let cs_n_used = N_EXPERT_USED as u32;
    let mut expert_w: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (cs_n_used as usize))?;
    expert_w.fill_zero()?;

    let max_per_expert = b;
    let mut group_count: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, n_work_items_target as usize)?;

    // Distinct experts per work item (same pattern as bench_iq2_isolated).
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

    let launch = |mid: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
        match (fmt.as_str(), use_kwide) {
            ("iq3", true) => iq3p.launch_fused_swiglu_kwide(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            ("iq3", false) => iq3p.launch_fused_swiglu_chunked(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            (_, true) => iq2xs.launch_fused_swiglu_kwide(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            (_, false) => iq2xs.launch_fused_swiglu_chunked(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
        }
    };

    launch(&mut mid)?;
    stream.synchronize()?;

    eprintln!("running {iters} iters under timing...");
    let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        launch(&mut mid)?;
        end.record(&stream)?;
        stream.synchronize()?;
        walls_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eprintln!(
        "{fmt} isolated: min={:.3} ms  median={:.3} ms  max={:.3} ms",
        walls_ms[0],
        walls_ms[walls_ms.len() / 2],
        walls_ms[walls_ms.len() - 1]
    );
    Ok(())
}
