//! Bench all 4 kernels of the decode-path iGPU MoE pipeline individually
//! so we know where the 225 µs/layer is actually spent:
//!   1. q8k_quantize (input)         — N_EMBD f32 → Q8_K blocks
//!   2. iq2_fused_swiglu_batch        — gate+up matvec + swiglu fusion
//!   3. q8k_quantize (mid)           — n_used × n_rows f32 → Q8_K blocks
//!   4. q2_k_matvec_par_batched       — down projection (sum over experts)
//!
//! Run normally:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_moe_chain_isolated \
//!     bench_moe_chain_isolated -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::forward::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP,
    SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::iq2_xxs::Iq2XxsPairMatvec;
use v4flash_kernels::q2_k::Q2KAccumulateMatvec;
use v4flash_kernels::q8_k::{Q8KQuantize, BLOCK_Q8_K_BYTES};

const BLOCK_IQ2_XXS_BYTES: usize = 66;
const BLOCK_Q2_K_BYTES: usize = 84;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn percentile(xs_sorted: &[f32], p: f32) -> f32 {
    if xs_sorted.is_empty() { return 0.0; }
    let k = ((xs_sorted.len() - 1) as f32 * p / 100.0).round() as usize;
    xs_sorted[k.min(xs_sorted.len() - 1)]
}

fn timed<F: FnMut(&Stream) -> eyre::Result<()>>(
    name: &str,
    stream: &Stream,
    iters: usize,
    warmup: usize,
    mut f: F,
) -> eyre::Result<()> {
    for _ in 0..warmup { f(stream)?; }
    stream.synchronize()?;
    let mut walls: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let s = Event::new()?;
        let e = Event::new()?;
        s.record(stream)?;
        f(stream)?;
        e.record(stream)?;
        stream.synchronize()?;
        walls.push(Event::elapsed_ms(&s, &e)?);
    }
    walls.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = walls[0];
    let p50 = percentile(&walls, 50.0);
    let p90 = percentile(&walls, 90.0);
    let mean: f32 = walls.iter().sum::<f32>() / walls.len() as f32;
    eprintln!(
        "{:<30}  min={:>7.1}  p50={:>7.1}  p90={:>7.1}  mean={:>7.1}  µs",
        name, min * 1000.0, p50 * 1000.0, p90 * 1000.0, mean * 1000.0
    );
    Ok(())
}

#[test]
#[ignore]
fn bench_moe_chain_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);

    let n_rows  = N_FF_EXP;       // 2048
    let n_used  = N_EXPERT_USED as u32; // 6
    let n_embd  = N_EMBD;         // 4096
    let n_expert = N_EXPERT as u32;
    let n_blocks_gate = BLOCKS_Q8K_GATE_IN;     // 16
    let n_blocks_down = BLOCKS_Q8K_DOWN_IN;     // 8
    let gate_bpe = (n_rows as usize) * (n_blocks_gate as usize) * BLOCK_IQ2_XXS_BYTES;
    let dbpe     = (n_embd as usize) * (n_blocks_down as usize) * BLOCK_Q2_K_BYTES;

    eprintln!("=== iGPU MoE decode-chain probe ===");
    eprintln!("n_rows={n_rows} n_used={n_used} n_embd={n_embd} n_expert={n_expert}");
    eprintln!("n_blocks_gate={n_blocks_gate} n_blocks_down={n_blocks_down}");
    eprintln!("gate_bpe={} KiB/expert  dbpe={} KiB/expert", gate_bpe/1024, dbpe/1024);
    eprintln!("total iq2 (gate+up) weights: {} MiB", (n_expert as usize) * gate_bpe * 2 / 1024 / 1024);
    eprintln!("total q2k (down) weights:    {} MiB", (n_expert as usize) * dbpe / 1024 / 1024);
    eprintln!();

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;

    let q8k = Q8KQuantize::for_arch(&arch)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;

    // Buffers (all zero-init — kernels handle zero data fine, just slower
    // possibly on cold pages).
    let mut x_in: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, n_embd as usize)?; x_in.fill_zero()?;
    let mut d_xq_q8k: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (n_blocks_gate as usize) * BLOCK_Q8_K_BYTES)?;
    d_xq_q8k.fill_zero()?;
    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (n_expert as usize) * gate_bpe)?; gate_w.fill_zero()?;
    let mut up_w: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (n_expert as usize) * gate_bpe)?; up_w.fill_zero()?;
    let mut d_mid_cat: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, (n_used as usize) * (n_rows as usize))?; d_mid_cat.fill_zero()?;
    let mut expert_w: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, n_used as usize)?; expert_w.fill_zero()?;
    let selected_host: Vec<i32> = (0..n_used as i32).collect();
    let mut selected: DeviceBuffer<i32> = DeviceBuffer::new(
        igpu.id, n_used as usize)?;
    selected.copy_from_host(&selected_host)?;

    let n_blocks_mid_total = (n_used as usize) * (n_blocks_down as usize);
    let mut d_midq_cat: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, n_blocks_mid_total * BLOCK_Q8_K_BYTES)?;
    d_midq_cat.fill_zero()?;
    let mut down_w: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (n_expert as usize) * dbpe)?;
    down_w.fill_zero()?;
    let mut ffn_moe: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, n_embd as usize)?; ffn_moe.fill_zero()?;

    // Warmup all once
    {
        q8k.launch(&stream, &mut d_xq_q8k, &x_in, n_blocks_gate)?;
        iq2.launch_fused_swiglu_batch(&stream, &mut d_mid_cat, &gate_w, &up_w,
            &d_xq_q8k, &expert_w, &selected,
            gate_bpe as u32, gate_bpe as u32, n_used, SWIGLU_CLAMP_EXP,
            n_rows, n_blocks_gate)?;
        // For q8k of mid: caller does it per-slot in one big buffer.
        // We just bench one slot's worth here; loop over n_used inside the f.
        for s in 0..n_used as usize {
            q8k.launch(&stream, &mut d_midq_cat, &d_mid_cat, n_blocks_down)?;
            // ^^^ ignoring offsets, just calling launch repeatedly
            let _ = s;
        }
        q2k.launch_batched(&stream, &mut ffn_moe, &down_w, &d_midq_cat, &selected,
            dbpe as u32,
            (n_blocks_down as u32) * (BLOCK_Q8_K_BYTES as u32),
            n_used, n_embd, n_blocks_down)?;
        stream.synchronize()?;
    }

    // Individual kernel timings
    timed("1. q8k_quantize input (16 blk)", &stream, iters, warmup, |s| {
        q8k.launch(s, &mut d_xq_q8k, &x_in, n_blocks_gate)?;
        Ok(())
    })?;

    timed("2. iq2_fused_swiglu_batch", &stream, iters, warmup, |s| {
        iq2.launch_fused_swiglu_batch(s, &mut d_mid_cat, &gate_w, &up_w,
            &d_xq_q8k, &expert_w, &selected,
            gate_bpe as u32, gate_bpe as u32, n_used, SWIGLU_CLAMP_EXP,
            n_rows, n_blocks_gate)?;
        Ok(())
    })?;

    // q8k of mid: production code calls it ONCE for all n_used × n_blocks_down
    // blocks via a concatenated launch. n_blocks=48 for V4-Flash.
    let n_blocks_mid = (n_used as u32) * n_blocks_down;
    timed(&format!("3. q8k_quantize mid ({} blk)", n_blocks_mid),
          &stream, iters, warmup, |s| {
        q8k.launch(s, &mut d_midq_cat, &d_mid_cat, n_blocks_mid)?;
        Ok(())
    })?;

    timed("4. q2_k_matvec_par_batched", &stream, iters, warmup, |s| {
        q2k.launch_batched(s, &mut ffn_moe, &down_w, &d_midq_cat, &selected,
            dbpe as u32,
            (n_blocks_down as u32) * (BLOCK_Q8_K_BYTES as u32),
            n_used, n_embd, n_blocks_down)?;
        Ok(())
    })?;

    // Full chain back-to-back (single stream sync per iter)
    timed("FULL CHAIN (1+2+3+4)", &stream, iters, warmup, |s| {
        q8k.launch(s, &mut d_xq_q8k, &x_in, n_blocks_gate)?;
        iq2.launch_fused_swiglu_batch(s, &mut d_mid_cat, &gate_w, &up_w,
            &d_xq_q8k, &expert_w, &selected,
            gate_bpe as u32, gate_bpe as u32, n_used, SWIGLU_CLAMP_EXP,
            n_rows, n_blocks_gate)?;
        q8k.launch(s, &mut d_midq_cat, &d_mid_cat, n_blocks_mid)?;
        q2k.launch_batched(s, &mut ffn_moe, &down_w, &d_midq_cat, &selected,
            dbpe as u32,
            (n_blocks_down as u32) * (BLOCK_Q8_K_BYTES as u32),
            n_used, n_embd, n_blocks_down)?;
        Ok(())
    })?;

    // BW summary for the big ones
    let iq2_bytes = (n_used as u64) * 2 * (n_rows as u64) * (n_blocks_gate as u64) * (BLOCK_IQ2_XXS_BYTES as u64);
    let q2k_bytes = (n_used as u64) * (n_embd as u64) * (n_blocks_down as u64) * (BLOCK_Q2_K_BYTES as u64);
    eprintln!();
    eprintln!("=== weight BW reference (per call) ===");
    eprintln!("iq2 gate+up weights: {} MiB", iq2_bytes / 1024 / 1024);
    eprintln!("q2k down weights:    {} MiB", q2k_bytes / 1024 / 1024);
    Ok(())
}
