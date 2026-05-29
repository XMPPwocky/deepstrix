//! Isolated bench for the DECODE (single-token) iq2 kernel
//! `iq2_xxs_pair_matvec_fused_swiglu_batch`. No model load, no GGUF.
//! Designed to run cleanly under rocprofv3 ATT.
//!
//! Run normally:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test bench_iq2_decode_isolated \
//!     bench_iq2_decode_isolated -- --ignored --nocapture
//!
//! Run under ATT (after building once):
//!   nix develop -c bash -c '
//!     export PATH=/nix/store/c9874ja4w6hkfbrv2fsx0r6zplrplwni-rocprofiler-sdk-7.2.3/bin:$PATH
//!     cargo build --release -p v4flash-kernels --test bench_iq2_decode_isolated
//!     TEST_BIN=$(find target/release/deps -name "bench_iq2_decode_isolated-*" -executable -not -name "*.d" | head -1)
//!     BENCH_ITERS=2 BENCH_WARMUP=1 rocprofv3 --att \
//!       --att-library-path /nix/store/qzy5bk596ljy2nlj9ig4pynf8qj0mprm-rocprof-trace-decoder-0.1.7/lib \
//!       --att-target-cu 0 --att-shader-engine-mask 0x1 \
//!       --att-consecutive-kernels 1 \
//!       -d /tmp/att_iq2_decode -o run \
//!       --kernel-include-regex "iq2_xxs_pair_matvec_fused_swiglu_batch$" -- \
//!       "$TEST_BIN" bench_iq2_decode_isolated --ignored --nocapture
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

fn percentile(xs_sorted: &[f32], p: f32) -> f32 {
    if xs_sorted.is_empty() { return 0.0; }
    let k = ((xs_sorted.len() - 1) as f32 * p / 100.0).round() as usize;
    xs_sorted[k.min(xs_sorted.len() - 1)]
}

#[test]
#[ignore]
fn bench_iq2_decode_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(100);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    // BENCH_VARIANT: 0=batch (block=256, 8 rows/WG), 1=b512 (block=512, 16 rows/WG)
    let variant: u32 = std::env::var("BENCH_VARIANT")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(0);
    eprintln!("variant={variant} (0=batch/block256, 1=b512/block512)");

    let n_rows  = N_FF_EXP;
    let n_used  = N_EXPERT_USED as u32;
    let n_blocks = BLOCKS_Q8K_GATE_IN;
    let n_expert = N_EXPERT as u32;
    let gate_bpe = (n_rows as usize) * (n_blocks as usize) * BLOCK_IQ2_XXS_BYTES;
    let up_bpe   = gate_bpe;

    eprintln!(
        "iq2 decode probe: n_rows={n_rows} n_used={n_used} n_blocks={n_blocks} \
         n_expert={n_expert} gate_bpe={gate_bpe} iters={iters} warmup={warmup}"
    );

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2 = Iq2XxsPairMatvec::for_arch(&arch)?;

    // gate_w, up_w: full expert pool (n_expert × n_rows × n_blocks × 66 bytes)
    // For N_EXPERT=256, N_FF_EXP=2048, n_blocks=16, BLOCK_IQ2_XXS_BYTES=66
    //   per-expert: 2048 * 16 * 66 = 2,162,688 bytes ≈ 2 MiB
    //   total: 256 * 2 MiB ≈ 540 MiB per matrix, × 2 for gate+up = 1.1 GiB
    let total_bytes = (n_expert as usize) * gate_bpe;
    eprintln!("allocating gate+up weights: 2 × {} MiB", total_bytes / 1024 / 1024);
    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_bytes)?;
    gate_w.fill_zero()?;
    let mut up_w:   DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_bytes)?;
    up_w.fill_zero()?;

    // xq: n_blocks × BLOCK_Q8_K_BYTES (one token's quantized activations)
    let xq_bytes = (n_blocks as usize) * BLOCK_Q8_K_BYTES;
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, xq_bytes)?;
    xq.fill_zero()?;

    // expert_w: per-slot expert weight (router output for selected experts)
    let mut expert_w: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_used as usize)?;
    expert_w.fill_zero()?;

    // selected: which expert per slot — use distinct indices to avoid weight reuse
    let selected_host: Vec<i32> = (0..n_used as i32).collect();
    let mut selected: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used as usize)?;
    selected.copy_from_host(&selected_host)?;

    // mid output: [n_used, n_rows]
    let mut mid: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, (n_used as usize) * (n_rows as usize))?;
    mid.fill_zero()?;

    let mut launch = |stream: &Stream, mid: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
        if variant == 1 {
            iq2.launch_fused_swiglu_batch_b512(
                stream, mid, &gate_w, &up_w, &xq, &expert_w, &selected,
                gate_bpe as u32, up_bpe as u32, n_used, SWIGLU_CLAMP_EXP,
                n_rows, n_blocks,
            )?;
        } else {
            iq2.launch_fused_swiglu_batch(
                stream, mid, &gate_w, &up_w, &xq, &expert_w, &selected,
                gate_bpe as u32, up_bpe as u32, n_used, SWIGLU_CLAMP_EXP,
                n_rows, n_blocks,
            )?;
        }
        Ok(())
    };

    for _ in 0..warmup { launch(&stream, &mut mid)?; }
    stream.synchronize()?;

    eprintln!("running {iters} timed iters...");
    let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
    for _ in 0..iters {
        let start = Event::new()?;
        let end = Event::new()?;
        start.record(&stream)?;
        launch(&stream, &mut mid)?;
        end.record(&stream)?;
        stream.synchronize()?;
        walls_ms.push(Event::elapsed_ms(&start, &end)?);
    }
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let min = walls_ms[0];
    let p50 = percentile(&walls_ms, 50.0);
    let p90 = percentile(&walls_ms, 90.0);
    let p99 = percentile(&walls_ms, 99.0);
    let max = walls_ms[walls_ms.len() - 1];
    let mean: f32 = walls_ms.iter().sum::<f32>() / walls_ms.len() as f32;
    eprintln!(
        "iq2 decode: min={:.3} mean={:.3} p50={:.3} p90={:.3} p99={:.3} max={:.3} (ms)",
        min, mean, p50, p90, p99, max,
    );
    // Bytes loaded per call: n_used experts × (gate + up) × n_rows × n_blocks × 66B
    let bytes_per_call: u64 = (n_used as u64) * 2
        * (n_rows as u64) * (n_blocks as u64) * (BLOCK_IQ2_XXS_BYTES as u64);
    let gbps = bytes_per_call as f32 / 1e9 / (p50 / 1000.0);
    eprintln!(
        "weight BW (p50): {:.1} GB/s (read {} MiB per call)",
        gbps, bytes_per_call / 1024 / 1024,
    );
    // Per-layer iGPU MoE rough wall (this kernel only; q8k_quantize + q2k_down
    // not included). 41 ratio>0 layers in V4-Flash.
    eprintln!(
        "= {:.1} µs/call; rough per-token (43 layers): {:.1} ms",
        p50 * 1000.0, p50 * 43.0,
    );
    Ok(())
}
