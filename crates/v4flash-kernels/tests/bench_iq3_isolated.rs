//! Isolated A/B: IQ3_XXS down kernels vs the Q2_K incumbents on identical
//! shapes + routing (Phase 1 de-risk spike for the unsloth UD mix).
//!
//! Prefill: iq3 by_expert_kwide2 vs q2k by_expert_kwide2 (production
//! default), both timed WITH partials zero-fill + reduce, same synthetic
//! Zipf routing. Decode: iq3 matvec_par_batched vs q2k matvec_par_batched
//! (single token, n_used=6).
//!
//! Roofline framing: IQ3_XXS reads 98/84 = 1.167× the weight bytes of Q2_K.
//! If the time ratio ≈ byte ratio → still BW-bound → proceed. Time ratio
//! ≫ 1.25× → grid-decode VALU cost dominates → mitigation needed.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_iq3_isolated -- --ignored --nocapture
//!
//! rocprofv3 (per-kernel):
//!   TEST_BIN=$(find target/release/deps -name "bench_iq3_isolated-*" -executable -not -name "*.d" | head -1)
//!   rocprofv3 --kernel-trace -d /tmp/iq3_prof -o run -- \
//!     "$TEST_BIN" --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::iq3_xxs::{Iq3XxsMatvec, BLOCK_IQ3_XXS_BYTES};
use v4flash_kernels::q2_k::{Q2KAccumulateMatvec, BLOCK_Q2_K_BYTES};
use v4flash_kernels::q8_k::BLOCK_Q8_K_BYTES;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1151") || arch.starts_with("gfx1150") {
            return Ok(d);
        }
    }
    Err(eyre!("no iGPU (gfx1150/gfx1151) found"))
}

fn build_group_arrays(
    selected_host: &[i32],
    b: u32,
    n_used: u32,
    n_expert: i32,
    max_per_expert: u32,
    chunk_size: u32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    let mut gc = vec![0i32; n_expert as usize];
    let mut em = vec![0i32; (n_expert as usize) * (max_per_expert as usize)];
    for bi in 0..(b as usize) {
        for s in 0..(n_used as usize) {
            let e = selected_host[bi * (n_used as usize) + s] as usize;
            let slot_in_grp = gc[e] as usize;
            em[e * (max_per_expert as usize) + slot_in_grp] = ((bi as i32) << 16) | (s as i32);
            gc[e] += 1;
        }
    }
    let mut wi: Vec<i32> = Vec::new();
    for e in 0..(n_expert as usize) {
        let n = gc[e] as u32;
        if n == 0 {
            continue;
        }
        let mut start = 0u32;
        while start < n {
            wi.push(((e as i32) << 16) | (start as i32));
            start += chunk_size;
        }
    }
    (gc, em, wi)
}

fn summarize(name: &str, walls_ms: &mut [f32], weight_bytes: usize) -> f32 {
    walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let min = walls_ms[0];
    let med = walls_ms[walls_ms.len() / 2];
    let gbps = (weight_bytes as f64 / 1e9) / (med as f64 / 1e3);
    eprintln!(
        "{name}: min={min:.3} ms  median={med:.3} ms  weight-read {:.1} MiB → {gbps:.0} GB/s (weights only)",
        weight_bytes as f64 / (1u64 << 20) as f64
    );
    med
}

#[test]
#[ignore]
fn bench_iq3_vs_q2k() -> eyre::Result<()> {
    install_panic_handler()?;

    let b: u32 = std::env::var("BENCH_B").ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let iters: usize = std::env::var("BENCH_ITERS").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let warmup: usize = std::env::var("BENCH_WARMUP").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let chunk_size: u32 = std::env::var("BENCH_CHUNK").ok().and_then(|s| s.parse().ok()).unwrap_or(32);

    let n_rows = N_EMBD;
    let n_used = N_EXPERT_USED as u32;
    let nb = BLOCKS_Q8K_DOWN_IN;
    let dbpe_q2k = (n_rows as usize) * (nb as usize) * BLOCK_Q2_K_BYTES;
    let dbpe_iq3 = (n_rows as usize) * (nb as usize) * BLOCK_IQ3_XXS_BYTES;
    let xq_slot_stride = (nb as u32) * (BLOCK_Q8_K_BYTES as u32);
    let max_per_expert = b;

    eprintln!("=== IQ3_XXS vs Q2_K down A/B (B={b}, chunk={chunk_size}) ===");
    eprintln!(
        "dbpe: q2k {} KiB, iq3 {} KiB (byte ratio {:.3})",
        dbpe_q2k / 1024,
        dbpe_iq3 / 1024,
        dbpe_iq3 as f64 / dbpe_q2k as f64
    );

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;
    let iq3 = Iq3XxsMatvec::for_arch(&arch)?;

    // Zero-filled synthetic buffers (timing is data-independent).
    let mut out: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (n_rows as usize))?;
    let mut w_q2k: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * dbpe_q2k)?;
    let mut w_iq3: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * dbpe_iq3)?;
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id,
        (b as usize) * (n_used as usize) * (xq_slot_stride as usize),
    )?;
    out.fill_zero()?;
    w_q2k.fill_zero()?;
    w_iq3.fill_zero()?;
    xq.fill_zero()?;

    // Same synthetic Zipf routing as bench_q2k_isolated.
    let mut sel_host = vec![0i32; (b as usize) * (n_used as usize)];
    let mut rng_state: u64 = 0xC0FFEE_1234_5678;
    let next_u32 = |s: &mut u64| -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 32) & 0xffffffff) as u32
    };
    for slot in sel_host.iter_mut() {
        let r = next_u32(&mut rng_state);
        *slot = if (r % 2) == 0 {
            (r >> 8) as i32 % 16
        } else {
            (r >> 8) as i32 % (N_EXPERT as i32)
        };
    }
    let mut selected: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (n_used as usize))?;
    selected.copy_from_host(&sel_host)?;

    let (gc_host, em_host, wi_host) =
        build_group_arrays(&sel_host, b, n_used, N_EXPERT as i32, max_per_expert, chunk_size);
    let n_work_items = wi_host.len() as u32;
    let distinct_e = gc_host.iter().filter(|&&c| c > 0).count();
    eprintln!("routing: distinct(E)={distinct_e}, n_work_items={n_work_items}");

    let mut gc_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, gc_host.len())?;
    let mut em_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, em_host.len())?;
    let mut wi_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_host.len())?;
    gc_d.copy_from_host(&gc_host)?;
    em_d.copy_from_host(&em_host)?;
    wi_d.copy_from_host(&wi_host)?;

    let mut partials: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (n_used as usize) * (n_rows as usize))?;

    // Per-launch weight bytes: each work item reads its expert's full tile.
    let wbytes_q2k = (n_work_items as usize) * dbpe_q2k;
    let wbytes_iq3 = (n_work_items as usize) * dbpe_iq3;

    // ---- prefill kwide2 A/B ----
    let mut med_q2k = 0f32;
    let mut med_iq3 = 0f32;
    for pass in 0..2 {
        let name = if pass == 0 { "q2k_kwide2" } else { "iq3_kwide2" };
        for _ in 0..warmup {
            partials.fill_zero()?;
            if pass == 0 {
                q2k.launch_by_expert_kwide2(
                    &stream, &mut partials, &w_q2k, &xq, &gc_d, &em_d, &wi_d,
                    dbpe_q2k as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                    n_rows, nb, n_work_items,
                )?;
            } else {
                iq3.launch_by_expert_kwide2(
                    &stream, &mut partials, &w_iq3, &xq, &gc_d, &em_d, &wi_d, n_work_items,
                    dbpe_iq3 as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                    n_rows, nb,
                )?;
            }
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
        }
        stream.synchronize()?;
        let mut walls: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Event::new()?;
            let end = Event::new()?;
            start.record(&stream)?;
            partials.fill_zero()?;
            if pass == 0 {
                q2k.launch_by_expert_kwide2(
                    &stream, &mut partials, &w_q2k, &xq, &gc_d, &em_d, &wi_d,
                    dbpe_q2k as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                    n_rows, nb, n_work_items,
                )?;
            } else {
                iq3.launch_by_expert_kwide2(
                    &stream, &mut partials, &w_iq3, &xq, &gc_d, &em_d, &wi_d, n_work_items,
                    dbpe_iq3 as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                    n_rows, nb,
                )?;
            }
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
            end.record(&stream)?;
            stream.synchronize()?;
            walls.push(Event::elapsed_ms(&start, &end)?);
        }
        let med = summarize(name, &mut walls, if pass == 0 { wbytes_q2k } else { wbytes_iq3 });
        if pass == 0 {
            med_q2k = med;
        } else {
            med_iq3 = med;
        }
    }
    eprintln!(
        "PREFILL VERDICT: time ratio {:.3} vs byte ratio {:.3} → {}",
        med_iq3 / med_q2k,
        dbpe_iq3 as f64 / dbpe_q2k as f64,
        if (med_iq3 / med_q2k) as f64 <= 1.25 {
            "BW-bound, PROCEED"
        } else {
            "VALU overhead — investigate"
        }
    );

    // ---- decode batched A/B (single token, n_used experts) ----
    let mut out1: DeviceBuffer<f32> = DeviceBuffer::new(igpu.id, n_rows as usize)?;
    let sel6: Vec<i32> = (0..n_used as i32).map(|i| i * 37 % (N_EXPERT as i32)).collect();
    let mut sel6_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_used as usize)?;
    sel6_d.copy_from_host(&sel6)?;
    for pass in 0..2 {
        let name = if pass == 0 { "q2k_decode_batched" } else { "iq3_decode_batched" };
        for _ in 0..warmup {
            if pass == 0 {
                q2k.launch_batched(&stream, &mut out1, &w_q2k, &xq, &sel6_d,
                    dbpe_q2k as u32, xq_slot_stride, n_used, n_rows, nb)?;
            } else {
                iq3.launch_batched(&stream, &mut out1, &w_iq3, &xq, &sel6_d,
                    dbpe_iq3 as u32, xq_slot_stride, n_used, n_rows, nb)?;
            }
        }
        stream.synchronize()?;
        let mut walls: Vec<f32> = Vec::with_capacity(iters.max(50));
        for _ in 0..iters.max(50) {
            let start = Event::new()?;
            let end = Event::new()?;
            start.record(&stream)?;
            if pass == 0 {
                q2k.launch_batched(&stream, &mut out1, &w_q2k, &xq, &sel6_d,
                    dbpe_q2k as u32, xq_slot_stride, n_used, n_rows, nb)?;
            } else {
                iq3.launch_batched(&stream, &mut out1, &w_iq3, &xq, &sel6_d,
                    dbpe_iq3 as u32, xq_slot_stride, n_used, n_rows, nb)?;
            }
            end.record(&stream)?;
            stream.synchronize()?;
            walls.push(Event::elapsed_ms(&start, &end)?);
        }
        let wb = (n_used as usize) * if pass == 0 { dbpe_q2k } else { dbpe_iq3 };
        summarize(name, &mut walls, wb);
    }
    Ok(())
}
