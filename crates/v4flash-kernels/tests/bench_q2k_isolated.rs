//! Isolated probe for `q2_k_matvec_par_batched_BxN` and the new
//! `q2_k_matvec_par_by_expert` at production prefill shape (B=512, n_used=6).
//!
//! Set ATT_SKIP_FILL=1 to make q2k the first kernel dispatched (no preceding
//! fill_zero), so rocprofv3 PMC / ATT can target it cleanly without a regex
//! filter (the regex filter hangs ATT on gfx1151/gfx1201 in 7.2.3).
//!
//! BENCH_VARIANT=0 → batched_BxN (production), 1 → by_expert (new).
//! Default: run BOTH back-to-back, A/B + correctness comparison.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::config::{
    BLOCKS_Q8K_DOWN_IN, N_EMBD, N_EXPERT, N_EXPERT_USED,
};
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

/// Build `group_count`, `expert_members`, and `work_items` from a host
/// `selected` array — same pattern as the iGPU moe_group_builder, but on host
/// for testability. Returns (gc_host, em_host, wi_host).
fn build_group_arrays(
    selected_host: &[i32],
    b: u32,
    n_used: u32,
    n_expert: i32,
    max_per_expert: u32,
    chunk_size: u32,
) -> (Vec<i32>, Vec<i32>, Vec<i32>) {
    // group_count[e] = #(b, slot) pairs that picked expert e.
    let mut gc = vec![0i32; n_expert as usize];
    let mut em = vec![0i32; (n_expert as usize) * (max_per_expert as usize)];
    for bi in 0..(b as usize) {
        for s in 0..(n_used as usize) {
            let e = selected_host[bi * (n_used as usize) + s] as usize;
            let slot_in_grp = gc[e] as usize;
            em[e * (max_per_expert as usize) + slot_in_grp] =
                ((bi as i32) << 16) | (s as i32);
            gc[e] += 1;
        }
    }
    // work_items: chunk popular experts. Each entry = (e<<16) | member_start.
    let mut wi: Vec<i32> = Vec::new();
    for e in 0..(n_expert as usize) {
        let n = gc[e] as u32;
        if n == 0 { continue; }
        let mut start = 0u32;
        while start < n {
            wi.push(((e as i32) << 16) | (start as i32));
            start += chunk_size;
        }
    }
    (gc, em, wi)
}

#[test]
#[ignore]
fn bench_q2k_isolated() -> eyre::Result<()> {
    install_panic_handler()?;

    let b: u32 = std::env::var("BENCH_B")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(512);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let chunk_size: u32 = std::env::var("BENCH_CHUNK")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(64);
    let skip_fill = std::env::var_os("ATT_SKIP_FILL").is_some();
    // BENCH_VARIANT: 0 = bxn only, 1 = by_expert only, default = both A/B.
    let variant: Option<u32> = std::env::var("BENCH_VARIANT")
        .ok().and_then(|s| s.parse().ok());

    let n_rows = N_EMBD;
    let n_used = N_EXPERT_USED as u32;
    let n_blocks_in = BLOCKS_Q8K_DOWN_IN;
    let dbpe = (n_rows as usize) * (n_blocks_in as usize) * BLOCK_Q2_K_BYTES;
    let xq_slot_stride = (n_blocks_in as u32) * (BLOCK_Q8_K_BYTES as u32);
    let max_per_expert = b; // upper bound: every batch slot could pick the same expert

    eprintln!("=== isolated q2_k MoE down probe ===");
    eprintln!("B={b} n_used={n_used} n_rows={n_rows} n_blocks_in={n_blocks_in} chunk={chunk_size}");
    eprintln!("dbpe={} KiB/expert  weight total: {} MiB",
              dbpe / 1024, (N_EXPERT as usize) * dbpe / 1024 / 1024);

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let q2k = Q2KAccumulateMatvec::for_arch(&arch)?;

    let mut out: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, (b as usize) * (n_rows as usize))?;
    let mut down_w: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (N_EXPERT as usize) * dbpe)?;
    let mut xq: DeviceBuffer<u8> = DeviceBuffer::new(
        igpu.id, (b as usize) * (n_used as usize) * (xq_slot_stride as usize))?;
    let mut selected: DeviceBuffer<i32> = DeviceBuffer::new(
        igpu.id, (b as usize) * (n_used as usize))?;

    if !skip_fill {
        out.fill_zero()?;
        down_w.fill_zero()?;
        xq.fill_zero()?;
    }

    // Build `selected` with a moderately Zipf-y distribution: half the picks
    // hit a 16-expert hot head, the rest spread evenly. Mimics real routing
    // pattern shape — distinct(E) ≈ 64-80 at B=512.
    let mut sel_host = vec![0i32; (b as usize) * (n_used as usize)];
    let mut rng_state: u64 = 0xC0FFEE_1234_5678;
    let next_u32 = |s: &mut u64| -> u32 {
        *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*s >> 32) & 0xffffffff) as u32
    };
    let n_hot: i32 = std::env::var("BENCH_HOT").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(16);
    // BENCH_ZIPF: fraction of picks that hit the hot head (default 0.5).
    // Larger value → fewer distinct experts → bigger by_expert win.
    let zipf_num: u32 = std::env::var("BENCH_ZIPF_NUM").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(1);
    let zipf_den: u32 = std::env::var("BENCH_ZIPF_DEN").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(2);
    for i in 0..(b as usize) * (n_used as usize) {
        let r = next_u32(&mut rng_state);
        let hot = (r % zipf_den) < zipf_num;
        sel_host[i] = if hot {
            (r >> 8) as i32 % n_hot
        } else {
            (r >> 8) as i32 % (N_EXPERT as i32)
        };
    }
    selected.copy_from_host(&sel_host)?;

    // Host-side group builder for the by_expert variant.
    let (gc_host, em_host, wi_host) = build_group_arrays(
        &sel_host, b, n_used, N_EXPERT as i32, max_per_expert, chunk_size);
    let n_work_items = wi_host.len() as u32;
    // distinct(E) for visibility
    let distinct_e = gc_host.iter().filter(|&&c| c > 0).count();
    eprintln!("synth routing: distinct(E)={distinct_e}/{N_EXPERT}, n_work_items={n_work_items}, max group={}",
              gc_host.iter().max().copied().unwrap_or(0));

    let mut group_count_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members_d: DeviceBuffer<i32> = DeviceBuffer::new(
        igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;
    let mut work_items_d: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, n_work_items as usize)?;
    group_count_d.copy_from_host(&gc_host)?;
    expert_members_d.copy_from_host(&em_host)?;
    work_items_d.copy_from_host(&wi_host)?;

    let do_bxn = variant.is_none() || variant == Some(0);
    let do_by_expert = variant.is_none() || variant == Some(1);
    let do_kwide = variant == Some(2);

    // by_expert needs a partials buffer (B*n_used, n_rows).
    let mut partials: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id, (b as usize) * (n_used as usize) * (n_rows as usize))?;

    // Warmup
    for _ in 0..warmup {
        if do_bxn {
            q2k.launch_batched_bxn(&stream, &mut out, &down_w, &xq, &selected,
                dbpe as u32, xq_slot_stride, n_used, n_rows, n_blocks_in, b)?;
        }
        if do_by_expert {
            partials.fill_zero()?;
            q2k.launch_by_expert(&stream, &mut partials, &down_w, &xq,
                &group_count_d, &expert_members_d, &work_items_d,
                dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                n_rows, n_blocks_in, n_work_items)?;
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
        }
    }
    stream.synchronize()?;

    // Correctness: capture outputs of both at zero-fill state, compare.
    if do_bxn && do_by_expert {
        let mut out_bxn = vec![0.0f32; (b as usize) * (n_rows as usize)];
        let mut out_be  = vec![0.0f32; (b as usize) * (n_rows as usize)];

        q2k.launch_batched_bxn(&stream, &mut out, &down_w, &xq, &selected,
            dbpe as u32, xq_slot_stride, n_used, n_rows, n_blocks_in, b)?;
        stream.synchronize()?;
        out.copy_to_host(&mut out_bxn)?;

        partials.fill_zero()?;
        q2k.launch_by_expert(&stream, &mut partials, &down_w, &xq,
            &group_count_d, &expert_members_d, &work_items_d,
            dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
            n_rows, n_blocks_in, n_work_items)?;
        q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
        stream.synchronize()?;
        out.copy_to_host(&mut out_be)?;

        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut any_diff = 0usize;
        for (a, b) in out_bxn.iter().zip(out_be.iter()) {
            let d = (a - b).abs();
            if d > max_abs { max_abs = d; }
            let denom = a.abs().max(b.abs()).max(1e-30);
            let r = d / denom;
            if r > max_rel { max_rel = r; }
            if d > 1e-3 { any_diff += 1; }
        }
        eprintln!("correctness vs bxn: max_abs={max_abs:.3e} max_rel={max_rel:.3e} n_diff>1e-3={any_diff}");
        // With zero inputs both should produce zero output identically.
    }

    let summarize = |name: &str, walls_ms: &mut [f32]| {
        walls_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let min = walls_ms[0];
        let med = walls_ms[walls_ms.len() / 2];
        let max = walls_ms[walls_ms.len() - 1];
        eprintln!("{name}: min={min:.3} ms  median={med:.3} ms  max={max:.3} ms");
    };

    if do_bxn {
        let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Event::new()?;
            let end = Event::new()?;
            start.record(&stream)?;
            q2k.launch_batched_bxn(&stream, &mut out, &down_w, &xq, &selected,
                dbpe as u32, xq_slot_stride, n_used, n_rows, n_blocks_in, b)?;
            end.record(&stream)?;
            stream.synchronize()?;
            walls_ms.push(Event::elapsed_ms(&start, &end)?);
        }
        summarize("q2k_bxn        ", &mut walls_ms);
    }
    if do_by_expert {
        // by_expert timing includes: zero partials + by_expert kernel +
        // reduce kernel. Matches production's per-call cost.
        let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Event::new()?;
            let end = Event::new()?;
            start.record(&stream)?;
            partials.fill_zero()?;
            q2k.launch_by_expert(&stream, &mut partials, &down_w, &xq,
                &group_count_d, &expert_members_d, &work_items_d,
                dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                n_rows, n_blocks_in, n_work_items)?;
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
            end.record(&stream)?;
            stream.synchronize()?;
            walls_ms.push(Event::elapsed_ms(&start, &end)?);
        }
        summarize("q2k_by_expert  ", &mut walls_ms);
    }
    if do_kwide {
        // Warmup (kwide wasn't part of the shared warmup loop above).
        for _ in 0..warmup {
            partials.fill_zero()?;
            q2k.launch_by_expert_kwide(&stream, &mut partials, &down_w, &xq,
                &group_count_d, &expert_members_d, &work_items_d,
                dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                n_rows, n_blocks_in, n_work_items)?;
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
        }
        stream.synchronize()?;
        let mut walls_ms: Vec<f32> = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = Event::new()?;
            let end = Event::new()?;
            start.record(&stream)?;
            partials.fill_zero()?;
            q2k.launch_by_expert_kwide(&stream, &mut partials, &down_w, &xq,
                &group_count_d, &expert_members_d, &work_items_d,
                dbpe as u32, xq_slot_stride, n_used, max_per_expert, chunk_size,
                n_rows, n_blocks_in, n_work_items)?;
            q2k.launch_reduce_partials(&stream, &mut out, &partials, n_used, n_rows, b)?;
            end.record(&stream)?;
            stream.synchronize()?;
            walls_ms.push(Event::elapsed_ms(&start, &end)?);
        }
        summarize("q2k_kwide      ", &mut walls_ms);
    }
    Ok(())
}
