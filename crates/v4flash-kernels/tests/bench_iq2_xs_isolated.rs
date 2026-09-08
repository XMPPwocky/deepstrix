//! Standalone IQ2_XS / IQ3_XXS-pair gate+up kernel probe — no GGUF, no
//! model load. Clone of bench_iq2_isolated for the unsloth UD-Q2_K_XL
//! pair kernels (chunked vs kwide).
//!
//! Env:
//!   BENCH_FMT     = iq2_xs (default) | iq2s | iq3 | iq3s
//!   BENCH_N_EXPERT = experts allocated (default N_EXPERT=256; lower it to
//!                    bound device memory — work items cycle over
//!                    min(BENCH_WI, BENCH_N_EXPERT) distinct experts)
//!   BENCH_VARIANT = 0 chunked (serial per-member) | 6 kwide
//!   BENCH_RANDOM_W = 1 fills gate/up with random bytes (default: zeros)
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
use v4flash_kernels::iq2_s::{Iq2SPairMatvec, BLOCK_IQ2_S_BYTES};
use v4flash_kernels::iq2_xs::{Iq2XsPairMatvec, BLOCK_IQ2_XS_BYTES};
use v4flash_kernels::iq3_s::{Iq3SPairMatvec, BLOCK_IQ3_S_BYTES};
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
    // 7 = f16 WMMA prototype (f16 grid LUT dequant), 8 = WMMA with per-byte
    // cvt dequant. Both take f16 activations; iq2_xs only.
    // 9..12 = ablations (no WMMA / no dequant / no x staging / no B loads).
    let use_wmma = (7..=13).contains(&variant);   // 13 = wmma magic
    let n_expert_alloc: u32 = std::env::var("BENCH_N_EXPERT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(N_EXPERT)
        .clamp(1, N_EXPERT);
    let block_bytes = match fmt.as_str() {
        "iq3" => BLOCK_IQ3_XXS_BYTES,
        "iq3s" => BLOCK_IQ3_S_BYTES,
        "iq2s" => BLOCK_IQ2_S_BYTES,
        _ => BLOCK_IQ2_XS_BYTES,
    };
    eprintln!("fmt={fmt} variant={variant} (0=chunked, 6=kwide, 7=wmma, 8=wmma_cvt, 9-12=wmma ablations, 13=wmma_lut)");
    eprintln!("isolated probe: B={b}, iters={iters}, n_work_items={n_work_items_target}, chunk={chunk_size}");

    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let arch = igpu.properties()?.gcn_arch_name;
    let stream = Stream::new(igpu.id)?;
    let iq2xs = Iq2XsPairMatvec::for_arch(&arch)?;
    let iq3p = Iq3XxsPairMatvec::for_arch(&arch)?;
    let iq3s = Iq3SPairMatvec::for_arch(&arch)?;
    let iq2s = Iq2SPairMatvec::for_arch(&arch)?;

    let gate_bpe = (N_FF_EXP as usize) * (BLOCKS_Q8K_GATE_IN as usize) * block_bytes;
    let up_bpe = gate_bpe;
    let total_gate_bytes = gate_bpe * (n_expert_alloc as usize);
    eprintln!("allocating: gate+up 2 x {} MB", total_gate_bytes / 1_000_000);
    let mut gate_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate_bytes)?;
    let mut up_w: DeviceBuffer<u8> = DeviceBuffer::new(igpu.id, total_gate_bytes)?;
    if std::env::var("BENCH_RANDOM_W").map(|v| v == "1").unwrap_or(false) {
        // Random block bytes so the data-dependent LDS grid/sign gathers
        // see real (conflicting) indices instead of all hitting entry 0.
        // Every bit pattern is a valid block for these formats.
        let mut state: u64 = 0x9E3779B97F4A7C15;
        let mut fill = |dst: &mut DeviceBuffer<u8>| -> eyre::Result<()> {
            let mut h = vec![0u8; total_gate_bytes];
            for chunk in h.chunks_exact_mut(8) {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                chunk.copy_from_slice(&state.to_le_bytes());
            }
            dst.copy_from_host(&h)
        };
        fill(&mut gate_w)?;
        fill(&mut up_w)?;
        eprintln!("weights: random bytes (BENCH_RANDOM_W=1)");
    } else {
        gate_w.fill_zero()?;
        up_w.fill_zero()?;
    }

    let xq_bytes_per_token = (BLOCKS_Q8K_GATE_IN as usize) * BLOCK_Q8_K_BYTES;
    let mut xq: DeviceBuffer<u8> =
        DeviceBuffer::new(igpu.id, xq_bytes_per_token * (b as usize))?;
    xq.fill_zero()?;
    // f16 activations for the WMMA variants: [B, K] halves, small non-zero
    // values (0.5) so the matrix pipe does real work.
    let k_dim = (BLOCKS_Q8K_GATE_IN as usize) * 256;
    let mut x16: DeviceBuffer<u16> = DeviceBuffer::new(igpu.id, k_dim * (b as usize))?;
    x16.copy_from_host(&vec![0x3800u16; k_dim * (b as usize)])?;

    let cs_n_used = N_EXPERT_USED as u32;
    let mut expert_w: DeviceBuffer<f32> =
        DeviceBuffer::new(igpu.id, (b as usize) * (cs_n_used as usize))?;
    expert_w.fill_zero()?;

    let max_per_expert = b;
    let mut group_count: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, N_EXPERT as usize)?;
    let mut expert_members: DeviceBuffer<i32> =
        DeviceBuffer::new(igpu.id, (N_EXPERT as usize) * (max_per_expert as usize))?;

    // Distinct experts per work item (same pattern as bench_iq2_isolated).
    let n_distinct = n_work_items_target.min(n_expert_alloc) as usize;
    let mut gc_host = vec![0i32; N_EXPERT as usize];
    let mut em_host = vec![0i32; (N_EXPERT as usize) * (max_per_expert as usize)];
    let mut wi_host = vec![0i32; n_work_items_target as usize];
    // BENCH_DIST=zipf: realistic routing. Draw B*8 selections per layer
    // proportional to the prefill counts in BENCH_STATS (expert_stats.json,
    // layer BENCH_LAYER, default 20), drop the top BENCH_HOT_K experts (the
    // het-split's dGPU-resident set, default 17), then build
    // group_count/expert_members/work_items exactly like moe_group_builder +
    // work_items_builder (chunks of chunk_size per expert). The uniform
    // default (every work item a full chunk) flatters tile-shaped kernels.
    let dist = std::env::var("BENCH_DIST").unwrap_or_else(|_| "uniform".into());
    let n_work_items_real: u32 = if dist == "zipf" {
        let stats_path = std::env::var("BENCH_STATS").map_err(|_| eyre!("BENCH_DIST=zipf needs BENCH_STATS=<expert_stats.json>"))?;
        let layer: usize = std::env::var("BENCH_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(20);
        let hot_k: usize = std::env::var("BENCH_HOT_K").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
        let txt = std::fs::read_to_string(&stats_path)?;
        let js: serde_json::Value = serde_json::from_str(&txt)?;
        let counts: Vec<f64> = js["prefill"]["counts"].as_array().ok_or_else(|| eyre!("no prefill.counts"))?
            .iter().skip(layer * 256).take(256).map(|v| v.as_f64().unwrap_or(0.0)).collect();
        let mut order: Vec<usize> = (0..256).collect();
        order.sort_by(|&a, &b| counts[b].partial_cmp(&counts[a]).unwrap());
        let hot: std::collections::HashSet<usize> = order.iter().take(hot_k).cloned().collect();
        // cumulative distribution over cold experts
        let mut cdf = Vec::with_capacity(256);
        let mut acc = 0f64;
        for e in 0..256 { if !hot.contains(&e) { acc += counts[e]; } cdf.push(acc); }
        let mut rng: u64 = 0xC0FFEE_2026_0908;
        let mut per_expert: Vec<Vec<i32>> = vec![Vec::new(); 256];
        for tok in 0..(b as usize) {
            let mut picked = std::collections::HashSet::new();
            let mut slot = 0usize;
            while slot < cs_n_used as usize {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((rng >> 11) as f64) / ((1u64 << 53) as f64) * acc;
                let e = cdf.partition_point(|&c| c <= u).min(255);
                if hot.contains(&e) || !picked.insert(e) { continue; }
                per_expert[e].push(((tok as i32) << 16) | (slot as i32));
                slot += 1;
            }
        }
        // hot slots are the dGPU's; on the iGPU those members simply don't exist
        // (mirrors moe_group_builder_hetsplit mode 0 with cap = n_used).
        wi_host.clear();
        let mut hist = std::collections::BTreeMap::new();
        for e in 0..256 {
            let n = per_expert[e].len();
            gc_host[e] = n as i32;
            for (i, &m) in per_expert[e].iter().enumerate() {
                em_host[e * (max_per_expert as usize) + i] = m;
            }
            let mut start = 0usize;
            while start < n { wi_host.push(((e as i32) << 16) | (start as i32)); start += chunk_size as usize; }
            if n > 0 { *hist.entry((n + 15) / 16).or_insert(0usize) += n; }
        }
        let total: usize = per_expert.iter().map(|v| v.len()).sum();
        eprintln!("zipf dist: layer={layer} hot_k={hot_k} members on iGPU={total} (of {}), work items={}, members by 16-tile count: {:?}",
            (b as usize) * (cs_n_used as usize), wi_host.len(), hist);
        wi_host.len() as u32
    } else {
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
        n_work_items_target
    };
    let mut work_items: DeviceBuffer<i32> = DeviceBuffer::new(igpu.id, wi_host.len().max(1))?;
    let n_work_items_target = n_work_items_real;
    group_count.copy_from_host(&gc_host)?;
    expert_members.copy_from_host(&em_host)?;
    work_items.copy_from_host(&wi_host)?;

    let mut mid: DeviceBuffer<f32> = DeviceBuffer::new(
        igpu.id,
        (b as usize) * (cs_n_used as usize) * (N_FF_EXP as usize),
    )?;

    let launch = |mid: &mut DeviceBuffer<f32>| -> eyre::Result<()> {
        if use_wmma {
            if fmt != "iq2_xs" {
                return Err(eyre!("wmma variants exist for iq2_xs only"));
            }
            return iq2xs.launch_fused_swiglu_wmma(
                &stream, mid, &gate_w, &up_w, &x16, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
                variant - 7,
            );
        }
        match (fmt.as_str(), use_kwide) {
            ("iq2s", true) => iq2s.launch_fused_swiglu_kwide(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            ("iq2s", false) => iq2s.launch_fused_swiglu_chunked(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            ("iq3s", true) => iq3s.launch_fused_swiglu_kwide(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
            ("iq3s", false) => iq3s.launch_fused_swiglu_chunked(
                &stream, mid, &gate_w, &up_w, &xq, &expert_w,
                &group_count, &expert_members, &work_items, n_work_items_target,
                gate_bpe as u32, up_bpe as u32, cs_n_used, max_per_expert,
                chunk_size, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            ),
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
