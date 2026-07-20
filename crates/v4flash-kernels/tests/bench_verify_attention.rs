//! True B=k DECODE-attention cost driver — the load-bearing kernel for an
//! MTP verify-k step, measured in isolation (no model load).
//!
//! ## Why this exists
//! The full-forward verify bench (bench_verify_decode) showed the dGPU
//! projection/MoE wall is flat across B (weights read once, applied to k
//! rows). The ONE part that genuinely grows with k is the attention CORE:
//! each of the k query rows attends over the shared KV window, so a naive
//! loop of the B=1 decode kernels costs ~k× the window read. The question
//! this driver answers: does batching the decode attention (share the KV
//! window across the k queries via LDS/occupancy) beat the k× loop?
//!
//! Two strategies, same decode-shaped shared window, per B:
//!   * `loop_b1`  : the PRODUCTION B=1 decode kernels
//!       (score_b1_htiled_wmma → softmax_only → wsum_b1_ksplit_ldsv →
//!        reduce), run k times (q offset per row). Cost ≈ k × B1.
//!   * `batched`  : the shared-window batched WMMA kernels prefill uses
//!       (score_batched_htiled_wmma_f16s → softmax_wsum_batched_ldsv_f16s)
//!       at batch=B. Shares the KV window across the k queries.
//!
//! The decode window is n_raw=128 + comp keys. At depth the comp keys are
//! SPARSE-gathered to ≤ INDEXER_TOP_K (≤512), so ATTN_SCORES_STRIDE=2048
//! bounds n_total. We sweep realistic gathered windows.
//!
//! Reports per (B, n_comp): loop_b1 µs, batched µs, C_attn(B) for each
//! strategy, and which wins. BENCH_DIFF=1 checks batched≡single at B=1.
//!
//! ## Run
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 VERIFY_B=1,2,3,4 VERIFY_NCOMP=384,512,1024 \
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test bench_verify_attention -- --ignored --nocapture
//! ```

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::attention::{AttentionMixed, ATTN_MIXED_MAX_KEYS, ATTN_SCORES_STRIDE};
use v4flash_kernels::config::{N_HEAD, N_HEAD_DIM};

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}

fn median(xs: &mut [f32]) -> f32 {
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

#[test]
#[ignore]
fn bench_verify_attention() -> eyre::Result<()> {
    install_panic_handler()?;

    let bs_list: Vec<u32> = std::env::var("VERIFY_B")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![1, 2, 3, 4]);
    let b_max = *bs_list.iter().max().unwrap();
    // Realistic sparse-gathered decode windows (comp keys). Default sweep
    // spans the sparse floor (INDEXER_TOP_K≈512) and a denser 1024.
    let ncomp_list: Vec<u32> = std::env::var("VERIFY_NCOMP")
        .ok()
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .filter(|v: &Vec<u32>| !v.is_empty())
        .unwrap_or_else(|| vec![384, 512, 1024]);
    let n_raw: u32 = std::env::var("VERIFY_NRAW")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(128);
    let iters: usize = std::env::var("BENCH_ITERS")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(200);
    let warmup: usize = std::env::var("BENCH_WARMUP")
        .ok().and_then(|s| s.parse().ok()).unwrap_or(20);
    const K_SPLIT: u32 = 16;

    let dgpu = pick_dgpu()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let attn = AttentionMixed::for_arch(&arch)?;
    let head_dim = N_HEAD_DIM;
    let n_head = N_HEAD;
    eprintln!(
        "verify-attn: B={bs_list:?} n_comp={ncomp_list:?} n_raw={n_raw} \
         n_head={n_head} head_dim={head_dim} iters={iters}"
    );

    // === buffers sized for the max B / max n_comp ===
    let max_ncomp = *ncomp_list.iter().max().unwrap();
    let hd = head_dim as usize;
    let nh = n_head as usize;
    let bm = b_max as usize;

    // q: [B, n_head, head_dim]
    let mut q: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, bm * nh * hd)?;
    q.fill_zero()?;
    // raw_kv shared: [128, head_dim] f16
    let mut raw_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, 128 * hd)?;
    raw_kv.fill_zero()?;
    // comp_kv shared: [max_ncomp, head_dim] f16
    let mut comp_kv: DeviceBuffer<u16> = DeviceBuffer::new(dgpu.id, (max_ncomp as usize) * hd)?;
    comp_kv.fill_zero()?;
    let mut sinks: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, nh)?;
    sinks.fill_zero()?;

    // loop_b1 scratch (single row, reused): scores [n_head, MAX_KEYS],
    // partials [k_split, n_head, head_dim], inv [n_head], out [n_head, head_dim]
    let mut b1_scores: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, nh * ATTN_MIXED_MAX_KEYS as usize)?;
    b1_scores.fill_zero()?;
    let mut b1_partials: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, K_SPLIT as usize * nh * hd)?;
    b1_partials.fill_zero()?;
    let mut b1_inv: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, nh)?;
    b1_inv.fill_zero()?;
    let mut b1_out: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, nh * hd)?;
    b1_out.fill_zero()?;

    // batched scratch: scores [B, n_head, ATTN_SCORES_STRIDE], out [B, n_head, head_dim]
    let mut bt_scores: DeviceBuffer<f32> =
        DeviceBuffer::new(dgpu.id, bm * nh * ATTN_SCORES_STRIDE as usize)?;
    bt_scores.fill_zero()?;
    let mut bt_out: DeviceBuffer<f32> = DeviceBuffer::new(dgpu.id, bm * nh * hd)?;
    bt_out.fill_zero()?;

    // Per-batch counters (all rows share the SAME decode window).
    let mut n_raw_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, bm)?;
    let mut n_off_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, bm)?;
    let mut n_comp_per: DeviceBuffer<i32> = DeviceBuffer::new(dgpu.id, bm)?;

    // loop_b1: run the production B=1 decode kernels once per query row.
    let run_loop_b1 = |stream: &Stream,
                       b: u32,
                       n_comp: u32,
                       b1_scores: &mut DeviceBuffer<f32>,
                       b1_partials: &mut DeviceBuffer<f32>,
                       b1_inv: &mut DeviceBuffer<f32>,
                       b1_out: &mut DeviceBuffer<f32>|
     -> eyre::Result<()> {
        let n_total = n_raw + n_comp;
        for r in 0..b as usize {
            let q_row = q.slice_view(r * nh * hd, nh * hd);
            attn.launch_score_b1_htiled_wmma(
                stream, b1_scores, &q_row, &raw_kv, Some(&comp_kv),
                n_raw, 0, n_comp, n_head, head_dim, n_total,
            )?;
            attn.launch_softmax_only(
                stream, b1_scores, &sinks, b1_inv, n_head, n_raw, n_comp,
            )?;
            attn.launch_wsum_b1_htiled_ksplit_ldsv(
                stream, b1_partials, b1_scores, &raw_kv, Some(&comp_kv),
                n_head, head_dim, n_raw, n_comp, K_SPLIT,
            )?;
            attn.launch_reduce_partials_apply_inv(
                stream, b1_out, b1_partials, b1_inv, n_head, head_dim, K_SPLIT,
            )?;
        }
        Ok(())
    };

    // batched: shared-window WMMA at batch=B (one launch each).
    let run_batched = |stream: &Stream,
                       b: u32,
                       n_comp: u32,
                       bt_scores: &mut DeviceBuffer<f32>,
                       bt_out: &mut DeviceBuffer<f32>,
                       n_raw_per: &DeviceBuffer<i32>,
                       n_off_per: &DeviceBuffer<i32>,
                       n_comp_per: &DeviceBuffer<i32>|
     -> eyre::Result<()> {
        let n_total = n_raw + n_comp;
        attn.launch_score_batched_htiled_wmma_f16s(
            stream, bt_scores, &q, &raw_kv, Some(&comp_kv),
            n_raw_per, n_off_per, n_comp_per, None,
            n_head, head_dim, n_total, b, /*comp_kv_batch_stride=*/ 0,
        )?;
        attn.launch_softmax_wsum_batched_htiled_wmma_ldsv_f16s(
            stream, bt_out, bt_scores, &sinks, &raw_kv, Some(&comp_kv),
            n_raw_per, n_off_per, n_comp_per, n_head, head_dim, b,
            /*comp_kv_batch_stride=*/ 0,
        )?;
        Ok(())
    };

    // results[n_comp] = Vec<(B, loop_us, batched_us)>
    for &n_comp in &ncomp_list {
        // Stamp counters: every row shares the window.
        let raws = vec![n_raw as i32; bm];
        let offs = vec![0i32; bm];
        let comps = vec![n_comp as i32; bm];
        n_raw_per.copy_from_host(&raws)?;
        n_off_per.copy_from_host(&offs)?;
        n_comp_per.copy_from_host(&comps)?;
        if n_raw + n_comp > ATTN_SCORES_STRIDE {
            eprintln!(
                "  (skip n_comp={n_comp}: n_total {}>{ATTN_SCORES_STRIDE} batched stride)",
                n_raw + n_comp
            );
            continue;
        }

        eprintln!("\n### n_comp={n_comp} (n_total={}) ###", n_raw + n_comp);
        eprintln!(
            "   B | loop_b1 µs  C | batched µs  C | batched/loop | winner"
        );
        let mut loop0 = 0.0f32;
        let mut bt0 = 0.0f32;
        for (i, &b) in bs_list.iter().enumerate() {
            // warmup
            for _ in 0..warmup {
                run_loop_b1(&stream, b, n_comp, &mut b1_scores, &mut b1_partials, &mut b1_inv, &mut b1_out)?;
                run_batched(&stream, b, n_comp, &mut bt_scores, &mut bt_out, &n_raw_per, &n_off_per, &n_comp_per)?;
            }
            stream.synchronize()?;
            // time loop_b1
            let mut loop_ms: Vec<f32> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let s = Event::new()?; let e = Event::new()?;
                s.record(&stream)?;
                run_loop_b1(&stream, b, n_comp, &mut b1_scores, &mut b1_partials, &mut b1_inv, &mut b1_out)?;
                e.record(&stream)?; stream.synchronize()?;
                loop_ms.push(Event::elapsed_ms(&s, &e)?);
            }
            // time batched
            let mut bt_ms: Vec<f32> = Vec::with_capacity(iters);
            for _ in 0..iters {
                let s = Event::new()?; let e = Event::new()?;
                s.record(&stream)?;
                run_batched(&stream, b, n_comp, &mut bt_scores, &mut bt_out, &n_raw_per, &n_off_per, &n_comp_per)?;
                e.record(&stream)?; stream.synchronize()?;
                bt_ms.push(Event::elapsed_ms(&s, &e)?);
            }
            let loop_us = median(&mut loop_ms) * 1000.0;
            let bt_us = median(&mut bt_ms) * 1000.0;
            if i == 0 {
                loop0 = loop_us;
                bt0 = bt_us;
            }
            let c_loop = loop_us / loop0;
            let c_bt = bt_us / bt0;
            let ratio = bt_us / loop_us;
            let winner = if bt_us < loop_us { "batched" } else { "loop_b1" };
            eprintln!(
                "  {b:>2} | {loop_us:>8.1}  {c_loop:>4.2} | {bt_us:>8.1}  {c_bt:>4.2} | {ratio:>11.3} | {winner}"
            );
        }
    }

    // Optional correctness anchor at B=1 (batched must equal single).
    if std::env::var_os("BENCH_DIFF").is_some() {
        let n_comp = ncomp_list[0];
        let raws = vec![n_raw as i32; bm];
        let offs = vec![0i32; bm];
        let comps = vec![n_comp as i32; bm];
        n_raw_per.copy_from_host(&raws)?;
        n_off_per.copy_from_host(&offs)?;
        n_comp_per.copy_from_host(&comps)?;
        run_loop_b1(&stream, 1, n_comp, &mut b1_scores, &mut b1_partials, &mut b1_inv, &mut b1_out)?;
        run_batched(&stream, 1, n_comp, &mut bt_scores, &mut bt_out, &n_raw_per, &n_off_per, &n_comp_per)?;
        stream.synchronize()?;
        let mut a = vec![0f32; nh * hd];
        let mut c = vec![0f32; nh * hd];
        b1_out.copy_to_host(&mut a)?;
        bt_out.slice_view(0, nh * hd).copy_to_host(&mut c)?;
        let mut mx = 0.0f32;
        for (x, y) in a.iter().zip(c.iter()) { let d=(x-y).abs(); if d>mx {mx=d;} }
        eprintln!("\nBENCH_DIFF loop_b1 vs batched out max_abs = {mx:.3e} (zeros → both 0)");
    }

    eprintln!(
        "\nInterpretation: C_attn(B) is the attention-CORE cost multiplier. The\n\
         winner column tells a true B=k decode which kernel to use per window.\n\
         The full-forward verify cost is dominated by the FLAT projection/MoE\n\
         wall; this core is a minority of the ~35 ms decode at 4K, larger at depth."
    );
    Ok(())
}
