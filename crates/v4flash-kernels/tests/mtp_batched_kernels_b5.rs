//! Every `_batched` kernel the DSpark drafter runs at B = MTP_BLOCK = 5,
//! checked row-by-row against its known-good B=1 twin.
//!
//! Motivation: `f16.gemm_batched_wmma` (BM/BN = 64) silently returns zeros for
//! rows 1..4 at B=5 and leaves row 4's result in row 0. It reported no error.
//! That one kernel fed BOTH the drafter's mHC mixes and its router, so the
//! drafter routed every token to experts [0,1,2] with weight 0.5 and still
//! produced fluent-looking drafts. Any other batched kernel written for
//! prefill-sized batches can do the same, so they all get checked here.
//!
//!   cargo test -p v4flash-kernels --features v41 --test mtp_batched_kernels_b5 \
//!     -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{
    GROUP_DIM, HC_DIM, HC_MIX_DIM, N_EMBD, N_HC, N_HEAD, N_HEAD_DIM, N_LORA_Q, N_ROT, OUT_LOW,
    Q_FLAT, RANK, RMS_EPS,
};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::mtp_rope;
use v4flash_kernels::het::weights::MtpWeights;

const B: usize = 5;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn noise(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
    let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-12)
}

/// Compare `got` (batched, [B, w]) against `want` (B=1 twin, [B, w]).
fn check(name: &str, want: &[f32], got: &[f32], w: usize, fails: &mut Vec<String>) {
    let mut worst = 1.0f32;
    for j in 0..B {
        let c = cos(&want[j * w..(j + 1) * w], &got[j * w..(j + 1) * w]);
        if c < worst {
            worst = c;
        }
    }
    println!("{name:34} worst-row cos {worst:.6}");
    if worst < 0.999 {
        fails.push(format!("{name}: worst-row cos {worst:.6}"));
    }
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn batched_kernels_agree_with_b1_at_block_size() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let model = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));
    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let l = &w.layers[0];
    let s = &e.compute;
    let id = dev.id;
    let mut fails: Vec<String> = Vec::new();

    let dbuf = |n: usize| DeviceBuffer::<f32>::new(id, n).unwrap();
    let up = |v: &[f32]| {
        let mut b = DeviceBuffer::<f32>::new(id, v.len()).unwrap();
        b.copy_from_host(v).unwrap();
        b
    };

    // ---- rms_w.launch_weighted_batched (attn_norm over [B, N_EMBD]) ----
    {
        let ne = N_EMBD as usize;
        let x = up(&noise(B * ne, 1, 1.0));
        let mut gb = dbuf(B * ne);
        e.rms_w
            .launch_weighted_batched(s, &mut gb, &x, &l.attn_norm, N_EMBD, RMS_EPS, B as u32)
            .unwrap();
        let mut want = vec![0.0f32; B * ne];
        for j in 0..B {
            let xj = x.slice_view(j * ne, ne);
            let mut o = dbuf(ne);
            e.rms_w
                .launch_weighted(s, &mut o, &xj, &l.attn_norm, N_EMBD, RMS_EPS)
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * ne..(j + 1) * ne]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * ne];
        gb.copy_to_host(&mut got).unwrap();
        check("rms_w.launch_weighted_batched", &want, &got, ne, &mut fails);
    }

    // ---- rms_nw.launch_batched (the mHC flat norm over [B, HC_DIM]) ----
    {
        let hd = HC_DIM as usize;
        let x = up(&noise(B * hd, 2, 1.0));
        let mut gb = dbuf(B * hd);
        e.rms_nw
            .launch_batched(s, &mut gb, &x, 1, HC_DIM, RMS_EPS, B as u32)
            .unwrap();
        let mut want = vec![0.0f32; B * hd];
        for j in 0..B {
            let xj = x.slice_view(j * hd, hd);
            let mut o = dbuf(hd);
            e.rms_nw.launch(s, &mut o, &xj, 1, HC_DIM, RMS_EPS).unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * hd..(j + 1) * hd]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * hd];
        gb.copy_to_host(&mut got).unwrap();
        check("rms_nw.launch_batched", &want, &got, hd, &mut fails);
    }

    // ---- q8.matvec_batched (attn_q_b: [N_LORA_Q] -> [Q_FLAT]) ----
    {
        let k = N_LORA_Q as usize;
        let x = up(&noise(B * k, 3, 1.0));
        let mut xq = DeviceBuffer::<i8>::new(id, B * k).unwrap();
        let mut xs = dbuf(B * k.div_ceil(32));
        e.q8
            .quantize_input_batched(s, &mut xq, &mut xs, &x, N_LORA_Q, B as u32)
            .unwrap();
        let mut gb = dbuf(B * Q_FLAT as usize);
        e.q8
            .matvec_batched(s, &mut gb, &l.attn_q_b.buffer, &xq, &xs, Q_FLAT, N_LORA_Q, B as u32)
            .unwrap();
        let qf = Q_FLAT as usize;
        let mut want = vec![0.0f32; B * qf];
        for j in 0..B {
            let xj = x.slice_view(j * k, k);
            let mut q1 = DeviceBuffer::<i8>::new(id, k).unwrap();
            let mut s1 = dbuf(k.div_ceil(32));
            e.q8.quantize_input(s, &mut q1, &mut s1, &xj, N_LORA_Q).unwrap();
            let mut o = dbuf(qf);
            e.q8
                .matvec(s, &mut o, &l.attn_q_b.buffer, &q1, &s1, Q_FLAT, N_LORA_Q)
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * qf..(j + 1) * qf]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * qf];
        gb.copy_to_host(&mut got).unwrap();
        check("q8.matvec_batched", &want, &got, qf, &mut fails);
    }

    // ---- q8_grouped.matvec_grouped_batched (wo_a) ----
    {
        let qf = Q_FLAT as usize;
        let x = up(&noise(B * qf, 4, 1.0));
        let mut xq = DeviceBuffer::<i8>::new(id, B * qf).unwrap();
        let mut xs = dbuf(B * qf.div_ceil(32));
        e.q8
            .quantize_input_batched(s, &mut xq, &mut xs, &x, Q_FLAT, B as u32)
            .unwrap();
        let ol = OUT_LOW as usize;
        let mut gb = dbuf(B * ol);
        e.q8_grouped
            .matvec_grouped_batched(
                s, &mut gb, &l.attn_output_a.buffer, &xq, &xs, GROUP_DIM, RANK,
                v4flash_kernels::config::N_GROUPS, B as u32,
            )
            .unwrap();
        let mut want = vec![0.0f32; B * ol];
        for j in 0..B {
            let xj = x.slice_view(j * qf, qf);
            let mut q1 = DeviceBuffer::<i8>::new(id, qf).unwrap();
            let mut s1 = dbuf(qf.div_ceil(32));
            e.q8.quantize_input(s, &mut q1, &mut s1, &xj, Q_FLAT).unwrap();
            let mut o = dbuf(ol);
            e.q8_grouped
                .matvec_grouped(
                    s, &mut o, &l.attn_output_a.buffer, &q1, &s1, GROUP_DIM, RANK,
                    v4flash_kernels::config::N_GROUPS,
                )
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * ol..(j + 1) * ol]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * ol];
        gb.copy_to_host(&mut got).unwrap();
        check("q8_grouped.matvec_grouped_batched", &want, &got, ol, &mut fails);
    }

    // ---- rope.launch_forward_batched (per-row positions) ----
    {
        let qf = Q_FLAT as usize;
        let base = noise(B * qf, 5, 1.0);
        let rope = mtp_rope();
        let mut pos_b = DeviceBuffer::<i32>::new(id, B).unwrap();
        let positions: Vec<i32> = (0..B).map(|j| 200 + j as i32).collect();
        pos_b.copy_from_host(&positions).unwrap();
        let mut gb = up(&base);
        e.rope
            .launch_forward_batched(s, &mut gb, &pos_b, N_HEAD, N_HEAD_DIM, N_ROT, B as u32, &rope)
            .unwrap();
        let mut want = vec![0.0f32; B * qf];
        for j in 0..B {
            let mut o = up(&base[j * qf..(j + 1) * qf]);
            let mut p1 = DeviceBuffer::<u32>::new(id, 1).unwrap();
            p1.copy_from_host(&[positions[j] as u32]).unwrap();
            e.rope
                .launch_forward_pdev(s, &mut o, &p1, N_HEAD, N_HEAD_DIM, N_ROT, &rope)
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * qf..(j + 1) * qf]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * qf];
        gb.copy_to_host(&mut got).unwrap();
        check("rope.launch_forward_batched", &want, &got, qf, &mut fails);
    }

    // ---- hc_weighted.launch_batched (hc_pre) ----
    {
        let hd = HC_DIM as usize;
        let ne = N_EMBD as usize;
        let x = up(&noise(B * hd, 6, 1.0));
        let wts = up(&noise(B * HC_MIX_DIM as usize, 7, 1.0));
        let mut gb = dbuf(B * ne);
        e.hc_weighted
            .launch_batched(s, &mut gb, &x, &wts, N_EMBD, N_HC, HC_MIX_DIM, B as u32)
            .unwrap();
        let mut want = vec![0.0f32; B * ne];
        for j in 0..B {
            let xj = x.slice_view(j * hd, hd);
            let wj = wts.slice_view(j * HC_MIX_DIM as usize, N_HC as usize);
            let mut o = dbuf(ne);
            e.hc_weighted.launch(s, &mut o, &xj, &wj, N_EMBD, N_HC).unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * ne..(j + 1) * ne]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * ne];
        gb.copy_to_host(&mut got).unwrap();
        check("hc_weighted.launch_batched", &want, &got, ne, &mut fails);
    }

    // ---- hc_sinkhorn.launch_batched ----
    {
        let m = HC_MIX_DIM as usize;
        let mix = up(&noise(B * m, 8, 1.0));
        let mut gb = dbuf(B * m);
        e.hc_sinkhorn
            .launch_batched(
                s, &mut gb, &mix, &l.hc_attn_scale, &l.hc_attn_base, N_HC,
                v4flash_kernels::config::SINKHORN_ITERS,
                v4flash_kernels::config::SINKHORN_EPS, B as u32,
            )
            .unwrap();
        let mut want = vec![0.0f32; B * m];
        for j in 0..B {
            let mj = mix.slice_view(j * m, m);
            let mut o = dbuf(m);
            e.hc_sinkhorn
                .launch(
                    s, &mut o, &mj, &l.hc_attn_scale, &l.hc_attn_base, N_HC,
                    v4flash_kernels::config::SINKHORN_ITERS,
                    v4flash_kernels::config::SINKHORN_EPS,
                )
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * m..(j + 1) * m]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * m];
        gb.copy_to_host(&mut got).unwrap();
        check("hc_sinkhorn.launch_batched", &want, &got, m, &mut fails);
    }

    // ---- hc_post.launch_from_split_batched ----
    {
        let hd = HC_DIM as usize;
        let ne = N_EMBD as usize;
        let resid = up(&noise(B * hd, 9, 1.0));
        let blk = up(&noise(B * ne, 10, 1.0));
        let split = up(&noise(B * HC_MIX_DIM as usize, 11, 0.3));
        let mut gb = dbuf(B * hd);
        e.hc_post
            .launch_from_split_batched(
                s, &mut gb, &blk, &resid, &split, N_HC, N_EMBD, N_HC, B as u32,
            )
            .unwrap();
        let mut want = vec![0.0f32; B * hd];
        for j in 0..B {
            let rj = resid.slice_view(j * hd, hd);
            let bj = blk.slice_view(j * ne, ne);
            let sj = split.slice_view(j * HC_MIX_DIM as usize, HC_MIX_DIM as usize);
            let mut o = dbuf(hd);
            e.hc_post
                .launch_from_split(s, &mut o, &bj, &rj, &sj, N_HC, N_EMBD, N_HC)
                .unwrap();
            s.synchronize().unwrap();
            o.copy_to_host(&mut want[j * hd..(j + 1) * hd]).unwrap();
        }
        s.synchronize().unwrap();
        let mut got = vec![0.0f32; B * hd];
        gb.copy_to_host(&mut got).unwrap();
        check("hc_post.launch_from_split_batched", &want, &got, hd, &mut fails);
    }

    assert!(
        fails.is_empty(),
        "batched kernels disagree with their B=1 twins at B={B}:\n  {}",
        fails.join("\n  ")
    );
}
