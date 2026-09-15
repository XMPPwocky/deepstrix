//! Does `f16.gemm_batched_wmma` agree with `f16.matvec` at a SMALL batch?
//!
//! The DSpark drafter runs every kernel at B = MTP_BLOCK = 5, far below the
//! B=512 prefill sizes this WMMA GEMM was written for (BM/BN are 64). Its
//! drafter router came out constant — every token picking experts [0,1,2] with
//! weight 0.5, which is a uniform softmax — so the question is whether this
//! kernel is even valid at B=5, and what layout it writes.
//!
//!   cargo test -p v4flash-kernels --features v41 --test f16_gemm_batched_small_b \
//!     -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::N_EMBD;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::weights::MtpWeights;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn gemm_batched_matches_matvec_at_b5() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let model = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));
    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    let gate = &w.layers[0].ffn_gate_inp;
    let n_exp = v4flash_core::hf_v41::MTP_N_EXPERT as u32;

    const B: usize = 5;
    let ne = N_EMBD as usize;
    let mut xs = vec![0.0f32; B * ne];
    let mut st = 12345u32;
    for v in xs.iter_mut() {
        st ^= st << 13;
        st ^= st >> 17;
        st ^= st << 5;
        *v = ((st >> 8) as f32 / (1u32 << 23) as f32 - 1.0) * 0.5;
    }
    let mut x = DeviceBuffer::<f32>::new(dev.id, B * ne).unwrap();
    x.copy_from_host(&xs).unwrap();

    // Reference: one matvec per row.
    let mut want = vec![0.0f32; B * n_exp as usize];
    for j in 0..B {
        let xj = x.slice_view(j * ne, ne);
        let mut o = DeviceBuffer::<f32>::new(dev.id, n_exp as usize).unwrap();
        e.f16
            .matvec(&e.compute, &mut o, &gate.buffer, &xj, n_exp, N_EMBD)
            .expect("matvec");
        e.compute.synchronize().unwrap();
        o.copy_to_host(&mut want[j * n_exp as usize..(j + 1) * n_exp as usize])
            .unwrap();
    }

    let mut got = DeviceBuffer::<f32>::new(dev.id, B * n_exp as usize).unwrap();
    e.f16
        .gemm_batched_wmma(&e.compute, &mut got, &gate.buffer, &x, n_exp, N_EMBD, B as u32)
        .expect("gemm_batched_wmma");
    e.compute.synchronize().unwrap();
    let mut g = vec![0.0f32; B * n_exp as usize];
    got.copy_to_host(&mut g).unwrap();

    let ne_exp = n_exp as usize;
    for j in 0..B {
        let a = &want[j * ne_exp..(j + 1) * ne_exp];
        let b = &g[j * ne_exp..(j + 1) * ne_exp];
        let dot: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
        let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!(
            "row {j}: |matvec| {na:.5} |gemm| {nb:.5} cos {:.6}",
            dot / (na * nb).max(1e-12)
        );
    }
    for j in 0..B {
        let a = &want[j * ne_exp..(j + 1) * ne_exp];
        let b = &g[j * ne_exp..(j + 1) * ne_exp];
        let dot: f32 = a.iter().zip(b).map(|(p, q)| p * q).sum();
        let na = a.iter().map(|v| v * v).sum::<f32>().sqrt();
        let nb = b.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cos = dot / (na * nb).max(1e-12);
        assert!(
            cos > 0.999,
            "row {j}: gemm_batched_wmma disagrees with matvec (cos {cos:.6}, \
             |matvec| {na:.5} |gemm| {nb:.5}) — the drafter runs this at B=5"
        );
    }
}
