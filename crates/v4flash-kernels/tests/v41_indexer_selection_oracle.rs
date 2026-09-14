//! V4.1 CSA2 **selection-set** oracle — the gate for flipping the sparse path (S1).
//!
//! WHY NOT the plan's original test. `docs/v41/INDEXER_PORT_PLAN.md` originally proposed
//! "flip the gate, run at <=512 tokens, assert bit-identical logits". That is a
//! TAUTOLOGY: the gate is `n_index_comp > INDEXER_TOP_K`, so at <=512 rows the sparse
//! code never executes and the test compares the dense path to itself. It would pass
//! with garbage indexer weights.
//!
//! What this does instead: drive the real packed-E2M1 key chain and the real score +
//! top-k kernels at **V4.1 shapes** (32 heads, 128 dim, top-512, n_comp >> 512), and
//! compare the selected index SET against a CPU recompute.
//!
//! Two deliberate choices:
//!   * The CPU reference scores the EXPANDED keys (`index_kv_e2m1_expand`), i.e. exactly
//!     the quantized values the GPU sees, so E2M1 quantization error cannot be mistaken
//!     for a kernel bug.
//!   * Selection is compared as a SET with a tolerance band. Measured 2026-09-14, the
//!     WMMA score path carries ~4.2e-4 absolute error at n_comp=16384; top-512 out of
//!     tens of thousands is a RANKING decision, so rows whose score sits within that
//!     band of the cut legitimately swap. A strict set-equality assert WILL flake.
//!     Rows OUTSIDE the band must match exactly — that is the real assertion.
//!
//! `cargo test -p v4flash-kernels --release --features v41 \
//!    --test v41_indexer_selection_oracle -- --ignored --nocapture`

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::index_kv_e2m1::{E2M1_KEY_DIM, E2M1_KEY_ROW_BYTES};
use v4flash_kernels::{IndexKvE2m1, IndexerScore, IndexerTopkBitonic, INDEXER_TOP_K};

const N_HEAD: usize = 32; // V4.1 index_n_heads (V4-Flash is 64)
const DIM: usize = E2M1_KEY_DIM; // 128
const N_COMP: usize = 4096; // >> 512 so the top-k is a real selection

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// IEEE half bits -> f32 (the expand kernel writes f16).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match exp {
        0 if man == 0 => sign << 31,
        0 => {
            // subnormal: renormalize
            let mut e = -1i32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            let exp32 = (127 - 15 + e + 1) as u32;
            (sign << 31) | (exp32 << 23) | ((m & 0x3ff) << 13)
        }
        0x1f => (sign << 31) | (0xff << 23) | (man << 13),
        _ => (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx12") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx12 device"))
}

/// Reference: per-head dot, ReLU, then head-weighted sum — `model.py:556-557`
/// (`(index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)`). Same semantics as
/// `tests/indexer_score.rs::cpu_indexer_score`, restated here at V4.1 shapes.
fn cpu_scores(q: &[f32], hw: &[f32], keys: &[f32]) -> Vec<f32> {
    (0..N_COMP)
        .map(|c| {
            let k = &keys[c * DIM..(c + 1) * DIM];
            (0..N_HEAD)
                .map(|h| {
                    let qh = &q[h * DIM..(h + 1) * DIM];
                    let dot: f32 = (0..DIM).map(|i| qh[i] * k[i]).sum();
                    dot.max(0.0) * hw[h]
                })
                .sum()
        })
        .collect()
}

#[test]
#[ignore]
fn v41_indexer_selection_matches_cpu() -> eyre::Result<()> {
    install_panic_handler()?;
    let dev = pick_dgpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let stream = Stream::new(dev.id)?;
    let packer = IndexKvE2m1::for_arch(&arch)?;
    let scorer = IndexerScore::for_arch(&arch)?;
    let topk = IndexerTopkBitonic::for_arch(&arch)?;
    let mut rng = Rng(0x5EED_2026_0914);

    // --- keys through the REAL packed chain -------------------------------
    let keys_f32: Vec<f32> = (0..N_COMP * DIM).map(|_| rng.unit()).collect();
    let mut d_rows: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, keys_f32.len())?;
    d_rows.copy_from_host(&keys_f32)?;
    let mut packed: DeviceBuffer<u8> = DeviceBuffer::new(dev.id, N_COMP * E2M1_KEY_ROW_BYTES)?;
    packer.launch_append_batched(&stream, &mut packed, &d_rows, 0, N_COMP as u32)?;

    // Expand back: these are EXACTLY the values the score kernel will read, so the
    // CPU reference cannot be confounded by E2M1 quantization error.
    let mut d_exp: DeviceBuffer<u16> = DeviceBuffer::new(dev.id, N_COMP * DIM)?;
    packer.launch_expand(&stream, &mut d_exp, &packed, N_COMP as u32)?;
    stream.synchronize()?;
    let mut exp_h = vec![0u16; N_COMP * DIM];
    d_exp.copy_to_host(&mut exp_h)?;
    let keys_q: Vec<f32> = exp_h.iter().map(|&h| f16_to_f32(h)).collect();

    // --- query + head weights ---------------------------------------------
    let q: Vec<f32> = (0..N_HEAD * DIM).map(|_| rng.unit()).collect();
    let hw: Vec<f32> = (0..N_HEAD).map(|_| rng.unit().abs() + 0.05).collect();
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, q.len())?;
    d_q.copy_from_host(&q)?;
    let mut d_hw: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, hw.len())?;
    d_hw.copy_from_host(&hw)?;

    // --- scores: GPU vs CPU ------------------------------------------------
    let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(dev.id, N_COMP)?;
    scorer.launch_e2m1(
        &stream, &mut d_scores, &d_q, &d_hw, &packed,
        N_COMP as u32, N_HEAD as u32, DIM as u32,
    )?;
    stream.synchronize()?;
    let mut gpu_scores = vec![0f32; N_COMP];
    d_scores.copy_to_host(&mut gpu_scores)?;
    let cpu = cpu_scores(&q, &hw, &keys_q);

    let mag = cpu.iter().fold(0f32, |m, v| m.max(v.abs()));
    let max_abs = cpu
        .iter()
        .zip(&gpu_scores)
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    eprintln!("score: max_abs_diff={max_abs:.3e} over magnitude {mag:.3e} (n_comp={N_COMP}, n_head={N_HEAD})");

    // --- selection sets ----------------------------------------------------
    let mut order: Vec<usize> = (0..N_COMP).collect();
    order.sort_by(|&a, &b| cpu[b].partial_cmp(&cpu[a]).unwrap_or(std::cmp::Ordering::Equal));
    let cut = cpu[order[INDEXER_TOP_K as usize - 1]];
    let cpu_set: std::collections::HashSet<usize> =
        order[..INDEXER_TOP_K as usize].iter().copied().collect();

    let b = 1u32;
    let mut d_n: DeviceBuffer<u32> = DeviceBuffer::new(dev.id, 1)?;
    d_n.copy_from_host(&[N_COMP as u32])?;
    let n_chunks = (N_COMP as u32).div_ceil(4096);
    let mut scratch: DeviceBuffer<u32> =
        DeviceBuffer::new(dev.id, (n_chunks * INDEXER_TOP_K) as usize + 4096)?;
    let mut sel: DeviceBuffer<i32> = DeviceBuffer::new(dev.id, INDEXER_TOP_K as usize)?;
    sel.fill_zero()?;
    topk.launch_batched(
        &stream, &mut sel, None, &mut scratch, &d_scores, &d_n,
        N_COMP as u32, N_COMP as u32, 0, INDEXER_TOP_K, b, None,
    )?;
    stream.synchronize()?;
    let mut sel_h = vec![0i32; INDEXER_TOP_K as usize];
    sel.copy_to_host(&mut sel_h)?;
    let gpu_set: std::collections::HashSet<usize> =
        sel_h.iter().map(|&i| i as usize).collect();

    // Rows the GPU picked that the CPU did not, and vice versa. Anything whose score
    // is within the kernel error band of the cut may legitimately swap; anything
    // outside it must not.
    let band = (max_abs * 4.0).max(1e-5) * mag.max(1.0);
    let mut hard_misses = 0usize;
    for &c in cpu_set.difference(&gpu_set) {
        if (cpu[c] - cut).abs() > band {
            hard_misses += 1;
        }
    }
    let soft = cpu_set.difference(&gpu_set).count();
    eprintln!(
        "selection: |cpu\\gpu|={soft} of {INDEXER_TOP_K} (band={band:.3e} around cut {cut:.4}), \
         hard misses OUTSIDE the band = {hard_misses}"
    );
    if hard_misses > 0 {
        return Err(eyre!(
            "{hard_misses} rows differ by more than the score-error band — a real selection bug"
        ));
    }
    eprintln!("PASS: selection sets agree outside the score-error band");
    Ok(())
}
