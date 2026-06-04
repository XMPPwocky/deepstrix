//! indexer_topk kernel synthetic oracle.
//!
//! Validates the greedy K-of-N top-K selection against a CPU reference that
//! mirrors ds4.c:7022-7032 exactly:
//!
//!     for k in 0..top_k:
//!         best = 0; best_score = -INF
//!         for c in 0..n_comp:
//!             if !allowed[c] && scores[c] > best_score:
//!                 best = c; best_score = scores[c]
//!         allowed[best] = true
//!         selected[k] = best
//!
//! The strict `>` comparison is load-bearing: tied scores resolve to the
//! FIRST-encountered (smallest) comp index. Our wave-parallel reduction must
//! preserve that ordering.
//!
//! Test cases cover:
//!  - n_comp < top_k          (degenerate: selects all rows)
//!  - n_comp == top_k         (degenerate: selects all rows)
//!  - n_comp slightly > top_k (one row dropped — boundary)
//!  - n_comp >> top_k         (production-scale, 32K → 512)
//!  - Heavy tie patterns      (verifies first-index tie-break)

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{IndexerTopk, IndexerTopkBitonic, INDEXER_TOP_K};

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

fn lcg_step(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    let v = (*s >> 8) as f32 / (1u32 << 24) as f32; // [0, 1)
    v * 2.0 - 1.0                                    // [-1, 1)
}

/// Bit-exact CPU port of ds4.c:7022-7032. Returns (selected, allowed).
fn cpu_topk(scores: &[f32], top_k: u32) -> (Vec<i32>, Vec<bool>) {
    let n_comp = scores.len();
    let k_actual = (top_k as usize).min(n_comp);
    let mut allowed = vec![false; n_comp];
    let mut selected = vec![-1i32; top_k as usize];
    for k in 0..k_actual {
        let mut best: usize = 0;
        let mut best_score = f32::NEG_INFINITY;
        for c in 0..n_comp {
            // Match ds4: strict `>`, first-index wins ties (because earlier
            // iterations of c lock in best, and subsequent equal scores don't
            // overwrite).
            if !allowed[c] && scores[c] > best_score {
                best = c;
                best_score = scores[c];
            }
        }
        allowed[best] = true;
        selected[k] = best as i32;
    }
    (selected, allowed)
}

fn run_case(
    kernel: &IndexerTopk,
    stream: &Stream,
    device: Device,
    label: &str,
    scores: Vec<f32>,
    top_k: u32,
) -> eyre::Result<()> {
    let n_comp = scores.len() as u32;
    let (cpu_selected, cpu_allowed) = cpu_topk(&scores, top_k);

    let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_comp as usize)?;
    d_scores.copy_from_host(&scores)?;

    let mut d_selected: DeviceBuffer<i32> = DeviceBuffer::new(device.id, top_k as usize)?;
    let n_words = ((n_comp + 31) / 32) as usize;
    let mut d_bits: DeviceBuffer<u32> = DeviceBuffer::new(device.id, n_words)?;

    kernel.launch(stream, &mut d_selected, &mut d_bits, &d_scores, n_comp, top_k)?;
    stream.synchronize()?;

    let mut got_selected = vec![0i32; top_k as usize];
    let mut got_bits = vec![0u32; n_words];
    d_selected.copy_to_host(&mut got_selected)?;
    d_bits.copy_to_host(&mut got_bits)?;

    // Compare selected[]. The ordering must match (selection order = greedy
    // descending; sentinel -1 in the tail).
    let mut sel_mismatches = 0usize;
    for k in 0..(top_k as usize) {
        if got_selected[k] != cpu_selected[k] {
            if sel_mismatches < 5 {
                eprintln!(
                    "  [{label}] selected[{k}]: gpu={} cpu={} (gpu score={}, cpu score={})",
                    got_selected[k],
                    cpu_selected[k],
                    if got_selected[k] >= 0 { scores[got_selected[k] as usize] } else { 0.0 },
                    if cpu_selected[k] >= 0 { scores[cpu_selected[k] as usize] } else { 0.0 },
                );
            }
            sel_mismatches += 1;
        }
    }

    // Compare bitmap. Bit c is set iff cpu_allowed[c].
    let mut bit_mismatches = 0usize;
    for c in 0..(n_comp as usize) {
        let gpu_bit = (got_bits[c >> 5] >> (c & 31)) & 1u32 != 0;
        if gpu_bit != cpu_allowed[c] {
            bit_mismatches += 1;
        }
    }

    eprintln!(
        "[{label}] n_comp={n_comp} top_k={top_k}: selected mismatches={sel_mismatches}/{top_k}, bit mismatches={bit_mismatches}/{n_comp}"
    );
    if sel_mismatches != 0 || bit_mismatches != 0 {
        return Err(eyre!(
            "[{label}] mismatch: selected={sel_mismatches} bits={bit_mismatches}"
        ));
    }
    Ok(())
}

fn run_case_bitonic(
    kernel: &IndexerTopkBitonic,
    stream: &Stream,
    device: Device,
    label: &str,
    scores: Vec<f32>,
    top_k: u32,
) -> eyre::Result<()> {
    let n_comp = scores.len() as u32;
    let (cpu_selected, cpu_allowed) = cpu_topk(&scores, top_k);

    let mut d_scores: DeviceBuffer<f32> = DeviceBuffer::new(device.id, n_comp as usize)?;
    d_scores.copy_from_host(&scores)?;

    let mut d_selected: DeviceBuffer<i32> = DeviceBuffer::new(device.id, top_k as usize)?;
    let n_words = ((n_comp + 31) / 32) as usize;
    let mut d_bits: DeviceBuffer<u32> = DeviceBuffer::new(device.id, n_words)?;
    // Scratch sized for the worst case (n_comp/4096 chunks × top_k).
    let max_chunks = (n_comp + 4095) / 4096;
    let mut d_scratch: DeviceBuffer<u32> =
        DeviceBuffer::new(device.id, (max_chunks * top_k).max(1) as usize)?;

    kernel.launch(
        stream,
        &mut d_selected,
        &mut d_bits,
        &mut d_scratch,
        &d_scores,
        n_comp,
        top_k,
    )?;
    stream.synchronize()?;

    let mut got_selected = vec![0i32; top_k as usize];
    let mut got_bits = vec![0u32; n_words];
    d_selected.copy_to_host(&mut got_selected)?;
    d_bits.copy_to_host(&mut got_bits)?;

    let mut sel_mismatches = 0usize;
    for k in 0..(top_k as usize) {
        if got_selected[k] != cpu_selected[k] {
            if sel_mismatches < 5 {
                eprintln!(
                    "  [{label}] selected[{k}]: gpu={} cpu={} (gpu score={}, cpu score={})",
                    got_selected[k],
                    cpu_selected[k],
                    if got_selected[k] >= 0 { scores[got_selected[k] as usize] } else { 0.0 },
                    if cpu_selected[k] >= 0 { scores[cpu_selected[k] as usize] } else { 0.0 },
                );
            }
            sel_mismatches += 1;
        }
    }
    let mut bit_mismatches = 0usize;
    for c in 0..(n_comp as usize) {
        let gpu_bit = (got_bits[c >> 5] >> (c & 31)) & 1u32 != 0;
        if gpu_bit != cpu_allowed[c] {
            bit_mismatches += 1;
        }
    }

    eprintln!(
        "[{label}] n_comp={n_comp} top_k={top_k}: selected mismatches={sel_mismatches}/{top_k}, bit mismatches={bit_mismatches}/{n_comp}"
    );
    if sel_mismatches != 0 || bit_mismatches != 0 {
        return Err(eyre!(
            "[{label}] mismatch: selected={sel_mismatches} bits={bit_mismatches}"
        ));
    }
    Ok(())
}

#[test]
#[ignore]
fn indexer_topk_synthetic() -> eyre::Result<()> {
    install_panic_handler()?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let kernel = IndexerTopk::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;
    let top_k = INDEXER_TOP_K;

    // Case 1: random scores at production size (32K).
    {
        let n_comp = 32 * 1024u32;
        let mut seed = 0xdeadbeefu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case(&kernel, &stream, device, "random-32K", scores, top_k)?;
    }

    // Case 2: random scores at boundary (n_comp = top_k + 1).
    {
        let n_comp = top_k + 1;
        let mut seed = 0xcafef00du32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case(&kernel, &stream, device, "boundary+1", scores, top_k)?;
    }

    // Case 3: degenerate n_comp == top_k (selects everything).
    {
        let n_comp = top_k;
        let mut seed = 0x1234abcdu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case(&kernel, &stream, device, "equal", scores, top_k)?;
    }

    // Case 4: degenerate n_comp < top_k. (Caller would early-permit; verify
    // kernel handles it defensively.)
    {
        let n_comp = 100u32;
        let mut seed = 0xfeedfaceu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case(&kernel, &stream, device, "small", scores, top_k)?;
    }

    // Case 5: heavy tie pattern. All scores equal — selection MUST be 0..top_k
    // in order (first-index wins every tie).
    {
        let n_comp = 2048u32;
        let scores: Vec<f32> = vec![0.5f32; n_comp as usize];
        run_case(&kernel, &stream, device, "all-tied", scores, top_k)?;
    }

    // Case 6: clustered ties. 256 distinct score levels, 128 indices per level.
    // Within each level the kernel must select indices in ascending order.
    {
        let n_comp = 256 * 128u32;
        let scores: Vec<f32> = (0..n_comp).map(|i| (i / 128) as f32 * 0.01).collect();
        run_case(&kernel, &stream, device, "clustered-ties", scores, top_k)?;
    }

    // --- Bitonic variant: same cases, must match CPU exactly. ---
    let kernel_b = IndexerTopkBitonic::for_arch(&arch)?;
    {
        let n_comp = 32 * 1024u32;
        let mut seed = 0xdeadbeefu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case_bitonic(&kernel_b, &stream, device, "bit:random-32K", scores, top_k)?;
    }
    {
        let n_comp = top_k + 1;
        let mut seed = 0xcafef00du32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case_bitonic(&kernel_b, &stream, device, "bit:boundary+1", scores, top_k)?;
    }
    {
        let n_comp = top_k;
        let mut seed = 0x1234abcdu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case_bitonic(&kernel_b, &stream, device, "bit:equal", scores, top_k)?;
    }
    {
        let n_comp = 100u32;
        let mut seed = 0xfeedfaceu32;
        let scores: Vec<f32> = (0..n_comp).map(|_| lcg_step(&mut seed)).collect();
        run_case_bitonic(&kernel_b, &stream, device, "bit:small", scores, top_k)?;
    }
    {
        let n_comp = 2048u32;
        let scores: Vec<f32> = vec![0.5f32; n_comp as usize];
        run_case_bitonic(&kernel_b, &stream, device, "bit:all-tied", scores, top_k)?;
    }
    {
        let n_comp = 256 * 128u32;
        let scores: Vec<f32> = (0..n_comp).map(|i| (i / 128) as f32 * 0.01).collect();
        run_case_bitonic(&kernel_b, &stream, device, "bit:clustered-ties", scores, top_k)?;
    }

    Ok(())
}
