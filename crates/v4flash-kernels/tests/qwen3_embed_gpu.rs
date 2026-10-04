//! Embed-phase GPU gates (docs/v41/EMBED_PHASE_DESIGN.md §10), dGPU only.
//!
//! `#[ignore]`: they drive the dGPU, so they need a GPU window (the hub
//! down -- it leaves the dGPU ~170 MiB free). Run with
//!
//! ```text
//! CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test qwen3_embed_gpu -- --ignored --nocapture --test-threads=1
//! ```
//!
//! - `tiny_gpu_matches_cpu`: a synthetic tiny Qwen3 (no weights needed), GPU
//!   forward vs the CPU oracle, sub-batches of 8 rows so inputs straddle
//!   sub-batches and sub-batches hold several inputs. Gate E2's machinery.
//! - `real_gpu_matches_cpu` (gate E2 proper): `QWEN3_EMBED_GGUF=<Q8_0 gguf>`;
//!   min cosine >= 0.9999 vs the CPU oracle on a few short inputs.
//! - `loan_round_trip`: a plain device buffer as the donor; lend it, let the
//!   forward clobber it, give it back, compare every byte.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::qwen3_embed::{cosine, cpu_forward, testing, Qwen3EmbedModel};
use v4flash_core::MappedGguf;
use v4flash_hip::{Device, DeviceBuffer, PinnedBuffer, Stream};
use v4flash_kernels::dgpu_loan::{Donor, Loan, LoanAlloc};
use v4flash_kernels::qwen3_embed::{run, EmbedBuffers, EmbedSizing, Qwen3EmbedKernels};

fn dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 device"))
}

fn tmp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("qwen3-embed-gpu-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// GPU embeddings of `inputs` (finished, full dims) with memory from one
/// plain allocation.
fn gpu_embed(dev: Device, model: &Qwen3EmbedModel, file: &MappedGguf, inputs: &[Vec<u32>], sizing: EmbedSizing) -> eyre::Result<Vec<Vec<f32>>> {
    dev.set_current()?;
    let k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    let total = sizing.total_bytes(&model.cfg, &model.slot);
    let backing = DeviceBuffer::<u8>::new(dev.id, total)?;
    let mut alloc = LoanAlloc::over(vec![backing.slice_view(0, total)]);
    let mut bufs = EmbedBuffers::carve(&mut alloc, sizing, &model.cfg, &model.slot)?;
    let mut host = [PinnedBuffer::<u8>::new(model.slot.bytes)?, PinnedBuffer::<u8>::new(model.slot.bytes)?];
    let (compute, copy) = (Stream::new(dev.id)?, Stream::new(dev.id)?);
    let refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    let (last, tm) = run(&k, model, file, &mut bufs, &mut host, &compute, &copy, &refs, &mut || {})?;
    eprintln!("gpu forward: {tm:?}");
    drop(backing);
    Ok(last.iter().map(|r| model.finish(r, None)).collect())
}

fn compare(gpu: &[Vec<f32>], cpu: &[Vec<f32>]) -> f64 {
    let mut min = f64::INFINITY;
    for (i, (g, c)) in gpu.iter().zip(cpu).enumerate() {
        let cs = cosine(g, c);
        eprintln!("input {i}: cos {cs:.7}");
        min = min.min(cs);
    }
    min
}

#[test]
#[ignore]
fn tiny_gpu_matches_cpu() -> eyre::Result<()> {
    let dev = dgpu()?;
    let p = tmp("tiny").join("tiny.gguf");
    testing::write_synthetic(&p, &testing::tiny_config(), 23)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let eos = model.eos_id;
    let inputs: Vec<Vec<u32>> = vec![
        (1..20).chain([eos]).collect(),
        vec![7, eos],
        vec![eos],
        (30..43).chain([eos]).collect(),
        (100..125).chain([eos]).collect(),
    ];
    let cpu = cpu_forward(&model, &file, &inputs)?;
    let gpu = gpu_embed(dev, &model, &file, &inputs, EmbedSizing { phase_tokens: 128, sub_rows: 8 })?;
    let min = compare(&gpu, &cpu);
    assert!(min >= 0.9999, "min cosine {min}");
    // Packing invariance on the GPU: alone == packed.
    let alone = gpu_embed(dev, &model, &file, &inputs[3..4], EmbedSizing { phase_tokens: 128, sub_rows: 8 })?;
    let cs = cosine(&alone[0], &gpu[3]);
    assert!(cs >= 0.99999, "alone vs packed {cs}");
    Ok(())
}

#[test]
#[ignore]
fn real_gpu_matches_cpu() -> eyre::Result<()> {
    let Ok(path) = std::env::var("QWEN3_EMBED_GGUF") else {
        eprintln!("QWEN3_EMBED_GGUF unset: skipped");
        return Ok(());
    };
    let dev = dgpu()?;
    let file = MappedGguf::open(&path)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let vocab = v4flash_core::tokenizer::BpeVocab::from_gguf(file.gguf())?;
    let texts = [
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:What is the capital of China?",
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other.",
        "def add(a, b):\n    return a + b\n",
        "日本の首都は東京です。",
    ];
    let inputs: Vec<Vec<u32>> = texts
        .iter()
        .map(|t| vocab.encode_qwen2(t).into_iter().map(|i| i as u32).chain([model.eos_id]).collect())
        .collect();
    let t0 = std::time::Instant::now();
    let cpu = cpu_forward(&model, &file, &inputs)?;
    eprintln!("cpu oracle: {:.1} s", t0.elapsed().as_secs_f64());
    let gpu = gpu_embed(dev, &model, &file, &inputs, EmbedSizing { phase_tokens: 1024, sub_rows: 64 })?;
    let min = compare(&gpu, &cpu);
    assert!(min >= 0.9999, "gate E2: min cosine {min} < 0.9999");
    Ok(())
}

#[test]
#[ignore]
fn loan_round_trip() -> eyre::Result<()> {
    let dev = dgpu()?;
    dev.set_current()?;
    let p = tmp("loan").join("tiny.gguf");
    testing::write_synthetic(&p, &testing::tiny_config(), 29)?;
    let file = MappedGguf::open(&p)?;
    let model = Qwen3EmbedModel::from_gguf(&file)?;
    let sizing = EmbedSizing { phase_tokens: 128, sub_rows: 16 };
    let sizes: Vec<usize> = sizing.buffer_sizes(&model.cfg, &model.slot).iter().map(|(_, b)| *b).collect();
    // Two donors with a recognizable pattern; the first is too small for the
    // ring slots, so placement must spill into the second.
    let mk = |n: usize, salt: u8| -> eyre::Result<(DeviceBuffer<u8>, Vec<u8>)> {
        let mut b = DeviceBuffer::<u8>::new(dev.id, n)?;
        let pat: Vec<u8> = (0..n).map(|i| (i as u8).wrapping_mul(31).wrapping_add(salt)).collect();
        b.copy_from_host(&pat)?;
        Ok((b, pat))
    };
    let (d0, p0) = mk(64 * 1024, 3)?;
    let (d1, p1) = mk(sizing.total_bytes(&model.cfg, &model.slot) + (1 << 20), 5)?;
    let donors = vec![
        Donor { name: "d0".into(), view: d0.slice_view(0, d0.len()) },
        Donor { name: "d1".into(), view: d1.slice_view(0, d1.len()) },
    ];
    let image = tmp("loan").join("embed-loan.img");
    let mut loan = Loan::new(donors, &sizes, image, 1 << 20)?;
    eprintln!("donors {:?}", loan.donor_summary());
    let stream = Stream::new(dev.id)?;
    loan.write_image(&stream)?;
    let mut alloc = loan.allocator()?;
    let mut bufs = EmbedBuffers::carve(&mut alloc, sizing, &model.cfg, &model.slot)?;
    let k = Qwen3EmbedKernels::for_arch("gfx1201")?;
    let mut host = [PinnedBuffer::<u8>::new(model.slot.bytes.max(1 << 20))?, PinnedBuffer::<u8>::new(model.slot.bytes.max(1 << 20))?];
    let (compute, copy) = (Stream::new(dev.id)?, Stream::new(dev.id)?);
    let inputs: Vec<Vec<u32>> = vec![(1..40).chain([model.eos_id]).collect()];
    let refs: Vec<&[u32]> = inputs.iter().map(|v| v.as_slice()).collect();
    run(&k, &model, &file, &mut bufs, &mut host, &compute, &copy, &refs, &mut || {})?;
    // The forward clobbered the donors ...
    let mut now1 = vec![0u8; d1.len()];
    d1.copy_to_host(&mut now1)?;
    assert_ne!(now1, p1, "the forward never touched the donor");
    // ... and the return puts back every byte.
    let st = loan.give_back(&stream, &mut host, true)?;
    eprintln!("return: {st:?}");
    let mut back0 = vec![0u8; d0.len()];
    let mut back1 = vec![0u8; d1.len()];
    d0.copy_to_host(&mut back0)?;
    d1.copy_to_host(&mut back1)?;
    assert_eq!(back0, p0);
    assert_eq!(back1, p1);
    assert_eq!(st.retried, 0);
    Ok(())
}
