//! Long-prompt logit dump for cross-binary bit-identity checks (2026-09-10).
//! Cycled dump residuals as inputs (identical for every binary), two-lane
//! pipelined prefill of T tokens (LOGIT_DUMP_T, default 120000), then
//! LOGIT_DUMP_N_DEC decode steps (default 24); writes the prefill logits and
//! every decode step's logits (f32 LE) to LOGIT_DUMP_OUT.
use std::io::Write;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::N_VOCAB;
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, BatchScratch,
    DgpuScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine, B_MAX,
};
use v4flash_kernels::{oracle::ActivationDump, RopeParams};

const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn pick(arch: &str) -> eyre::Result<Device> {
    Device::all()?.into_iter().find(|d| d.properties().map(|p| p.gcn_arch_name.starts_with(arch)).unwrap_or(false))
        .ok_or_else(|| eyre!("no {arch}"))
}

#[test]
#[ignore]
fn logit_dump_long() -> eyre::Result<()> {
    install_panic_handler()?;
    let t: usize = std::env::var("LOGIT_DUMP_T").ok().and_then(|s| s.parse().ok()).unwrap_or(120000);
    let n_dec: usize = std::env::var("LOGIT_DUMP_N_DEC").ok().and_then(|s| s.parse().ok()).unwrap_or(24);
    let out_path = std::env::var("LOGIT_DUMP_OUT").map_err(|_| eyre!("LOGIT_DUMP_OUT required"))?;
    let dump_dir = std::env::var("DEEPSTRIX_DUMP_DIR").map(std::path::PathBuf::from).unwrap_or_else(|_| {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..").join("reference/v4flash-cpu-activations")
    });
    let dump = ActivationDump::open(dump_dir)?;
    let gguf = MappedGguf::open(std::env::var("DEEPSTRIX_GGUF").map_err(|_| eyre!("DEEPSTRIX_GGUF required"))?)?;
    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump.weight("rope_params", layer).ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope_for_layer)?;
    let engine = HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let n_real = PROMPT_TOKENS.len();
    let mut real: Vec<Vec<f32>> = Vec::new();
    for i in 0..n_real {
        let e = dump.tensor("layer_input_residual", 0, i as i32).ok_or_else(|| eyre!("missing residual T{i}"))?;
        real.push(dump.read_f32(e)?);
    }
    let hcs: Vec<Vec<f32>> = (0..t + n_dec).map(|i| real[i % n_real].clone()).collect();
    let toks: Vec<i32> = (0..t + n_dec).map(|i| PROMPT_TOKENS[i % n_real]).collect();
    let lane_rows = B_MAX.div_ceil(2);
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut sd = BatchDgpuShared::alloc_rows(dgpu, lane_rows)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    let mut head_scratch = DgpuScratch::alloc(dgpu)?;
    let mut bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut st = HetModelState::alloc(dgpu, igpu, (t + n_dec) as u32 + 4)?;
    eprintln!("prefill {t} ...");
    let t0 = std::time::Instant::now();
    let p = engine.forward_prefill_pipelined(
        &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b, &mut sd, &mut si, &mut head_scratch,
        &mut st, &weights, &hcs[..t], &toks[..t], 0, true, None, None, None, None,
    )?;
    eprintln!("prefill done in {:.1} s, argmax {}", t0.elapsed().as_secs_f64(), p.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0);
    let mut f = std::io::BufWriter::new(std::fs::File::create(&out_path)?);
    for v in &p { f.write_all(&v.to_le_bytes())?; }
    for i in 0..n_dec {
        let pos = (t + i) as u32;
        engine.forward_token(&mut bs.shared_dgpu, &mut bs.shared_igpu, &mut st, &weights, &hcs[t + i], pos, toks[t + i])?;
        bs.shared_dgpu.residual.copy_from_buffer(&bs.shared_dgpu.residual_next)?;
        engine.forward_head(&mut bs.shared_dgpu, &weights.global)?;
        let mut l = vec![0f32; N_VOCAB as usize];
        bs.shared_dgpu.logits.copy_to_host(&mut l)?;
        for v in &l { f.write_all(&v.to_le_bytes())?; }
    }
    f.flush()?;
    eprintln!("wrote {} ({} + {} x {} logits)", out_path, p.len(), n_dec, N_VOCAB);
    engine.shutdown()?;
    Ok(())
}
