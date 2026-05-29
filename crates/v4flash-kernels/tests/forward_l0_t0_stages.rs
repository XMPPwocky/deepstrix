//! Layer-0 token-0 per-stage diff vs ds4 dump. Runs forward_layer_pair_mode
//! for (L=0, T=0) and compares each named scratch buffer to the corresponding
//! ds4 dump tag, so we can see which stage contributes the bulk of the ~1e-2
//! layer-0 divergence.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{
    HC_DIM, N_EMBD, N_HEAD_DIM, OUT_LOW, Q_FLAT,
};
use v4flash_kernels::het::{
    BatchScratch, ExecMode, HetModelState, HetModelWeights, HeterogeneousEngine,
};
use v4flash_kernels::{ActivationDump, RopeParams};

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201"))
}
fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize, f32, f32) {
    assert_eq!(a.len(), b.len());
    let mut maxd = 0.0f32;
    let mut idx = 0usize;
    for i in 0..a.len() {
        let d = (a[i] - b[i]).abs();
        if d > maxd {
            maxd = d;
            idx = i;
        }
    }
    (maxd, idx, a[idx], b[idx])
}

fn mean_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += (a[i] - b[i]).abs();
    }
    sum / n
}

#[test]
#[ignore]
fn forward_l0_t0_stages() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let main_gguf = MappedGguf::open(MAIN_MODEL_PATH)?;
    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing rope_params L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading main weights...");
    let main_weights = HetModelWeights::load_all(&main_gguf, dgpu, igpu, &rope_for_layer)?;
    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;

    let mut bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, 8)?;

    // Seed L0 residual with dump's layer_input_residual.
    let inp = dump.read_f32(
        dump.tensor("layer_input_residual", 0, 0)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T0"))?,
    )?;
    dgpu.set_current()?;
    bs.shared_dgpu.residual.copy_from_host(&inp)?;

    // Run layer 0 for token 0.
    engine.forward_layer_pair_mode(
        &mut bs.shared_dgpu,
        &mut bs.shared_igpu,
        &mut state.layers[0],
        &main_weights.dgpu_layers[0],
        &main_weights.igpu_layers[0],
        0,
        PROMPT_TOKENS[0],
    )?;

    // Helper: compare a scratch buffer against a dump tag.
    let mut compare = |tag: &str,
                       got_host: &[f32]|
     -> eyre::Result<()> {
        let entry = dump
            .tensor(tag, 0, 0)
            .ok_or_else(|| eyre!("missing dump tag {tag} L0 T0"))?;
        let expected = dump.read_f32(entry)?;
        if expected.len() != got_host.len() {
            println!(
                "  {tag:<28}  SHAPE MISMATCH got={} exp={}",
                got_host.len(),
                expected.len()
            );
            return Ok(());
        }
        let (maxd, idx, gv, ev) = max_abs_diff(got_host, &expected);
        let meand = mean_abs_diff(got_host, &expected);
        println!(
            "  {tag:<28}  max={:.4e}  mean={:.4e}  @i={:>6}  got={:>+10.4}  exp={:>+10.4}",
            maxd, meand, idx, gv, ev
        );
        Ok(())
    };

    println!("\n=== L0 T0 per-stage diff vs ds4 ===");

    // Block 1: HC pre-attn → attn_input_norm + attn_cur
    let mut h = vec![0f32; N_EMBD as usize];
    bs.shared_dgpu.attn_cur.copy_to_host(&mut h)?;
    compare("attn_cur", &h)?;
    bs.shared_dgpu.attn_input_norm.copy_to_host(&mut h)?;
    compare("attn_input_norm", &h)?;

    // Block 2: Q chain. q_normed = q_post_rope (after rope on q_b output rms-normed).
    // qr_normed = q_a_normed. q = q_b_out (pre rms).
    let mut hq_low = vec![0f32; v4flash_kernels::config::N_LORA_Q as usize];
    bs.shared_dgpu.qr.copy_to_host(&mut hq_low)?;
    compare("q_a_out", &hq_low)?;
    bs.shared_dgpu.qr_normed.copy_to_host(&mut hq_low)?;
    compare("q_a_normed", &hq_low)?;
    let mut hq = vec![0f32; Q_FLAT as usize];
    bs.shared_dgpu.q.copy_to_host(&mut hq)?;
    compare("q_b_out", &hq)?;
    bs.shared_dgpu.q_normed.copy_to_host(&mut hq)?;
    compare("q_post_rope", &hq)?;

    // Block 3: KV chain. kv_raw = kv_raw_out (pre rms). kv_normed = kv_post_rope.
    let mut hkv = vec![0f32; N_HEAD_DIM as usize];
    bs.shared_dgpu.kv_raw.copy_to_host(&mut hkv)?;
    compare("kv_raw_out", &hkv)?;
    bs.shared_dgpu.kv_normed.copy_to_host(&mut hkv)?;
    compare("kv_post_rope", &hkv)?;

    // Block 4: attention output (post inv-rope, post output-proj).
    // dgpu_scratch.heads holds attn_heads_inv_rope; .low holds attn_out_low; .attn_out is the projection result.
    let mut hheads = vec![0f32; Q_FLAT as usize];
    bs.shared_dgpu.heads.copy_to_host(&mut hheads)?;
    compare("attn_heads_inv_rope", &hheads)?;
    let mut hlow = vec![0f32; OUT_LOW as usize];
    bs.shared_dgpu.low.copy_to_host(&mut hlow)?;
    compare("attn_out_low", &hlow)?;
    let mut hattn_out = vec![0f32; N_EMBD as usize];
    bs.shared_dgpu.attn_out.copy_to_host(&mut hattn_out)?;
    compare("attn_out", &hattn_out)?;

    // Block 5: HC post-attn / pre-ffn.
    let mut hffn_cur = vec![0f32; N_EMBD as usize];
    bs.shared_dgpu.ffn_cur.copy_to_host(&mut hffn_cur)?;
    compare("ffn_cur", &hffn_cur)?;
    bs.shared_dgpu.ffn_input_norm.copy_to_host(&mut hffn_cur)?;
    compare("ffn_input_norm", &hffn_cur)?;

    // Block 6: final residual (post FFN/MoE + HC post-ffn).
    let mut hres = vec![0f32; HC_DIM as usize];
    bs.shared_dgpu.residual_next.copy_to_host(&mut hres)?;
    compare("layer_output_residual", &hres)?;

    Ok(())
}
