//! Per-layer compositional bisect. Runs forward_layer one at a time at T=0
//! and reports diff vs dump tags at every major checkpoint.
//!
//! Outputs (per layer, per stage):
//!   attn_cur          vs dump.attn_cur[L,0]
//!   attn_input_norm   vs dump.attn_input_norm[L,0]   (if dumped)
//!   q_normed (post-rope) vs dump.q_post_rope[L,0]
//!   kv_normed (post-rope+f16rt) vs dump.kv_cached_row[L,0]
//!   attn_out          vs dump.attn_out[L,0]
//!   ffn_input_norm    vs dump.ffn_input_norm[L,0]
//!   ffn_moe (post-add ffn_shared)  vs (ffn_moe + ffn_shared)
//!   ffn_shared        vs dump.ffn_shared[L,0]
//!   residual_next     vs dump.layer_input_residual[L+1,0]
//!                        or dump.layer_output_residual[L,0]
//!
//! Goal: find the first L where divergence enters at >1e-2 on any stage.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test forward_bisect -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::forward::{
    Engine, ModelState, ModelWeights, Scratch, HC_DIM, N_EMBD, N_HEAD_DIM, N_LAYER, OUT_LOW,
    Q_FLAT,
};
use v4flash_kernels::{ActivationDump, RopeParams};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";
const PROMPT_TOKENS: [i32; 7] = [53091, 4374, 1465, 13582, 22, 32958, 344];
const ROPE_ORIG_CTX: u64 = 65536;

fn dump_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("reference/v4flash-cpu-activations")
}

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

fn diff(got: &[f32], expected: &[f32]) -> (f32, f64) {
    let mut max_abs: f32 = 0.0;
    let mut sum_abs: f64 = 0.0;
    for (g, e) in got.iter().zip(expected.iter()) {
        let d = (g - e).abs();
        if d > max_abs {
            max_abs = d;
        }
        sum_abs += d as f64;
    }
    let mean = sum_abs / got.len() as f64;
    (max_abs, mean)
}

fn check(
    label: &str,
    layer: i32,
    got_dev: &v4flash_hip::DeviceBuffer<f32>,
    dump: &ActivationDump,
    tag: &str,
    dump_layer: i32,
    dump_token: i32,
    host_buf: &mut Vec<f32>,
) -> eyre::Result<(f32, f64)> {
    got_dev.copy_to_host(host_buf)?;
    let entry = dump
        .tensor(tag, dump_layer, dump_token)
        .ok_or_else(|| eyre!("missing {tag} L{dump_layer} T{dump_token}"))?;
    let expected = dump.read_f32(entry)?;
    if expected.len() != host_buf.len() {
        return Err(eyre!(
            "{label} L{layer}: size {} != dump {} ({tag})",
            host_buf.len(),
            expected.len()
        ));
    }
    let (mx, mn) = diff(host_buf, &expected);
    eprintln!("  L{layer:>2} {label:<22} max={mx:.3e}  mean={mn:.3e}");
    Ok((mx, mn))
}

#[test]
#[ignore]
fn forward_bisect_T0() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let rope_for_layer = |layer: i32| -> eyre::Result<RopeParams> {
        let entry = dump
            .weight("rope_params", layer)
            .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
        let floats = dump.read_f32(entry)?;
        let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
        RopeParams::from_dump_blob(&floats, n_ctx_orig)
    };

    eprintln!("loading weights…");
    let weights = ModelWeights::load_all(&gguf, device.id, &rope_for_layer)?;
    eprintln!("weights loaded.");

    let engine = Engine::for_arch(device, &arch)?;
    let mut scratch = Scratch::alloc(device.id)?;
    let mut state = ModelState::alloc(device.id, 64)?;

    // Install debug dump pointer for in-engine router overrides via env.
    unsafe { v4flash_kernels::forward::set_debug_dump(Some(&dump)); }

    // T=0 only. Seed residual from dump's layer_input_residual[L=0, T=0].
    let pos: u32 = 0;
    let token_id = PROMPT_TOKENS[0];
    let inp = dump.read_f32(
        dump.tensor("layer_input_residual", 0, 0)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T0"))?,
    )?;
    assert_eq!(inp.len(), HC_DIM as usize);
    scratch.residual.copy_from_host(&inp)?;

    let mut host_embd = vec![0f32; N_EMBD as usize];
    let mut host_hc = vec![0f32; HC_DIM as usize];
    let mut host_hd = vec![0f32; N_HEAD_DIM as usize];
    let mut host_qflat = vec![0f32; Q_FLAT as usize];
    let mut host_low = vec![0f32; OUT_LOW as usize];

    let mut first_breakpoint: Option<i32> = None;

    for layer in 0..N_LAYER {
        let lw = &weights.layers[layer as usize];
        // OPTIONAL: clobber residual with dump's layer_input_residual at this layer
        // (DS4_BISECT_RESET_AT=N). Lets us isolate per-layer behavior with clean input.
        if let Ok(reset_at) = std::env::var("DS4_BISECT_RESET_AT") {
            if reset_at.parse::<i32>().ok() == Some(layer) {
                let clean = dump.read_f32(
                    dump.tensor("layer_input_residual", layer, 0)
                        .ok_or_else(|| eyre!("missing layer_input_residual L{layer}"))?,
                )?;
                scratch.residual.copy_from_host(&clean)?;
                eprintln!("  >>> reset residual from dump at L{layer}");
            }
        }
        // DS4_BISECT_FFNIN_AT=N: after forward_layer, BEFORE bisect-check,
        // re-run the layer's FFN stage with dump's exact ffn_input_norm to
        // isolate MoE noise from input drift. Implemented by running
        // forward_layer normally then noticing this is too invasive — we'd
        // need a separate forward_ffn_only API. Skip for now.
        engine.forward_layer(
            &mut scratch,
            &mut state,
            lw,
            &gguf,
            layer as usize,
            pos,
            token_id,
        )?;
        engine.stream.synchronize()?;

        // Compare every per-stage checkpoint we have a dump tag for.
        let (m1, _) = check(
            "attn_cur",
            layer,
            &scratch.attn_cur,
            &dump,
            "attn_cur",
            layer,
            0,
            &mut host_embd,
        )?;
        let _ = check(
            "attn_input_norm",
            layer,
            &scratch.attn_input_norm,
            &dump,
            "attn_input_norm",
            layer,
            0,
            &mut host_embd,
        )?;
        let _ = check(
            "q_post_rope",
            layer,
            &scratch.q_normed,
            &dump,
            "q_post_rope",
            layer,
            0,
            &mut host_qflat,
        )?;
        // Extra attention-subpipeline checks. These won't be available for
        // ratio>0 layers in the dump (those have mixed attention) but
        // for L=0,1 they pinpoint the failing op.
        let _ = check(
            "kv_normed vs kv_post_rope",
            layer,
            &scratch.kv_normed,
            &dump,
            "kv_post_rope",
            layer,
            0,
            &mut host_hd,
        )
        .ok();
        let _ = check(
            "kv_normed vs kv_cached_row",
            layer,
            &scratch.kv_normed,
            &dump,
            "kv_cached_row",
            layer,
            0,
            &mut host_hd,
        )
        .ok();
        // Direct dump-vs-dump diff to determine whether kv_cached_row == kv_post_rope.
        if layer == 0 {
            let pr = dump.read_f32(dump.tensor("kv_post_rope", 0, 0).unwrap())?;
            let cr = dump.read_f32(dump.tensor("kv_cached_row", 0, 0).unwrap())?;
            let mut mx: f32 = 0.0;
            let mut sm: f64 = 0.0;
            for (a, b) in pr.iter().zip(cr.iter()) {
                let d = (a - b).abs();
                if d > mx { mx = d; }
                sm += d as f64;
            }
            eprintln!(
                "  ** dump kv_post_rope vs kv_cached_row: max={mx:.3e} mean={:.3e}",
                sm / pr.len() as f64
            );
        }
        let _ = check(
            "attn_heads_inv_rope",
            layer,
            &scratch.heads,
            &dump,
            "attn_heads_inv_rope",
            layer,
            0,
            &mut host_qflat,
        )
        .ok();
        let _ = check(
            "attn_out_low",
            layer,
            &scratch.low,
            &dump,
            "attn_out_low",
            layer,
            0,
            &mut host_low,
        )
        .ok();
        let _ = check(
            "attn_out",
            layer,
            &scratch.attn_out,
            &dump,
            "attn_out",
            layer,
            0,
            &mut host_embd,
        )?;
        let _ = check(
            "ffn_input_norm",
            layer,
            &scratch.ffn_input_norm,
            &dump,
            "ffn_input_norm",
            layer,
            0,
            &mut host_embd,
        )?;
        let _ = check(
            "ffn_shared",
            layer,
            &scratch.ffn_shared,
            &dump,
            "ffn_shared",
            layer,
            0,
            &mut host_embd,
        )?;

        // L33 magnitude probe.
        if layer == 33 {
            scratch.ffn_input_norm.copy_to_host(&mut host_embd)?;
            let our_max = host_embd.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let dump_fin = dump.read_f32(dump.tensor("ffn_input_norm", layer, 0).unwrap())?;
            let dump_max = dump_fin.iter().map(|x| x.abs()).fold(0f32, f32::max);
            let dump_moe = dump.read_f32(dump.tensor("ffn_moe", layer, 0).unwrap())?;
            let dmoe_max = dump_moe.iter().map(|x| x.abs()).fold(0f32, f32::max);
            eprintln!(
                "  L33 mag: ffn_input_norm our={:.3e} dump={:.3e}  ffn_moe dump={:.3e}",
                our_max, dump_max, dmoe_max
            );
        }
        // ffn_moe (post-vec_add): should match dump.ffn_moe + dump.ffn_shared
        scratch.ffn_moe.copy_to_host(&mut host_embd)?;
        let mut our_shared = vec![0f32; N_EMBD as usize];
        scratch.ffn_shared.copy_to_host(&mut our_shared)?;
        let dm = dump.read_f32(dump.tensor("ffn_moe", layer, 0).unwrap())?;
        let ds = dump.read_f32(dump.tensor("ffn_shared", layer, 0).unwrap())?;
        let mut mx: f32 = 0.0;
        let mut sm: f64 = 0.0;
        let mut mx_routed: f32 = 0.0;
        let mut sm_routed: f64 = 0.0;
        for (((g, dm_i), ds_i), our_sh) in host_embd
            .iter()
            .zip(dm.iter())
            .zip(ds.iter())
            .zip(our_shared.iter())
        {
            let e = dm_i + ds_i;
            let d = (g - e).abs();
            if d > mx { mx = d; }
            sm += d as f64;
            // isolate routed: our_ffn_moe_routed = scratch.ffn_moe - our_ffn_shared
            let our_routed = g - our_sh;
            let dr = (our_routed - dm_i).abs();
            if dr > mx_routed { mx_routed = dr; }
            sm_routed += dr as f64;
        }
        eprintln!(
            "  L{layer:>2} ffn_moe(post-add)      max={mx:.3e}  mean={:.3e}",
            sm / host_embd.len() as f64
        );
        eprintln!(
            "  L{layer:>2} ffn_moe(routed only)   max={mx_routed:.3e}  mean={:.3e}",
            sm_routed / host_embd.len() as f64
        );

        // residual_next ⇔ layer_input_residual[L+1, T=0] (or output_residual[L])
        let next_tag = if layer + 1 < N_LAYER {
            ("layer_input_residual", layer + 1)
        } else {
            ("layer_output_residual", layer)
        };
        let (m_next, _) = check(
            "residual_next",
            layer,
            &scratch.residual_next,
            &dump,
            next_tag.0,
            next_tag.1,
            0,
            &mut host_hc,
        )?;

        if first_breakpoint.is_none() && (m1 > 1.0e-2 || m_next > 1.0e-2) {
            first_breakpoint = Some(layer);
            eprintln!("  ^^ first break at L{layer} (any stage > 1e-2) ^^");
        }

        // Swap to advance.
        std::mem::swap(&mut scratch.residual, &mut scratch.residual_next);
    }

    if let Some(l) = first_breakpoint {
        eprintln!("\nFIRST BREAK: L{l}");
    } else {
        eprintln!("\nNo break: all layers within 1e-2");
    }
    Ok(())
}
