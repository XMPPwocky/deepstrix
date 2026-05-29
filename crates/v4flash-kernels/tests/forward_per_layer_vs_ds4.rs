//! Per-layer per-token diff of our sequential forward chain vs the ds4 CPU
//! reference dump. Localizes where (which layer, which token) our impl
//! starts diverging from ds4. Uses the existing 7-token reference dump.
//!
//! Setup:
//!   For each token T in 0..B (in order so KV cache builds up correctly):
//!     Seed `shared_dgpu.residual` with `dump.layer_input_residual[L=0, T]`.
//!     For each layer L in 0..N_LAYER:
//!       Call `forward_layer_pair_mode(L, T)`. This writes layer-L KV for T
//!         and produces residual_next (= input to layer L+1).
//!       Compare `residual_next` to `dump.layer_output_residual[L, T]`.
//!       Then copy residual_next → residual for the next layer iteration.
//!
//! Output: per-(layer, token) max-abs-diff, plus a per-layer summary.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!     --test forward_per_layer_vs_ds4 -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_LAYER};
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
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
    (maxd, idx)
}

#[test]
#[ignore]
fn forward_per_layer_vs_ds4() -> eyre::Result<()> {
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

    let b: usize = std::env::var("BENCH_B")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7)
        .min(PROMPT_TOKENS.len());
    eprintln!("per-layer vs ds4: B={b}");

    let tokens: Vec<i32> = PROMPT_TOKENS[..b].to_vec();

    let mut bs = BatchScratch::alloc(dgpu, igpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, b as u32 + 4)?;

    // diffs[layer][token]
    let mut diffs: Vec<Vec<f32>> = vec![vec![0.0; b]; N_LAYER as usize];

    for token in 0..b {
        // Seed residual with dump's layer_input_residual at L=0 for this token.
        let inp_entry = dump
            .tensor("layer_input_residual", 0, token as i32)
            .ok_or_else(|| eyre!("missing layer_input_residual L0 T{token}"))?;
        let inp = dump.read_f32(inp_entry)?;
        assert_eq!(inp.len(), HC_DIM as usize);
        dgpu.set_current()?;
        bs.shared_dgpu.residual.copy_from_host(&inp)?;

        for layer in 0..N_LAYER as usize {
            engine.forward_layer_pair_mode(
                &mut bs.shared_dgpu,
                &mut bs.shared_igpu,
                &mut state.layers[layer],
                &main_weights.dgpu_layers[layer],
                &main_weights.igpu_layers[layer],
                token as u32,
                tokens[token],
            )?;

            // Compare residual_next to ds4's layer_output_residual.
            let mut got = vec![0f32; HC_DIM as usize];
            bs.shared_dgpu.residual_next.copy_to_host(&mut got)?;
            let exp_entry = dump
                .tensor("layer_output_residual", layer as i32, token as i32)
                .ok_or_else(|| {
                    eyre!("missing layer_output_residual L{layer} T{token}")
                })?;
            let expected = dump.read_f32(exp_entry)?;
            let (maxd, _idx) = max_abs_diff(&got, &expected);
            diffs[layer][token] = maxd;

            // Chain: this layer's output becomes next layer's input.
            bs.shared_dgpu
                .residual
                .copy_from_buffer(&bs.shared_dgpu.residual_next)?;
        }
    }

    // Print per-layer summary.
    println!("\nlayer | max diff across tokens | (worst token)");
    for layer in 0..N_LAYER as usize {
        let (worst_t, worst_d) = diffs[layer]
            .iter()
            .enumerate()
            .map(|(t, &d)| (t, d))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        println!("L{:>2}   | {:.4e}              | T{}", layer, worst_d, worst_t);
    }

    // Full grid for closer inspection of small B.
    if b <= 16 {
        println!("\nfull diff grid (layers × tokens):");
        print!("       ");
        for t in 0..b {
            print!("    T{:<6}", t);
        }
        println!();
        for layer in 0..N_LAYER as usize {
            print!("L{:>2}   ", layer);
            for t in 0..b {
                print!("  {:>8.2e}", diffs[layer][t]);
            }
            println!();
        }
    }

    // Highlight the first layer where ANY token exceeds 1e-2 (heuristic).
    for layer in 0..N_LAYER as usize {
        let max_for_layer: f32 = diffs[layer].iter().cloned().fold(0.0, f32::max);
        if max_for_layer > 1e-2 {
            println!(
                "\nfirst layer with max diff > 1e-2: L{} ({:.4e})",
                layer, max_for_layer
            );
            break;
        }
    }

    Ok(())
}
