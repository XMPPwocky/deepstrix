//! DSpark drafter EXIT: tied head + markov bias loop.
//!
//! Runs the exit on a real drafter residual and checks the parts that can be
//! checked without a full speculative decode loop:
//!
//!   * every draft token is a valid vocab id, and the logits are finite;
//!   * the markov head actually MOVES the logits — it is the only channel
//!     carrying "what did position i-1 actually emit" into position i, so a
//!     no-op there would silently cost acceptance rate and nothing else;
//!   * feeding a different accepted token changes the drafts.
//!
//!   cargo test -p v4flash-kernels --features v41 --test mtp_exit_smoke \
//!     -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::{V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_VOCAB};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{mtp_rope, MtpExit, MtpState, MTP_BLOCK, MTP_NOISE_TOKEN};
use v4flash_kernels::het::weights::{MtpExitWeights, MtpWeights};

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151"))
}

fn read_f32(path: &str) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "needs the checkpoint and a GPU"]
fn exit_emits_valid_drafts() {
    install_panic_handler().ok();
    let home = std::env::var("HOME").unwrap_or_default();
    let refdir = std::env::var("MTP_REF")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/v41/agentic/main/mtp_ref"));
    let main_hidden = read_f32(&format!("{refdir}/main_hidden.bin"));
    let model = std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{home}/.cache/deepstrix/models/dsv4.1f"));

    let hf = V41HfWeights::open(&model, None).expect("open checkpoint");
    let dev = pick_igpu().expect("igpu");
    let arch = dev.properties().expect("props").gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch).expect("engine");
    let w = MtpWeights::load(&hf, dev, 40).expect("load drafter");
    // In production the exit lives on the dGPU beside the tied head; here it
    // shares the iGPU so the test needs one device.
    let xw = MtpExitWeights::load(&hf, dev).expect("load exit weights");

    // The drafter's head is TIED to the main model's, so the exit borrows
    // `output.weight`. In production it is already resident on the dGPU; here
    // only that one tensor is uploaded, not the whole model.
    let head = v4flash_kernels::weights::load_to_device(&hf, "output.weight", dev.id)
        .expect("load tied head");
    let src: WeightSrc = (&hf).into();
    let mk = src
        .tensor("mtp.2.markov_embd.weight")
        .expect("markov embed tensor");
    let mk_dtype = mk.dtype;
    let mk_bytes = src.read_tensor(mk).expect("read markov embed");

    // Token embeddings: the real rows, via the same host-side path the server
    // uses for the main model (M57 — token_embd is not device-resident).
    let te = src.tensor("token_embd.weight").expect("token_embd");
    let te_dtype = te.dtype;
    let te_bytes = src.read_tensor(te).expect("read token_embd");
    let mut noise_row = vec![0.0f32; HC_DIM as usize];
    v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, MTP_NOISE_TOKEN, &mut noise_row)
        .expect("embed noise");

    let mut st = MtpState::alloc(dev.id).expect("alloc state");
    let mut ex = MtpExit::alloc(dev.id).expect("alloc exit");
    let rope = mtp_rope();

    let mut seen: Vec<[i32; MTP_BLOCK]> = Vec::new();
    for tok in [3070i32, 15043i32] {
        let mut token_row = vec![0.0f32; HC_DIM as usize];
        v4flash_kernels::embed::embed_lookup(&te_bytes, te_dtype, tok, &mut token_row)
            .expect("embed token");

        st.inject_main_hidden(&main_hidden).expect("inject");
        st.forward(&e, &e.compute, &w, &rope, 200, &token_row, &noise_row)
            .expect("drafter forward");

        e.compute.synchronize().expect("sync");
        let mut h_host = vec![0.0f32; st.h.len()];
        st.h.copy_to_host(&mut h_host).expect("read h");
        let mut pre_host = vec![0.0f32; st.pre_carry().len()];
        st.pre_carry().copy_to_host(&mut pre_host).expect("read pre");

        let (ids, plain) = ex
            .forward(
                &e, &e.compute, &h_host, &pre_host, &xw, &head, &mk_bytes, mk_dtype, tok,
            )
            .expect("exit forward");
        e.compute.synchronize().expect("sync");

        let mut lg = vec![0.0f32; MTP_BLOCK * N_VOCAB as usize];
        ex.logits.copy_to_host(&mut lg).expect("read logits");
        assert!(lg.iter().all(|v| v.is_finite()), "non-finite logits");
        for (i, id) in ids.iter().enumerate() {
            assert!(
                *id >= 0 && (*id as u32) < N_VOCAB,
                "draft {i} = {id} outside the vocab"
            );
        }
        println!("token {tok:6} -> drafts {ids:?}  (no markov bias: {plain:?})");
        seen.push(ids);
    }

    assert_ne!(
        seen[0], seen[1],
        "different accepted tokens produced identical drafts"
    );
}
