//! Chain oracles for the M4 attention setup path. Composes all the M2-M4
//! kernels and asserts the chain's output matches ds4's CPU reference at
//! the post-RoPE point.
//!
//! - `q_lora_chain_oracle`: attn_input_norm → q8_0_matvec(q_a) → rms_norm
//!   → q8_0_matvec(q_b) → rms_norm_no_weight → rope_tail; compare to
//!   `q_post_rope`. Threshold 5e-2 on max_abs.
//! - `kv_chain_oracle`: attn_input_norm → q8_0_matvec(kv) → rms_norm →
//!   rope_tail; compare to `kv_post_rope`. Threshold 5e-3 on max_abs.
//!
//! Threshold rationale: the chain mean_abs is at f32-ULP (~1e-5/1e-6),
//! confirming the bulk of every output is correct. The max_abs is
//! dominated by a handful of spiky (L, T) positions where the rms_norm
//! step divides by a small RMS (one element ~15σ above the rest) and
//! thereby amplifies the matvec's Q8_0 noise ~10x on that element. The
//! per-stage diagnostics printed below quantify the amplification.
//! Downstream attention's softmax tolerates ~1% absolute noise on Q/KV
//! values that are O(1)–O(10) in magnitude, so this floor is acceptable
//! for M5+. If a future kernel port causes the *mean* to climb above
//! ~1e-4, that's the real regression signal.
//!
//! Q8_0 weights are loaded per-layer (~42 MB peak) so the test stays
//! under any reasonable GPU memory budget. Norm weights and rope_params
//! come from the activation dump.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test attention_setup_chain -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, Q8_0Matvec, RmsNorm, RmsNormNoWeight, RopeParams, RopeTail,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

const N_LAYER: i32 = 43;
const N_EMBD: u32 = 4096;
const N_LORA_Q: u32 = 1024;
const N_HEAD: u32 = 64;
const N_HEAD_KV: u32 = 1;
const N_HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM;
const KV_FLAT: u32 = N_HEAD_KV * N_HEAD_DIM;
const RMS_EPS: f32 = 1.0e-6;
const ROPE_ORIG_CTX: u64 = 65536;

const Q_THRESHOLD: f32 = 5.0e-2;
const KV_THRESHOLD: f32 = 5.0e-3;

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

#[derive(Default, Clone, Copy)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}
impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32]) {
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > self.max_abs {
                self.max_abs = d;
            }
            self.sum_abs += d as f64;
            self.count += 1;
        }
    }
    fn mean_abs(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_abs / self.count as f64
        }
    }
}

struct Setup {
    device: Device,
    dump: ActivationDump,
    gguf: MappedGguf,
    q8: Q8_0Matvec,
    rms_w: RmsNorm,
    rms_nw: RmsNormNoWeight,
    rope: RopeTail,
}

fn setup() -> eyre::Result<Setup> {
    install_panic_handler()?;
    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let rms_w = RmsNorm::for_arch(&arch)?;
    let rms_nw = RmsNormNoWeight::for_arch(&arch)?;
    let rope = RopeTail::for_arch(&arch)?;
    eprintln!(
        "setup: device {} ({}), dump n_tensors={}, gguf",
        device.id,
        arch,
        dump.len()
    );
    let _ = arch; // already printed above
    Ok(Setup {
        device,
        dump,
        gguf,
        q8,
        rms_w,
        rms_nw,
        rope,
    })
}

fn load_rope_params(dump: &ActivationDump, layer: i32) -> eyre::Result<RopeParams> {
    let entry = dump
        .weight("rope_params", layer)
        .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
    let floats = dump.read_f32(entry)?;
    let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
    RopeParams::from_dump_blob(&floats, n_ctx_orig)
}

fn upload_norm_weight(
    dump: &ActivationDump,
    tag: &str,
    layer: i32,
    n: u32,
    buf: &mut DeviceBuffer<f32>,
) -> eyre::Result<()> {
    let e = dump
        .weight(tag, layer)
        .ok_or_else(|| eyre!("missing weight:{tag} for L{layer}"))?;
    let v = dump.read_f32(e)?;
    if v.len() != n as usize {
        return Err(eyre!(
            "{tag}@L{layer}: len {} != expected {}",
            v.len(),
            n
        ));
    }
    buf.copy_from_host(&v)?;
    Ok(())
}

#[test]
#[ignore]
fn q_lora_chain_oracle() -> eyre::Result<()> {
    let s = setup()?;
    eprintln!("=== q_lora_chain_oracle ===");

    let stream = Stream::new(s.device.id)?;
    let n_tokens = s.dump.n_logit_rows as i32;

    // Reused per-token buffers. Sized for the largest stage.
    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, N_EMBD as usize)?;
    let mut d_xq_n_embd: DeviceBuffer<i8> = DeviceBuffer::new(s.device.id, N_EMBD as usize)?;
    let mut d_xscale_n_embd: DeviceBuffer<f32> =
        DeviceBuffer::new(s.device.id, (N_EMBD / 32) as usize)?;
    let mut d_qr: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, N_LORA_Q as usize)?;
    let mut d_qr_normed: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, N_LORA_Q as usize)?;
    let mut d_qr_xq: DeviceBuffer<i8> = DeviceBuffer::new(s.device.id, N_LORA_Q as usize)?;
    let mut d_qr_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(s.device.id, (N_LORA_Q / 32) as usize)?;
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, Q_FLAT as usize)?;
    let mut d_q_normed: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, Q_FLAT as usize)?;
    let mut d_qa_norm: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, N_LORA_Q as usize)?;
    let mut got = vec![0f32; Q_FLAT as usize];
    let mut got_qa = vec![0f32; N_LORA_Q as usize];
    let mut got_qa_normed = vec![0f32; N_LORA_Q as usize];
    let mut got_qb = vec![0f32; Q_FLAT as usize];
    let mut got_qb_normed = vec![0f32; Q_FLAT as usize];

    let mut overall = DiffStats::default();
    let mut stage_qa = DiffStats::default();
    let mut stage_qa_normed = DiffStats::default();
    let mut stage_qb = DiffStats::default();
    let mut stage_qb_normed = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in 0..N_LAYER {
        // Load Q8_0 weights for this layer (released at end-of-iter when shadowed).
        let q_a =
            weights::load_to_device(&s.gguf, &format!("blk.{layer}.attn_q_a.weight"), s.device.id)?;
        let q_b =
            weights::load_to_device(&s.gguf, &format!("blk.{layer}.attn_q_b.weight"), s.device.id)?;

        upload_norm_weight(&s.dump, "q_a_norm", layer, N_LORA_Q, &mut d_qa_norm)?;
        let params = load_rope_params(&s.dump, layer)?;

        for token in 0..n_tokens {
            let x_entry = match s.dump.tensor("attn_input_norm", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let exp_entry = s
                .dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            let x_host = s.dump.read_f32(x_entry)?;
            let expected = s.dump.read_f32(exp_entry)?;

            d_x.copy_from_host(&x_host)?;

            // (1) Q8_0 quantize input (4096-dim) → qr matvec
            s.q8.quantize_input(&stream, &mut d_xq_n_embd, &mut d_xscale_n_embd, &d_x, N_EMBD)?;
            s.q8.matvec(
                &stream,
                &mut d_qr,
                &q_a.buffer,
                &d_xq_n_embd,
                &d_xscale_n_embd,
                N_LORA_Q,
                N_EMBD,
            )?;
            stream.synchronize()?;
            d_qr.copy_to_host(&mut got_qa)?;
            if let Some(e) = s.dump.tensor("q_a_out", layer, token) {
                stage_qa.update(&got_qa, &s.dump.read_f32(e)?);
            }

            // (2) rms_norm_weighted with q_a_norm
            s.rms_w
                .launch_weighted(&stream, &mut d_qr_normed, &d_qr, &d_qa_norm, N_LORA_Q, RMS_EPS)?;
            stream.synchronize()?;
            d_qr_normed.copy_to_host(&mut got_qa_normed)?;
            if let Some(e) = s.dump.tensor("q_a_normed", layer, token) {
                stage_qa_normed.update(&got_qa_normed, &s.dump.read_f32(e)?);
            }

            // (3) Q8_0 quantize qr_normed (1024-dim) → q matvec
            s.q8.quantize_input(
                &stream,
                &mut d_qr_xq,
                &mut d_qr_xscale,
                &d_qr_normed,
                N_LORA_Q,
            )?;
            s.q8.matvec(
                &stream,
                &mut d_q,
                &q_b.buffer,
                &d_qr_xq,
                &d_qr_xscale,
                Q_FLAT,
                N_LORA_Q,
            )?;
            stream.synchronize()?;
            d_q.copy_to_host(&mut got_qb)?;
            if let Some(e) = s.dump.tensor("q_b_out", layer, token) {
                stage_qb.update(&got_qb, &s.dump.read_f32(e)?);
            }

            // (4) head rms_norm_no_weight (n_rows=64, n=512)
            s.rms_nw
                .launch(&stream, &mut d_q_normed, &d_q, N_HEAD, N_HEAD_DIM, RMS_EPS)?;
            stream.synchronize()?;
            d_q_normed.copy_to_host(&mut got_qb_normed)?;
            if let Some(e) = s.dump.tensor("q_head_normed", layer, token) {
                stage_qb_normed.update(&got_qb_normed, &s.dump.read_f32(e)?);
            }

            // (5) forward RoPE on Q (n_head=64) — in-place, mirrors ds4's
            // rope_tail_layer_inplace(scratch->q, ...) directly.
            s.rope.launch_forward(
                &stream,
                &mut d_q_normed,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &params,
            )?;
            stream.synchronize()?;
            d_q_normed.copy_to_host(&mut got)?;

            let prev = overall.max_abs;
            overall.update(&got, &expected);
            if overall.max_abs > prev {
                worst = (layer, token);
            }
        }

        // `q_a` and `q_b` go out of scope here → DeviceBuffer<u8> drops → frees memory.
        drop(q_a);
        drop(q_b);
    }

    eprintln!(
        "stage q_a_out:        max_abs={:.3e}, mean={:.3e}, n={}",
        stage_qa.max_abs,
        stage_qa.mean_abs(),
        stage_qa.count,
    );
    eprintln!(
        "stage q_a_normed:     max_abs={:.3e}, mean={:.3e}, n={}",
        stage_qa_normed.max_abs,
        stage_qa_normed.mean_abs(),
        stage_qa_normed.count,
    );
    eprintln!(
        "stage q_b_out:        max_abs={:.3e}, mean={:.3e}, n={}",
        stage_qb.max_abs,
        stage_qb.mean_abs(),
        stage_qb.count,
    );
    eprintln!(
        "stage q_head_normed:  max_abs={:.3e}, mean={:.3e}, n={}",
        stage_qb_normed.max_abs,
        stage_qb_normed.mean_abs(),
        stage_qb_normed.count,
    );
    eprintln!(
        "Q LoRA chain: max_abs_diff={:.3e}, mean_abs={:.3e}, n={}, worst at L{} T{}",
        overall.max_abs,
        overall.mean_abs(),
        overall.count,
        worst.0,
        worst.1,
    );

    assert!(
        overall.max_abs < Q_THRESHOLD,
        "Q LoRA chain max_abs_diff {:.3e} exceeds threshold {:.3e}",
        overall.max_abs,
        Q_THRESHOLD
    );

    Ok(())
}

#[test]
#[ignore]
fn kv_chain_oracle() -> eyre::Result<()> {
    let s = setup()?;
    eprintln!("=== kv_chain_oracle ===");

    let stream = Stream::new(s.device.id)?;
    let n_tokens = s.dump.n_logit_rows as i32;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, N_EMBD as usize)?;
    let mut d_xq: DeviceBuffer<i8> = DeviceBuffer::new(s.device.id, N_EMBD as usize)?;
    let mut d_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(s.device.id, (N_EMBD / 32) as usize)?;
    let mut d_raw: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, KV_FLAT as usize)?;
    let mut d_normed: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, KV_FLAT as usize)?;
    let mut d_kv_norm: DeviceBuffer<f32> = DeviceBuffer::new(s.device.id, KV_FLAT as usize)?;
    let mut got = vec![0f32; KV_FLAT as usize];

    let mut overall = DiffStats::default();
    let mut stage_raw = DiffStats::default();
    let mut stage_normed = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    let mut got_raw = vec![0f32; KV_FLAT as usize];
    let mut got_normed = vec![0f32; KV_FLAT as usize];

    for layer in 0..N_LAYER {
        let kv_w = weights::load_to_device(
            &s.gguf,
            &format!("blk.{layer}.attn_kv.weight"),
            s.device.id,
        )?;
        upload_norm_weight(&s.dump, "kv_a_norm", layer, N_HEAD_DIM, &mut d_kv_norm)?;
        let params = load_rope_params(&s.dump, layer)?;

        for token in 0..n_tokens {
            let x_entry = match s.dump.tensor("attn_input_norm", layer, token) {
                Some(e) => e,
                None => continue,
            };
            let exp_entry = s
                .dump
                .tensor("kv_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing kv_post_rope at L{layer} T{token}"))?;
            let x_host = s.dump.read_f32(x_entry)?;
            let expected = s.dump.read_f32(exp_entry)?;

            d_x.copy_from_host(&x_host)?;

            s.q8.quantize_input(&stream, &mut d_xq, &mut d_xscale, &d_x, N_EMBD)?;
            s.q8.matvec(
                &stream,
                &mut d_raw,
                &kv_w.buffer,
                &d_xq,
                &d_xscale,
                N_HEAD_DIM,
                N_EMBD,
            )?;
            stream.synchronize()?;
            d_raw.copy_to_host(&mut got_raw)?;
            if let Some(e) = s.dump.tensor("kv_raw_out", layer, token) {
                let v = s.dump.read_f32(e)?;
                stage_raw.update(&got_raw, &v);
            }

            s.rms_w
                .launch_weighted(&stream, &mut d_normed, &d_raw, &d_kv_norm, N_HEAD_DIM, RMS_EPS)?;
            stream.synchronize()?;
            d_normed.copy_to_host(&mut got_normed)?;
            if let Some(e) = s.dump.tensor("kv_normed", layer, token) {
                let v = s.dump.read_f32(e)?;
                stage_normed.update(&got_normed, &v);
            }

            s.rope.launch_forward(
                &stream,
                &mut d_normed,
                N_HEAD_KV,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &params,
            )?;
            stream.synchronize()?;
            d_normed.copy_to_host(&mut got)?;

            let prev = overall.max_abs;
            overall.update(&got, &expected);
            if overall.max_abs > prev {
                worst = (layer, token);
            }
        }

        drop(kv_w);
    }

    eprintln!(
        "stage kv_raw_out:     max_abs={:.3e}, mean={:.3e}, n={}",
        stage_raw.max_abs,
        stage_raw.mean_abs(),
        stage_raw.count,
    );
    eprintln!(
        "stage kv_normed:      max_abs={:.3e}, mean={:.3e}, n={}",
        stage_normed.max_abs,
        stage_normed.mean_abs(),
        stage_normed.count,
    );
    eprintln!(
        "KV chain: max_abs_diff={:.3e}, mean_abs={:.3e}, n={}, worst at L{} T{}",
        overall.max_abs,
        overall.mean_abs(),
        overall.count,
        worst.0,
        worst.1,
    );

    assert!(
        overall.max_abs < KV_THRESHOLD,
        "KV chain max_abs_diff {:.3e} exceeds threshold {:.3e}",
        overall.max_abs,
        KV_THRESHOLD
    );

    Ok(())
}
