//! End-to-end SWA attention chain oracle (L=0, L=1).
//!
//! Composes the full M5 attention path: `attention_swa → rope_tail(inverse)
//! → q8_0_grouped_matvec → q8_0_matvec` and compares against `attn_out`.
//! Per-stage diagnostics mirror the M4 chain pattern.
//!
//! Weights loaded per layer (and freed at end of iter): `attn_sinks`
//! (from dump), `attn_output_a` Q8_0, `attn_output_b` Q8_0. Per-token
//! inputs from dump: `q_post_rope`, `kv_cached_row` (accumulated across T).
//!
//! Pass: max_abs_diff < 1e-1 on the final `attn_out`; mean<1e-4 as the
//! regression signal. Two Q8_0 matvecs after softmax amplify quantisation
//! noise on spiky inputs (same mechanism as M4 chain). Downstream HC-post
//! composition (M10) tolerates this — V4 Flash's HC stream averaging
//! further attenuates per-element noise.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-kernels \
//!                              --test attention_compute_chain -- --ignored --nocapture

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, AttentionSwa, Q8_0GroupedMatvec, Q8_0Matvec, RopeParams, RopeTail,
    ATTN_SWA_MAX_KV,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_HEAD: u32 = 64;
const N_HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM; // 32768
const N_GROUPS: u32 = 8;
const GROUP_DIM: u32 = 4096; // Q_FLAT / N_GROUPS
const RANK: u32 = 1024;
const OUT_LOW: u32 = N_GROUPS * RANK; // 8192
const N_EMBD: u32 = 4096;
const BLOCKS_GROUPED: u32 = (GROUP_DIM / 32) * N_GROUPS; // 1024
const BLOCKS_OUT_LOW: u32 = OUT_LOW / 32; // 256
const SWA_LAYERS: &[i32] = &[0, 1];
const ROPE_ORIG_CTX: u64 = 65536;
const FINAL_THRESHOLD: f32 = 1.0e-1;

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

fn load_rope_params(dump: &ActivationDump, layer: i32) -> eyre::Result<RopeParams> {
    let entry = dump
        .weight("rope_params", layer)
        .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
    let floats = dump.read_f32(entry)?;
    let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
    RopeParams::from_dump_blob(&floats, n_ctx_orig)
}

#[test]
#[ignore]
fn swa_attention_chain_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;
    assert!((n_tokens as u32) <= ATTN_SWA_MAX_KV);

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let attn = AttentionSwa::for_arch(&arch)?;
    let rope = RopeTail::for_arch(&arch)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let grouped = Q8_0GroupedMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Reused buffers.
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_heads: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_sinks: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD as usize)?;
    let mut d_kv: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (ATTN_SWA_MAX_KV as usize) * (N_HEAD_DIM as usize))?;
    let mut d_heads_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_heads_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, BLOCKS_GROUPED as usize)?;
    let mut d_low: DeviceBuffer<f32> = DeviceBuffer::new(device.id, OUT_LOW as usize)?;
    let mut d_low_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, OUT_LOW as usize)?;
    let mut d_low_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, BLOCKS_OUT_LOW as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut got_heads = vec![0f32; Q_FLAT as usize];
    let mut got_inv_rope = vec![0f32; Q_FLAT as usize];
    let mut got_low = vec![0f32; OUT_LOW as usize];
    let mut got_out = vec![0f32; N_EMBD as usize];

    let mut stage_heads = DiffStats::default();
    let mut stage_inv_rope = DiffStats::default();
    let mut stage_low = DiffStats::default();
    let mut final_stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for &layer in SWA_LAYERS {
        let sinks_entry = dump
            .weight("attn_sinks", layer)
            .ok_or_else(|| eyre!("missing weight:attn_sinks for L{layer}"))?;
        let sinks = dump.read_f32(sinks_entry)?;
        d_sinks.copy_from_host(&sinks)?;

        let w_out_a = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_output_a.weight"),
            device.id,
        )?;
        let w_out_b = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_output_b.weight"),
            device.id,
        )?;
        let params = load_rope_params(&dump, layer)?;

        let mut host_cache = vec![0f32; (ATTN_SWA_MAX_KV as usize) * (N_HEAD_DIM as usize)];

        for token in 0..n_tokens {
            // Accumulate KV cache.
            let kv_entry = dump
                .tensor("kv_cached_row", layer, token)
                .ok_or_else(|| eyre!("missing kv_cached_row at L{layer} T{token}"))?;
            let kv_row = dump.read_f32(kv_entry)?;
            let off = (token as usize) * (N_HEAD_DIM as usize);
            host_cache[off..off + (N_HEAD_DIM as usize)].copy_from_slice(&kv_row);
            d_kv.copy_from_host(&host_cache)?;

            let q_entry = dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            let q_host = dump.read_f32(q_entry)?;
            d_q.copy_from_host(&q_host)?;

            // Stage 1: SWA attention compute.
            attn.launch(
                &stream,
                &mut d_heads,
                &d_q,
                &d_kv,
                &d_sinks,
                N_HEAD,
                N_HEAD_DIM,
                (token as u32) + 1,
            )?;
            stream.synchronize()?;
            d_heads.copy_to_host(&mut got_heads)?;
            if let Some(e) = dump.tensor("attn_heads", layer, token) {
                stage_heads.update(&got_heads, &dump.read_f32(e)?);
            }

            // Stage 2: inverse RoPE on heads (in-place).
            rope.launch_inverse(
                &stream,
                &mut d_heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &params,
            )?;
            stream.synchronize()?;
            d_heads.copy_to_host(&mut got_inv_rope)?;
            if let Some(e) = dump.tensor("attn_heads_inv_rope", layer, token) {
                stage_inv_rope.update(&got_inv_rope, &dump.read_f32(e)?);
            }

            // Stage 3: grouped Q8_0 matvec → attn_out_low.
            q8.quantize_input(
                &stream,
                &mut d_heads_xq,
                &mut d_heads_xscale,
                &d_heads,
                Q_FLAT,
            )?;
            grouped.matvec_grouped(
                &stream,
                &mut d_low,
                &w_out_a.buffer,
                &d_heads_xq,
                &d_heads_xscale,
                GROUP_DIM,
                RANK,
                N_GROUPS,
            )?;
            stream.synchronize()?;
            d_low.copy_to_host(&mut got_low)?;
            if let Some(e) = dump.tensor("attn_out_low", layer, token) {
                stage_low.update(&got_low, &dump.read_f32(e)?);
            }

            // Stage 4: standard Q8_0 matvec → attn_out.
            q8.quantize_input(&stream, &mut d_low_xq, &mut d_low_xscale, &d_low, OUT_LOW)?;
            q8.matvec(
                &stream,
                &mut d_out,
                &w_out_b.buffer,
                &d_low_xq,
                &d_low_xscale,
                N_EMBD,
                OUT_LOW,
            )?;
            stream.synchronize()?;
            d_out.copy_to_host(&mut got_out)?;

            let exp_entry = dump
                .tensor("attn_out", layer, token)
                .ok_or_else(|| eyre!("missing attn_out at L{layer} T{token}"))?;
            let expected = dump.read_f32(exp_entry)?;
            let prev = final_stats.max_abs;
            final_stats.update(&got_out, &expected);
            if final_stats.max_abs > prev {
                worst = (layer, token);
            }
        }

        drop(w_out_a);
        drop(w_out_b);
    }

    eprintln!(
        "stage attn_heads:           max_abs={:.3e}, mean={:.3e}, n={}",
        stage_heads.max_abs,
        stage_heads.mean_abs(),
        stage_heads.count,
    );
    eprintln!(
        "stage attn_heads_inv_rope:  max_abs={:.3e}, mean={:.3e}, n={}",
        stage_inv_rope.max_abs,
        stage_inv_rope.mean_abs(),
        stage_inv_rope.count,
    );
    eprintln!(
        "stage attn_out_low:         max_abs={:.3e}, mean={:.3e}, n={}",
        stage_low.max_abs,
        stage_low.mean_abs(),
        stage_low.count,
    );
    eprintln!(
        "FINAL attn_out:             max_abs={:.3e}, mean={:.3e}, n={}, worst at L{} T{}",
        final_stats.max_abs,
        final_stats.mean_abs(),
        final_stats.count,
        worst.0,
        worst.1,
    );

    assert!(
        final_stats.max_abs < FINAL_THRESHOLD,
        "attn_out max_abs_diff {:.3e} exceeds threshold {:.3e}",
        final_stats.max_abs,
        FINAL_THRESHOLD
    );

    Ok(())
}
