//! End-to-end compressor oracle — composes F16 matvec + APE add + state
//! write + pool + RMS norm + RoPE + FP8 quantize, and validates against
//! ds4's `comp_post_fp8` (the value right before the F16 roundtrip happens
//! on push to the cache). Includes our own F16 roundtrip and compares the
//! final value to `comp_kv_row` as a stretch goal.
//!
//! Mirrors ds4's `compressor_decode_one_decode_scratch` (ds4.c:6589) for
//! head_dim=512 (main compressor) on every ratio>0 layer.
//!
//! For our 57-token M1 prompt:
//!   - ratio==4: pool fires every 4 tokens, ~14 firings × 21 layers = 294 outputs
//!   - ratio==128: pool never fires (n_tokens < 128), so only the per-token
//!     state-write path is exercised; no comp_post_fp8 entries to compare.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, CompressorPool, CompressorStateShuffleR4, CompressorStateWrite,
    F16Matvec, Fp8E4m3fnQuantize, RmsNorm, RopeParams, RopeTail,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const ROPE_ORIG_CTX: u64 = 65536;
const RMS_EPS: f32 = 1.0e-6;
// FP8 has coarse quantisation (step size = 0.125 around |x|=1). Tiny f32
// noise upstream (from pool / rms_norm / rope) can flip the FP8 bucket
// for a single value — that's a 1-step jump (~0.125) at the boundary.
// Mean stays at f32-ULP (~1e-6); max captures the per-bucket-flip case.
const THRESHOLD_POST_FP8: f32 = 2.0e-1;

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

fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("tensor {name} has dtype {:?}, expected F32", t.dtype));
    }
    let bytes = gguf
        .read_tensor(t).wrap_err("tensor {name} bytes missing")?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "tensor {name}: have {} bytes, expected {}",
            bytes.len(),
            expected_len * 4
        ));
    }
    let mut v = vec![0f32; expected_len];
    for (i, c) in bytes.chunks_exact(4).enumerate() {
        v[i] = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
    }
    let mut buf: DeviceBuffer<f32> = DeviceBuffer::new(device_id, expected_len)?;
    buf.copy_from_host(&v)?;
    Ok(buf)
}

fn load_rope_params(dump: &ActivationDump, layer: i32) -> eyre::Result<RopeParams> {
    let entry = dump
        .weight("rope_params", layer)
        .ok_or_else(|| eyre!("missing weight:rope_params for L{layer}"))?;
    let floats = dump.read_f32(entry)?;
    let n_ctx_orig = if floats[2] != 0.0 { ROPE_ORIG_CTX } else { 0 };
    RopeParams::from_dump_blob(&floats, n_ctx_orig)
}

fn layer_compress_ratio(il: i32) -> u32 {
    if il < 2 {
        0
    } else if (il & 1) == 0 {
        4
    } else {
        128
    }
}

#[derive(Default)]
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

#[test]
#[ignore]
fn compressor_end_to_end_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    let matvec = F16Matvec::for_arch(&arch)?;
    let state_write = CompressorStateWrite::for_arch(&arch)?;
    let state_shuffle = CompressorStateShuffleR4::for_arch(&arch)?;
    let pool = CompressorPool::for_arch(&arch)?;
    let rms = RmsNorm::for_arch(&arch)?;
    let rope = RopeTail::for_arch(&arch)?;
    let fp8 = Fp8E4m3fnQuantize::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    const NEG_INF: f32 = -3.4028235e38;

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    // Sized for the larger of ratio==4 (8 rows × 1024 comp_width = 8192)
    // and ratio==128 (128 rows × 512 comp_width = 65536).
    let max_comp_width: usize = 2 * HEAD_DIM as usize;            // 1024 (ratio==4)
    let max_state_total: usize = 128 * HEAD_DIM as usize;          // 65536 (ratio==128)
    let mut d_kv_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, max_comp_width)?;
    let mut d_sc_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, max_comp_width)?;
    let mut d_state_kv: DeviceBuffer<f32> = DeviceBuffer::new(device.id, max_state_total)?;
    let mut d_state_score: DeviceBuffer<f32> = DeviceBuffer::new(device.id, max_state_total)?;
    let mut d_pooled: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HEAD_DIM as usize)?;
    let mut d_out_comp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HEAD_DIM as usize)?;

    let mut got = vec![0f32; HEAD_DIM as usize];

    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in 2..43 {
        let ratio = layer_compress_ratio(layer);
        if ratio == 0 {
            continue;
        }
        let coff: u32 = if ratio == 4 { 2 } else { 1 };
        let comp_width = coff * HEAD_DIM;
        let state_rows = ratio * coff;

        // Load F16 weights (raw bytes) + F32 norm + rope params.
        let wkv = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_compressor_kv.weight"),
            device.id,
        )?;
        let wgate = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_compressor_gate.weight"),
            device.id,
        )?;
        let ape = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.attn_compressor_ape.weight"),
            device.id,
        )?;
        let norm = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.attn_compressor_norm.weight"),
            device.id,
            HEAD_DIM as usize,
        )?;
        let rope_params = load_rope_params(&dump, layer)?;

        // Init state on device: state_kv = 0, state_score = NEG_INF. Pad
        // to the device-buffer length (sized for the worst case across
        // both ratios — kernel only reads the in-use portion).
        let zeros = vec![0f32; max_state_total];
        let mut neg_inf_buf = vec![NEG_INF; max_state_total];
        // Sanity: ensure we don't accidentally read beyond the in-use rows.
        let in_use = (state_rows as usize) * (comp_width as usize);
        for v in &mut neg_inf_buf[in_use..] {
            *v = 0.0;
        }
        d_state_kv.copy_from_host(&zeros)?;
        d_state_score.copy_from_host(&neg_inf_buf)?;

        for token in 0..n_tokens {
            let pos_mod = (token as u32) % ratio;
            let row = if ratio == 4 { 4 + pos_mod } else { pos_mod };

            let x_entry = dump
                .tensor("attn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing attn_input_norm at L{layer} T{token}"))?;
            let x_host = dump.read_f32(x_entry)?;
            d_x.copy_from_host(&x_host)?;

            // matvec wkv → kv_cur (size comp_width)
            matvec.matvec(&stream, &mut d_kv_cur, &wkv.buffer, &d_x, comp_width, N_EMBD)?;
            matvec.matvec(&stream, &mut d_sc_cur, &wgate.buffer, &d_x, comp_width, N_EMBD)?;

            // APE add + state row write
            state_write.launch(
                &stream,
                &mut d_state_kv,
                &mut d_state_score,
                &d_kv_cur,
                &d_sc_cur,
                &ape.buffer,
                comp_width,
                row,
                pos_mod,
            )?;

            if (token + 1) as u32 % ratio != 0 {
                continue;
            }

            // Boundary: pool → rms_norm → rope → fp8
            pool.launch(&stream, &mut d_pooled, &d_state_kv, &d_state_score, HEAD_DIM, ratio)?;
            rms.launch_weighted(
                &stream,
                &mut d_out_comp,
                &d_pooled,
                &norm,
                HEAD_DIM,
                RMS_EPS,
            )?;
            let comp_pos = (token as u32) + 1 - ratio;
            rope.launch_forward(
                &stream,
                &mut d_out_comp,
                1,
                HEAD_DIM,
                N_ROT,
                comp_pos,
                &rope_params,
            )?;
            if HEAD_DIM == 512 {
                fp8.launch(&stream, &mut d_out_comp, HEAD_DIM - N_ROT)?;
            }
            stream.synchronize()?;
            d_out_comp.copy_to_host(&mut got)?;

            // Compare against comp_post_fp8.
            let exp_entry = dump
                .tensor("comp_post_fp8", layer, token)
                .ok_or_else(|| eyre!("missing comp_post_fp8 at L{layer} T{token}"))?;
            let expected = dump.read_f32(exp_entry)?;
            let prev = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }

            // State shuffle for ratio==4 (rows 4..7 → 0..3).
            if ratio == 4 {
                state_shuffle.launch(&stream, &mut d_state_kv, &mut d_state_score, comp_width)?;
            }
        }

        drop(wkv);
        drop(wgate);
        drop(ape);
        drop(norm);
    }

    eprintln!(
        "OVERALL: max_abs_diff={:.3e}, mean_abs_diff={:.3e}, n={}, worst at L{} T{}",
        stats.max_abs,
        stats.mean_abs(),
        stats.count,
        worst.0,
        worst.1,
    );

    assert!(
        stats.max_abs < THRESHOLD_POST_FP8,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD_POST_FP8
    );

    Ok(())
}
