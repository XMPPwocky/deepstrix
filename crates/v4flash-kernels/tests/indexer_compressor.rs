//! Indexer compressor end-to-end oracle. Same orchestration as the main
//! compressor (M7.6) but with the indexer's smaller dims:
//!   - head_dim = DS4_N_INDEXER_HEAD_DIM = 128
//!   - ratio = 4 (only ratio==4 layers have an indexer)
//!   - comp_width = 2 * head_dim = 256
//!   - state rows = 8
//!
//! FP8 quantize is skipped (head_dim != 512 → ds4 doesn't apply it for
//! the indexer compressor; index_comp_post_fp8 == index_comp_pre_fp8).
//!
//! Validates against `index_comp_post_fp8` (post-RoPE, no-op-FP8).
//! Mean<1e-5 as regression signal; max as the noise budget.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, CompressorPool, CompressorStateShuffleR4, CompressorStateWrite,
    F16Matvec, RmsNorm, RopeParams, RopeTail,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const INDEXER_HEAD_DIM: u32 = 128;
const INDEXER_COMP_WIDTH: u32 = 2 * INDEXER_HEAD_DIM; // 256, ratio==4 coff=2
const N_ROT: u32 = 64;
const ROPE_ORIG_CTX: u64 = 65536;
const RMS_EPS: f32 = 1.0e-6;
const THRESHOLD: f32 = 1.0e-3;

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
    let t = gguf
        .gguf()
        .tensor(name)
        .ok_or_else(|| eyre!("tensor {name} missing"))?;
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
fn indexer_compressor_oracle() -> eyre::Result<()> {
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
    let stream = Stream::new(device.id)?;

    const NEG_INF: f32 = -3.4028235e38;
    const STATE_ROWS: u32 = 8; // ratio*coff = 4*2

    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_kv_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, INDEXER_COMP_WIDTH as usize)?;
    let mut d_sc_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, INDEXER_COMP_WIDTH as usize)?;
    let total_state = (STATE_ROWS * INDEXER_COMP_WIDTH) as usize;
    let mut d_state_kv: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_state)?;
    let mut d_state_score: DeviceBuffer<f32> = DeviceBuffer::new(device.id, total_state)?;
    let mut d_pooled: DeviceBuffer<f32> = DeviceBuffer::new(device.id, INDEXER_HEAD_DIM as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, INDEXER_HEAD_DIM as usize)?;

    let mut got = vec![0f32; INDEXER_HEAD_DIM as usize];
    let mut stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    for layer in (2..=42).step_by(2) {
        // Indexer weights are F16 (raw bytes) + F32 norm.
        let wkv = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.indexer_compressor_kv.weight"),
            device.id,
        )?;
        let wgate = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.indexer_compressor_gate.weight"),
            device.id,
        )?;
        let ape = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.indexer_compressor_ape.weight"),
            device.id,
        )?;
        let norm = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.indexer_compressor_norm.weight"),
            device.id,
            INDEXER_HEAD_DIM as usize,
        )?;
        let rope_params = load_rope_params(&dump, layer)?;

        // Init state: kv = 0, score = NEG_INF.
        let zeros = vec![0f32; total_state];
        let neg_inf = vec![NEG_INF; total_state];
        d_state_kv.copy_from_host(&zeros)?;
        d_state_score.copy_from_host(&neg_inf)?;

        for token in 0..n_tokens {
            let pos_mod = (token as u32) % 4;
            let row = 4 + pos_mod;

            let x_entry = dump
                .tensor("attn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing attn_input_norm at L{layer} T{token}"))?;
            let x_host = dump.read_f32(x_entry)?;
            d_x.copy_from_host(&x_host)?;

            matvec.matvec(&stream, &mut d_kv_cur, &wkv.buffer, &d_x, INDEXER_COMP_WIDTH, N_EMBD)?;
            matvec.matvec(&stream, &mut d_sc_cur, &wgate.buffer, &d_x, INDEXER_COMP_WIDTH, N_EMBD)?;
            state_write.launch(
                &stream,
                &mut d_state_kv,
                &mut d_state_score,
                &d_kv_cur,
                &d_sc_cur,
                &ape.buffer,
                INDEXER_COMP_WIDTH,
                row,
                pos_mod,
            )?;

            if (token + 1) % 4 != 0 {
                continue;
            }

            pool.launch(&stream, &mut d_pooled, &d_state_kv, &d_state_score, INDEXER_HEAD_DIM, 4)?;
            rms.launch_weighted(
                &stream,
                &mut d_out,
                &d_pooled,
                &norm,
                INDEXER_HEAD_DIM,
                RMS_EPS,
            )?;
            let comp_pos = (token as u32) + 1 - 4;
            rope.launch_forward(
                &stream,
                &mut d_out,
                1,
                INDEXER_HEAD_DIM,
                N_ROT,
                comp_pos,
                &rope_params,
            )?;
            // No FP8 for indexer (head_dim != 512).
            stream.synchronize()?;
            d_out.copy_to_host(&mut got)?;

            let exp_entry = dump
                .tensor("index_comp_post_fp8", layer, token)
                .ok_or_else(|| eyre!("missing index_comp_post_fp8 at L{layer} T{token}"))?;
            let expected = dump.read_f32(exp_entry)?;
            let prev = stats.max_abs;
            stats.update(&got, &expected);
            if stats.max_abs > prev {
                worst = (layer, token);
            }

            state_shuffle.launch(&stream, &mut d_state_kv, &mut d_state_score, INDEXER_COMP_WIDTH)?;
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
        stats.max_abs < THRESHOLD,
        "max_abs_diff {:.3e} exceeds threshold {:.3e}",
        stats.max_abs,
        THRESHOLD
    );

    Ok(())
}
