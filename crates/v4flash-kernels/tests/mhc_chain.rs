//! mHC pre + post oracle. Validates two sub-chains:
//!
//! (1) hc_pre_attn: layer_input_residual → attn_cur
//!     - rms_norm_no_weight(hc_dim=16384)
//!     - F16Matvec(hc_attn_fn[16384, 24], flat) → mix[24]
//!     - hc_sinkhorn(mix, hc_attn_scale[3], hc_attn_base[24]) → split[24]
//!     - hc_weighted_sum(layer_input_residual, split[0..4]) → attn_cur[4096]
//!
//! (2) hc_post_attn + hc_pre_ffn: with dumped attn_out as block_out,
//!     layer_input_residual as residual_hc, and re-computed (post, comb)
//!     from (1)'s Sinkhorn:
//!     - hc_post(attn_out, layer_input_residual, post, comb) → after_attn_hc
//!     - rms_norm_no_weight(after_attn_hc) → flat2
//!     - F16Matvec(hc_ffn_fn[16384, 24], flat2) → mix2
//!     - hc_sinkhorn(mix2, hc_ffn_scale, hc_ffn_base) → split2
//!     - hc_weighted_sum(after_attn_hc, split2[0..4]) → ffn_cur[4096]
//!
//! Both `attn_cur` and `ffn_cur` are dumped tags. Coverage: all layers
//! L=0..42 × all 51 tokens.

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use color_eyre::eyre::WrapErr;
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, oracle::ActivationDump, F16Matvec, HcPost, HcSinkhorn, HcWeightedSum, RmsNormNoWeight,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_HC: u32 = 4;
const HC_DIM: u32 = N_EMBD * N_HC; // 16384
const HC_MIX_DIM: u32 = 2 * N_HC + N_HC * N_HC; // 24
const N_LAYER: i32 = 43;
const N_SINKHORN_ITERS: u32 = 20;
const SINKHORN_EPS: f32 = 1.0e-6;
const RMS_EPS: f32 = 1.0e-6;

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
        return Err(eyre!("tensor {name} dtype {:?} != F32", t.dtype));
    }
    let bytes = gguf.read_tensor(t).wrap_err("{name} bytes missing")?;
    if bytes.len() != expected_len * 4 {
        return Err(eyre!(
            "{name}: have {} bytes, expected {}",
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

#[derive(Default)]
struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
    worst: (i32, i32),
}
impl DiffStats {
    fn update(&mut self, a: &[f32], b: &[f32], l: i32, t: i32) {
        let prev = self.max_abs;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = (x - y).abs();
            if d > self.max_abs {
                self.max_abs = d;
            }
            self.sum_abs += d as f64;
            self.count += 1;
        }
        if self.max_abs > prev {
            self.worst = (l, t);
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
fn mhc_chain_oracle() -> eyre::Result<()> {
    install_panic_handler()?;

    let dump = ActivationDump::open(dump_dir())?;
    let gguf = MappedGguf::open(MODEL_PATH)?;
    let n_tokens = dump.n_logit_rows as i32;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch}); n_tokens={n_tokens}", device.id);

    let rms = RmsNormNoWeight::for_arch(&arch)?;
    let f16 = F16Matvec::for_arch(&arch)?;
    let sinkhorn = HcSinkhorn::for_arch(&arch)?;
    let weighted = HcWeightedSum::for_arch(&arch)?;
    let post_k = HcPost::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    // Per-token / per-layer scratch.
    let mut d_inp: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_flat: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_mix: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_MIX_DIM as usize)?;
    let mut d_split: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_MIX_DIM as usize)?;
    let mut d_attn_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_attn_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_after_attn: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_flat2: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_DIM as usize)?;
    let mut d_mix2: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_MIX_DIM as usize)?;
    let mut d_split2: DeviceBuffer<f32> = DeviceBuffer::new(device.id, HC_MIX_DIM as usize)?;
    let mut d_ffn_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;

    let mut got_attn_cur = vec![0f32; N_EMBD as usize];
    let mut got_ffn_cur = vec![0f32; N_EMBD as usize];

    let mut s_attn = DiffStats::default();
    let mut s_ffn = DiffStats::default();

    for layer in 0..N_LAYER {
        let w_attn_fn = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.hc_attn_fn.weight"),
            device.id,
        )?;
        let w_attn_scale = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.hc_attn_scale.weight"),
            device.id,
            3,
        )?;
        let w_attn_base = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.hc_attn_base.weight"),
            device.id,
            HC_MIX_DIM as usize,
        )?;
        let w_ffn_fn = weights::load_to_device(
            &gguf,
            &format!("blk.{layer}.hc_ffn_fn.weight"),
            device.id,
        )?;
        let w_ffn_scale = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.hc_ffn_scale.weight"),
            device.id,
            3,
        )?;
        let w_ffn_base = load_f32_weight(
            &gguf,
            &format!("blk.{layer}.hc_ffn_base.weight"),
            device.id,
            HC_MIX_DIM as usize,
        )?;

        for token in 0..n_tokens {
            let inp_entry = dump
                .tensor("layer_input_residual", layer, token)
                .ok_or_else(|| eyre!("missing layer_input_residual L{layer} T{token}"))?;
            d_inp.copy_from_host(&dump.read_f32(inp_entry)?)?;

            // (1) hc_pre_attn
            rms.launch(&stream, &mut d_flat, &d_inp, 1, HC_DIM, RMS_EPS)?;
            f16.matvec(&stream, &mut d_mix, &w_attn_fn.buffer, &d_flat, HC_MIX_DIM, HC_DIM)?;
            sinkhorn.launch(
                &stream,
                &mut d_split,
                &d_mix,
                &w_attn_scale,
                &w_attn_base,
                N_HC,
                N_SINKHORN_ITERS,
                SINKHORN_EPS,
            )?;
            weighted.launch(&stream, &mut d_attn_cur, &d_inp, &d_split, N_EMBD, N_HC)?;
            stream.synchronize()?;
            d_attn_cur.copy_to_host(&mut got_attn_cur)?;

            let expected_attn = dump.read_f32(
                dump.tensor("attn_cur", layer, token)
                    .ok_or_else(|| eyre!("missing attn_cur L{layer} T{token}"))?,
            )?;
            s_attn.update(&got_attn_cur, &expected_attn, layer, token);

            // (2) hc_post_attn + hc_pre_ffn (uses dumped attn_out as block_out).
            // post is d_split[n_hc..2*n_hc], comb is d_split[2*n_hc..2*n_hc+n_hc*n_hc].
            // For HcPost we pass offset pointers — emulate via index-shifted
            // bind: allocate separate buffers and copy from d_split. Tiny cost.
            let mut post_host = vec![0f32; HC_MIX_DIM as usize];
            d_split.copy_to_host(&mut post_host)?;
            let post_only: Vec<f32> = post_host[N_HC as usize..2 * N_HC as usize].to_vec();
            let comb_only: Vec<f32> =
                post_host[2 * N_HC as usize..2 * N_HC as usize + (N_HC * N_HC) as usize].to_vec();
            let mut d_post: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HC as usize)?;
            let mut d_comb: DeviceBuffer<f32> =
                DeviceBuffer::new(device.id, (N_HC * N_HC) as usize)?;
            d_post.copy_from_host(&post_only)?;
            d_comb.copy_from_host(&comb_only)?;

            let attn_out_entry = dump
                .tensor("attn_out", layer, token)
                .ok_or_else(|| eyre!("missing attn_out L{layer} T{token}"))?;
            d_attn_out.copy_from_host(&dump.read_f32(attn_out_entry)?)?;

            post_k.launch(
                &stream,
                &mut d_after_attn,
                &d_attn_out,
                &d_inp,
                &d_post,
                &d_comb,
                N_EMBD,
                N_HC,
            )?;

            rms.launch(&stream, &mut d_flat2, &d_after_attn, 1, HC_DIM, RMS_EPS)?;
            f16.matvec(&stream, &mut d_mix2, &w_ffn_fn.buffer, &d_flat2, HC_MIX_DIM, HC_DIM)?;
            sinkhorn.launch(
                &stream,
                &mut d_split2,
                &d_mix2,
                &w_ffn_scale,
                &w_ffn_base,
                N_HC,
                N_SINKHORN_ITERS,
                SINKHORN_EPS,
            )?;
            weighted.launch(
                &stream,
                &mut d_ffn_cur,
                &d_after_attn,
                &d_split2,
                N_EMBD,
                N_HC,
            )?;
            stream.synchronize()?;
            d_ffn_cur.copy_to_host(&mut got_ffn_cur)?;

            let expected_ffn = dump.read_f32(
                dump.tensor("ffn_cur", layer, token)
                    .ok_or_else(|| eyre!("missing ffn_cur L{layer} T{token}"))?,
            )?;
            s_ffn.update(&got_ffn_cur, &expected_ffn, layer, token);
        }
    }

    eprintln!(
        "attn_cur: max={:.3e} mean={:.3e} worst L{} T{}, n={}",
        s_attn.max_abs, s_attn.mean_abs(), s_attn.worst.0, s_attn.worst.1, s_attn.count
    );
    eprintln!(
        "ffn_cur:  max={:.3e} mean={:.3e} worst L{} T{}, n={}",
        s_ffn.max_abs, s_ffn.mean_abs(), s_ffn.worst.0, s_ffn.worst.1, s_ffn.count
    );

    // Threshold: chain noise from F16 matvec + sigmoid expf gap + Sinkhorn
    // (multiple exp/normalize iters) + weighted_sum projection.
    const THRESHOLD: f32 = 5.0e-2;
    assert!(s_attn.max_abs < THRESHOLD, "attn_cur max {:.3e}", s_attn.max_abs);
    assert!(s_ffn.max_abs < THRESHOLD, "ffn_cur max {:.3e}", s_ffn.max_abs);
    Ok(())
}
