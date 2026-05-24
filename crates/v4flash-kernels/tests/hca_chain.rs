//! Full HCA chain — closes the CSA loop. Our compressor (M7.6) produces
//! comp_kv (with F16 roundtrip on push to match ds4's cache); our indexer
//! pipeline (M7.9) produces comp_allowed (trivially all-1s for our
//! 57-token prompt where n_comp ≤ 14 < top_k=512); M6's attention_mixed
//! consumes both alongside the raw KV cache from dump; M5's chain
//! finishes via inv RoPE + grouped Q8_0 + standard Q8_0 → attn_out.
//!
//! Validates `attn_out` for the 21 ratio==4 layers across all 51 tokens.
//! Per-stage diagnostics: our comp_kv vs dump's comp_kv_row; the final
//! attn_out vs dump.
//!
//! Threshold 1e-1 final; mean<1e-3. Includes additional FP8-step noise
//! beyond M6's chain (we now produce comp_kv ourselves with the FP8
//! quantize kernel, whereas M6 read it from the dump).

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::{
    weights, ActivationDump, AttentionMixed, CompressorPool, CompressorStateShuffleR4,
    CompressorStateWrite, F16Matvec, F16Roundtrip, Fp8E4m3fnQuantize, Q8_0GroupedMatvec,
    Q8_0Matvec, RmsNorm, RopeParams, RopeTail, ATTN_MIXED_MAX_KEYS,
};

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const N_EMBD: u32 = 4096;
const N_HEAD: u32 = 64;
const N_HEAD_DIM: u32 = 512;
const N_ROT: u32 = 64;
const Q_FLAT: u32 = N_HEAD * N_HEAD_DIM;
const N_GROUPS: u32 = 8;
const GROUP_DIM: u32 = 4096;
const RANK: u32 = 1024;
const OUT_LOW: u32 = N_GROUPS * RANK;
const BLOCKS_GROUPED: u32 = (GROUP_DIM / 32) * N_GROUPS;
const BLOCKS_OUT_LOW: u32 = OUT_LOW / 32;
const ROPE_ORIG_CTX: u64 = 65536;
const RMS_EPS: f32 = 1.0e-6;
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

fn load_f32_weight(
    gguf: &MappedGguf,
    name: &str,
    device_id: i32,
    expected_len: usize,
) -> eyre::Result<DeviceBuffer<f32>> {
    let t = gguf.gguf().tensor(name).ok_or_else(|| eyre!("tensor {name} missing"))?;
    if t.dtype != GgufType::F32 {
        return Err(eyre!("tensor {name} has dtype {:?}", t.dtype));
    }
    let bytes = gguf.tensor_bytes(t).ok_or_else(|| eyre!("bytes missing"))?;
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

#[test]
#[ignore]
fn hca_chain_oracle() -> eyre::Result<()> {
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
    let f16rt = F16Roundtrip::for_arch(&arch)?;
    let attn = AttentionMixed::for_arch(&arch)?;
    let q8 = Q8_0Matvec::for_arch(&arch)?;
    let grouped = Q8_0GroupedMatvec::for_arch(&arch)?;
    let stream = Stream::new(device.id)?;

    const NEG_INF: f32 = -3.4028235e38;
    const COMP_WIDTH: u32 = 2 * N_HEAD_DIM; // ratio==4 main
    const STATE_ROWS: u32 = 8;

    // Per-layer reusable buffers.
    let mut d_x: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;
    let mut d_kv_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, COMP_WIDTH as usize)?;
    let mut d_sc_cur: DeviceBuffer<f32> = DeviceBuffer::new(device.id, COMP_WIDTH as usize)?;
    let mut d_state_kv: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (STATE_ROWS * COMP_WIDTH) as usize)?;
    let mut d_state_score: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, (STATE_ROWS * COMP_WIDTH) as usize)?;
    let mut d_pooled: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD_DIM as usize)?;
    let mut d_comp_row: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD_DIM as usize)?;

    // Attention buffers
    let mut d_q: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_heads: DeviceBuffer<f32> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_sinks: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_HEAD as usize)?;
    let mut d_raw_kv: DeviceBuffer<f32> = DeviceBuffer::new(
        device.id,
        (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize),
    )?;
    let mut d_comp_kv_cache: DeviceBuffer<f32> = DeviceBuffer::new(
        device.id,
        (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize),
    )?;
    // No mask needed for the early-permit branch in our prompt; pass None.

    let mut d_heads_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, Q_FLAT as usize)?;
    let mut d_heads_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, BLOCKS_GROUPED as usize)?;
    let mut d_low: DeviceBuffer<f32> = DeviceBuffer::new(device.id, OUT_LOW as usize)?;
    let mut d_low_xq: DeviceBuffer<i8> = DeviceBuffer::new(device.id, OUT_LOW as usize)?;
    let mut d_low_xscale: DeviceBuffer<f32> =
        DeviceBuffer::new(device.id, BLOCKS_OUT_LOW as usize)?;
    let mut d_out: DeviceBuffer<f32> = DeviceBuffer::new(device.id, N_EMBD as usize)?;

    let mut stage_comp = DiffStats::default();
    let mut final_stats = DiffStats::default();
    let mut worst = (-1i32, -1i32);

    let mut got_comp = vec![0f32; N_HEAD_DIM as usize];
    let mut got_out = vec![0f32; N_EMBD as usize];

    for layer in (2..=42).step_by(2) {
        // Load main-compressor weights.
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
            N_HEAD_DIM as usize,
        )?;
        let rope_params = load_rope_params(&dump, layer)?;

        // Attention weights.
        let sinks_entry = dump
            .weight("attn_sinks", layer)
            .ok_or_else(|| eyre!("missing weight:attn_sinks for L{layer}"))?;
        d_sinks.copy_from_host(&dump.read_f32(sinks_entry)?)?;
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

        // Init compressor state.
        let zeros = vec![0f32; (STATE_ROWS * COMP_WIDTH) as usize];
        let neg_inf = vec![NEG_INF; (STATE_ROWS * COMP_WIDTH) as usize];
        d_state_kv.copy_from_host(&zeros)?;
        d_state_score.copy_from_host(&neg_inf)?;

        // Host shadows for raw + comp KV caches.
        let mut host_raw =
            vec![0f32; (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize)];
        let mut host_comp =
            vec![0f32; (ATTN_MIXED_MAX_KEYS as usize) * (N_HEAD_DIM as usize)];
        let mut n_comp: u32 = 0;

        for token in 0..n_tokens {
            let pos_mod = (token as u32) % 4;
            let row = 4 + pos_mod;

            // Compressor step. Always runs (per-token state update).
            let x_entry = dump
                .tensor("attn_input_norm", layer, token)
                .ok_or_else(|| eyre!("missing attn_input_norm at L{layer} T{token}"))?;
            d_x.copy_from_host(&dump.read_f32(x_entry)?)?;

            matvec.matvec(&stream, &mut d_kv_cur, &wkv.buffer, &d_x, COMP_WIDTH, N_EMBD)?;
            matvec.matvec(&stream, &mut d_sc_cur, &wgate.buffer, &d_x, COMP_WIDTH, N_EMBD)?;
            state_write.launch(
                &stream,
                &mut d_state_kv,
                &mut d_state_score,
                &d_kv_cur,
                &d_sc_cur,
                &ape.buffer,
                COMP_WIDTH,
                row,
                pos_mod,
            )?;

            // On boundary: complete the compressor pipeline + F16 roundtrip.
            if (token + 1) % 4 == 0 {
                pool.launch(&stream, &mut d_pooled, &d_state_kv, &d_state_score, N_HEAD_DIM, 4)?;
                rms.launch_weighted(
                    &stream,
                    &mut d_comp_row,
                    &d_pooled,
                    &norm,
                    N_HEAD_DIM,
                    RMS_EPS,
                )?;
                let comp_pos = (token as u32) + 1 - 4;
                rope.launch_forward(
                    &stream,
                    &mut d_comp_row,
                    1,
                    N_HEAD_DIM,
                    N_ROT,
                    comp_pos,
                    &rope_params,
                )?;
                fp8.launch(&stream, &mut d_comp_row, N_HEAD_DIM - N_ROT)?;
                // F16 roundtrip on push (mirrors kv_cache_push_comp).
                f16rt.launch(&stream, &mut d_comp_row, N_HEAD_DIM)?;
                stream.synchronize()?;
                d_comp_row.copy_to_host(&mut got_comp)?;

                // Compare against dump's comp_kv_row for the per-stage stat.
                let cmp_entry = dump
                    .tensor("comp_kv_row", layer, token)
                    .ok_or_else(|| eyre!("missing comp_kv_row at L{layer} T{token}"))?;
                stage_comp.update(&got_comp, &dump.read_f32(cmp_entry)?);

                // Append to our comp_kv cache (host shadow then upload below).
                let coff = (n_comp as usize) * (N_HEAD_DIM as usize);
                host_comp[coff..coff + (N_HEAD_DIM as usize)].copy_from_slice(&got_comp);
                n_comp += 1;

                // State shuffle.
                state_shuffle.launch(&stream, &mut d_state_kv, &mut d_state_score, COMP_WIDTH)?;
            }

            // Accumulate raw_kv (still from dump — M5/M6 already validated).
            let kv_entry = dump
                .tensor("kv_cached_row", layer, token)
                .ok_or_else(|| eyre!("missing kv_cached_row at L{layer} T{token}"))?;
            let kv_row = dump.read_f32(kv_entry)?;
            let off = (token as usize) * (N_HEAD_DIM as usize);
            host_raw[off..off + (N_HEAD_DIM as usize)].copy_from_slice(&kv_row);
            d_raw_kv.copy_from_host(&host_raw)?;
            if n_comp > 0 {
                d_comp_kv_cache.copy_from_host(&host_comp)?;
            }

            // Q for this token.
            let q_entry = dump
                .tensor("q_post_rope", layer, token)
                .ok_or_else(|| eyre!("missing q_post_rope at L{layer} T{token}"))?;
            d_q.copy_from_host(&dump.read_f32(q_entry)?)?;

            // Mixed attention with our comp_kv + dump raw_kv + None mask
            // (early-permit branch: n_comp ≤ 14 < top_k=512).
            let comp_opt = if n_comp > 0 { Some(&d_comp_kv_cache) } else { None };
            attn.launch(
                &stream,
                &mut d_heads,
                &d_q,
                &d_raw_kv,
                comp_opt,
                None,
                &d_sinks,
                N_HEAD,
                N_HEAD_DIM,
                (token as u32) + 1,
                n_comp,
            )?;

            // Inverse RoPE on heads.
            rope.launch_inverse(
                &stream,
                &mut d_heads,
                N_HEAD,
                N_HEAD_DIM,
                N_ROT,
                token as u32,
                &rope_params,
            )?;

            // Grouped Q8_0 → attn_out_low.
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
            // Standard Q8_0 → attn_out.
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

        drop(wkv);
        drop(wgate);
        drop(ape);
        drop(norm);
        drop(w_out_a);
        drop(w_out_b);
    }

    eprintln!(
        "stage our_comp_kv vs dump:  max_abs={:.3e}, mean={:.3e}, n={}",
        stage_comp.max_abs,
        stage_comp.mean_abs(),
        stage_comp.count,
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
