//! gguf-dequant-dense — emit the single-file "dumper variant" GGUF that
//! lets the ds4 CPU reference run the unsloth UD-IQ2_XXS quant mix
//! (Track R of the unsloth plan).
//!
//! ds4's dense matmul paths only understand F32/F16/Q8_0, and its
//! compressor/indexer/HC paths are F16-only. Rather than teach the C
//! reference five new dense quant formats, this tool pre-converts offline:
//!
//! - exact-dequant to **F32** (llama.cpp-scalar math, `kquants.rs`):
//!   blk.*.attn_q_a.weight, blk.*.ffn_{gate,up,down}_shexp.weight,
//!   output.weight, token_embd.weight
//! - convert to **F16** with the engine's load-time semantics
//!   (`weight_contract::convert_to_f16` replica): blk.*.ffn_gate_inp
//!   (BF16), hc_*_fn / output_hc_fn (F32), attn/indexer compressor kv +
//!   gate + blk.*.indexer.attn_q_b (Q8_0), *_ape + indexer.proj (F32)
//! - everything else (routed experts, norms, tid2eid, ...) passes through
//!   byte-identical
//!
//! Split inputs are merged (pass shard 1's path); `split.*` keys are
//! dropped; `deepseek4.vocab_size` is synthesized from the tokenizer if
//! absent (unsloth's converter omits it, ds4 requires it).
//!
//! Usage: gguf-dequant-dense INPUT.gguf OUTPUT.gguf [--force]

use std::fs::File;
use std::io::{BufWriter, Seek};
use std::path::Path;

use color_eyre::eyre::{self, eyre};

use v4flash_core::gguf::{Gguf, GgufType, GgufValue};
use v4flash_core::gguf_write::{GgufWriter, TensorSpec};
use v4flash_core::kquants;
use v4flash_core::mapped::MappedGguf;

/// Streaming chunk size (source bytes, rounded down to whole blocks).
const CHUNK: usize = 8 << 20;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Action {
    Pass,
    ToF32,
    ToF16,
}

/// Collapse `blk.<n>.` to `blk.N.` (mirror of weight_contract::role_of).
fn role_of(name: &str) -> String {
    if let Some(rest) = name.strip_prefix("blk.") {
        if let Some(dot) = rest.find('.') {
            if rest[..dot].bytes().all(|b| b.is_ascii_digit()) {
                return format!("blk.N.{}", &rest[dot + 1..]);
            }
        }
    }
    name.to_string()
}

fn action_for(role: &str) -> Action {
    match role {
        // ds4 dense-matvec consumers: exact dequant to F32.
        "blk.N.attn_q_a.weight"
        | "blk.N.ffn_gate_shexp.weight"
        | "blk.N.ffn_up_shexp.weight"
        | "blk.N.ffn_down_shexp.weight"
        | "output.weight"
        | "token_embd.weight" => Action::ToF32,
        // ds4 F16-only consumers: engine load-time conversion semantics.
        "blk.N.ffn_gate_inp.weight"
        | "blk.N.hc_attn_fn.weight"
        | "blk.N.hc_ffn_fn.weight"
        | "output_hc_fn.weight"
        | "blk.N.attn_compressor_kv.weight"
        | "blk.N.attn_compressor_gate.weight"
        | "blk.N.indexer_compressor_kv.weight"
        | "blk.N.indexer_compressor_gate.weight"
        | "blk.N.indexer.attn_q_b.weight"
        | "blk.N.attn_compressor_ape.weight"
        | "blk.N.indexer_compressor_ape.weight"
        | "blk.N.indexer.proj.weight" => Action::ToF16,
        _ => Action::Pass,
    }
}

/// Source dtypes each action accepts (defensive: an unexpected dtype means
/// the quant mix drifted from what this tool was written for — fail loudly
/// rather than emit a silently-wrong dumper variant).
fn check_src_dtype(action: Action, dt: GgufType, name: &str) -> eyre::Result<()> {
    use GgufType::*;
    let ok = match action {
        Action::Pass => true,
        Action::ToF32 => matches!(dt, Q4_K | Q5_K | Q6_K | Q8_0 | F32 | F16 | BF16),
        Action::ToF16 => matches!(dt, F16 | F32 | BF16 | Q8_0),
    };
    if ok {
        Ok(())
    } else {
        Err(eyre!("{name}: dtype {} unsupported for {action:?}", dt.name()))
    }
}

fn free_bytes(dir: &Path) -> eyre::Result<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(dir.as_os_str().as_bytes())?;
    let mut sv: libc::statvfs = unsafe { std::mem::zeroed() };
    let rc = unsafe { libc::statvfs(c.as_ptr(), &mut sv) };
    if rc != 0 {
        return Err(eyre!("statvfs({}) failed", dir.display()));
    }
    Ok(sv.f_bavail as u64 * sv.f_frsize as u64)
}

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let args: Vec<String> = std::env::args().collect();
    let mut force = false;
    let mut paths = Vec::new();
    for a in &args[1..] {
        match a.as_str() {
            "--force" => force = true,
            other => paths.push(other.to_string()),
        }
    }
    if paths.len() != 2 {
        eprintln!("usage: gguf-dequant-dense INPUT.gguf OUTPUT.gguf [--force]");
        std::process::exit(2);
    }
    let (input, output) = (Path::new(&paths[0]), Path::new(&paths[1]));

    let src = MappedGguf::open(input)?;
    let g = src.gguf();
    eprintln!(
        "input: {} ({} shard(s), {} tensors, {} KVs)",
        input.display(),
        src.n_shards(),
        g.n_tensors,
        g.n_kv
    );

    // ---- metadata: keep order, drop split.*, synthesize vocab_size ----
    let mut kvs: Vec<(&str, &GgufValue)> = Vec::new();
    for (k, v) in g.metadata_in_order() {
        if k.starts_with("split.") {
            continue;
        }
        kvs.push((k, v));
    }
    let synth_vocab: Option<GgufValue> = if g.metadata("deepseek4.vocab_size").is_none() {
        let n = g
            .metadata("tokenizer.ggml.tokens")
            .and_then(|v| match v {
                GgufValue::Array(a) => Some(a.len() as u32),
                _ => None,
            })
            .ok_or_else(|| {
                eyre!("deepseek4.vocab_size missing and tokenizer.ggml.tokens absent")
            })?;
        eprintln!("synthesizing deepseek4.vocab_size = {n} (ds4 requires it; unsloth omits it)");
        Some(GgufValue::U32(n))
    } else {
        None
    };
    if let Some(v) = &synth_vocab {
        kvs.push(("deepseek4.vocab_size", v));
    }

    // ---- tensor directory: same order, converted dtypes ----
    let mut specs = Vec::with_capacity(g.tensors().len());
    let mut n_f32 = 0usize;
    let mut n_f16 = 0usize;
    for t in g.tensors() {
        let role = role_of(&t.name);
        let action = action_for(&role);
        check_src_dtype(action, t.dtype, &t.name)?;
        let dtype = match action {
            Action::Pass => t.dtype,
            Action::ToF32 => {
                if t.dtype != GgufType::F32 {
                    n_f32 += 1;
                }
                GgufType::F32
            }
            Action::ToF16 => {
                if t.dtype != GgufType::F16 {
                    n_f16 += 1;
                }
                GgufType::F16
            }
        };
        specs.push(TensorSpec {
            name: t.name.clone(),
            dims: t.dims.clone(),
            dtype,
        });
    }

    let est: u64 = specs
        .iter()
        .map(|s| s.byte_size().unwrap_or(0) + 64)
        .sum::<u64>()
        + (64 << 20);
    eprintln!(
        "converting {n_f32} tensors -> f32, {n_f16} -> f16; estimated output ~{:.1} GiB",
        est as f64 / (1u64 << 30) as f64
    );

    let out_dir = output.parent().unwrap_or_else(|| Path::new("."));
    let free = free_bytes(out_dir)?;
    if free < est + (est / 50) {
        let msg = format!(
            "only {:.1} GiB free in {} but ~{:.1} GiB needed",
            free as f64 / (1u64 << 30) as f64,
            out_dir.display(),
            est as f64 / (1u64 << 30) as f64
        );
        if force {
            eprintln!("WARNING: {msg} (--force given, continuing)");
        } else {
            return Err(eyre!("{msg}; pass --force to override"));
        }
    }

    // ---- stream ----
    let out_file = File::create(output)?;
    let mut w = GgufWriter::new(BufWriter::with_capacity(4 << 20, out_file), &kvs, &specs, 32)?;

    let mut src_buf = vec![0u8; CHUNK];
    let mut f32_buf: Vec<f32> = Vec::new();
    let mut out_bytes: Vec<u8> = Vec::new();
    let t_start = std::time::Instant::now();
    let mut done_bytes: u64 = 0;

    for (idx, t) in g.tensors().iter().enumerate() {
        let role = role_of(&t.name);
        let action = action_for(&role);
        let (block_elems, block_bytes) = t
            .dtype
            .block_shape()
            .ok_or_else(|| eyre!("{}: unknown dtype", t.name))?;
        let block_bytes = block_bytes as usize;
        // Whole blocks per chunk; every tensor here is block-aligned.
        let chunk_src = (CHUNK / block_bytes).max(1) * block_bytes;
        let mut off = 0u64;
        while off < t.byte_size {
            let take = ((t.byte_size - off) as usize).min(chunk_src);
            let buf = &mut src_buf[..take];
            src.read_range_into(t.shard, t.abs_offset + off, buf)?;
            match action {
                Action::Pass => {
                    w.write_tensor_chunk(buf)?;
                }
                Action::ToF32 => {
                    f32_buf.clear();
                    kquants::dequant_to_f32(t.dtype, buf, &mut f32_buf)?;
                    out_bytes.clear();
                    out_bytes.reserve(f32_buf.len() * 4);
                    for v in &f32_buf {
                        out_bytes.extend_from_slice(&v.to_le_bytes());
                    }
                    w.write_tensor_chunk(&out_bytes)?;
                }
                Action::ToF16 => {
                    let elems = take as u64 / block_bytes as u64 * block_elems as u64;
                    let converted = kquants::convert_to_f16(t.dtype, buf, elems)?;
                    w.write_tensor_chunk(&converted)?;
                }
            }
            off += take as u64;
        }
        done_bytes += w.tensor_bytes(idx);
        if idx % 100 == 0 || idx + 1 == g.tensors().len() {
            eprintln!(
                "  [{:4}/{}] {:>7.1} GiB written  ({:.0} s)  {}",
                idx + 1,
                g.tensors().len(),
                done_bytes as f64 / (1u64 << 30) as f64,
                t_start.elapsed().as_secs_f64(),
                t.name
            );
        }
    }

    let inner = w.finish()?;
    let mut out_file = inner.into_inner().map_err(|e| eyre!("flush: {e}"))?;
    out_file.sync_all()?;
    let final_size = out_file.stream_position().unwrap_or(0);
    // Drop what we just wrote from the page cache — the box's RAM belongs
    // to the inference server.
    {
        use std::os::unix::io::AsRawFd;
        unsafe {
            libc::posix_fadvise(out_file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED);
        }
    }
    drop(out_file);

    // ---- verify: re-parse the directory, check dtype + sizes ----
    let check = Gguf::open(output).map_err(|e| eyre!("re-parse {}: {e}", output.display()))?;
    if check.n_tensors != g.n_tensors {
        return Err(eyre!(
            "verify: wrote {} tensors, re-parse sees {}",
            g.n_tensors,
            check.n_tensors
        ));
    }
    for spec in &specs {
        let t = check
            .tensor(&spec.name)
            .ok_or_else(|| eyre!("verify: {} missing from output", spec.name))?;
        if t.dtype != spec.dtype || t.dims != spec.dims {
            return Err(eyre!(
                "verify: {} is {:?} {:?}, wanted {:?} {:?}",
                spec.name,
                t.dtype,
                t.dims,
                spec.dtype,
                spec.dims
            ));
        }
    }
    eprintln!(
        "OK: {} ({:.1} GiB, {} tensors, {} KVs) in {:.0} s",
        output.display(),
        final_size as f64 / (1u64 << 30) as f64,
        check.n_tensors,
        check.n_kv,
        t_start.elapsed().as_secs_f64()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn role_mapping() {
        assert_eq!(action_for(&role_of("blk.26.attn_q_a.weight")), Action::ToF32);
        assert_eq!(action_for(&role_of("blk.3.ffn_down_shexp.weight")), Action::ToF32);
        assert_eq!(action_for(&role_of("output.weight")), Action::ToF32);
        assert_eq!(action_for(&role_of("token_embd.weight")), Action::ToF32);
        assert_eq!(action_for(&role_of("blk.7.ffn_gate_inp.weight")), Action::ToF16);
        assert_eq!(action_for(&role_of("blk.7.indexer.attn_q_b.weight")), Action::ToF16);
        assert_eq!(action_for(&role_of("blk.7.indexer.proj.weight")), Action::ToF16);
        assert_eq!(action_for(&role_of("output_hc_fn.weight")), Action::ToF16);
        // must NOT touch the routed experts or the Q8 attention stack
        assert_eq!(action_for(&role_of("blk.26.ffn_gate_exps.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.26.ffn_down_exps.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.attn_q_b.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.attn_kv.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.attn_output_a.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.0.ffn_gate_tid2eid.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.exp_probs_b.bias")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.attn_sinks.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.hc_attn_base.weight")), Action::Pass);
        assert_eq!(action_for(&role_of("blk.5.hc_attn_scale.weight")), Action::Pass);
    }
}
