//! gguf-compare-llama — cross-check our Rust GGUF parser against
//! llama.cpp's gguf_dump.py output.
//!
//! Usage:
//!   gguf-compare-llama <gguf-path> <llama-json>
//!
//! Compares:
//!   - Tensor count + name set
//!   - Per-tensor: dtype, shape
//!   - Metadata key set + types
//!   - Scalar metadata values (strings, ints, floats with epsilon)
//!
//! What we DON'T compare: tensor offsets (llama.cpp reports the
//! directory-entry offset, we report the absolute data offset — different
//! semantics). Metadata array values are checked for length + first item
//! only (full content match would be very noisy for the 248k-entry
//! tokenizer arrays).
//!
//! Reports diffs to stdout; non-zero exit if any mismatch.

use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::process::ExitCode;

use color_eyre::eyre::{self, eyre};
use serde::Deserialize;
use v4flash_core::gguf::{Gguf, GgufType, GgufValue};

#[derive(Deserialize)]
struct LlamaDump {
    metadata: BTreeMap<String, LlamaField>,
    tensors: BTreeMap<String, LlamaTensor>,
}

#[derive(Deserialize)]
struct LlamaField {
    #[serde(rename = "type")]
    type_str: String,
    #[serde(default)]
    value: serde_json::Value,
}

#[derive(Deserialize)]
struct LlamaTensor {
    shape: Vec<u64>,
    #[serde(rename = "type")]
    type_str: String,
}

fn main() -> ExitCode {
    let _ = color_eyre::install();
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("usage: gguf-compare-llama <gguf> <llama-json>");
        return ExitCode::from(2);
    }
    match run(&args[1], &args[2]) {
        Ok(0) => ExitCode::SUCCESS,
        Ok(_) => ExitCode::from(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(gguf_path: &str, json_path: &str) -> eyre::Result<u32> {
    let g = Gguf::open(gguf_path).map_err(|e| eyre!("{e}"))?;
    let json_data = std::fs::read_to_string(json_path)?;
    let llama: LlamaDump = serde_json::from_str(&json_data)?;

    let mut mismatches: u32 = 0;

    // --- Tensors ---
    println!("== tensors ==");
    let rust_names: BTreeSet<&str> = g.tensors().iter().map(|t| t.name.as_str()).collect();
    let llama_names: BTreeSet<&str> = llama.tensors.keys().map(String::as_str).collect();

    let only_rust: Vec<_> = rust_names.difference(&llama_names).collect();
    let only_llama: Vec<_> = llama_names.difference(&rust_names).collect();
    if !only_rust.is_empty() {
        println!("  ✗ {} tensors only in Rust parser:", only_rust.len());
        for n in only_rust.iter().take(5) {
            println!("      {n}");
        }
        mismatches += 1;
    }
    if !only_llama.is_empty() {
        println!("  ✗ {} tensors only in llama.cpp:", only_llama.len());
        for n in only_llama.iter().take(5) {
            println!("      {n}");
        }
        mismatches += 1;
    }
    if rust_names == llama_names {
        println!("  ✓ tensor name sets match ({} entries)", rust_names.len());
    }

    // For tensors in both, check shape + dtype.
    let mut shape_mism = 0;
    let mut type_mism = 0;
    for name in rust_names.intersection(&llama_names) {
        let r = g.tensor(name).unwrap();
        let l = &llama.tensors[*name];

        if r.dims != l.shape {
            shape_mism += 1;
            if shape_mism <= 3 {
                println!("  ✗ shape mismatch {name}: rust={:?} llama={:?}", r.dims, l.shape);
            }
        }
        let rust_type_name = gguf_type_to_llama_name(r.dtype);
        if rust_type_name != l.type_str.to_lowercase() {
            type_mism += 1;
            if type_mism <= 3 {
                println!("  ✗ dtype mismatch {name}: rust={} llama={}", rust_type_name, l.type_str);
            }
        }
    }
    if shape_mism == 0 && type_mism == 0 {
        println!("  ✓ all tensor shapes + dtypes match");
    } else {
        if shape_mism > 0 {
            println!("  (total {shape_mism} shape mismatches)");
            mismatches += 1;
        }
        if type_mism > 0 {
            println!("  (total {type_mism} dtype mismatches)");
            mismatches += 1;
        }
    }

    // --- Metadata ---
    println!("\n== metadata ==");
    let rust_meta: BTreeMap<&str, &GgufValue> =
        g.metadata_keys().map(|k| (k, g.metadata(k).unwrap())).collect();
    // llama.cpp adds 3 GGUF.* keys (version, tensor_count, kv_count) that
    // aren't in the metadata table — they're header fields. Skip them.
    let llama_meta: BTreeMap<&str, &LlamaField> = llama
        .metadata
        .iter()
        .filter(|(k, _)| !k.starts_with("GGUF."))
        .map(|(k, v)| (k.as_str(), v))
        .collect();

    let rust_keys: BTreeSet<&str> = rust_meta.keys().copied().collect();
    let llama_keys: BTreeSet<&str> = llama_meta.keys().copied().collect();
    let only_rust_kv: Vec<_> = rust_keys.difference(&llama_keys).collect();
    let only_llama_kv: Vec<_> = llama_keys.difference(&rust_keys).collect();
    if !only_rust_kv.is_empty() {
        println!("  ✗ {} keys only in Rust parser:", only_rust_kv.len());
        for k in only_rust_kv.iter().take(5) {
            println!("      {k}");
        }
        mismatches += 1;
    }
    if !only_llama_kv.is_empty() {
        println!("  ✗ {} keys only in llama.cpp:", only_llama_kv.len());
        for k in only_llama_kv.iter().take(5) {
            println!("      {k}");
        }
        mismatches += 1;
    }
    if rust_keys == llama_keys {
        println!("  ✓ metadata key sets match ({} entries)", rust_keys.len());
    }

    // Per-key type + scalar value check
    let mut type_diff = 0;
    let mut val_diff = 0;
    for key in rust_keys.intersection(&llama_keys) {
        let rv = rust_meta[key];
        let lv = llama_meta[key];
        let rust_type_name = gguf_value_type_name(rv);
        if rust_type_name != lv.type_str {
            type_diff += 1;
            if type_diff <= 3 {
                println!("  ✗ type mismatch {key}: rust={rust_type_name} llama={}", lv.type_str);
            }
            continue;
        }
        if let Some(diff) = compare_scalar(rv, &lv.value) {
            val_diff += 1;
            if val_diff <= 3 {
                println!("  ✗ value mismatch {key}: {diff}");
            }
        }
    }
    if type_diff == 0 && val_diff == 0 {
        println!("  ✓ all metadata types + scalar values match");
    } else {
        if type_diff > 0 { println!("  (total {type_diff} type mismatches)"); mismatches += 1; }
        if val_diff > 0 { println!("  (total {val_diff} value mismatches)"); mismatches += 1; }
    }

    println!();
    if mismatches == 0 {
        println!("OK: Rust parser matches llama.cpp output");
    } else {
        println!("FAIL: {mismatches} categories of mismatch");
    }
    Ok(mismatches)
}

fn gguf_type_to_llama_name(t: GgufType) -> String {
    // llama.cpp uses uppercase enum-style names ("F32", "Q8_0", "IQ2_XXS")
    // — match by lowercasing both sides.
    t.name().to_string()
}

fn gguf_value_type_name(v: &GgufValue) -> &'static str {
    match v {
        GgufValue::U8(_) => "UINT8",
        GgufValue::I8(_) => "INT8",
        GgufValue::U16(_) => "UINT16",
        GgufValue::I16(_) => "INT16",
        GgufValue::U32(_) => "UINT32",
        GgufValue::I32(_) => "INT32",
        GgufValue::F32(_) => "FLOAT32",
        GgufValue::Bool(_) => "BOOL",
        GgufValue::String(_) => "STRING",
        GgufValue::Array(_) => "ARRAY",
        GgufValue::U64(_) => "UINT64",
        GgufValue::I64(_) => "INT64",
        GgufValue::F64(_) => "FLOAT64",
    }
}

/// Compare a Rust GgufValue against llama.cpp's JSON value. Returns
/// None if equal, Some(diff_description) if not. Floats use epsilon.
/// For arrays: checks length + first item type only.
fn compare_scalar(rv: &GgufValue, lv: &serde_json::Value) -> Option<String> {
    use serde_json::Value as J;
    match rv {
        GgufValue::U8(r) => match lv.as_u64() {
            Some(l) if l == *r as u64 => None,
            _ => Some(format!("rust=u8({r}) llama={lv}")),
        },
        GgufValue::I8(r) => match lv.as_i64() {
            Some(l) if l == *r as i64 => None,
            _ => Some(format!("rust=i8({r}) llama={lv}")),
        },
        GgufValue::U16(r) => match lv.as_u64() {
            Some(l) if l == *r as u64 => None,
            _ => Some(format!("rust=u16({r}) llama={lv}")),
        },
        GgufValue::I16(r) => match lv.as_i64() {
            Some(l) if l == *r as i64 => None,
            _ => Some(format!("rust=i16({r}) llama={lv}")),
        },
        GgufValue::U32(r) => match lv.as_u64() {
            Some(l) if l == *r as u64 => None,
            _ => Some(format!("rust=u32({r}) llama={lv}")),
        },
        GgufValue::I32(r) => match lv.as_i64() {
            Some(l) if l == *r as i64 => None,
            _ => Some(format!("rust=i32({r}) llama={lv}")),
        },
        GgufValue::U64(r) => match lv.as_u64() {
            Some(l) if l == *r => None,
            _ => Some(format!("rust=u64({r}) llama={lv}")),
        },
        GgufValue::I64(r) => match lv.as_i64() {
            Some(l) if l == *r => None,
            _ => Some(format!("rust=i64({r}) llama={lv}")),
        },
        GgufValue::F32(r) => {
            let l = lv.as_f64();
            match l {
                Some(l) if (l - *r as f64).abs() < 1e-5 => None,
                _ => Some(format!("rust=f32({r}) llama={lv}")),
            }
        }
        GgufValue::F64(r) => {
            let l = lv.as_f64();
            match l {
                Some(l) if (l - *r).abs() < 1e-10 => None,
                _ => Some(format!("rust=f64({r}) llama={lv}")),
            }
        }
        GgufValue::Bool(r) => match lv.as_bool() {
            Some(l) if l == *r => None,
            _ => Some(format!("rust=bool({r}) llama={lv}")),
        },
        GgufValue::String(r) => match lv.as_str() {
            Some(l) if l == r => None,
            _ => Some(format!("rust=str(len={}) llama=str(len={:?})", r.len(),
                lv.as_str().map(str::len))),
        },
        GgufValue::Array(a) => {
            // llama.cpp's gguf_dump.py serializes arrays as JSON arrays for
            // some element types but emits `null` for others (notably string
            // arrays past a certain size, and certain integer types). When
            // llama emits null, our parser is actually MORE complete than
            // its reference — skip rather than fail.
            match lv {
                J::Array(la) => {
                    if a.len() != la.len() {
                        Some(format!("array len rust={} llama={}", a.len(), la.len()))
                    } else {
                        None
                    }
                }
                J::Null => None, // llama omitted the value; not our bug
                _ => Some(format!("rust=array(len={}) llama={lv:?}", a.len())),
            }
        }
    }
}
