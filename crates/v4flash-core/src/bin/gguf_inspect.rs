//! gguf-inspect — print a GGUF's metadata + tensor inventory.
//!
//! Usage:
//!   gguf-inspect <path>                — header + metadata + tensor stats
//!   gguf-inspect <path> --tensors      — also dump tensor directory
//!   gguf-inspect <path> --metadata=KEY — dump a specific metadata value

use std::env;
use std::process::ExitCode;

use color_eyre::eyre;
use v4flash_core::gguf::{Gguf, GgufArray, GgufValue};

fn main() -> ExitCode {
    color_eyre::install().ok();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: gguf-inspect PATH [--tensors] [--metadata=KEY]");
        return ExitCode::from(2);
    }
    let path = &args[1];
    let show_tensors = args.iter().any(|a| a == "--tensors");
    let key_filter: Option<String> = args
        .iter()
        .find_map(|a| a.strip_prefix("--metadata=").map(str::to_string));

    match run(path, show_tensors, key_filter.as_deref()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("gguf-inspect: {e:#}");
            ExitCode::from(1)
        }
    }
}

fn run(path: &str, show_tensors: bool, key_filter: Option<&str>) -> eyre::Result<()> {
    let g = Gguf::open(path).map_err(|e| eyre::eyre!("{e}"))?;

    println!("== {path} ==");
    println!("  version             {}", g.version);
    println!("  alignment           {}", g.alignment);
    println!("  n_kv                {}", g.n_kv);
    println!("  n_tensors           {}", g.n_tensors);
    println!("  tensor_data_offset  {} (0x{:x})", g.tensor_data_offset, g.tensor_data_offset);
    println!("  file_size           {} ({:.2} GiB)", g.file_size, g.file_size as f64 / (1u64 << 30) as f64);
    println!("  architecture        {:?}", g.architecture());

    if let Some(key) = key_filter {
        match g.metadata(key) {
            Some(v) => print_value(&format!("{key}"), v, 0),
            None => println!("  (no metadata key {key:?})"),
        }
        return Ok(());
    }

    // Print all metadata keys, with sensible summaries for big arrays.
    println!("\n== metadata ({} entries) ==", g.n_kv);
    let mut keys: Vec<&str> = g.metadata_keys().collect();
    keys.sort();
    for k in keys {
        let v = g.metadata(k).unwrap();
        print_value(k, v, 0);
    }

    // Tensor-type histogram.
    let mut by_type: std::collections::BTreeMap<&str, (usize, u64)> = Default::default();
    let mut total_bytes: u64 = 0;
    for t in g.tensors() {
        let entry = by_type.entry(t.dtype.name()).or_insert((0, 0));
        entry.0 += 1;
        entry.1 += t.byte_size;
        total_bytes += t.byte_size;
    }
    println!("\n== tensor dtype histogram ==");
    for (name, (count, bytes)) in &by_type {
        println!("  {:>9}  {:>6} tensors  {:>10.2} MiB", name, count, *bytes as f64 / (1 << 20) as f64);
    }
    println!("  TOTAL                       {:>10.2} GiB", total_bytes as f64 / (1u64 << 30) as f64);

    if show_tensors {
        println!("\n== tensors ({} entries) ==", g.n_tensors);
        for t in g.tensors() {
            let dims: Vec<String> = t.dims.iter().map(|d| d.to_string()).collect();
            println!(
                "  {:<60} {:>8} [{}] @ 0x{:x}  ({} bytes)",
                t.name,
                t.dtype.name(),
                dims.join("x"),
                t.abs_offset,
                t.byte_size,
            );
        }
    }
    Ok(())
}

fn print_value(key: &str, v: &GgufValue, indent: usize) {
    let pad = " ".repeat(indent);
    match v {
        GgufValue::String(s) => {
            let preview: String = s.chars().take(80).collect();
            let suffix = if s.chars().count() > 80 { "…" } else { "" };
            println!("  {pad}{key} = {preview:?}{suffix}");
        }
        GgufValue::Array(a) => match a {
            GgufArray::String(items) => {
                println!("  {pad}{key} = [string × {}]", items.len());
                if items.len() <= 4 {
                    for (i, s) in items.iter().enumerate() {
                        let preview: String = s.chars().take(40).collect();
                        println!("  {pad}    [{i}] = {preview:?}");
                    }
                }
            }
            GgufArray::I32(items) => {
                println!("  {pad}{key} = [i32 × {}] {:?}", items.len(), preview_slice(items));
            }
            GgufArray::U32(items) => {
                println!("  {pad}{key} = [u32 × {}] {:?}", items.len(), preview_slice(items));
            }
            GgufArray::F32(items) => {
                println!("  {pad}{key} = [f32 × {}] {:?}", items.len(), preview_slice(items));
            }
            other => println!("  {pad}{key} = [array len={}]", other.len()),
        },
        _ => println!("  {pad}{key} = {v:?}"),
    }
}

fn preview_slice<T: std::fmt::Debug>(s: &[T]) -> &[T] {
    &s[..s.len().min(8)]
}
