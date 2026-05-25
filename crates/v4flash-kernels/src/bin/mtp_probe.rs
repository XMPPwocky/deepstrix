//! M40-P2.0: dump all `mtp.*` tensors in the V4-Flash GGUF.
//! Confirms MTP weights are present and lets us see their shapes.
//!
//! Usage:
//!   cargo run --release -p v4flash-kernels --bin mtp_probe -- <gguf_path>

use color_eyre::eyre;

fn main() -> eyre::Result<()> {
    let path = std::env::args()
        .nth(1)
        .ok_or_else(|| eyre::eyre!("usage: mtp_probe <gguf_path>"))?;
    let gguf = v4flash_core::MappedGguf::open(&path)?;
    // List total tensor count + extract any tensor that looks MTP-related
    // (mtp/draft/spec/predict prefix, or any layer index >= 43 since main
    // model is 43 layers).
    let all = gguf.gguf().tensors();
    println!("total tensors: {}", all.len());

    // Block-prefixed tensors above L=42 (= the 44th layer = MTP layer).
    let mut high_blk: Vec<_> = all
        .iter()
        .filter(|t| {
            if let Some(rest) = t.name.strip_prefix("blk.") {
                if let Some(dot) = rest.find('.') {
                    let layer_str = &rest[..dot];
                    if let Ok(layer) = layer_str.parse::<i32>() {
                        return layer >= 43;
                    }
                }
            }
            false
        })
        .collect();
    high_blk.sort_by(|a, b| a.name.cmp(&b.name));
    println!("blk.>=43 tensors: {}", high_blk.len());
    for t in &high_blk {
        println!("  {:60} dims={:?} dtype={:?}", t.name, t.dims, t.dtype);
    }

    // Anything matching common spec-decode keywords.
    let keywords = ["mtp", "draft", "spec", "predict", "head_extra", "mediumhead"];
    for kw in keywords {
        let matches: Vec<_> = all.iter().filter(|t| t.name.contains(kw)).collect();
        if !matches.is_empty() {
            println!("matched '{}': {}", kw, matches.len());
            for t in &matches[..] {
                println!("  {:60} dims={:?} dtype={:?}", t.name, t.dims, t.dtype);
            }
        }
    }

    // List ALL distinct top-level prefixes (the part before first dot).
    let mut prefixes: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for t in all {
        let p = t.name.split('.').next().unwrap_or("");
        prefixes.insert(p);
    }
    let mut prefixes_v: Vec<_> = prefixes.into_iter().collect();
    prefixes_v.sort();
    println!("\nALL top-level tensor name prefixes:");
    for p in &prefixes_v {
        let count = all.iter().filter(|t| t.name.split('.').next() == Some(p)).count();
        println!("  {:30} ({} tensors)", p, count);
    }
    Ok(())
}
