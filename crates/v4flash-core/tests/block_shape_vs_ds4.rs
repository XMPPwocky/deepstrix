//! Pin `GgufType::block_shape()` against ds4's `ds4q_type_traits` table
//! (`external/ds4/gguf-tools/quants.c`). The two tables describe the same
//! on-disk format; a mismatch is always a bug on our side (this caught
//! IQ4_NL and IQ1_S carrying wrong shapes for months — harmless only
//! because nothing loaded those types).

use std::collections::HashMap;

use v4flash_core::gguf::GgufType;

/// Parse `ds4q_type_traits` entries: `[DS4Q_TYPE_X] = { "name", elems, bytes, ... }`.
/// `QK_K` is 256.
fn parse_ds4_traits(src: &str) -> HashMap<String, (u32, u32)> {
    let mut out = HashMap::new();
    for line in src.lines() {
        let line = line.trim();
        if !line.starts_with("[DS4Q_TYPE_") {
            continue;
        }
        let Some(brace) = line.find('{') else { continue };
        let body = line[brace + 1..].trim_end_matches(&[',', ' ', '}'][..]);
        let fields: Vec<&str> = body.split(',').map(str::trim).collect();
        if fields.len() < 3 {
            continue;
        }
        let name = fields[0].trim_matches('"').to_ascii_lowercase();
        let elems: u32 = match fields[1] {
            "QK_K" => 256,
            n => n.parse().unwrap_or_else(|_| panic!("bad elems in {line:?}")),
        };
        let bytes: u32 = fields[2]
            .parse()
            .unwrap_or_else(|_| panic!("bad bytes in {line:?}"));
        out.insert(name, (elems, bytes));
    }
    out
}

#[test]
fn block_shape_matches_ds4_type_traits() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../external/ds4/gguf-tools/quants.c"
    );
    let src = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("read {path}: {e} (ds4 submodule checked out?)"));
    let ds4 = parse_ds4_traits(&src);
    assert!(
        ds4.len() >= 30,
        "parsed only {} entries from ds4q_type_traits — parser broke?",
        ds4.len()
    );

    let ours: &[GgufType] = &[
        GgufType::F32,
        GgufType::F16,
        GgufType::Q4_0,
        GgufType::Q4_1,
        GgufType::Q5_0,
        GgufType::Q5_1,
        GgufType::Q8_0,
        GgufType::Q8_1,
        GgufType::Q2_K,
        GgufType::Q3_K,
        GgufType::Q4_K,
        GgufType::Q5_K,
        GgufType::Q6_K,
        GgufType::Q8_K,
        GgufType::IQ2_XXS,
        GgufType::IQ2_XS,
        GgufType::IQ3_XXS,
        GgufType::IQ1_S,
        GgufType::IQ4_NL,
        GgufType::IQ3_S,
        GgufType::IQ2_S,
        GgufType::IQ4_XS,
        GgufType::I8,
        GgufType::I16,
        GgufType::I32,
        GgufType::I64,
        GgufType::F64,
        GgufType::IQ1_M,
        GgufType::BF16,
        GgufType::MXFP4,
    ];
    for t in ours {
        let name = t.name();
        let (elems, bytes) = t.block_shape().expect("known type");
        let Some(&(d_elems, d_bytes)) = ds4.get(name) else {
            panic!("type {name} missing from ds4q_type_traits");
        };
        assert_eq!(
            (elems, bytes),
            (d_elems, d_bytes),
            "block_shape mismatch for {name}: ours ({elems},{bytes}) vs ds4 ({d_elems},{d_bytes})"
        );
    }
}
