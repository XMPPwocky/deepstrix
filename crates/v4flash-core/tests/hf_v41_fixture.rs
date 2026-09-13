//! Byte-level oracle for `V41HfWeights`: every tensor in a GGUF written by
//! `scripts/v41_convert/to_gguf.py` (the reference transforms, themselves
//! validated against gguf-py's decoders) must come out of the HF reader with
//! identical name, dtype, dims and bytes.
//!
//! Fixture (server may stay up; ~1 GB RAM):
//!   PYTHONPATH=~/llama.cpp/gguf-py python3 scripts/v41_convert/to_gguf.py \
//!       --out $S/v41-dry.gguf --layers 0-0 --experts 8 --no-globals
//! Run:
//!   V41_FIXTURE_GGUF=$S/v41-dry-00001-of-00001.gguf \
//!       cargo test -p v4flash-core --release --test hf_v41_fixture -- --nocapture
use std::collections::BTreeSet;

use v4flash_core::{MappedGguf, V41HfWeights};

fn layer_of(name: &str) -> Option<usize> {
    name.strip_prefix("blk.")?.split('.').next()?.parse().ok()
}

#[test]
fn hf_reader_matches_converter_fixture() {
    let Ok(fixture) = std::env::var("V41_FIXTURE_GGUF") else {
        eprintln!("skip: V41_FIXTURE_GGUF unset");
        return;
    };
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap())
    });
    let experts: usize = std::env::var("V41_FIXTURE_EXPERTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);

    let g = MappedGguf::open(&fixture).expect("open fixture");
    let hf = V41HfWeights::open(&dir, Some(experts)).expect("open HF checkpoint");
    eprintln!(
        "HF view: {} tensors over {} layers ({} experts), {} raw tensors in {} shards",
        hf.tensors().len(),
        hf.n_layers(),
        hf.n_expert(),
        hf.raw().n_tensors(),
        hf.raw().n_shards()
    );

    let mut n = 0usize;
    let mut bytes = 0u64;
    let mut layers = BTreeSet::new();
    for t in g.gguf().tensors() {
        if let Some(l) = layer_of(&t.name) {
            layers.insert(l);
        }
        let vt = hf.tensor(&t.name).unwrap_or_else(|| panic!("{}: missing from HF view", t.name));
        assert_eq!(vt.dtype, t.dtype, "{}: dtype", t.name);
        assert_eq!(vt.dims, t.dims, "{}: dims", t.name);
        assert_eq!(vt.byte_size, t.byte_size, "{}: byte_size", t.name);
        let a = g.read_tensor(t).expect("gguf read");
        let b = hf.read(vt).expect("hf read");
        if a != b {
            let first = a.iter().zip(&b).position(|(x, y)| x != y).unwrap();
            let cnt = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            panic!(
                "{}: {cnt} of {} bytes differ; first at {first} (gguf {:02x} vs hf {:02x})",
                t.name,
                a.len(),
                a[first],
                b[first]
            );
        }
        n += 1;
        bytes += t.byte_size;
    }

    // Completeness the other way: for the layers the fixture covers, the HF
    // view must not present tensors the converter did not write.
    for vt in hf.tensors() {
        if let Some(l) = layer_of(&vt.name) {
            if layers.contains(&l) {
                assert!(g.gguf().tensor(&vt.name).is_some(), "{}: in HF view but not in fixture", vt.name);
            }
        }
    }

    // Sub-range reads of a stacked expert tensor.
    let l0 = *layers.iter().next().expect("fixture has a layer");
    let vt = hf.get(&format!("blk.{l0}.ffn_down_exps.weight")).unwrap();
    let whole = hf.read(vt).unwrap();
    let per = hf.expert_bytes(vt);
    assert_eq!(per * experts, whole.len());
    let e = experts - 1;
    let mut part = vec![0u8; per];
    hf.read_range_into(vt, (e * per) as u64, &mut part).unwrap();
    assert_eq!(part, whole[e * per..(e + 1) * per], "aligned expert range");
    let off = per - 500;
    let mut part = vec![0u8; 1000];
    hf.read_range_into(vt, off as u64, &mut part).unwrap();
    assert_eq!(part, whole[off..off + 1000], "unaligned range straddling experts 0/1");
    let mut one = vec![0u8; per];
    hf.read_expert_into(vt, 1, &mut one).unwrap();
    assert_eq!(one, whole[per..2 * per], "read_expert_into");

    eprintln!("OK: {n} tensors, {} MiB byte-identical to the converter fixture", bytes >> 20);
}

/// Load-time transform throughput at partial scale (no fixture needed):
///   V41_HF_BENCH=1 V41_FIXTURE_EXPERTS=64 DEEPSTRIX_HF_THREADS=8 \
///       cargo test -p v4flash-core --release --test hf_v41_fixture transform_throughput -- --nocapture
/// Reads are cold each time (the reader drops pages after every pread).
#[test]
fn transform_throughput() {
    if std::env::var("V41_HF_BENCH").is_err() {
        eprintln!("skip: V41_HF_BENCH unset");
        return;
    }
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap())
    });
    let experts: usize = std::env::var("V41_FIXTURE_EXPERTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let layer: usize = std::env::var("V41_BENCH_LAYER").ok().and_then(|s| s.parse().ok()).unwrap_or(2);
    let hf = V41HfWeights::open(&dir, Some(experts)).expect("open HF checkpoint");
    let threads = std::env::var("DEEPSTRIX_HF_THREADS").unwrap_or_else(|_| "default".into());
    for suffix in [
        "ffn_gate_exps.weight",
        "ffn_down_exps.weight",
        "attn_q_b.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "ffn_gate_shexp.weight",
        "attn_compressor_kv.weight",
        "ffn_gate_inp.weight",
    ] {
        let name = format!("blk.{layer}.{suffix}");
        let Some(vt) = hf.tensor(&name) else {
            eprintln!("{name:34} (not present on this layer)");
            continue;
        };
        let mut buf = vec![0u8; vt.byte_size as usize];
        let t0 = std::time::Instant::now();
        hf.read_into(vt, &mut buf).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        let mib = vt.byte_size as f64 / (1u64 << 20) as f64;
        eprintln!(
            "{name:34} {:?} {mib:>8.1} MiB out  {dt:>7.3} s  {:>7.0} MiB/s  (threads={threads})",
            vt.dtype,
            mib / dt
        );
    }
}
