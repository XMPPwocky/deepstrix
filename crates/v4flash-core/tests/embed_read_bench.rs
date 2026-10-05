//! How fast can an embed phase stream the layers? (docs/v41/EMBED_PHASE_DESIGN.md
//! §15 E7: the 10-05 window measured ~1.8 GB/s with 4 buffered readers and a
//! DONTNEED per piece.) CPU and disk only, no GPU:
//!
//! ```text
//! QWEN3_EMBED_GGUF=/path/Q8_0.gguf [QWEN3_EMBED_GGUF_REPLICAS=/other/disk/copy.gguf] \
//! [QWEN3_EMBED_BENCH_THREADS=4,8,16] \
//!   nix develop -c cargo test --release -p v4flash-core --test embed_read_bench -- --ignored --nocapture
//! ```
//!
//! Reads all 36 layers per configuration: the current buffered reader
//! (`read_layer_into_par`, 4 threads), then O_DIRECT spans
//! (`DirectFiles::read_span`) over every replica at each thread count. Each
//! configuration starts cold (the file's pages are dropped first).

use std::path::PathBuf;
use std::time::Instant;

use v4flash_core::direct_io::{AlignedBuf, DirectFiles};
use v4flash_core::qwen3_embed::Qwen3EmbedModel;
use v4flash_core::MappedGguf;

#[test]
#[ignore]
fn layer_stream_rate() {
    let gguf = std::env::var("QWEN3_EMBED_GGUF").expect("QWEN3_EMBED_GGUF");
    let mut paths = vec![PathBuf::from(&gguf)];
    if let Ok(r) = std::env::var("QWEN3_EMBED_GGUF_REPLICAS") {
        paths.extend(r.split(':').filter(|s| !s.is_empty()).map(PathBuf::from));
    }
    let threads: Vec<usize> = std::env::var("QWEN3_EMBED_BENCH_THREADS")
        .unwrap_or_else(|_| "4,8,16".into())
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let file = MappedGguf::open(&gguf).expect("open");
    let model = Qwen3EmbedModel::from_gguf(&file).expect("model");
    let n = model.cfg.n_layer;
    let total: u64 = (0..n).map(|il| model.layer_span(il).expect("span").1 as u64).sum();
    eprintln!("{n} layers, {:.2} GB per pass", total as f64 / 1e9);

    // Current path: buffered, 4 readers, DONTNEED per piece.
    let _ = file.drop_page_cache();
    let mut buf = vec![0u8; model.layout.bytes];
    let t = Instant::now();
    for il in 0..n {
        model.read_layer_into_par(&file, il, &mut buf, 4).expect("read");
    }
    let s = t.elapsed().as_secs_f64();
    eprintln!("buffered read_layer_into_par x4            : {s:5.2} s  {:5.2} GB/s", total as f64 / s / 1e9);

    let df = DirectFiles::open(&paths).expect("open direct");
    eprintln!("replicas: {:?}", df.describe());
    let mut abuf = AlignedBuf::new(model.max_layer_span_bytes().expect("spans"));
    for &th in &threads {
        let _ = file.drop_page_cache();
        let t = Instant::now();
        for il in 0..n {
            let (off, len) = model.layer_span(il).expect("span");
            df.read_span(off, len, abuf.as_mut_slice(), th).expect("read_span");
        }
        let s = t.elapsed().as_secs_f64();
        eprintln!("O_DIRECT x{th:<3} over {} replica(s)            : {s:5.2} s  {:5.2} GB/s", df.replicas(), total as f64 / s / 1e9);
    }
    // Spot check: the direct span equals the buffered layer read.
    let (off, len) = model.layer_span(n - 1).expect("span");
    let head = df.read_span(off, len, abuf.as_mut_slice(), 4).expect("read_span");
    model.read_layer_into_par(&file, n - 1, &mut buf, 4).expect("read");
    for (t, at) in model.layout.placements(&model.layers[n - 1]) {
        let src = head + (t.offset - off) as usize;
        assert_eq!(&abuf.as_slice()[src..src + t.bytes as usize], &buf[at..at + t.bytes as usize], "{}", t.name);
    }
    eprintln!("direct span == buffered layer read: OK");
}
