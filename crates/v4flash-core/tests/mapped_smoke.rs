//! Smoke test for MappedGguf: opens a real GGUF on disk, reads tensor
//! bytes, and sanity-checks that the byte slice matches what we'd
//! expect by manual file seek+read.
//!
//! Requires /persist/lumi/models/gemma-4-E4B-it-Q8_0.gguf. Run with
//! `cargo test -p v4flash-core --test mapped_smoke -- --ignored` on a
//! machine that has it.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use v4flash_core::MappedGguf;

const TEST_GGUF: &str = "/persist/lumi/models/gemma-4-E4B-it-Q8_0.gguf";

#[test]
#[ignore]
fn open_and_read_first_tensor() {
    if !std::path::Path::new(TEST_GGUF).exists() {
        eprintln!("skipping: {TEST_GGUF} not present");
        return;
    }
    let m = MappedGguf::open(TEST_GGUF).expect("open");
    eprintln!(
        "opened: version={} n_tensors={} n_kv={} file_size={}",
        m.gguf().version,
        m.gguf().n_tensors,
        m.gguf().n_kv,
        m.gguf().file_size
    );

    let t = m
        .gguf()
        .tensors()
        .iter()
        .find(|t| t.byte_size > 0)
        .expect("at least one non-empty tensor");
    eprintln!(
        "first non-empty tensor: {:?} type={} bytes={} abs_offset={}",
        t.name, t.dtype.name(), t.byte_size, t.abs_offset
    );

    let mmap_bytes = m.read_tensor(t).expect("read_tensor");
    assert_eq!(mmap_bytes.len(), t.byte_size as usize);

    // Cross-check: pread the same range from the file and compare.
    let mut f = File::open(TEST_GGUF).expect("open file");
    f.seek(SeekFrom::Start(t.abs_offset)).expect("seek");
    let mut pread_bytes = vec![0u8; t.byte_size as usize];
    f.read_exact(&mut pread_bytes).expect("read");

    // Compare a prefix to keep test output small.
    let n = (t.byte_size as usize).min(64);
    assert_eq!(
        &mmap_bytes[..n],
        &pread_bytes[..n],
        "mmap bytes don't match pread for tensor {:?}",
        t.name
    );

    // Also check tail
    let tail = t.byte_size as usize;
    assert_eq!(
        &mmap_bytes[tail - n..tail],
        &pread_bytes[tail - n..tail],
        "mmap tail bytes don't match pread for tensor {:?}",
        t.name
    );

    eprintln!("first 32 bytes: {:02x?}", &mmap_bytes[..32.min(mmap_bytes.len())]);
}

#[test]
#[ignore]
fn tensor_by_name() {
    if !std::path::Path::new(TEST_GGUF).exists() {
        eprintln!("skipping: {TEST_GGUF} not present");
        return;
    }
    let m = MappedGguf::open(TEST_GGUF).expect("open");
    // gemma's LM head — known existing tensor name
    let head_name = m
        .gguf()
        .tensors()
        .iter()
        .find(|t| t.name == "token_embd.weight" || t.name == "output.weight")
        .map(|t| t.name.clone())
        .expect("expected token_embd.weight or output.weight to exist in gemma");

    let t = m.gguf().tensor(&head_name).expect("tensor by name");
    let bytes = m.read_tensor(t).expect("bytes");
    assert!(bytes.len() > 0);
    eprintln!("{}: {} bytes", head_name, bytes.len());
}
