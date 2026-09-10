//! Snapshot v5 round trip for the packed stores (FP8 compressed KV + E2M1
//! indexer keys), plus the refusal path:
//!
//! 1. packed state -> save (v5) -> restore into a fresh packed state: main
//!    rows, head shadow and index rows byte-identical.
//! 2. the same file restored into a state whose stores are f16
//!    (INDEXER_KEYS_E2M1=0 / COMP_KV_FP8=0 shape) is REFUSED (encoding
//!    mismatch) — snapshots are a cache, there is no conversion.
//!
//! Needs the GGUF header for the vocab (snapshot keys hash decoded token
//! bytes) — `DEEPSTRIX_GGUF` or the production default — and both GPUs;
//! allocates a 4K-context state (a few tens of MiB), so it is safe beside
//! the live server. No weights are loaded.
//!
//! `cargo test -p deepstrix-server --release --test snapshot_fp8_roundtrip -- --ignored --nocapture`

use std::path::PathBuf;

use color_eyre::eyre::{self, eyre};
use deepstrix_server::embed::build_gpt2_byte_decoder;
use deepstrix_server::snapshot::{self, ModelFingerprint, RestoreKernels};
use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Stream};
use v4flash_kernels::comp_kv_fp8::{unpack_row_host, FP8_KV_HEAD_DIM, FP8_KV_HEAD_ROWS, FP8_KV_ROW_BYTES};
use v4flash_kernels::config::COMPRESS_RATIOS;
use v4flash_kernels::het::state::{CompKvStore, HetModelState};
use v4flash_kernels::index_kv_e2m1::{E2M1_KEY_DIM, E2M1_KEY_ROW_BYTES};
use v4flash_kernels::{CompKvAppend, CompKvFp8, F16Roundtrip, Fp8E4m3fnQuantize, IndexKvE2m1, IndexerQat};

const DEFAULT_GGUF: &str = "/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf";
const N_KV_MAX: u32 = 4096;
const N_COMP_R4: u32 = 700; // > FP8_KV_HEAD_ROWS so the shadow cut-off is exercised
const N_COMP_R128: u32 = 20;
const N_ICOMP: u32 = 300;

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn devices() -> eyre::Result<(Device, Device)> {
    let all = Device::all()?;
    let mut dgpu = None;
    let mut igpu = None;
    for d in &all {
        let arch = d.properties()?.gcn_arch_name;
        if arch.starts_with("gfx1201") {
            dgpu = Some(*d);
        } else if arch.starts_with("gfx1151") {
            igpu = Some(*d);
        }
    }
    Ok((dgpu.ok_or_else(|| eyre!("no gfx1201"))?, igpu.ok_or_else(|| eyre!("no gfx1151"))?))
}

fn gguf_path() -> PathBuf {
    std::env::var("DEEPSTRIX_GGUF").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(DEFAULT_GGUF))
}

fn fp() -> ModelFingerprint {
    ModelFingerprint {
        n_layer: 43,
        n_head_dim: 512,
        vocab_size: 1,
        token_embd_prefix_blake3: "fp8-kv-roundtrip".into(),
        tensor_directory_blake3: "test".into(),
    }
}

fn scratch_root() -> PathBuf {
    let base = std::env::var("CLAUDE_SCRATCHPAD")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let p = base.join(format!("snapshot-fp8-roundtrip-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

struct Kernels {
    fp8: Fp8E4m3fnQuantize,
    f16rt: F16Roundtrip,
    append: CompKvAppend,
    packed: CompKvFp8,
    qat: IndexerQat,
    keys: IndexKvE2m1,
}

/// Fill every compressor with random rows through the production
/// producers; returns per-layer (rows bytes | f16 words) images read back
/// from the device for later comparison.
fn fill(state: &mut HetModelState, k: &Kernels, dev: i32, stream: &Stream, rng: &mut Rng) -> eyre::Result<(Vec<Vec<u8>>, Vec<Vec<u8>>)> {
    let mut images = Vec::new();
    let mut index_images = Vec::new();
    for (li, layer) in state.layers.iter_mut().enumerate() {
        let ratio = COMPRESS_RATIOS[li];
        let Some(comp) = layer.compressor.as_mut() else {
            images.push(Vec::new());
            index_images.push(Vec::new());
            continue;
        };
        let n_comp = if ratio == 4 { N_COMP_R4 } else { N_COMP_R128 };
        let head_dim = comp.head_dim as usize;
        let mut rows = vec![0f32; n_comp as usize * head_dim];
        for v in rows.iter_mut() {
            let u = rng.unit() * 2.0 - 1.0;
            *v = match rng.next() % 5 {
                0 => u * 1e-5,
                1 => u * 300.0,
                _ => u * 10f32.powf(rng.unit() * 3.0 - 1.5),
            };
        }
        let mut x = DeviceBuffer::<f32>::new(dev, rows.len())?;
        x.copy_from_host(&rows)?;
        match &mut comp.comp_kv {
            CompKvStore::Fp8 { rows: packed, head } => {
                k.packed.launch_append_batched(stream, packed, head, &x, 0, n_comp, FP8_KV_HEAD_ROWS as u32)?;
                stream.synchronize()?;
                let mut img = vec![0u8; n_comp as usize * FP8_KV_ROW_BYTES];
                packed.slice_view(0, img.len()).copy_to_host(&mut img)?;
                images.push(img);
            }
            CompKvStore::F16(buf) => {
                k.fp8.launch_batched(stream, &mut x, (head_dim - 64) as u32, head_dim as u32, n_comp)?;
                k.f16rt.launch(stream, &mut x, rows.len() as u32)?;
                k.append.launch_batched(stream, buf, &x, 0, head_dim as u32, n_comp)?;
                stream.synchronize()?;
                let mut img16 = vec![0u16; n_comp as usize * head_dim];
                buf.slice_view(0, img16.len()).copy_to_host(&mut img16)?;
                images.push(img16.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
            CompKvStore::E2m1(_) => unreachable!("main compressor is never E2M1"),
        }
        comp.n_comp = n_comp;
        layer.n_raw = 0;
        // Indexer compressor (ratio-4 only): random rows -> QAT -> packed
        // (or f16) keys.
        if let Some(icomp) = layer.indexer_compressor.as_mut() {
            let mut krows = vec![0f32; N_ICOMP as usize * E2M1_KEY_DIM];
            for v in krows.iter_mut() {
                *v = (rng.unit() * 2.0 - 1.0) * 10f32.powf(rng.unit() * 2.0 - 1.0);
            }
            let mut kx = DeviceBuffer::<f32>::new(dev, krows.len())?;
            kx.copy_from_host(&krows)?;
            k.qat.launch(stream, &mut kx, N_ICOMP)?;
            match &mut icomp.comp_kv {
                CompKvStore::E2m1(rows) => {
                    k.keys.launch_append_batched(stream, rows, &kx, 0, N_ICOMP)?;
                    stream.synchronize()?;
                    let mut img = vec![0u8; N_ICOMP as usize * E2M1_KEY_ROW_BYTES];
                    rows.slice_view(0, img.len()).copy_to_host(&mut img)?;
                    index_images.push(img);
                }
                CompKvStore::F16(buf) => {
                    k.f16rt.launch(stream, &mut kx, krows.len() as u32)?;
                    k.append.launch_batched(stream, buf, &kx, 0, E2M1_KEY_DIM as u32, N_ICOMP)?;
                    stream.synchronize()?;
                    let mut img16 = vec![0u16; krows.len()];
                    buf.slice_view(0, img16.len()).copy_to_host(&mut img16)?;
                    index_images.push(img16.iter().flat_map(|v| v.to_le_bytes()).collect());
                }
                CompKvStore::Fp8 { .. } => unreachable!(),
            }
            icomp.n_comp = N_ICOMP;
        } else {
            index_images.push(Vec::new());
        }
    }
    Ok((images, index_images))
}

/// Read back what a state holds, in the same shape `fill` returned, plus
/// the head shadows of FP8 stores.
fn read_back(state: &HetModelState) -> eyre::Result<(Vec<Vec<u8>>, Vec<Vec<u16>>, Vec<Vec<u8>>)> {
    let mut images = Vec::new();
    let mut heads = Vec::new();
    let mut index_images = Vec::new();
    for layer in &state.layers {
        match layer.indexer_compressor.as_ref().map(|c| (&c.comp_kv, c.n_comp as usize)) {
            Some((CompKvStore::E2m1(rows), n)) => {
                let mut img = vec![0u8; n * E2M1_KEY_ROW_BYTES];
                rows.slice_view(0, img.len()).copy_to_host(&mut img)?;
                index_images.push(img);
            }
            Some((CompKvStore::F16(buf), n)) => {
                let mut img16 = vec![0u16; n * E2M1_KEY_DIM];
                buf.slice_view(0, img16.len()).copy_to_host(&mut img16)?;
                index_images.push(img16.iter().flat_map(|v| v.to_le_bytes()).collect());
            }
            _ => index_images.push(Vec::new()),
        }
        let Some(comp) = layer.compressor.as_ref() else {
            images.push(Vec::new());
            heads.push(Vec::new());
            continue;
        };
        let n_comp = comp.n_comp as usize;
        let head_dim = comp.head_dim as usize;
        match &comp.comp_kv {
            CompKvStore::Fp8 { rows, head } => {
                let mut img = vec![0u8; n_comp * FP8_KV_ROW_BYTES];
                rows.slice_view(0, img.len()).copy_to_host(&mut img)?;
                images.push(img);
                let hn = n_comp.min(FP8_KV_HEAD_ROWS) * head_dim;
                let mut h = vec![0u16; hn];
                head.slice_view(0, hn).copy_to_host(&mut h)?;
                heads.push(h);
            }
            CompKvStore::F16(buf) => {
                let mut img16 = vec![0u16; n_comp * head_dim];
                buf.slice_view(0, img16.len()).copy_to_host(&mut img16)?;
                images.push(img16.iter().flat_map(|v| v.to_le_bytes()).collect());
                heads.push(Vec::new());
            }
            CompKvStore::E2m1(_) => unreachable!("main compressor is never E2M1"),
        }
    }
    Ok((images, heads, index_images))
}

fn expected_heads(images: &[Vec<u8>], state: &HetModelState) -> Vec<Vec<u16>> {
    let mut out = Vec::new();
    for (li, layer) in state.layers.iter().enumerate() {
        match layer.compressor.as_ref().map(|c| &c.comp_kv) {
            Some(CompKvStore::Fp8 { .. }) => {
                let n = (images[li].len() / FP8_KV_ROW_BYTES).min(FP8_KV_HEAD_ROWS);
                let mut h = vec![0u16; n * FP8_KV_HEAD_DIM];
                for r in 0..n {
                    unpack_row_host(&images[li][r * FP8_KV_ROW_BYTES..], &mut h[r * FP8_KV_HEAD_DIM..(r + 1) * FP8_KV_HEAD_DIM]);
                }
                out.push(h);
            }
            _ => out.push(Vec::new()),
        }
    }
    out
}

#[test]
#[ignore]
fn snapshot_fp8_round_trip_and_conversions() -> eyre::Result<()> {
    install_panic_handler()?;
    let (dgpu, igpu) = devices()?;
    dgpu.set_current()?;
    let arch = dgpu.properties()?.gcn_arch_name;
    let stream = Stream::new(dgpu.id)?;
    let k = Kernels {
        fp8: Fp8E4m3fnQuantize::for_arch(&arch)?,
        f16rt: F16Roundtrip::for_arch(&arch)?,
        append: CompKvAppend::for_arch(&arch)?,
        packed: CompKvFp8::for_arch(&arch)?,
        qat: IndexerQat::for_arch(&arch)?,
        keys: IndexKvE2m1::for_arch(&arch)?,
    };
    let kernels = RestoreKernels { fp8: &k.packed, stream: &stream };

    let gguf = MappedGguf::open(&gguf_path())?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;
    let byte_decoder = build_gpt2_byte_decoder();
    let root = scratch_root();
    let fingerprint = fp();
    let tokens: Vec<i32> = (0..300).map(|i| 1000 + (i * 7) % 5000).collect();

    // 1. packed -> v5 -> packed.
    let mut src = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    assert!(
        src.layers.iter().any(|l| matches!(l.compressor.as_ref().map(|c| &c.comp_kv), Some(CompKvStore::Fp8 { .. }))),
        "expected FP8 main stores (COMP_KV_FP8 unset)"
    );
    assert!(
        src.layers.iter().any(|l| matches!(l.indexer_compressor.as_ref().map(|c| &c.comp_kv), Some(CompKvStore::E2m1(_)))),
        "expected E2M1 index stores (INDEXER_KEYS_E2M1 unset)"
    );
    let mut rng = Rng(0xC0FFEE_1234_5678);
    let (images, index_images) = fill(&mut src, &k, dgpu.id, &stream, &mut rng)?;
    let entry = snapshot::save(&src, &tokens, &[], dgpu, igpu, &fingerprint, &root, &vocab, &byte_decoder, None)?;
    let dir = entry.dir.clone();
    let comp_kv_bytes = std::fs::metadata(dir.join("comp_kv.bin"))?.len();
    let index_bytes = std::fs::metadata(dir.join("index_comp_kv.bin"))?.len();
    println!("saved v5 snapshot: {} ({comp_kv_bytes} B comp_kv.bin, {index_bytes} B index_comp_kv.bin)", dir.display());
    let expect_index: u64 = 21 * N_ICOMP as u64 * E2M1_KEY_ROW_BYTES as u64;
    assert_eq!(index_bytes, expect_index, "index blob must be the packed size");

    let mut dst = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    let r = snapshot::restore_vl(&mut dst, &dir, dgpu, igpu, &fingerprint, kernels)?;
    assert_eq!(r.tokens, tokens);
    let (got, heads, got_index) = read_back(&dst)?;
    let want_heads = expected_heads(&images, &dst);
    for li in 0..got.len() {
        assert_eq!(dst.layers[li].compressor.as_ref().map(|c| c.n_comp), src.layers[li].compressor.as_ref().map(|c| c.n_comp), "L{li} n_comp");
        assert!(got[li] == images[li], "L{li}: restored main rows differ from saved");
        assert!(heads[li] == want_heads[li], "L{li}: rebuilt head shadow differs");
        assert!(got_index[li] == index_images[li], "L{li}: restored index rows differ from saved");
    }
    println!("  v5 -> packed stores: {} layers main rows + head shadow + index rows byte-identical", got.len());

    // 2. The same file into f16 stores must be REFUSED (no conversion).
    let mut dst16 = HetModelState::alloc(dgpu, igpu, N_KV_MAX)?;
    for layer in dst16.layers.iter_mut() {
        if let Some(c) = layer.indexer_compressor.as_mut() {
            if c.comp_kv.is_e2m1() {
                let cap = c.comp_kv.capacity_rows(c.head_dim);
                c.comp_kv = CompKvStore::F16(DeviceBuffer::new(dgpu.id, cap * c.head_dim as usize)?);
            }
        }
    }
    match snapshot::restore_vl(&mut dst16, &dir, dgpu, igpu, &fingerprint, kernels) {
        Ok(_) => return Err(eyre!("restore into an f16 index store must be refused")),
        Err(e) => {
            let msg = format!("{e:#}");
            assert!(msg.contains("does not match the live store"), "unexpected error: {msg}");
            println!("  encoding mismatch refused as expected: {}", msg.lines().next().unwrap_or(""));
        }
    }
    let _ = FP8_KV_HEAD_DIM;
    let _ = std::fs::remove_dir_all(&root);
    Ok(())
}
