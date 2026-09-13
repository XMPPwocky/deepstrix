//! Correctness oracle for the 2026-09-12 fast model-load path.
//!
//! `DEEPSTRIX_FAST_LOAD=1` (default) allocates the DeviceBuffer first and
//! `pread`s the tensor straight into it, skipping the host `Vec` (and its
//! pointless `resize(n, 0)`) and the staged host->device copy. The bytes on
//! the device must be **exactly** the bytes in the file — a silent corruption
//! here would poison every weight in the model, so this checks them directly.
//!
//! Covers the eligible (no host transformation) tensors; the largest ones plus
//! a spread of small ones, since the reader switches between a single `pread`
//! and a 64-way parallel one based on size.
//!
//! Run:
//!   HIP_VISIBLE_DEVICES=0,1 nix develop -c cargo test --release \
//!     -p v4flash-kernels --test fast_load_bytes_match -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::weights::load_to_device;

const MAIN_MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";

#[test]
#[ignore]
fn fast_load_bytes_match() -> eyre::Result<()> {
    install_panic_handler()?;
    let path = std::env::var("DEEPSTRIX_GGUF").unwrap_or_else(|_| MAIN_MODEL_PATH.to_string());
    let gguf = MappedGguf::open(&path)?;
    let dev = Device::all()?
        .into_iter()
        .next()
        .ok_or_else(|| eyre!("no device"))?;

    // Eligible = no host-side transformation, so device bytes == file bytes.
    let mut eligible: Vec<_> = gguf
        .gguf()
        .tensors()
        .iter()
        .filter(|t| t.byte_size > 0 && t.dtype != GgufType::Q8_0)
        .collect();
    eligible.sort_by_key(|t| std::cmp::Reverse(t.byte_size));

    // The reader switches strategy at 2 MB/thread, so test both regimes.
    let big: Vec<_> = eligible.iter().take(6).collect();
    let small: Vec<_> = eligible.iter().rev().take(6).collect();
    let picked: Vec<_> = big.into_iter().chain(small).collect();

    println!("checking {} tensors ({})", picked.len(), path);
    let mut checked_bytes = 0u64;
    for t in picked {
        let name = t.name.clone();
        let w = match load_to_device(&gguf, &name, dev.id) {
            Ok(w) => w,
            Err(e) => {
                // ToF16-role tensors are not eligible; skip rather than fail.
                println!("  skip {name}: {e}");
                continue;
            }
        };
        let mut got = vec![0u8; w.buffer.len()];
        w.buffer.copy_to_host(&mut got)?;
        let want = gguf.read_tensor(t)?;
        if got.len() != want.len() {
            return Err(eyre!(
                "{name}: device {} bytes != file {} bytes",
                got.len(),
                want.len()
            ));
        }
        if got != want {
            let at = got.iter().zip(&want).position(|(a, b)| a != b).unwrap();
            return Err(eyre!(
                "{name}: MISMATCH at byte {at} (device 0x{:02X}, file 0x{:02X}) — \
                 fast load is corrupting weights",
                got[at],
                want[at]
            ));
        }
        checked_bytes += got.len() as u64;
        println!(
            "  ok {name}: {} bytes ({}) identical",
            got.len(),
            t.dtype.name()
        );
    }
    println!(
        "PASS — {:.2} GiB of device weights byte-identical to the file",
        checked_bytes as f64 / 2f64.powi(30)
    );
    Ok(())
}
