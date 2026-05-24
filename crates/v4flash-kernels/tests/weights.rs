//! Smoke test: load the V4 Flash output projection (Q8_0 [4096, 129280],
//! ~537 MB) onto a HIP device via `v4flash_kernels::weights::load_to_device`.
//!
//! Asserts byte count matches the Q8_0 layout formula:
//!   n_vocab * n_embd / 32 (blocks/row) * 34 (bytes/block)
//!   = 129280 * 128 * 34 = 562,814,976 bytes
//!
//! Marked `#[ignore]` — requires the V4 Flash GGUF on disk and a HIP device.

use color_eyre::eyre;
use v4flash_core::MappedGguf;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::weights;

const MODEL_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf";

const EXPECTED_OUTPUT_PROJ_BYTES: u64 = 129_280 * 4_096 / 32 * 34;

#[test]
#[ignore]
fn weights_load_output_proj() -> eyre::Result<()> {
    install_panic_handler()?;

    let gguf = MappedGguf::open(MODEL_PATH)?;
    let devices = Device::all()?;
    let device = devices
        .iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151"))
                .unwrap_or(false)
        })
        .copied()
        .or_else(|| devices.first().copied())
        .ok_or_else(|| eyre::eyre!("no HIP devices"))?;
    device.set_current()?;

    let w = weights::load_to_device(&gguf, "output.weight", device.id)?;

    eprintln!(
        "output.weight: dtype={:?}, shape={:?}, n_elements={}, bytes={}",
        w.dtype,
        w.shape,
        w.n_elements,
        w.buffer.byte_len(),
    );

    assert_eq!(w.shape, vec![4096, 129280], "output.weight shape mismatch");
    assert_eq!(
        w.buffer.byte_len() as u64,
        EXPECTED_OUTPUT_PROJ_BYTES,
        "Q8_0 byte-size formula mismatch"
    );
    Ok(())
}
