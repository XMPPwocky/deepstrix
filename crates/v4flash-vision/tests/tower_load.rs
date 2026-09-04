//! GPU smoke test for `Tower::load` — IGNORED by default. Run ONE at a time:
//!
//!   DEEPSTRIX_MMPROJ=/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf \
//!   DEEPSTRIX_VISION_DEVICE=1 cargo test --release -p v4flash-vision --test tower_load -- --ignored
//!
//! Device 1 (gfx1151 iGPU, host RAM) is the only allowed target while the
//! production server holds the dGPU. Everything is freed on drop.

use std::path::PathBuf;

use v4flash_hip::Device;
use v4flash_vision::mmproj::{read_meta, MmprojHost};
use v4flash_vision::tower::PATCH_K_PAD;
use v4flash_vision::{Tower, PATCH_ELEMS, VIT_DIM};

fn mmproj_path() -> Option<PathBuf> {
    std::env::var_os("DEEPSTRIX_MMPROJ").map(PathBuf::from)
}

#[test]
fn mmproj_header_parses() {
    let Some(p) = mmproj_path() else {
        eprintln!("DEEPSTRIX_MMPROJ unset; skipping");
        return;
    };
    let m = v4flash_core::MappedGguf::open(&p).unwrap();
    let meta = read_meta(&m).unwrap();
    assert_eq!(meta.n_layers, 32);
    assert_eq!(m.gguf().tensors().len(), 427);
}

#[test]
#[ignore]
fn host_load_all_tensors() {
    let p = mmproj_path().expect("DEEPSTRIX_MMPROJ");
    let h = MmprojHost::load(&p).unwrap();
    assert_eq!(h.blocks.len(), 32);
    assert_eq!(h.f16_bytes(), 932_339_712); // 889.15 MiB of f16 per gguf-inspect
    assert_eq!(h.patch_embd_w.len(), 1024 * 588);
    assert_eq!(h.mm1_w.len(), 4096 * 9216);
}

#[test]
#[ignore]
fn tower_load_on_device() {
    let p = mmproj_path().expect("DEEPSTRIX_MMPROJ");
    let dev_id: i32 = std::env::var("DEEPSTRIX_VISION_DEVICE").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    assert_ne!(dev_id, 0, "refusing to touch the dGPU (device 0) while the server is live");
    let device = Device::new(dev_id);
    let name = device.properties().unwrap().name;
    eprintln!("loading tower on device {dev_id} ({name})");
    let t = Tower::load(&p, device).unwrap();
    eprintln!("device bytes = {:.1} MiB", t.device_bytes() as f64 / (1u64 << 20) as f64);
    // f16 weights + f32 biases/norms on device (sentinels stay host-side).
    // The device copy of `v.patch_embd.weight` is zero-padded on K from
    // PATCH_ELEMS (588) to PATCH_K_PAD (608), so it costs 1024*(608-588)*2
    // = 40_960 B MORE than the 932_339_712 B of f16 in the file.
    const PATCH_PAD_BYTES: usize = VIT_DIM * (PATCH_K_PAD - PATCH_ELEMS) * 2; // 40_960
    assert_eq!(PATCH_PAD_BYTES, 40_960);
    assert_eq!(t.device_bytes(), 932_339_712 + PATCH_PAD_BYTES + 827_392);
    // Read one buffer back to prove the upload landed.
    let mut back = vec![0u16; 16];
    t.dev.patch_embd_w.slice_view(0, 16).copy_to_host(&mut back).unwrap();
    assert_eq!(back, t.host.as_ref().unwrap().patch_embd_w[..16]);
    drop(t);
    device.synchronize().unwrap();
}
