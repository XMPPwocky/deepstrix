//! DSpark kept-row ring writes enqueued WITHOUT a sync (docs/v41/DSPARK_SINGLE_STREAM_PERF.md 4;
//! `MsDspark::keep_rows` / `settle_writes`): the same sequence of ring writes and drafter forwards
//! run (a) synchronizing after every write, as before, and (b) async -- an event recorded after
//! the write and waited on only before the next write or draft -- with TWO slots' rings swapped
//! into one `MtpState` by pointer, as `MsDspark::with_ring` does, so a write of one slot is still
//! in flight when the other slot's upload starts. Every drafter output (`h` after `forward`) and
//! both slots' final ring bytes must be bit-identical.
//!
//! Loads the drafter only (iGPU, ~8.7 GB): hub DOWN. Run:
//! ```text
//! HIP_VISIBLE_DEVICES=0,1 CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --release --features v41 --test mtp_ring_async -- --ignored --nocapture
//! ```
#![cfg(feature = "v41")]

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer, Event};
use v4flash_kernels::config::HC_DIM;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{mtp_rope, MtpState, MTP_SRC_LAYERS};
use v4flash_kernels::het::weights::MtpWeights;

const HF_DIR_DEFAULT: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

/// Deterministic residual-like rows.
fn rows(seed: u64, n: usize, len: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n * len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 4.0
        })
        .collect()
}

struct Slot {
    rings: Vec<DeviceBuffer<u16>>,
    writes: usize,
    pos: u32,
    done: Event,
    pending: bool,
}

/// One arm: the scripted sequence; returns every drafter output and both rings' bytes.
fn run(e: &DeviceEngine, w: &MtpWeights, dev: &Device, asynchronous: bool) -> eyre::Result<(Vec<Vec<f32>>, Vec<Vec<u16>>)> {
    let k = MTP_SRC_LAYERS.len() * v4flash_kernels::config::N_EMBD as usize;
    let rope = mtp_rope();
    let mut st = MtpState::alloc(dev.id)?;
    let mut slots: Vec<Slot> = (0..2)
        .map(|i| -> eyre::Result<Slot> {
            Ok(Slot {
                rings: st.rings.iter().map(|r| DeviceBuffer::<u16>::new(dev.id, r.len())).collect::<eyre::Result<_>>()?,
                writes: 0,
                pos: 1000 + 37 * i as u32,
                done: Event::new_no_timing()?,
                pending: false,
            })
        })
        .collect::<eyre::Result<_>>()?;
    let settle = |slots: &mut Vec<Slot>| -> eyre::Result<()> {
        for s in slots.iter_mut() {
            if s.pending {
                s.pending = false;
                s.done.synchronize()?;
            }
        }
        Ok(())
    };
    let token_row = rows(7, 1, HC_DIM as usize);
    let noise_row = rows(11, 1, HC_DIM as usize);
    let mut outs = Vec::new();
    // (slot, kept rows, draft after?) -- alternating slots, 1..8-row writes,
    // drafts interleaved, a write of one slot followed by the other's write.
    let script: [(usize, usize, bool); 14] = [
        (0, 8, true), (1, 5, false), (0, 3, true), (1, 1, true), (0, 1, false), (0, 6, true), (1, 8, false),
        (1, 2, true), (0, 4, true), (1, 7, true), (0, 2, false), (1, 3, false), (0, 5, true), (1, 4, true),
    ];
    for (step, &(si, r, draft)) in script.iter().enumerate() {
        settle(&mut slots)?; // as `keep_rows` does before its blocking upload
        let hidden = rows(100 + step as u64, r, k);
        {
            let s = &mut slots[si];
            std::mem::swap(&mut st.rings, &mut s.rings);
            st.set_ring_writes(s.writes);
            st.ring_write_rows(e, &e.compute, w, &rope, s.pos, &hidden)?;
            s.writes = st.ring_writes();
            std::mem::swap(&mut st.rings, &mut s.rings);
            if asynchronous {
                s.done.record(&e.compute)?;
                s.pending = true;
            } else {
                e.compute.synchronize()?;
            }
            s.pos += r as u32;
        }
        if draft {
            settle(&mut slots)?; // as the top of `decode_rows` does
            let s = &mut slots[si];
            let pos = s.pos - 1;
            std::mem::swap(&mut st.rings, &mut s.rings);
            // The draft rewrites the latest ring row (`with_ring(rewind)`).
            st.set_ring_writes(s.writes - 1);
            st.inject_main_hidden(&hidden[(r - 1) * k..r * k])?;
            st.forward(e, &e.compute, w, &rope, pos, &token_row, &noise_row)?;
            e.compute.synchronize()?;
            s.writes = st.ring_writes();
            std::mem::swap(&mut st.rings, &mut s.rings);
            let mut h = vec![0f32; st.h.len()];
            st.h.copy_to_host(&mut h)?;
            outs.push(h);
        }
    }
    settle(&mut slots)?;
    let mut rings = Vec::new();
    for s in &slots {
        for r in &s.rings {
            let mut v = vec![0u16; r.len()];
            r.copy_to_host(&mut v)?;
            rings.push(v);
        }
    }
    Ok((outs, rings))
}

#[test]
#[ignore]
fn async_ring_writes_match_synchronous() -> eyre::Result<()> {
    install_panic_handler()?;
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let hf = V41HfWeights::open(&dir, None)?;
    let dev = pick_igpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let w = MtpWeights::load(&hf, dev, 40)?;
    let (h_sync, ring_sync) = run(&e, &w, &dev, false)?;
    let (h_async, ring_async) = run(&e, &w, &dev, true)?;
    let h_diff = h_sync.iter().zip(&h_async).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.to_bits() != y.to_bits())).count();
    let r_diff = ring_sync.iter().zip(&ring_async).filter(|(a, b)| a != b).count();
    println!("mtp ring async: {} drafter outputs, {h_diff} differ; {} ring buffers, {r_diff} differ", h_sync.len(), ring_sync.len());
    if h_diff > 0 || r_diff > 0 {
        return Err(eyre!("async ring writes are not bit-identical to synchronous ones ({h_diff} outputs, {r_diff} rings)"));
    }
    Ok(())
}
