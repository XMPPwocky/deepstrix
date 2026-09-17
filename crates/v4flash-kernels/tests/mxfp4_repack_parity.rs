//! `mxfp4_repack_hf_to_ggml` must be BIT-IDENTICAL to the CPU repack it replaces.
//!
//! The pager's miss path now uploads raw HF bytes and permutes on the iGPU
//! (`V41_PAGER_GPU_REPACK`). That permutation feeds every MXFP4 matvec, so a
//! wrong nibble here is silent numeric corruption in the MoE — no error, just
//! worse tokens. This test pins it against the exact scalar loop from
//! `hf_v41.rs::read_expert_raw` at both real expert geometries.

use color_eyre::eyre::{self, eyre};
use v4flash_hip::{Device, DeviceBuffer, Stream};
use v4flash_kernels::mxfp4_repack::Mxfp4Repack;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (iGPU) visible"))
}

/// The reference: the CPU loop this kernel replaces, verbatim.
fn cpu_repack(packed: &[u8], scale: &[u8], out: usize, nb: usize) -> Vec<u8> {
    assert_eq!(nb % 8, 0, "super-block layout needs nb % 8 == 0");
    let half = nb * 16;
    let mut dst = vec![0u8; out * nb * 17];
    for r in 0..out {
        let prow = &packed[r * half..(r + 1) * half];
        let srow = &scale[r * nb..(r + 1) * nb];
        let drow = &mut dst[r * nb * 17..(r + 1) * nb * 17];
        // SUPER-BLOCK v2: each 136-byte super-block is [8 x 16 B nibbles][8 B
        // scales]; block k's nibbles at k*16, its scale at 128+k.
        for (sb, sup) in drow.chunks_exact_mut(136).enumerate() {
            for k in 0..8 {
                let j = sb * 8 + k;
                sup[128 + k] = srow[j];
                let pb = &prow[j * 16..(j + 1) * 16];
                for i in 0..8 {
                    let lo = pb[i];
                    let hi = pb[8 + i];
                    sup[k * 16 + 2 * i] = (lo & 0x0F) | (hi << 4);
                    sup[k * 16 + 1 + 2 * i] = (lo >> 4) | (hi & 0xF0);
                }
            }
        }
    }
    dst
}

fn case(dev: &Device, stream: &Stream, rp: &Mxfp4Repack, out: usize, nb: usize, seed: u64) {
    // Deterministic pseudo-random bytes: every nibble position must be exercised,
    // and an all-zero or all-0xFF buffer would hide a swapped lo/hi.
    let mut st = seed | 1;
    let mut next = || {
        st ^= st << 13;
        st ^= st >> 7;
        st ^= st << 17;
        (st >> 24) as u8
    };
    let packed: Vec<u8> = (0..out * nb * 16).map(|_| next()).collect();
    let scale: Vec<u8> = (0..out * nb).map(|_| next()).collect();

    let want = cpu_repack(&packed, &scale, out, nb);

    // The pager uploads the two tensors back to back in ONE buffer, which is what
    // makes the HF form a single contiguous pread. Mirror that exactly.
    let mut src_host = packed.clone();
    src_host.extend_from_slice(&scale);
    assert_eq!(src_host.len(), out * nb * 17, "HF form must match ggml size");

    let mut src = DeviceBuffer::<u8>::new(dev.id, src_host.len()).unwrap();
    src.copy_from_host(&src_host).unwrap();

    // Offset the destination so a slot-base bug cannot pass by landing at 0.
    let off = 3 * out * nb * 17;
    let mut dst = DeviceBuffer::<u8>::new(dev.id, off + out * nb * 17).unwrap();
    dst.copy_from_host(&vec![0xABu8; off + out * nb * 17]).unwrap();

    rp.launch(stream, &mut dst, off, &src, out as u32, nb as u32).unwrap();
    stream.synchronize().unwrap();

    let mut got = vec![0u8; off + out * nb * 17];
    dst.copy_to_host(&mut got).unwrap();

    let tail = &got[off..];
    if tail != want.as_slice() {
        let bad = tail.iter().zip(&want).position(|(a, b)| a != b).unwrap();
        panic!(
            "out={out} nb={nb}: first mismatch at byte {bad} (super {}, off {}): got {:#04x} want {:#04x}",
            bad / 136, bad % 136, tail[bad], want[bad]
        );
    }
    // The bytes BELOW the offset must be untouched.
    assert!(got[..off].iter().all(|&b| b == 0xAB), "repack wrote below dst_off");
    println!("  out={out} nb={nb}: {} bytes bit-identical", want.len());
}

#[test]
#[ignore = "needs a HIP device"]
fn mxfp4_repack_matches_cpu() {
    let dev = pick_igpu().unwrap();
    dev.set_current().unwrap();
    let arch = dev.properties().unwrap().gcn_arch_name;
    let stream = Stream::new(dev.id).unwrap();
    let rp = Mxfp4Repack::for_arch(&arch).unwrap();
    println!("mxfp4_repack parity on {arch}");
    // The two real expert geometries: gate/up are [N_FF_EXP, N_EMBD/32] and
    // down is [N_EMBD, N_FF_EXP/32] — same block count, different shape, so a
    // row/block index swap passes one and fails the other.
    case(&dev, &stream, &rp, 2304, 160, 0x1234_5678);
    case(&dev, &stream, &rp, 5120, 72, 0x9abc_def0);
    // A grid tail: total blocks not a multiple of the 256-wide block. `nb` must
    // stay a multiple of 8 — a super-block is 8 blocks and the matvec reads whole
    // super-blocks, so a partial one has no defined layout.
    case(&dev, &stream, &rp, 7, 8, 0xdead_beef);
}
