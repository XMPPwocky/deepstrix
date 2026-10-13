//! `read_range_into_direct_mirrored` (box 1's striped layout read, `V41_EXPERT_MIRROR_LAYOUT`) vs the single-drive
//! `read_range_into_direct`, byte for byte, on real expert tensors: whole tensors and odd, unaligned subranges, several
//! mirror shares, into UNALIGNED destinations (the box-1 prefetcher reads into plain heap vectors). Needs the checkpoint
//! (`V41_MODEL_DIR`, e.g. the E100 copy) and a mirror dir (`V41_EXPERT_MIRROR_DIR`, e.g. the YMTC snapshot). Read-only.
use v4flash_core::safetensors::SafetensorsDir;

#[test]
#[ignore]
fn mirrored_layout_read_matches_single_drive() -> color_eyre::eyre::Result<()> {
    let dir = std::env::var("V41_MODEL_DIR").expect("V41_MODEL_DIR");
    let st = SafetensorsDir::open(&dir)?;
    let mut checked = 0;
    for layer in [0usize, 7, 20, 39] {
        for e in [0usize, 5, 383] {
            for which in ["w1", "w2", "w3"] {
                let name = format!("layers.{layer}.ffn.experts.{e}.{which}.weight");
                let Ok(t) = st.get(&name) else { continue };
                assert!(st.has_mirror(t.shard), "no mirror for shard {}", t.shard);
                let len = t.len as usize;
                // (offset, length) cases: whole, an unaligned middle, a tail, tiny.
                for (off, n) in [(0usize, len), (4097, len / 3), (len - 100_001, 100_001), (13, 5000)] {
                    // One byte of slack in front so `&v[1..]` is NOT 4096-aligned.
                    let mut want = vec![0u8; n + 1];
                    assert!(st.read_range_into_direct(t, off as u64, &mut want[1..])?, "single-drive direct");
                    for frac in [0.3f32, 0.4, 0.6, 1.0] {
                        let mut got = vec![0xAAu8; n + 1];
                        assert!(st.read_range_into_direct_mirrored(t, off as u64, &mut got[1..], frac)?, "mirrored");
                        assert!(got[1..] == want[1..], "{name} [{off}, +{n}) frac {frac}: bytes differ");
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 0, "no expert tensors found");
    eprintln!("mirrored layout read == single-drive read on {checked} ranges x 4 shares");
    Ok(())
}
