//! HEDGED READS on the real drives (`v4flash_core::hedge`; `V41_ENGRAM_HEDGE_MS` / `V41_EXPERT_HEDGE_MS`), with the
//! primary FORCED to stall (`hedge::set_test_primary_delay_ms`, test-only): the hedged result must be BIT-IDENTICAL to a
//! plain read and the mirror must have won. Reads a few MB from each drive (no GPU, no model load); run it in a gate
//! window or a quiet moment -- it adds a little I/O to the production drives. Needs the primary checkpoint dir and the
//! mirror:
//! ```text
//! V41_MODEL_DIR=/weights2/dsv4.1f \
//! V41_EXPERT_MIRROR_DIR=/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277 \
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-core --release --test hedged_reads -- --ignored --nocapture --test-threads=1
//! ```

use std::time::{Duration, Instant};

use v4flash_core::engram_table::EngramTable;
use v4flash_core::hedge;
use v4flash_core::safetensors::SafetensorsDir;

fn open() -> SafetensorsDir {
    let dir = std::env::var("V41_MODEL_DIR").expect("V41_MODEL_DIR=<primary checkpoint dir>");
    assert!(std::env::var("V41_EXPERT_MIRROR_DIR").is_ok(), "set V41_EXPERT_MIRROR_DIR (the mirror copy)");
    SafetensorsDir::open(&dir).expect("open checkpoint")
}

#[test]
#[ignore]
fn engram_hedge_is_bit_identical_and_wins_on_a_stalled_primary() {
    let st = open();
    let reference = EngramTable::open(&st, 1).unwrap();
    let hedged = EngramTable::open(&st, 1).unwrap(); // its own (empty) row cache: every row is a disk miss
    let mut seed: u64 = 0x5DEECE66D;
    let ids: Vec<i64> = (0..96)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % reference.rows) as i64
        })
        .collect();
    let n = ids.len() * v4flash_core::engram_hash::ENGRAM_ROW_DIM;
    let mut a = vec![0f32; n];
    reference.gather(&st, &ids, &mut a, 32).unwrap();
    let s0 = hedge::ENGRAM.snapshot();
    hedge::set_test_primary_delay_ms(300);
    let t = Instant::now();
    let mut b = vec![0f32; n];
    let r = hedged.gather_hedged(&st, &ids, &mut b, 32, Some(Duration::from_millis(5)));
    let took = t.elapsed();
    hedge::set_test_primary_delay_ms(0);
    r.unwrap();
    let s1 = hedge::ENGRAM.snapshot();
    eprintln!("engram hedged gather of {} rows: {took:?}; fired {} chunks {} won by mirror {} by primary {}", ids.len(),
        s1[1] - s0[1], s1[2] - s0[2], s1[4] - s0[4], s1[3] - s0[3]);
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "hedged rows differ from the plain gather");
    assert!(s1[1] > s0[1] && s1[4] > s0[4], "the hedge never fired / the mirror never won");
    assert!(took < Duration::from_millis(300), "waited for the stalled primary ({took:?})");
    // Let the stalled losers finish before the process tears down.
    std::thread::sleep(Duration::from_millis(400));
}

#[test]
#[ignore]
fn expert_hedge_is_bit_identical_and_wins_on_a_stalled_primary() {
    let st = open();
    let name = st
        .names()
        .filter(|n| n.contains(".experts.") && n.ends_with(".weight"))
        .find(|n| st.tensor(n).is_some_and(|t| t.len >= 1 << 20 && st.has_mirror(t.shard)))
        .expect("an expert weight with a mirror")
        .to_string();
    let t = st.tensor(&name).unwrap().clone();
    let frac = 0.4f32;
    let mut a = vec![0u8; t.len as usize];
    hedge::set_expert_hedge_ms(0);
    assert!(st.read_range_into_direct_mirrored(&t, 0, &mut a, frac).unwrap());
    let s0 = hedge::EXPERT.snapshot();
    hedge::set_expert_hedge_ms(5);
    hedge::set_test_primary_delay_ms(300);
    let t0 = Instant::now();
    let mut b = vec![0u8; t.len as usize];
    let r = st.read_range_into_direct_mirrored(&t, 0, &mut b, frac);
    let took = t0.elapsed();
    hedge::set_test_primary_delay_ms(0);
    hedge::set_expert_hedge_ms(0);
    assert!(r.unwrap());
    let s1 = hedge::EXPERT.snapshot();
    eprintln!("expert {name} ({} B) hedged: {took:?}; fired {} won by mirror {}", t.len, s1[1] - s0[1], s1[4] - s0[4]);
    assert!(a == b, "hedged expert bytes differ");
    assert!(s1[1] > s0[1] && s1[4] > s0[4], "the hedge never fired / the mirror never won");
    assert!(took < Duration::from_millis(300), "waited for the stalled primary ({took:?})");
    std::thread::sleep(Duration::from_millis(400));
}
