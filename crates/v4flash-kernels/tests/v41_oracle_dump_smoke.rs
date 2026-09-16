//! The V4.1 CPU-oracle dumps, exported by `scripts/v41_oracle/export_bins.py`,
//! must open through the engine's standard activation-dump loader.
//!   V41_ORACLE_BINS=~/.cache/deepstrix/v41/oracle_full_bins cargo test -p v4flash-kernels --release --test v41_oracle_dump_smoke -- --nocapture
use v4flash_kernels::oracle::ActivationDump;

#[test]
fn v41_dump_opens_and_indexes() {
    let Ok(root) = std::env::var("V41_ORACLE_BINS") else {
        {
            if std::env::var("V41_REQUIRE_FIXTURES").as_deref() == Ok("1") {
                panic!("FIXTURE MISSING (V41_REQUIRE_FIXTURES=1): V41_ORACLE_BINS unset -- this is the only non-ignored v41-named test");
            }
            eprintln!("*** SKIPPED, NOTHING TESTED: V41_ORACLE_BINS unset -- this is the only non-ignored v41-named test ***");
        }
        return;
    };
    let d = ActivationDump::open(&root).expect("open dump");
    assert!(d.prompt_len >= 6, "prompt_len {}", d.prompt_len);
    assert_eq!(d.vocab_size, 129_280);
    let e = d.tensor("embed_hc", -1, 0).expect("embed_hc T0");
    assert_eq!(e.shape, vec![4, 5120]);
    let r = d.tensor("residual", 39, (d.prompt_len - 1) as i32).expect("residual L39 last token");
    assert_eq!(r.shape, vec![4, 5120]);
    let bytes = d.read_bytes(r).expect("read residual");
    assert_eq!(bytes.len(), 4 * 5120 * 4);
    let f: Vec<f32> = bytes.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    assert!(f.iter().all(|x| x.is_finite()));
    let ids = d.tensor("topk_ids", 0, 0).expect("topk_ids L0 T0");
    let ids: Vec<i32> = d.read_bytes(ids).unwrap().chunks_exact(4).map(|b| i32::from_le_bytes(b.try_into().unwrap())).collect();
    assert_eq!(ids.len(), 6, "top-6 routed ids");
    assert!(ids.iter().all(|&e| (0..384).contains(&e)), "expert ids in range: {ids:?}");
    let lg = d.tensor("logits", -1, (d.prompt_len - 1) as i32).expect("logits");
    assert_eq!(lg.shape, vec![129_280]);
    eprintln!("OK: T={} layers indexed, residual L39 max|x| = {:.1}", d.prompt_len,
        f.iter().fold(0f32, |a, &x| a.max(x.abs())));
}
