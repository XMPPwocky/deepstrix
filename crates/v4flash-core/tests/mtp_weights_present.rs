//! DSpark drafter weights: every `mtp.{0,1,2}.*` tensor is presented, shaped
//! correctly, and actually LOADED.
//!
//! The last point is the one that matters. `dspark_accept.py` reported 0.44
//! acceptance for weeks because `ffn_norm` was silently left at RMSNorm's init of
//! 1.0 — a 4-6x input-scale error that still produced plausible tokens, so no
//! output-quality check caught it. The trained gains have mean 0.157 / 0.200 /
//! 0.241 per stage, so "not all ones" is a precise regression test for exactly
//! that failure.
//!
//! Run:
//!   nix develop -c cargo test --release -p v4flash-core --test mtp_weights_present -- --ignored --nocapture

use v4flash_core::hf_v41::{MTP_N_EXPERT, MTP_STAGES};
use v4flash_core::V41HfWeights;

fn model() -> String {
    std::env::var("V41_MODEL").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap_or_default())
    })
}

#[test]
#[ignore = "needs the V4.1 checkpoint"]
fn mtp_stages_are_presented_and_loaded() {
    let hf = V41HfWeights::open(&model(), None).expect("open checkpoint");

    for s in 0..MTP_STAGES {
        // --- every tensor the drafter needs resolves, with the right geometry ---
        let expect: &[(&str, &[u64])] = &[
            ("attn_norm.weight", &[5120]),
            ("ffn_norm.weight", &[5120]),
            ("attn_q_a.weight", &[5120, 1280]),
            ("attn_kv.weight", &[5120, 512]),
            ("attn_output_b.weight", &[8192, 5120]),
            ("ffn_gate_inp.weight", &[5120, MTP_N_EXPERT as u64]),
        ];
        for (suffix, dims) in expect {
            let name = format!("mtp.{s}.{suffix}");
            let t = hf.get(&name).unwrap_or_else(|e| panic!("mtp.{s}: {name}: {e}"));
            assert_eq!(
                &t.dims[..],
                *dims,
                "{name}: dims {:?} != expected {dims:?}",
                t.dims
            );
        }

        // --- the router is 128-wide, not the main model's 384 ---
        for role in ["ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"] {
            let t = hf.get(&format!("mtp.{s}.{role}.weight")).expect("expert tensor");
            assert_eq!(
                t.dims[2], MTP_N_EXPERT as u64,
                "mtp.{s}.{role}: {} experts, expected {MTP_N_EXPERT}",
                t.dims[2]
            );
        }

        // --- THE regression test: ffn_norm must be the TRAINED gain, not 1.0 ---
        let t = hf.get(&format!("mtp.{s}.ffn_norm.weight")).unwrap();
        let mut buf = vec![0u8; t.byte_size as usize];
        hf.read_range_into(t, 0, &mut buf).expect("read ffn_norm");
        let v: Vec<f32> = buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        assert!(v.iter().all(|x| x.is_finite()), "mtp.{s}.ffn_norm has non-finite values");
        assert!(
            (mean - 1.0).abs() > 0.1,
            "mtp.{s}.ffn_norm mean is {mean:.4} — that is RMSNorm's INIT, not the trained \
             gain (expected ~0.157/0.200/0.241). This is the exact bug that made DSpark \
             acceptance look like 0.44."
        );
        println!("mtp.{s}: ok, ffn_norm mean {mean:.4}, {MTP_N_EXPERT} experts");
    }

    // The three groups are a 3-LAYER stack, not three drafters: the entry layer
    // carries main_proj/main_norm and the exit layer carries the final norm and
    // the confidence/markov heads. Assert that shape explicitly — getting it
    // wrong is how the first version of this loader failed.
    for (name, dims) in [
        ("mtp.0.main_proj.weight", &[15360u64, 5120][..]),
        ("mtp.0.main_norm.weight", &[5120][..]),
        ("mtp.2.norm.weight", &[5120][..]),
        ("mtp.2.confidence.weight", &[5376, 1][..]),
        // auxiliary n-gram head: a 256-dim hidden, not the 5120 residual
        ("mtp.2.markov_head.weight", &[256, 129280][..]),
    ] {
        let t = hf.get(name).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(&t.dims[..], dims, "{name}: dims {:?} != {dims:?}", t.dims);
        println!("  {name}: {:?}", t.dims);
    }
    for absent in ["mtp.1.main_proj.weight", "mtp.0.confidence.weight", "mtp.1.norm.weight"] {
        assert!(hf.tensor(absent).is_none(), "{absent} should NOT exist — stack shape is wrong");
    }
}
