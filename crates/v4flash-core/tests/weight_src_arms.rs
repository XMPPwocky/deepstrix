//! `WeightSrc` must be a transparent seam: the GGUF arm returns exactly the
//! bytes `MappedGguf` returned before (including the inline
//! `abs_offset + e * bpe` expert arithmetic it replaced), and the HF arm returns
//! the same bytes for the same names. Uses the converter fixture (small).
//!   V41_FIXTURE_GGUF=~/.cache/deepstrix/v41/fixtures/v41-dry-00001-of-00001.gguf \
//!     cargo test -p v4flash-core --release --test weight_src_arms -- --nocapture
use v4flash_core::{MappedGguf, V41HfWeights, WeightSrc};

#[test]
fn gguf_arm_is_transparent_and_hf_arm_agrees() {
    let Ok(fixture) = std::env::var("V41_FIXTURE_GGUF") else {
        eprintln!("skip: V41_FIXTURE_GGUF unset");
        return;
    };
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap())
    });
    let experts: usize = std::env::var("V41_FIXTURE_EXPERTS").ok().and_then(|s| s.parse().ok()).unwrap_or(8);
    let m = MappedGguf::open(&fixture).expect("fixture");
    let hf = V41HfWeights::open(&dir, Some(experts)).expect("hf");
    let g = WeightSrc::from(&m);
    let v = WeightSrc::from(&hf);
    assert_eq!(g.tensors().len(), m.gguf().tensors().len());
    let mut n = 0;
    for t in m.gguf().tensors() {
        let direct = m.read_tensor(t).unwrap();
        let via = g.read_tensor(g.tensor(&t.name).unwrap()).unwrap();
        assert_eq!(direct, via, "{}: GGUF arm read_tensor", t.name);
        let mut par = vec![0u8; t.byte_size as usize];
        g.read_tensor_into_slice_parallel(t, &mut par).unwrap();
        assert_eq!(direct, par, "{}: GGUF arm parallel read", t.name);
        let vt = v.tensor(&t.name).expect("HF arm has the tensor");
        assert_eq!((vt.dtype, &vt.dims, vt.byte_size), (t.dtype, &t.dims, t.byte_size), "{}: desc", t.name);
        assert_eq!(direct, v.read_tensor(vt).unwrap(), "{}: HF arm read_tensor", t.name);
        if t.name.ends_with("_exps.weight") {
            let bpe = (t.byte_size / experts as u64) as usize;
            for e in [0usize, 3, experts - 1] {
                let mut inline = vec![0u8; bpe];
                m.read_range_into(t.shard, t.abs_offset + (e as u64) * (bpe as u64), &mut inline).unwrap();
                let mut a = vec![0u8; bpe];
                g.read_expert_into(t, e, &mut a).unwrap();
                assert_eq!(inline, a, "{}: GGUF arm expert {e}", t.name);
                let mut b = vec![0u8; bpe];
                v.read_expert_into(vt, e, &mut b).unwrap();
                assert_eq!(inline, b, "{}: HF arm expert {e}", t.name);
            }
        }
        n += 1;
    }
    let shape_g = g.model_shape().unwrap();
    let shape_v = v.model_shape().unwrap().unwrap();
    eprintln!("OK: {n} tensors through both arms; gguf shape {shape_g:?}, hf shape {shape_v:?}");
    assert_eq!(shape_v.n_layer, 40);
    assert_eq!(shape_v.n_embd, 5120);
    assert_eq!(shape_v.n_expert, 384);
}
