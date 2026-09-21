//! `read_range_into_direct_split` vs the unsplit direct read, byte for byte,
//! on real expert tensors. Needs the HF checkpoint (V41_MODEL_DIR) and a mirror
//! dir (V41_EXPERT_MIRROR_DIR) -- for the check the mirror can be symlinks to
//! the same shards, which exercises the two-handle split without a 2nd drive.
use v4flash_core::safetensors::SafetensorsDir;

/// 4096-aligned byte buffer (O_DIRECT needs an aligned destination).
struct Aligned { ptr: *mut u8, len: usize, layout: std::alloc::Layout }
impl Aligned {
    fn new(len: usize) -> Self {
        let layout = std::alloc::Layout::from_size_align(len.max(4096), 4096).unwrap();
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null());
        Self { ptr, len, layout }
    }
}
impl AsMut<[u8]> for Aligned { fn as_mut(&mut self) -> &mut [u8] { unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) } } }
impl AsRef<[u8]> for Aligned { fn as_ref(&self) -> &[u8] { unsafe { std::slice::from_raw_parts(self.ptr, self.len) } } }
impl Drop for Aligned { fn drop(&mut self) { unsafe { std::alloc::dealloc(self.ptr, self.layout) } } }

#[test]
#[ignore]
fn split_read_matches_unsplit() -> color_eyre::eyre::Result<()> {
    let dir = std::env::var("V41_MODEL_DIR").expect("V41_MODEL_DIR");
    let st = SafetensorsDir::open(&dir)?;
    let mut checked = 0;
    for layer in [0usize, 7, 20, 39] {
        for e in [0usize, 5, 383] {
            for which in ["w1", "w2", "w3"] {
                let name = format!("layers.{layer}.ffn.experts.{e}.{which}.weight");
                let Ok(t) = st.get(&name) else { continue };
                let len = t.len as usize;
                assert!(st.has_mirror(t.shard), "no mirror for shard {}", t.shard);
                let cap = SafetensorsDir::direct_capacity_for(len);
                let mut a = Aligned::new(cap);
                let mut b = Aligned::new(cap);
                let pa = st.read_range_into_direct_padded(t, 0, len, a.as_mut())?.expect("direct");
                for frac in [0.3f32, 0.5, 0.6, 0.9] {
                    b.as_mut().fill(0);
                    let pb = st.read_range_into_direct_split(t, 0, len, b.as_mut(), frac)?.expect("split");
                    assert_eq!(pa, pb, "{name}: pad");
                    assert_eq!(&a.as_ref()[pa..pa + len], &b.as_ref()[pb..pb + len], "{name}: bytes differ at frac {frac}");
                    checked += 1;
                }
            }
        }
    }
    println!("mirror split: {checked} reads byte-identical");
    assert!(checked > 0);
    Ok(())
}
