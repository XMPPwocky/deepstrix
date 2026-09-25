//! Box 2's drive routing, on a synthetic checkpoint (no model needed): a
//! `model.safetensors` with two U8 tensors and a same-size MIRROR copy holding
//! DIFFERENT bytes, so every read proves which drive it came from.
//! `read_range_into_direct_split` at exactly 1 / 0 must read wholly from the
//! mirror / primary (no clamped block on the other side); in between, a
//! 4096-aligned head from the primary and the tail from the mirror; the small
//! tensor takes the unsplit path (mirror only at exactly 1);
//! `read_range_into_direct_padded_on` picks the drive; with NO mirror every
//! route falls back to the primary. O_DIRECT needs a real filesystem, so the
//! files go under CARGO_TARGET_TMPDIR (inside target/).
use v4flash_core::safetensors::SafetensorsDir;

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

const NX: usize = 7 * 4096 + 123; // split path, not block-aligned
const NY: usize = 3000; // < 2 blocks: the unsplit small-read path

/// `model.safetensors` with tensors x (NX bytes) then y (NY bytes); data byte
/// i (over both) = f(i).
fn write_st(dir: &std::path::Path, f: impl Fn(usize) -> u8) {
    std::fs::create_dir_all(dir).unwrap();
    let mut header = format!(
        r#"{{"x":{{"dtype":"U8","shape":[{NX}],"data_offsets":[0,{NX}]}},"y":{{"dtype":"U8","shape":[{NY}],"data_offsets":[{NX},{}]}}}}"#,
        NX + NY
    )
    .into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend((0..NX + NY).map(f));
    std::fs::write(dir.join("model.safetensors"), out).unwrap();
}

fn fp(i: usize) -> u8 { (i % 251) as u8 }
fn fm(i: usize) -> u8 { ((i * 7 + 3) % 253) as u8 }

#[test]
fn drive_routing_reads_the_right_drive() {
    let base = std::path::PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("drive-route-{}", std::process::id()));
    let (prim, mirr) = (base.join("primary"), base.join("mirror"));
    write_st(&prim, fp);
    write_st(&mirr, fm);
    std::env::set_var("V41_EXPERT_MIRROR_DIR", &mirr);
    let st = SafetensorsDir::open(&prim).expect("open");
    let x = st.get("x").expect("tensor x");
    let y = st.get("y").expect("tensor y");
    let mut buf = Aligned::new(SafetensorsDir::direct_capacity_for(NX));
    let want = |f: fn(usize) -> u8, off: usize, n: usize| -> Vec<u8> { (off..off + n).map(f).collect() };

    // Primary, unsplit (also: is O_DIRECT available here at all?).
    let Some(pad) = st.read_range_into_direct_padded(x, 0, NX, buf.as_mut()).expect("padded") else {
        eprintln!("SKIP: no O_DIRECT on {}", prim.display());
        let _ = std::fs::remove_dir_all(&base);
        return;
    };
    assert!(st.has_mirror(x.shard), "mirror not picked up (same size, same name)");
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fp, 0, NX)[..], "padded: primary bytes");
    // Mirror, unsplit (the scale plane under mirror_only).
    buf.as_mut().fill(0);
    let pad = st.read_range_into_direct_padded_on(x, 0, NX, buf.as_mut(), true).unwrap().unwrap();
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fm, 0, NX)[..], "padded_on(mirror): mirror bytes");
    // Split at exactly 1.0: ALL from the mirror (no clamped primary block).
    buf.as_mut().fill(0);
    let pad = st.read_range_into_direct_split(x, 0, NX, buf.as_mut(), 1.0).unwrap().unwrap();
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fm, 0, NX)[..], "split 1.0: all mirror");
    // Split at exactly 0.0: ALL from the primary.
    buf.as_mut().fill(0);
    let pad = st.read_range_into_direct_split(x, 0, NX, buf.as_mut(), 0.0).unwrap().unwrap();
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fp, 0, NX)[..], "split 0.0: all primary");
    // In between: a 4096-aligned (in the padded span) head from the primary,
    // the rest from the mirror, both non-empty.
    let (wp, wm) = (want(fp, 0, NX), want(fm, 0, NX));
    for frac in [0.3f32, 0.7, 0.99] {
        buf.as_mut().fill(0);
        let pad = st.read_range_into_direct_split(x, 0, NX, buf.as_mut(), frac).unwrap().unwrap();
        let got = &buf.as_ref()[pad..pad + NX];
        let cut = (0..NX).find(|&i| got[i] != wp[i]).expect("some bytes must come from the mirror");
        assert!(cut > 0, "frac {frac}: nothing from the primary");
        assert_eq!((pad + cut) % 4096, 0, "frac {frac}: cut not block-aligned in the span");
        assert_eq!(&got[..cut], &wp[..cut], "frac {frac}: head");
        assert_eq!(&got[cut..], &wm[cut..], "frac {frac}: tail");
    }
    // Small tensor (< 2 blocks): unsplit, mirror only at exactly 1.0.
    for (frac, f, what) in [(1.0f32, fm as fn(usize) -> u8, "mirror"), (0.7, fp, "primary"), (0.0, fp, "primary")] {
        buf.as_mut().fill(0);
        let pad = st.read_range_into_direct_split(y, 0, NY, buf.as_mut(), frac).unwrap().unwrap();
        assert_eq!(&buf.as_ref()[pad..pad + NY], &want(f, NX, NY)[..], "small tensor at {frac}: {what}");
    }

    // NO mirror: every route falls back to the primary.
    std::env::remove_var("V41_EXPERT_MIRROR_DIR");
    let st2 = SafetensorsDir::open(&prim).expect("open without mirror");
    let x2 = st2.get("x").unwrap();
    assert!(!st2.has_mirror(x2.shard));
    buf.as_mut().fill(0);
    let pad = st2.read_range_into_direct_padded_on(x2, 0, NX, buf.as_mut(), true).unwrap().unwrap();
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fp, 0, NX)[..], "no mirror, padded_on(mirror): primary bytes");
    buf.as_mut().fill(0);
    let pad = st2.read_range_into_direct_split(x2, 0, NX, buf.as_mut(), 1.0).unwrap().unwrap();
    assert_eq!(&buf.as_ref()[pad..pad + NX], &want(fp, 0, NX)[..], "no mirror, split 1.0: primary bytes");
    std::fs::remove_dir_all(&base).unwrap();
}
