//! Expert-mirror acceptance (`v4flash_core::mirror_check`, `SafetensorsDir::open_with_mirror`) on tiny synthetic
//! checkpoints (no model needed): a mirror is used for a shard only when it is a COPY of the primary's -- the 10-17
//! model swap (DeepSeek-V4.1-Flash-1017: same names, same headers, same shard sizes, different bytes) must be refused,
//! as must a short, missing or header-different shard and a dir whose HF identity / config differs. Files go under
//! CARGO_TARGET_TMPDIR (a real filesystem: the mirror handles are O_DIRECT).
use std::path::{Path, PathBuf};

use v4flash_core::mirror_check::{self, check_dirs, check_shard, Verdict};
use v4flash_core::safetensors::SafetensorsDir;

/// Data bytes per shard: two U8 tensors, > SAMPLES x SAMPLE_BYTES so the windows are spread.
const NA: usize = 700_000;
const NB: usize = 400_123;

fn shard_bytes(seed: u32) -> Vec<u8> {
    let mut header = format!(r#"{{"a":{{"dtype":"U8","shape":[{NA}],"data_offsets":[0,{NA}]}},"b":{{"dtype":"U8","shape":[{NB}],"data_offsets":[{NA},{}]}}}}"#, NA + NB).into_bytes();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut out = (header.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&header);
    out.extend((0..(NA + NB) as u32).map(|i| (i.wrapping_add(seed).wrapping_mul(2_654_435_761) >> 13) as u8));
    out
}

/// A checkpoint dir with shards `names` (bytes from `seed`), their index and a config.
fn write_dir(dir: &Path, names: &[&str], seed: u32) {
    std::fs::create_dir_all(dir).unwrap();
    for (i, n) in names.iter().enumerate() {
        std::fs::write(dir.join(n), shard_bytes(seed + i as u32 * 1000)).unwrap();
    }
    let map: Vec<String> = names.iter().enumerate().map(|(i, n)| format!(r#""s{i}.a":"{n}""#)).collect();
    std::fs::write(dir.join("model.safetensors.index.json"), format!(r#"{{"weight_map":{{{}}}}}"#, map.join(","))).unwrap();
    std::fs::write(dir.join("config.json"), r#"{"model_type":"fake"}"#).unwrap();
}

fn base(tag: &str) -> PathBuf {
    let b = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(format!("mirror-verify-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&b);
    b
}

/// Tensor-name prefix of shard `k` in `write_ckpt`: shards 0 and 1 hold routed experts, shard 2 is expert-free (the
/// role of the hub-only Engram shards).
fn pre(k: usize) -> String {
    if k < 2 { format!("layers.{k}.ffn.experts.0.") } else { format!("engram.{k}.") }
}

/// `shard_bytes` with its tensors renamed `{pre(k)}a` / `{pre(k)}b`: `SafetensorsDir` refuses a tensor name defined by
/// two shards, so a multi-shard checkpoint needs distinct names.
fn shard_bytes_named(seed: u32, k: usize) -> Vec<u8> {
    let raw = shard_bytes(seed);
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let h = String::from_utf8(raw[8..8 + n].to_vec()).unwrap();
    let h2 = h.replacen(r#""a":"#, &format!(r#""{}a":"#, pre(k)), 1).replacen(r#""b":"#, &format!(r#""{}b":"#, pre(k)), 1);
    let mut h2 = h2.trim_end().as_bytes().to_vec();
    while h2.len() % 8 != 0 {
        h2.push(b' ');
    }
    let mut out = (h2.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h2);
    out.extend_from_slice(&raw[8 + n..]);
    out
}

fn write_ckpt(dir: &Path, names: &[&str], seed: u32) {
    write_dir(dir, names, seed);
    for (k, n) in names.iter().enumerate() {
        std::fs::write(dir.join(n), shard_bytes_named(seed + k as u32 * 1000, k)).unwrap();
    }
}

#[test]
fn shard_check_accepts_a_copy_and_rejects_everything_else() {
    let b = base("shard");
    let (p, m) = (b.join("p"), b.join("m"));
    write_dir(&p, &["x.safetensors"], 1);
    write_dir(&m, &["x.safetensors"], 1);
    let (ps, ms) = (p.join("x.safetensors"), m.join("x.safetensors"));
    assert_eq!(check_shard(&ps, &ms), Verdict::Same, "identical copy");

    // The 1017 swap: same size, same header, different data.
    std::fs::write(&ms, shard_bytes(77)).unwrap();
    assert_eq!(std::fs::metadata(&ps).unwrap().len(), std::fs::metadata(&ms).unwrap().len());
    let v = check_shard(&ps, &ms);
    assert!(matches!(&v, Verdict::Differs(w) if w.contains("data differs")), "same size + header, other bytes: {v:?}");

    // One flipped byte inside a sampled window (the middle one) is seen.
    let mut one = shard_bytes(1);
    let n = u64::from_le_bytes(one[..8].try_into().unwrap());
    let wins = mirror_check::sample_windows(8 + n, one.len() as u64);
    assert_eq!(wins.len(), mirror_check::SAMPLES);
    let at = wins[wins.len() / 2] as usize + 4321;
    one[at] ^= 0x5a;
    std::fs::write(&ms, &one).unwrap();
    assert!(matches!(check_shard(&ps, &ms), Verdict::Differs(w) if w.contains(&format!("at byte {at}"))));
    // ... and so is the very last byte (the last window ends at EOF).
    let mut last = shard_bytes(1);
    *last.last_mut().unwrap() ^= 1;
    std::fs::write(&ms, &last).unwrap();
    assert!(matches!(check_shard(&ps, &ms), Verdict::Differs(_)), "last byte");

    // Short (half-copied).
    let full = shard_bytes(1);
    std::fs::write(&ms, &full[..full.len() / 2]).unwrap();
    assert!(matches!(check_shard(&ps, &ms), Verdict::Differs(w) if w.contains("size")));
    // Shorter than a header length.
    std::fs::write(&ms, [1u8, 2, 3]).unwrap();
    assert!(matches!(check_shard(&ps, &ms), Verdict::Differs(_)));
    // Same size, header differs (a dtype/name byte), data identical.
    let mut hd = shard_bytes(1);
    let i = hd.windows(4).position(|w| w == b"\"U8\"").unwrap();
    hd[i + 1] = b'I';
    std::fs::write(&ms, &hd).unwrap();
    assert!(matches!(check_shard(&ps, &ms), Verdict::Differs(w) if w.contains("header")));
    // Missing.
    std::fs::remove_file(&ms).unwrap();
    assert_eq!(check_shard(&ps, &ms), Verdict::Missing);
    std::fs::remove_dir_all(&b).unwrap();
}

#[test]
fn dir_check_refuses_another_checkpoint_identity() {
    let b = base("dirs");
    let (p, m) = (b.join("p"), b.join("m"));
    write_dir(&p, &["x.safetensors"], 1);
    write_dir(&m, &["x.safetensors"], 1);
    assert!(check_dirs(&p, &m).is_ok());
    // Identity known on one side only: content decides.
    std::fs::write(p.join(".hf_fetch_rev"), "deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e10\n").unwrap();
    assert!(check_dirs(&p, &m).is_ok());
    // Both known and different.
    std::fs::write(m.join(".hf_fetch_rev"), "deepseek-ai/DeepSeek-V4.1-Flash dba1be0a\n").unwrap();
    assert!(check_dirs(&p, &m).is_err_and(|w| w.contains("HF identity")));
    std::fs::write(m.join(".hf_fetch_rev"), "deepseek-ai/DeepSeek-V4.1-Flash-1017 1bae0e10\n").unwrap();
    assert!(check_dirs(&p, &m).is_ok());
    // HF cache snapshot path on the mirror side vs a fetch-rev primary.
    let snap = b.join("hf/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a");
    write_dir(&snap, &["x.safetensors"], 1);
    assert!(check_dirs(&p, &snap).is_err_and(|w| w.contains("DeepSeek-V4.1-Flash@dba1be0a")));
    // config.json / index differ.
    std::fs::write(m.join("config.json"), r#"{"model_type":"other"}"#).unwrap();
    assert!(check_dirs(&p, &m).is_err_and(|w| w.contains("config.json")));
    std::fs::write(m.join("config.json"), r#"{"model_type":"fake"}"#).unwrap();
    std::fs::write(m.join("model.safetensors.index.json"), r#"{"weight_map":{}}"#).unwrap();
    assert!(check_dirs(&p, &m).is_err_and(|w| w.contains("index")));
    std::fs::remove_dir_all(&b).unwrap();
}

/// The only test here that opens a `SafetensorsDir` (the counters are process-global).
#[test]
fn open_uses_only_verified_mirror_shards_and_counts_the_rest() {
    let b = base("open");
    let (p, m) = (b.join("p"), b.join("m"));
    let names = ["s-00001.safetensors", "s-00002.safetensors", "s-00003.safetensors"];
    write_ckpt(&p, &names, 1);
    write_ckpt(&m, &names, 1);
    let probe = {
        use std::os::unix::fs::OpenOptionsExt;
        std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(p.join(names[0])).is_ok()
    };
    if !probe {
        eprintln!("SKIP: no O_DIRECT under {}", b.display());
        let _ = std::fs::remove_dir_all(&b);
        return;
    }
    let c0 = mirror_check::counters();
    // All three copies: every shard mirrored, hedge handles available.
    let st = SafetensorsDir::open_with_mirror(&p, Some(m.clone())).expect("open");
    assert!(st.mirror_complete());
    assert!((0..3).all(|s| st.hedge_handles(s).unwrap().is_some()));
    let c1 = mirror_check::counters();
    assert_eq!((c1.0 - c0.0, c1.1 - c0.1, c1.2 - c0.2), (3, 0, 0));

    let shard_of = |st: &SafetensorsDir, k: usize| st.get(&format!("{}a", pre(k))).unwrap().shard;

    // PARTIAL mirror (box 2's /weights2 after the 1017 swap: no Engram shards): the expert-free shard is missing, the
    // expert shards stay mirrored, and mirror-only expert reads (`route=urgency`) are still allowed.
    std::fs::remove_file(m.join(names[2])).unwrap();
    let st = SafetensorsDir::open_with_mirror(&p, Some(m.clone())).expect("open");
    assert!(st.has_mirror(shard_of(&st, 0)) && st.has_mirror(shard_of(&st, 1)));
    assert!(!st.has_mirror(shard_of(&st, 2)) && st.hedge_handles(shard_of(&st, 2)).unwrap().is_none());
    assert!(!st.mirror_complete() && st.mirror_covers_experts());
    // Reads of the missing shard fall back to the primary on every mirror route (no error, primary bytes).
    {
        let t = st.get(&format!("{}a", pre(2))).unwrap().clone();
        let n = 300_001usize;
        let mut want = vec![0u8; n];
        assert!(st.read_range_into_direct(&t, 12_345, &mut want).unwrap());
        let mut got = vec![0xAAu8; n];
        assert!(st.read_range_into_direct_mirrored(&t, 12_345, &mut got, 1.0).unwrap());
        assert!(got == want, "mirrored read of a missing mirror shard");
        let mut got = vec![0xAAu8; n];
        st.read_range_into_cached_hedged(&t, 12_345, &mut got).unwrap();
        assert!(got == want, "hedged read of a missing mirror shard");
    }
    let c2 = mirror_check::counters();
    assert_eq!((c2.0 - c1.0, c2.1 - c1.1, c2.2 - c1.2), (0, 0, 1), "shards 1-2 memoized; 3 missing");
    // A re-open does not recount or relog (memoized while the files are unchanged).
    let _ = SafetensorsDir::open_with_mirror(&p, Some(m.clone())).expect("open");
    assert_eq!(mirror_check::counters(), c2);

    // Expert shard 2 replaced by "the other model" (same size + same header, other bytes).
    std::fs::write(m.join(names[1]), {
        let mut v = shard_bytes_named(1 + 1000, 1);
        let n = u64::from_le_bytes(v[..8].try_into().unwrap()) as usize;
        v[8 + n..].iter_mut().for_each(|x| *x = x.wrapping_add(1));
        v
    })
    .unwrap();
    assert_eq!(std::fs::metadata(m.join(names[1])).unwrap().len(), std::fs::metadata(p.join(names[1])).unwrap().len());
    let st = SafetensorsDir::open_with_mirror(&p, Some(m.clone())).expect("open");
    assert!(st.has_mirror(shard_of(&st, 0)), "the intact copy stays mirrored");
    assert!(!st.has_mirror(shard_of(&st, 1)), "a different checkpoint's shard must NOT be used");
    assert!(st.hedge_handles(shard_of(&st, 1)).unwrap().is_none(), "nor hedged on");
    assert!(!st.mirror_complete() && !st.mirror_covers_experts());
    let c2b = mirror_check::counters();
    assert_eq!((c2b.0 - c2.0, c2b.1 - c2.1, c2b.2 - c2.2), (0, 1, 0), "shard 2 mismatched, the rest memoized");
    let c2 = c2b;

    // Whole-dir refusal: another HF identity -> no shard is mirrored, all counted as mismatched.
    let m2 = b.join("m2");
    write_ckpt(&m2, &names, 1);
    std::fs::write(p.join(".hf_fetch_rev"), "org/new rev2\n").unwrap();
    std::fs::write(m2.join(".hf_fetch_rev"), "org/old rev1\n").unwrap();
    let st = SafetensorsDir::open_with_mirror(&p, Some(m2.clone())).expect("open");
    assert!((0..3).all(|s| !st.has_mirror(s)));
    let c3 = mirror_check::counters();
    assert_eq!((c3.0 - c2.0, c3.1 - c2.1, c3.2 - c2.2), (0, 3, 0));

    // No mirror at all.
    let st = SafetensorsDir::open_with_mirror(&p, None).expect("open");
    assert!((0..3).all(|s| !st.has_mirror(s) && st.hedge_handles(s).unwrap().is_none()));
    std::fs::remove_dir_all(&b).unwrap();
}

/// Real checkpoints, READ-ONLY (headers + a few 64 KiB O_DIRECT samples per shard): the verdict for every shard of
/// `MIRROR_CHECK_PRIMARY` against `MIRROR_CHECK_MIRROR`, and the time it took. E.g. the live pairing
/// (/weights2/dsv4.1f vs the HF snapshot: expect all Same) and the 1017 swap (/weights2/dsv4.1f-1017 vs either old
/// copy: expect every shard refused).
#[test]
#[ignore]
fn real_dirs_report() {
    let (p, m) = (PathBuf::from(std::env::var("MIRROR_CHECK_PRIMARY").unwrap()), PathBuf::from(std::env::var("MIRROR_CHECK_MIRROR").unwrap()));
    println!("dirs: {:?} (identity primary {:?}, mirror {:?})", check_dirs(&p, &m), mirror_check::dir_identity(&p), mirror_check::dir_identity(&m));
    let idx: serde_json::Value = serde_json::from_slice(&std::fs::read(p.join("model.safetensors.index.json")).unwrap()).unwrap();
    let mut names: Vec<String> = idx["weight_map"].as_object().unwrap().values().map(|v| v.as_str().unwrap().to_owned()).collect();
    names.sort();
    names.dedup();
    let t0 = std::time::Instant::now();
    let mut tally = std::collections::BTreeMap::new();
    for n in &names {
        let v = check_shard(&p.join(n), &m.join(n));
        let k = match &v {
            Verdict::Same => "same".to_string(),
            Verdict::Missing => "missing".into(),
            Verdict::Differs(w) => w.split(" (").next().unwrap_or(w).split(" at byte").next().unwrap_or(w).to_string(),
        };
        println!("{n}: {v:?}");
        *tally.entry(k).or_insert(0) += 1;
    }
    println!("{} shards in {:?}: {tally:?}", names.len(), t0.elapsed());
}
