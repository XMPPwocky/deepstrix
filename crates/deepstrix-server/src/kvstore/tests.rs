//! Store-level tests over real files in a temp dir (synthetic tokens and
//! payloads; no GPU), and the M1 measurements (`#[ignore]`).
//!
//! Set TMPDIR to a disk-backed directory when running these: chunk files are
//! 2.83 MB and tails up to ~5.7 MB, the real sizes.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::format::{self, BuildId, TailKind, TailOrigin};
use super::index::{JobId, Why};
use super::io::tests::unique_dir;
use super::keys::{ChainCursor, ImageRecord, Key, NamespaceInputs};
use super::*;

const BUILD_A: &str = "aaaaaaa0aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const BUILD_B: &str = "bbbbbbb0bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn ns_inputs(tag: u8) -> NamespaceInputs {
    NamespaceInputs::v41([tag; 32], [0; 32], [1; 32], true, true)
}

fn config(root: &Path, cap: u64, build: &str) -> StoreConfig {
    StoreConfig {
        root: root.to_path_buf(),
        cap_bytes: cap,
        write_queue_bytes: 512 << 20,
        write_wait: Duration::from_millis(200),
        purge: Vec::new(),
        build: BuildId::parse(build).unwrap(),
        knob_hash: [7; 16],
        check_invariants: true,
    }
}

fn conv(seed: u64, len: usize) -> Vec<i32> {
    let mut r = StdRng::seed_from_u64(seed);
    (0..len).map(|_| r.gen_range(0..128_000)).collect()
}

/// Two images: one straddling the chunk 0/1 boundary, one starting in the
/// open part of chunk 1.
fn imgs(seed: u8) -> Vec<ImageRecord> {
    vec![ImageRecord { start: 1000, len: 100, hash: [seed; 32] }, ImageRecord { start: 1500, len: 40, hash: [seed ^ 0x5a; 32] }]
}

/// Deterministic bytes standing in for a capture: a function of what is
/// captured (the ids it covers), so a reread can be checked exactly.
fn fake_bytes(len: u64, ids: &[i32], salt: u8) -> Vec<u8> {
    let mut h = blake3::Hasher::new();
    h.update(&[salt]);
    for t in ids {
        h.update(&t.to_le_bytes());
    }
    let mut out = vec![0u8; len as usize];
    h.finalize_xof().fill(&mut out);
    out
}

fn sec_e_for(tokens: &[i32], t: u32) -> Vec<u8> {
    let n_raw = t.min(128);
    fake_bytes(format::section_e_len(&format::v41_stores(), t, n_raw), &tokens[..t as usize], 1)
}

fn sec_d_for(tokens: &[i32], t: u32) -> Vec<u8> {
    fake_bytes(64 * 1024, &tokens[..t as usize], 2)
}

fn chunk_payload(tokens: &[i32], k: u32) -> Vec<u8> {
    fake_bytes(format::chunk_payload_len(&format::v41_stores()), &tokens[(k * C) as usize..((k + 1) * C) as usize], 3)
}

fn chunk_req<'a>(s: &mut Store, tokens: &'a [i32], images: &'a [ImageRecord], k: u32, job: Option<JobId>) -> ChunkWriteReq<'a> {
    let mut payload = s.buffer();
    payload.extend_from_slice(&chunk_payload(tokens, k));
    ChunkWriteReq { tokens, images, k, payload, provenance: format::Provenance::Prefill, job, knob_hash: None }
}

fn tail_req<'a>(tokens: &'a [i32], images: &'a [ImageRecord], t: u32, kind: TailKind, origin: TailOrigin, job: Option<JobId>) -> TailWriteReq<'a> {
    let full = kind == TailKind::Full;
    TailWriteReq {
        tokens,
        images,
        t,
        kind,
        origin,
        n_raw: t.min(128),
        n_raw_dec: if full { t.min(128) } else { 0 },
        drafter: if full { [5; 32] } else { [0; 32] },
        session_id: Some("sess"),
        sec_e: sec_e_for(tokens, t),
        sec_d: if full { sec_d_for(tokens, t) } else { Vec::new() },
        job,
        knob_hash: None,
    }
}

/// How an admission runs.
#[derive(Clone, Copy)]
struct Adm {
    /// The job fails: no prompt-end tail, ended as failed.
    fail: bool,
    /// Flush the IO queue after the job ended (else completions land later).
    flush: bool,
    /// Also write an anchor at the deepest K multiple ≤ L.
    anchor: bool,
}

const SYNC: Adm = Adm { fail: false, flush: true, anchor: false };

/// One admission through the store, as M2's job will run it: walk, select,
/// verify + touch the plan (a failing file is evicted and selection re-runs,
/// 6.2), write the missing chunks, waypoints (at L too when L is a K
/// multiple: then the prompt end has the same key) and the prompt-end tail,
/// then END THE JOB IN THE SAME TICK, before any completion is processed.
/// Returns the restored t.
fn admit_opts(s: &mut Store, tokens: &[i32], images: &[ImageRecord], now: u64, job: u64, o: Adm) -> u32 {
    let l = tokens.len() as u32;
    let job = JobId(job);
    let (walk, plan) = loop {
        let (walk, _) = s.walk(tokens, images, now);
        let Some(p) = Store::select(&walk, l, false) else { break (walk, None) };
        let pin = s.pin_plan(&p.key, Some(job)).unwrap();
        let mut ok = true;
        for (k, key) in walk.chunks.iter().enumerate().take((p.t / C) as usize) {
            let k = k as u32;
            match s.read_chunk(key, &tokens[(k * C) as usize..((k + 1) * C) as usize], now) {
                Ok(f) => assert_eq!(f.payload, chunk_payload(tokens, k)),
                Err(StoreError::Corrupt { .. }) => ok = false,
                Err(e) => panic!("restore read: {e}"),
            }
        }
        let a = p.t / C * C;
        let want_d = l - p.t <= 128;
        match s.read_tail(&p.key, &tokens[a as usize..p.t as usize], want_d, now) {
            Ok(tf) if ok => {
                assert_eq!(tf.sec_e, sec_e_for(tokens, p.t));
                if want_d {
                    assert_eq!(tf.sec_d.unwrap(), sec_d_for(tokens, p.t));
                }
            }
            Ok(_) | Err(StoreError::Corrupt { .. }) | Err(StoreError::Missing) => ok = false,
            Err(e) => panic!("restore read: {e}"),
        }
        s.unpin(pin, now);
        if ok {
            s.touch(&p.key, &walk.ancestors_below(p.t), now);
            break (walk, Some(p));
        }
    };
    let pos0 = plan.map_or(0, |p| p.t);
    let mut cur: ChainCursor = s.chain().cursor();
    cur.seed(&walk.chunks);
    let mut budget = s.tick_budget();
    for k in pos0 / C..l / C {
        let req = chunk_req(s, tokens, images, k, Some(job));
        s.write_chunk(&mut cur, req, &mut budget, now).unwrap();
        let end = (k + 1) * C;
        if end % K == 0 && end > pos0 && end <= l {
            let req = tail_req(tokens, images, end, TailKind::Enc, TailOrigin::Waypoint, Some(job));
            s.write_tail(&mut cur, req, &mut budget, now).unwrap();
        }
    }
    if o.anchor && l >= K {
        let req = tail_req(tokens, images, l / K * K, TailKind::Full, TailOrigin::Anchor, Some(job));
        s.write_tail(&mut cur, req, &mut budget, now).unwrap();
    }
    if !o.fail {
        let req = tail_req(tokens, images, l, TailKind::Full, TailOrigin::PromptEnd, Some(job));
        s.write_tail(&mut cur, req, &mut budget, now).unwrap();
    }
    let _ = s.job_finished(job, o.fail, now);
    if o.flush {
        let _ = s.flush(false, now);
        s.check_invariants().unwrap();
        assert!(s.index().finished(job).is_none(), "a drained job is forgotten");
    }
    pos0
}

fn admit(s: &mut Store, tokens: &[i32], now: u64, job: u64) -> u32 {
    admit_opts(s, tokens, &[], now, job, SYNC)
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in fs::read_dir(&d).unwrap().flatten() {
            if e.file_type().unwrap().is_dir() {
                stack.push(e.path());
            } else {
                out.push(e.path());
            }
        }
    }
    out
}

fn store_files(s: &Store) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = files_under(&s.dirs().base).into_iter().filter(|p| p.extension().is_some_and(|e| e == "kvc" || e == "kvt")).collect();
    v.sort();
    v
}

fn store_bytes_on_disk(s: &Store) -> u64 {
    store_files(s).iter().map(|p| fs::metadata(p).unwrap().len()).sum()
}

fn store_bytes_on_disk_dir(dir: &Path) -> u64 {
    files_under(dir).iter().filter(|p| p.extension().is_some_and(|e| e == "kvc" || e == "kvt")).map(|p| fs::metadata(p).unwrap().len()).sum()
}

fn assert_disk_is_index(s: &Store) {
    assert_eq!(store_bytes_on_disk(s), s.index().chunk_bytes() + s.index().tail_bytes(), "disk != index");
    assert_eq!(store_files(s).len(), s.index().n_chunks() + s.index().n_tails(), "file count != index");
}

/// Read and verify every indexed file; evict the ones that fail (a test
/// stand-in for the reads restores do). Returns how many failed.
fn scrub(s: &mut Store, now: u64) -> usize {
    let ns = *s.chain().ns();
    let mut chunks: Vec<Key> = s.index().chunk_keys().copied().collect();
    let mut tails: Vec<Key> = s.index().tail_keys().copied().collect();
    chunks.sort();
    tails.sort();
    let mut bad = 0;
    for k in chunks {
        if s.index().chunk(&k).is_some() && format::read_chunk(&s.dirs().chunk_path(&k), &ns).is_err() {
            s.evict_corrupt(&k, EntryKind::Chunk, now);
            bad += 1;
        }
    }
    for k in tails {
        let want_d = s.index().tail(&k).is_some_and(|e| e.kind == TailKind::Full);
        if s.index().tail(&k).is_some() && format::read_tail(&s.dirs().tail_path(&k), &ns, want_d).is_err() {
            s.evict_corrupt(&k, EntryKind::Tail, now);
            bad += 1;
        }
    }
    bad
}

type ChunkSnap = (Key, u32, u64, u32);
type TailSnap = (Key, u32, TailKind, bool, bool, u32, u64, u64);

/// What a restart must preserve, per entry. `path_last_used` is returned
/// apart: it is compared as "rebuilt ≤ live", because live values only grow
/// while a rebuild sees only the tails still below (evicted or thinned
/// descendants no longer count). Detached chunks (their parent's write was
/// dropped) are live-only by design: the scan drops them as unreachable.
fn snapshot(s: &Store) -> (Vec<ChunkSnap>, Vec<TailSnap>, Vec<(Key, u64)>) {
    let idx = s.index();
    let mut c: Vec<_> = idx
        .chunk_keys()
        .filter(|k| idx.reachable(k))
        .map(|k| {
            let e = idx.chunk(k).unwrap();
            (*k, e.k, e.bytes, e.refs)
        })
        .collect();
    let mut t: Vec<_> = idx
        .tail_keys()
        .map(|k| {
            let e = idx.tail(k).unwrap();
            (*k, e.t, e.kind, e.anchor, e.demoted, e.hits, e.bytes, e.last_used)
        })
        .collect();
    let mut p: Vec<_> = idx.tail_keys().map(|k| (*k, idx.tail(k).unwrap().path_last_used)).collect();
    c.sort();
    t.sort();
    p.sort();
    (c, t, p)
}

/// Permissions bind (not root): EACCES tests are meaningful.
fn perms_bind(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(p, fs::Permissions::from_mode(0o000)).unwrap();
    let bind = fs::read(p).is_err();
    fs::set_permissions(p, fs::Permissions::from_mode(0o600)).unwrap();
    bind
}

fn chmod(p: &Path, mode: u32) {
    use std::os::unix::fs::PermissionsExt;
    fs::set_permissions(p, fs::Permissions::from_mode(mode)).unwrap();
}

#[test]
fn roundtrip_restore_and_reopen() {
    let root = unique_dir("st-roundtrip");
    let ns = ns_inputs(1);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(1, 12_000);
    let im = imgs(1);
    assert_eq!(admit_opts(&mut s, &a[..10_000], &im, 1000, 1, SYNC), 0, "cold");
    assert_eq!((s.index().n_chunks(), s.index().n_tails()), (9, 2), "9 chunks, a waypoint and the prompt end");
    // The next turn restores the prompt end, reading and verifying every byte.
    assert_eq!(admit_opts(&mut s, &a[..10_500], &im, 2000, 2, SYNC), 10_000);
    // A t ≤ 128 continuation reads section D too.
    assert_eq!(admit_opts(&mut s, &a[..10_600], &im, 3000, 3, SYNC), 10_500);
    // Other pixels: nothing past the first IMAGE_START is shared.
    let (w, _) = s.walk(&a[..10_600], &imgs(2), 3000);
    assert!(w.chunks.is_empty() && w.tails.is_empty());
    // Disk equals the index.
    let _ = s.flush(true, 3000);
    assert_disk_is_index(&s);
    let before = snapshot(&s);
    // The hourly byte total goes through the IO thread (2.).
    let bytes_file = s.dirs().base.join(scan::NS_BYTES);
    let _ = fs::remove_file(&bytes_file);
    let _ = s.tick(3000 + INVARIANT_CHECK_EVERY_S);
    let _ = s.flush(false, 3000);
    let persisted: u64 = fs::read_to_string(&bytes_file).unwrap().trim().parse().unwrap();
    assert_eq!(persisted, s.index().chunk_bytes() + s.index().tail_bytes());
    let stats = s.stats();
    assert!(stats.log_line().starts_with("kv.store chunks=10 tails=4"), "{}", stats.log_line());
    s.shutdown(Duration::from_secs(1), 3000);

    // Reopen: the index is rebuilt from the files alone.
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 4000).unwrap();
    s2.check_invariants().unwrap();
    let r = s2.scan_report();
    assert_eq!((r.chunks, r.tails, r.invalid, r.unreachable, r.missing_ancestor), (10, 4, 0, 0, 0));
    assert!(!r.gen_new);
    assert_eq!(snapshot(&s2), before, "kinds, hits, last_used and path_last_used survive a restart");
    assert_eq!(s2.stats().chunk_bytes, stats.chunk_bytes);
}

#[test]
fn path_last_used_is_recomputed_from_leaves() {
    // 8.3: propagated touches are memory-only; at startup path_last_used comes
    // from the leaves' mtimes and must equal what the live index had.
    let root = unique_dir("st-plu");
    let ns = ns_inputs(2);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let y = conv(2, 9_600);
    admit(&mut s, &y[..9_100], 2000, 1); // waypoint 8192 + Y at 9100 (unaligned)
    admit(&mut s, &y[..9_500], 4000, 2); // X below Y, at node 9
    let mut w = y[..9_600].to_vec();
    w[9_000] ^= 1; // diverges inside Y's open part: not below Y
    admit(&mut s, &w, 8000, 3);
    let snap = |s: &Store| -> Vec<(u32, u64, u64)> {
        let mut v: Vec<_> = s
            .index()
            .tail_keys()
            .map(|k| {
                let e = s.index().tail(k).unwrap();
                (e.t, e.last_used, e.path_last_used)
            })
            .collect();
        v.sort();
        v
    };
    let live = snap(&s);
    let at = |v: &[(u32, u64, u64)], t: u32| v.iter().find(|x| x.0 == t).copied().unwrap();
    assert_eq!(at(&live, 8192).2, 8000, "the shared waypoint is young through W");
    assert_eq!(at(&live, 9100).2, 4000, "Y is young through X only");
    s.shutdown(Duration::from_secs(1), 8000);
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 9000).unwrap();
    assert_eq!(snap(&s2), live);
}

#[test]
fn torn_short_and_corrupt_files_read_as_a_miss() {
    // G5: a damaged file never reads as data. Byte flips anywhere are caught,
    // except in the tail's `hits` (deliberately outside the checksum), which
    // then changes only `hits`. The chunk carries an image record.
    let root = unique_dir("st-torn");
    let ns = ns_inputs(3);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(3, 2_100);
    let im = imgs(3);
    admit_opts(&mut s, &a, &im, 1000, 1, SYNC);
    let (walk, _) = s.walk(&a, &im, 1000);
    let ck = walk.chunks[1];
    let tk = walk.tails.last().unwrap().key;
    let cpath = s.dirs().chunk_path(&ck);
    let tpath = s.dirs().tail_path(&tk);
    let nsk = *s.chain().ns();
    let corig = fs::read(&cpath).unwrap();
    let torig = fs::read(&tpath).unwrap();
    let cgood = format::read_chunk(&cpath, &nsk).unwrap();
    assert_eq!(cgood.images.len(), 1, "chunk 1 carries the image starting at 1500");
    let tgood = format::read_tail(&tpath, &nsk, true).unwrap();
    let scratch = root.join("scratch.bin");
    let mut rng = StdRng::seed_from_u64(77);
    for i in 0..400 {
        let (orig, is_tail) = if i % 2 == 0 { (&corig, false) } else { (&torig, true) };
        let mut b = orig.clone();
        if i % 4 < 2 {
            b.truncate(rng.gen_range(0..b.len()));
        } else {
            let at = rng.gen_range(0..b.len());
            b[at] ^= 1 << rng.gen_range(0..8);
        }
        fs::write(&scratch, &b).unwrap();
        if is_tail {
            match format::read_tail(&scratch, &nsk, true) {
                Err(e) => assert!(e.evicts(), "case {i}: {e} would not evict"),
                Ok(t) => {
                    let off = format::TAIL_HITS_OFFSET as usize;
                    assert!(
                        b.len() == orig.len() && (0..b.len()).filter(|&j| b[j] != orig[j]).all(|j| (off..off + 4).contains(&j)),
                        "case {i}: damaged tail read as data"
                    );
                    assert_eq!(format::TailFile { header: format::TailHeader { hits: tgood.header.hits, ..t.header.clone() }, ..t }, tgood);
                }
            }
        } else {
            match format::read_chunk(&scratch, &nsk) {
                Err(e) => assert!(e.evicts(), "case {i}: {e} would not evict"),
                Ok(_) => panic!("case {i}: damaged chunk read as data"),
            }
        }
    }
    // Through the store: a corrupt chunk on read is evicted with every tail
    // beneath it (6.2), and the next walk stops before it.
    let mut b = corig.clone();
    let n = b.len();
    b[n - 10] ^= 0xff;
    fs::write(&cpath, &b).unwrap();
    let err = s.read_chunk(&ck, &a[C as usize..2 * C as usize], 1000).unwrap_err();
    assert!(matches!(err, StoreError::Corrupt { .. }), "{err}");
    assert_eq!(s.index().n_tails(), 0);
    assert_eq!(s.index().n_chunks(), 0, "chunk 0 cascaded with its only tail");
    assert!(s.walk(&a, &im, 1000).0.chunks.is_empty());
    assert_eq!(s.recent_event(&ck).map(|e| e.0), Some(RecentEvent::Removed(Why::Corrupt)));
    let _ = s.flush(true, 1000);
    assert!(!cpath.exists() && !tpath.exists(), "evicted files are unlinked");
    s.check_invariants().unwrap();
    // A token-id mismatch against the request is a data failure too.
    let c = conv(33, 1_500);
    admit(&mut s, &c, 1100, 2);
    let k0 = s.walk(&c, &[], 1100).0.chunks[0];
    let mut wrong = c[..C as usize].to_vec();
    wrong[5] ^= 1;
    assert!(matches!(s.read_chunk(&k0, &wrong, 1100), Err(StoreError::Corrupt { .. })));
    assert_eq!((s.index().n_chunks(), s.index().n_tails()), (0, 0));
    s.check_invariants().unwrap();
}

#[test]
fn store_level_shape_and_id_checks_evict() {
    // The checks the store adds on top of the format (9b): the chunk's store
    // shape, the tail's section-E shape and n_raw, the ids against the request.
    let root = unique_dir("st-shape");
    let ns = ns_inputs(4);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    for (i, seed) in [40u64, 41, 42].into_iter().enumerate() {
        let a = conv(seed, 2_100);
        admit(&mut s, &a, 1000, i as u64 + 1);
        let (w, _) = s.walk(&a, &[], 1000);
        let (k1, tk) = (w.chunks[1], w.tails.last().unwrap().key);
        match i {
            0 => {
                // Swap two store entries: same payload length, wrong shape.
                let p = s.dirs().chunk_path(&k1);
                let mut b = fs::read(&p).unwrap();
                let (s0, s3) = (116, 116 + 3 * 8);
                let tmp: Vec<u8> = b[s0..s0 + 8].to_vec();
                b.copy_within(s3..s3 + 8, s0);
                b[s3..s3 + 8].copy_from_slice(&tmp);
                format::reseal_chunk(&mut b);
                fs::write(&p, &b).unwrap();
                let e = s.read_chunk(&k1, &a[C as usize..2 * C as usize], 1000).unwrap_err();
                assert!(matches!(&e, StoreError::Corrupt { error: format::FormatError::BadField("store shape"), .. }), "{e}");
            }
            1 => {
                // n_raw ≠ min(t, 128).
                let p = s.dirs().tail_path(&tk);
                let mut b = fs::read(&p).unwrap();
                b[116..120].copy_from_slice(&127u32.to_le_bytes());
                format::reseal_tail(&mut b);
                fs::write(&p, &b).unwrap();
                let e = s.read_tail(&tk, &a[2 * C as usize..], false, 1000).unwrap_err();
                assert!(matches!(&e, StoreError::Corrupt { error: format::FormatError::BadField("section E shape"), .. }), "{e}");
            }
            _ => {
                let mut open = a[2 * C as usize..].to_vec();
                open[3] ^= 1;
                let e = s.read_tail(&tk, &open, false, 1000).unwrap_err();
                assert!(matches!(e, StoreError::Corrupt { .. }), "{e}");
            }
        }
        assert!(s.index().tail(&tk).is_none(), "case {i}: the tail was evicted");
        s.check_invariants().unwrap();
    }
}

#[test]
fn transient_io_errors_never_evict() {
    // 3: EACCES (or EIO, EMFILE...) says nothing about the file: the read
    // returns StoreError::Io and evicts nothing; the scan leaves the file and
    // everything that hangs off it on disk for the next startup.
    let root = unique_dir("st-eio");
    let ns = ns_inputs(5);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(50, 3_000);
    admit(&mut s, &a, 1000, 1);
    let (w, _) = s.walk(&a, &[], 1000);
    let c1 = w.chunks[1];
    let p = s.dirs().chunk_path(&c1);
    if !perms_bind(&p) {
        return; // root: permissions do not bind
    }
    chmod(&p, 0o000);
    let e = s.read_chunk(&c1, &a[C as usize..2 * C as usize], 1000).unwrap_err();
    assert!(matches!(e, StoreError::Io { .. }), "{e}");
    assert!(s.index().chunk(&c1).is_some() && s.index().n_tails() == 1, "nothing evicted");
    s.shutdown(Duration::from_secs(1), 1000);
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 2000).unwrap();
    let r = s2.scan_report().clone();
    assert_eq!((r.io_errors, r.invalid, r.missing_ancestor), (1, 0, 0), "{r:?}");
    assert!(r.deferred >= 1, "the tail above the unreadable chunk is deferred, not dropped: {r:?}");
    s2.shutdown(Duration::from_secs(1), 2000);
    assert!(p.exists());
    chmod(&p, 0o600);
    let s3 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 3000).unwrap();
    assert_eq!((s3.index().n_chunks(), s3.index().n_tails()), (2, 1), "all back once readable");
}

#[test]
fn startup_scan_repairs_and_drops() {
    let root = unique_dir("st-scan");
    let ns = ns_inputs(6);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(4, 6_000);
    let b = conv(5, 3_000);
    admit(&mut s, &a, 1000, 1);
    admit(&mut s, &b, 1000, 2);
    let ta = s.walk(&a, &[], 1000).0.tails.last().unwrap().key;
    let tb = s.walk(&b, &[], 1000).0.tails.last().unwrap().key;
    let (pa, pb) = (s.dirs().tail_path(&ta), s.dirs().tail_path(&tb));
    let nsk = *s.chain().ns();
    let tmp_left = s.dirs().tmp().join("deadbeef-1.kvc.tmp");
    s.shutdown(Duration::from_secs(1), 1000);
    let mtime_b = io::mtime_secs(&fs::metadata(&pb).unwrap());
    // (1) A crash between a demotion's truncate and its header rewrite.
    let ha = format::read_tail(&pa, &nsk, false).unwrap().header;
    fs::OpenOptions::new().write(true).open(&pa).unwrap().set_len(ha.e_end()).unwrap();
    // (2) A file longer than its sections.
    let mut f = fs::OpenOptions::new().append(true).open(&pb).unwrap();
    std::io::Write::write_all(&mut f, &[0u8; 100]).unwrap();
    f.set_modified(std::time::UNIX_EPOCH + Duration::from_secs(mtime_b)).unwrap();
    drop(f);
    // (3) A write in flight at the crash.
    fs::write(&tmp_left, b"partial").unwrap();
    // (4) A foreign file.
    fs::write(root.join(keys::ns16(&nsk)).join("chunks").join("README"), b"x").unwrap();
    let mut s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 2000).unwrap();
    let r = s2.scan_report().clone();
    assert_eq!((r.repaired_demotions, r.truncated, r.invalid), (1, 1, 0));
    let ea = s2.index().tail(&ta).unwrap();
    assert!(ea.kind == TailKind::Enc && ea.demoted && ea.bytes == ha.e_end());
    let _ = s2.flush(true, 2000);
    let fa = format::read_tail(&pa, &nsk, false).unwrap();
    assert!(fa.header.kind == TailKind::Enc && fa.header.demoted);
    assert!(format::read_tail(&pb, &nsk, true).is_ok());
    assert_eq!(io::mtime_secs(&fs::metadata(&pb).unwrap()), mtime_b, "the truncate repair keeps last_used");
    assert!(!tmp_left.exists() && fs::read_dir(s2.dirs().tmp()).unwrap().count() == 0);
    assert_eq!(fs::read_dir(root.join("trash")).unwrap().count(), 0, "trash drained in the background");
    s2.check_invariants().unwrap();
    // A torn chunk: dropped, and the tails above it with it.
    let c0 = s2.walk(&a, &[], 2000).0.chunks[0];
    let cpath = s2.dirs().chunk_path(&c0);
    s2.shutdown(Duration::from_secs(1), 2000);
    let len = fs::metadata(&cpath).unwrap().len();
    fs::OpenOptions::new().write(true).open(&cpath).unwrap().set_len(len / 2).unwrap();
    let mut s3 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 3000).unwrap();
    let r = s3.scan_report().clone();
    assert_eq!((r.invalid, r.unreachable, r.missing_ancestor), (1, 4, 1), "{r:?}");
    assert!(s3.walk(&a, &[], 3000).0.tails.is_empty());
    assert_eq!(s3.walk(&b, &[], 3000).0.tails.len(), 1, "the other conversation is untouched");
    let _ = s3.flush(true, 3000);
    assert_disk_is_index(&s3);
}

#[test]
fn scan_keeps_the_valid_copy_of_a_misplaced_duplicate() {
    // 14: the same key in two fan-out directories: the misplaced copy goes,
    // the valid one stays indexed.
    let root = unique_dir("st-dup");
    let ns = ns_inputs(7);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(6, 2_100);
    admit(&mut s, &a, 1000, 1);
    let c0 = s.walk(&a, &[], 1000).0.chunks[0];
    let good = s.dirs().chunk_path(&c0);
    let h = keys::hex(&c0);
    let wrong_fan = if &h[..2] == "00" { "01" } else { "00" };
    let dup = s.dirs().base.join("chunks").join(wrong_fan).join(format!("{h}.kvc"));
    s.shutdown(Duration::from_secs(1), 1000);
    fs::create_dir_all(dup.parent().unwrap()).unwrap();
    fs::copy(&good, &dup).unwrap();
    let mut s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 2000).unwrap();
    assert_eq!(s2.scan_report().invalid, 1);
    assert!(s2.index().chunk(&c0).is_some());
    let _ = s2.flush(true, 2000);
    assert!(good.exists() && !dup.exists());
    assert_disk_is_index(&s2);
}

#[test]
fn namespace_gc_keeps_active_and_previous_and_evicts_inactive_first() {
    let root = unique_dir("st-ns");
    let a = conv(6, 3_000);
    for (i, tag) in [10u8, 11, 12].iter().enumerate() {
        let now = 1000 + 100 * i as u64;
        let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(*tag), now).unwrap();
        admit(&mut s, &a, now, 1);
        s.shutdown(Duration::from_secs(1), now);
    }
    let ns_dir = |tag: u8| root.join(keys::ns16(&ns_inputs(tag).key()));
    let one_ns = store_bytes_on_disk_dir(&ns_dir(12));
    // The byte total was persisted at shutdown (24).
    assert_eq!(fs::read_to_string(ns_dir(12).join(scan::NS_BYTES)).unwrap().trim().parse::<u64>().unwrap(), one_ns);
    // Every open keeps the active namespace and the most recently active
    // other: opening 12 already trashed 10. Opening 13 keeps 12, trashes 11.
    assert!(!ns_dir(10).exists() && ns_dir(11).exists());
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(13), 2000).unwrap();
    let r = s.scan_report().clone();
    assert_eq!(r.trashed_namespaces, 1);
    assert_eq!(r.kept_inactive, Some((keys::ns16(&ns_inputs(12).key()), one_ns)));
    let _ = s.flush(true, 2000);
    assert!(!ns_dir(10).exists() && !ns_dir(11).exists() && ns_dir(12).exists());
    assert_eq!(fs::read_dir(root.join("trash")).unwrap().count(), 0);
    // A second process cannot open the same root (8).
    let e = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(13), 2000).err().unwrap();
    assert_eq!(e.kind(), std::io::ErrorKind::WouldBlock, "{e}");
    // One global cap: the inactive namespace is evicted first, whole.
    admit(&mut s, &a, 2100, 1);
    let mine = s.index().chunk_bytes() + s.index().tail_bytes();
    s.index.set_cap_bytes(mine + one_ns / 2);
    assert!(s.index.enforce_cap(2200));
    s.apply(2200);
    let _ = s.flush(true, 2200);
    assert!(!ns_dir(12).exists(), "the inactive namespace went first");
    assert_eq!(s.index().n_tails(), 1, "the active one is untouched");
}

#[test]
fn a_format_bump_is_a_new_namespace_not_an_unlink() {
    // 7: the old binary's files are kept as the inactive namespace, so a
    // rollback finds them; neither side unlinks the other's as "bad version".
    let root = unique_dir("st-fmt");
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(60), 1000).unwrap();
    admit(&mut s, &conv(7, 2_100), 1000, 1);
    s.shutdown(Duration::from_secs(1), 1000);
    let old = keys::ns16(&ns_inputs(60).key());
    let mut v2 = ns_inputs(60);
    v2.format += 1;
    assert_ne!(keys::ns16(&v2.key()), old);
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &v2, 2000).unwrap();
    assert_eq!(s2.scan_report().invalid, 0);
    assert_eq!(s2.scan_report().kept_inactive.as_ref().map(|k| k.0.clone()), Some(old.clone()));
    assert!(root.join(&old).exists());
}

#[test]
fn purge_by_build_cascades_through_shared_prefixes() {
    let root = unique_dir("st-purge");
    let ns = ns_inputs(20);
    let a = conv(7, 8_000);
    let other = conv(8, 2_100);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    admit(&mut s, &a[..3_000], 1000, 1);
    s.shutdown(Duration::from_secs(1), 1000);
    // A later build continues the conversation on top of build A's chunks.
    let mut s = Store::open(config(&root, 100 << 30, BUILD_B), &ns, 2000).unwrap();
    admit(&mut s, &a[..7_000], 2000, 1);
    admit(&mut s, &other, 2000, 2);
    s.shutdown(Duration::from_secs(1), 2000);
    // A purge that matches the RUNNING build is refused (17).
    let mut cfg = config(&root, 100 << 30, BUILD_B);
    cfg.purge = PurgeSpec::parse_list(&BUILD_B[..8]).unwrap();
    let s = Store::open(cfg, &ns, 2500).unwrap();
    assert!(s.scan_report().purge_refused && s.scan_report().purged == 0);
    s.shutdown(Duration::from_secs(1), 2500);
    let mut cfg = config(&root, 100 << 30, BUILD_B);
    cfg.purge = PurgeSpec::parse_list(&BUILD_A[..8]).unwrap();
    let mut s = Store::open(cfg, &ns, 3000).unwrap();
    let r = s.scan_report().clone();
    // Build A wrote chunks 0-1 and the tail at 3000; build B chunks 2-5 and
    // the tail at 7000 on top of them.
    assert_eq!((r.purged, r.unreachable, r.missing_ancestor), (3, 4, 1), "{r:?}");
    assert!(s.walk(&a, &[], 3000).0.tails.is_empty(), "everything beneath a purged chunk is gone");
    assert_eq!(s.walk(&other, &[], 3000).0.tails.len(), 1, "build B's own conversation stays");
    s.check_invariants().unwrap();
}

#[test]
fn eviction_under_cap_keeps_disk_equal_to_index() {
    let root = unique_dir("st-evict");
    let ns = ns_inputs(21);
    let cap = 60 << 20;
    let mut s = Store::open(config(&root, cap, BUILD_A), &ns, 1000).unwrap();
    for i in 0..8u64 {
        let p = conv(100 + i, 5_200);
        admit(&mut s, &p, 1000 + 100 * i, i + 1);
        assert!(s.index().total_bytes() <= cap);
    }
    let _ = s.flush(true, 2000);
    assert_disk_is_index(&s);
    assert!(s.stats().evicted_bytes > 0);
    // The newest conversations survive, the oldest went.
    let old = conv(100, 5_200);
    let (w, _) = s.walk(&old, &[], 2000);
    assert!(w.tails.is_empty());
    assert_eq!(s.walk(&conv(107, 5_200), &[], 2000).0.tails.len(), 1);
    let gone = s.chain().tail_key(&mut s.chain().cursor(), &old, &[], 5_200).1;
    assert_eq!(s.recent_event(&gone).map(|e| e.0), Some(RecentEvent::Removed(Why::Cap)), "12e: removals are queryable");
    s.check_invariants().unwrap();
}

#[test]
fn write_dedup_pending_and_full_replaces_enc() {
    let root = unique_dir("st-dedup");
    let ns = ns_inputs(22);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(9, 2_500);
    let mut b = s.tick_budget();
    let mut cur = s.chain().cursor();
    for (k, want) in [(0, WriteOutcome::Queued), (0, WriteOutcome::Pending), (1, WriteOutcome::Queued)] {
        let req = chunk_req(&mut s, &a, &[], k, None);
        assert_eq!(s.write_chunk(&mut cur, req, &mut b, 1000).unwrap(), want, "chunk {k}");
    }
    let _ = s.flush(false, 1000);
    let req = chunk_req(&mut s, &a, &[], 0, None);
    assert_eq!(s.write_chunk(&mut cur, req, &mut b, 1000).unwrap(), WriteOutcome::Stored);
    // The chunk's stored payload hash is kept for kv.dedup_mismatch (12f), and
    // its buffer came back to the pool (12d).
    let c0 = cur.chain(&a, &[], 1);
    assert_eq!(s.chunk_payload_hash(&c0), Some(*blake3::hash(&chunk_payload(&a, 0)).as_bytes()));
    assert!(s.buffer().capacity() >= format::chunk_payload_len(&format::v41_stores()) as usize);
    // An encoder tail and the full tail at the SAME key, back to back, no
    // flush between: both queued, applied in order.
    let t = 2_100;
    let enc = || tail_req(&a, &[], t, TailKind::Enc, TailOrigin::Waypoint, None);
    let full = || tail_req(&a, &[], t, TailKind::Full, TailOrigin::PromptEnd, None);
    assert_eq!(s.write_tail(&mut cur, enc(), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    assert_eq!(s.write_tail(&mut cur, full(), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    assert_eq!(s.write_tail(&mut cur, full(), &mut b, 1000).unwrap(), WriteOutcome::Pending);
    assert_eq!(s.write_tail(&mut cur, enc(), &mut b, 1000).unwrap(), WriteOutcome::Pending);
    let _ = s.flush(true, 1000);
    let key = s.chain().tail_key(&mut cur, &a, &[], t).1;
    let e = s.index().tail(&key).unwrap();
    assert_eq!(e.kind, TailKind::Full);
    assert_eq!(e.bytes, fs::metadata(s.dirs().tail_path(&key)).unwrap().len(), "the index holds the full file's size");
    assert!(s.read_tail(&key, &a[2 * C as usize..t as usize], true, 1000).is_ok());
    assert_eq!(s.write_tail(&mut cur, enc(), &mut b, 1300).unwrap(), WriteOutcome::Stored);
    // Shape checks refuse a malformed capture instead of storing it.
    let bad = TailWriteReq { sec_e: vec![0; 10], ..enc() };
    assert!(matches!(s.write_tail(&mut cur, bad, &mut b, 1300), Err(StoreError::Invalid(_))));
    let short = ChunkWriteReq { payload: vec![0; 5], ..chunk_req(&mut s, &a, &[], 1, None) };
    assert!(matches!(s.write_chunk(&mut cur, short, &mut b, 1300), Err(StoreError::Invalid(_))));
    assert_disk_is_index(&s);
    s.check_invariants().unwrap();
}

#[test]
fn same_key_writes_failures_and_late_unlinks() {
    // Blocker 1: two in-flight writes of one tail key, a failure of the
    // second, and evictions that race a write whose completion is not yet
    // processed. The index must always agree with the disk.
    let root = unique_dir("st-pend");
    let ns = ns_inputs(25);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(40, 2 * C as usize + 300);
    let mut cur = s.chain().cursor();
    let mut b = s.tick_budget();
    for k in 0..2 {
        let req = chunk_req(&mut s, &a, &[], k, None);
        s.write_chunk(&mut cur, req, &mut b, 1000).unwrap();
    }
    // A tail at 2C keeps chunks 0-1 referenced through what follows.
    let req = tail_req(&a, &[], 2 * C, TailKind::Enc, TailOrigin::Waypoint, None);
    s.write_tail(&mut cur, req, &mut b, 1000).unwrap();
    let _ = s.flush(false, 1000);
    let t = 2 * C + 100;
    let open = &a[2 * C as usize..t as usize];
    let key = s.chain().tail_key(&mut cur, &a, &[], t).1;
    let path = s.dirs().tail_path(&key);

    // (A) Enc then Full of the same key, no flush between; the Full fails.
    s.io.inject_write_faults(&[false, true]);
    assert_eq!(s.write_tail(&mut cur, tail_req(&a, &[], t, TailKind::Enc, TailOrigin::Cancel, None), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    assert_eq!(s.write_tail(&mut cur, tail_req(&a, &[], t, TailKind::Full, TailOrigin::PromptEnd, None), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    let _ = s.flush(true, 1000);
    let e = s.index().tail(&key).unwrap();
    assert_eq!(e.kind, TailKind::Enc, "the failed full write left the encoder tail");
    assert_eq!(e.bytes, fs::metadata(&path).unwrap().len());
    assert!(matches!(s.read_tail(&key, open, true, 1000), Err(StoreError::Invalid(_))), "no section D from an encoder tail");
    assert!(s.read_tail(&key, open, false, 1000).is_ok());
    assert_disk_is_index(&s);
    s.check_invariants().unwrap();

    // (B) The full write RUNS, then the tail is evicted before the completion
    // is processed: the unlink must be skipped, the completion indexes the
    // new file.
    assert_eq!(s.write_tail(&mut cur, tail_req(&a, &[], t, TailKind::Full, TailOrigin::PromptEnd, None), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    let done = s.io.flush();
    s.index.remove_tail(&key, Why::Cap, 1000);
    s.apply(1000);
    let ev = s.complete(done, 1000);
    assert!(ev.contains(&StoreEvent::TailStored(key)), "{ev:?}");
    let _ = s.flush(true, 1000);
    assert!(path.exists(), "the stale unlink did not delete the new file");
    assert_eq!(s.index().tail(&key).unwrap().kind, TailKind::Full);
    assert!(s.read_tail(&key, open, true, 1000).is_ok());
    assert_disk_is_index(&s);

    // (C) Same race, but the write FAILS: the old file, whose unlink was
    // skipped, is cleaned up once nothing is in flight.
    let t2 = 2 * C + 200;
    let key2 = s.chain().tail_key(&mut cur, &a, &[], t2).1;
    s.write_tail(&mut cur, tail_req(&a, &[], t2, TailKind::Enc, TailOrigin::Cancel, None), &mut b, 1000).unwrap();
    let _ = s.flush(false, 1000);
    s.io.inject_write_faults(&[true]);
    s.write_tail(&mut cur, tail_req(&a, &[], t2, TailKind::Full, TailOrigin::PromptEnd, None), &mut b, 1000).unwrap();
    let done = s.io.flush();
    s.index.remove_tail(&key2, Why::Cap, 1000);
    s.apply(1000);
    s.complete(done, 1000);
    let _ = s.flush(true, 1000);
    assert!(s.index().tail(&key2).is_none() && !s.dirs().tail_path(&key2).exists());
    assert_eq!(s.recent_event(&key2).map(|e| e.0), Some(RecentEvent::Dropped));
    assert_disk_is_index(&s);
    s.check_invariants().unwrap();
}

#[test]
fn late_completions_after_job_finished() {
    // Blocker 2: M2 ends a job in the tick that queues its last writes. Late
    // completions land unowned; no pin outlives the job.
    let root = unique_dir("st-late");
    let ns = ns_inputs(26);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    for (case, seed) in [(0, 70u64), (1, 71), (2, 72), (3, 73)] {
        let a = conv(seed, 5_000);
        let job = JobId(100 + case);
        let mut cur = s.chain().cursor();
        let mut b = s.tick_budget();
        if case == 3 {
            // A dropped chunk 1: the tail above it lands as broken_path.
            s.io.inject_write_faults(&[false, true]);
        }
        for k in 0..4 {
            let req = chunk_req(&mut s, &a, &[], k, Some(job));
            s.write_chunk(&mut cur, req, &mut b, 1000).unwrap();
        }
        let with_tail = matches!(case, 0 | 3);
        if with_tail {
            s.write_tail(&mut cur, tail_req(&a, &[], 5_000, TailKind::Full, TailOrigin::PromptEnd, Some(job)), &mut b, 1000).unwrap();
        }
        let failed = case == 2;
        let _ = s.job_finished(job, failed, 1000);
        assert_eq!(s.index().finished(job), Some(failed), "remembered while its writes are in flight");
        // The contract: nothing is written for a job after it ended.
        let late = chunk_req(&mut s, &a, &[], 3, Some(job));
        assert!(matches!(s.write_chunk(&mut cur, late, &mut b, 1000), Err(StoreError::Invalid(_))));
        s.check_invariants().unwrap();
        let ev = s.flush(true, 1000);
        let keys: Vec<Key> = (1..=4).map(|b| cur.chain(&a, &[], b)).collect();
        let stored = |k: &Key| ev.contains(&StoreEvent::ChunkStored(*k));
        let dropped = |k: &Key| ev.contains(&StoreEvent::ChunkDropped(*k));
        s.check_invariants().unwrap();
        assert_eq!(s.index().finished(job), None, "forgotten once drained");
        assert_eq!(s.index().job_pin_count(job), 0);
        let (w, _) = s.walk(&a, &[], 1000);
        match case {
            0 => assert_eq!((w.chunks.len(), w.tails.len()), (4, 1)),
            1 => assert_eq!(w.tails.len(), 0, "cancelled without a tail: orphans"),
            2 => {
                assert_eq!(w.chunks.len(), 0, "a failed job's late chunks are deleted at once");
                assert!(!s.dirs().chunk_path(&keys[0]).exists());
                // ... and reported dropped, never stored: a job subscribed on
                // Pending re-enqueues them (9.4).
                assert!(keys.iter().all(|k| dropped(k) && !stored(k)), "{ev:?}");
            }
            _ => {
                assert!(ev.iter().any(|e| matches!(e, StoreEvent::TailDropped { why: "broken_path", .. })), "{ev:?}");
                assert_eq!(w.chunks.len(), 1, "the walk stops at the dropped chunk");
                assert!(dropped(&keys[1]) && stored(&keys[0]) && stored(&keys[2]), "{ev:?}");
            }
        }
    }
    // A failed job whose chunks already LANDED: release_job deletes the
    // unreferenced ones, and job_finished reports each as dropped.
    let a = conv(74, 5_000);
    let job = JobId(200);
    let mut cur = s.chain().cursor();
    let mut b = s.tick_budget();
    for k in 0..4 {
        let req = chunk_req(&mut s, &a, &[], k, Some(job));
        s.write_chunk(&mut cur, req, &mut b, 1000).unwrap();
    }
    let landed = s.flush(false, 1000);
    assert_eq!(landed.iter().filter(|e| matches!(e, StoreEvent::ChunkStored(_))).count(), 4);
    let ev = s.job_finished(job, true, 1000);
    for kb in 1..=4 {
        assert!(ev.contains(&StoreEvent::ChunkDropped(cur.chain(&a, &[], kb))), "{ev:?}");
    }
    s.check_invariants().unwrap();
    // Everything not referenced is an orphan, unpinned: evictable first.
    s.index.set_cap_bytes(1);
    s.index.enforce_cap(2000);
    s.apply(2000);
    let _ = s.flush(true, 2000);
    assert_eq!((s.index().n_chunks(), s.index().n_tails()), (0, 0), "no pin kept anything alive");
    assert_disk_is_index(&s);
}

#[test]
fn dropped_writes_are_deterministic_with_a_held_worker() {
    // 9.4 / 10: hold the IO worker on a barrier; with room for one chunk, the
    // second chunk waits out its budget and is dropped; the tail above the
    // hole lands as broken_path and its file is removed.
    let root = unique_dir("st-drop");
    let ns = ns_inputs(23);
    let mut cfg = config(&root, 100 << 30, BUILD_A);
    cfg.write_queue_bytes = format::chunk_payload_len(&format::v41_stores()) + 256 + 4096;
    cfg.write_wait = Duration::from_millis(20);
    let mut s = Store::open(cfg, &ns, 1000).unwrap();
    let a = conv(10, 3_000);
    let mut cur = s.chain().cursor();
    let mut b = s.tick_budget();
    let (tx, rx) = mpsc::sync_channel(0);
    s.io.submit(IoJob::Barrier(tx));
    let r0 = chunk_req(&mut s, &a, &[], 0, None);
    assert_eq!(s.write_chunk(&mut cur, r0, &mut b, 1000).unwrap(), WriteOutcome::Queued);
    let r1 = chunk_req(&mut s, &a, &[], 1, None);
    assert_eq!(s.write_chunk(&mut cur, r1, &mut b, 1000).unwrap(), WriteOutcome::Dropped);
    let k1 = cur.chain(&a, &[], 2);
    assert_eq!(s.recent_event(&k1).map(|e| e.0), Some(RecentEvent::Dropped));
    rx.recv().unwrap();
    let _ = s.flush(false, 1000);
    assert_eq!(s.stats().writes_dropped, 1);
    let mut b = s.tick_budget();
    let req = tail_req(&a, &[], 2_500, TailKind::Enc, TailOrigin::Cancel, None);
    assert_eq!(s.write_tail(&mut cur, req, &mut b, 1000).unwrap(), WriteOutcome::Queued);
    let ev = s.flush(true, 1000);
    assert!(ev.iter().any(|e| matches!(e, StoreEvent::TailDropped { why: "broken_path", .. })), "{ev:?}");
    assert_eq!(s.index().n_tails(), 0);
    assert_disk_is_index(&s);
    s.check_invariants().unwrap();
}

#[test]
fn recent_log_forgets_the_least_recently_noted() {
    // 7.: a re-noted key moves to the back; it is not forgotten as old.
    let mut r = RecentLog::default();
    let key = |i: u64| -> Key {
        let mut k = [0u8; 32];
        k[..8].copy_from_slice(&i.to_le_bytes());
        k
    };
    r.note(key(0), RecentEvent::Dropped, 1);
    for i in 1..RECENT_EVENTS as u64 {
        r.note(key(i), RecentEvent::Demoted, 2);
    }
    r.note(key(0), RecentEvent::Removed(Why::Cap), 3); // fresh again
    r.note(key(RECENT_EVENTS as u64), RecentEvent::Demoted, 4); // over the cap by one
    assert_eq!(r.get(&key(0)), Some((RecentEvent::Removed(Why::Cap), 3)), "the fresh event survived");
    assert_eq!(r.get(&key(1)), None, "the least recently noted went");
    assert_eq!(r.map.len(), RECENT_EVENTS);
    for i in 0..5 * RECENT_EVENTS as u64 {
        r.note(key(i % 7), RecentEvent::Dropped, 5);
    }
    assert!(r.order.len() <= 2 * RECENT_EVENTS + 1, "stale order entries are compacted");
}

#[test]
fn mode_parse_defaults_off() {
    assert_eq!(Mode::parse(""), Some(Mode::Off));
    assert_eq!(Mode::parse("shadow"), Some(Mode::Shadow));
    assert_eq!(Mode::parse("on"), Some(Mode::On));
    assert_eq!(Mode::parse("bogus"), None);
    if std::env::var_os("V41_KV_STORE").is_none() {
        assert_eq!(Mode::from_env(), Mode::Off);
    }
}

#[test]
fn corrupt_mutation_evicts() {
    let root = unique_dir("st-mut");
    let ns = ns_inputs(24);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(11, 4_000);
    for (i, l) in [2_000usize, 2_500, 3_000].into_iter().enumerate() {
        if i == 2 {
            // Damage the oldest prompt end before the third prompt end demotes it.
            let k = s.walk(&a[..2_000], &[], 1000).0.tails.last().unwrap().key;
            fs::write(s.dirs().tail_path(&k), b"junk").unwrap();
        }
        admit(&mut s, &a[..l], 1000 + i as u64, i as u64 + 1);
    }
    let _ = s.flush(true, 1100);
    // The demotion found a bad header: the tail is evicted, not left half-done.
    assert_eq!(s.walk(&a[..2_000], &[], 1100).0.tails.iter().filter(|t| t.t == 2_000).count(), 0);
    s.check_invariants().unwrap();
}

/// Random admissions (new, continued, branched, retried, K-multiple lengths,
/// anchors), with ASYNCHRONOUS completions (jobs end before their writes
/// land; completions drained at random points), injected write failures,
/// cap pressure, damaged files and restarts, through real files and the IO
/// thread. Every step: invariants and the cap; whenever flushed, disk ==
/// index; after every restart, the rebuilt index equals the live one.
#[test]
fn randomized_store_with_restarts() {
    for seed in 1..=2u64 {
        let mut rng = StdRng::seed_from_u64(seed * 7919);
        let root = unique_dir("st-rand");
        let ns = ns_inputs(30 + seed as u8);
        let cap = 45 << 20;
        let mut now = 10_000u64;
        let mut s = Store::open(config(&root, cap, BUILD_A), &ns, now).unwrap();
        let bases: Vec<(Vec<i32>, Vec<ImageRecord>)> = (0..3).map(|i| (conv(seed * 1000 + i, 9_000), if i == 0 { imgs(seed as u8) } else { vec![] })).collect();
        let mut lines: Vec<(usize, Vec<i32>)> = Vec::new();
        let (mut warm, mut restarts, mut job, mut late, mut flushes) = (0, 0, 0u64, 0, 0);
        for step in 0..32 {
            now += rng.gen_range(10..5000);
            job += 1;
            let op = rng.gen_range(0..100);
            if op < 65 || lines.is_empty() {
                let (bi, tokens) = if !lines.is_empty() && rng.gen_bool(0.6) {
                    let i = rng.gen_range(0..lines.len());
                    let (bi, mut t) = lines[i].clone();
                    let base = &bases[bi].0;
                    let end = (t.len() + rng.gen_range(1..1500)).min(base.len());
                    t.extend_from_slice(&base[t.len().min(end)..end]);
                    if rng.gen_bool(0.25) {
                        let p = rng.gen_range(100..t.len());
                        t[p] ^= 1; // a branch
                    }
                    lines[i] = (bi, t.clone());
                    (bi, t)
                } else {
                    let bi = rng.gen_range(0..bases.len());
                    let l = if rng.gen_bool(0.25) { K as usize } else { rng.gen_range(200..3500) };
                    let t = bases[bi].0[..l].to_vec();
                    lines.push((bi, t.clone()));
                    (bi, t)
                };
                if rng.gen_bool(0.2) {
                    let pattern: Vec<bool> = (0..6).map(|_| rng.gen_bool(0.3)).collect();
                    s.io.inject_write_faults(&pattern);
                }
                let o = Adm { fail: rng.gen_bool(0.05), flush: rng.gen_bool(0.4), anchor: rng.gen_bool(0.3) };
                late += !o.flush as u32;
                let images = bases[bi].1.clone();
                warm += (admit_opts(&mut s, &tokens, &images, now, job, o) > 0) as u32;
            } else if op < 75 {
                // Land whatever completed so far.
                let _ = s.process_completions(now);
            } else if op < 85 {
                // Damage a random file; the next read of it must evict it.
                let _ = s.flush(true, now);
                let files = store_files(&s);
                if !files.is_empty() {
                    let p = &files[rng.gen_range(0..files.len())];
                    let mut b = fs::read(p).unwrap();
                    let mut at = rng.gen_range(0..b.len());
                    let hits = format::TAIL_HITS_OFFSET as usize;
                    if p.extension().is_some_and(|e| e == "kvt") && (hits..hits + 4).contains(&at) {
                        at = 0; // hits is outside the checksum by design
                    }
                    b[at] ^= 0x40;
                    fs::write(p, &b).unwrap();
                    assert_eq!(scrub(&mut s, now), 1, "seed {seed} step {step}: the damaged file was not caught");
                }
            } else {
                let _ = s.flush(true, now);
                let live = snapshot(&s);
                s.shutdown(Duration::from_secs(5), now);
                s = Store::open(config(&root, cap, BUILD_A), &ns, now).unwrap();
                let back = snapshot(&s);
                assert_eq!(back.0, live.0, "seed {seed} step {step}: chunks differ after restart");
                assert_eq!(back.1, live.1, "seed {seed} step {step}: tails differ after restart");
                for ((k, b), (_, l)) in back.2.iter().zip(&live.2) {
                    let own = s.index().tail(k).unwrap().last_used;
                    assert!(*b <= *l && *b >= own, "seed {seed} step {step}: path_last_used {b} vs live {l}");
                }
                restarts += 1;
            }
            s.check_invariants().unwrap_or_else(|e| panic!("seed {seed} step {step}: {e}"));
            assert!(!s.needs_rescan(), "seed {seed} step {step}: counter drift");
            assert!(s.index().total_bytes() <= cap, "seed {seed} step {step}: over the cap");
            if rng.gen_bool(0.4) {
                let _ = s.flush(true, now);
                flushes += 1;
                s.check_invariants().unwrap_or_else(|e| panic!("seed {seed} step {step} (flushed): {e}"));
                assert_disk_is_index(&s);
            }
        }
        let _ = s.flush(true, now);
        assert_disk_is_index(&s);
        for j in 1..=job {
            assert_eq!(s.index().finished(JobId(j)), None, "seed {seed}: job {j} never forgotten");
            assert_eq!(s.index().job_pin_count(JobId(j)), 0, "seed {seed}: job {j} kept pins");
        }
        assert!(warm >= 5 && restarts >= 1 && late >= 5 && flushes >= 5, "seed {seed}: not exercised ({warm}, {restarts}, {late}, {flushes})");
        s.shutdown(Duration::from_secs(5), now);
    }
}

/// M1 measurements (13): blake3 throughput and unlink rate on this box.
///
/// `V41_KV_BENCH_DIR` must be on the store's filesystem (btrfs over dm-crypt
/// under /home), NOT /tmp (tmpfs). Run once, niced:
/// `V41_KV_BENCH_DIR=... nice -n 19 cargo test --release -p deepstrix-server
///  --features v41 --lib kvstore::tests::bench_blake3_and_unlink -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_blake3_and_unlink() {
    // blake3, single thread, over buffers larger than the L3 (a restore
    // verifies from the page cache, i.e. from DRAM).
    let chunk = format::chunk_payload_len(&format::v41_stores()) as usize;
    let n_chunks = 64; // 181 MB resident, well past the L3
    let mut buf = vec![0u8; chunk * n_chunks];
    StdRng::seed_from_u64(1).fill(&mut buf[..]);
    let rounds = 4;
    let t0 = Instant::now();
    let mut x = 0u8;
    for _ in 0..rounds {
        for c in buf.chunks(chunk) {
            x ^= blake3::hash(c).as_bytes()[0];
        }
    }
    let secs = t0.elapsed().as_secs_f64();
    let gbs = (buf.len() * rounds) as f64 / secs / 1e9;
    println!("blake3 1 thread: {gbs:.2} GB/s over {} x 2.83 MB chunks; 911 MB (330K prefix) = {:.0} ms [{x}]", n_chunks * rounds, 0.911 / gbs * 1e3);
    for threads in [2usize, 4] {
        let t0 = Instant::now();
        std::thread::scope(|sc| {
            for part in buf.chunks(buf.len().div_ceil(threads)) {
                sc.spawn(move || {
                    for _ in 0..rounds {
                        for c in part.chunks(chunk) {
                            std::hint::black_box(blake3::hash(c));
                        }
                    }
                });
            }
        });
        let g = (buf.len() * rounds) as f64 / t0.elapsed().as_secs_f64() / 1e9;
        println!("blake3 {threads} threads (per-chunk parallel): {g:.2} GB/s; 911 MB = {:.0} ms", 0.911 / g * 1e3);
    }
    drop(buf);

    let Some(dir) = std::env::var_os("V41_KV_BENCH_DIR").map(PathBuf::from) else {
        println!("V41_KV_BENCH_DIR unset: unlink bench skipped");
        return;
    };
    let dir = dir.join(format!("kvbench-{}", std::process::id()));
    // Real chunk files on the production disk: kept to ~570 MB of writes.
    for (n, size) in [(4000usize, 4096usize), (200, chunk)] {
        fs::create_dir_all(&dir).unwrap();
        let data = vec![0x5au8; size];
        let t0 = Instant::now();
        let mut paths = Vec::with_capacity(n);
        for i in 0..n {
            let sub = dir.join(format!("{:02x}", i % 256));
            if i < 256 {
                fs::create_dir_all(&sub).unwrap();
            }
            let p = sub.join(format!("{i}.kvc"));
            let mut f = fs::File::create(&p).unwrap();
            std::io::Write::write_all(&mut f, &data).unwrap();
            f.sync_all().unwrap(); // on disk before we time the unlinks
            paths.push(p);
        }
        let create = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        for p in &paths {
            fs::remove_file(p).unwrap();
        }
        let unlink = t1.elapsed().as_secs_f64();
        // The deferred part of the cost (extent/csum tree updates) lands in the
        // next transaction commit.
        let t2 = Instant::now();
        fs::File::open(&dir).unwrap().sync_all().unwrap();
        let dsync = t2.elapsed().as_secs_f64();
        println!(
            "unlink {n} x {} KiB: {:.0} files/s ({:.1} us each; dir fsync after {:.0} ms); create+fsync {:.1} s; 37K files = {:.1} s",
            size / 1024,
            n as f64 / unlink,
            unlink / n as f64 * 1e6,
            dsync * 1e3,
            create,
            37_000.0 * unlink / n as f64
        );
        fs::remove_dir_all(&dir).unwrap();
    }
}
