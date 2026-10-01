//! Store-level tests over real files in a temp dir (synthetic tokens and
//! payloads; no GPU), and the M1 measurements (`#[ignore]`).
//!
//! Set TMPDIR to a disk-backed directory when running these: chunk files are
//! 2.83 MB and tails up to ~5.7 MB, the real sizes.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use super::format::{self, BuildId, TailKind, TailOrigin};
use super::index::JobId;
use super::io::tests::unique_dir;
use super::keys::{ChainCursor, Key, NamespaceInputs};
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

/// One admission through the store, as M2's job will run it: walk, select,
/// verify + touch the plan, write the missing chunks, waypoints and the
/// prompt-end tail, then end the job. Returns the restored t.
fn admit(s: &mut Store, tokens: &[i32], now: u64, job: u64) -> u32 {
    let l = tokens.len() as u32;
    // Verify the plan before using it; a file that fails is evicted and
    // selection re-runs without it (6.2).
    let (walk, plan) = loop {
        let (walk, _) = s.walk(tokens, &[], now);
        let Some(p) = Store::select(&walk, l, false) else { break (walk, None) };
        let pin = s.pin_plan(&p.key).unwrap();
        let mut ok = true;
        for (k, key) in walk.chunks.iter().enumerate().take((p.t / C) as usize) {
            let k = k as u32;
            match s.read_chunk(key, &tokens[(k * C) as usize..((k + 1) * C) as usize], now) {
                Ok(f) => assert_eq!(f.payload, chunk_payload(tokens, k)),
                Err(_) => ok = false,
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
            _ => ok = false,
        }
        s.unpin(pin, now);
        if ok {
            s.touch(&p.key, &walk.ancestors_below(p.t), now);
            break (walk, Some(p));
        }
    };
    let pos0 = plan.map_or(0, |p| p.t);
    let job = JobId(job);
    let mut cur: ChainCursor = s.chain().cursor();
    cur.seed(&walk.chunks);
    let mut ancestors = walk.ancestors_below(l);
    let mut budget = s.tick_budget();
    for k in pos0 / C..l / C {
        let req = ChunkWriteReq { tokens, images: &[], k, payload: chunk_payload(tokens, k), provenance: format::Provenance::Prefill, job: Some(job) };
        s.write_chunk(&mut cur, req, &mut budget, now).unwrap();
        let end = (k + 1) * C;
        if end % K == 0 && end > pos0 && end < l {
            let req = TailWriteReq {
                tokens,
                images: &[],
                t: end,
                kind: TailKind::Enc,
                origin: TailOrigin::Waypoint,
                n_raw: 128,
                n_raw_dec: 0,
                drafter: [0; 32],
                session_id: None,
                sec_e: sec_e_for(tokens, end),
                sec_d: Vec::new(),
                ancestors: ancestors.clone(),
                job: Some(job),
            };
            s.write_tail(&mut cur, req, &mut budget, now).unwrap();
            ancestors.push(s.chain().tail_key(&mut cur, tokens, &[], end).1);
        }
    }
    let req = TailWriteReq {
        tokens,
        images: &[],
        t: l,
        kind: TailKind::Full,
        origin: TailOrigin::PromptEnd,
        n_raw: l.min(128),
        n_raw_dec: l.min(128),
        drafter: [5; 32],
        session_id: Some("sess"),
        sec_e: sec_e_for(tokens, l),
        sec_d: sec_d_for(tokens, l),
        ancestors,
        job: Some(job),
    };
    s.write_tail(&mut cur, req, &mut budget, now).unwrap();
    s.flush(false, now);
    s.job_finished(job, false, now);
    s.check_invariants().unwrap();
    pos0
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

fn store_bytes_on_disk(s: &Store) -> u64 {
    files_under(&s.dirs().base)
        .iter()
        .filter(|p| p.extension().is_some_and(|e| e == "kvc" || e == "kvt"))
        .map(|p| fs::metadata(p).unwrap().len())
        .sum()
}

#[test]
fn roundtrip_restore_and_reopen() {
    let root = unique_dir("st-roundtrip");
    let ns = ns_inputs(1);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(1, 12_000);
    assert_eq!(admit(&mut s, &a[..10_000], 1000, 1), 0, "cold");
    assert_eq!((s.index().n_chunks(), s.index().n_tails()), (9, 2), "9 chunks, a waypoint and the prompt end");
    // The next turn restores the prompt end, reading and verifying every byte.
    assert_eq!(admit(&mut s, &a[..10_500], 2000, 2), 10_000);
    // A t ≤ 128 continuation reads section D too.
    assert_eq!(admit(&mut s, &a[..10_600], 3000, 3), 10_500);
    // Disk equals the index.
    s.flush(true, 3000);
    assert_eq!(store_bytes_on_disk(&s), s.index().chunk_bytes() + s.index().tail_bytes());
    let before: Vec<(Key, u32, TailKind, u32, u64)> = {
        let mut v: Vec<_> = s.index().tail_keys().map(|k| {
            let e = s.index().tail(k).unwrap();
            (*k, e.t, e.kind, e.hits, e.path_last_used)
        }).collect();
        v.sort();
        v
    };
    let stats = s.stats();
    assert!(stats.log_line().starts_with("kv.store chunks=10 tails=4"), "{}", stats.log_line());
    s.shutdown(Duration::from_secs(1));

    // Reopen: the index is rebuilt from the files alone.
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 4000).unwrap();
    s2.check_invariants().unwrap();
    let r = s2.scan_report();
    assert_eq!((r.chunks, r.tails, r.invalid, r.unreachable, r.missing_ancestor), (10, 4, 0, 0, 0));
    assert!(!r.gen_new);
    let mut after: Vec<_> = s2.index().tail_keys().map(|k| {
        let e = s2.index().tail(k).unwrap();
        (*k, e.t, e.kind, e.hits, e.path_last_used)
    }).collect();
    after.sort();
    assert_eq!(before, after, "kinds, hits and path_last_used survive a restart");
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
        let mut v: Vec<_> = s.index().tail_keys().map(|k| {
            let e = s.index().tail(k).unwrap();
            (e.t, e.last_used, e.path_last_used)
        }).collect();
        v.sort();
        v
    };
    let live = snap(&s);
    let at = |v: &[(u32, u64, u64)], t: u32| v.iter().find(|x| x.0 == t).copied().unwrap();
    assert_eq!(at(&live, 8192).2, 8000, "the shared waypoint is young through W");
    assert_eq!(at(&live, 9100).2, 4000, "Y is young through X only");
    s.shutdown(Duration::from_secs(1));
    let s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 9000).unwrap();
    assert_eq!(snap(&s2), live);
}

#[test]
fn torn_short_and_corrupt_files_read_as_a_miss() {
    // G5: a damaged file never reads as data. Byte flips anywhere are caught,
    // except in the tail's `hits` (deliberately outside the checksum), which
    // then changes only `hits`.
    let root = unique_dir("st-torn");
    let ns = ns_inputs(3);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    let a = conv(3, 2_100);
    admit(&mut s, &a, 1000, 1);
    let (walk, _) = s.walk(&a, &[], 1000);
    let ck = walk.chunks[1];
    let tk = walk.tails.last().unwrap().key;
    let cpath = s.dirs().chunk_path(&ck);
    let tpath = s.dirs().tail_path(&tk);
    let nsk = *s.chain().ns();
    let corig = fs::read(&cpath).unwrap();
    let torig = fs::read(&tpath).unwrap();
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
                Err(_) => {}
                Ok(t) => {
                    let off = format::TAIL_HITS_OFFSET as usize;
                    assert!(b.len() == orig.len() && (0..b.len()).filter(|&j| b[j] != orig[j]).all(|j| (off..off + 4).contains(&j)), "case {i}: damaged tail read as data");
                    assert_eq!(format::TailFile { header: format::TailHeader { hits: tgood.header.hits, ..t.header.clone() }, ..t }, tgood);
                }
            }
        } else {
            assert!(format::read_chunk(&scratch, &nsk).is_err(), "case {i}: damaged chunk read as data");
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
    assert!(s.walk(&a, &[], 1000).0.chunks.is_empty());
    s.flush(true, 1000);
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
fn startup_scan_repairs_and_drops() {
    let root = unique_dir("st-scan");
    let ns = ns_inputs(4);
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
    s.shutdown(Duration::from_secs(1));
    // (1) A crash between a demotion's truncate and its header rewrite.
    let ha = format::read_tail(&pa, &nsk, false).unwrap().header;
    fs::OpenOptions::new().write(true).open(&pa).unwrap().set_len(ha.e_end()).unwrap();
    // (2) A file longer than its sections.
    let mut f = fs::OpenOptions::new().append(true).open(&pb).unwrap();
    std::io::Write::write_all(&mut f, &[0u8; 100]).unwrap();
    drop(f);
    // (3) A write in flight at the crash.
    fs::write(&tmp_left, b"partial").unwrap();
    // (4) A foreign file and a torn chunk.
    fs::write(root.join(keys::ns16(&nsk)).join("chunks").join("README"), b"x").unwrap();
    let mut s2 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 2000).unwrap();
    let r = s2.scan_report().clone();
    assert_eq!((r.repaired_demotions, r.truncated, r.invalid), (1, 1, 0));
    let ea = s2.index().tail(&ta).unwrap();
    assert!(ea.kind == TailKind::Enc && ea.demoted && ea.bytes == ha.e_end());
    s2.flush(true, 2000);
    let fa = format::read_tail(&pa, &nsk, true).unwrap();
    assert!(fa.header.kind == TailKind::Enc && fa.header.demoted && fa.sec_d.is_none());
    assert!(format::read_tail(&pb, &nsk, true).is_ok());
    assert!(!tmp_left.exists() && fs::read_dir(s2.dirs().tmp()).unwrap().count() == 0);
    assert_eq!(fs::read_dir(root.join("trash")).unwrap().count(), 0, "trash drained in the background");
    s2.check_invariants().unwrap();
    // A torn chunk: dropped, and the tails above it with it.
    let c0 = s2.walk(&a, &[], 2000).0.chunks[0];
    let cpath = s2.dirs().chunk_path(&c0);
    s2.shutdown(Duration::from_secs(1));
    let len = fs::metadata(&cpath).unwrap().len();
    fs::OpenOptions::new().write(true).open(&cpath).unwrap().set_len(len / 2).unwrap();
    let mut s3 = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 3000).unwrap();
    let r = s3.scan_report().clone();
    assert_eq!((r.invalid, r.unreachable, r.missing_ancestor), (1, 4, 1), "{r:?}");
    assert!(s3.walk(&a, &[], 3000).0.tails.is_empty());
    assert_eq!(s3.walk(&b, &[], 3000).0.tails.len(), 1, "the other conversation is untouched");
    s3.flush(true, 3000);
    assert_eq!(store_bytes_on_disk(&s3), s3.index().chunk_bytes() + s3.index().tail_bytes());
}

#[test]
fn namespace_gc_keeps_active_and_previous_and_evicts_inactive_first() {
    let root = unique_dir("st-ns");
    let a = conv(6, 3_000);
    for (i, tag) in [10u8, 11, 12].iter().enumerate() {
        let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(*tag), 1000 + 100 * i as u64).unwrap();
        admit(&mut s, &a, 1000 + 100 * i as u64, 1);
        s.shutdown(Duration::from_secs(1));
    }
    let ns_dir = |tag: u8| root.join(keys::ns16(&ns_inputs(tag).key()));
    let one_ns = store_bytes_on_disk_dir(&ns_dir(12));
    // Every open keeps the active namespace and the most recently active
    // other: opening 12 already trashed 10. Opening 13 keeps 12, trashes 11.
    assert!(!ns_dir(10).exists() && ns_dir(11).exists());
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns_inputs(13), 2000).unwrap();
    let r = s.scan_report().clone();
    assert_eq!(r.trashed_namespaces, 1);
    assert_eq!(r.kept_inactive, Some((keys::ns16(&ns_inputs(12).key()), one_ns)));
    s.flush(true, 2000);
    assert!(!ns_dir(10).exists() && !ns_dir(11).exists() && ns_dir(12).exists());
    assert_eq!(fs::read_dir(root.join("trash")).unwrap().count(), 0);
    // One global cap: the inactive namespace is evicted first, whole.
    admit(&mut s, &a, 2100, 1);
    let mine = s.index().chunk_bytes() + s.index().tail_bytes();
    s.index.set_cap_bytes(mine + one_ns / 2);
    assert!(s.index.enforce_cap(2200));
    s.apply();
    s.flush(true, 2200);
    assert!(!ns_dir(12).exists(), "the inactive namespace went first");
    assert_eq!(s.index().n_tails(), 1, "the active one is untouched");
}

fn store_bytes_on_disk_dir(dir: &Path) -> u64 {
    files_under(dir).iter().map(|p| fs::metadata(p).unwrap().len()).sum()
}

#[test]
fn purge_by_build_cascades_through_shared_prefixes() {
    let root = unique_dir("st-purge");
    let ns = ns_inputs(20);
    let a = conv(7, 8_000);
    let other = conv(8, 2_100);
    let mut s = Store::open(config(&root, 100 << 30, BUILD_A), &ns, 1000).unwrap();
    admit(&mut s, &a[..3_000], 1000, 1);
    s.shutdown(Duration::from_secs(1));
    // A later build continues the conversation on top of build A's chunks.
    let mut s = Store::open(config(&root, 100 << 30, BUILD_B), &ns, 2000).unwrap();
    admit(&mut s, &a[..7_000], 2000, 1);
    admit(&mut s, &other, 2000, 2);
    s.shutdown(Duration::from_secs(1));
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
    s.flush(true, 2000);
    assert_eq!(store_bytes_on_disk(&s), s.index().chunk_bytes() + s.index().tail_bytes());
    assert!(s.stats().evicted_bytes > 0);
    // The newest conversations survive, the oldest went.
    assert!(s.walk(&conv(100, 5_200), &[], 2000).0.tails.is_empty());
    assert_eq!(s.walk(&conv(107, 5_200), &[], 2000).0.tails.len(), 1);
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
    let req = |k| ChunkWriteReq { tokens: &a, images: &[], k, payload: chunk_payload(&a, k), provenance: format::Provenance::Prefill, job: None };
    assert_eq!(s.write_chunk(&mut cur, req(0), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    assert_eq!(s.write_chunk(&mut cur, req(0), &mut b, 1000).unwrap(), WriteOutcome::Pending);
    assert_eq!(s.write_chunk(&mut cur, req(1), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    s.flush(false, 1000);
    assert_eq!(s.write_chunk(&mut cur, req(0), &mut b, 1000).unwrap(), WriteOutcome::Stored);
    let t = 2_100;
    let enc = || TailWriteReq {
        tokens: &a,
        images: &[],
        t,
        kind: TailKind::Enc,
        origin: TailOrigin::Cancel,
        n_raw: 128,
        n_raw_dec: 0,
        drafter: [0; 32],
        session_id: None,
        sec_e: sec_e_for(&a, t),
        sec_d: vec![],
        ancestors: vec![],
        job: None,
    };
    assert_eq!(s.write_tail(&mut cur, enc(), &mut b, 1000).unwrap(), WriteOutcome::Queued);
    s.flush(false, 1000);
    let key = s.chain().tail_key(&mut cur, &a, &[], t).1;
    s.touch(&key, &[], 1100); // one hit on the encoder tail
    let full = TailWriteReq { kind: TailKind::Full, origin: TailOrigin::PromptEnd, n_raw_dec: 128, sec_d: sec_d_for(&a, t), ..enc() };
    assert_eq!(s.write_tail(&mut cur, full, &mut b, 1200).unwrap(), WriteOutcome::Queued);
    s.flush(false, 1200);
    let e = s.index().tail(&key).unwrap();
    assert_eq!((e.kind, e.hits), (TailKind::Full, 1), "the full tail replaced the encoder tail and kept its hit");
    assert_eq!(s.write_tail(&mut cur, enc(), &mut b, 1300).unwrap(), WriteOutcome::Stored);
    // Shape checks refuse a malformed capture instead of storing it.
    let bad = TailWriteReq { sec_e: vec![0; 10], ..enc() };
    assert!(matches!(s.write_tail(&mut cur, bad, &mut b, 1300), Err(StoreError::Invalid(_))));
    let short = ChunkWriteReq { payload: vec![0; 5], ..req(1) };
    assert!(matches!(s.write_chunk(&mut cur, short, &mut b, 1300), Err(StoreError::Invalid(_))));
    s.check_invariants().unwrap();
}

#[test]
fn dropped_writes_and_broken_paths() {
    // 9.4: a full queue drops after the budget; a tail whose path lost a chunk
    // is dropped when it lands (broken per key, not per path).
    let root = unique_dir("st-drop");
    let ns = ns_inputs(23);
    let mut cfg = config(&root, 100 << 30, BUILD_A);
    cfg.write_queue_bytes = 1; // every write but the first waits for an empty queue
    cfg.write_wait = Duration::from_millis(1);
    let mut s = Store::open(cfg, &ns, 1000).unwrap();
    let a = conv(10, 3_000);
    let mut cur = s.chain().cursor();
    let mut b = s.tick_budget();
    let mut outcomes = Vec::new();
    for k in 0..2 {
        let req = ChunkWriteReq { tokens: &a, images: &[], k, payload: chunk_payload(&a, k), provenance: format::Provenance::Prefill, job: None };
        outcomes.push(s.write_chunk(&mut cur, req, &mut b, 1000).unwrap());
    }
    assert_eq!(outcomes[0], WriteOutcome::Queued);
    // The second either waited out the first (fast disk) or was dropped.
    s.flush(false, 1000);
    if outcomes[1] == WriteOutcome::Dropped {
        assert_eq!(s.stats().writes_dropped, 1);
        // The tail above the hole lands and is dropped as broken_path.
        let req = TailWriteReq {
            tokens: &a,
            images: &[],
            t: 2_500,
            kind: TailKind::Enc,
            origin: TailOrigin::Cancel,
            n_raw: 128,
            n_raw_dec: 0,
            drafter: [0; 32],
            session_id: None,
            sec_e: sec_e_for(&a, 2_500),
            sec_d: vec![],
            ancestors: vec![],
            job: None,
        };
        let mut b = s.tick_budget();
        assert_eq!(s.write_tail(&mut cur, req, &mut b, 1000).unwrap(), WriteOutcome::Queued);
        let ev = s.flush(true, 1000);
        assert!(ev.iter().any(|e| matches!(e, StoreEvent::TailDropped { why: "broken_path", .. })), "{ev:?}");
        assert_eq!(s.index().n_tails(), 0);
        assert_eq!(fs::read_dir(s.dirs().base.join("tails")).unwrap().flatten().map(|d| fs::read_dir(d.path()).unwrap().count()).sum::<usize>(), 0);
    }
    s.check_invariants().unwrap();
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
    s.flush(true, 1100);
    // The demotion found a bad header: the tail is evicted, not left half-done.
    assert_eq!(s.walk(&a[..2_000], &[], 1100).0.tails.iter().filter(|t| t.t == 2_000).count(), 0);
    s.check_invariants().unwrap();
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
            s.index.remove_chunk(&k, index::Why::Corrupt, now);
            bad += 1;
        }
    }
    for k in tails {
        if s.index().tail(&k).is_some() && format::read_tail(&s.dirs().tail_path(&k), &ns, true).is_err() {
            s.index.remove_tail(&k, index::Why::Corrupt, now);
            bad += 1;
        }
    }
    s.apply();
    bad
}

type ChunkSnap = (Key, u32, u64, u32);
type TailSnap = (Key, u32, TailKind, bool, bool, u32, u64, u64);

/// What a restart must preserve, per entry. `path_last_used` is returned
/// apart: it is compared as "rebuilt ≤ live", because live values only grow
/// while a rebuild sees only the tails still below (evicted or thinned
/// descendants no longer count).
fn snapshot(s: &Store) -> (Vec<ChunkSnap>, Vec<TailSnap>, Vec<(Key, u64)>) {
    let idx = s.index();
    let mut c: Vec<_> = idx
        .chunk_keys()
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

/// Random admissions (new, continued, branched, retried), cap pressure,
/// damaged files and restarts, through real files and the IO thread. After
/// every step: invariants, the cap, and the bytes on disk equal the index.
/// After every restart: the rebuilt index equals the live one.
#[test]
fn randomized_store_with_restarts() {
    for seed in 1..=2u64 {
        let mut rng = StdRng::seed_from_u64(seed * 7919);
        let root = unique_dir("st-rand");
        let ns = ns_inputs(30 + seed as u8);
        let cap = 45 << 20;
        let mut now = 10_000u64;
        let mut s = Store::open(config(&root, cap, BUILD_A), &ns, now).unwrap();
        let bases: Vec<Vec<i32>> = (0..3).map(|i| conv(seed * 1000 + i, 6_000)).collect();
        let mut lines: Vec<Vec<i32>> = Vec::new();
        let (mut warm, mut restarts, mut job) = (0, 0, 0u64);
        for step in 0..28 {
            now += rng.gen_range(10..5000);
            job += 1;
            let op = rng.gen_range(0..100);
            if op < 70 || lines.is_empty() {
                let tokens = if !lines.is_empty() && rng.gen_bool(0.6) {
                    let i = rng.gen_range(0..lines.len());
                    let base = &bases[i % bases.len()];
                    let mut t = lines[i].clone();
                    let end = (t.len() + rng.gen_range(1..1500)).min(base.len());
                    t.extend_from_slice(&base[t.len().min(end)..end]);
                    if rng.gen_bool(0.25) {
                        let p = rng.gen_range(100..t.len());
                        t[p] ^= 1; // a branch
                    }
                    lines[i] = t.clone();
                    t
                } else {
                    let base = &bases[rng.gen_range(0..bases.len())];
                    let t = base[..rng.gen_range(200..3500)].to_vec();
                    lines.push(t.clone());
                    t
                };
                warm += (admit(&mut s, &tokens, now, job) > 0) as u32;
            } else if op < 85 {
                // Damage a random file; the next read of it must evict it.
                let mut files: Vec<PathBuf> = files_under(&s.dirs().base)
                    .into_iter()
                    .filter(|p| p.extension().is_some_and(|e| e == "kvc" || e == "kvt"))
                    .collect();
                files.sort();
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
                s.flush(true, now);
                let live = snapshot(&s);
                s.shutdown(Duration::from_secs(5));
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
            s.flush(true, now);
            s.check_invariants().unwrap_or_else(|e| panic!("seed {seed} step {step}: {e}"));
            assert!(s.index().total_bytes() <= cap, "seed {seed} step {step}: over the cap");
            let disk = store_bytes_on_disk(&s);
            assert_eq!(disk, s.index().chunk_bytes() + s.index().tail_bytes(), "seed {seed} step {step}: disk != index");
        }
        assert!(warm >= 5 && restarts >= 1, "seed {seed}: not exercised (warm {warm}, restarts {restarts})");
        s.shutdown(Duration::from_secs(5));
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
