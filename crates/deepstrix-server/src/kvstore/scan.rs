//! Startup: the root lock, namespace GC and the header scan (design 9.5, 4.4,
//! 4.5, 8.3).
//!
//! The scan rebuilds the whole index from the files: nothing about the index
//! is persisted except what each file says about itself (its header, its
//! mtime = `last_used` for tails). It never waits on a deletion: stale
//! namespaces and the old `tmp/` are RENAMED into `trash/` (one rename each)
//! and the IO thread removes them in the background; files that fail
//! validation are queued for unlink.
//!
//! What the scan repairs (4.5):
//! - a full tail whose file ends right after section E: a crash between the
//!   demotion's truncate and its header rewrite. It is indexed as demoted and
//!   its header rewrite is queued.
//! - a file longer than its sections: truncated.
//!
//! What it drops: torn or short files, bad headers, foreign-namespace or
//! misnamed files, chunks whose parent chain does not reach the root, tails
//! with a missing chunk on their path (and so everything beneath a purged
//! chunk). A file whose read failed for a reason that says nothing about its
//! data (EACCES, EIO, EMFILE...) is left alone and counted.

use std::fs::{self, File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::format::{
    self, BuildId, ChunkHeader, FormatError, GenPair, StoreRows, TailHeader, TailKind, CHUNK_HEADER_LEN, GEN_V6,
    TAIL_HEADER_LEN,
};
use super::index::{ChunkInsert, Index, TailInsert};
use super::io::{mkdir_private, mtime_secs, unix_now, IoJob, NsDirs};
use super::keys::{self, images_in, Key, KeyMap};
use super::C;

/// flock(2), declared here rather than pulling in a crate for one call.
mod sys {
    extern "C" {
        pub fn flock(fd: i32, operation: i32) -> i32;
    }
    pub const LOCK_EX: i32 = 2;
    pub const LOCK_NB: i32 = 4;
}

/// Take the store root's lock, `kvstore-v1/.lock`, with
/// `flock(LOCK_EX | LOCK_NB)`. Opening a store renames the other namespaces
/// and its own `tmp/` into trash, so a second process on the same root would
/// break a live one. The lock lives as long as the returned file is open; on
/// failure the caller runs with the store off and logs why.
pub fn lock_root(root: &Path) -> io::Result<File> {
    mkdir_private(root)?;
    let f = OpenOptions::new().create(true).truncate(false).write(true).mode(0o600).open(root.join(".lock"))?;
    // SAFETY: `flock` reads two ints; the descriptor belongs to `f`, which is
    // open for the duration of the call.
    let r = unsafe { sys::flock(f.as_raw_fd(), sys::LOCK_EX | sys::LOCK_NB) };
    if r != 0 {
        let e = io::Error::last_os_error();
        return Err(io::Error::new(e.kind(), format!("kvstore root {} is locked by another process: {e}", root.display())));
    }
    Ok(f)
}

/// `V41_KV_STORE_PURGE_BUILD` (4.4): an explicit operator action, never
/// routine retention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PurgeSpec {
    /// A git sha prefix (≥ 7 hex digits).
    Build(String),
    /// `knob:<hex prefix>` of the knob hash (≥ 8 hex digits).
    Knob(String),
    /// `gen:<n>`: one `KV_NUMERICS_GEN`.
    Gen(u32),
    /// `v6`: everything backfilled from v6 snapshots.
    V6,
}

impl PurgeSpec {
    /// Comma-separated list.
    pub fn parse_list(s: &str) -> Result<Vec<Self>, String> {
        let is_hex = |x: &str, min: usize| x.len() >= min && x.bytes().all(|b| b.is_ascii_hexdigit());
        s.split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(|x| {
                if x == "v6" {
                    Ok(Self::V6)
                } else if let Some(h) = x.strip_prefix("knob:") {
                    is_hex(h, 8).then(|| Self::Knob(h.to_ascii_lowercase())).ok_or(format!("bad knob hash {h:?} (≥ 8 hex digits)"))
                } else if let Some(g) = x.strip_prefix("gen:") {
                    g.parse().map(Self::Gen).map_err(|_| format!("bad generation {g:?}"))
                } else if is_hex(x, 7) {
                    Ok(Self::Build(x.to_ascii_lowercase()))
                } else {
                    Err(format!("bad purge spec {x:?} (want <sha ≥ 7 hex>, knob:<hash ≥ 8 hex>, gen:<n> or v6)"))
                }
            })
            .collect()
    }

    pub fn matches(&self, build: &BuildId, gen: &GenPair) -> bool {
        match self {
            Self::Build(p) => !build.v6 && keys::hex(&build.sha).starts_with(p.as_str()),
            Self::Knob(p) => keys::hex(&gen.knob).starts_with(p.as_str()),
            Self::Gen(n) => gen.gen == *n,
            Self::V6 => build.v6 || gen.gen == GEN_V6,
        }
    }
}

/// What the scan saw and did (logged as `kv.scan`).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ScanReport {
    pub chunks: u64,
    pub tails: u64,
    pub bytes: u64,
    /// Bad header, wrong size, wrong namespace or name, misplaced: unlinked.
    pub invalid: u64,
    pub unreachable: u64,
    pub missing_ancestor: u64,
    pub purged: u64,
    /// A purge that matched the running build or generation: not applied.
    pub purge_refused: bool,
    /// Crash between a demotion's truncate and its header rewrite.
    pub repaired_demotions: u64,
    pub truncated: u64,
    /// Files in our directories that are not ours (left alone).
    pub foreign: u64,
    /// Files whose read failed transiently (left alone, not indexed).
    pub io_errors: u64,
    /// Unreachable / missing-ancestor files NOT unlinked because some read
    /// failed transiently: the missing link may be one of those files, so the
    /// next startup judges them again.
    pub deferred: u64,
    pub kept_inactive: Option<(String, u64)>,
    pub trashed_namespaces: u64,
    /// The running effective generation has no file in the store yet
    /// (`kv.gen_new`, 4.4).
    pub gen_new: bool,
    pub elapsed_ms: u64,
}

/// The result of opening a namespace: the index plus the file work the IO
/// thread must do once it runs, and the root lock.
pub struct Opened {
    pub index: Index,
    pub jobs: Vec<IoJob>,
    pub unlinks: Vec<(PathBuf, u64)>,
    pub report: ScanReport,
    pub lock: File,
}

const LAST_ACTIVE: &str = "last_active";
/// The namespace's byte total, written by the store at shutdown and hourly,
/// so a kept inactive namespace is counted without statting its ~37K files.
pub const NS_BYTES: &str = "bytes";

/// Write a small file atomically (tmp + rename).
pub fn write_small(path: &Path, contents: &str) -> io::Result<()> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, contents)?;
    fs::rename(&tmp, path)
}

fn read_u64_file(path: &Path) -> Option<u64> {
    fs::read_to_string(path).ok().and_then(|s| s.trim().parse().ok())
}

fn is_ns16(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Total bytes of the regular files under `dir` (the fallback when a
/// namespace has no persisted total).
fn dir_bytes(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(e.path()),
                Ok(_) => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                Err(_) => {}
            }
        }
    }
    total
}

/// Rename `from` into `trash/` under a unique name.
fn rename_to_trash(trash: &Path, from: &Path, seq: &mut u32) -> io::Result<PathBuf> {
    mkdir_private(trash)?;
    *seq += 1;
    let name = from.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let to = trash.join(format!("{name}-{}-{}-{}", unix_now(), std::process::id(), seq));
    fs::rename(from, &to)?;
    Ok(to)
}

struct ScannedChunk {
    h: ChunkHeader,
    size: u64,
    path: PathBuf,
}

struct ScannedTail {
    h: TailHeader,
    /// The size after the queued repair, if any.
    size: u64,
    mtime: u64,
    path: PathBuf,
    repair: Option<IoJob>,
}

/// Open the namespace `ns` under `root` (= `kvstore-v1/`): lock the root, GC
/// the other namespaces, clear `tmp/`, scan, rebuild.
#[allow(clippy::too_many_arguments)]
pub fn open_namespace(
    root: &Path,
    ns: &Key,
    cap_bytes: u64,
    expect_stores: &[StoreRows],
    purge: &[PurgeSpec],
    current_build: &BuildId,
    current_gen: GenPair,
    now: u64,
) -> io::Result<Opened> {
    let t0 = Instant::now();
    let lock = lock_root(root)?;
    let mut report = ScanReport::default();
    let mut jobs = Vec::new();
    let mut unlinks: Vec<(PathBuf, u64)> = Vec::new();
    let trash = root.join("trash");
    mkdir_private(&trash)?;
    let mut seq = 0u32;

    // Namespace GC (4.4): keep the active namespace and the most recently
    // active other one (for rollback); rename the rest into trash/.
    let active = keys::ns16(ns);
    let mut others: Vec<(u64, String)> = Vec::new();
    for e in fs::read_dir(root)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name != active && is_ns16(&name) && e.file_type().is_ok_and(|t| t.is_dir()) {
            others.push((read_u64_file(&e.path().join(LAST_ACTIVE)).unwrap_or(0), name));
        }
    }
    others.sort_unstable_by(|a, b| b.cmp(a));
    for (i, (_, name)) in others.iter().enumerate() {
        let dir = root.join(name);
        if i == 0 {
            let bytes = read_u64_file(&dir.join(NS_BYTES)).unwrap_or_else(|| dir_bytes(&dir));
            report.kept_inactive = Some((name.clone(), bytes));
        } else {
            rename_to_trash(&trash, &dir, &mut seq)?;
            report.trashed_namespaces += 1;
        }
    }
    // Leftovers of earlier runs (and the renames above) are removed in the
    // background.
    for e in fs::read_dir(&trash)?.flatten() {
        jobs.push(IoJob::Trash { from: e.path() });
    }

    let dirs = NsDirs::new(root.join(&active));
    dirs.create()?;
    write_small(&dirs.base.join(LAST_ACTIVE), &format!("{now}\n"))?;
    // Writes in flight at a crash: the whole tmp/ goes (one rename).
    if fs::read_dir(dirs.tmp())?.next().is_some() {
        let to = rename_to_trash(&trash, &dirs.tmp(), &mut seq)?;
        jobs.push(IoJob::Trash { from: to });
        dirs.create()?;
    }

    // Header scan.
    let mut chunks: KeyMap<ScannedChunk> = KeyMap::default();
    let mut tails: KeyMap<ScannedTail> = KeyMap::default();
    for (sub, ext) in [("chunks", "kvc"), ("tails", "kvt")] {
        let Ok(fan) = fs::read_dir(dirs.base.join(sub)) else { continue };
        for d in fan.flatten() {
            let Ok(rd) = fs::read_dir(d.path()) else { continue };
            for f in rd.flatten() {
                let path = f.path();
                let name = f.file_name().to_string_lossy().into_owned();
                let Some(key) = name.strip_suffix(&format!(".{ext}")).and_then(keys::parse_hex_key) else {
                    report.foreign += 1;
                    continue;
                };
                let Ok(meta) = f.metadata() else {
                    report.io_errors += 1;
                    continue;
                };
                let size = meta.len();
                // A copy outside its fan-out directory is never looked up: drop
                // it, without touching a valid copy of the same key.
                let in_fan = d.file_name().to_string_lossy() == keys::hex(&key)[..2];
                let r = if ext == "kvc" {
                    scan_chunk(&path, size, ns, &key, expect_stores).map(|h| {
                        if in_fan {
                            chunks.insert(key, ScannedChunk { h, size, path: path.clone() });
                        }
                    })
                } else {
                    scan_tail(&path, size, ns, &key).map(|(h, repair)| {
                        if in_fan {
                            let size = h.file_len();
                            tails.insert(key, ScannedTail { h, size, mtime: mtime_secs(&meta), path: path.clone(), repair });
                        }
                    })
                };
                match r {
                    Ok(()) if in_fan => {}
                    Err(e) if !e.evicts() => {
                        tracing::warn!(path = %path.display(), error = %e, "kv.suspect scan read failed; file left alone");
                        report.io_errors += 1;
                    }
                    _ => {
                        report.invalid += 1;
                        unlinks.push((path, size));
                    }
                }
            }
        }
    }

    // Purge (4.4): matching files go; everything beneath a purged chunk then
    // fails reachability below. A purge that would match what this binary
    // writes is a misconfiguration: refused, not applied.
    if purge.iter().any(|p| p.matches(current_build, &current_gen)) {
        tracing::error!(?purge, build = %current_build, gen = %current_gen, "kv.purge refused: it matches the running build or generation");
        report.purge_refused = true;
    } else if !purge.is_empty() {
        let hit = |b: &BuildId, g: &GenPair| purge.iter().any(|p| p.matches(b, g));
        chunks.retain(|_, c| {
            let keep = !hit(&c.h.build, &c.h.gen);
            if !keep {
                report.purged += 1;
                unlinks.push((c.path.clone(), c.size));
            }
            keep
        });
        tails.retain(|_, t| {
            let keep = !hit(&t.h.build, &t.h.gen);
            if !keep {
                report.purged += 1;
                unlinks.push((t.path.clone(), t.size));
            }
            keep
        });
    }

    // Rebuild: chunks in breadth-first order from the root, so every chunk
    // finds its parent; anything not reached is unreachable.
    let mut index = Index::new(keys::chain_root(ns), cap_bytes);
    let mut by_parent: KeyMap<Vec<Key>> = KeyMap::default();
    for (k, c) in &chunks {
        by_parent.entry(c.h.parent).or_default().push(*k);
    }
    let mut frontier = vec![(*index.root(), u32::MAX)];
    while let Some((node, k_node)) = frontier.pop() {
        for child in by_parent.remove(&node).unwrap_or_default() {
            let c = &chunks[&child];
            if c.h.k != k_node.wrapping_add(1) {
                continue; // misnumbered: dropped below as unreachable
            }
            index.insert_chunk(
                ChunkInsert {
                    key: child,
                    parent: node,
                    k: c.h.k,
                    bytes: c.size,
                    created: c.h.created,
                    gen: c.h.gen,
                    payload_hash: c.h.payload_hash,
                    job: None,
                },
                now,
            );
            frontier.push((child, c.h.k));
        }
    }
    // A transient read failure may be the missing link of a whole subtree:
    // then nothing is dropped for reachability, only left out of the index.
    let defer = report.io_errors > 0;
    for (k, c) in &chunks {
        if index.chunk(k).is_none() {
            if defer {
                report.deferred += 1;
            } else {
                report.unreachable += 1;
                unlinks.push((c.path.clone(), c.size));
            }
        }
    }
    let mut paths: KeyMap<PathBuf> = chunks.iter().map(|(k, c)| (*k, c.path.clone())).collect();
    for (k, t) in tails {
        let h = &t.h;
        let ins = TailInsert {
            key: h.key,
            base: h.base,
            t: h.t,
            kind: h.kind,
            origin: h.origin,
            anchor: h.anchor,
            demoted: h.demoted,
            bytes: t.size,
            sec_d_bytes: h.sec_d_len,
            // The mtime is the persisted last_used (8.3).
            created: t.mtime,
            gen: h.gen,
            hits: h.hits,
            ancestors: Vec::new(),
            job: None,
        };
        if index.load_tail(ins).is_err() {
            if defer {
                report.deferred += 1;
            } else {
                report.missing_ancestor += 1;
                unlinks.push((t.path.clone(), fs::metadata(&t.path).map(|m| m.len()).unwrap_or(t.size)));
            }
            continue;
        }
        if let Some(job) = t.repair {
            match job {
                IoJob::Demote { .. } => report.repaired_demotions += 1,
                IoJob::Truncate { .. } => report.truncated += 1,
                _ => {}
            }
            jobs.push(job);
        }
        paths.insert(k, t.path);
    }
    recompute_path_last_used(&mut index, &paths);

    report.chunks = index.n_chunks() as u64;
    report.tails = index.n_tails() as u64;
    report.bytes = index.chunk_bytes() + index.tail_bytes();
    report.gen_new = !index.gens().any(|(g, _)| g == current_gen);
    if let Some((ns16, bytes)) = &report.kept_inactive {
        index.set_inactive(vec![(ns16.clone(), *bytes)]);
    }
    report.elapsed_ms = t0.elapsed().as_millis() as u64;
    Ok(Opened { index, jobs, unlinks, report, lock })
}

fn read_block(path: &Path, len: usize, size: u64) -> Result<Vec<u8>, FormatError> {
    if size < len as u64 {
        return Err(FormatError::Short { want: len as u64, got: size });
    }
    let f = File::open(path)?;
    let mut b = vec![0u8; len];
    f.read_exact_at(&mut b, 0)?;
    Ok(b)
}

fn scan_chunk(path: &Path, size: u64, ns: &Key, key: &Key, expect: &[StoreRows]) -> Result<ChunkHeader, FormatError> {
    let h = ChunkHeader::decode(&read_block(path, CHUNK_HEADER_LEN, size)?)?;
    if &h.ns != ns {
        return Err(FormatError::Namespace);
    }
    if &h.key != key {
        return Err(FormatError::Key);
    }
    if h.stores != expect {
        return Err(FormatError::BadField("store shape"));
    }
    if h.k.checked_mul(C).is_none() {
        return Err(FormatError::BadField("k"));
    }
    // Chunks are never mutated in place: any other size is a torn file.
    if size != h.file_len() {
        return Err(FormatError::Short { want: h.file_len(), got: size });
    }
    Ok(h)
}

/// The header as it will be once the returned repair (if any) ran.
fn scan_tail(path: &Path, size: u64, ns: &Key, key: &Key) -> Result<(TailHeader, Option<IoJob>), FormatError> {
    let h = TailHeader::decode(&read_block(path, TAIL_HEADER_LEN, size)?)?;
    if &h.ns != ns {
        return Err(FormatError::Namespace);
    }
    if &h.key != key {
        return Err(FormatError::Key);
    }
    let (e_end, full_end) = (h.e_end(), h.file_len());
    if size == full_end {
        return Ok((h, None));
    }
    if size < e_end {
        return Err(FormatError::Short { want: e_end, got: size });
    }
    if h.kind == TailKind::Full && size < full_end && !h.anchor {
        // The demotion's truncate ran, its header rewrite did not (or section
        // D is torn): finish the demotion. Truncating to e_end also covers a
        // torn D.
        let job = IoJob::Demote { path: path.to_path_buf(), key: Some(h.key), hits: None };
        return Ok((h.demoted(h.hits), Some(job)));
    }
    if size > full_end {
        return Ok((h, Some(IoJob::Truncate { path: path.to_path_buf(), len: full_end })));
    }
    Err(FormatError::Short { want: full_end, got: size })
}

/// `path_last_used` = max(last_used) over a tail and every tail below it on
/// the same path (8.3), from the persisted leaves' mtimes. A tail Y at node b
/// with open ids o is above:
/// - every tail and chunk under node b, if o is empty;
/// - a tail X at the same node with t_X > t_Y whose open ids start with Y's;
/// - the subtree of a child chunk c whose ids start with Y's.
///
/// "Start with" is tested by recomputing Y's key over the other file's ids,
/// read on demand: at most one extra 4 KiB read per child chunk of a node that
/// carries an unaligned tail, and per tail at such a node.
fn recompute_path_last_used(index: &mut Index, paths: &KeyMap<PathBuf>) {
    // Subtree maxima, children before parents.
    let root = *index.root();
    let mut order = vec![root];
    let mut i = 0;
    while i < order.len() {
        let n = order[i];
        order.extend(index.children(&n).iter().filter(|c| index.chunk(c).is_some()).copied());
        i += 1;
    }
    let mut sub_max: KeyMap<u64> = KeyMap::default();
    for n in order.iter().rev() {
        let own = index.tails_at(n).iter().filter_map(|t| index.tail(t)).map(|e| e.last_used).max().unwrap_or(0);
        let kids = index.children(n).iter().filter_map(|c| sub_max.get(c)).copied().max().unwrap_or(0);
        sub_max.insert(*n, own.max(kids));
    }
    let mut updates: Vec<(Key, u64)> = Vec::new();
    for n in &order {
        let ts: Vec<Key> = index.tails_at(n).to_vec();
        if ts.is_empty() {
            continue;
        }
        // Ids are needed only when some tail here has an open part.
        let unaligned = ts.iter().any(|t| index.tail(t).is_some_and(|e| e.t % C != 0));
        let ids_cache: KeyMap<Option<(u32, Vec<i32>, Vec<keys::ImageRecord>)>> = if unaligned {
            ts.iter()
                .chain(index.children(n).iter())
                .map(|k| {
                    let is_tail = index.tail(k).is_some();
                    (*k, paths.get(k).and_then(|p| format::read_ids(p, is_tail).ok()))
                })
                .collect()
        } else {
            KeyMap::default()
        };
        let starts_with = |y: &Key, t_y: u32, other: &Key| -> bool {
            let Some(Some((a, ids, imgs))) = ids_cache.get(other) else { return false };
            let o = (t_y - a) as usize;
            ids.len() >= o && keys::tail_step(n, *a, &ids[..o], images_in(imgs, *a, t_y)) == *y
        };
        for y in &ts {
            let Some(ey) = index.tail(y) else { continue };
            let (t_y, mut plu) = (ey.t, ey.last_used);
            if t_y % C == 0 {
                plu = plu.max(sub_max[n]);
            } else {
                for x in &ts {
                    if let Some(ex) = index.tail(x) {
                        if ex.t > t_y && starts_with(y, t_y, x) {
                            plu = plu.max(ex.last_used);
                        }
                    }
                }
                for c in index.children(n) {
                    if let Some(m) = sub_max.get(c) {
                        if starts_with(y, t_y, c) {
                            plu = plu.max(*m);
                        }
                    }
                }
            }
            updates.push((*y, plu));
        }
    }
    for (k, plu) in updates {
        index.set_path_last_used(&k, plu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn purge_spec_parse_and_match() {
        let v = PurgeSpec::parse_list("0123456, knob:ab12cd34,gen:7,v6").unwrap();
        assert_eq!(v, vec![PurgeSpec::Build("0123456".into()), PurgeSpec::Knob("ab12cd34".into()), PurgeSpec::Gen(7), PurgeSpec::V6]);
        assert!(PurgeSpec::parse_list("012").is_err());
        assert!(PurgeSpec::parse_list("knob:ab12").is_err(), "knob prefixes need ≥ 8 hex digits");
        assert!(PurgeSpec::parse_list("gen:x").is_err());
        let b = BuildId::parse("0123456789abcdef0123456789abcdef01234567").unwrap();
        let mut knob = [0u8; 16];
        knob[..4].copy_from_slice(&[0xab, 0x12, 0xcd, 0x34]);
        let g = GenPair { gen: 7, knob };
        assert!(v[0].matches(&b, &g));
        assert!(v[1].matches(&b, &g));
        assert!(v[2].matches(&b, &g));
        assert!(!v[3].matches(&b, &g));
        assert!(v[3].matches(&BuildId::V6, &g));
        assert!(!v[0].matches(&BuildId::V6, &g), "a sha purge never matches v6 backfill");
    }

    #[test]
    fn root_lock_is_exclusive() {
        let root = crate::kvstore::io::tests::unique_dir("lock");
        let a = lock_root(&root).unwrap();
        let e = lock_root(&root).unwrap_err();
        assert_eq!(e.kind(), io::ErrorKind::WouldBlock, "{e}");
        drop(a);
        assert!(lock_root(&root).is_ok(), "released when the holder closes it");
    }
}
