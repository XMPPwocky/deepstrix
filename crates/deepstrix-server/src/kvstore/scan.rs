//! Startup: namespace GC and the header scan (design 9.5, 4.4, 4.5, 8.3).
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
//! chunk).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::format::{
    self, BuildId, ChunkHeader, GenPair, StoreRows, TailHeader, TailKind, CHUNK_HEADER_LEN, GEN_V6, TAIL_HEADER_LEN,
};
use super::index::{ChunkInsert, Index, TailInsert};
use super::io::{mkdir_private, mtime_secs, unix_now, IoJob, NsDirs};
use super::keys::{self, images_in, Key};
use super::C;

/// `V41_KV_STORE_PURGE_BUILD` (4.4): an explicit operator action, never
/// routine retention.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PurgeSpec {
    /// A git sha prefix (≥ 7 hex digits).
    Build(String),
    /// `knob:<hex prefix>` of the knob hash.
    Knob(String),
    /// `gen:<n>`: one `KV_NUMERICS_GEN`.
    Gen(u32),
    /// `v6`: everything backfilled from v6 snapshots.
    V6,
}

impl PurgeSpec {
    /// Comma-separated list.
    pub fn parse_list(s: &str) -> Result<Vec<Self>, String> {
        let is_hex = |x: &str| !x.is_empty() && x.bytes().all(|b| b.is_ascii_hexdigit());
        s.split(',')
            .map(str::trim)
            .filter(|x| !x.is_empty())
            .map(|x| {
                if x == "v6" {
                    Ok(Self::V6)
                } else if let Some(h) = x.strip_prefix("knob:") {
                    is_hex(h).then(|| Self::Knob(h.to_ascii_lowercase())).ok_or(format!("bad knob hash {h:?}"))
                } else if let Some(g) = x.strip_prefix("gen:") {
                    g.parse().map(Self::Gen).map_err(|_| format!("bad generation {g:?}"))
                } else if is_hex(x) && x.len() >= 7 {
                    Ok(Self::Build(x.to_ascii_lowercase()))
                } else {
                    Err(format!("bad purge spec {x:?} (want <sha>, knob:<hash>, gen:<n> or v6)"))
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
    /// Bad header, wrong size, wrong namespace or name: unlinked.
    pub invalid: u64,
    pub unreachable: u64,
    pub missing_ancestor: u64,
    pub purged: u64,
    /// Crash between a demotion's truncate and its header rewrite.
    pub repaired_demotions: u64,
    pub truncated: u64,
    /// Files in our directories that are not ours (left alone).
    pub foreign: u64,
    pub kept_inactive: Option<(String, u64)>,
    pub trashed_namespaces: u64,
    /// The running effective generation has no file in the store yet
    /// (`kv.gen_new`, 4.4).
    pub gen_new: bool,
    pub elapsed_ms: u64,
}

/// The result of opening a namespace: the index plus the file work the IO
/// thread must do once it runs.
pub struct Opened {
    pub index: Index,
    pub jobs: Vec<IoJob>,
    pub unlinks: Vec<(PathBuf, u64)>,
    pub report: ScanReport,
}

const LAST_ACTIVE: &str = "last_active";

fn read_last_active(dir: &Path) -> u64 {
    fs::read_to_string(dir.join(LAST_ACTIVE)).ok().and_then(|s| s.trim().parse().ok()).unwrap_or(0)
}

fn is_ns16(name: &str) -> bool {
    name.len() == 16 && name.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// Total bytes of the regular files under `dir`.
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
}

struct ScannedTail {
    h: TailHeader,
    size: u64,
    mtime: u64,
}

/// Open the namespace `ns` under `root` (= `kvstore-v1/`): GC the other
/// namespaces, clear `tmp/`, scan, rebuild.
pub fn open_namespace(
    root: &Path,
    ns: &Key,
    cap_bytes: u64,
    expect_stores: &[StoreRows],
    purge: &[PurgeSpec],
    current_gen: GenPair,
    now: u64,
) -> io::Result<Opened> {
    let t0 = Instant::now();
    let mut report = ScanReport::default();
    let mut jobs = Vec::new();
    let mut unlinks: Vec<(PathBuf, u64)> = Vec::new();
    let trash = root.join("trash");
    mkdir_private(root)?;
    mkdir_private(&trash)?;
    let mut seq = 0u32;

    // Namespace GC (4.4): keep the active namespace and the most recently
    // active other one (for rollback); rename the rest into trash/.
    let active = keys::ns16(ns);
    let mut others: Vec<(u64, String)> = Vec::new();
    for e in fs::read_dir(root)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name != active && is_ns16(&name) && e.file_type().is_ok_and(|t| t.is_dir()) {
            others.push((read_last_active(&e.path()), name));
        }
    }
    others.sort_unstable_by(|a, b| b.cmp(a));
    for (i, (_, name)) in others.iter().enumerate() {
        if i == 0 {
            let bytes = dir_bytes(&root.join(name));
            report.kept_inactive = Some((name.clone(), bytes));
        } else {
            rename_to_trash(&trash, &root.join(name), &mut seq)?;
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
    fs::write(dirs.base.join(LAST_ACTIVE), format!("{now}\n"))?;
    // Writes in flight at a crash: the whole tmp/ goes (one rename).
    let tmp_has_files = fs::read_dir(dirs.tmp())?.next().is_some();
    if tmp_has_files {
        let to = rename_to_trash(&trash, &dirs.tmp(), &mut seq)?;
        jobs.push(IoJob::Trash { from: to });
        dirs.create()?;
    }

    // Header scan.
    let mut chunks: HashMap<Key, ScannedChunk> = HashMap::new();
    let mut tails: Vec<ScannedTail> = Vec::new();
    let mut paths: HashMap<Key, PathBuf> = HashMap::new();
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
                let Ok(meta) = f.metadata() else { continue };
                let size = meta.len();
                let in_fan = d.file_name().to_string_lossy() == keys::hex(&key)[..2];
                let scanned = if ext == "kvc" {
                    scan_chunk(&path, size, ns, &key, expect_stores).map(|h| {
                        chunks.insert(key, ScannedChunk { h, size });
                    })
                } else {
                    scan_tail(&path, size, ns, &key, &mut jobs, &mut report).map(|h| {
                        tails.push(ScannedTail { h, size: 0, mtime: mtime_secs(&meta) });
                    })
                };
                match scanned {
                    Ok(()) if in_fan => {
                        paths.insert(key, path);
                    }
                    _ => {
                        report.invalid += 1;
                        chunks.remove(&key);
                        tails.retain(|t| t.h.key != key);
                        unlinks.push((path, size));
                    }
                }
            }
        }
    }
    // A tail's indexed size is its size after the queued repairs.
    for t in &mut tails {
        t.size = t.h.file_len();
    }

    // Purge (4.4): matching files go; everything beneath a purged chunk then
    // fails reachability below.
    if !purge.is_empty() {
        let hit = |b: &BuildId, g: &GenPair| purge.iter().any(|p| p.matches(b, g));
        chunks.retain(|k, c| {
            let keep = !hit(&c.h.build, &c.h.gen);
            if !keep {
                report.purged += 1;
                unlinks.push((paths[k].clone(), c.size));
            }
            keep
        });
        tails.retain(|t| {
            let keep = !hit(&t.h.build, &t.h.gen);
            if !keep {
                report.purged += 1;
                unlinks.push((paths[&t.h.key].clone(), t.size));
            }
            keep
        });
    }

    // Rebuild: chunks in breadth-first order from the root, so every chunk
    // finds its parent; anything not reached is unreachable.
    let mut index = Index::new(keys::chain_root(ns), cap_bytes);
    let mut by_parent: HashMap<Key, Vec<Key>> = HashMap::new();
    for (k, c) in &chunks {
        by_parent.entry(c.h.parent).or_default().push(*k);
    }
    let mut frontier = vec![(*index.root(), u32::MAX)];
    while let Some((node, k_node)) = frontier.pop() {
        for child in by_parent.remove(&node).unwrap_or_default() {
            let c = &chunks[&child];
            if c.h.k != k_node.wrapping_add(1) {
                continue; // misnumbered: left in `chunks`, dropped below
            }
            index.insert_chunk(ChunkInsert {
                key: child,
                parent: node,
                k: c.h.k,
                bytes: c.size,
                created: c.h.created,
                gen: c.h.gen,
                job: None,
            });
            frontier.push((child, c.h.k));
        }
    }
    for (k, c) in &chunks {
        if index.chunk(k).is_none() {
            report.unreachable += 1;
            unlinks.push((paths[k].clone(), c.size));
        }
    }
    for t in &tails {
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
            created: h.created,
            gen: h.gen,
            hits: h.hits,
            ancestors: Vec::new(),
            job: None,
        };
        if index.load_tail(ins, t.mtime).is_err() {
            report.missing_ancestor += 1;
            unlinks.push((paths[&h.key].clone(), fs::metadata(&paths[&h.key]).map(|m| m.len()).unwrap_or(t.size)));
        }
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
    Ok(Opened { index, jobs, unlinks, report })
}

fn read_block(path: &Path, len: usize, size: u64) -> Result<Vec<u8>, format::FormatError> {
    if size < len as u64 {
        return Err(format::FormatError::Short { want: len as u64, got: size });
    }
    let f = File::open(path)?;
    let mut b = vec![0u8; len];
    f.read_exact_at(&mut b, 0)?;
    Ok(b)
}

fn scan_chunk(path: &Path, size: u64, ns: &Key, key: &Key, expect: &[StoreRows]) -> Result<ChunkHeader, format::FormatError> {
    let h = ChunkHeader::decode(&read_block(path, CHUNK_HEADER_LEN, size)?)?;
    if &h.ns != ns {
        return Err(format::FormatError::Namespace);
    }
    if &h.key != key {
        return Err(format::FormatError::Key);
    }
    if h.stores != expect {
        return Err(format::FormatError::BadField("store shape"));
    }
    if h.k.checked_mul(C).is_none() {
        return Err(format::FormatError::BadField("k"));
    }
    // Chunks are never mutated in place: any other size is a torn file.
    if size != h.file_len() {
        return Err(format::FormatError::Short { want: h.file_len(), got: size });
    }
    Ok(h)
}

/// Returns the header as it will be once the queued repair ran.
fn scan_tail(path: &Path, size: u64, ns: &Key, key: &Key, jobs: &mut Vec<IoJob>, report: &mut ScanReport) -> Result<TailHeader, format::FormatError> {
    let h = TailHeader::decode(&read_block(path, TAIL_HEADER_LEN, size)?)?;
    if &h.ns != ns {
        return Err(format::FormatError::Namespace);
    }
    if &h.key != key {
        return Err(format::FormatError::Key);
    }
    let (e_end, full_end) = (h.e_end(), h.file_len());
    if size == full_end {
        return Ok(h);
    }
    if size < e_end {
        return Err(format::FormatError::Short { want: e_end, got: size });
    }
    if h.kind == TailKind::Full && size < full_end && !h.anchor {
        // The demotion's truncate ran, its header rewrite did not (or section
        // D is torn): finish the demotion. Truncating to e_end also covers a
        // torn D.
        jobs.push(IoJob::Demote { path: path.to_path_buf(), key: Some(h.key), hits: None });
        report.repaired_demotions += 1;
        return Ok(h.demoted(h.hits));
    }
    if size > full_end {
        jobs.push(IoJob::Truncate { path: path.to_path_buf(), len: full_end });
        report.truncated += 1;
        return Ok(h);
    }
    Err(format::FormatError::Short { want: full_end, got: size })
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
fn recompute_path_last_used(index: &mut Index, paths: &HashMap<Key, PathBuf>) {
    // Subtree maxima, children before parents.
    let root = *index.root();
    let mut order = vec![root];
    let mut i = 0;
    while i < order.len() {
        let n = order[i];
        order.extend(index.children(&n).iter().filter(|c| index.chunk(c).is_some()).copied());
        i += 1;
    }
    let mut sub_max: HashMap<Key, u64> = HashMap::with_capacity(order.len());
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
        let ids_cache: HashMap<Key, Option<(u32, Vec<i32>, Vec<keys::ImageRecord>)>> = if unaligned {
            ts.iter()
                .chain(index.children(n).iter())
                .map(|k| {
                    let is_tail = index.tail(k).is_some();
                    (*k, paths.get(k).and_then(|p| format::read_ids(p, is_tail).ok()))
                })
                .collect()
        } else {
            HashMap::new()
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
        let v = PurgeSpec::parse_list("0123456, knob:ab12,gen:7,v6").unwrap();
        assert_eq!(v, vec![PurgeSpec::Build("0123456".into()), PurgeSpec::Knob("ab12".into()), PurgeSpec::Gen(7), PurgeSpec::V6]);
        assert!(PurgeSpec::parse_list("012").is_err());
        assert!(PurgeSpec::parse_list("gen:x").is_err());
        let b = BuildId::parse("0123456789abcdef0123456789abcdef01234567").unwrap();
        let g = GenPair { gen: 7, knob: [0xab; 16] };
        assert!(v[0].matches(&b, &g));
        assert!(v[1].matches(&b, &GenPair { gen: 1, knob: [0xab, 0x12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] }));
        assert!(v[2].matches(&b, &g));
        assert!(!v[3].matches(&b, &g));
        assert!(v[3].matches(&BuildId::V6, &g));
        assert!(!v[0].matches(&BuildId::V6, &g), "a sha purge never matches v6 backfill");
    }
}
