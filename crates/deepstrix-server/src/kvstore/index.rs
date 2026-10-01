//! The in-memory index: the chunk trie, tails, refcounts, pins, the eviction
//! order, demotion and thinning (design 6.1, 6.2, 8, 9.4).
//!
//! It does no IO. Every change to a file is returned as an [`Action`] that the
//! store hands to the IO thread, so the whole policy is testable in memory and
//! the scheduler thread never blocks on the disk (8.3: "index removal on the
//! scheduler thread, unlinks on the IO thread").
//!
//! Model (8.1):
//! - a chunk node is keyed by its chain key; its `parent` is chain_k (the root
//!   for chunk 0). A tail hangs off node b = ⌊T/C⌋ (its `base`, chain_b) and its
//!   PATH is the chunks chain_1..chain_b.
//! - `refs` of a chunk = the number of tails whose path contains it. A chunk
//!   goes when that reaches 0, with the last tail beneath it ("the chunk bytes
//!   go only with the last tail beneath them").
//! - `pins` of a chunk = the number of pin records anchored at it OR BELOW it.
//!   Pins are PATH pins (6.5, 9.4): a pinned chunk keeps its ancestors too,
//!   otherwise an in-flight job's restored prefix could be cascaded away under
//!   the chunks it is writing, and its next tail would find a hole. Refs and
//!   pins are therefore both monotone up a path, so a deletable chunk's whole
//!   subtree is deletable.
//! - orphans = chunks with refs 0 and pins 0 that no tail removal deleted:
//!   left by a killed or cancelled job (or found by the startup scan). They are
//!   the first thing eviction takes (8.3 order 1).
//!
//! Completions are asynchronous: a job may end (`release_job`) while writes of
//! its are still in flight. Such late completions are indexed as unowned
//! (no pin), and a failed job's late chunks go at once if nothing references
//! them (9.4); the index remembers finished jobs until the store forgets them.
//!
//! Counter drift (a refcount or byte total that would go negative) never
//! panics the scheduler thread: it saturates, logs `kv.invariant`, and sets
//! [`Index::needs_rescan`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::format::{GenPair, TailKind, TailOrigin};
use super::keys::{self, images_in, ImageRecord, Key, KeyMap, KeySet};
use super::{C, DEMOTE_KEEP_FULL, ENC_MIN_SUFFIX, K, MIN_RESTORE_T, SCORE_HIT_S};

/// A prefill job, as the owner of chunk pins and plan pins (9.4, 6.5). Ids
/// must not be reused while the store remembers the job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct JobId(pub u64);

/// A pin record (a restore plan's tail + path, or one chunk a job wrote).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct PinId(u64);

/// Index into the generation-pair table (`kv.store gens=`).
pub type GenId = u16;

#[derive(Debug, Clone)]
pub struct ChunkEntry {
    /// chain_k (the root for k = 0).
    pub parent: Key,
    pub k: u32,
    pub bytes: u64,
    pub created: u64,
    pub gen: GenId,
    /// The stored payload's blake3 (`kv.dedup_mismatch` compares a recomputed
    /// chunk against it, 7.1 E2).
    pub payload_hash: [u8; 32],
    pub refs: u32,
    pub pins: u32,
}

#[derive(Debug, Clone)]
pub struct TailEntry {
    /// chain_⌊t/C⌋.
    pub base: Key,
    pub t: u32,
    pub kind: TailKind,
    pub origin: TailOrigin,
    pub anchor: bool,
    pub demoted: bool,
    /// Restores of this tail itself; persisted (8.3).
    pub hits: u32,
    /// This tail's own last restore or insert; persisted as the file mtime.
    pub last_used: u64,
    /// max(last_used) over this tail and every tail below it on the same path.
    /// Memory only: propagated touches never persist and never count as hits.
    pub path_last_used: u64,
    pub created: u64,
    /// Current file size.
    pub bytes: u64,
    /// Section D's bytes (0 for an encoder tail): what a demotion frees.
    pub sec_d_bytes: u64,
    pub gen: GenId,
    pub pins: u32,
    /// Demotion decided while pinned; applied at the last unpin (8.2).
    pub deferred_demote: bool,
    /// Thinning decided while pinned: superseded by this tail in its K-window.
    pub thin_by: Option<Key>,
}

/// Why something left the index (`kv.evict why=`, shadow reason codes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Why {
    Cap,
    Orphan,
    /// A failed job's unreferenced chunks (9.4).
    Failed,
    /// A checksum, token-id, shape or ABI failure on read (6.2, 9.4).
    Corrupt,
    /// An inactive namespace evicted whole (4.4).
    Namespace,
    Thinned,
    /// `V41_KV_STORE_PURGE_BUILD` (4.4).
    Purge,
    /// A tail whose chunk path is incomplete (9.5).
    MissingAncestor,
    /// A chunk whose parent chain does not reach the root (startup).
    Unreachable,
    /// A tail removed because the cascade of another removal took its path.
    Cascade,
}

impl Why {
    pub fn as_str(self) -> &'static str {
        match self {
            Why::Cap => "cap",
            Why::Orphan => "orphan",
            Why::Failed => "failed",
            Why::Corrupt => "corrupt",
            Why::Namespace => "namespace",
            Why::Thinned => "thinned",
            Why::Purge => "purge",
            Why::MissingAncestor => "missing_ancestor",
            Why::Unreachable => "unreachable",
            Why::Cascade => "cascade",
        }
    }
}

/// File work for the IO thread (all mutations of existing files run there in
/// order, 4.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    UnlinkChunk { key: Key, bytes: u64, why: Why },
    UnlinkTail { key: Key, bytes: u64, why: Why },
    /// `ftruncate` section D, then rewrite the header with `hits` (8.2, 4.5).
    Demote { key: Key, hits: u32 },
    /// pwrite `hits`, set the mtime to `last_used` (8.3).
    Touch { key: Key, hits: u32, last_used: u64 },
    /// Rename the inactive namespace into `trash/` (4.4).
    TrashNamespace { ns16: String, bytes: u64 },
}

/// One removal, for `kv.evict` and the shadow reason codes (11.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Removal {
    pub what: Removed,
    pub key: Key,
    /// Bytes of the entry itself.
    pub bytes: u64,
    /// Bytes of chunks that went with it.
    pub cascaded: u64,
    pub age_s: u64,
    pub hits: u32,
    pub why: Why,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Removed {
    Tail { t: u32, kind: TailKind },
    Chunk { k: u32 },
    Namespace,
}

/// The request to index a chunk whose file is now on disk.
#[derive(Debug, Clone)]
pub struct ChunkInsert {
    pub key: Key,
    pub parent: Key,
    pub k: u32,
    pub bytes: u64,
    pub created: u64,
    pub gen: GenPair,
    pub payload_hash: [u8; 32],
    /// Pinned for this job until the next tail on its path completes (9.4).
    pub job: Option<JobId>,
}

/// The request to index a tail whose file is now on disk.
#[derive(Debug, Clone)]
pub struct TailInsert {
    pub key: Key,
    pub base: Key,
    pub t: u32,
    pub kind: TailKind,
    pub origin: TailOrigin,
    pub anchor: bool,
    pub demoted: bool,
    pub bytes: u64,
    pub sec_d_bytes: u64,
    /// The insert time: the header's `created`, the file's mtime, and so this
    /// tail's `last_used` (8.3: the mtime IS the persisted last_used).
    pub created: u64,
    pub gen: GenPair,
    pub hits: u32,
    /// The tails on this tail's path (8.2: "the job's walk matched them"),
    /// computed by the store from the request ([`Index::ancestors_on_path`]).
    /// Drives demotion, thinning and touch propagation.
    pub ancestors: Vec<Key>,
    pub job: Option<JobId>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refused {
    /// A chunk on the tail's path is not indexed (dropped, evicted or never
    /// written): the tail is unreachable and its file must go.
    MissingAncestor,
}

#[derive(Debug, Clone, Copy)]
struct PinRec {
    /// The chunk the path pin is anchored at; `None` once that chunk was
    /// force-removed (the contribution was already taken off its ancestors).
    chunk: Option<Key>,
    tail: Option<Key>,
    /// The job that owns the record (released by `release_job`).
    job: Option<JobId>,
}

/// A tail the walk matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WalkTail {
    pub key: Key,
    pub t: u32,
    pub kind: TailKind,
    pub anchor: bool,
}

/// The result of a walk over one request (6.1).
#[derive(Debug, Clone, Default)]
pub struct Walk {
    /// chain_1..chain_m, the matched chunk keys in order.
    pub chunks: Vec<Key>,
    /// Every matched tail, by increasing t.
    pub tails: Vec<WalkTail>,
}

impl Walk {
    /// The matched tails strictly below `t`.
    pub fn ancestors_below(&self, t: u32) -> Vec<Key> {
        self.tails.iter().filter(|w| w.t < t).map(|w| w.key).collect()
    }
}

/// Selection (6.2): the deepest usable tail; a full tail wins a tie.
///
/// - full: t ≤ L − 1, or t = L when the request ends with a trailing marker
///   (the marker is the one row prefilled);
/// - encoder: L − t > 128, because only then does `prefill_job_finish` discard
///   the decoder windows anyway (5.3);
/// - and t ≥ 64.
pub fn select(walk: &Walk, l: u32, trailing_marker: bool) -> Option<WalkTail> {
    walk.tails
        .iter()
        .copied()
        .filter(|w| usable(w.kind, w.t, l, trailing_marker))
        .max_by_key(|w| (w.t, w.kind))
}

pub fn usable(kind: TailKind, t: u32, l: u32, trailing_marker: bool) -> bool {
    if t < MIN_RESTORE_T || t > l {
        return false;
    }
    match kind {
        TailKind::Full => t < l || trailing_marker,
        TailKind::Enc => l - t > ENC_MIN_SUFFIX,
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GenStat {
    pub files: u64,
    pub bytes: u64,
}

pub struct Index {
    root: Key,
    chunks: KeyMap<ChunkEntry>,
    /// parent key (the root included) -> child chunk keys. A child may arrive
    /// before its parent (a dropped write re-enqueued behind it, 9.4): it is
    /// listed here, unreachable for walks until the parent lands.
    children: KeyMap<Vec<Key>>,
    tails: KeyMap<TailEntry>,
    /// node (chain_b) -> the tails hanging off it.
    tails_at: KeyMap<Vec<Key>>,
    /// (created, k, key): oldest first, and a parent before its children.
    orphans: BTreeSet<(u64, u32, Key)>,
    by_score: BTreeSet<(u64, Key)>,
    pins: HashMap<PinId, PinRec>,
    job_pins: HashMap<JobId, Vec<PinId>>,
    /// Jobs that ended, with `failed`, until the store forgets them (their
    /// writes drained). Their late completions are indexed unowned.
    finished: HashMap<JobId, bool>,
    next_pin: u64,
    gens: Vec<GenPair>,
    gen_stats: Vec<GenStat>,
    chunk_bytes: u64,
    tail_bytes: u64,
    /// Kept inactive namespaces (4.4): (ns16, bytes), evicted first, whole.
    inactive: Vec<(String, u64)>,
    cap_bytes: u64,
    /// A counter would have gone negative: the incremental state drifted.
    drift: bool,
    actions: Vec<Action>,
    removals: Vec<Removal>,
}

/// `a -= b`, saturating; true on underflow.
fn sub_sat(a: &mut u64, b: u64) -> bool {
    let under = b > *a;
    *a = a.saturating_sub(b);
    under
}

impl Index {
    pub fn new(root: Key, cap_bytes: u64) -> Self {
        Self {
            root,
            chunks: KeyMap::default(),
            children: KeyMap::default(),
            tails: KeyMap::default(),
            tails_at: KeyMap::default(),
            orphans: BTreeSet::new(),
            by_score: BTreeSet::new(),
            pins: HashMap::new(),
            job_pins: HashMap::new(),
            finished: HashMap::new(),
            next_pin: 1,
            gens: Vec::new(),
            gen_stats: Vec::new(),
            chunk_bytes: 0,
            tail_bytes: 0,
            inactive: Vec::new(),
            cap_bytes,
            drift: false,
            actions: Vec::new(),
            removals: Vec::new(),
        }
    }

    pub fn root(&self) -> &Key {
        &self.root
    }
    pub fn chunk(&self, key: &Key) -> Option<&ChunkEntry> {
        self.chunks.get(key)
    }
    pub fn tail(&self, key: &Key) -> Option<&TailEntry> {
        self.tails.get(key)
    }
    pub fn n_chunks(&self) -> usize {
        self.chunks.len()
    }
    pub fn n_tails(&self) -> usize {
        self.tails.len()
    }
    pub fn n_orphans(&self) -> usize {
        self.orphans.len()
    }
    pub fn chunk_bytes(&self) -> u64 {
        self.chunk_bytes
    }
    pub fn tail_bytes(&self) -> u64 {
        self.tail_bytes
    }
    pub fn inactive_bytes(&self) -> u64 {
        self.inactive.iter().map(|(_, b)| b).sum()
    }
    /// Everything counted against the global cap (4.4: one cap over every
    /// namespace).
    pub fn total_bytes(&self) -> u64 {
        self.chunk_bytes + self.tail_bytes + self.inactive_bytes()
    }
    pub fn cap_bytes(&self) -> u64 {
        self.cap_bytes
    }
    pub fn set_cap_bytes(&mut self, cap: u64) {
        self.cap_bytes = cap;
    }
    pub fn set_inactive(&mut self, ns: Vec<(String, u64)>) {
        self.inactive = ns;
    }
    pub fn tail_keys(&self) -> impl Iterator<Item = &Key> {
        self.tails.keys()
    }
    pub fn chunk_keys(&self) -> impl Iterator<Item = &Key> {
        self.chunks.keys()
    }
    pub fn tails_at(&self, node: &Key) -> &[Key] {
        self.tails_at.get(node).map(|v| v.as_slice()).unwrap_or(&[])
    }
    pub fn children(&self, node: &Key) -> &[Key] {
        self.children.get(node).map(|v| v.as_slice()).unwrap_or(&[])
    }
    /// Files and bytes by effective generation pair (`kv.store gens=`).
    pub fn gens(&self) -> impl Iterator<Item = (GenPair, GenStat)> + '_ {
        self.gens.iter().copied().zip(self.gen_stats.iter().copied()).filter(|(_, s)| s.files > 0)
    }
    /// A counter drifted (saturated instead of going negative): the store
    /// should rebuild from disk (a restart's scan) at a convenient time.
    pub fn needs_rescan(&self) -> bool {
        self.drift
    }
    /// The job ended and the store has not forgotten it yet; `Some(failed)`.
    pub fn finished(&self, job: JobId) -> Option<bool> {
        self.finished.get(&job).copied()
    }
    /// Pin records the job still owns.
    pub fn job_pin_count(&self, job: JobId) -> usize {
        self.job_pins.get(&job).map_or(0, |v| v.len())
    }

    /// File work produced since the last call.
    pub fn take_actions(&mut self) -> Vec<Action> {
        std::mem::take(&mut self.actions)
    }
    /// Removals since the last call (telemetry).
    pub fn take_removals(&mut self) -> Vec<Removal> {
        std::mem::take(&mut self.removals)
    }

    fn flag_drift(&mut self, what: &'static str) {
        if !self.drift {
            tracing::error!(what, "kv.invariant counter drift; saturated, a rescan is due");
        }
        self.drift = true;
    }

    fn intern_gen(&mut self, g: GenPair) -> GenId {
        if let Some(i) = self.gens.iter().position(|x| *x == g) {
            return i as GenId;
        }
        assert!(self.gens.len() < GenId::MAX as usize, "too many generation pairs");
        self.gens.push(g);
        self.gen_stats.push(GenStat::default());
        (self.gens.len() - 1) as GenId
    }

    fn gen_add(&mut self, g: GenId, files: i64, bytes: i64) {
        let s = &mut self.gen_stats[g as usize];
        let (f, b) = (s.files as i64 + files, s.bytes as i64 + bytes);
        s.files = f.max(0) as u64;
        s.bytes = b.max(0) as u64;
        if f < 0 || b < 0 {
            self.flag_drift("generation stats");
        }
    }

    fn sub_chunk_bytes(&mut self, b: u64) {
        if sub_sat(&mut self.chunk_bytes, b) {
            self.flag_drift("chunk bytes");
        }
    }

    fn sub_tail_bytes(&mut self, b: u64) {
        if sub_sat(&mut self.tail_bytes, b) {
            self.flag_drift("tail bytes");
        }
    }

    /// A job that ended is no owner any more: its late writes go in unowned.
    fn live_job(&self, job: Option<JobId>) -> Option<JobId> {
        job.filter(|j| !self.finished.contains_key(j))
    }

    // ---- chunks -------------------------------------------------------

    /// Is this chunk on a path from the root (not detached)?
    pub fn reachable(&self, key: &Key) -> bool {
        let mut node = *key;
        while node != self.root {
            match self.chunks.get(&node) {
                Some(c) => node = c.parent,
                None => return false,
            }
        }
        true
    }

    /// The chunk keys of a path ending at node `base` = chain_b, in order
    /// chain_1..chain_b. `None` if any chunk is missing or misnumbered.
    pub fn chunk_path(&self, base: &Key, b: u32) -> Option<Vec<Key>> {
        let mut out = Vec::with_capacity(b as usize);
        let mut node = *base;
        for k in (0..b).rev() {
            let c = self.chunks.get(&node)?;
            if c.k != k {
                return None;
            }
            out.push(node);
            node = c.parent;
        }
        (node == self.root).then(|| {
            out.reverse();
            out
        })
    }

    /// Add `d_refs` / `d_pins` to `from` and every existing ancestor.
    fn add_path(&mut self, from: &Key, d_refs: i64, d_pins: i64) {
        let mut node = *from;
        let mut under = false;
        while node != self.root {
            let Some(c) = self.chunks.get_mut(&node) else { break };
            let (r, p) = (c.refs as i64 + d_refs, c.pins as i64 + d_pins);
            under |= r < 0 || p < 0;
            c.refs = r.max(0) as u32;
            c.pins = p.max(0) as u32;
            let parent = c.parent;
            self.orphan_sync(&node);
            node = parent;
        }
        if under {
            self.flag_drift("chunk refs/pins");
        }
    }

    fn orphan_sync(&mut self, key: &Key) {
        let Some(c) = self.chunks.get(key) else { return };
        let item = (c.created, c.k, *key);
        if c.refs == 0 && c.pins == 0 {
            self.orphans.insert(item);
        } else {
            self.orphans.remove(&item);
        }
    }

    /// Index a chunk whose file is on disk. Returns false if it was already
    /// indexed (the caller's dedup failed; the first writer wins, E5).
    ///
    /// A chunk of a job that already ended is indexed unowned; if that job
    /// FAILED and nothing references or pins the chunk, it goes at once (9.4).
    pub fn insert_chunk(&mut self, c: ChunkInsert, now: u64) -> bool {
        if self.chunks.contains_key(&c.key) || c.key == self.root {
            return false;
        }
        let late_failed = c.job.and_then(|j| self.finished.get(&j).copied()).unwrap_or(false);
        let job = self.live_job(c.job);
        let gen = self.intern_gen(c.gen);
        // Detached children that arrived before this chunk (9.4) carry pins.
        let pins_below: u32 = self.children(&c.key).iter().filter_map(|k| self.chunks.get(k)).map(|e| e.pins).sum();
        self.chunks.insert(
            c.key,
            ChunkEntry {
                parent: c.parent,
                k: c.k,
                bytes: c.bytes,
                created: c.created,
                gen,
                payload_hash: c.payload_hash,
                refs: 0,
                pins: pins_below,
            },
        );
        self.children.entry(c.parent).or_default().push(c.key);
        self.chunk_bytes += c.bytes;
        self.gen_add(gen, 1, c.bytes as i64);
        if pins_below > 0 {
            let parent = c.parent;
            self.add_path(&parent, 0, pins_below as i64);
        }
        self.orphan_sync(&c.key);
        if let Some(job) = job {
            let id = self.new_pin(PinRec { chunk: Some(c.key), tail: None, job: Some(job) });
            self.add_path(&c.key, 0, 1);
            self.job_pins.entry(job).or_default().push(id);
        } else if late_failed && self.chunks.get(&c.key).is_some_and(|e| e.refs == 0 && e.pins == 0) {
            self.delete_subtree(&c.key, Why::Failed, now);
        }
        true
    }

    fn new_pin(&mut self, rec: PinRec) -> PinId {
        let id = PinId(self.next_pin);
        self.next_pin += 1;
        self.pins.insert(id, rec);
        id
    }

    /// The subtree rooted at `key` (itself first), over attached children.
    fn subtree(&self, key: &Key) -> Vec<Key> {
        let mut out = vec![*key];
        let mut i = 0;
        while i < out.len() {
            let k = out[i];
            out.extend(self.children(&k).iter().filter(|c| self.chunks.contains_key(*c)).copied());
            i += 1;
        }
        out
    }

    /// Delete a chunk with refs 0 and pins 0 and its whole subtree (all of
    /// which then have refs 0 and pins 0 too). Returns the bytes freed.
    fn delete_subtree(&mut self, key: &Key, why: Why, now: u64) -> u64 {
        let Some(top) = self.chunks.get(key) else { return 0 };
        if top.refs != 0 || top.pins != 0 {
            self.flag_drift("deleting a referenced or pinned chunk");
            return 0;
        }
        let (top_k, top_bytes, top_created, parent) = (top.k, top.bytes, top.created, top.parent);
        let mut freed = 0;
        for k in self.subtree(key) {
            let Some(c) = self.chunks.remove(&k) else { continue };
            self.orphans.remove(&(c.created, c.k, k));
            self.children.remove(&k);
            self.sub_chunk_bytes(c.bytes);
            self.gen_add(c.gen, -1, -(c.bytes as i64));
            freed += c.bytes;
            self.actions.push(Action::UnlinkChunk { key: k, bytes: c.bytes, why });
        }
        if let Some(sib) = self.children.get_mut(&parent) {
            sib.retain(|c| c != key);
            if sib.is_empty() {
                self.children.remove(&parent);
            }
        }
        self.removals.push(Removal {
            what: Removed::Chunk { k: top_k },
            key: *key,
            bytes: top_bytes,
            cascaded: freed - top_bytes,
            age_s: now.saturating_sub(top_created),
            hits: 0,
            why,
        });
        freed
    }

    /// Remove a chunk regardless of refs and pins (corrupt file, purge): every
    /// tail beneath it goes, and the pins anchored in its subtree are
    /// taken off its ancestors and their records retired.
    pub fn remove_chunk(&mut self, key: &Key, why: Why, now: u64) {
        if !self.chunks.contains_key(key) {
            return;
        }
        let sub = self.subtree(key);
        let beneath: Vec<Key> = sub.iter().flat_map(|n| self.tails_at(n).to_vec()).collect();
        for t in beneath {
            self.remove_tail(&t, why, now);
        }
        // remove_tail's cascade may already have taken the chunk.
        let Some(c) = self.chunks.get(key) else { return };
        let (pins, parent) = (c.pins, c.parent);
        if pins > 0 {
            self.add_path(&parent, 0, -(pins as i64));
            let set: KeySet = self.subtree(key).into_iter().collect();
            for rec in self.pins.values_mut() {
                if rec.chunk.is_some_and(|c| set.contains(&c)) {
                    rec.chunk = None;
                }
            }
            for k in &set {
                if let Some(c) = self.chunks.get_mut(k) {
                    c.pins = 0;
                }
            }
        }
        let mut stray_refs = false;
        for k in self.subtree(key) {
            if let Some(c) = self.chunks.get_mut(&k) {
                stray_refs |= c.refs != 0;
                c.refs = 0;
            }
        }
        if stray_refs {
            self.flag_drift("refs beneath a removed chunk");
        }
        self.delete_subtree(key, why, now);
    }

    // ---- tails --------------------------------------------------------

    fn score_of(e: &TailEntry) -> u64 {
        e.path_last_used + (SCORE_HIT_S as f64 * (1.0 + e.hits as f64).log2()) as u64
    }

    /// Apply `f` to a tail, keeping the score order in sync.
    fn rescore(&mut self, key: &Key, f: impl FnOnce(&mut TailEntry)) {
        let Some(e) = self.tails.get_mut(key) else { return };
        let old = Self::score_of(e);
        f(e);
        let new = Self::score_of(e);
        if old != new {
            self.by_score.remove(&(old, *key));
            self.by_score.insert((new, *key));
        }
    }

    /// Index a tail whose file is on disk (and demote / thin its path).
    ///
    /// Refused when a chunk on its path is missing; the job's pins on the
    /// path are released either way (the tail was its "next tail", 9.4).
    pub fn insert_tail(&mut self, ins: TailInsert, now: u64) -> Result<(), Refused> {
        let b = ins.t / C;
        let job = self.live_job(ins.job);
        let path = self.chunk_path(&ins.base, b);
        // The tail's completion releases the job's chunk pins on its path:
        // every chunk the job pinned below b is on it (a job writes one request).
        if let Some(job) = job {
            self.release_job_pins(job, now, |idx, rec| {
                rec.tail.is_none() && rec.chunk.map_or(true, |c| idx.chunks.get(&c).map_or(true, |e| e.k < b))
            });
        }
        let path = path.ok_or(Refused::MissingAncestor)?;
        self.place_tail(&ins, b);
        // Inserting touches the tail and every ancestor on its path (8.3).
        let anc = self.valid_ancestors(&ins, &path);
        self.propagate(&ins.key, &anc, ins.created);
        if ins.kind == TailKind::Full {
            self.demote_and_thin(&ins.key, &anc, now);
        }
        Ok(())
    }

    /// Insert or replace the entry, with path refs; no policy. The startup
    /// scan uses this directly (the disk state is the state).
    fn place_tail(&mut self, ins: &TailInsert, b: u32) {
        let gen = self.intern_gen(ins.gen);
        let last_used = ins.created;
        if let Some(old) = self.tails.get(&ins.key).cloned() {
            // Same key: a full tail replacing an encoder tail at the same T
            // (5.1). The file was renamed over; adopt the new one, keep hits,
            // pins and the anchor flag.
            self.sub_tail_bytes(old.bytes);
            self.tail_bytes += ins.bytes;
            self.gen_add(old.gen, -1, -(old.bytes as i64));
            self.gen_add(gen, 1, ins.bytes as i64);
            self.rescore(&ins.key, |e| {
                e.kind = ins.kind;
                e.origin = ins.origin;
                e.anchor |= ins.anchor;
                e.demoted = ins.demoted && !e.anchor;
                e.bytes = ins.bytes;
                e.sec_d_bytes = ins.sec_d_bytes;
                e.gen = gen;
                e.hits = e.hits.max(ins.hits);
                e.last_used = e.last_used.max(last_used);
                e.path_last_used = e.path_last_used.max(last_used);
                e.deferred_demote = false;
                e.thin_by = None;
            });
            return;
        }
        let e = TailEntry {
            base: ins.base,
            t: ins.t,
            kind: ins.kind,
            origin: ins.origin,
            anchor: ins.anchor,
            demoted: ins.demoted && !ins.anchor,
            hits: ins.hits,
            last_used,
            path_last_used: last_used,
            created: ins.created,
            bytes: ins.bytes,
            sec_d_bytes: ins.sec_d_bytes,
            gen,
            pins: 0,
            deferred_demote: false,
            thin_by: None,
        };
        self.by_score.insert((Self::score_of(&e), ins.key));
        self.tails.insert(ins.key, e);
        self.tails_at.entry(ins.base).or_default().push(ins.key);
        self.tail_bytes += ins.bytes;
        self.gen_add(gen, 1, ins.bytes as i64);
        if b > 0 {
            self.add_path(&ins.base, 1, 0);
        }
    }

    /// Startup scan: index a tail exactly as found on disk; `created` carries
    /// the file's mtime (its persisted last_used).
    pub(crate) fn load_tail(&mut self, ins: TailInsert) -> Result<(), Refused> {
        let b = ins.t / C;
        self.chunk_path(&ins.base, b).ok_or(Refused::MissingAncestor)?;
        self.place_tail(&ins, b);
        Ok(())
    }

    /// Startup scan: set the recomputed `path_last_used` (8.3).
    pub(crate) fn set_path_last_used(&mut self, key: &Key, plu: u64) {
        self.rescore(key, |e| e.path_last_used = e.last_used.max(plu));
    }

    /// The caller's ancestor list, restricted to tails that still exist and
    /// structurally sit on this tail's path (their node is on it, t below).
    fn valid_ancestors(&self, ins: &TailInsert, path: &[Key]) -> Vec<Key> {
        let nodes: KeySet = path.iter().copied().chain(std::iter::once(self.root)).collect();
        ins.ancestors
            .iter()
            .filter(|a| **a != ins.key)
            .filter(|a| self.tails.get(*a).is_some_and(|e| e.t < ins.t && nodes.contains(&e.base)))
            .copied()
            .collect()
    }

    fn propagate(&mut self, key: &Key, ancestors: &[Key], when: u64) {
        self.rescore(key, |e| {
            e.last_used = e.last_used.max(when);
            e.path_last_used = e.path_last_used.max(when);
        });
        for a in ancestors {
            self.rescore(a, |e| e.path_last_used = e.path_last_used.max(when));
        }
    }

    /// 8.2: the newest N = 2 full tails on the path stay full (anchors are
    /// exempt and do not count); older ones are demoted. Then the path keeps
    /// at most one demoted tail per K-window, the deepest. "Newest" on one
    /// path is "deepest": a conversation's prompt ends grow with its turns.
    fn demote_and_thin(&mut self, new: &Key, ancestors: &[Key], now: u64) {
        let mut on_path: Vec<(u32, Key)> = ancestors
            .iter()
            .chain(std::iter::once(new))
            .filter_map(|k| self.tails.get(k).map(|e| (e.t, *k)))
            .collect();
        on_path.sort_unstable_by(|a, b| b.cmp(a));
        on_path.dedup();
        let fulls: Vec<Key> = on_path
            .iter()
            .filter(|(_, k)| self.tails.get(k).is_some_and(|e| e.kind == TailKind::Full && !e.anchor))
            .map(|(_, k)| *k)
            .collect();
        for k in fulls.iter().skip(DEMOTE_KEEP_FULL) {
            self.demote(k);
        }
        let mut windows: BTreeMap<u32, Vec<(u32, Key)>> = BTreeMap::new();
        for (t, k) in &on_path {
            if self.tails.get(k).is_some_and(|e| e.demoted && !e.anchor) {
                windows.entry(t / K).or_default().push((*t, *k));
            }
        }
        for (_, mut v) in windows {
            v.sort_unstable_by(|a, b| b.cmp(a));
            let keeper = v[0].1;
            for (_, k) in v.into_iter().skip(1) {
                self.thin(&k, keeper, now);
            }
        }
    }

    /// Demote a full tail to encoder-only (or defer it while pinned).
    pub fn demote(&mut self, key: &Key) {
        let Some(e) = self.tails.get_mut(key) else { return };
        if e.kind != TailKind::Full || e.anchor {
            return;
        }
        if e.pins > 0 {
            e.deferred_demote = true;
            return;
        }
        let freed = e.sec_d_bytes;
        let under = sub_sat(&mut e.bytes, freed);
        e.kind = TailKind::Enc;
        e.demoted = true;
        e.sec_d_bytes = 0;
        e.deferred_demote = false;
        let (gen, hits) = (e.gen, e.hits);
        if under {
            self.flag_drift("tail bytes below section D");
        }
        self.sub_tail_bytes(freed);
        self.gen_add(gen, 0, -(freed as i64));
        self.actions.push(Action::Demote { key: *key, hits });
    }

    fn thin(&mut self, key: &Key, keeper: Key, now: u64) {
        let Some(e) = self.tails.get_mut(key) else { return };
        if e.pins > 0 {
            e.thin_by = Some(keeper);
            return;
        }
        self.remove_tail(key, Why::Thinned, now);
    }

    /// Take a tail out of the index (its unlink and the cascade of chunks
    /// that only it referenced become actions). Returns the bytes freed.
    pub fn remove_tail(&mut self, key: &Key, why: Why, now: u64) -> u64 {
        let Some(e) = self.tails.remove(key) else { return 0 };
        self.by_score.remove(&(Self::score_of(&e), *key));
        if let Some(v) = self.tails_at.get_mut(&e.base) {
            v.retain(|k| k != key);
            if v.is_empty() {
                self.tails_at.remove(&e.base);
            }
        }
        self.sub_tail_bytes(e.bytes);
        self.gen_add(e.gen, -1, -(e.bytes as i64));
        if e.pins > 0 {
            // Forced removal of a pinned tail (corrupt): its pin records keep
            // only their chunk part.
            for rec in self.pins.values_mut() {
                if rec.tail == Some(*key) {
                    rec.tail = None;
                }
            }
        }
        self.actions.push(Action::UnlinkTail { key: *key, bytes: e.bytes, why });
        let mut cascaded = 0;
        if e.t >= C {
            self.add_path(&e.base, -1, 0);
            // The deletable chunks (refs 0, pins 0) are the deepest stretch
            // of the path; delete from its top down.
            let mut top = None;
            let mut node = e.base;
            while let Some(c) = self.chunks.get(&node) {
                if c.refs != 0 || c.pins != 0 {
                    break;
                }
                top = Some(node);
                node = c.parent;
            }
            if let Some(top) = top {
                let n_before = self.removals.len();
                cascaded = self.delete_subtree(&top, why, now);
                // Fold the chunk cascade into this tail's record.
                self.removals.truncate(n_before);
            }
        }
        self.removals.push(Removal {
            what: Removed::Tail { t: e.t, kind: e.kind },
            key: *key,
            bytes: e.bytes,
            cascaded,
            age_s: now.saturating_sub(e.last_used),
            hits: e.hits,
            why,
        });
        e.bytes + cascaded
    }

    /// A restore of this tail (8.3): one hit, persisted with the mtime; the
    /// ancestors' `path_last_used` moves in memory only.
    pub fn touch(&mut self, key: &Key, ancestors: &[Key], now: u64) {
        if !self.tails.contains_key(key) {
            return;
        }
        self.rescore(key, |e| e.hits = e.hits.saturating_add(1).min(super::format::MAX_HITS));
        self.propagate(key, ancestors, now);
        let e = &self.tails[key];
        self.actions.push(Action::Touch { key: *key, hits: e.hits, last_used: e.last_used });
    }

    // ---- pins ---------------------------------------------------------

    /// Pin a restore plan: the tail and its chunk path, until the restore
    /// completes (6.5). Pins block deletion, demotion and thinning. A plan
    /// owned by `job` is also released when the job ends.
    pub fn pin_plan(&mut self, tail: &Key, job: Option<JobId>) -> Option<PinId> {
        let job = self.live_job(job);
        let e = self.tails.get_mut(tail)?;
        e.pins += 1;
        let (base, t) = (e.base, e.t);
        let chunk = (t >= C).then_some(base);
        if let Some(c) = chunk {
            self.add_path(&c, 0, 1);
        }
        let id = self.new_pin(PinRec { chunk, tail: Some(*tail), job });
        if let Some(j) = job {
            self.job_pins.entry(j).or_default().push(id);
        }
        Some(id)
    }

    pub fn unpin(&mut self, id: PinId, now: u64) {
        let Some(rec) = self.pins.remove(&id) else { return };
        if let Some(j) = rec.job {
            if let Some(v) = self.job_pins.get_mut(&j) {
                v.retain(|x| *x != id);
                if v.is_empty() {
                    self.job_pins.remove(&j);
                }
            }
        }
        self.drop_pin(rec, now);
    }

    fn drop_pin(&mut self, rec: PinRec, now: u64) {
        if let Some(c) = rec.chunk {
            self.add_path(&c, 0, -1);
        }
        let Some(t) = rec.tail else { return };
        let Some(e) = self.tails.get_mut(&t) else { return };
        if e.pins == 0 {
            self.flag_drift("tail pins");
            return;
        }
        e.pins -= 1;
        if e.pins > 0 {
            return;
        }
        let (deferred, thin_by) = (e.deferred_demote, e.thin_by.take());
        if deferred {
            self.demote(&t);
        }
        if let Some(by) = thin_by {
            // Thin only if the tail that superseded it is still there.
            if self.tails.contains_key(&by) {
                self.remove_tail(&t, Why::Thinned, now);
            }
        }
    }

    fn release_job_pins(&mut self, job: JobId, now: u64, mut pred: impl FnMut(&Self, &PinRec) -> bool) {
        let Some(ids) = self.job_pins.remove(&job) else { return };
        let (mut keep, mut drop) = (Vec::new(), Vec::new());
        for id in ids {
            match self.pins.get(&id) {
                Some(rec) if pred(self, rec) => drop.push(id),
                Some(_) => keep.push(id),
                None => {}
            }
        }
        if !keep.is_empty() {
            self.job_pins.insert(job, keep);
        }
        for id in drop {
            if let Some(rec) = self.pins.remove(&id) {
                self.drop_pin(rec, now);
            }
        }
    }

    /// A job ended. Every pin it owns goes; on a failure (an error, not a
    /// cancel) its chunks that no tail references are deleted at once (9.4).
    /// After a cancel or a normal end they stay, as orphans if unreferenced.
    /// Writes of the job still in flight land unowned ([`Index::insert_chunk`])
    /// until [`Index::forget_job`].
    pub fn release_job(&mut self, job: JobId, failed: bool, now: u64) {
        let chunks: Vec<Key> = self
            .job_pins
            .get(&job)
            .map(|ids| ids.iter().filter_map(|id| self.pins.get(id)).filter(|r| r.tail.is_none()).filter_map(|r| r.chunk).collect())
            .unwrap_or_default();
        self.release_job_pins(job, now, |_, _| true);
        self.finished.insert(job, failed);
        if failed {
            for k in chunks {
                if self.chunks.get(&k).is_some_and(|c| c.refs == 0 && c.pins == 0) {
                    self.delete_subtree(&k, Why::Failed, now);
                }
            }
        }
    }

    /// The job's writes have all completed: stop remembering it.
    pub fn forget_job(&mut self, job: JobId) {
        self.finished.remove(&job);
    }

    // ---- eviction (8.3) ---------------------------------------------------

    /// Bytes freed by removing this tail now: its own, plus the deepest
    /// stretch of its path that only it references.
    fn freed_by(&self, e: &TailEntry) -> u64 {
        let mut freed = e.bytes;
        if e.t >= C {
            let mut node = e.base;
            while let Some(c) = self.chunks.get(&node) {
                if c.refs != 1 || c.pins != 0 {
                    break;
                }
                freed += c.bytes;
                node = c.parent;
            }
        }
        freed
    }

    /// The lowest-score unpinned tail; among equal scores the one freeing the
    /// most bytes, cascade included (so a dead conversation's leaves go first
    /// and its chunks cascade).
    fn pick_victim(&self) -> Option<Key> {
        const TIE_SCAN: usize = 64;
        let mut first = None;
        let mut best: Option<(u64, Key)> = None;
        let mut seen = 0;
        for &(score, key) in &self.by_score {
            let e = &self.tails[&key];
            if e.pins > 0 {
                continue;
            }
            match first {
                None => first = Some(score),
                Some(s) if score > s => break,
                _ => {}
            }
            let freed = self.freed_by(e);
            if best.is_none_or(|(f, _)| freed > f) {
                best = Some((freed, key));
            }
            seen += 1;
            if seen >= TIE_SCAN {
                break;
            }
        }
        best.map(|(_, k)| k)
    }

    /// Enforce the global cap: above it, evict down to cap − 1% (headroom), in
    /// the order inactive namespaces, orphan chunks (oldest first), tails by
    /// lowest score. Pinned entries are skipped. Returns false if the target
    /// could not be reached because everything left is pinned.
    pub fn enforce_cap(&mut self, now: u64) -> bool {
        if self.total_bytes() <= self.cap_bytes {
            return true;
        }
        let target = self.cap_bytes - self.cap_bytes / 100;
        while self.total_bytes() > target {
            if let Some((ns16, bytes)) = self.inactive.pop() {
                self.actions.push(Action::TrashNamespace { ns16, bytes });
                self.removals.push(Removal {
                    what: Removed::Namespace,
                    key: [0; 32],
                    bytes,
                    cascaded: 0,
                    age_s: 0,
                    hits: 0,
                    why: Why::Namespace,
                });
                continue;
            }
            if let Some(&(_, _, key)) = self.orphans.iter().next() {
                self.delete_subtree(&key, Why::Orphan, now);
                continue;
            }
            let Some(victim) = self.pick_victim() else { return false };
            self.remove_tail(&victim, Why::Cap, now);
        }
        true
    }

    // ---- lookup (6.1) ---------------------------------------------------

    /// The tails at `node` (= chain_j, a = jC) whose key matches `tokens`, with
    /// t ≤ `max_t`, by increasing t.
    fn match_tails_at(&self, node: &Key, a: usize, tokens: &[i32], images: &[ImageRecord], max_t: usize, out: &mut Vec<WalkTail>) {
        let Some(ts) = self.tails_at.get(node) else { return };
        let mut cands: Vec<u32> = ts.iter().map(|k| self.tails[k].t).filter(|&t| t as usize <= max_t).collect();
        cands.sort_unstable();
        cands.dedup();
        for t in cands {
            let key = keys::tail_step(node, a as u32, &tokens[a..t as usize], images_in(images, a as u32, t));
            if let Some(e) = self.tails.get(&key) {
                if e.base == *node {
                    out.push(WalkTail { key, t, kind: e.kind, anchor: e.anchor });
                }
            }
        }
    }

    /// Walk a request: chain keys while they are indexed, testing the tails at
    /// every matched node (b = 0 included). Pending writes are not indexed, so
    /// they are never matched; removed entries leave the index at once.
    pub fn walk(&self, tokens: &[i32], images: &[ImageRecord]) -> Walk {
        let l = tokens.len();
        let mut w = Walk::default();
        let mut node = self.root;
        let mut a = 0usize;
        loop {
            self.match_tails_at(&node, a, tokens, images, l, &mut w.tails);
            let end = a + C as usize;
            if end > l {
                break;
            }
            let next = keys::chunk_step(&node, a as u32, &tokens[a..end], images_in(images, a as u32, end as u32));
            if !self.chunks.contains_key(&next) {
                break;
            }
            w.chunks.push(next);
            node = next;
            a = end;
        }
        w
    }

    /// The indexed tails on the path of a tail at `t` of this request (t' <
    /// t), given its chain keys chain_0..chain_⌊t/C⌋: the walk's matching
    /// without re-hashing any chunk. Stops at the first chain key not indexed.
    pub fn ancestors_on_path(&self, chain: &[Key], tokens: &[i32], images: &[ImageRecord], t: u32) -> Vec<Key> {
        let mut out = Vec::new();
        if t == 0 {
            return Vec::new();
        }
        for (j, node) in chain.iter().enumerate() {
            if j > 0 && !self.chunks.contains_key(node) {
                break;
            }
            self.match_tails_at(node, j * C as usize, tokens, images, t as usize - 1, &mut out);
        }
        out.into_iter().map(|w| w.key).collect()
    }

    // ---- invariant checker (9.5) ------------------------------------------

    /// Rebuild refcounts and pins bottom-up over the trie (O(chunks + tails)),
    /// the orphan and score sets, byte totals, generation stats and the maps,
    /// and compare with the incremental state. Run in tests after every
    /// operation, and hourly in debug builds and in shadow.
    pub fn check_invariants(&self) -> Result<(), String> {
        let h = |k: &Key| keys::hex(&k[..4]);
        // Reachable chunks, and the k numbering along every edge.
        let mut reach: KeySet = KeySet::default();
        let mut stack = vec![(self.root, u32::MAX)];
        while let Some((n, kn)) = stack.pop() {
            for c in self.children(&n) {
                let e = self.chunks.get(c).ok_or("children map names a missing chunk")?;
                if e.parent != n || e.k != kn.wrapping_add(1) {
                    return Err(format!("chunk {} misplaced under its parent", h(c)));
                }
                reach.insert(*c);
                stack.push((*c, e.k));
            }
        }
        let mut tail_bytes = 0u64;
        let mut own_refs: KeyMap<u32> = KeyMap::default();
        let mut tail_pins: KeyMap<u32> = KeyMap::default();
        for (k, e) in &self.tails {
            tail_bytes += e.bytes;
            let b = e.t / C;
            if b == 0 {
                if e.base != self.root {
                    return Err(format!("tail {} at t={} not on the root", h(k), e.t));
                }
            } else {
                let ok = reach.contains(&e.base) && self.chunks.get(&e.base).is_some_and(|c| c.k + 1 == b);
                if !ok {
                    return Err(format!("tail {} at t={} has an incomplete chunk path", h(k), e.t));
                }
                *own_refs.entry(e.base).or_default() += 1;
            }
            if !self.tails_at.get(&e.base).is_some_and(|v| v.contains(k)) {
                return Err(format!("tail {} missing from tails_at", h(k)));
            }
            if !self.by_score.contains(&(Self::score_of(e), *k)) {
                return Err(format!("tail {} missing from the score order", h(k)));
            }
            if e.kind == TailKind::Enc && e.sec_d_bytes != 0 {
                return Err("encoder tail with section D bytes".into());
            }
            if e.demoted && (e.kind != TailKind::Enc || e.anchor) {
                return Err("demoted tail that is full or an anchor".into());
            }
            if e.path_last_used < e.last_used {
                return Err("path_last_used below last_used".into());
            }
        }
        if self.by_score.len() != self.tails.len() || self.tails_at.values().map(|v| v.len()).sum::<usize>() != self.tails.len() {
            return Err("score order / tails_at count differs from the tails".into());
        }
        let mut own_pins: KeyMap<u32> = KeyMap::default();
        for rec in self.pins.values() {
            if let Some(c) = rec.chunk {
                if !self.chunks.contains_key(&c) {
                    return Err("a pin record is anchored at a missing chunk".into());
                }
                *own_pins.entry(c).or_default() += 1;
            }
            if let Some(t) = rec.tail {
                *tail_pins.entry(t).or_default() += 1;
            }
        }
        for (j, ids) in &self.job_pins {
            if self.finished.contains_key(j) {
                return Err(format!("finished job {j:?} still owns {} pins", ids.len()));
            }
            if ids.iter().any(|id| self.pins.get(id).is_none_or(|r| r.job != Some(*j))) {
                return Err("job pin list names a retired or foreign record".into());
            }
        }
        let owned: std::collections::HashSet<(JobId, PinId)> = self.job_pins.iter().flat_map(|(j, ids)| ids.iter().map(move |id| (*j, *id))).collect();
        if self.pins.iter().any(|(id, r)| r.job.is_some_and(|j| !owned.contains(&(j, *id)))) {
            return Err("a job-owned pin record is missing from its job's list".into());
        }
        // Bottom-up: children (k + 1) before parents; totals flow to existing
        // parents only, exactly like add_path.
        let mut order: Vec<(&Key, &ChunkEntry)> = self.chunks.iter().collect();
        order.sort_unstable_by(|a, b| b.1.k.cmp(&a.1.k));
        let mut refs: KeyMap<u32> = own_refs;
        let mut pins: KeyMap<u32> = own_pins;
        let mut chunk_bytes = 0u64;
        let mut orphans = 0usize;
        for (k, c) in &order {
            chunk_bytes += c.bytes;
            let (r, p) = (refs.get(*k).copied().unwrap_or(0), pins.get(*k).copied().unwrap_or(0));
            if c.refs != r {
                return Err(format!("chunk {} k={} refs {} != rebuilt {}", h(k), c.k, c.refs, r));
            }
            if c.pins != p {
                return Err(format!("chunk {} k={} pins {} != rebuilt {}", h(k), c.k, c.pins, p));
            }
            if self.chunks.contains_key(&c.parent) {
                *refs.entry(c.parent).or_default() += r;
                *pins.entry(c.parent).or_default() += p;
            } else if c.parent == self.root && c.k != 0 {
                return Err("a child of the root is not chunk 0".into());
            }
            let is_orphan = c.refs == 0 && c.pins == 0;
            if is_orphan != self.orphans.contains(&(c.created, c.k, **k)) {
                return Err(format!("chunk {} orphan set membership wrong", h(k)));
            }
            orphans += is_orphan as usize;
            if !self.children.get(&c.parent).is_some_and(|v| v.contains(k)) {
                return Err(format!("chunk {} missing from its parent's children", h(k)));
            }
        }
        if orphans != self.orphans.len() {
            return Err(format!("orphan set has {} entries, rebuilt {}", self.orphans.len(), orphans));
        }
        for (k, e) in &self.tails {
            let want = tail_pins.get(k).copied().unwrap_or(0);
            if e.pins != want {
                return Err(format!("tail {} pins {} != rebuilt {}", h(k), e.pins, want));
            }
        }
        if chunk_bytes != self.chunk_bytes || tail_bytes != self.tail_bytes {
            return Err(format!(
                "byte totals drifted: chunks {} vs {}, tails {} vs {}",
                self.chunk_bytes, chunk_bytes, self.tail_bytes, tail_bytes
            ));
        }
        let mut gen_bytes = vec![GenStat::default(); self.gens.len()];
        for (g, b) in self.chunks.values().map(|c| (c.gen, c.bytes)).chain(self.tails.values().map(|e| (e.gen, e.bytes))) {
            gen_bytes[g as usize].files += 1;
            gen_bytes[g as usize].bytes += b;
        }
        if gen_bytes != self.gen_stats {
            return Err("generation stats drifted".into());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kvstore::keys::{ChainCursor, KeyChain};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::VecDeque;

    const GEN: GenPair = GenPair { gen: 1, knob: [0; 16] };
    const CHUNK_B: u64 = 1000;
    const ENC_B: u64 = 300;
    const D_B: u64 = 200;

    enum Pend {
        Chunk(ChunkInsert),
        Tail(TailInsert),
    }

    impl Pend {
        fn job(&self) -> Option<JobId> {
            match self {
                Pend::Chunk(c) => c.job,
                Pend::Tail(t) => t.job,
            }
        }
    }

    /// An in-memory driver that mimics M2's job flow and the store's
    /// completion handling over the index: writes go into an in-flight FIFO
    /// (the IO thread completes in order) and land when drained, possibly
    /// after the job has ended; a write may be dropped.
    struct Sim {
        chain: KeyChain,
        idx: Index,
        next_job: u64,
        inflight: VecDeque<(Key, Pend, bool)>,
        job_writes: HashMap<JobId, u32>,
        refused: u32,
    }

    /// How one admission runs.
    #[derive(Clone, Copy, Default)]
    struct Opts {
        /// The job fails (no prompt-end tail, release as failed).
        fail: bool,
        /// Leave the writes in flight when the job ends (drained later).
        async_end: bool,
        /// Drop every write with this probability (needs an rng).
        drop_p: f64,
        /// Also write an anchor at the deepest K multiple ≤ L.
        anchor: bool,
    }

    impl Sim {
        fn new(cap: u64) -> Self {
            let chain = KeyChain::new([42; 32]);
            let idx = Index::new(*chain.root(), cap);
            Self { chain, idx, next_job: 1, inflight: VecDeque::new(), job_writes: HashMap::new(), refused: 0 }
        }

        fn chunk_ins(&self, cur: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], k: u32, job: Option<JobId>, now: u64) -> ChunkInsert {
            let parent = cur.chain(tokens, images, k);
            let key = cur.chain(tokens, images, k + 1);
            ChunkInsert { key, parent, k, bytes: CHUNK_B, created: now, gen: GEN, payload_hash: [k as u8; 32], job }
        }

        #[allow(clippy::too_many_arguments)]
        fn tail_ins(&self, cur: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], t: u32, kind: TailKind, origin: TailOrigin, ancestors: Vec<Key>, job: Option<JobId>, now: u64) -> TailInsert {
            let (base, key) = self.chain.tail_key(cur, tokens, images, t);
            TailInsert {
                key,
                base,
                t,
                kind,
                origin,
                anchor: origin == TailOrigin::Anchor,
                demoted: false,
                bytes: ENC_B + if kind == TailKind::Full { D_B } else { 0 },
                sec_d_bytes: if kind == TailKind::Full { D_B } else { 0 },
                created: now,
                gen: GEN,
                hits: 0,
                ancestors,
                job,
            }
        }

        /// The store's ancestor rule: indexed tails on the path, plus
        /// in-flight tails whose key matches this request.
        fn ancestors(&self, cur: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], t: u32) -> Vec<Key> {
            let chain = cur.keys_to(tokens, images, t / C).to_vec();
            let mut anc = self.idx.ancestors_on_path(&chain, tokens, images, t);
            for (pk, p, _) in &self.inflight {
                let Pend::Tail(p) = p else { continue };
                let b = (p.t / C) as usize;
                if p.t < t && chain[b] == p.base && !anc.contains(pk) {
                    let a = b as u32 * C;
                    if keys::tail_step(&chain[b], a, &tokens[a as usize..p.t as usize], images_in(images, a, p.t)) == *pk {
                        anc.push(*pk);
                    }
                }
            }
            anc
        }

        fn in_flight_kind(&self, key: &Key) -> Option<TailKind> {
            self.inflight.iter().filter(|(k, _, _)| k == key).filter_map(|(_, p, _)| match p {
                Pend::Tail(t) => Some(t.kind),
                Pend::Chunk(_) => None,
            }).max()
        }

        fn push(&mut self, key: Key, p: Pend, dropped: bool) {
            if let Some(j) = p.job() {
                *self.job_writes.entry(j).or_default() += 1;
            }
            self.inflight.push_back((key, p, dropped));
        }

        /// Land up to `n` in-flight writes, in order, the way the store does.
        fn drain(&mut self, n: usize, now: u64) {
            for _ in 0..n {
                let Some((_, p, dropped)) = self.inflight.pop_front() else { break };
                let job = p.job();
                if !dropped {
                    match p {
                        Pend::Chunk(c) => {
                            self.idx.insert_chunk(c, now);
                        }
                        Pend::Tail(t) => {
                            if self.idx.insert_tail(t, now).is_err() {
                                self.refused += 1;
                            }
                        }
                    }
                }
                // After the insert, like the store: forgetting the job first
                // would let its last write pin for a job nobody releases.
                if let Some(j) = job {
                    let w = self.job_writes.get_mut(&j).unwrap();
                    *w -= 1;
                    if *w == 0 {
                        self.job_writes.remove(&j);
                        if self.idx.finished(j).is_some() {
                            self.idx.forget_job(j);
                        }
                    }
                }
                self.idx.enforce_cap(now);
            }
        }

        fn drain_all(&mut self, now: u64) {
            let n = self.inflight.len();
            self.drain(n, now);
        }

        fn queue_tail(&mut self, cur: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], t: u32, kind: TailKind, origin: TailOrigin, job: JobId, now: u64, drop: bool) -> Key {
            let anc = self.ancestors(cur, tokens, images, t);
            let ins = self.tail_ins(cur, tokens, images, t, kind, origin, anc, Some(job), now);
            let key = ins.key;
            let have = self.idx.tail(&key).map(|e| e.kind).max(self.in_flight_kind(&key));
            if have.is_none_or(|k| k < kind) {
                self.push(key, Pend::Tail(ins), drop);
            }
            key
        }

        /// One admission of `tokens`: returns the restored t (0 = cold).
        fn admit_with(&mut self, tokens: &[i32], images: &[ImageRecord], now: u64, o: Opts, rng: Option<&mut StdRng>) -> u32 {
            let mut rng = rng;
            let mut roll = |p: f64| rng.as_mut().is_some_and(|r| r.gen_bool(p));
            let l = tokens.len() as u32;
            let walk = self.idx.walk(tokens, images);
            let plan = select(&walk, l, false);
            let pos0 = plan.map_or(0, |p| p.t);
            let job = JobId(self.next_job);
            self.next_job += 1;
            let pin = plan.and_then(|p| self.idx.pin_plan(&p.key, Some(job)));
            if let Some(p) = plan {
                self.idx.touch(&p.key, &walk.ancestors_below(p.t), now);
            }
            if let Some(id) = pin {
                self.idx.unpin(id, now);
            }
            let mut cur = self.chain.cursor();
            cur.seed(&walk.chunks);
            for k in pos0 / C..l / C {
                let ins = self.chunk_ins(&mut cur, tokens, images, k, Some(job), now);
                let key = ins.key;
                let pending = self.inflight.iter().any(|(pk, _, _)| *pk == key);
                if self.idx.chunk(&key).is_none() && !pending {
                    let d = roll(o.drop_p);
                    self.push(key, Pend::Chunk(ins), d);
                }
                let end = (k + 1) * C;
                // A waypoint at every K multiple crossed, L included when L is
                // one: then the prompt-end full tail has the SAME key.
                if end % K == 0 && end > pos0 && end <= l {
                    let d = roll(o.drop_p);
                    self.queue_tail(&mut cur, tokens, images, end, TailKind::Enc, TailOrigin::Waypoint, job, now, d);
                }
            }
            if o.anchor && l >= K {
                let a = l / K * K;
                let d = roll(o.drop_p);
                self.queue_tail(&mut cur, tokens, images, a, TailKind::Full, TailOrigin::Anchor, job, now, d);
            }
            if !o.fail && l >= 64 {
                let d = roll(o.drop_p);
                self.queue_tail(&mut cur, tokens, images, l, TailKind::Full, TailOrigin::PromptEnd, job, now, d);
            }
            if !o.async_end {
                self.drain_all(now);
            }
            // M2 ends the job in the tick that queues its last writes.
            self.idx.release_job(job, o.fail, now);
            if !self.job_writes.contains_key(&job) {
                self.idx.forget_job(job);
            }
            self.idx.enforce_cap(now);
            pos0
        }

        fn admit(&mut self, tokens: &[i32], now: u64, fail: bool, check: bool) -> u32 {
            let pos0 = self.admit_with(tokens, &[], now, Opts { fail, ..Opts::default() }, None);
            if check {
                self.idx.check_invariants().unwrap();
            }
            pos0
        }
    }

    /// A uniformly random element of a HashMap's keys, independent of the
    /// map's (per-process random) iteration order.
    fn pick<'a>(rng: &mut StdRng, keys: impl Iterator<Item = &'a Key>) -> Option<Key> {
        let mut v: Vec<Key> = keys.copied().collect();
        v.sort_unstable();
        (!v.is_empty()).then(|| v[rng.gen_range(0..v.len())])
    }

    fn conv(seed: u64, len: usize) -> Vec<i32> {
        let mut r = StdRng::seed_from_u64(seed);
        (0..len).map(|_| r.gen_range(0..128_000)).collect()
    }

    /// Two images per conversation: one straddling the chunk 0/1 boundary,
    /// one starting in the open part of the second chunk.
    fn images_for(seed: u64) -> Vec<ImageRecord> {
        vec![
            ImageRecord { start: 1000, len: 100, hash: [seed as u8; 32] },
            ImageRecord { start: 1500, len: 40, hash: [seed as u8 ^ 0x5a; 32] },
        ]
    }

    #[test]
    fn selection_table() {
        let w = |t, kind| WalkTail { key: [t as u8; 32], t, kind, anchor: false };
        let walk = Walk { chunks: vec![], tails: vec![w(64, TailKind::Enc), w(500, TailKind::Full), w(872, TailKind::Enc)] };
        // L − t = 128 never picks an encoder tail; 129 does.
        assert_eq!(select(&walk, 1000, false).unwrap().t, 500);
        assert_eq!(select(&walk, 1001, false).unwrap().t, 872);
        // A full tail at T = L only with a trailing marker.
        let walk2 = Walk { chunks: vec![], tails: vec![w(100, TailKind::Full), w(300, TailKind::Full)] };
        assert_eq!(select(&walk2, 300, false).unwrap().t, 100);
        assert_eq!(select(&walk2, 300, true).unwrap().t, 300);
        assert_eq!(select(&walk2, 301, false).unwrap().t, 300);
        // t ≥ 64.
        let walk3 = Walk { chunks: vec![], tails: vec![w(63, TailKind::Full)] };
        assert!(select(&walk3, 1000, false).is_none());
        // A full tail wins a tie.
        let walk4 = Walk { chunks: vec![], tails: vec![w(300, TailKind::Enc), w(300, TailKind::Full)] };
        assert_eq!(select(&walk4, 1000, false).unwrap().kind, TailKind::Full);
        for t in 0..2000u32 {
            for l in t..t + 300 {
                if usable(TailKind::Enc, t, l, false) {
                    assert!(l - t > 128 && t >= 64);
                }
            }
        }
    }

    #[test]
    fn walk_finds_every_written_tail_at_every_prefix_length() {
        // G5 "walk keys = write keys for every prefix length": a tail written
        // at T through the writer's key path (ChainCursor / tail_key) is found
        // by the walk (its own slicing of ids and images) of any request that
        // extends tokens[..T], and by no walk of a request that does not. With
        // and without images (one straddling the chunk 0/1 boundary, one in
        // the open part of chunk 1).
        for images in [vec![], images_for(7)] {
            let mut s = Sim::new(u64::MAX);
            let tokens = conv(1, 3 * C as usize + 50);
            let mut cur = s.chain.cursor();
            for k in 0..3 {
                let ins = s.chunk_ins(&mut cur, &tokens, &images, k, None, 0);
                s.idx.insert_chunk(ins, 0);
            }
            for t in 0..=tokens.len() as u32 {
                let ins = s.tail_ins(&mut cur, &tokens, &images, t, TailKind::Enc, TailOrigin::Waypoint, vec![], None, 0);
                let key = ins.key;
                s.idx.insert_tail(ins, 0).unwrap();
                let w = s.idx.walk(&tokens[..t as usize], &images);
                assert_eq!(w.tails.last().map(|x| (x.t, x.key)), Some((t, key)), "t={t}");
                assert_eq!(w.chunks.len() as u32, t / C);
                if t % 97 == 0 {
                    let anc = s.idx.ancestors_on_path(cur.keys_to(&tokens, &images, t / C), &tokens, &images, t);
                    assert_eq!(anc.len() as u32, t, "every earlier tail is an ancestor (t={t})");
                }
            }
            let w = s.idx.walk(&tokens, &images);
            assert_eq!(w.tails.len(), tokens.len() + 1);
            // A request that differs at position p sees only tails at t ≤ p.
            let mut other = tokens.clone();
            other[1500] ^= 1;
            let w = s.idx.walk(&other, &images);
            assert_eq!(w.tails.last().unwrap().t, 1500);
            assert_eq!(w.chunks.len(), 1);
            if !images.is_empty() {
                // Different pixels: nothing past the first IMAGE_START matches.
                let w = s.idx.walk(&tokens, &images_for(8));
                assert_eq!(w.tails.last().unwrap().t, 1000);
                assert_eq!(w.chunks.len(), 0);
            }
            s.idx.check_invariants().unwrap();
        }
    }

    #[test]
    fn bug_a_checkpoint_keys_come_from_request_tokens() {
        // Bug (a) (1.1): the checkpoint key was built from
        // `pf.prefix + job.tokens()[..done]` after `start_prefill` had already
        // extended `pf.prefix` to the whole prompt, giving pos0 + total + done
        // tokens. Keys are now built from the REQUEST tokens and an explicit
        // position, so the cancel tail of a retry's own prompt is found.
        let s = Sim::new(u64::MAX);
        let req = conv(2, 5000);
        let (pos0, done) = (1000u32, 2200u32);
        // What the bug fed the old store:
        let buggy: Vec<i32> = req.iter().copied().chain(req[pos0 as usize..(pos0 + done) as usize].iter().copied()).collect();
        assert_eq!(buggy.len() as u32, 5000 + done);
        let t = pos0 + done;
        let (_, good) = s.chain.tail_key(&mut s.chain.cursor(), &req, &[], t);
        // The bug's key (whole buggy vector as the key's tokens) is not the
        // key of any position of the request:
        let (_, bad) = s.chain.tail_key(&mut s.chain.cursor(), &buggy, &[], buggy.len() as u32);
        let mut cur = s.chain.cursor();
        assert!((0..=req.len() as u32).all(|x| s.chain.tail_key(&mut cur, &req, &[], x).1 != bad));
        // ... while the API, given the request and t, can only produce `good`.
        let mut sim = Sim::new(u64::MAX);
        let mut cur = sim.chain.cursor();
        for k in 0..t / C {
            let ins = sim.chunk_ins(&mut cur, &req, &[], k, None, 0);
            sim.idx.insert_chunk(ins, 0);
        }
        let ins = sim.tail_ins(&mut cur, &req, &[], t, TailKind::Enc, TailOrigin::Cancel, vec![], None, 0);
        assert_eq!(ins.key, good);
        sim.idx.insert_tail(ins, 0).unwrap();
        let w = sim.idx.walk(&req, &[]);
        assert_eq!(select(&w, req.len() as u32, false).unwrap().t, t, "the retry restores at its cancel tail");
    }

    #[test]
    fn demotion_keeps_two_full_and_thinning_one_per_window() {
        let mut s = Sim::new(u64::MAX);
        let base = conv(3, 40_000);
        // Ten turns of one conversation, 3,000 tokens apart.
        let mut now = 100;
        for turn in 1..=10usize {
            now += 60;
            s.admit(&base[..3000 * turn + 500], now, false, true);
        }
        let w = s.idx.walk(&base, &[]);
        let fulls: Vec<u32> = w.tails.iter().filter(|x| x.kind == TailKind::Full).map(|x| x.t).collect();
        assert_eq!(fulls, vec![27_500, 30_500], "the newest two prompt ends stay full");
        // Demoted prompt ends: at most one per K-window, the deepest.
        let mut per_window: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
        for x in &w.tails {
            let e = s.idx.tail(&x.key).unwrap();
            if e.demoted {
                per_window.entry(x.t / K).or_default().push(x.t);
            }
        }
        assert!(per_window.values().all(|v| v.len() == 1), "{per_window:?}");
        assert_eq!(per_window[&0], vec![6500]);
        // Waypoints are never thinned.
        let wps: Vec<u32> = w.tails.iter().filter(|x| s.idx.tail(&x.key).unwrap().origin == TailOrigin::Waypoint).map(|x| x.t).collect();
        assert_eq!(wps, vec![8192, 16_384, 24_576]);
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn anchor_is_exempt_from_demotion_and_thinning() {
        let mut s = Sim::new(u64::MAX);
        let base = conv(4, 20_000);
        let a = 5282u32;
        // The anchor tail at A, written on second sight (5.2).
        s.admit(&base[..6000], 10, false, true);
        let mut cur = s.chain.cursor();
        let w = s.idx.walk(&base[..a as usize], &[]);
        cur.seed(&w.chunks);
        let anc = s.tail_ins(&mut cur, &base, &[], a, TailKind::Full, TailOrigin::Anchor, vec![], None, 20);
        let anchor_key = anc.key;
        s.idx.insert_tail(anc, 20).unwrap();
        for (i, l) in [6200usize, 6400, 6600, 7000, 7500].into_iter().enumerate() {
            s.admit(&base[..l], 30 + i as u64, false, true);
        }
        let e = s.idx.tail(&anchor_key).unwrap();
        assert!(e.anchor && e.kind == TailKind::Full && !e.demoted, "anchor demoted: {e:?}");
        // The prompt ends in the same K-window were thinned to one.
        let w = s.idx.walk(&base, &[]);
        let demoted: Vec<u32> = w.tails.iter().filter(|x| s.idx.tail(&x.key).unwrap().demoted).map(|x| x.t).collect();
        assert_eq!(demoted, vec![6600]);
        // An anchor leaves only by score eviction.
        let freed = s.idx.remove_tail(&anchor_key, Why::Cap, 100);
        assert_eq!(freed, ENC_B + D_B);
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn pins_defer_demotion_and_thinning_and_block_eviction() {
        let mut s = Sim::new(u64::MAX);
        let base = conv(5, 20_000);
        s.admit(&base[..3000], 1, false, true);
        let w = s.idx.walk(&base[..3000], &[]);
        let first = w.tails.last().unwrap().key;
        let pin = s.idx.pin_plan(&first, None).unwrap();
        s.admit(&base[..3500], 2, false, true);
        s.admit(&base[..4000], 3, false, true);
        // The third full tail would demote `first`; it is pinned: deferred.
        assert_eq!(s.idx.tail(&first).unwrap().kind, TailKind::Full);
        assert!(s.idx.tail(&first).unwrap().deferred_demote);
        // Eviction skips it.
        s.idx.set_cap_bytes(1);
        s.idx.enforce_cap(4);
        assert!(s.idx.tail(&first).is_some(), "a pinned tail was evicted");
        assert_eq!(s.idx.n_tails(), 1);
        s.idx.check_invariants().unwrap();
        s.idx.unpin(pin, 5);
        assert_eq!(s.idx.tail(&first).unwrap().kind, TailKind::Enc, "demoted at unpin");
        assert!(s.idx.take_actions().iter().any(|a| matches!(a, Action::Demote { key, .. } if *key == first)));
        s.idx.enforce_cap(6);
        assert_eq!(s.idx.n_tails(), 0);
        assert_eq!(s.idx.n_chunks(), 0, "chunks cascade with the last tail");
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn plan_pins_owned_by_a_job_go_with_it() {
        // 12a: a restore plan pinned for a job is released when the job ends,
        // even if the job never unpins it.
        let mut s = Sim::new(u64::MAX);
        let base = conv(13, 5000);
        s.admit(&base[..3000], 1, false, true);
        let k = s.idx.walk(&base[..3000], &[]).tails[0].key;
        let job = JobId(500);
        s.idx.pin_plan(&k, Some(job)).unwrap();
        assert_eq!(s.idx.job_pin_count(job), 1);
        s.idx.set_cap_bytes(1);
        s.idx.enforce_cap(2);
        assert!(s.idx.tail(&k).is_some(), "the plan's tail is pinned");
        s.idx.release_job(job, false, 3);
        s.idx.forget_job(job);
        s.idx.enforce_cap(3);
        assert_eq!(s.idx.n_tails(), 0);
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn eviction_order_namespace_then_orphans_then_score() {
        let mut s = Sim::new(u64::MAX);
        let a = conv(6, 5000);
        let b = conv(7, 5000);
        s.admit(&a, 10, false, true);
        s.admit(&b, 20, false, true);
        // A failed job deletes its chunks at once; a cancelled one leaves orphans.
        s.admit(&conv(8, 3000), 30, true, true);
        assert_eq!(s.idx.n_orphans(), 0);
        let job = JobId(999);
        let c = conv(9, 3000);
        let mut cur = s.chain.cursor();
        for k in 0..2 {
            let ins = s.chunk_ins(&mut cur, &c, &[], k, Some(job), 40);
            s.idx.insert_chunk(ins, 40);
        }
        s.idx.release_job(job, false, 40);
        assert_eq!(s.idx.n_orphans(), 2);
        s.idx.set_inactive(vec![("deadbeefdeadbeef".into(), 500)]);
        s.idx.take_removals();
        // A cap that holds one conversation (with the 1% headroom): the
        // namespace, then the orphans, then the older conversation go.
        let one_conv = 4 * CHUNK_B + ENC_B + D_B;
        assert_eq!(s.idx.total_bytes(), 2 * one_conv + 2 * CHUNK_B + 500);
        s.idx.set_cap_bytes(one_conv + 100);
        s.idx.enforce_cap(50);
        let r = s.idx.take_removals();
        let whys: Vec<Why> = r.iter().map(|x| x.why).collect();
        assert_eq!(whys, vec![Why::Namespace, Why::Orphan, Why::Cap], "{r:?}");
        assert!(matches!(r[1].what, Removed::Chunk { k: 0 }) && r[1].cascaded == CHUNK_B, "the orphan subtree goes whole");
        assert_eq!(r[2].cascaded, 4 * CHUNK_B, "the tail's chunks cascade");
        assert!(s.idx.walk(&a, &[]).tails.is_empty(), "the older conversation went first");
        assert_eq!(s.idx.walk(&b, &[]).tails.len(), 1);
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn hits_and_path_touches_keep_live_lineage_young() {
        let mut s = Sim::new(u64::MAX);
        let a = conv(10, 30_000);
        s.admit(&a[..20_000], 0, false, true);
        let wp = s.idx.walk(&a[..20_000], &[]).tails[0];
        assert_eq!(wp.t, 8192);
        // A later turn restores the prompt end; its path touches the waypoint.
        s.admit(&a[..21_000], 7200, false, true);
        let e = s.idx.tail(&wp.key).unwrap();
        assert_eq!(e.last_used, 0, "a propagated touch is not the tail's own");
        assert_eq!(e.path_last_used, 7200);
        assert_eq!(e.hits, 0, "a propagated touch is not a hit");
        let end = s.idx.walk(&a[..20_000], &[]).tails.last().unwrap().key;
        assert_eq!(s.idx.tail(&end).unwrap().hits, 1);
    }

    #[test]
    fn detached_chunk_waits_for_its_parent() {
        // 9.4: a dropped chunk k re-enqueued behind chunk k+1.
        let mut s = Sim::new(u64::MAX);
        let tokens = conv(11, 3 * C as usize);
        let job = JobId(1);
        let mut cur = s.chain.cursor();
        let c0 = s.chunk_ins(&mut cur, &tokens, &[], 0, Some(job), 1);
        let c1 = s.chunk_ins(&mut cur, &tokens, &[], 1, Some(job), 1);
        let c2 = s.chunk_ins(&mut cur, &tokens, &[], 2, Some(job), 1);
        s.idx.insert_chunk(c0, 1);
        s.idx.insert_chunk(c2.clone(), 1);
        s.idx.check_invariants().unwrap();
        assert_eq!(s.idx.walk(&tokens, &[]).chunks.len(), 1, "the walk stops at the hole");
        // A tail above the hole is refused, and releases the job's pins on its
        // path (it was the job's "next tail", 9.4).
        let t = s.tail_ins(&mut cur, &tokens, &[], 3 * C, TailKind::Enc, TailOrigin::Cancel, vec![], Some(job), 1);
        assert_eq!(s.idx.insert_tail(t.clone(), 1), Err(Refused::MissingAncestor));
        assert_eq!(s.idx.job_pin_count(job), 0);
        s.idx.check_invariants().unwrap();
        let c1 = ChunkInsert { job: None, ..c1 };
        s.idx.insert_chunk(c1, 2);
        s.idx.check_invariants().unwrap();
        assert_eq!(s.idx.walk(&tokens, &[]).chunks.len(), 3);
        s.idx.insert_tail(t, 2).unwrap();
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn late_completions_after_the_job_ended() {
        // 2.: M2 ends a job in the tick that queues its last writes. Its late
        // chunks land unowned (no pin leaks); a FAILED job's late chunks go at
        // once when nothing references them; a cancelled job's stay as orphans.
        for (fail, drop_tail) in [(false, false), (true, false), (false, true)] {
            let mut s = Sim::new(u64::MAX);
            let tokens = conv(14, 5000);
            let o = Opts { fail, async_end: true, ..Opts::default() };
            s.admit_with(&tokens, &[], 1, o, None);
            if drop_tail {
                // The prompt-end tail is the last write: drop it.
                if let Some(last) = s.inflight.back_mut() {
                    last.2 = true;
                }
            }
            assert!(!s.inflight.is_empty(), "the job ended with writes in flight");
            assert!(s.idx.finished(JobId(1)).is_some());
            s.idx.check_invariants().unwrap();
            s.drain_all(2);
            s.idx.check_invariants().unwrap();
            assert!(s.idx.finished(JobId(1)).is_none(), "forgotten once its writes drained");
            assert!(s.idx.pins.is_empty(), "no pin outlives the job");
            match (fail, drop_tail) {
                (true, _) => assert_eq!(s.idx.n_chunks(), 0, "a failed job's late chunks are deleted"),
                (false, true) => assert_eq!(s.idx.n_orphans(), 4, "a cancelled job's late chunks are orphans"),
                _ => assert_eq!((s.idx.n_tails(), s.idx.n_orphans()), (1, 0)),
            }
        }
    }

    #[test]
    fn corrupt_chunk_takes_every_tail_beneath() {
        let mut s = Sim::new(u64::MAX);
        let base = conv(12, 12_000);
        s.admit(&base[..9000], 1, false, true);
        let mut branch = base[..9500].to_vec();
        branch[5000] ^= 7;
        s.admit(&branch, 2, false, true);
        assert_eq!(s.idx.n_tails(), 4, "a waypoint and a prompt end per branch");
        let w = s.idx.walk(&base[..9000], &[]);
        let pin = s.idx.pin_plan(&w.tails.last().unwrap().key, None).unwrap();
        // Chunk 3 (positions 3072..4096) is shared by both branches.
        s.idx.remove_chunk(&w.chunks[3], Why::Corrupt, 3);
        s.idx.check_invariants().unwrap();
        assert_eq!(s.idx.n_tails(), 0, "every tail beneath chunk 3 is gone");
        // The plan's pin was anchored beneath the corrupt chunk: retired with
        // it, so chunks 0..3 are left as orphans (evicted first).
        assert_eq!(s.idx.n_chunks(), 3);
        assert_eq!(s.idx.n_orphans(), 3);
        s.idx.unpin(pin, 4);
        assert_eq!(s.idx.n_orphans(), 3);
        s.idx.check_invariants().unwrap();
    }

    #[test]
    fn checker_matches_pin_records_to_their_owner_lists() {
        // 4.: job A owns two records; one is missing from A's list. The old
        // check passed (A still owned "some" record); membership must fail.
        let mut s = Sim::new(u64::MAX);
        let base = conv(16, 3000);
        s.admit(&base, 1, false, true);
        let k = s.idx.walk(&base, &[]).tails[0].key;
        let a = JobId(700);
        s.idx.pin_plan(&k, Some(a)).unwrap();
        let id2 = s.idx.pin_plan(&k, Some(a)).unwrap();
        s.idx.check_invariants().unwrap();
        s.idx.job_pins.get_mut(&a).unwrap().retain(|x| *x != id2);
        assert!(s.idx.check_invariants().is_err(), "a record missing from its owner's list must fail");
    }

    #[test]
    fn drift_saturates_and_flags_a_rescan() {
        let mut s = Sim::new(u64::MAX);
        let base = conv(15, 3000);
        s.admit(&base, 1, false, true);
        let c0 = s.idx.walk(&base, &[]).chunks[0];
        // Simulate drift: a refcount decrement nobody incremented.
        s.idx.add_path(&c0, -5, 0);
        assert!(s.idx.needs_rescan());
        assert_eq!(s.idx.chunk(&c0).unwrap().refs, 0, "saturated, not wrapped or panicked");
        assert!(s.idx.check_invariants().is_err(), "the checker sees the drift");
    }

    /// G5: refcounts, pins, orphans, scores and byte totals under random
    /// admissions with ASYNCHRONOUS completions (drained at random points,
    /// after jobs ended), dropped writes, Enc / Full / Anchor writes of the same
    /// key, images, pins, demotion, thinning, corrupt evictions and cap
    /// pressure, checked against a full rebuild after every step. Plus the
    /// policy invariants: pinned tails survive, anchors stay full and are never
    /// thinned, no pin outlives its job, no drift.
    #[test]
    fn randomized_refcounts_against_rebuild() {
        for seed in 1..=12u64 {
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let cap = rng.gen_range(20..200) * CHUNK_B;
            let mut s = Sim::new(cap);
            let roots: Vec<(Vec<i32>, Vec<ImageRecord>)> =
                (0..4).map(|i| (conv(seed * 100 + i, 30_000), if i % 2 == 0 { images_for(seed * 10 + i) } else { vec![] })).collect();
            let mut lineages: Vec<(usize, Vec<i32>)> = Vec::new();
            let mut held: Vec<(PinId, Key)> = Vec::new();
            let mut anchors: KeySet = KeySet::default();
            let mut anchors_seen = 0u32;
            let mut now = 1000u64;
            let (mut restored, mut steps, mut late) = (0u32, 0u32, 0u32);
            for step in 0..300 {
                now += rng.gen_range(1..4000);
                let op = rng.gen_range(0..100);
                if op < 55 || lineages.is_empty() {
                    // Extend an existing lineage, or start one from a shared root.
                    let (ri, tokens) = if !lineages.is_empty() && rng.gen_bool(0.7) {
                        let i = rng.gen_range(0..lineages.len());
                        let (ri, mut t) = lineages[i].clone();
                        let grow = rng.gen_range(1..4000);
                        let r = &roots[ri].0;
                        let end = (t.len() + grow).min(r.len());
                        t.extend_from_slice(&r[t.len().min(end)..end]);
                        if rng.gen_bool(0.2) && t.len() > 2000 {
                            let p = rng.gen_range(1000..t.len());
                            t[p] ^= 1; // a branch
                        }
                        lineages[i] = (ri, t.clone());
                        (ri, t)
                    } else {
                        let ri = rng.gen_range(0..roots.len());
                        // Sometimes exactly a K multiple: waypoint and prompt
                        // end share a key.
                        let l = if rng.gen_bool(0.2) { K as usize * rng.gen_range(1..3) } else { rng.gen_range(10..12_000) };
                        let t = roots[ri].0[..l].to_vec();
                        lineages.push((ri, t.clone()));
                        (ri, t)
                    };
                    let o = Opts {
                        fail: rng.gen_bool(0.05),
                        async_end: rng.gen_bool(0.5),
                        drop_p: if rng.gen_bool(0.2) { 0.15 } else { 0.0 },
                        anchor: rng.gen_bool(0.2),
                    };
                    let images = roots[ri].1.clone();
                    restored += s.admit_with(&tokens, &images, now, o, Some(&mut rng)).min(1);
                    late += o.async_end as u32;
                    steps += 1;
                } else if op < 62 {
                    // Land some of the in-flight writes.
                    let n = rng.gen_range(0..=s.inflight.len());
                    s.drain(n, now);
                } else if op < 72 {
                    // Pin a random tail as a restore in flight.
                    if let Some(k) = pick(&mut rng, s.idx.tails.keys()) {
                        let id = s.idx.pin_plan(&k, None).unwrap();
                        held.push((id, k));
                    }
                } else if op < 84 {
                    if !held.is_empty() {
                        let (id, _) = held.swap_remove(rng.gen_range(0..held.len()));
                        s.idx.unpin(id, now);
                    }
                } else if op < 90 {
                    // Retry of a lineage's prompt.
                    if !lineages.is_empty() {
                        let (ri, t) = lineages[rng.gen_range(0..lineages.len())].clone();
                        let images = roots[ri].1.clone();
                        s.admit_with(&t, &images, now, Opts { async_end: rng.gen_bool(0.5), ..Opts::default() }, Some(&mut rng));
                    }
                } else if op < 95 {
                    // A data-attributable read failure on a random chunk.
                    if let Some(k) = pick(&mut rng, s.idx.chunks.keys()) {
                        s.idx.remove_chunk(&k, Why::Corrupt, now);
                        // Tails under it went regardless of pins; their
                        // restores end (the unpin is then a no-op).
                        let (gone, keep): (Vec<_>, Vec<_>) = held.into_iter().partition(|(_, t)| s.idx.tail(t).is_none());
                        held = keep;
                        for (id, _) in gone {
                            s.idx.unpin(id, now);
                        }
                    }
                } else {
                    s.idx.set_cap_bytes(rng.gen_range(10..200) * CHUNK_B);
                }
                s.idx.enforce_cap(now);
                s.idx.take_actions();
                // The first removal of a key seen as an anchor is that anchor's
                // end: never thinning. (The key may come back as a plain tail.)
                for r in s.idx.take_removals() {
                    if anchors.remove(&r.key) {
                        assert_ne!(r.why, Why::Thinned, "seed {seed} step {step}: an anchor was thinned");
                    }
                }
                s.idx.check_invariants().unwrap_or_else(|e| panic!("seed {seed} step {step}: {e}"));
                assert!(!s.idx.needs_rescan(), "seed {seed} step {step}: counter drift");
                for (_, k) in &held {
                    // Pinned tails are never evicted, demoted or thinned
                    // (only a corrupt chunk beneath them removes them).
                    let e = s.idx.tail(k).unwrap_or_else(|| panic!("seed {seed} step {step}: a pinned tail vanished"));
                    assert!(e.pins > 0);
                }
                for (k, e) in &s.idx.tails {
                    assert!(!(e.anchor && e.demoted));
                    if e.anchor {
                        assert_eq!(e.kind, TailKind::Full, "seed {seed} step {step}: an anchor lost section D");
                        anchors_seen += anchors.insert(*k) as u32;
                    }
                }
            }
            s.drain_all(now);
            for (id, _) in held.drain(..) {
                s.idx.unpin(id, now);
            }
            s.idx.check_invariants().unwrap_or_else(|e| panic!("seed {seed} end: {e}"));
            assert!(s.idx.pins.is_empty() && s.idx.job_pins.is_empty(), "seed {seed}: pins leaked");
            assert!(s.idx.finished.is_empty(), "seed {seed}: finished jobs not forgotten");
            assert!(steps > 100 && restored > 20 && late > 30, "seed {seed}: not exercised ({steps}, {restored} warm, {late} async)");
            assert!(anchors_seen > 0 && s.refused > 0, "seed {seed}: no anchors ({anchors_seen}) or no refused tails ({})", s.refused);
        }
    }
}
