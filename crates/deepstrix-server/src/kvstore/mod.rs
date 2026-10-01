//! KV prefix store: chunked, content-addressed
//! (docs/v41/KV_PREFIX_STORE_DESIGN.md, rev 7).
//!
//! The positional KV of a prompt is stored in chunks of C = 1024 positions,
//! each one file keyed by a blake3 chain over the token ids and written once,
//! shared by every prompt with that prefix. The state that is not positional
//! (windows, accumulators, the open rows, the DSpark ring) lives in tails at
//! prompt ends, every K = 8192 positions of a prefill (waypoints), at the
//! system-prefix anchor and on cancel. A lookup walks the chain over the
//! request and restores from the deepest usable tail (6).
//!
//! **M1 (design 13): the store without GPU and without the request path.**
//! Nothing in the server reaches this module yet: `V41_KV_STORE` defaults to
//! `off` and only M2 will read it. M2 adds the device-side capture/restore
//! (`het/kv_capture.rs`), the writes w1-w4, snapping and the shadow mode.
//!
//! Layout:
//! - [`keys`]: the namespace and the chunk / tail key chain (4.3, 4.4);
//! - [`format`]: chunk and tail files, two tail sections, the `anchor` flag;
//! - [`index`]: trie, tails, refcounts, pins, demotion, thinning, eviction,
//!   walk and selection, the invariant checker;
//! - [`io`]: the one IO thread (writes, mutations, unlinks, trash);
//! - [`scan`]: startup (root lock, namespace GC, header scan, crash repair,
//!   rebuild);
//! - [`knobs`]: the knob classification and the knob hash;
//! - [`Store`] (here): the scheduler-thread facade over all of them.

pub mod format;
pub mod index;
pub mod io;
pub mod keys;
pub mod knobs;
pub mod scan;
#[cfg(test)]
mod tests;

use std::collections::{HashMap, VecDeque};
use std::fs::File;
use std::path::{Path, PathBuf};
use std::time::Duration;

use format::{
    BuildId, ChunkHeader, ChunkMeta, FormatError, GenPair, Provenance, StoreLayout, StoreRows, TailHeader, TailKind, TailMeta,
    TailOrigin,
};
use index::{Action, ChunkInsert, Index, JobId, PinId, Removal, Removed, TailInsert, Walk, WalkTail, Why};
use io::{ChunkWrite, Completion, IoHandle, IoJob, IoStats, NsDirs, TailWrite, WaitBudget};
use keys::{ChainCursor, ImageRecord, Key, KeyChain, KeyMap, NamespaceInputs};
pub use scan::{PurgeSpec, ScanReport};

/// Positions per chunk (4.2). Equals the production prefill chunk
/// (`V41_MS_CHUNK_ROWS` = `B_MAX`), so cold prefills already end their chunks
/// at multiples of C. Part of the namespace.
pub const C: u32 = 1024;
/// Waypoint spacing (5.2): an encoder tail every K positions of a prefill.
pub const K: u32 = 8192;
/// Bumped only on an explicit decision (4.4.1): a fix that corrected what
/// stored KV contains, or a numerics change measured beyond the M0 null.
/// Bumping it starts a new namespace = cold-starts every conversation.
pub const KV_EPOCH: u32 = 1;
/// Bumped in the same commit as any change that alters prefill bits (4.4).
/// Recorded in headers only, never in the namespace: a bump cold-starts
/// nothing. A diff touching the prefill crates without bumping it says "not
/// bit-changing" in its commit message.
pub const KV_NUMERICS_GEN: u32 = 1;
/// Full tails kept full on one path (8.2): two cover "regenerate the last turn".
pub const DEMOTE_KEEP_FULL: usize = 2;
/// An encoder tail serves only a suffix longer than this (5.3): exactly when
/// `prefill_job_finish` discards the decoder windows anyway. = SWA_WINDOW.
pub const ENC_MIN_SUFFIX: u32 = 128;
/// No restore below this many positions (6.2).
pub const MIN_RESTORE_T: u32 = 64;
/// Eviction score weight of hits (8.3): score = path_last_used + 6 h × log2(1 + hits).
pub const SCORE_HIT_S: u64 = 6 * 3600;
/// How often the invariant checker runs in debug builds and in shadow (9.5),
/// and how often the namespace byte total is persisted.
pub const INVARIANT_CHECK_EVERY_S: u64 = 3600;
/// Recent removals, demotions and drops kept for the shadow reason codes
/// (11.1: "the store's own record of that event"), like the frontier LRU.
pub const RECENT_EVENTS: usize = 10_000;
/// Write buffers kept for reuse (a chunk payload is 2.83 MB; 8 ≈ 23 MB).
pub const BUFFER_POOL: usize = 8;

const _: () = assert!(K % C == 0);
const _: () = assert!(ENC_MIN_SUFFIX == v4flash_kernels::config::SWA_WINDOW);

/// `V41_KV_STORE` (11).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The default until M3: the store is not opened.
    Off,
    /// The old store serves; this one writes, walks, demotes and evicts, and
    /// logs its plan next to the actual restore. Restores nothing.
    Shadow,
    /// This store serves; the old one is neither read nor written.
    On,
}

impl Mode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "" | "off" | "0" => Some(Self::Off),
            "shadow" => Some(Self::Shadow),
            "on" | "1" => Some(Self::On),
            _ => None,
        }
    }

    /// Unset or unparsable = off.
    pub fn from_env() -> Self {
        match std::env::var("V41_KV_STORE") {
            Ok(v) => Self::parse(&v).unwrap_or_else(|| {
                tracing::warn!(value = %v, "V41_KV_STORE: want off|shadow|on; store stays off");
                Self::Off
            }),
            Err(_) => Self::Off,
        }
    }
}

#[derive(Debug, Clone)]
pub struct StoreConfig {
    /// `~/.cache/deepstrix/kvstore-v1`: every namespace lives below (4.4).
    pub root: PathBuf,
    /// One global cap over every namespace (`V41_KV_STORE_CAP_GIB`, 100).
    pub cap_bytes: u64,
    /// `V41_KV_WRITE_QUEUE_MB` (512).
    pub write_queue_bytes: u64,
    /// `V41_KV_WRITE_WAIT_MS` (200 per tick).
    pub write_wait: Duration,
    /// `V41_KV_STORE_PURGE_BUILD`, applied at open.
    pub purge: Vec<PurgeSpec>,
    pub build: BuildId,
    /// The knob hash of the launch env. Knobs re-read at run time (a job's
    /// resolved `V41_MS_LM_FILE` values, box 2's own knobs) are not in it:
    /// writes may carry their own (`ChunkWriteReq::knob_hash`), see `knobs`.
    pub knob_hash: [u8; 16],
    /// Run the invariant checker hourly (debug builds and shadow, 9.5).
    pub check_invariants: bool,
}

fn env_u64(k: &str, d: u64) -> u64 {
    std::env::var(k).ok().and_then(|v| v.trim().parse().ok()).unwrap_or(d)
}

impl StoreConfig {
    pub fn default_root() -> PathBuf {
        let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("."));
        home.join(".cache/deepstrix/kvstore-v1")
    }

    /// The configuration from the env. `build` is the binary's git sha (M2
    /// supplies it; no build-id mechanism exists yet).
    pub fn from_env(build: BuildId, mode: Mode) -> Result<Self, String> {
        let purge = match std::env::var("V41_KV_STORE_PURGE_BUILD") {
            Ok(v) => PurgeSpec::parse_list(&v)?,
            Err(_) => Vec::new(),
        };
        Ok(Self {
            root: Self::default_root(),
            cap_bytes: env_u64("V41_KV_STORE_CAP_GIB", 100) << 30,
            write_queue_bytes: env_u64("V41_KV_WRITE_QUEUE_MB", 512) << 20,
            write_wait: Duration::from_millis(env_u64("V41_KV_WRITE_WAIT_MS", 200)),
            purge,
            build,
            knob_hash: knobs::knob_hash_from_env(),
            check_invariants: cfg!(debug_assertions) || mode == Mode::Shadow,
        })
    }

    pub fn gen(&self) -> GenPair {
        GenPair { gen: KV_NUMERICS_GEN, knob: self.knob_hash }
    }
}

#[derive(Debug)]
pub enum StoreError {
    /// A request that cannot be right (wrong shape or position): a caller bug.
    Invalid(String),
    /// A stored file failed validation; it was evicted (6.2).
    Corrupt { key: Key, error: FormatError },
    /// A read failed for a reason that says nothing about the file (EIO,
    /// EMFILE, ENOMEM, EACCES...): nothing was evicted (`kv.suspect`).
    Io { key: Key, error: FormatError },
    /// The file is not indexed (evicted meanwhile).
    Missing,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid store request: {s}"),
            Self::Corrupt { key, error } => write!(f, "corrupt file {}: {error}", keys::hex(&key[..8])),
            Self::Io { key, error } => write!(f, "transient read failure on {}: {error}", keys::hex(&key[..8])),
            Self::Missing => f.write_str("not in the store"),
        }
    }
}

impl std::error::Error for StoreError {}

/// What happened to a write request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    Queued,
    /// Already indexed (the first writer wins, E5).
    Stored,
    /// Another write of the same key (at least this kind) is in flight:
    /// subscribe to its completion (9.4).
    Pending,
    /// The queue stayed full past the tick's wait budget (`kv.write_dropped`).
    Dropped,
}

/// Completion news for M2's jobs (subscriptions, pins, frontier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreEvent {
    /// The chunk is indexed (and still there when this is reported).
    ChunkStored(Key),
    /// The chunk's write failed or was dropped, OR an indexed chunk left the
    /// store (a failed job's chunk deleted at once, an eviction, a corrupt
    /// file): a job that counted on it re-enqueues it from its own state if
    /// it still can (9.4).
    ChunkDropped(Key),
    TailStored(Key),
    /// `why`: "io" (the write failed) or "broken_path" (a chunk on the path
    /// was missing when the tail landed).
    TailDropped { key: Key, why: &'static str },
}

/// The store's own record of what happened to a key recently (11.1 reason
/// codes: `dropped`, `evicted`, `thinned`, `demoted`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecentEvent {
    Removed(Why),
    Demoted,
    Dropped,
}

/// A chunk or a tail, for [`Store::evict_corrupt`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EntryKind {
    Chunk,
    Tail,
}

/// A chunk to write: chunk `k` of `tokens` (the REQUEST's ids, never a
/// reconstructed prefix: bug (a), 1.1).
pub struct ChunkWriteReq<'a> {
    pub tokens: &'a [i32],
    pub images: &'a [ImageRecord],
    pub k: u32,
    /// Store by store, rows then keys (4.1). Take it from [`Store::buffer`]:
    /// it comes back to the pool when the write completes.
    pub payload: Vec<u8>,
    pub provenance: Provenance,
    pub job: Option<JobId>,
    /// The knob hash of the values this job actually ran with, when they
    /// differ from the launch env's (knobs re-read at run time); `None` = the
    /// store's.
    pub knob_hash: Option<[u8; 16]>,
}

/// A tail to write at `t` of `tokens` (the request's ids). Its ancestors on
/// the path (for demotion, thinning, touches) are computed by the store.
pub struct TailWriteReq<'a> {
    pub tokens: &'a [i32],
    pub images: &'a [ImageRecord],
    pub t: u32,
    pub kind: TailKind,
    pub origin: TailOrigin,
    pub n_raw: u32,
    pub n_raw_dec: u32,
    pub drafter: [u8; 32],
    pub session_id: Option<&'a str>,
    pub sec_e: Vec<u8>,
    /// Empty for an encoder tail.
    pub sec_d: Vec<u8>,
    pub job: Option<JobId>,
    pub knob_hash: Option<[u8; 16]>,
}

#[derive(Debug, Clone)]
enum PendingWrite {
    Chunk(ChunkInsert),
    Tail(TailInsert),
}

impl PendingWrite {
    fn job(&self) -> Option<JobId> {
        match self {
            Self::Chunk(c) => c.job,
            Self::Tail(t) => t.job,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct StoreStats {
    pub chunks: u64,
    pub tails: u64,
    pub orphans: u64,
    pub chunk_bytes: u64,
    pub tail_bytes: u64,
    pub inactive_bytes: u64,
    pub cap_bytes: u64,
    pub pending_writes: u64,
    pub writes_dropped: u64,
    pub evicted_bytes: u64,
    pub needs_rescan: bool,
    pub gens: Vec<(GenPair, index::GenStat)>,
    pub io: IoStats,
}

impl StoreStats {
    /// The `kv.store` line (8.6), key=value, ANSI-free.
    pub fn log_line(&self) -> String {
        let gens: Vec<String> = self.gens.iter().map(|(g, s)| format!("{g}:{}", s.files)).collect();
        format!(
            "kv.store chunks={} tails={} orphans={} chunk_mb={} tail_mb={} inactive_mb={} cap_mb={} pending={} \
             dropped={} evicted_mb={} written_mb={} trash_bytes={} trash_dirs={} trash_errors={} rescan={} gens={}",
            self.chunks,
            self.tails,
            self.orphans,
            self.chunk_bytes >> 20,
            self.tail_bytes >> 20,
            self.inactive_bytes >> 20,
            self.cap_bytes >> 20,
            self.pending_writes,
            self.writes_dropped,
            self.evicted_bytes >> 20,
            self.io.written_bytes >> 20,
            self.io.delete_backlog_bytes,
            self.io.trash_dirs,
            self.io.trash_errors,
            self.needs_rescan as u8,
            gens.join(",")
        )
    }
}

/// A bounded memory of recent events per key, the least recently NOTED
/// forgotten first. A re-noted key moves to the back: `order` keeps a
/// sequence number per entry and stale entries are skipped (and compacted).
#[derive(Default)]
struct RecentLog {
    map: KeyMap<(RecentEvent, u64, u64)>,
    order: VecDeque<(Key, u64)>,
    seq: u64,
}

impl RecentLog {
    fn note(&mut self, key: Key, ev: RecentEvent, now: u64) {
        self.seq += 1;
        self.map.insert(key, (ev, now, self.seq));
        self.order.push_back((key, self.seq));
        while self.map.len() > RECENT_EVENTS {
            let Some((k, seq)) = self.order.pop_front() else { break };
            if self.map.get(&k).is_some_and(|e| e.2 == seq) {
                self.map.remove(&k);
            }
        }
        if self.order.len() > 2 * RECENT_EVENTS {
            let map = &self.map;
            self.order.retain(|(k, seq)| map.get(k).is_some_and(|e| e.2 == *seq));
        }
    }

    fn get(&self, key: &Key) -> Option<(RecentEvent, u64)> {
        self.map.get(key).map(|e| (e.0, e.1))
    }
}

/// The store, owned by the scheduler thread. Every method is cheap: index
/// updates and queue pushes; the IO thread does the files.
pub struct Store {
    cfg: StoreConfig,
    chain: KeyChain,
    stores: Vec<StoreLayout>,
    chunk_rows: Vec<StoreRows>,
    index: Index,
    io: IoHandle,
    /// Writes in flight per key, oldest first. A key can have more than one:
    /// an encoder tail and then the full tail at the same T (a waypoint at
    /// L % K == 0, an anchor on a K multiple, two jobs). Completions arrive in
    /// FIFO order, so each pops the front.
    pending: KeyMap<VecDeque<PendingWrite>>,
    /// Writes in flight per job: a finished job is forgotten when they drain.
    job_writes: HashMap<JobId, u32>,
    recent: RecentLog,
    /// Events produced outside a completion (an eviction, a failed job's
    /// chunks deleted), handed out with the next batch.
    events: Vec<StoreEvent>,
    pool: Vec<Vec<u8>>,
    dropped: u64,
    evicted_bytes: u64,
    last_check: u64,
    scan: ScanReport,
    /// The root lock (`kvstore-v1/.lock`), held for the store's lifetime.
    _lock: File,
}

impl Store {
    /// Open (or create) the namespace of `ns`: lock the root, GC, scan,
    /// rebuild, start the IO thread. Never waits on deletions. Fails (and the
    /// caller runs with the store off, logging why) if another process holds
    /// the root.
    pub fn open(cfg: StoreConfig, ns: &NamespaceInputs, now: u64) -> std::io::Result<Self> {
        let ns_key = ns.key();
        let chain = KeyChain::new(ns_key);
        let chunk_rows: Vec<StoreRows> = ns.stores.iter().map(StoreRows::chunk_of).collect();
        let opened = scan::open_namespace(&cfg.root, &ns_key, cfg.cap_bytes, &chunk_rows, &cfg.purge, &cfg.build, cfg.gen(), now)?;
        let dirs = NsDirs::new(cfg.root.join(keys::ns16(&ns_key)));
        let io = IoHandle::spawn(dirs, cfg.root.join("trash"), cfg.write_queue_bytes)?;
        for j in opened.jobs {
            io.submit(j);
        }
        for (p, b) in opened.unlinks {
            io.delete(p, b);
        }
        let r = &opened.report;
        tracing::info!(
            ns = %keys::ns16(&ns_key),
            chunks = r.chunks,
            tails = r.tails,
            mb = r.bytes >> 20,
            invalid = r.invalid,
            unreachable = r.unreachable,
            missing_ancestor = r.missing_ancestor,
            purged = r.purged,
            purge_refused = r.purge_refused,
            repaired = r.repaired_demotions,
            truncated = r.truncated,
            io_errors = r.io_errors,
            trashed_ns = r.trashed_namespaces,
            ms = r.elapsed_ms,
            "kv.scan"
        );
        if r.gen_new {
            let set: Vec<String> = knobs::numerics_knobs_set().into_iter().map(|(k, v)| format!("{k}={v}")).collect();
            tracing::info!(gen = %cfg.gen(), knobs = %set.join(" "), "kv.gen_new");
        }
        let mut s = Self {
            stores: ns.stores.clone(),
            chunk_rows,
            index: opened.index,
            scan: opened.report,
            _lock: opened.lock,
            cfg,
            chain,
            io,
            pending: KeyMap::default(),
            job_writes: HashMap::new(),
            recent: RecentLog::default(),
            events: Vec::new(),
            pool: Vec::new(),
            dropped: 0,
            evicted_bytes: 0,
            last_check: now,
        };
        s.index.enforce_cap(now);
        s.apply(now);
        Ok(s)
    }

    pub fn chain(&self) -> &KeyChain {
        &self.chain
    }
    pub fn index(&self) -> &Index {
        &self.index
    }
    pub fn scan_report(&self) -> &ScanReport {
        &self.scan
    }
    pub fn dirs(&self) -> &NsDirs {
        self.io.dirs()
    }
    /// A fresh wait budget for one scheduler tick (9.4).
    pub fn tick_budget(&self) -> WaitBudget {
        WaitBudget::new(self.cfg.write_wait)
    }
    /// Counter drift was seen: rebuild from disk (restart) when convenient.
    pub fn needs_rescan(&self) -> bool {
        self.index.needs_rescan()
    }
    /// The store's record of a recent removal, demotion or dropped write of
    /// `key` (shadow reason codes, 11.1).
    pub fn recent_event(&self, key: &Key) -> Option<(RecentEvent, u64)> {
        self.recent.get(key)
    }
    /// The stored payload blake3 of an indexed chunk (`kv.dedup_mismatch`).
    pub fn chunk_payload_hash(&self, key: &Key) -> Option<[u8; 32]> {
        self.index.chunk(key).map(|c| c.payload_hash)
    }
    /// A write buffer from the pool (cleared), or a new one.
    pub fn buffer(&mut self) -> Vec<u8> {
        let mut b = self.pool.pop().unwrap_or_default();
        b.clear();
        b
    }

    fn recycle(&mut self, bufs: Vec<Vec<u8>>) {
        for b in bufs {
            if self.pool.len() < BUFFER_POOL && b.capacity() > 0 {
                self.pool.push(b);
            }
        }
    }

    /// Top of every scheduler tick: completions, then the hourly checker and
    /// the namespace byte total.
    pub fn tick(&mut self, now: u64) -> Vec<StoreEvent> {
        let ev = self.process_completions(now);
        if now >= self.last_check + INVARIANT_CHECK_EVERY_S {
            self.last_check = now;
            self.persist_bytes();
            if self.cfg.check_invariants {
                if let Err(e) = self.check_invariants() {
                    tracing::error!(error = %e, "kv.invariant violated");
                }
            }
        }
        ev
    }

    /// Persist the namespace byte total (4.4) -- on the IO thread: the
    /// scheduler thread never touches the disk (4.5, 8.3).
    fn persist_bytes(&self) {
        let b = self.index.chunk_bytes() + self.index.tail_bytes();
        self.io.submit(IoJob::WriteSmall { path: self.dirs().base.join(scan::NS_BYTES), contents: format!("{b}\n") });
    }

    /// Apply the IO thread's completions to the index (9.4).
    pub fn process_completions(&mut self, now: u64) -> Vec<StoreEvent> {
        let done = self.io.completions();
        self.complete(done, now)
    }

    fn pop_pending(&mut self, key: &Key) -> Option<PendingWrite> {
        let q = self.pending.get_mut(key)?;
        let p = q.pop_front();
        if q.is_empty() {
            self.pending.remove(key);
        }
        p
    }

    /// One write of `job` is fully applied. Called AFTER its insert: a
    /// finished job forgotten before its last write is indexed would make
    /// that write pin for a job nobody releases.
    fn job_write_done(&mut self, job: Option<JobId>) {
        let Some(job) = job else { return };
        let n = self.job_writes.entry(job).or_default();
        *n = n.saturating_sub(1);
        if *n == 0 {
            self.job_writes.remove(&job);
            if self.index.finished(job).is_some() {
                self.index.forget_job(job);
            }
        }
    }

    /// The key has no write in flight and is not indexed: whatever file may
    /// be left at its path is garbage (an unlink skipped while a write was
    /// pending, 9.4) and goes.
    fn delete_if_unindexed(&mut self, key: &Key, tail: bool) {
        if self.pending.contains_key(key) {
            return;
        }
        if tail && self.index.tail(key).is_none() {
            self.io.delete(self.io.dirs().tail_path(key), 0);
        } else if !tail && self.index.chunk(key).is_none() {
            self.io.delete(self.io.dirs().chunk_path(key), 0);
        }
    }

    fn complete(&mut self, done: Vec<Completion>, now: u64) -> Vec<StoreEvent> {
        let mut ev = Vec::new();
        for c in done {
            match c {
                Completion::Written { key, tail, result, payload_hash, bufs } => {
                    self.recycle(bufs);
                    let Some(p) = self.pop_pending(&key) else { continue };
                    let job = p.job();
                    match (p, result) {
                        (PendingWrite::Chunk(mut ins), Ok(bytes)) => {
                            ins.bytes = bytes;
                            ins.payload_hash = payload_hash.unwrap_or_default();
                            self.index.insert_chunk(ins, now);
                            // A failed job's late chunk is deleted inside the
                            // insert: then it is reported dropped (by apply),
                            // never stored.
                            if self.index.chunk(&key).is_some() {
                                ev.push(StoreEvent::ChunkStored(key));
                            }
                        }
                        (PendingWrite::Tail(mut ins), Ok(bytes)) => {
                            ins.bytes = bytes;
                            match self.index.insert_tail(ins, now) {
                                Ok(()) => ev.push(StoreEvent::TailStored(key)),
                                Err(index::Refused::MissingAncestor) => {
                                    tracing::info!(key = %keys::hex(&key[..8]), "kv.write_dropped kind=tail why=broken_path");
                                    self.delete_if_unindexed(&key, true);
                                    self.dropped += 1;
                                    self.recent.note(key, RecentEvent::Dropped, now);
                                    ev.push(StoreEvent::TailDropped { key, why: "broken_path" });
                                }
                            }
                        }
                        (_, Err((kind, msg))) => {
                            tracing::warn!(key = %keys::hex(&key[..8]), tail, ?kind, error = %msg, "kv.write_dropped why=io");
                            self.dropped += 1;
                            self.recent.note(key, RecentEvent::Dropped, now);
                            self.delete_if_unindexed(&key, tail);
                            ev.push(if tail { StoreEvent::TailDropped { key, why: "io" } } else { StoreEvent::ChunkDropped(key) });
                        }
                    }
                    self.job_write_done(job);
                }
                Completion::MutationFailed { key, error } => {
                    if error.evicts() {
                        tracing::warn!(key = %keys::hex(&key[..8]), error = %error, "kv.corrupt demotion found a bad file");
                        self.index.remove_tail(&key, Why::Corrupt, now);
                    } else {
                        // The index already counts it as demoted; the file
                        // keeps section D until the next scan sees it.
                        tracing::warn!(key = %keys::hex(&key[..8]), error = %error, "kv.suspect demotion failed transiently");
                    }
                }
            }
        }
        // Events queued before this batch (outside a completion) come first.
        self.events.extend(ev);
        self.index.enforce_cap(now);
        self.apply(now);
        std::mem::take(&mut self.events)
    }

    /// Hand the index's file work to the IO thread and record removals. An
    /// unlink of a key with a write in flight is skipped: that write already
    /// ran or will run, and the file at the path is (or will be) the new one;
    /// if the write fails, `delete_if_unindexed` cleans up. Every chunk that
    /// leaves is reported as `ChunkDropped` with the next batch of events.
    fn apply(&mut self, now: u64) {
        for a in self.index.take_actions() {
            match a {
                Action::UnlinkChunk { key, bytes, why } => {
                    self.recent.note(key, RecentEvent::Removed(why), now);
                    self.events.push(StoreEvent::ChunkDropped(key));
                    if !self.pending.contains_key(&key) {
                        self.io.delete(self.io.dirs().chunk_path(&key), bytes);
                    }
                }
                Action::UnlinkTail { key, bytes, why } => {
                    self.recent.note(key, RecentEvent::Removed(why), now);
                    if !self.pending.contains_key(&key) {
                        self.io.delete(self.io.dirs().tail_path(&key), bytes);
                    }
                }
                Action::Demote { key, hits } => {
                    tracing::info!(key = %keys::hex(&key[..8]), "kv.demote");
                    self.recent.note(key, RecentEvent::Demoted, now);
                    self.io.submit(IoJob::Demote { path: self.io.dirs().tail_path(&key), key: Some(key), hits: Some(hits) })
                }
                Action::Touch { key, hits, last_used } => self.io.submit(IoJob::Touch { key, hits, last_used }),
                Action::TrashNamespace { ns16, .. } => self.io.submit(IoJob::Trash { from: self.cfg.root.join(ns16) }),
            }
        }
        for r in self.index.take_removals() {
            self.evicted_bytes += r.bytes + r.cascaded;
            log_removal(&r);
        }
    }

    /// The walk (6.1), after the writer's completions (so a retry right after
    /// the cancel that queued its tail finds it).
    pub fn walk(&mut self, tokens: &[i32], images: &[ImageRecord], now: u64) -> (Walk, Vec<StoreEvent>) {
        let ev = self.process_completions(now);
        (self.index.walk(tokens, images), ev)
    }

    /// Selection (6.2).
    pub fn select(walk: &Walk, l: u32, trailing_marker: bool) -> Option<WalkTail> {
        index::select(walk, l, trailing_marker)
    }

    /// Indexed, in flight, or neither.
    pub fn chunk_state(&self, key: &Key) -> WriteOutcome {
        if self.index.chunk(key).is_some() {
            WriteOutcome::Stored
        } else if self.pending.contains_key(key) {
            WriteOutcome::Pending
        } else {
            WriteOutcome::Queued
        }
    }

    fn gen_for(&self, knob: Option<[u8; 16]>) -> GenPair {
        GenPair { gen: KV_NUMERICS_GEN, knob: knob.unwrap_or(self.cfg.knob_hash) }
    }

    fn queue(&mut self, key: Key, p: PendingWrite) {
        if let Some(job) = p.job() {
            *self.job_writes.entry(job).or_default() += 1;
        }
        self.pending.entry(key).or_default().push_back(p);
    }

    /// Queue chunk `k` of the request (w1). The key comes from the job's
    /// cursor over the request ids (the same `chunk_step` the walk uses).
    pub fn write_chunk(&mut self, cur: &mut ChainCursor, req: ChunkWriteReq<'_>, budget: &mut WaitBudget, now: u64) -> Result<WriteOutcome, StoreError> {
        self.check_job_live(req.job)?;
        let a = req.k.checked_mul(C).ok_or_else(|| StoreError::Invalid("k overflows".into()))?;
        if (a + C) as usize > req.tokens.len() {
            return Err(StoreError::Invalid(format!("chunk {} past the request's {} ids", req.k, req.tokens.len())));
        }
        let want = format::chunk_payload_len(&self.stores);
        if req.payload.len() as u64 != want {
            return Err(StoreError::Invalid(format!("chunk payload {} B, layout says {want}", req.payload.len())));
        }
        let parent = cur.chain(req.tokens, req.images, req.k);
        let key = cur.chain(req.tokens, req.images, req.k + 1);
        if self.index.chunk(&key).is_some() {
            return Ok(WriteOutcome::Stored);
        }
        if self.pending.contains_key(&key) {
            return Ok(WriteOutcome::Pending);
        }
        let images: Vec<ImageRecord> = keys::images_in(req.images, a, a + C).to_vec();
        let gen = self.gen_for(req.knob_hash);
        let backfill = req.provenance == Provenance::BackfillV6;
        let header = ChunkHeader {
            ns: *self.chain.ns(),
            key,
            parent,
            k: req.k,
            n_tokens: C as u16,
            n_images: images.len() as u16,
            stores: self.chunk_rows.clone(),
            provenance: req.provenance,
            gen: if backfill { GenPair { gen: format::GEN_V6, ..gen } } else { gen },
            build: if backfill { BuildId::V6 } else { self.cfg.build },
            created: now,
            payload_len: want,
            payload_hash: [0; 32],
        };
        let bytes = header.data_offset() + want;
        let ins = ChunkInsert { key, parent, k: req.k, bytes, created: now, gen: header.gen, payload_hash: [0; 32], job: req.job };
        let w = ChunkWrite { header, tokens: req.tokens[a as usize..(a + C) as usize].to_vec(), images, payload: req.payload };
        if self.io.submit_write(IoJob::Chunk(Box::new(w)), bytes, budget).is_err() {
            tracing::info!(k = req.k, "kv.write_dropped kind=chunk why=queue_full");
            self.dropped += 1;
            self.recent.note(key, RecentEvent::Dropped, now);
            return Ok(WriteOutcome::Dropped);
        }
        self.queue(key, PendingWrite::Chunk(ins));
        Ok(WriteOutcome::Queued)
    }

    /// Queue a tail at `t` (w2-w4). Skipped when the same key is stored or in
    /// flight with at least this kind; a full tail replaces an encoder tail at
    /// the same key (5.1) and keeps its hits and anchor flag. The ancestors
    /// (8.2) are the indexed and in-flight tails on this request's path.
    pub fn write_tail(&mut self, cur: &mut ChainCursor, req: TailWriteReq<'_>, budget: &mut WaitBudget, now: u64) -> Result<WriteOutcome, StoreError> {
        self.check_job_live(req.job)?;
        let t = req.t;
        if t as usize > req.tokens.len() {
            return Err(StoreError::Invalid(format!("tail at {t} past the request's {} ids", req.tokens.len())));
        }
        let want_e = format::section_e_len(&self.stores, t, req.n_raw);
        // Encoder windows hold min(t, SWA_WINDOW) rows (6.5).
        let want_raw = t.min(v4flash_kernels::config::SWA_WINDOW);
        if req.sec_e.len() as u64 != want_e || req.n_raw != want_raw {
            return Err(StoreError::Invalid(format!(
                "section E {} B / n_raw {} at t={t}, layout says {want_e} B / {want_raw}",
                req.sec_e.len(),
                req.n_raw
            )));
        }
        match req.kind {
            TailKind::Enc if !req.sec_d.is_empty() || req.n_raw_dec != 0 => {
                return Err(StoreError::Invalid("an encoder tail carries no section D".into()));
            }
            TailKind::Full if req.sec_d.is_empty() => return Err(StoreError::Invalid("a full tail needs section D".into())),
            _ => {}
        }
        let (base, key) = self.chain.tail_key(cur, req.tokens, req.images, t);
        let existing = self.index.tail(&key).map(|e| (e.kind, e.anchor, e.hits));
        if existing.is_some_and(|(k, _, _)| k >= req.kind) {
            return Ok(WriteOutcome::Stored);
        }
        let in_flight = self.pending.get(&key).is_some_and(|q| q.iter().any(|p| matches!(p, PendingWrite::Tail(p) if p.kind >= req.kind)));
        if in_flight {
            return Ok(WriteOutcome::Pending);
        }
        let ancestors = self.ancestors(cur, req.tokens, req.images, t);
        let a = t / C * C;
        let images: Vec<ImageRecord> = keys::images_in(req.images, a, t).to_vec();
        let pending_anchor = self.pending.get(&key).is_some_and(|q| q.iter().any(|p| matches!(p, PendingWrite::Tail(p) if p.anchor)));
        let anchor = req.origin == TailOrigin::Anchor || existing.is_some_and(|(_, an, _)| an) || pending_anchor;
        let hits = existing.map_or(0, |(_, _, h)| h);
        let gen = self.gen_for(req.knob_hash);
        let header = TailHeader {
            ns: *self.chain.ns(),
            key,
            base,
            t,
            kind: req.kind,
            anchor,
            demoted: false,
            origin: req.origin,
            n_open: (t - a) as u16,
            n_images: images.len() as u16,
            n_raw: req.n_raw,
            n_raw_dec: req.n_raw_dec,
            drafter: req.drafter,
            hits,
            gen,
            build: self.cfg.build,
            created: now,
            sec_e_len: want_e,
            sec_e_hash: [0; 32],
            sec_d_len: req.sec_d.len() as u64,
            sec_d_hash: [0; 32],
            session_id: req.session_id.unwrap_or_default().to_string(),
        };
        let bytes = header.file_len();
        let ins = TailInsert {
            key,
            base,
            t,
            kind: req.kind,
            origin: req.origin,
            anchor,
            demoted: false,
            bytes,
            sec_d_bytes: req.sec_d.len() as u64,
            created: now,
            gen,
            hits,
            ancestors,
            job: req.job,
        };
        let w = TailWrite { header, open: req.tokens[a as usize..t as usize].to_vec(), images, sec_e: req.sec_e, sec_d: req.sec_d };
        if self.io.submit_write(IoJob::Tail(Box::new(w)), bytes, budget).is_err() {
            tracing::info!(t, kind = ?req.kind, "kv.write_dropped kind=tail why=queue_full");
            self.dropped += 1;
            self.recent.note(key, RecentEvent::Dropped, now);
            return Ok(WriteOutcome::Dropped);
        }
        self.queue(key, PendingWrite::Tail(ins));
        Ok(WriteOutcome::Queued)
    }

    /// The tails on the path of a tail at `t` of this request: indexed ones
    /// (the walk's matching over the cursor's chain keys, no chunk re-hashed)
    /// and in-flight ones (a job's own waypoints land before its prompt end:
    /// FIFO).
    fn ancestors(&self, cur: &mut ChainCursor, tokens: &[i32], images: &[ImageRecord], t: u32) -> Vec<Key> {
        let chain = cur.keys_to(tokens, images, t / C).to_vec();
        let mut anc = self.index.ancestors_on_path(&chain, tokens, images, t);
        for (pk, q) in &self.pending {
            for p in q {
                let PendingWrite::Tail(p) = p else { continue };
                if p.t >= t || anc.contains(pk) {
                    continue;
                }
                let b = (p.t / C) as usize;
                let a = b as u32 * C;
                if chain[b] == p.base && keys::tail_step(&chain[b], a, &tokens[a as usize..p.t as usize], keys::images_in(images, a, p.t)) == *pk {
                    anc.push(*pk);
                }
            }
        }
        anc
    }

    /// A restore of `tail` (8.3): a hit, and its walk ancestors' paths young.
    pub fn touch(&mut self, tail: &Key, ancestors: &[Key], now: u64) {
        self.index.touch(tail, ancestors, now);
        self.apply(now);
    }

    /// Pin a restore plan's tail and chunk path until the restore completes;
    /// owned by `job`, it also goes when the job ends.
    pub fn pin_plan(&mut self, tail: &Key, job: Option<JobId>) -> Option<PinId> {
        self.index.pin_plan(tail, job)
    }

    pub fn unpin(&mut self, id: PinId, now: u64) {
        self.index.unpin(id, now);
        self.apply(now);
    }

    /// A job ended; `failed` = an error, not a cancel (9.4). It may be called
    /// while writes of the job are in flight: they land unowned, and a failed
    /// job's late chunks go at once if nothing references them. Returns the
    /// events this produced (a failed job's deleted chunks as `ChunkDropped`,
    /// for the other jobs that counted on them).
    ///
    /// Contract: a job writes nothing after this call (refused while the store
    /// still remembers the job; once forgotten, such a write would pin for a
    /// job nobody releases).
    pub fn job_finished(&mut self, job: JobId, failed: bool, now: u64) -> Vec<StoreEvent> {
        self.index.release_job(job, failed, now);
        if !self.job_writes.contains_key(&job) {
            self.index.forget_job(job);
        }
        self.index.enforce_cap(now);
        self.apply(now);
        std::mem::take(&mut self.events)
    }

    fn check_job_live(&self, job: Option<JobId>) -> Result<(), StoreError> {
        match job {
            Some(j) if self.index.finished(j).is_some() => Err(StoreError::Invalid(format!("write from finished job {j:?}"))),
            _ => Ok(()),
        }
    }

    /// Evict a file for a data-attributable failure the caller found (shape,
    /// ABI, a restore-side check), with everything beneath it (6.2).
    pub fn evict_corrupt(&mut self, key: &Key, kind: EntryKind, now: u64) {
        match kind {
            EntryKind::Chunk => self.index.remove_chunk(key, Why::Corrupt, now),
            EntryKind::Tail => {
                self.index.remove_tail(key, Why::Corrupt, now);
            }
        }
        self.apply(now);
    }

    fn read_failed(&mut self, key: &Key, kind: EntryKind, error: FormatError, now: u64) -> StoreError {
        if error.evicts() {
            tracing::warn!(key = %keys::hex(&key[..8]), ?kind, error = %error, "kv.corrupt");
            self.evict_corrupt(key, kind, now);
            StoreError::Corrupt { key: *key, error }
        } else {
            tracing::warn!(key = %keys::hex(&key[..8]), ?kind, error = %error, "kv.suspect read failed; file kept");
            StoreError::Io { key: *key, error }
        }
    }

    /// Read and verify an indexed chunk into `payload` (a reused staging
    /// buffer) and check its ids against the request's. A file that fails a
    /// data check is evicted with every tail beneath it (6.2); a transient IO
    /// failure evicts nothing.
    pub fn read_chunk_into(&mut self, key: &Key, expect: &[i32], payload: &mut Vec<u8>, now: u64) -> Result<ChunkMeta, StoreError> {
        if self.index.chunk(key).is_none() {
            return Err(StoreError::Missing);
        }
        let r = format::read_chunk_into(&self.dirs().chunk_path(key), self.chain.ns(), Some(key), payload).and_then(|m| {
            if m.header.stores != self.chunk_rows {
                Err(FormatError::BadField("store shape"))
            } else if m.tokens != expect {
                Err(FormatError::BadField("token ids differ from the request"))
            } else {
                Ok(m)
            }
        });
        r.map_err(|e| self.read_failed(key, EntryKind::Chunk, e, now))
    }

    pub fn read_chunk(&mut self, key: &Key, expect: &[i32], now: u64) -> Result<format::ChunkFile, StoreError> {
        let mut payload = Vec::new();
        let m = self.read_chunk_into(key, expect, &mut payload, now)?;
        Ok(format::ChunkFile { header: m.header, tokens: m.tokens, images: m.images, payload })
    }

    /// Read and verify an indexed tail against the request's open ids;
    /// `want_d` reads section D (only a t ≤ 128 restore uses it, 6.6) and is
    /// refused for a tail the index holds as encoder-only.
    pub fn read_tail_into(
        &mut self,
        key: &Key,
        expect_open: &[i32],
        want_d: bool,
        sec_e: &mut Vec<u8>,
        sec_d: &mut Vec<u8>,
        now: u64,
    ) -> Result<TailMeta, StoreError> {
        let Some(e) = self.index.tail(key) else { return Err(StoreError::Missing) };
        if want_d && e.kind != TailKind::Full {
            return Err(StoreError::Invalid("section D asked of an encoder tail".into()));
        }
        let stores = &self.stores;
        let r = format::read_tail_into(&self.io.dirs().tail_path(key), self.chain.ns(), Some(key), want_d, sec_e, sec_d).and_then(|m| {
            let h = &m.header;
            if h.n_raw != h.t.min(v4flash_kernels::config::SWA_WINDOW) || h.sec_e_len != format::section_e_len(stores, h.t, h.n_raw) {
                Err(FormatError::BadField("section E shape"))
            } else if m.open != expect_open {
                Err(FormatError::BadField("token ids differ from the request"))
            } else {
                Ok(m)
            }
        });
        r.map_err(|e| self.read_failed(key, EntryKind::Tail, e, now))
    }

    pub fn read_tail(&mut self, key: &Key, expect_open: &[i32], want_d: bool, now: u64) -> Result<format::TailFile, StoreError> {
        let (mut e, mut d) = (Vec::new(), Vec::new());
        let m = self.read_tail_into(key, expect_open, want_d, &mut e, &mut d, now)?;
        Ok(format::TailFile { header: m.header, open: m.open, images: m.images, sec_e: e, sec_d: want_d.then_some(d) })
    }

    /// The index invariants, plus the store's own: every pending entry and
    /// every job write counter agree, and no finished job owns a pin.
    pub fn check_invariants(&self) -> Result<(), String> {
        self.index.check_invariants()?;
        let mut per_job: HashMap<JobId, u32> = HashMap::new();
        for q in self.pending.values() {
            if q.is_empty() {
                return Err("empty pending queue left in the map".into());
            }
            for p in q {
                if let Some(j) = p.job() {
                    *per_job.entry(j).or_default() += 1;
                }
            }
        }
        if per_job != self.job_writes {
            return Err(format!("job write counters drifted: {:?} vs {:?}", self.job_writes, per_job));
        }
        Ok(())
    }

    pub fn stats(&self) -> StoreStats {
        StoreStats {
            chunks: self.index.n_chunks() as u64,
            tails: self.index.n_tails() as u64,
            orphans: self.index.n_orphans() as u64,
            chunk_bytes: self.index.chunk_bytes(),
            tail_bytes: self.index.tail_bytes(),
            inactive_bytes: self.index.inactive_bytes(),
            cap_bytes: self.index.cap_bytes(),
            pending_writes: self.pending.values().map(|q| q.len() as u64).sum(),
            writes_dropped: self.dropped,
            evicted_bytes: self.evicted_bytes,
            needs_rescan: self.index.needs_rescan(),
            gens: self.index.gens().collect(),
            io: self.io.stats(),
        }
    }

    /// Block until queued writes (and, with `all`, unlinks and trash) are done,
    /// then apply the completions. For tests and shutdown.
    pub fn flush(&mut self, all: bool, now: u64) -> Vec<StoreEvent> {
        let done = if all { self.io.flush_all() } else { self.io.flush() };
        let mut ev = self.complete(done, now);
        if all {
            // The completions may have queued more unlinks.
            let more = self.io.flush_all();
            ev.extend(self.complete(more, now));
        }
        ev
    }

    /// Drain the write queue (9.4), persist the byte total, stop the IO thread.
    pub fn shutdown(mut self, delete_grace: Duration, now: u64) {
        let n: usize = self.pending.values().map(|q| q.len()).sum();
        let done = self.io.flush();
        self.complete(done, now);
        self.persist_bytes();
        let Self { io, .. } = self;
        let _ = io.shutdown(delete_grace);
        tracing::info!(drained = n, "kv.store shutdown");
    }

    pub fn root(&self) -> &Path {
        &self.cfg.root
    }
}

fn log_removal(r: &Removal) {
    match r.what {
        Removed::Tail { t, kind } => tracing::info!(
            kind = ?kind,
            t,
            freed = r.bytes,
            cascaded = r.cascaded,
            age_h = r.age_s / 3600,
            hits = r.hits,
            why = r.why.as_str(),
            "kv.evict"
        ),
        Removed::Chunk { k } => {
            tracing::info!(kind = "chunk", k, freed = r.bytes, cascaded = r.cascaded, age_h = r.age_s / 3600, why = r.why.as_str(), "kv.evict")
        }
        Removed::Namespace => tracing::info!(kind = "namespace", freed = r.bytes, why = r.why.as_str(), "kv.evict"),
    }
}
