//! KV prefix store: chunked, content-addressed
//! (docs/v41/KV_PREFIX_STORE_DESIGN.md, rev 6).
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
//! - [`scan`]: startup (namespace GC, header scan, crash repair, rebuild);
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

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

use format::{BuildId, ChunkHeader, FormatError, GenPair, Provenance, StoreLayout, StoreRows, TailHeader, TailKind, TailOrigin};
use index::{Action, ChunkInsert, Index, JobId, PinId, Removal, Removed, TailInsert, Walk, WalkTail, Why};
use io::{ChunkWrite, Completion, IoHandle, IoJob, IoStats, NsDirs, TailWrite, WaitBudget};
use keys::{ChainCursor, ImageRecord, Key, KeyChain, NamespaceInputs};
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
/// How often the invariant checker runs in debug builds and in shadow (9.5).
pub const INVARIANT_CHECK_EVERY_S: u64 = 3600;

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
    /// The file is not indexed (evicted meanwhile).
    Missing,
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) => write!(f, "invalid store request: {s}"),
            Self::Corrupt { key, error } => write!(f, "corrupt file {}: {error}", keys::hex(&key[..8])),
            Self::Missing => f.write_str("not in the store"),
        }
    }
}

/// What happened to a write request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteOutcome {
    Queued,
    /// Already indexed (the first writer wins, E5).
    Stored,
    /// Another job's write of the same key is in flight: subscribe to its
    /// completion (9.4).
    Pending,
    /// The queue stayed full past the tick's wait budget (`kv.write_dropped`).
    Dropped,
}

/// Completion news for M2's jobs (subscriptions, pins, frontier).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreEvent {
    ChunkStored(Key),
    ChunkDropped(Key),
    TailStored(Key),
    /// `why`: "io" (the write failed) or "broken_path" (a chunk on the path
    /// was missing when the tail landed).
    TailDropped { key: Key, why: &'static str },
}

/// A chunk to write: chunk `k` of `tokens` (the REQUEST's ids, never a
/// reconstructed prefix: bug (a), 1.1).
pub struct ChunkWriteReq<'a> {
    pub tokens: &'a [i32],
    pub images: &'a [ImageRecord],
    pub k: u32,
    /// Store by store, rows then keys (4.1).
    pub payload: Vec<u8>,
    pub provenance: Provenance,
    pub job: Option<JobId>,
}

/// A tail to write at `t` of `tokens` (the request's ids).
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
    /// The walk's matched tails below `t` plus the job's own earlier tails.
    pub ancestors: Vec<Key>,
    pub job: Option<JobId>,
}

enum PendingWrite {
    Chunk(ChunkInsert),
    Tail(TailInsert),
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
    pub gens: Vec<(GenPair, index::GenStat)>,
    pub io: IoStats,
}

impl StoreStats {
    /// The `kv.store` line (8.6), key=value, ANSI-free.
    pub fn log_line(&self) -> String {
        let gens: Vec<String> = self.gens.iter().map(|(g, s)| format!("{g}:{}", s.files)).collect();
        format!(
            "kv.store chunks={} tails={} orphans={} chunk_mb={} tail_mb={} inactive_mb={} cap_mb={} pending={} \
             dropped={} evicted_mb={} written_mb={} trash_bytes={} trash_dirs={} gens={}",
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
            gens.join(",")
        )
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
    pending: HashMap<Key, PendingWrite>,
    dropped: u64,
    evicted_bytes: u64,
    last_check: u64,
    scan: ScanReport,
}

impl Store {
    /// Open (or create) the namespace of `ns`: GC, scan, rebuild, start the IO
    /// thread. Never waits on deletions.
    pub fn open(cfg: StoreConfig, ns: &NamespaceInputs, now: u64) -> std::io::Result<Self> {
        let ns_key = ns.key();
        let chain = KeyChain::new(ns_key);
        let chunk_rows: Vec<StoreRows> = ns.stores.iter().map(StoreRows::chunk_of).collect();
        let opened = scan::open_namespace(&cfg.root, &ns_key, cfg.cap_bytes, &chunk_rows, &cfg.purge, cfg.gen(), now)?;
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
            repaired = r.repaired_demotions,
            truncated = r.truncated,
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
            cfg,
            chain,
            io,
            pending: HashMap::new(),
            dropped: 0,
            evicted_bytes: 0,
            last_check: now,
        };
        s.index.enforce_cap(now);
        s.apply();
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

    /// Top of every scheduler tick: completions, then the hourly checker.
    pub fn tick(&mut self, now: u64) -> Vec<StoreEvent> {
        let ev = self.process_completions(now);
        if self.cfg.check_invariants && now >= self.last_check + INVARIANT_CHECK_EVERY_S {
            self.last_check = now;
            if let Err(e) = self.index.check_invariants() {
                tracing::error!(error = %e, "kv.invariant violated");
            }
        }
        ev
    }

    /// Apply the IO thread's completions to the index (9.4).
    pub fn process_completions(&mut self, now: u64) -> Vec<StoreEvent> {
        let done = self.io.completions();
        self.complete(done, now)
    }

    fn complete(&mut self, done: Vec<Completion>, now: u64) -> Vec<StoreEvent> {
        let mut ev = Vec::new();
        for c in done {
            match c {
                Completion::Written { key, tail, result } => {
                    let Some(p) = self.pending.remove(&key) else { continue };
                    match (p, result) {
                        (PendingWrite::Chunk(mut ins), Ok(bytes)) => {
                            ins.bytes = bytes;
                            self.index.insert_chunk(ins);
                            ev.push(StoreEvent::ChunkStored(key));
                        }
                        (PendingWrite::Tail(mut ins), Ok(bytes)) => {
                            ins.bytes = bytes;
                            match self.index.insert_tail(ins, now) {
                                Ok(()) => ev.push(StoreEvent::TailStored(key)),
                                Err(index::Refused::MissingAncestor) => {
                                    tracing::info!(key = %keys::hex(&key[..8]), "kv.write_dropped kind=tail why=broken_path");
                                    self.io.delete(self.dirs().tail_path(&key), bytes);
                                    self.dropped += 1;
                                    ev.push(StoreEvent::TailDropped { key, why: "broken_path" });
                                }
                            }
                        }
                        (_, Err(e)) => {
                            tracing::warn!(key = %keys::hex(&key[..8]), tail, error = %e, "kv.write_dropped why=io");
                            self.dropped += 1;
                            ev.push(if tail { StoreEvent::TailDropped { key, why: "io" } } else { StoreEvent::ChunkDropped(key) });
                        }
                    }
                }
                Completion::Corrupt { key, error } => {
                    tracing::warn!(key = %keys::hex(&key[..8]), error = %error, "kv.corrupt (demote)");
                    self.index.remove_tail(&key, Why::Corrupt, now);
                }
            }
        }
        self.index.enforce_cap(now);
        self.apply();
        ev
    }

    /// Hand the index's file work to the IO thread; log removals.
    fn apply(&mut self) {
        for a in self.index.take_actions() {
            match a {
                Action::UnlinkChunk { key, bytes } => self.io.delete(self.io.dirs().chunk_path(&key), bytes),
                Action::UnlinkTail { key, bytes } => self.io.delete(self.io.dirs().tail_path(&key), bytes),
                Action::Demote { key, hits } => {
                    tracing::info!(key = %keys::hex(&key[..8]), "kv.demote");
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

    /// Queue chunk `k` of the request (w1). The key comes from the job's
    /// cursor over the request ids (the same `chunk_step` the walk uses).
    pub fn write_chunk(&mut self, cur: &mut ChainCursor, req: ChunkWriteReq<'_>, budget: &mut WaitBudget, now: u64) -> Result<WriteOutcome, StoreError> {
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
        let gen = self.cfg.gen();
        let header = ChunkHeader {
            ns: *self.chain.ns(),
            key,
            parent,
            k: req.k,
            n_tokens: C as u16,
            n_images: images.len() as u16,
            stores: self.chunk_rows.clone(),
            provenance: req.provenance,
            gen: if req.provenance == Provenance::BackfillV6 { GenPair { gen: format::GEN_V6, ..gen } } else { gen },
            build: if req.provenance == Provenance::BackfillV6 { BuildId::V6 } else { self.cfg.build },
            created: now,
            payload_len: want,
            payload_hash: [0; 32],
        };
        let bytes = header.data_offset() + want;
        let ins = ChunkInsert { key, parent, k: req.k, bytes, created: now, gen: header.gen, job: req.job };
        let w = ChunkWrite { header, tokens: req.tokens[a as usize..(a + C) as usize].to_vec(), images, payload: req.payload };
        if self.io.submit_write(IoJob::Chunk(Box::new(w)), bytes, budget).is_err() {
            tracing::info!(k = req.k, "kv.write_dropped kind=chunk why=queue_full");
            self.dropped += 1;
            return Ok(WriteOutcome::Dropped);
        }
        self.pending.insert(key, PendingWrite::Chunk(ins));
        Ok(WriteOutcome::Queued)
    }

    /// Queue a tail at `t` (w2-w4). Skipped when the same key is stored or in
    /// flight with at least this kind; a full tail replaces an encoder tail at
    /// the same key (5.1) and keeps its hits and anchor flag.
    pub fn write_tail(&mut self, cur: &mut ChainCursor, req: TailWriteReq<'_>, budget: &mut WaitBudget, now: u64) -> Result<WriteOutcome, StoreError> {
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
        if let Some(PendingWrite::Tail(p)) = self.pending.get(&key) {
            if p.kind >= req.kind {
                return Ok(WriteOutcome::Pending);
            }
        }
        let a = t / C * C;
        let images: Vec<ImageRecord> = keys::images_in(req.images, a, t).to_vec();
        let anchor = req.origin == TailOrigin::Anchor || existing.is_some_and(|(_, an, _)| an);
        let hits = existing.map_or(0, |(_, _, h)| h);
        let gen = self.cfg.gen();
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
            ancestors: req.ancestors,
            job: req.job,
        };
        let w = TailWrite { header, open: req.tokens[a as usize..t as usize].to_vec(), images, sec_e: req.sec_e, sec_d: req.sec_d };
        if self.io.submit_write(IoJob::Tail(Box::new(w)), bytes, budget).is_err() {
            tracing::info!(t, kind = ?req.kind, "kv.write_dropped kind=tail why=queue_full");
            self.dropped += 1;
            return Ok(WriteOutcome::Dropped);
        }
        self.pending.insert(key, PendingWrite::Tail(ins));
        Ok(WriteOutcome::Queued)
    }

    /// A restore of `tail` (8.3): a hit, and its walk ancestors' paths young.
    pub fn touch(&mut self, tail: &Key, ancestors: &[Key], now: u64) {
        self.index.touch(tail, ancestors, now);
        self.apply();
    }

    /// Pin a restore plan's tail and chunk path until the restore completes.
    pub fn pin_plan(&mut self, tail: &Key) -> Option<PinId> {
        self.index.pin_plan(tail)
    }

    pub fn unpin(&mut self, id: PinId, now: u64) {
        self.index.unpin(id, now);
        self.apply();
    }

    /// A job ended; `failed` = an error, not a cancel (9.4).
    pub fn job_finished(&mut self, job: JobId, failed: bool, now: u64) {
        self.index.release_job(job, failed, now);
        self.index.enforce_cap(now);
        self.apply();
    }

    /// Read and verify chunk `k` of a plan against the request's ids. A file
    /// that fails is evicted with every tail beneath it (6.2) and reported.
    pub fn read_chunk(&mut self, key: &Key, expect: &[i32], now: u64) -> Result<format::ChunkFile, StoreError> {
        if self.index.chunk(key).is_none() {
            return Err(StoreError::Missing);
        }
        let r = format::read_chunk(&self.dirs().chunk_path(key), self.chain.ns()).and_then(|f| {
            if f.header.stores != self.chunk_rows {
                Err(FormatError::BadField("store shape"))
            } else if f.tokens != expect {
                Err(FormatError::BadField("token ids differ from the request"))
            } else {
                Ok(f)
            }
        });
        r.map_err(|error| {
            tracing::warn!(key = %keys::hex(&key[..8]), error = %error, "kv.corrupt chunk");
            self.index.remove_chunk(key, Why::Corrupt, now);
            self.apply();
            StoreError::Corrupt { key: *key, error }
        })
    }

    /// Read and verify a tail against the request's open ids; `want_d` reads
    /// section D (only when t ≤ 128 will use it, 6.6).
    pub fn read_tail(&mut self, key: &Key, expect_open: &[i32], want_d: bool, now: u64) -> Result<format::TailFile, StoreError> {
        if self.index.tail(key).is_none() {
            return Err(StoreError::Missing);
        }
        let stores = self.stores.clone();
        let r = format::read_tail(&self.dirs().tail_path(key), self.chain.ns(), want_d).and_then(|f| {
            if f.header.sec_e_len != format::section_e_len(&stores, f.header.t, f.header.n_raw) {
                Err(FormatError::BadField("section E shape"))
            } else if f.open != expect_open {
                Err(FormatError::BadField("token ids differ from the request"))
            } else {
                Ok(f)
            }
        });
        r.map_err(|error| {
            tracing::warn!(key = %keys::hex(&key[..8]), error = %error, "kv.corrupt tail");
            self.index.remove_tail(key, Why::Corrupt, now);
            self.apply();
            StoreError::Corrupt { key: *key, error }
        })
    }

    pub fn check_invariants(&self) -> Result<(), String> {
        self.index.check_invariants()
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
            pending_writes: self.pending.len() as u64,
            writes_dropped: self.dropped,
            evicted_bytes: self.evicted_bytes,
            gens: self.index.gens().collect(),
            io: self.io.stats(),
        }
    }

    /// Block until queued writes (and, with `all`, unlinks and trash) are done,
    /// then apply the completions. For tests and shutdown.
    pub fn flush(&mut self, all: bool, now: u64) -> Vec<StoreEvent> {
        let done = if all { self.io.flush_all() } else { self.io.flush() };
        let ev = self.complete(done, now);
        if all {
            // The completions may have queued more unlinks.
            let more = self.io.flush_all();
            let mut ev = ev;
            ev.extend(self.complete(more, now));
            return ev;
        }
        ev
    }

    /// Drain the write queue (9.4) and stop the IO thread.
    pub fn shutdown(self, delete_grace: Duration) {
        let n = self.pending.len();
        let _ = self.io.shutdown(delete_grace);
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
