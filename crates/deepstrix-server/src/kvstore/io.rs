//! The IO thread (design 9.4, 9.5, 4.5, 8.3).
//!
//! One thread does every file operation of the store, so the scheduler thread
//! only ever pays a device-to-host copy (M2) and an in-memory index update:
//! - **writes**: checksums, `tmp/` then `rename`, no fsync (this is a cache;
//!   the startup scan and the per-read checks catch what a crash leaves);
//! - **mutations of existing files**, in FIFO order with the writes: the hits
//!   pwrite + mtime of a touch, the truncate + header rewrite of a demotion,
//!   startup repairs. One thread applying them in order means a demotion's
//!   header rewrite can never lose a concurrent hits update (4.5);
//! - **deletions**, on a separate queue served only when no write or mutation
//!   is waiting ("behind pending writes", 8.3), and the background removal of
//!   `trash/` (stale namespaces, the old `tmp/`), so neither startup nor the
//!   scheduler ever waits on unlinks.
//!
//! Backpressure (9.4): the write FIFO is bounded in bytes
//! (`V41_KV_WRITE_QUEUE_MB`). A full queue makes the scheduler wait up to its
//! per-tick budget (`V41_KV_WRITE_WAIT_MS`) for room, then the write is
//! dropped and the caller logs `kv.write_dropped`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::os::unix::fs::{DirBuilderExt, FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use super::format::{self, ChunkHeader, TailHeader, TailKind, TAIL_HEADER_LEN, TAIL_HITS_OFFSET};
use super::keys::{self, ImageRecord, Key};

/// Create `dir` and its missing parents with mode 0700: the files contain
/// prompts (4.4, risk 11).
pub fn mkdir_private(dir: &Path) -> io::Result<()> {
    fs::DirBuilder::new().recursive(true).mode(0o700).create(dir)
}

/// The directories of one namespace: `kvstore-v1/<ns16>/{chunks,tails,tmp}`.
#[derive(Debug, Clone)]
pub struct NsDirs {
    pub base: PathBuf,
}

impl NsDirs {
    pub fn new(base: PathBuf) -> Self {
        Self { base }
    }
    /// `chunks/<h2>/<key>.kvc` (4.5): 256 fan-out directories.
    pub fn chunk_path(&self, key: &Key) -> PathBuf {
        let h = keys::hex(key);
        self.base.join("chunks").join(&h[..2]).join(format!("{h}.kvc"))
    }
    pub fn tail_path(&self, key: &Key) -> PathBuf {
        let h = keys::hex(key);
        self.base.join("tails").join(&h[..2]).join(format!("{h}.kvt"))
    }
    pub fn tmp(&self) -> PathBuf {
        self.base.join("tmp")
    }
    pub fn create(&self) -> io::Result<()> {
        mkdir_private(&self.base.join("chunks"))?;
        mkdir_private(&self.base.join("tails"))?;
        mkdir_private(&self.tmp())
    }
}

pub struct ChunkWrite {
    /// `payload_hash` is filled in by the IO thread.
    pub header: ChunkHeader,
    pub tokens: Vec<i32>,
    /// Absolute positions.
    pub images: Vec<ImageRecord>,
    pub payload: Vec<u8>,
}

pub struct TailWrite {
    /// The section hashes are filled in by the IO thread.
    pub header: TailHeader,
    pub open: Vec<i32>,
    pub images: Vec<ImageRecord>,
    pub sec_e: Vec<u8>,
    /// Empty for an encoder tail.
    pub sec_d: Vec<u8>,
}

pub enum IoJob {
    Chunk(Box<ChunkWrite>),
    Tail(Box<TailWrite>),
    /// pwrite `hits`, set the mtime (8.3).
    Touch { key: Key, hits: u32, last_used: u64 },
    /// Truncate section D, then rewrite the header as demoted, carrying the
    /// scheduler's in-memory hits (8.2, 4.5). `hits: None` keeps the file's
    /// (startup repair of a crash between the two steps).
    Demote { path: PathBuf, key: Option<Key>, hits: Option<u32> },
    /// Startup repair: a file longer than its sections.
    Truncate { path: PathBuf, len: u64 },
    /// Rename a directory into `trash/` and remove it in the background.
    Trash { from: PathBuf },
    /// Signalled when every earlier main job is done.
    Barrier(mpsc::SyncSender<()>),
    /// Signalled when every earlier main job AND all deletions and trash are
    /// done (tests, shutdown accounting).
    BarrierAll(mpsc::SyncSender<()>),
}

#[derive(Debug, Clone)]
pub enum Completion {
    /// A chunk or tail write finished: `Ok(file bytes)` or the error (the tmp
    /// file is removed; nothing was renamed). The write's buffers come back
    /// for reuse; `payload_hash` is the stored chunk payload's blake3.
    Written {
        key: Key,
        tail: bool,
        result: Result<u64, (io::ErrorKind, String)>,
        payload_hash: Option<[u8; 32]>,
        bufs: Vec<Vec<u8>>,
    },
    /// A demotion failed on this tail. `error.evicts()` says whether the file
    /// itself is bad (evict it) or the failure was transient (`kv.suspect`).
    MutationFailed { key: Key, error: format::FormatError },
}

/// The write FIFO stayed full past the tick's wait budget: the write was
/// dropped (`kv.write_dropped`, 9.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueFull;

impl std::fmt::Display for QueueFull {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("kvstore write queue full")
    }
}

impl std::error::Error for QueueFull {}

/// Attempts at removing a trash directory before it is left alone (a file in
/// it cannot be unlinked); it is then counted in `IoStats::trash_errors`.
const TRASH_ATTEMPTS: u32 = 3;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct IoStats {
    /// Bytes waiting in the write FIFO.
    pub queued_bytes: u64,
    /// The unlink backlog (`kv.store trash_bytes`).
    pub delete_backlog_bytes: u64,
    pub delete_backlog_files: u64,
    /// Directories in `trash/` not yet removed.
    pub trash_dirs: u64,
    pub written_files: u64,
    pub written_bytes: u64,
    pub write_errors: u64,
    pub unlinked_files: u64,
    pub unlinked_bytes: u64,
    pub unlink_errors: u64,
    pub trash_files_removed: u64,
    /// Trash directories given up on (something in them cannot be removed).
    pub trash_errors: u64,
}

struct Queues {
    main: VecDeque<(IoJob, u64)>,
    cap_bytes: u64,
    /// FIFO of paths to unlink; an entry whose path is no longer in
    /// `delete_set` was cancelled by a later write to the same path.
    deletes: VecDeque<PathBuf>,
    delete_set: HashMap<PathBuf, u64>,
    trash: Vec<PathBuf>,
    /// Failed removals per trash directory, and the ones given up on (their
    /// parents skip them instead of pushing them again).
    trash_attempts: HashMap<PathBuf, u32>,
    trash_failed: HashSet<PathBuf>,
    barriers_all: Vec<mpsc::SyncSender<()>>,
    shutdown: Option<Instant>,
    stats: IoStats,
    /// Test fault injection: per queued write, true = fail it.
    #[cfg(test)]
    faults: VecDeque<bool>,
}

struct Shared {
    q: Mutex<Queues>,
    work: Condvar,
    room: Condvar,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, Queues> {
        self.q.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// How long the scheduler may still wait for queue room in this tick
/// (`V41_KV_WRITE_WAIT_MS` per tick, 9.4).
#[derive(Debug, Clone, Copy)]
pub struct WaitBudget {
    pub remaining: Duration,
}

impl WaitBudget {
    pub fn new(per_tick: Duration) -> Self {
        Self { remaining: per_tick }
    }
}

/// The scheduler's handle on the IO thread.
pub struct IoHandle {
    shared: Arc<Shared>,
    thread: Option<std::thread::JoinHandle<()>>,
    completions: mpsc::Receiver<Completion>,
    dirs: NsDirs,
}

impl IoHandle {
    /// Start the IO thread for one namespace. `trash_root` is
    /// `kvstore-v1/trash/`.
    pub fn spawn(dirs: NsDirs, trash_root: PathBuf, queue_cap_bytes: u64) -> io::Result<Self> {
        let shared = Arc::new(Shared {
            q: Mutex::new(Queues {
                main: VecDeque::new(),
                cap_bytes: queue_cap_bytes,
                deletes: VecDeque::new(),
                delete_set: HashMap::new(),
                trash: Vec::new(),
                trash_attempts: HashMap::new(),
                trash_failed: HashSet::new(),
                barriers_all: Vec::new(),
                shutdown: None,
                stats: IoStats::default(),
                #[cfg(test)]
                faults: VecDeque::new(),
            }),
            work: Condvar::new(),
            room: Condvar::new(),
        });
        let (tx, rx) = mpsc::channel();
        let worker = Worker { shared: shared.clone(), dirs: dirs.clone(), trash_root, tx, seq: 0 };
        let thread = std::thread::Builder::new().name("kvstore-io".into()).spawn(move || worker.run())?;
        Ok(Self { shared, thread: Some(thread), completions: rx, dirs })
    }

    pub fn dirs(&self) -> &NsDirs {
        &self.dirs
    }

    /// Queue a write. Waits for room up to `budget` (which it decrements),
    /// then gives up: `Err` means dropped (`kv.write_dropped`).
    pub fn submit_write(&self, job: IoJob, bytes: u64, budget: &mut WaitBudget) -> Result<(), QueueFull> {
        let start = Instant::now();
        let deadline = start + budget.remaining;
        let mut q = self.shared.lock();
        // An empty queue always takes the write, even one larger than the cap.
        while !q.main.is_empty() && q.stats.queued_bytes + bytes > q.cap_bytes {
            let now = Instant::now();
            if now >= deadline {
                budget.remaining = Duration::ZERO;
                return Err(QueueFull);
            }
            q = self.shared.room.wait_timeout(q, deadline - now).unwrap_or_else(|e| e.into_inner()).0;
        }
        budget.remaining = budget.remaining.saturating_sub(start.elapsed());
        q.stats.queued_bytes += bytes;
        q.main.push_back((job, bytes));
        self.shared.work.notify_one();
        Ok(())
    }

    /// Test fault injection: the next writes the worker runs fail (true) or
    /// succeed (false), in order.
    #[cfg(test)]
    pub fn inject_write_faults(&self, pattern: &[bool]) {
        self.shared.lock().faults.extend(pattern.iter().copied());
    }

    /// Queue a mutation (never dropped: they are tiny).
    pub fn submit(&self, job: IoJob) {
        let mut q = self.shared.lock();
        q.main.push_back((job, 0));
        self.shared.work.notify_one();
    }

    /// Queue an unlink behind the pending writes.
    pub fn delete(&self, path: PathBuf, bytes: u64) {
        let mut q = self.shared.lock();
        if q.delete_set.insert(path.clone(), bytes).is_none() {
            q.stats.delete_backlog_bytes += bytes;
            q.stats.delete_backlog_files += 1;
            q.deletes.push_back(path);
            self.shared.work.notify_one();
        }
    }

    /// Completions since the last call (processed at the top of every
    /// scheduler tick and before every walk, 9.4).
    pub fn completions(&self) -> Vec<Completion> {
        self.completions.try_iter().collect()
    }

    /// Block until every completion queued before this call has been sent.
    pub fn flush(&self) -> Vec<Completion> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.submit(IoJob::Barrier(tx));
        let _ = rx.recv();
        self.completions()
    }

    /// Block until the write FIFO, the deletion queue and the trash are empty.
    pub fn flush_all(&self) -> Vec<Completion> {
        let (tx, rx) = mpsc::sync_channel(1);
        self.submit(IoJob::BarrierAll(tx));
        let _ = rx.recv();
        self.completions()
    }

    pub fn stats(&self) -> IoStats {
        self.shared.lock().stats
    }

    /// Drain the write FIFO (bounded by its size, ≤ 512 MB, 9.4), then spend
    /// at most `delete_grace` on the unlink backlog. Unlinks left undone leave
    /// valid files that the next startup scan indexes again.
    pub fn shutdown(mut self, delete_grace: Duration) -> Vec<Completion> {
        {
            let mut q = self.shared.lock();
            q.shutdown = Some(Instant::now() + delete_grace);
            self.shared.work.notify_all();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        self.completions.try_iter().collect()
    }
}

impl Drop for IoHandle {
    fn drop(&mut self) {
        if let Some(t) = self.thread.take() {
            {
                let mut q = self.shared.lock();
                q.shutdown = Some(Instant::now());
                self.shared.work.notify_all();
            }
            let _ = t.join();
        }
    }
}

enum Work {
    Main(IoJob, u64),
    Delete(Vec<(PathBuf, u64)>),
    Trash,
    Exit,
}

struct Worker {
    shared: Arc<Shared>,
    dirs: NsDirs,
    trash_root: PathBuf,
    tx: mpsc::Sender<Completion>,
    seq: u64,
}

/// Unlinks per batch before the worker looks at the write FIFO again.
const DELETE_BATCH: usize = 64;

impl Worker {
    fn run(mut self) {
        loop {
            let work = {
                let mut q = self.shared.lock();
                loop {
                    if let Some((job, bytes)) = q.main.pop_front() {
                        break Work::Main(job, bytes);
                    }
                    let past_grace = q.shutdown.is_some_and(|d| Instant::now() >= d);
                    if !q.deletes.is_empty() && !past_grace {
                        let mut batch = Vec::new();
                        while batch.len() < DELETE_BATCH {
                            let Some(p) = q.deletes.pop_front() else { break };
                            if let Some(b) = q.delete_set.remove(&p) {
                                batch.push((p, b));
                            }
                        }
                        break Work::Delete(batch);
                    }
                    if !q.trash.is_empty() && q.shutdown.is_none() {
                        break Work::Trash;
                    }
                    if q.deletes.is_empty() && (q.trash.is_empty() || q.shutdown.is_some()) {
                        for b in q.barriers_all.drain(..) {
                            let _ = b.send(());
                        }
                    }
                    if q.shutdown.is_some() {
                        break Work::Exit;
                    }
                    q = self.shared.work.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            match work {
                Work::Main(job, bytes) => {
                    self.main(job);
                    let mut q = self.shared.lock();
                    q.stats.queued_bytes -= bytes;
                    self.shared.room.notify_all();
                }
                Work::Delete(batch) => self.unlink_batch(batch),
                Work::Trash => self.trash_step(),
                Work::Exit => return,
            }
        }
    }

    fn main(&mut self, job: IoJob) {
        match job {
            IoJob::Chunk(mut w) => {
                let key = w.header.key;
                let r = if self.injected_fault() { Err(io::Error::other("injected write fault")) } else { self.write_chunk(&mut w) };
                let hash = r.is_ok().then_some(w.header.payload_hash);
                self.written(key, false, r, hash, vec![std::mem::take(&mut w.payload)]);
            }
            IoJob::Tail(mut w) => {
                let key = w.header.key;
                let r = if self.injected_fault() { Err(io::Error::other("injected write fault")) } else { self.write_tail(&mut w) };
                self.written(key, true, r, None, vec![std::mem::take(&mut w.sec_e), std::mem::take(&mut w.sec_d)]);
            }
            IoJob::Touch { key, hits, last_used } => {
                let path = self.dirs.tail_path(&key);
                if let Err(e) = touch(&path, hits, last_used) {
                    tracing::warn!(path = %path.display(), error = %e, "kv.io touch failed");
                }
            }
            IoJob::Demote { path, key, hits } => {
                if let Err(error) = demote(&path, hits) {
                    tracing::warn!(path = %path.display(), error = %error, "kv.io demote failed");
                    if let Some(key) = key {
                        let _ = self.tx.send(Completion::MutationFailed { key, error });
                    }
                }
            }
            IoJob::Truncate { path, len } => {
                // Keep the mtime: it is the tail's persisted last_used (8.3).
                let r = OpenOptions::new().write(true).open(&path).and_then(|f| {
                    let mtime = f.metadata()?.modified()?;
                    f.set_len(len)?;
                    f.set_modified(mtime)
                });
                if let Err(e) = r {
                    tracing::warn!(path = %path.display(), error = %e, "kv.io truncate failed");
                }
            }
            IoJob::Trash { from } => {
                if from.parent() == Some(self.trash_root.as_path()) {
                    // Already in trash/ (startup renames, earlier runs).
                    let mut q = self.shared.lock();
                    q.trash.push(from);
                    q.stats.trash_dirs = q.trash.len() as u64;
                } else if let Some(to) = self.trash_target(&from) {
                    match fs::rename(&from, &to) {
                        Ok(()) => {
                            let mut q = self.shared.lock();
                            q.trash.push(to);
                            q.stats.trash_dirs = q.trash.len() as u64;
                        }
                        Err(e) => tracing::warn!(from = %from.display(), error = %e, "kv.io trash rename failed"),
                    }
                }
            }
            IoJob::Barrier(tx) => {
                let _ = tx.send(());
            }
            IoJob::BarrierAll(tx) => self.shared.lock().barriers_all.push(tx),
        }
    }

    fn trash_target(&mut self, from: &Path) -> Option<PathBuf> {
        mkdir_private(&self.trash_root).ok()?;
        self.seq += 1;
        let name = from.file_name()?.to_string_lossy().into_owned();
        Some(self.trash_root.join(format!("{name}-{}-{}", unix_now(), self.seq)))
    }

    #[cfg(test)]
    fn injected_fault(&self) -> bool {
        self.shared.lock().faults.pop_front().unwrap_or(false)
    }

    #[cfg(not(test))]
    fn injected_fault(&self) -> bool {
        false
    }

    fn written(&mut self, key: Key, tail: bool, r: io::Result<u64>, payload_hash: Option<[u8; 32]>, bufs: Vec<Vec<u8>>) {
        {
            let mut q = self.shared.lock();
            match &r {
                Ok(b) => {
                    q.stats.written_files += 1;
                    q.stats.written_bytes += b;
                }
                Err(_) => q.stats.write_errors += 1,
            }
        }
        let result = r.map_err(|e| (e.kind(), e.to_string()));
        let _ = self.tx.send(Completion::Written { key, tail, result, payload_hash, bufs });
    }

    /// A write to `path` makes a queued unlink of the same path stale: the
    /// rename replaces whatever is there, so the unlink must not run after.
    fn cancel_deletion(&self, path: &Path) {
        let mut q = self.shared.lock();
        if let Some(b) = q.delete_set.remove(path) {
            q.stats.delete_backlog_bytes -= b;
            q.stats.delete_backlog_files -= 1;
        }
    }

    fn tmp_path(&mut self, key: &Key, ext: &str) -> PathBuf {
        self.seq += 1;
        self.dirs.tmp().join(format!("{}-{}.{ext}.tmp", keys::hex(&key[..8]), self.seq))
    }

    fn write_chunk(&mut self, w: &mut ChunkWrite) -> io::Result<u64> {
        w.header.payload_len = w.payload.len() as u64;
        w.header.payload_hash = *blake3::hash(&w.payload).as_bytes();
        let prefix = format::encode_chunk_prefix(&w.header, &w.tokens, &w.images);
        let dst = self.dirs.chunk_path(&w.header.key);
        let tmp = self.tmp_path(&w.header.key, "kvc");
        self.cancel_deletion(&dst);
        write_atomic(&tmp, &dst, &[&prefix, &w.payload], w.header.created)
    }

    fn write_tail(&mut self, w: &mut TailWrite) -> io::Result<u64> {
        let h = &mut w.header;
        h.sec_e_len = w.sec_e.len() as u64;
        h.sec_e_hash = *blake3::hash(&w.sec_e).as_bytes();
        h.sec_d_len = w.sec_d.len() as u64;
        h.sec_d_hash = if w.sec_d.is_empty() { [0; 32] } else { *blake3::hash(&w.sec_d).as_bytes() };
        let prefix = format::encode_tail_prefix(&w.header, &w.open, &w.images);
        let dst = self.dirs.tail_path(&w.header.key);
        let tmp = self.tmp_path(&w.header.key, "kvt");
        self.cancel_deletion(&dst);
        write_atomic(&tmp, &dst, &[&prefix, &w.sec_e, &w.sec_d], w.header.created)
    }

    fn unlink_batch(&mut self, batch: Vec<(PathBuf, u64)>) {
        let (mut files, mut bytes, mut errors) = (0, 0, 0);
        for (p, b) in &batch {
            match fs::remove_file(p) {
                Ok(()) => {
                    files += 1;
                    bytes += b;
                }
                // Already gone (a rename over it was cancelled late, or a
                // crash): the goal state.
                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                Err(e) => {
                    errors += 1;
                    tracing::warn!(path = %p.display(), error = %e, "kv.io unlink failed");
                }
            }
        }
        let mut q = self.shared.lock();
        let backlog: u64 = batch.iter().map(|(_, b)| b).sum();
        q.stats.delete_backlog_bytes -= backlog;
        q.stats.delete_backlog_files -= batch.len() as u64;
        q.stats.unlinked_files += files;
        q.stats.unlinked_bytes += bytes;
        q.stats.unlink_errors += errors;
    }

    /// Remove up to [`DELETE_BATCH`] entries of the deepest trash directory,
    /// then return to the queues. A directory that will not go (a file in it
    /// cannot be unlinked) is retried [`TRASH_ATTEMPTS`] times, then left on
    /// disk and counted, so the worker never spins and `flush_all` returns.
    fn trash_step(&mut self) {
        let (dir, failed) = {
            let q = self.shared.lock();
            let Some(d) = q.trash.last().cloned() else { return };
            (d, q.trash_failed.clone())
        };
        let mut removed = 0u64;
        let mut subdirs = Vec::new();
        let mut more = false;
        match fs::read_dir(&dir) {
            Ok(rd) => {
                for e in rd.flatten() {
                    if removed as usize >= DELETE_BATCH {
                        more = true;
                        break;
                    }
                    let p = e.path();
                    if e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        if !failed.contains(&p) {
                            subdirs.push(p);
                        }
                    } else if fs::remove_file(&p).is_ok() {
                        removed += 1;
                    }
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => tracing::warn!(dir = %dir.display(), error = %e, "kv.io trash read failed"),
        }
        let mut q = self.shared.lock();
        q.stats.trash_files_removed += removed;
        if !subdirs.is_empty() {
            q.trash.extend(subdirs);
        } else if !more {
            match fs::remove_dir(&dir) {
                Err(e) if e.kind() != io::ErrorKind::NotFound => {
                    let n = q.trash_attempts.entry(dir.clone()).or_default();
                    *n += 1;
                    if *n >= TRASH_ATTEMPTS {
                        tracing::warn!(dir = %dir.display(), error = %e, "kv.io trash dir left in place");
                        q.trash_attempts.remove(&dir);
                        q.trash.retain(|d| d != &dir);
                        q.trash_failed.insert(dir);
                        q.stats.trash_errors += 1;
                    }
                }
                _ => {
                    q.trash_attempts.remove(&dir);
                    q.trash.retain(|d| d != &dir);
                }
            }
        }
        q.stats.trash_dirs = q.trash.len() as u64;
    }
}

/// tmp file, write, close, rename (no fsync, 9.5). The `rename` replaces an
/// existing file of the same key: a full tail over the encoder tail at the
/// same T (5.1). Chunks are deduplicated before they are queued, so a chunk
/// is never renamed over a live one (the first writer wins, E5).
///
/// The mtime is set to the header's `created`: a tail's mtime IS its
/// persisted `last_used` (8.3), so it must be the time the index used.
fn write_atomic(tmp: &Path, dst: &Path, parts: &[&[u8]], mtime: u64) -> io::Result<u64> {
    let r = (|| {
        let mut f = OpenOptions::new().write(true).create_new(true).mode(0o600).open(tmp)?;
        let mut n = 0u64;
        for p in parts {
            f.write_all(p)?;
            n += p.len() as u64;
        }
        f.set_modified(UNIX_EPOCH + Duration::from_secs(mtime))?;
        drop(f);
        match fs::rename(tmp, dst) {
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // First file in this fan-out directory.
                mkdir_private(dst.parent().unwrap())?;
                fs::rename(tmp, dst)?;
            }
            r => r?,
        }
        Ok(n)
    })();
    if r.is_err() {
        let _ = fs::remove_file(tmp);
    }
    r
}

fn touch(path: &Path, hits: u32, last_used: u64) -> io::Result<()> {
    let f = OpenOptions::new().write(true).open(path)?;
    f.write_all_at(&hits.to_le_bytes(), TAIL_HITS_OFFSET)?;
    f.set_modified(UNIX_EPOCH + Duration::from_secs(last_used))
}

/// Demotion (8.2): truncate first, then rewrite the header. A crash between
/// the two leaves a header that claims section D over a file that ends after
/// section E; the startup scan recognizes that and finishes the job.
pub(crate) fn demote(path: &Path, hits: Option<u32>) -> Result<(), format::FormatError> {
    let f = OpenOptions::new().read(true).write(true).open(path)?;
    let mut hb = [0u8; TAIL_HEADER_LEN];
    f.read_exact_at(&mut hb, 0)?;
    let h = TailHeader::decode(&hb)?;
    if h.anchor {
        return Err(format::FormatError::BadField("demoting an anchor"));
    }
    if h.kind != TailKind::Full {
        return Err(format::FormatError::BadField("demoting a tail that is not full"));
    }
    let meta = f.metadata()?;
    // set_len past the end would EXTEND a torn file with zeros.
    if meta.len() < h.e_end() {
        return Err(format::FormatError::Short { want: h.e_end(), got: meta.len() });
    }
    let mtime = meta.modified().ok();
    f.set_len(h.e_end())?;
    let d = h.demoted(hits.unwrap_or(h.hits));
    f.write_all_at(&d.encode(), 0)?;
    // The header rewrite is not a use: keep last_used (the mtime, 8.3).
    if let Some(t) = mtime {
        f.set_modified(t)?;
    }
    Ok(())
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// A file's mtime in unix seconds (a tail's persisted `last_used`, 8.3).
pub fn mtime_secs(m: &fs::Metadata) -> u64 {
    m.modified().ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs()).unwrap_or(0)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::kvstore::format::tests::{sample_chunk_header, sample_tail_header};
    use crate::kvstore::format::{read_chunk, read_tail, TailKind};
    use crate::kvstore::keys::{chunk_step, tail_step};
    use crate::kvstore::C;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh directory under TMPDIR, removed when dropped (store tests
    /// write real-size files: 2.83 MB chunks).
    pub struct TestDir(PathBuf);

    impl std::ops::Deref for TestDir {
        type Target = Path;
        fn deref(&self) -> &Path {
            &self.0
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    pub fn unique_dir(tag: &str) -> TestDir {
        static N: AtomicU32 = AtomicU32::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let p = std::env::temp_dir().join(format!("deepstrix-kvstore-{tag}-{}-{n}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        TestDir(p)
    }

    fn spawn(root: &Path, cap: u64) -> IoHandle {
        let dirs = NsDirs::new(root.join("ns"));
        dirs.create().unwrap();
        IoHandle::spawn(dirs, root.join("trash"), cap).unwrap()
    }

    pub fn chunk_write(k: u32, fill: u8) -> ChunkWrite {
        let mut header = sample_chunk_header();
        let tokens: Vec<i32> = (0..C as i32).map(|i| i + fill as i32).collect();
        header.k = k;
        header.parent = [fill; 32];
        header.key = chunk_step(&header.parent, k * C, &tokens, &[]);
        let payload = vec![fill; header.payload_len as usize];
        ChunkWrite { header, tokens, images: vec![], payload }
    }

    pub fn tail_write(t: u32, kind: TailKind, fill: u8) -> TailWrite {
        let mut header = sample_tail_header();
        header.anchor = false;
        header.t = t;
        header.n_open = (t % C) as u16;
        header.kind = kind;
        let open: Vec<i32> = (0..(t % C) as i32).map(|i| i * 3 + fill as i32).collect();
        header.base = [fill; 32];
        header.key = tail_step(&header.base, t / C * C, &open, &[]);
        let sec_d = if kind == TailKind::Full { vec![fill ^ 0xff; 3000] } else { vec![] };
        if kind == TailKind::Enc {
            header.n_raw_dec = 0;
        }
        TailWrite { header, open, images: vec![], sec_e: vec![fill; 5000], sec_d }
    }

    #[test]
    fn writes_land_verified_and_tmp_is_clean() {
        let root = unique_dir("io-write");
        let io = spawn(&root, 1 << 30);
        let mut b = WaitBudget::new(Duration::from_millis(200));
        let cw = chunk_write(3, 7);
        let ck = cw.header.key;
        let ns = cw.header.ns;
        io.submit_write(IoJob::Chunk(Box::new(cw)), 10, &mut b).unwrap();
        let tw = tail_write(3 * C + 10, TailKind::Full, 9);
        let tk = tw.header.key;
        io.submit_write(IoJob::Tail(Box::new(tw)), 10, &mut b).unwrap();
        let done = io.flush();
        assert_eq!(done.len(), 2);
        assert!(done.iter().all(|c| matches!(c, Completion::Written { result: Ok(_), .. })));
        let c = read_chunk(&io.dirs().chunk_path(&ck), &ns).unwrap();
        assert_eq!(c.header.k, 3);
        let t = read_tail(&io.dirs().tail_path(&tk), &ns, true).unwrap();
        assert_eq!(t.sec_d.unwrap().len(), 3000);
        assert_eq!(fs::read_dir(io.dirs().tmp()).unwrap().count(), 0);
        let s = io.stats();
        assert_eq!((s.written_files, s.queued_bytes), (2, 0));
    }

    #[test]
    fn backpressure_waits_then_drops() {
        let root = unique_dir("io-bp");
        let io = spawn(&root, 100);
        // Stall the worker behind a barrier we hold.
        let (tx, rx) = mpsc::sync_channel(0);
        io.submit(IoJob::Barrier(tx));
        std::thread::sleep(Duration::from_millis(20));
        let mut b = WaitBudget::new(Duration::from_millis(50));
        // The first write enters (the queue only holds the barrier, 0 bytes,
        // but is not empty, so 80 <= 100 must fit).
        assert!(io.submit_write(IoJob::Chunk(Box::new(chunk_write(0, 1))), 80, &mut b).is_ok());
        let t0 = Instant::now();
        assert!(io.submit_write(IoJob::Chunk(Box::new(chunk_write(1, 2))), 80, &mut b).is_err(), "no room: dropped");
        let waited = t0.elapsed();
        assert!(waited >= Duration::from_millis(40) && waited < Duration::from_millis(2000), "{waited:?}");
        assert_eq!(b.remaining, Duration::ZERO);
        // The budget is per tick: once spent, the next write drops at once.
        let t1 = Instant::now();
        assert!(io.submit_write(IoJob::Chunk(Box::new(chunk_write(2, 3))), 80, &mut b).is_err());
        assert!(t1.elapsed() < Duration::from_millis(20));
        rx.recv().unwrap();
        let done = io.flush();
        assert_eq!(done.len(), 1);
    }

    #[test]
    fn a_write_cancels_a_queued_unlink_of_the_same_path() {
        let root = unique_dir("io-cancel");
        let io = spawn(&root, 1 << 30);
        let mut b = WaitBudget::new(Duration::from_millis(200));
        let tw = tail_write(100, TailKind::Enc, 1);
        let key = tw.header.key;
        let ns = tw.header.ns;
        io.submit_write(IoJob::Tail(Box::new(tw)), 1, &mut b).unwrap();
        io.flush();
        // Hold the worker, queue the unlink, then a rewrite of the same key.
        let (tx, rx) = mpsc::sync_channel(0);
        io.submit(IoJob::Barrier(tx));
        io.delete(io.dirs().tail_path(&key), 5000);
        io.submit_write(IoJob::Tail(Box::new(tail_write(100, TailKind::Enc, 1))), 1, &mut b).unwrap();
        rx.recv().unwrap();
        io.flush_all();
        assert!(read_tail(&io.dirs().tail_path(&key), &ns, false).is_ok(), "the rewrite survived the stale unlink");
        let s = io.stats();
        assert_eq!((s.delete_backlog_files, s.delete_backlog_bytes, s.unlinked_files), (0, 0, 0));
        // An unlink that is not cancelled runs.
        io.delete(io.dirs().tail_path(&key), 5000);
        io.flush_all();
        assert!(!io.dirs().tail_path(&key).exists());
        assert_eq!(io.stats().unlinked_files, 1);
    }

    #[test]
    fn touch_and_demote_mutate_in_place() {
        let root = unique_dir("io-mut");
        let io = spawn(&root, 1 << 30);
        let mut b = WaitBudget::new(Duration::from_millis(200));
        let tw = tail_write(C + 5, TailKind::Full, 4);
        let (key, ns) = (tw.header.key, tw.header.ns);
        io.submit_write(IoJob::Tail(Box::new(tw)), 1, &mut b).unwrap();
        io.submit(IoJob::Touch { key, hits: 17, last_used: 1_800_000_000 });
        let path = io.dirs().tail_path(&key);
        io.submit(IoJob::Demote { path: path.clone(), key: Some(key), hits: Some(18) });
        assert!(io.flush().iter().all(|c| matches!(c, Completion::Written { result: Ok(_), .. })));
        let t = read_tail(&path, &ns, false).unwrap();
        assert_eq!((t.header.kind, t.header.demoted, t.header.hits), (TailKind::Enc, true, 18));
        assert_eq!(read_tail(&path, &ns, true), Err(format::FormatError::NoSectionD));
        assert_eq!(fs::metadata(&path).unwrap().len(), t.header.e_end());
        assert_eq!(mtime_secs(&fs::metadata(&path).unwrap()), 1_800_000_000, "demotion keeps last_used");
        // Demoting a corrupt file reports it.
        fs::write(&path, b"garbage").unwrap();
        io.submit(IoJob::Demote { path, key: Some(key), hits: Some(1) });
        match io.flush().as_slice() {
            [Completion::MutationFailed { error, .. }] => assert!(error.evicts(), "{error}"),
            other => panic!("{other:?}"),
        }
        // A file already demoted (or torn short of section E) is refused, not
        // extended by set_len.
        let tw = tail_write(C + 7, TailKind::Enc, 5);
        let p2 = io.dirs().tail_path(&tw.header.key);
        io.submit_write(IoJob::Tail(Box::new(tw)), 1, &mut b).unwrap();
        io.submit(IoJob::Demote { path: p2.clone(), key: None, hits: None });
        io.flush();
        let len = fs::metadata(&p2).unwrap().len();
        assert_eq!(read_tail(&p2, &ns, false).unwrap().header.kind, TailKind::Enc);
        assert_eq!(fs::metadata(&p2).unwrap().len(), len);
    }

    #[test]
    fn trash_is_removed_in_the_background() {
        let root = unique_dir("io-trash");
        let io = spawn(&root, 1 << 30);
        let victim = root.join("oldns");
        for d in 0..3 {
            let sub = victim.join("chunks").join(format!("{d:02x}"));
            fs::create_dir_all(&sub).unwrap();
            for i in 0..150 {
                fs::write(sub.join(format!("{i}.kvc")), b"x").unwrap();
            }
        }
        io.submit(IoJob::Trash { from: victim.clone() });
        io.flush_all();
        assert!(!victim.exists());
        assert_eq!(fs::read_dir(root.join("trash")).unwrap().count(), 0);
        let s = io.stats();
        assert_eq!((s.trash_files_removed, s.trash_dirs), (450, 0));
    }

    #[test]
    fn a_stuck_trash_dir_is_given_up_not_spun_on() {
        // A file that cannot be unlinked (its directory is read-only) keeps its
        // directory, and so every parent, non-empty forever. The worker must
        // give up after a few attempts so it idles and flush_all returns.
        use std::os::unix::fs::PermissionsExt;
        let root = unique_dir("io-stuck");
        let io = spawn(&root, 1 << 30);
        // Already inside trash/ (as startup leaves it), so it is not renamed
        // and the test can restore the permissions afterwards.
        let victim = root.join("trash").join("oldns");
        let locked = victim.join("chunks").join("ab");
        fs::create_dir_all(&locked).unwrap();
        fs::write(locked.join("x.kvc"), b"x").unwrap();
        fs::write(victim.join("chunks").join("y.kvc"), b"y").unwrap();
        fs::set_permissions(&locked, fs::Permissions::from_mode(0o500)).unwrap();
        let undo = || fs::set_permissions(&locked, fs::Permissions::from_mode(0o700)).unwrap();
        if fs::remove_file(locked.join("x.kvc")).is_ok() {
            undo();
            return; // running as root: permissions do not bind, nothing to test
        }
        io.submit(IoJob::Trash { from: victim.clone() });
        let (tx, rx) = mpsc::channel();
        let waiter = std::thread::spawn(move || {
            io.flush_all();
            let _ = tx.send(io.stats());
            io
        });
        let r = rx.recv_timeout(Duration::from_secs(20));
        undo();
        let s = r.expect("flush_all hung: the worker spins on the stuck trash dir");
        drop(waiter.join().unwrap());
        assert!(s.trash_errors >= 1 && s.trash_dirs == 0, "{s:?}");
        assert!(s.trash_files_removed >= 1, "the removable file went");
    }

    #[test]
    fn injected_write_faults_fail_only_their_writes() {
        let root = unique_dir("io-fault");
        let io = spawn(&root, 1 << 30);
        let mut b = WaitBudget::new(Duration::from_millis(200));
        io.inject_write_faults(&[false, true]);
        let w0 = chunk_write(0, 1);
        let w1 = chunk_write(1, 2);
        let (k0, k1) = (w0.header.key, w1.header.key);
        io.submit_write(IoJob::Chunk(Box::new(w0)), 1, &mut b).unwrap();
        io.submit_write(IoJob::Chunk(Box::new(w1)), 1, &mut b).unwrap();
        let done = io.flush();
        let ok: Vec<(Key, bool)> = done
            .iter()
            .map(|c| match c {
                Completion::Written { key, result, payload_hash, bufs, .. } => {
                    assert_eq!(bufs.len(), 1, "the payload buffer comes back");
                    assert_eq!(payload_hash.is_some(), result.is_ok());
                    (*key, result.is_ok())
                }
                other => panic!("{other:?}"),
            })
            .collect();
        assert_eq!(ok, vec![(k0, true), (k1, false)]);
        assert!(io.dirs().chunk_path(&k0).exists() && !io.dirs().chunk_path(&k1).exists());
        assert_eq!(fs::read_dir(io.dirs().tmp()).unwrap().count(), 0);
    }

    #[test]
    fn shutdown_drains_writes() {
        let root = unique_dir("io-shutdown");
        let io = spawn(&root, 1 << 30);
        let mut b = WaitBudget::new(Duration::from_millis(200));
        let mut keys = Vec::new();
        for k in 0..20 {
            let w = chunk_write(k, k as u8);
            keys.push(w.header.key);
            io.submit_write(IoJob::Chunk(Box::new(w)), 1, &mut b).unwrap();
        }
        let dirs = io.dirs().clone();
        let done = io.shutdown(Duration::from_millis(100));
        assert_eq!(done.len(), 20);
        assert!(keys.iter().all(|k| dirs.chunk_path(k).exists()));
    }
}
