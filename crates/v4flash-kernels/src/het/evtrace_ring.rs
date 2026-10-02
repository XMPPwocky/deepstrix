//! TIER B: the flight recorder (docs/v41/EVTRACE_REBUILD_PLAN.md 2.1, phase P1).
//!
//! High-rate records -- device stage intervals (P3), CPU spans (P2), knob
//! changes -- never touch Tier A's channel. A producer serializes into ITS
//! THREAD's buffer (`emit_b`: the file format's bytes, its own mutex, no
//! allocation once warm). A full buffer is copied into the ring's open block
//! at once if the ring's lock is free (a producer never waits on it; it never
//! seals or evicts either -- only the Tier B thread does), and the Tier B
//! thread drains every registered buffer every 100 ms. The ring is a deque of
//! RECORD-ALIGNED blocks (~64 KB, sealed as `Arc<[u8]>`) bounded by
//! `V41_EVTRACE_RING_MB` (default 128, owner 10-01; 0 = Tier B off), oldest
//! first out. Records are in BATCH order (per thread, at drain time), not time
//! order: readers sort.
//!
//! Nothing reaches the disk until a DUMP. A request file
//! `<ring dir>/dump-request` (contents: seconds, or `all`; write it
//! temp-then-rename; an empty read is retried at the next poll) is polled by
//! the dump thread every 250 ms and removed when acted on: no edge-trigger
//! traps, nothing left to fire at the next start. A dump drains the buffers,
//! clones the block list, and STREAMS block by block the records whose own
//! time is in the window into `<ring dir>/<role>-dump-<utc>-<pid>-<n>.evt`
//! (the Tier A file format, the CURRENT string table and knob values in its
//! header, `"tier": "B"`), each 4 MB synced to the disk before the next (paced
//! writeback: box 2's OS disk is its primary expert drive). Ring dir:
//! `V41_EVTRACE_RING_DIR` (default: `/dev/shm/evtrace-ring` on box 2, role
//! `b2`; `<V41_EVTRACE_DIR>-ring` elsewhere; never the Tier A dir), dumps kept
//! under `V41_EVTRACE_RING_KEEP_MB` (default 256) -- pruning touches only this
//! role's `-dump-` files.

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock, Weak};
use std::time::{Duration, Instant};

use super::evtrace::{Kind, MAX_FIELDS};

/// A block is sealed once it holds at least this many bytes.
const BLOCK: usize = 64 << 10;
/// A thread's buffer is handed to the ring at this size.
const LOCAL_FLUSH: usize = 16 << 10;
/// A thread's buffer past this size (the ring's lock kept busy) drops records.
const LOCAL_MAX: usize = 4 << 20;

static ON: AtomicBool = AtomicBool::new(false);
static DROPPED_B: AtomicU64 = AtomicU64::new(0);
static RING: OnceLock<Mutex<Ring>> = OnceLock::new();
static REG: Mutex<Vec<Weak<Mutex<Vec<u8>>>>> = Mutex::new(Vec::new());
/// Buffers of exiting threads that found the ring busy (an O(1) push; the
/// Tier B thread moves them): a parent joining a helper thread must never
/// wait on the ring through the child's TLS destructor.
static ORPHANS: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Is Tier B on? One relaxed load.
#[inline]
pub fn enabled() -> bool {
    ON.load(Relaxed)
}

/// Records dropped by Tier B (a full thread buffer, thread teardown) and the
/// bytes the ring holds: the Tier A `meta` heartbeat carries both.
pub fn stats() -> (u64, u64) {
    let bytes = RING.get().map(|r| r.lock().unwrap_or_else(|p| p.into_inner()).bytes as u64).unwrap_or(0);
    (DROPPED_B.load(Relaxed), bytes)
}

/// The ring: sealed blocks + the open one, record-aligned (only whole records
/// are ever appended).
pub(crate) struct Ring {
    blocks: VecDeque<Arc<[u8]>>,
    open: Vec<u8>,
    bytes: usize,
    cap: usize,
}

impl Ring {
    pub(crate) fn new(cap: usize) -> Self {
        Self { blocks: VecDeque::new(), open: Vec::with_capacity(BLOCK + LOCAL_FLUSH), bytes: 0, cap }
    }

    /// Copy whole records into the open block (producers: a memcpy only).
    pub(crate) fn push(&mut self, recs: &[u8]) {
        self.open.extend_from_slice(recs);
        self.bytes += recs.len();
    }

    /// Seal the open block past `BLOCK`, evict the oldest blocks past the cap
    /// (the Tier B thread only).
    pub(crate) fn maintain(&mut self) {
        if self.open.len() >= BLOCK {
            let sealed: Arc<[u8]> = Arc::from(std::mem::replace(&mut self.open, Vec::with_capacity(BLOCK + LOCAL_FLUSH)));
            self.blocks.push_back(sealed);
        }
        while self.bytes > self.cap {
            match self.blocks.pop_front() {
                Some(b) => self.bytes -= b.len(),
                None => break,
            }
        }
    }

    /// The blocks (shared, not copied) and a copy of the open block.
    pub(crate) fn snapshot(&self) -> (Vec<Arc<[u8]>>, Vec<u8>) {
        (self.blocks.iter().cloned().collect(), self.open.clone())
    }
}

/// One record in the file format (u16 kind, u16 n, n x f64), NaN-padded.
fn serialize(out: &mut Vec<u8>, k: &Kind, v: &[f64]) {
    let n = k.fields.len().min(MAX_FIELDS);
    out.extend_from_slice(&k.id.to_le_bytes());
    out.extend_from_slice(&(n as u16).to_le_bytes());
    for i in 0..n {
        out.extend_from_slice(&v.get(i).copied().unwrap_or(f64::NAN).to_le_bytes());
    }
}

/// This thread's buffer, registered for the Tier B thread's drain; on thread
/// exit its bytes go to the ring if the ring is free, else to `ORPHANS`.
struct Local {
    buf: Arc<Mutex<Vec<u8>>>,
}

impl Local {
    fn new() -> Self {
        let buf = Arc::new(Mutex::new(Vec::with_capacity(LOCAL_FLUSH * 2)));
        REG.lock().unwrap_or_else(|p| p.into_inner()).push(Arc::downgrade(&buf));
        Self { buf }
    }
}

impl Drop for Local {
    fn drop(&mut self) {
        let bytes = std::mem::take(&mut *self.buf.lock().unwrap_or_else(|p| p.into_inner()));
        if bytes.is_empty() {
            return;
        }
        match RING.get().map(|r| r.try_lock()) {
            Some(Ok(mut r)) => r.push(&bytes),
            Some(Err(_)) => ORPHANS.lock().unwrap_or_else(|p| p.into_inner()).push(bytes),
            None => {}
        }
    }
}

thread_local! {
    static LOCAL: Local = Local::new();
}

/// The kinds a dump can place in time (`evtrace::time_fields`), cached.
fn tidx() -> &'static std::collections::HashMap<u16, Vec<usize>> {
    static T: OnceLock<std::collections::HashMap<u16, Vec<usize>>> = OnceLock::new();
    T.get_or_init(super::evtrace::time_fields)
}

/// Record one Tier B event (`v` in `k.fields` order; short = NaN-padded).
/// Never blocks on the ring: the thread's own buffer, the ring through
/// `try_lock`. `k` must be registered in `evtrace::kinds()` (a dump reads a
/// record's time from the kind's fields; an unknown kind is kept at any age).
#[inline]
pub fn emit_b(k: &Kind, v: &[f64]) {
    if !enabled() {
        return;
    }
    debug_assert!(tidx().get(&k.id).is_some_and(|t| !t.is_empty()), "Tier B kind {} not registered with a time field", k.name);
    let ok = LOCAL.try_with(|l| {
        let mut b = l.buf.lock().unwrap_or_else(|p| p.into_inner());
        if b.len() >= LOCAL_MAX {
            DROPPED_B.fetch_add(1, Relaxed);
            return;
        }
        serialize(&mut b, k, v);
        if b.len() >= LOCAL_FLUSH {
            if let Some(Ok(mut r)) = RING.get().map(|r| r.try_lock()) {
                r.push(&b);
                b.clear();
            }
        }
    });
    if ok.is_err() {
        DROPPED_B.fetch_add(1, Relaxed); // thread teardown
    }
}

/// Hand this thread's buffer to the ring now (a tick's end, plan N5); left
/// for the Tier B thread if the ring is busy.
pub fn flush_local() {
    if !enabled() {
        return;
    }
    let _ = LOCAL.try_with(|l| {
        let mut b = l.buf.lock().unwrap_or_else(|p| p.into_inner());
        if !b.is_empty() {
            if let Some(Ok(mut r)) = RING.get().map(|r| r.try_lock()) {
                r.push(&b);
                b.clear();
            }
        }
    });
}

/// Move every registered buffer and every orphan into the ring, then seal and
/// evict (the Tier B thread every ~100 ms, and a dump first).
fn drain_all() {
    let Some(ring) = RING.get() else { return };
    let mut reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    reg.retain(|w| w.strong_count() > 0);
    let bufs: Vec<Arc<Mutex<Vec<u8>>>> = reg.iter().filter_map(|w| w.upgrade()).collect();
    drop(reg);
    for b in bufs {
        // The replacement is allocated BEFORE the producer's lock: under it,
        // only a swap.
        let mut fresh = Vec::with_capacity(LOCAL_FLUSH * 2);
        {
            let mut g = b.lock().unwrap_or_else(|p| p.into_inner());
            if g.is_empty() {
                continue;
            }
            std::mem::swap(&mut *g, &mut fresh);
        }
        ring.lock().unwrap_or_else(|p| p.into_inner()).push(&fresh);
    }
    let orphans = std::mem::take(&mut *ORPHANS.lock().unwrap_or_else(|p| p.into_inner()));
    let mut r = ring.lock().unwrap_or_else(|p| p.into_inner());
    for o in &orphans {
        r.push(o);
    }
    r.maintain();
}

/// Call `f` with every record of `data` whose first time field is at least
/// `t_min` (a record whose kind has no time field is kept).
pub(crate) fn for_each_selected(data: &[u8], t_min: f64, tidx: &std::collections::HashMap<u16, Vec<usize>>, mut f: impl FnMut(&[u8])) -> u64 {
    let mut n_rec = 0u64;
    let mut off = 0;
    while off + 4 <= data.len() {
        let kid = u16::from_le_bytes([data[off], data[off + 1]]);
        let n = u16::from_le_bytes([data[off + 2], data[off + 3]]) as usize;
        let end = off + 4 + 8 * n;
        if end > data.len() {
            break; // cannot happen: blocks hold whole records
        }
        let t = tidx
            .get(&kid)
            .and_then(|ix| ix.iter().filter(|&&i| i < n).map(|&i| f64::from_le_bytes(data[off + 4 + 8 * i..off + 12 + 8 * i].try_into().unwrap())).find(|x| !x.is_nan()));
        if t.is_none_or(|t| t >= t_min) {
            f(&data[off..end]);
            n_rec += 1;
        }
        off = end;
    }
    n_rec
}

struct DumpCfg {
    dir: PathBuf,
    role: String,
    keep_bytes: u64,
}

/// Start Tier B (from `evtrace::init`): the ring, the drain thread, the dump
/// thread. `tier_a_dir` = `V41_EVTRACE_DIR` (the ring dir may never be it).
pub(crate) fn init(role: &str, tier_a_dir: &Path) {
    let mb = std::env::var("V41_EVTRACE_RING_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(128);
    if mb == 0 || RING.get().is_some() {
        return;
    }
    let default_dir = if role == "b2" { PathBuf::from("/dev/shm/evtrace-ring") } else { PathBuf::from(format!("{}-ring", tier_a_dir.display())) };
    let dir = std::env::var("V41_EVTRACE_RING_DIR").map(PathBuf::from).unwrap_or(default_dir);
    if std::fs::canonicalize(&dir).ok().zip(std::fs::canonicalize(tier_a_dir).ok()).is_some_and(|(a, b)| a == b) || dir == tier_a_dir {
        tracing::warn!(dir = %dir.display(), "evtrace ring: the ring dir is the Tier A dir; Tier B off");
        return;
    }
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %e, dir = %dir.display(), "evtrace ring: cannot create dir; Tier B off");
        return;
    }
    if RING.set(Mutex::new(Ring::new((mb << 20) as usize))).is_err() {
        return;
    }
    let keep_bytes = std::env::var("V41_EVTRACE_RING_KEEP_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(256) << 20;
    let cfg = DumpCfg { dir: dir.clone(), role: role.to_string(), keep_bytes };
    // The Tier B thread also times the device buffers (`evtrace_dev`).
    let rx = super::evtrace_dev::channel();
    let drain = std::thread::Builder::new().name("evtrace-tierb".into()).spawn(move || super::evtrace_dev::tier_b_loop(rx, drain_all));
    let dump = std::thread::Builder::new().name("evtrace-dump".into()).spawn(move || dump_loop(cfg));
    if drain.is_err() || dump.is_err() {
        tracing::warn!("evtrace ring: thread failed to start; Tier B off");
        return;
    }
    ON.store(true, Relaxed);
    tracing::info!(ring_mb = mb, dir = %dir.display(), "evtrace ring: on (dump: write seconds or `all` to <dir>/dump-request)");
}

fn dump_loop(cfg: DumpCfg) {
    let req = cfg.dir.join("dump-request");
    let mut n = 0u32;
    let mut warned_stuck = false;
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let Ok(text) = std::fs::read_to_string(&req) else { continue };
        let text = text.trim().to_string();
        if text.is_empty() {
            continue; // a non-atomic write caught half-way: read it again next poll
        }
        // A request that cannot be removed would fire every poll: act on none.
        if let Err(e) = std::fs::remove_file(&req) {
            if !warned_stuck {
                tracing::warn!(error = %e, path = %req.display(), "evtrace ring: cannot remove the dump request; ignoring it");
                warned_stuck = true;
            }
            continue;
        }
        let secs = if text.eq_ignore_ascii_case("all") {
            None
        } else {
            match text.parse::<f64>().ok().filter(|s| *s > 0.0) {
                Some(s) => Some(s),
                None => {
                    // Only `all` means everything (a 128 MB dump).
                    tracing::warn!(request = %text, "evtrace ring: dump request is neither seconds nor `all`; ignored");
                    continue;
                }
            }
        };
        let t0 = Instant::now();
        match dump(&cfg, n, secs) {
            Ok((path, bytes, recs)) => {
                n += 1;
                tracing::info!(path = %path.display(), bytes, records = recs, wall_ms = t0.elapsed().as_millis() as u64, "evtrace ring: dumped");
            }
            Err(e) => tracing::warn!(error = %e, "evtrace ring: dump failed"),
        }
    }
}

fn dump(cfg: &DumpCfg, n: u32, secs: Option<f64>) -> std::io::Result<(PathBuf, u64, u64)> {
    // Everything emitted before the request: the threads' buffers first.
    drain_all();
    let (blocks, open) = match RING.get() {
        Some(r) => r.lock().unwrap_or_else(|p| p.into_inner()).snapshot(),
        None => return Err(std::io::Error::other("Tier B off")),
    };
    let now = super::evtrace::now();
    let t_min = secs.map_or(f64::NEG_INFINITY, |s| now - s * 1e9);
    let extra = serde_json::json!({ "tier": "B", "dump": { "seconds": secs, "t_from_raw": t_min, "t_to_raw": now } });
    let (header, _) = super::evtrace::header_bytes(&extra);
    let path = cfg.dir.join(format!("{}-dump-{}-{}-{:03}.evt", cfg.role, super::evtrace::utc_stamp(), std::process::id(), n));
    let file = std::fs::File::create(&path)?;
    let mut f = std::io::BufWriter::with_capacity(4 << 20, &file);
    f.write_all(b"EVT1")?;
    f.write_all(&(header.len() as u32).to_le_bytes())?;
    f.write_all(&header)?;
    // STREAM block by block (each block's Arc dropped once written: O(4 MB) of
    // extra memory, not a second ring), and every 4 MB SYNCED to the disk
    // before the next with a breath between: paced writeback, not a burst the
    // kernel writes later (box 2's OS disk is its primary expert drive; ionice
    // does nothing on these drives).
    let (mut bytes, mut recs, mut since_sync) = (8 + header.len() as u64, 0u64, 0usize);
    let tidx = tidx();
    let mut piece = |data: &[u8], f: &mut std::io::BufWriter<&std::fs::File>| -> std::io::Result<()> {
        let mut write_err = None;
        recs += for_each_selected(data, t_min, tidx, |r| {
            if write_err.is_none() {
                if let Err(e) = f.write_all(r) {
                    write_err = Some(e);
                }
                bytes += r.len() as u64;
                since_sync += r.len();
            }
        });
        if let Some(e) = write_err {
            return Err(e);
        }
        if since_sync >= 4 << 20 {
            f.flush()?;
            file.sync_data()?;
            since_sync = 0;
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    };
    // Each block's Arc is released once written (no copy).
    for b in blocks {
        piece(&b, &mut f)?;
    }
    piece(&open, &mut f)?;
    f.flush()?;
    file.sync_data()?;
    drop(f);
    prune_bytes(&cfg.dir, &format!("{}-dump-", cfg.role), cfg.keep_bytes, &path);
    Ok((path, bytes, recs))
}

/// Keep the newest of this role's dumps (`<prefix>*.evt`, by modification
/// time) within `keep` bytes; never the one just written. Nothing else in the
/// directory is touched.
fn prune_bytes(dir: &Path, prefix: &str, keep: u64, current: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut v: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.starts_with(prefix) && n.ends_with(".evt")))
        .filter_map(|e| {
            let m = e.metadata().ok()?;
            Some((m.modified().ok()?, m.len(), e.path()))
        })
        .collect();
    v.sort();
    let mut total: u64 = v.iter().map(|x| x.1).sum();
    for (_, len, p) in v {
        if total <= keep {
            break;
        }
        if p != current {
            let _ = std::fs::remove_file(&p);
            total -= len;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    static T: Kind = Kind { id: 31, name: "dev", fields: &["t_start", "t_end", "stage"] };

    fn recs(ts: &[f64]) -> Vec<u8> {
        let mut out = Vec::new();
        for &t in ts {
            serialize(&mut out, &T, &[t, t + 1.0, 7.0]);
        }
        out
    }

    fn tidx31() -> std::collections::HashMap<u16, Vec<usize>> {
        std::collections::HashMap::from([(31u16, vec![0usize, 1])])
    }

    #[test]
    fn the_ring_stays_record_aligned_and_bounded() {
        let one = recs(&[1.0]).len();
        let mut r = Ring::new(BLOCK * 3);
        for i in 0..20_000 {
            r.push(&recs(&[i as f64]));
            r.maintain();
        }
        assert!(r.bytes <= BLOCK * 3 + BLOCK, "{}", r.bytes);
        let (blocks, open) = r.snapshot();
        let mut body = Vec::new();
        let mut n = 0;
        for data in blocks.iter().map(|b| &b[..]).chain(std::iter::once(&open[..])) {
            n += for_each_selected(data, f64::NEG_INFINITY, &tidx31(), |rec| body.extend_from_slice(rec));
        }
        assert_eq!(body.len() as u64, n * one as u64, "whole records only");
        let last = f64::from_le_bytes(body[body.len() - one + 4..body.len() - one + 12].try_into().unwrap());
        assert_eq!(last, 19_999.0);
        let first = f64::from_le_bytes(body[4..12].try_into().unwrap());
        assert!(first > 0.0, "evicted from the front");
    }

    #[test]
    fn a_dump_selects_by_record_time() {
        let data = recs(&[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(for_each_selected(&data, 25.0, &tidx31(), |_| {}), 2);
    }

    #[test]
    fn pruning_touches_only_this_roles_dumps() {
        let dir = std::env::temp_dir().join(format!("evtrace-prune-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mk = |n: &str, len: usize| {
            let p = dir.join(n);
            std::fs::write(&p, vec![0u8; len]).unwrap();
            std::thread::sleep(Duration::from_millis(15));
            p
        };
        let tier_a = mk("hub-20261001-000000-1-000.evt", 1000);
        let other = mk("b2-dump-20261001-000000-2-000.evt", 1000);
        let old = mk("hub-dump-20261001-000000-1-000.evt", 1000);
        let new = mk("hub-dump-20261001-000001-1-001.evt", 1000);
        prune_bytes(&dir, "hub-dump-", 1500, &new);
        assert!(tier_a.exists() && other.exists() && new.exists());
        assert!(!old.exists());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
