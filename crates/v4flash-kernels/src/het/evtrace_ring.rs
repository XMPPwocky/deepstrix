//! TIER B: the flight recorder (docs/v41/EVTRACE_REBUILD_PLAN.md 2.1, phase P1).
//!
//! High-rate records -- device stage intervals (P3), CPU spans (P2), knob
//! changes -- never touch Tier A's channel. A producer serializes into ITS
//! THREAD's buffer (`emit_b`: the file format's bytes, an uncontended lock, no
//! allocation once warm); a full buffer is handed to the ring at once (only if
//! the ring's lock is free: a producer never waits), and the Tier B thread
//! drains every registered buffer every 100 ms. The ring is a deque of
//! RECORD-ALIGNED blocks (~64 KB, sealed as `Arc<[u8]>`) bounded by
//! `V41_EVTRACE_RING_MB` (default 128, owner 10-01; 0 = Tier B off), oldest
//! blocks evicted first.
//!
//! Nothing reaches the disk until a DUMP. A request file
//! `<ring dir>/dump-request` (contents: seconds; empty = everything the ring
//! holds; write it temp-then-rename) is polled by the dump thread every 250 ms
//! and removed when acted on: no edge-trigger traps, nothing left to fire at
//! the next start. A dump clones the block list (no copy; no lock held while
//! writing), keeps the records whose own time is in the requested window, and
//! writes `<ring dir>/<role>-<utc>-<pid>-<n>.evt` in the Tier A file format with
//! the CURRENT string table and knob values in its header (`"tier": "B"`).
//! Ring dir: `V41_EVTRACE_RING_DIR` (default `<V41_EVTRACE_DIR>-ring`: its own
//! directory, so Tier A pruning, `ls hub-*.evt` and `evtrace.py <dir>` never
//! see dumps), kept under `V41_EVTRACE_RING_KEEP_MB` (default 256) of dumps.

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

    /// Append whole records; seal the open block past `BLOCK`; evict the oldest
    /// blocks past the cap.
    pub(crate) fn append(&mut self, recs: &[u8]) {
        self.open.extend_from_slice(recs);
        self.bytes += recs.len();
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

/// This thread's buffer, registered for the Tier B thread's drain; flushed into
/// the ring when the thread exits.
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
        if let (false, Some(r)) = (bytes.is_empty(), RING.get()) {
            r.lock().unwrap_or_else(|p| p.into_inner()).append(&bytes);
        }
    }
}

thread_local! {
    static LOCAL: Local = Local::new();
}

/// Record one Tier B event (`v` in `k.fields` order; short = NaN-padded).
/// Never blocks: the thread's own buffer, and the ring only through `try_lock`.
#[inline]
pub fn emit_b(k: &Kind, v: &[f64]) {
    if !enabled() {
        return;
    }
    let ok = LOCAL.try_with(|l| {
        let mut b = l.buf.lock().unwrap_or_else(|p| p.into_inner());
        if b.len() >= LOCAL_MAX {
            DROPPED_B.fetch_add(1, Relaxed);
            return;
        }
        serialize(&mut b, k, v);
        if b.len() >= LOCAL_FLUSH {
            if let Some(Ok(mut r)) = RING.get().map(|r| r.try_lock()) {
                r.append(&b);
                b.clear();
            }
        }
    });
    if ok.is_err() {
        DROPPED_B.fetch_add(1, Relaxed); // thread teardown
    }
}

/// Hand this thread's buffer to the ring now (a tick's end, plan N5).
pub fn flush_local() {
    if !enabled() {
        return;
    }
    let _ = LOCAL.try_with(|l| {
        let mut b = l.buf.lock().unwrap_or_else(|p| p.into_inner());
        if !b.is_empty() {
            if let Some(Ok(mut r)) = RING.get().map(|r| r.try_lock()) {
                r.append(&b);
                b.clear();
            }
        }
    });
}

/// Move every registered buffer into the ring (the Tier B thread, ~100 ms).
fn drain_all() {
    let Some(ring) = RING.get() else { return };
    let mut reg = REG.lock().unwrap_or_else(|p| p.into_inner());
    reg.retain(|w| w.strong_count() > 0);
    let bufs: Vec<Arc<Mutex<Vec<u8>>>> = reg.iter().filter_map(|w| w.upgrade()).collect();
    drop(reg);
    for b in bufs {
        let bytes = {
            let mut g = b.lock().unwrap_or_else(|p| p.into_inner());
            if g.is_empty() {
                continue;
            }
            let cap = g.capacity();
            std::mem::replace(&mut *g, Vec::with_capacity(cap))
        };
        ring.lock().unwrap_or_else(|p| p.into_inner()).append(&bytes);
    }
}

/// The records of `blocks` (then `open`) whose first time field is at least
/// `t_min`, in order. `tidx[kind]` = the index of the kind's first `t*` field.
pub(crate) fn select(blocks: &[Arc<[u8]>], open: &[u8], t_min: f64, tidx: &std::collections::HashMap<u16, Vec<usize>>) -> (Vec<u8>, u64) {
    let mut out = Vec::new();
    let mut n_rec = 0u64;
    for data in blocks.iter().map(|b| &b[..]).chain(std::iter::once(open)) {
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
            // A record without a time (none of today's kinds) is kept.
            if t.is_none_or(|t| t >= t_min) {
                out.extend_from_slice(&data[off..end]);
                n_rec += 1;
            }
            off = end;
        }
    }
    (out, n_rec)
}

struct DumpCfg {
    dir: PathBuf,
    role: String,
    keep_bytes: u64,
}

/// Start Tier B (from `evtrace::init`): the ring, the drain thread, the dump
/// thread. `default_dir` = `<V41_EVTRACE_DIR>-ring`.
pub(crate) fn init(role: &str, default_dir: PathBuf) {
    let mb = std::env::var("V41_EVTRACE_RING_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(128);
    if mb == 0 || RING.get().is_some() {
        return;
    }
    let dir = std::env::var("V41_EVTRACE_RING_DIR").map(PathBuf::from).unwrap_or(default_dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %e, dir = %dir.display(), "evtrace ring: cannot create dir; Tier B off");
        return;
    }
    if RING.set(Mutex::new(Ring::new((mb << 20) as usize))).is_err() {
        return;
    }
    let keep_bytes = std::env::var("V41_EVTRACE_RING_KEEP_MB").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(256) << 20;
    let cfg = DumpCfg { dir: dir.clone(), role: role.to_string(), keep_bytes };
    let drain = std::thread::Builder::new().name("evtrace-tierb".into()).spawn(|| loop {
        std::thread::sleep(Duration::from_millis(100));
        drain_all();
    });
    let dump = std::thread::Builder::new().name("evtrace-dump".into()).spawn(move || dump_loop(cfg));
    if drain.is_err() || dump.is_err() {
        tracing::warn!("evtrace ring: thread failed to start; Tier B off");
        return;
    }
    ON.store(true, Relaxed);
    tracing::info!(ring_mb = mb, dir = %dir.display(), "evtrace ring: on (dump: write seconds to <dir>/dump-request)");
}

fn dump_loop(cfg: DumpCfg) {
    let req = cfg.dir.join("dump-request");
    let mut n = 0u32;
    loop {
        std::thread::sleep(Duration::from_millis(250));
        let Ok(text) = std::fs::read_to_string(&req) else { continue };
        let _ = std::fs::remove_file(&req);
        let secs = text.trim().parse::<f64>().ok().filter(|s| *s > 0.0);
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
    let (body, recs) = select(&blocks, &open, t_min, &super::evtrace::time_fields());
    drop(blocks);
    let extra = serde_json::json!({ "tier": "B", "dump": { "seconds": secs, "t_from_raw": t_min, "t_to_raw": now } });
    let (header, _) = super::evtrace::header_bytes(&extra);
    let path = cfg.dir.join(format!("{}-{}-{}-{:03}.evt", cfg.role, super::evtrace::utc_stamp(), std::process::id(), n));
    let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path)?);
    f.write_all(b"EVT1")?;
    f.write_all(&(header.len() as u32).to_le_bytes())?;
    f.write_all(&header)?;
    // In 4 MB pieces with a breath between them: the hub's disk also serves
    // Engram and box-1 expert reads (ionice does nothing on these drives).
    for chunk in body.chunks(4 << 20) {
        f.write_all(chunk)?;
        f.flush()?;
        std::thread::sleep(Duration::from_millis(2));
    }
    f.flush()?;
    let bytes = 8 + header.len() as u64 + body.len() as u64;
    prune_bytes(&cfg.dir, cfg.keep_bytes, &path);
    Ok((path, bytes, recs))
}

/// Keep the newest dumps (by modification time) within `keep` bytes; never
/// the one just written.
fn prune_bytes(dir: &Path, keep: u64, current: &Path) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut v: Vec<(std::time::SystemTime, u64, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_str().is_some_and(|n| n.ends_with(".evt")))
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

    #[test]
    fn the_ring_stays_record_aligned_and_bounded() {
        let one = recs(&[1.0]).len();
        let mut r = Ring::new(BLOCK * 3);
        for i in 0..20_000 {
            r.append(&recs(&[i as f64]));
        }
        assert!(r.bytes <= BLOCK * 3 + BLOCK, "{}", r.bytes);
        let (blocks, open) = r.snapshot();
        let tidx = std::collections::HashMap::from([(31u16, vec![0usize, 1])]);
        let (body, n) = select(&blocks, &open, f64::NEG_INFINITY, &tidx);
        assert_eq!(body.len() as u64, n * one as u64, "whole records only");
        // The newest record survives; the oldest went first.
        let last = f64::from_le_bytes(body[body.len() - one + 4..body.len() - one + 12].try_into().unwrap());
        assert_eq!(last, 19_999.0);
        let first = f64::from_le_bytes(body[4..12].try_into().unwrap());
        assert!(first > 0.0, "evicted from the front");
    }

    #[test]
    fn a_dump_selects_by_record_time() {
        let mut r = Ring::new(1 << 30);
        r.append(&recs(&[10.0, 20.0, 30.0, 40.0]));
        let (blocks, open) = r.snapshot();
        let tidx = std::collections::HashMap::from([(31u16, vec![0usize, 1])]);
        let (_, n) = select(&blocks, &open, 25.0, &tidx);
        assert_eq!(n, 2);
    }
}
