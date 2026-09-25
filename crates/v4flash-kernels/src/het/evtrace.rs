//! EVENT TRACE (2026-09-25): one binary record per EVENT -- per box-2 request
//! (on both boxes), per expert read on box 2, per decode step, per scheduler
//! phase change, plus periodic system samples -- so the analysis can build
//! HISTOGRAMS and join events across the two boxes. The `ms.stage` window
//! means cannot: step variance is ~all box-2 paging time and half of that is a
//! per-miss TAIL (3.8 ms typical, 14+ ms in 10% of windows), which a mean over
//! 20 steps hides.
//!
//! Off unless `V41_EVTRACE_DIR` is set: `init` is then a no-op and `emit`
//! costs one relaxed load. Emitters copy a fixed-size record into a bounded
//! channel with `try_send` -- a full queue DROPS the record and counts it
//! (`meta.dropped`), so the hot path never blocks on the disk. One writer
//! thread appends to `<dir>/<role>-<utc>-<pid>-<n>.evt`, rotating at
//! `V41_EVTRACE_MAX_MB` (default 1024) and keeping the newest
//! `V41_EVTRACE_KEEP` (default 16) files of this role.
//!
//! File format (little endian): `b"EVT1"`, u32 header length, a JSON header
//! (role, pid, clock pair, every kind's field names, extras), then records:
//! u16 kind, u16 n, n x f64. Every timestamp field (`t_*`) is
//! CLOCK_MONOTONIC_RAW ns of the EMITTING box -- the clock of the wire's
//! t1..t4 -- so the hub's per-request NTP quadruples put both boxes on one
//! timeline. Reader: scripts/evtrace.py.

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::mpsc::{sync_channel, Receiver, RecvTimeoutError, SyncSender};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::remote_experts::{monotonic_raw_ns, realtime_ns};

/// Fields a record can carry (the largest kind must fit).
pub const MAX_FIELDS: usize = 112;

/// One event type: a stable id and its field names, written into every file's
/// header so the reader needs no copy of this source.
pub struct Kind {
    pub id: u16,
    pub name: &'static str,
    pub fields: &'static [&'static str],
}

/// Writer heartbeat, once a second: the clock pair (drift between the raw and
/// wall clocks over a run), and the drop / write counters.
pub static META: Kind = Kind {
    id: 0,
    name: "meta",
    fields: &["t_mono_raw", "t_realtime", "written", "dropped", "files", "queue_cap"],
};

/// Periodic system sample (`V41_EVTRACE_SYS_MS`, default 100). `dK_*` are the
/// cumulative /proc/diskstats counters of the K-th NVMe whole disk (names in
/// the header's `sys_devices`); `psi_*` cumulative /proc/pressure totals (us);
/// `self_*` this process (/proc/self/io, /proc/self/status).
pub static SYS: Kind = Kind {
    id: 1,
    name: "sys",
    fields: &[
        "t",
        "d0_reads", "d0_rsect", "d0_rms", "d0_writes", "d0_wsect", "d0_wms", "d0_inflight", "d0_io_ms", "d0_wtd_ms",
        "d1_reads", "d1_rsect", "d1_rms", "d1_writes", "d1_wsect", "d1_wms", "d1_inflight", "d1_io_ms", "d1_wtd_ms",
        "d2_reads", "d2_rsect", "d2_rms", "d2_writes", "d2_wsect", "d2_wms", "d2_inflight", "d2_io_ms", "d2_wtd_ms",
        "psi_io_some", "psi_io_full", "psi_cpu_some", "psi_mem_some", "psi_mem_full",
        "load1", "procs_running", "procs_blocked",
        "mem_avail_kb", "cached_kb", "dirty_kb", "writeback_kb",
        "self_read_bytes", "self_write_bytes", "self_rchar", "self_threads",
        "self_vol_ctxsw", "self_invol_ctxsw", "ctxt_total", "sample_us",
    ],
};

/// Every kind this binary can emit, in the header of every file.
fn kinds() -> Vec<&'static Kind> {
    let mut v: Vec<&'static Kind> = vec![&META, &SYS];
    v.extend(super::evtrace_kinds::ALL.iter().copied());
    v
}

#[derive(Clone, Copy)]
struct Rec {
    kind: u16,
    n: u16,
    v: [f64; MAX_FIELDS],
}

static ENABLED: AtomicBool = AtomicBool::new(false);
static TX: OnceLock<SyncSender<Rec>> = OnceLock::new();
static DROPPED: AtomicU64 = AtomicU64::new(0);
static WRITTEN: AtomicU64 = AtomicU64::new(0);
const QUEUE_CAP: usize = 8192;

/// Is the trace on? One relaxed load: call sites that must gather fields
/// check this first.
#[inline]
pub fn enabled() -> bool {
    ENABLED.load(Relaxed)
}

/// CLOCK_MONOTONIC_RAW ns as f64 (exact below 2^53 ns, ~104 days of uptime).
#[inline]
pub fn now() -> f64 {
    monotonic_raw_ns() as f64
}

/// An `Instant` in the past as CLOCK_MONOTONIC_RAW ns: now's raw stamp minus
/// its age (CLOCK_MONOTONIC and _RAW differ only by NTP slew, ppm over the
/// milliseconds this is used for).
#[inline]
pub fn inst_to_raw(t: Instant) -> f64 {
    let (raw, now) = (monotonic_raw_ns() as f64, Instant::now());
    raw - now.saturating_duration_since(t).as_nanos() as f64
}

/// Record one event. `v` must list `k.fields` in order; a short `v` is padded
/// with NaN (= "not measured"), a long one truncated.
#[inline]
pub fn emit(k: &Kind, v: &[f64]) {
    if !enabled() {
        return;
    }
    debug_assert_eq!(v.len(), k.fields.len(), "evtrace kind {} field count", k.name);
    let n = k.fields.len().min(MAX_FIELDS);
    let mut r = Rec { kind: k.id, n: n as u16, v: [f64::NAN; MAX_FIELDS] };
    let m = v.len().min(n);
    r.v[..m].copy_from_slice(&v[..m]);
    if let Some(tx) = TX.get() {
        if tx.try_send(r).is_err() {
            DROPPED.fetch_add(1, Relaxed);
        }
    }
}

/// Record one event given as `(field, value)` pairs; unnamed fields are NaN,
/// unknown names are ignored (debug-asserted). For the rare, partial records
/// (a park-served request); hot paths build the vector in order.
pub fn emit_named(k: &Kind, pairs: &[(&str, f64)]) {
    if !enabled() {
        return;
    }
    let mut v = vec![f64::NAN; k.fields.len()];
    for &(name, x) in pairs {
        set_named(k, &mut v, name, x);
    }
    emit(k, &v);
}

/// Overwrite field `name` of an in-order record `v` of kind `k`.
pub fn set_named(k: &Kind, v: &mut [f64], name: &str, x: f64) {
    match k.fields.iter().position(|f| *f == name) {
        Some(i) if i < v.len() => v[i] = x,
        _ => debug_assert!(false, "evtrace kind {} has no field {name}", k.name),
    }
}

struct Cfg {
    dir: PathBuf,
    role: String,
    max_bytes: u64,
    keep: usize,
    header: Vec<u8>,
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The NVMe whole disks (`nvme0n1`, ...), sorted, at most 3.
fn nvme_disks() -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_to_string("/proc/diskstats")
        .unwrap_or_default()
        .lines()
        .filter_map(|l| l.split_whitespace().nth(2).map(str::to_string))
        .filter(|n| {
            n.strip_prefix("nvme").is_some_and(|r| {
                let mut it = r.splitn(2, 'n');
                let (a, b) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
                !a.is_empty() && !b.is_empty() && a.bytes().all(|c| c.is_ascii_digit()) && b.bytes().all(|c| c.is_ascii_digit())
            })
        })
        .collect();
    v.sort();
    v.truncate(3);
    v
}

/// Start the trace if `V41_EVTRACE_DIR` is set: `role` names the files
/// ("hub", "b2"), `extras` goes into the header verbatim (config worth
/// knowing at analysis time). Idempotent; errors disable the trace with a
/// warning rather than failing the caller.
pub fn init(role: &str, extras: serde_json::Value) {
    let Ok(dir) = std::env::var("V41_EVTRACE_DIR") else { return };
    if TX.get().is_some() {
        return;
    }
    let dir = PathBuf::from(dir);
    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(error = %e, dir = %dir.display(), "evtrace: cannot create dir; trace off");
        return;
    }
    let disks = nvme_disks();
    let kinds_json: Vec<serde_json::Value> = kinds()
        .iter()
        .map(|k| serde_json::json!({ "id": k.id, "name": k.name, "fields": k.fields }))
        .collect();
    let header = serde_json::json!({
        "format": "EVT1",
        "role": role,
        "pid": std::process::id(),
        "clock": "CLOCK_MONOTONIC_RAW ns",
        "t_mono_raw_at_open": monotonic_raw_ns(),
        "t_realtime_at_open": realtime_ns(),
        "sys_devices": disks,
        "kinds": kinds_json,
        "extras": extras,
    });
    let cfg = Cfg {
        dir,
        role: role.to_string(),
        max_bytes: env_u64("V41_EVTRACE_MAX_MB", 1024) << 20,
        keep: env_u64("V41_EVTRACE_KEEP", 16).max(1) as usize,
        header: serde_json::to_vec(&header).unwrap_or_default(),
    };
    let (tx, rx) = sync_channel::<Rec>(QUEUE_CAP);
    if TX.set(tx).is_err() {
        return;
    }
    let spawned = std::thread::Builder::new().name("evtrace-writer".into()).spawn(move || writer(rx, cfg));
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "evtrace: writer thread failed to start; trace off");
        return;
    }
    ENABLED.store(true, Relaxed);
    let period = env_u64("V41_EVTRACE_SYS_MS", 100);
    if period > 0 {
        let _ = std::thread::Builder::new()
            .name("evtrace-sys".into())
            .spawn(move || sys_sampler(Duration::from_millis(period), disks));
    }
    tracing::info!(role, "evtrace: on");
}

/// `init` with this process's argv and every V41_* / GPU_* / HIP_* variable
/// in the header (for binaries without serde_json of their own).
pub fn init_env(role: &str) {
    let env: serde_json::Map<String, serde_json::Value> = std::env::vars()
        .filter(|(k, _)| k.starts_with("V41_") || k.starts_with("GPU_") || k.starts_with("HIP_"))
        .map(|(k, v)| (k, serde_json::Value::String(v)))
        .collect();
    let argv: Vec<String> = std::env::args().collect();
    init(role, serde_json::json!({ "argv": argv, "env": env }));
}

fn utc_stamp() -> String {
    // yyyymmdd-hhmmss from CLOCK_REALTIME, without a date crate (days-from-civil).
    let s = (realtime_ns() / 1_000_000_000) as i64;
    let (days, sod) = (s.div_euclid(86_400), s.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}{m:02}{d:02}-{:02}{:02}{:02}", sod / 3600, sod % 3600 / 60, sod % 60)
}

fn open_file(cfg: &Cfg, n: u32) -> std::io::Result<(BufWriter<File>, u64)> {
    let path = cfg.dir.join(format!("{}-{}-{}-{:03}.evt", cfg.role, utc_stamp(), std::process::id(), n));
    let mut w = BufWriter::with_capacity(1 << 20, File::create(&path)?);
    w.write_all(b"EVT1")?;
    w.write_all(&(cfg.header.len() as u32).to_le_bytes())?;
    w.write_all(&cfg.header)?;
    prune(&cfg.dir, &cfg.role, cfg.keep, &path);
    Ok((w, 8 + cfg.header.len() as u64))
}

/// Keep only the newest `keep` files of `role`, newest by MODIFICATION time
/// (the name's stamp is the wall clock, which box 2 has had ~36 h off), and
/// never `current`, the file just opened.
fn prune(dir: &Path, role: &str, keep: usize, current: &Path) {
    let prefix = format!("{role}-");
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut v: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name().to_str().is_some_and(|n| n.starts_with(&prefix) && n.ends_with(".evt"))
        })
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .filter(|(_, p)| p != current)
        .collect();
    v.sort();
    // `current` is the newest by construction and counts toward `keep`.
    while v.len() + 1 > keep {
        let _ = std::fs::remove_file(v.remove(0).1);
    }
}

fn write_rec(w: &mut BufWriter<File>, r: &Rec) -> std::io::Result<u64> {
    let n = r.n as usize;
    let mut buf = [0u8; 4 + 8 * MAX_FIELDS];
    buf[..2].copy_from_slice(&r.kind.to_le_bytes());
    buf[2..4].copy_from_slice(&r.n.to_le_bytes());
    for (i, x) in r.v[..n].iter().enumerate() {
        buf[4 + 8 * i..12 + 8 * i].copy_from_slice(&x.to_le_bytes());
    }
    w.write_all(&buf[..4 + 8 * n])?;
    Ok(4 + 8 * n as u64)
}

fn writer(rx: Receiver<Rec>, cfg: Cfg) {
    let fail = |e: std::io::Error| {
        tracing::warn!(error = %e, "evtrace: write failed; trace off");
        ENABLED.store(false, Relaxed);
    };
    let mut n_file = 0u32;
    let (mut w, mut bytes) = match open_file(&cfg, n_file) {
        Ok(x) => x,
        Err(e) => return fail(e),
    };
    let mut last_meta = Instant::now().checked_sub(Duration::from_secs(2)).unwrap_or_else(Instant::now);
    let mut dirty = false;
    loop {
        let got = rx.recv_timeout(Duration::from_millis(250));
        let r = match got {
            Ok(r) => Some(r),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => break,
        };
        if let Some(r) = r {
            match write_rec(&mut w, &r) {
                Ok(b) => {
                    bytes += b;
                    dirty = true;
                    WRITTEN.fetch_add(1, Relaxed);
                }
                Err(e) => return fail(e),
            }
        } else if dirty {
            if let Err(e) = w.flush() {
                return fail(e);
            }
            dirty = false;
        }
        if last_meta.elapsed() >= Duration::from_secs(1) {
            last_meta = Instant::now();
            let mut m = Rec { kind: META.id, n: META.fields.len() as u16, v: [f64::NAN; MAX_FIELDS] };
            m.v[..6].copy_from_slice(&[
                monotonic_raw_ns() as f64,
                realtime_ns() as f64,
                WRITTEN.load(Relaxed) as f64,
                DROPPED.load(Relaxed) as f64,
                (n_file + 1) as f64,
                QUEUE_CAP as f64,
            ]);
            if let Err(e) = write_rec(&mut w, &m).and_then(|b| {
                bytes += b;
                w.flush()
            }) {
                return fail(e);
            }
            dirty = false;
        }
        if bytes >= cfg.max_bytes {
            if let Err(e) = w.flush() {
                return fail(e);
            }
            n_file += 1;
            match open_file(&cfg, n_file) {
                Ok((nw, nb)) => {
                    w = nw;
                    bytes = nb;
                }
                Err(e) => return fail(e),
            }
        }
    }
    let _ = w.flush();
}

/// Parse "some avg10=.. avg60=.. avg300=.. total=N" / "full ..." lines.
fn psi_totals(path: &str) -> (f64, f64) {
    let s = std::fs::read_to_string(path).unwrap_or_default();
    let mut out = (f64::NAN, f64::NAN);
    for l in s.lines() {
        let tot = l.split_whitespace().find_map(|t| t.strip_prefix("total=")).and_then(|v| v.parse::<f64>().ok());
        if l.starts_with("some") {
            out.0 = tot.unwrap_or(f64::NAN);
        } else if l.starts_with("full") {
            out.1 = tot.unwrap_or(f64::NAN);
        }
    }
    out
}

fn kv_field(s: &str, key: &str) -> f64 {
    s.lines()
        .find(|l| l.starts_with(key))
        .and_then(|l| l[key.len()..].split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .unwrap_or(f64::NAN)
}

fn sys_sampler(period: Duration, disks: Vec<String>) {
    loop {
        if !enabled() {
            return;
        }
        let t0 = Instant::now();
        let mut v = Vec::with_capacity(SYS.fields.len());
        v.push(now());
        let ds = std::fs::read_to_string("/proc/diskstats").unwrap_or_default();
        for k in 0..3 {
            let row: Option<Vec<f64>> = disks.get(k).and_then(|name| {
                ds.lines().find_map(|l| {
                    let f: Vec<&str> = l.split_whitespace().collect();
                    (f.len() >= 14 && f[2] == name.as_str())
                        .then(|| [3usize, 5, 6, 7, 9, 10, 11, 12, 13].iter().map(|&i| f[i].parse().unwrap_or(f64::NAN)).collect())
                })
            });
            v.extend(row.unwrap_or_else(|| vec![f64::NAN; 9]));
        }
        let io = psi_totals("/proc/pressure/io");
        let cpu = psi_totals("/proc/pressure/cpu");
        let mem = psi_totals("/proc/pressure/memory");
        v.extend([io.0, io.1, cpu.0, mem.0, mem.1]);
        let la = std::fs::read_to_string("/proc/loadavg").unwrap_or_default();
        v.push(la.split_whitespace().next().and_then(|x| x.parse().ok()).unwrap_or(f64::NAN));
        let stat = std::fs::read_to_string("/proc/stat").unwrap_or_default();
        v.push(kv_field(&stat, "procs_running"));
        v.push(kv_field(&stat, "procs_blocked"));
        let mi = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        for key in ["MemAvailable:", "Cached:", "Dirty:", "Writeback:"] {
            v.push(kv_field(&mi, key));
        }
        let sio = std::fs::read_to_string("/proc/self/io").unwrap_or_default();
        v.push(kv_field(&sio, "read_bytes:"));
        v.push(kv_field(&sio, "write_bytes:"));
        v.push(kv_field(&sio, "rchar:"));
        let st = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
        v.push(kv_field(&st, "Threads:"));
        v.push(kv_field(&st, "voluntary_ctxt_switches:"));
        v.push(kv_field(&st, "nonvoluntary_ctxt_switches:"));
        v.push(kv_field(&stat, "ctxt"));
        v.push(t0.elapsed().as_secs_f64() * 1e6);
        emit(&SYS, &v);
        std::thread::sleep(period.saturating_sub(t0.elapsed()));
    }
}

/// Small dense ids for things the trace names by address (a lane's scratch):
/// the first address seen gets 0, the next 1, ... (up to 16; later ones 15).
pub fn small_id(addr: usize) -> f64 {
    static SEEN: [AtomicU64; 16] = [const { AtomicU64::new(0) }; 16];
    for (i, s) in SEEN.iter().enumerate() {
        let cur = s.load(Relaxed);
        if cur == addr as u64 {
            return i as f64;
        }
        if cur == 0 && s.compare_exchange(0, addr as u64, Relaxed, Relaxed).is_ok() {
            return i as f64;
        }
        if s.load(Relaxed) == addr as u64 {
            return i as f64;
        }
    }
    15.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn utc_stamp_shape() {
        let s = utc_stamp();
        assert_eq!(s.len(), 15, "{s}");
        assert_eq!(&s[8..9], "-");
        assert!(s[..4].parse::<u32>().unwrap() >= 2024);
    }

    #[test]
    fn record_round_trip_bytes() {
        let dir = std::env::temp_dir().join(format!("evtrace-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = Cfg { dir: dir.clone(), role: "t".into(), max_bytes: 1 << 30, keep: 2, header: b"{}".to_vec() };
        let (mut w, _) = open_file(&cfg, 0).unwrap();
        let mut r = Rec { kind: 7, n: 3, v: [0.0; MAX_FIELDS] };
        r.v[..3].copy_from_slice(&[1.5, f64::NAN, 1e15]);
        assert_eq!(write_rec(&mut w, &r).unwrap(), 4 + 24);
        w.flush().unwrap();
        drop(w);
        let f = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
        let b = std::fs::read(&f).unwrap();
        assert_eq!(&b[..4], b"EVT1");
        assert_eq!(u32::from_le_bytes(b[4..8].try_into().unwrap()), 2);
        let p = &b[10..];
        assert_eq!(u16::from_le_bytes([p[0], p[1]]), 7);
        assert_eq!(u16::from_le_bytes([p[2], p[3]]), 3);
        assert_eq!(f64::from_le_bytes(p[4..12].try_into().unwrap()), 1.5);
        assert!(f64::from_le_bytes(p[12..20].try_into().unwrap()).is_nan());
        assert_eq!(f64::from_le_bytes(p[20..28].try_into().unwrap()), 1e15);
        // keep=2 prunes to the newest two (by mtime; the one just opened survives).
        for n in 1..4 {
            std::thread::sleep(std::time::Duration::from_millis(20));
            let _ = open_file(&cfg, n).unwrap();
        }
        let names: Vec<String> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        assert!(names.iter().any(|n| n.ends_with("-003.evt")), "newest pruned: {names:?}");
        let left = std::fs::read_dir(&dir).unwrap().count();
        assert_eq!(left, 2);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn kinds_fit_and_ids_unique() {
        let ks = kinds();
        let mut ids: Vec<u16> = ks.iter().map(|k| k.id).collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), ks.len(), "duplicate kind id");
        for k in ks {
            assert!(k.fields.len() <= MAX_FIELDS, "{} has {} fields", k.name, k.fields.len());
        }
    }

    #[test]
    fn small_ids_are_dense() {
        let a = small_id(0x1000_0001);
        let b = small_id(0x1000_0002);
        assert_ne!(a, b);
        assert_eq!(small_id(0x1000_0001), a);
    }
}
