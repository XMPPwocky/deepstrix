//! `het::evtrace` end to end (no GPU): init into a temp dir, emit one record
//! of every kind with known values, let the writer flush, then parse the
//! file back -- header, kinds, values, NaN padding, the SYS sampler and the
//! META heartbeat. `EVTRACE_KEEP=1` leaves the file for scripts/evtrace.py.
//!
//!   cargo test -p v4flash-kernels --release --features v41 --test evtrace_e2e -- --nocapture
use v4flash_kernels::het::evtrace::{self, Kind};
use v4flash_kernels::het::evtrace_kinds::{ALL, B2_READ, HUB_REQ};

fn parse(bytes: &[u8]) -> (serde_json::Value, Vec<(u16, Vec<f64>)>) {
    assert_eq!(&bytes[..4], b"EVT1");
    let hlen = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + hlen]).unwrap();
    let mut off = 8 + hlen;
    let mut recs = Vec::new();
    while off + 4 <= bytes.len() {
        let kind = u16::from_le_bytes([bytes[off], bytes[off + 1]]);
        let n = u16::from_le_bytes([bytes[off + 2], bytes[off + 3]]) as usize;
        off += 4;
        let v: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(bytes[off + 8 * i..off + 8 * i + 8].try_into().unwrap())).collect();
        off += 8 * n;
        recs.push((kind, v));
    }
    assert_eq!(off, bytes.len(), "trailing partial record");
    (header, recs)
}

/// Cost of `emit` on the caller's thread with the writer draining (the hot
/// path's price per record), and of the disabled check. Ignored: it writes
/// ~100 MB. `cargo test ... --test evtrace_e2e -- --ignored --nocapture`
#[test]
#[ignore]
fn bench_emit() {
    let dir = std::env::temp_dir().join(format!("evtrace-bench-{}", std::process::id()));
    // A separate process from the e2e test (cargo runs ignored tests alone
    // when asked), so init here is this binary's only init.
    std::env::set_var("V41_EVTRACE_DIR", &dir);
    std::env::set_var("V41_EVTRACE_SYS_MS", "0");
    evtrace::init("bench", serde_json::json!({}));
    let k = &v4flash_kernels::het::evtrace_kinds::B2_REQ;
    let v: Vec<f64> = (0..k.fields.len()).map(|i| i as f64).collect();
    let mut per = Vec::new();
    for _round in 0..20 {
        let n = 10_000;
        let t = std::time::Instant::now();
        for _ in 0..n {
            evtrace::emit(k, &v);
        }
        per.push(t.elapsed().as_nanos() as f64 / n as f64);
        // Let the writer drain so each round starts with an empty queue.
        std::thread::sleep(std::time::Duration::from_millis(300));
    }
    per.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let t = std::time::Instant::now();
    let mut acc = 0.0;
    for _ in 0..1_000_000 {
        acc += evtrace::now();
    }
    let now_ns = t.elapsed().as_nanos() as f64 / 1e6;
    std::hint::black_box(acc);
    std::thread::sleep(std::time::Duration::from_millis(1200));
    let meta_dropped = {
        let f = std::fs::read_dir(&dir).unwrap().next().unwrap().unwrap().path();
        let (_, recs) = parse(&std::fs::read(f).unwrap());
        recs.iter().filter(|(id, _)| *id == evtrace::META.id).map(|(_, m)| m[3]).fold(0.0, f64::max)
    };
    println!(
        "emit ({} fields): p10 {:.0} ns  p50 {:.0} ns  p90 {:.0} ns per record; now() {now_ns:.0} ns; dropped {meta_dropped}",
        k.fields.len(), per[2], per[10], per[18]
    );
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn evtrace_end_to_end() {
    let dir = std::env::temp_dir().join(format!("evtrace-e2e-{}", std::process::id()));
    std::env::set_var("V41_EVTRACE_DIR", &dir);
    std::env::set_var("V41_EVTRACE_SYS_MS", "20");
    evtrace::init("e2e", serde_json::json!({ "test": true }));
    assert!(evtrace::enabled());
    // One record per kind: field i = kind id * 1000 + i.
    let kinds: Vec<&Kind> = ALL.to_vec();
    for k in &kinds {
        let v: Vec<f64> = (0..k.fields.len()).map(|i| f64::from(k.id) * 1000.0 + i as f64).collect();
        evtrace::emit(k, &v);
    }
    // A short one: padded with NaN (release builds; debug asserts the length).
    if !cfg!(debug_assertions) {
        evtrace::emit(&HUB_REQ, &[1.0, 2.0]);
    }
    // A joinable pair (hub_req <-> b2_req on (seq, t2)) + its step, by name:
    // exercises `emit_named` and scripts/evtrace.py `join`.
    let t = evtrace::now();
    evtrace::emit_named(&HUB_REQ, &[
        ("t_submit", t), ("t_submit_end", t + 1e4), ("t1", t + 2e4), ("t4", t + 3.1e6), ("t_wait_enter", t + 1e6),
        ("t_wait_exit", t + 3.2e6), ("t2_b2", 5e12), ("t3_b2", 5e12 + 2.9e6), ("step", 3.0), ("seq", 7.0),
        ("clock_offset_ns", 5e12 - t - 1e5), ("clock_delay_ns", 1e5), ("rtt_us", 3100.0), ("srv_us", 2900.0),
    ]);
    evtrace::emit_named(&v4flash_kernels::het::evtrace_kinds::B2_REQ, &[
        ("seq", 7.0), ("t_frame", 5e12), ("t_dequeue", 5e12 + 1e5), ("t_merge_end", 5e12 + 1e5), ("t_hints_end", 5e12 + 1.1e5),
        ("t_run_start", 5e12 + 1.2e5), ("t_run_end", 5e12 + 2.5e6), ("t_d2h_end", 5e12 + 2.7e6), ("t_ready", 5e12 + 2.8e6),
        ("merged", 0.0), ("n_miss", 1.0),
    ]);
    evtrace::emit_named(&v4flash_kernels::het::evtrace_kinds::HUB_STEP, &[("step", 3.0), ("fwd_ms", 100.0), ("rows", 1.0)]);
    let t0 = evtrace::now();
    assert!(t0 > 0.0);
    // Writer flushes on a 250 ms idle tick and writes META every second.
    std::thread::sleep(std::time::Duration::from_millis(1400));
    let files: Vec<_> = std::fs::read_dir(&dir).unwrap().map(|e| e.unwrap().path()).collect();
    assert_eq!(files.len(), 1, "{files:?}");
    let name = files[0].file_name().unwrap().to_str().unwrap().to_string();
    assert!(name.starts_with("e2e-") && name.ends_with("-000.evt"), "{name}");
    let (header, recs) = parse(&std::fs::read(&files[0]).unwrap());
    assert_eq!(header["role"], "e2e");
    assert_eq!(header["extras"]["test"], true);
    let hk = header["kinds"].as_array().unwrap();
    for k in &kinds {
        let h = hk.iter().find(|x| x["id"] == k.id).unwrap_or_else(|| panic!("kind {} not in header", k.name));
        assert_eq!(h["fields"].as_array().unwrap().len(), k.fields.len(), "{}", k.name);
        let got = recs.iter().find(|(id, _)| *id == k.id).unwrap_or_else(|| panic!("no {} record", k.name));
        assert_eq!(got.1.len(), k.fields.len(), "{}", k.name);
        for (i, x) in got.1.iter().enumerate() {
            assert_eq!(*x, f64::from(k.id) * 1000.0 + i as f64, "{} field {i}", k.name);
        }
    }
    if !cfg!(debug_assertions) {
        let short = recs.iter().filter(|(id, _)| *id == HUB_REQ.id).nth(1).expect("padded record");
        assert_eq!(&short.1[..2], &[1.0, 2.0]);
        assert!(short.1[2..].iter().all(|x| x.is_nan()));
    }
    // SYS samples (20 ms) with a real clock and at least the first disk's counters.
    let sys: Vec<&Vec<f64>> = recs.iter().filter(|(id, _)| *id == evtrace::SYS.id).map(|(_, v)| v).collect();
    assert!(sys.len() >= 10, "{} sys samples", sys.len());
    assert!(sys.windows(2).all(|w| w[1][0] > w[0][0]), "sys clock not increasing");
    let has_disk = !header["sys_devices"].as_array().unwrap().is_empty();
    if has_disk {
        assert!(!sys[0][1].is_nan(), "d0_reads missing");
    }
    // META heartbeat: written counts, nothing dropped.
    let meta: Vec<&Vec<f64>> = recs.iter().filter(|(id, _)| *id == evtrace::META.id).map(|(_, v)| v).collect();
    assert!(!meta.is_empty(), "no meta record");
    assert_eq!(meta[0][3], 0.0, "dropped records");
    let _ = B2_READ.fields.len();
    if std::env::var("EVTRACE_KEEP").is_err() {
        std::fs::remove_dir_all(&dir).unwrap();
    } else {
        println!("kept {}", files[0].display());
    }
}
