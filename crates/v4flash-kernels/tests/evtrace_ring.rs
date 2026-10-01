//! Tier B end to end (docs/v41/EVTRACE_REBUILD_PLAN.md P1), in its own
//! process: evtrace on (env), Tier B records from two threads (one exits before
//! the dump: its buffer must reach the ring on exit), a `dump-request` file,
//! and the dump written to the ring dir with the strings and knobs in its
//! header and only the requested window's records. Host only (no GPU).

use std::time::{Duration, Instant};

use v4flash_kernels::het::evtrace;
use v4flash_kernels::het::evtrace_ring;

// A REGISTERED kind with a time field (an unregistered one has no time for a
// dump's window to read, and is kept whatever its age).
use v4flash_kernels::het::evtrace::KNOB as DEV;

#[test]
fn tier_b_dumps_on_request() {
    let base = std::env::temp_dir().join(format!("evtrace-ring-it-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let dir = base.join("evt");
    let ring = base.join("ring");
    std::env::set_var("V41_EVTRACE_DIR", &dir);
    std::env::set_var("V41_EVTRACE_RING_DIR", &ring);
    std::env::set_var("V41_EVTRACE_RING_MB", "8");
    std::env::set_var("V41_EVTRACE_SYS_MS", "0");
    evtrace::init("it", serde_json::json!({}));
    assert!(evtrace_ring::enabled());

    let stage = evtrace::intern("dgpu.it_stage");
    let old = evtrace::now();
    evtrace_ring::emit_b(&DEV, &[old - 60e9, stage, stage, 3.0]); // outside a 10 s window
    let t = std::thread::spawn(move || {
        for i in 0..100 {
            let now = evtrace::now();
            evtrace_ring::emit_b(&DEV, &[now + i as f64, stage, stage, 3.0]);
        }
        // exits without flushing: its buffer reaches the ring on thread exit
    });
    t.join().unwrap();
    for _ in 0..50 {
        let now = evtrace::now();
        evtrace_ring::emit_b(&DEV, &[now, stage, stage, 3.0]);
    }
    evtrace_ring::flush_local();

    std::fs::write(ring.join("dump-request.tmp"), "10").unwrap();
    std::fs::rename(ring.join("dump-request.tmp"), ring.join("dump-request")).unwrap();
    let t0 = Instant::now();
    let dump = loop {
        let found = std::fs::read_dir(&ring)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.extension().is_some_and(|x| x == "evt"));
        if let Some(p) = found {
            std::thread::sleep(Duration::from_millis(200)); // let the write finish
            break p;
        }
        assert!(t0.elapsed() < Duration::from_secs(10), "no dump");
        std::thread::sleep(Duration::from_millis(50));
    };
    assert!(!ring.join("dump-request").exists(), "the request is consumed");

    let b = std::fs::read(&dump).unwrap();
    assert_eq!(&b[..4], b"EVT1");
    let hlen = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
    let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hlen]).unwrap();
    assert_eq!(h["tier"], "B");
    assert_eq!(h["format_rev"], 2);
    let strings = h["strings"].as_array().unwrap();
    assert_eq!(strings[stage as usize], "dgpu.it_stage");
    assert!(h["knobs_at_open"].is_object());
    let (mut off, mut n_dev) = (8 + hlen, 0);
    while off + 4 <= b.len() {
        let kind = u16::from_le_bytes([b[off], b[off + 1]]);
        let n = u16::from_le_bytes([b[off + 2], b[off + 3]]) as usize;
        if kind == DEV.id {
            n_dev += 1;
        }
        off += 4 + 8 * n;
    }
    assert_eq!(off, b.len(), "whole records");
    assert_eq!(n_dev, 150, "the exited thread's 100 + this thread's 50; the 60 s old one is outside the window");
    let _ = std::fs::remove_dir_all(&base);
}
