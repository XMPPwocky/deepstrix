//! Tier B device timing on the GPUs (docs/v41/EVTRACE_REBUILD_PLAN.md P3), in its
//! own process: stages recorded on every device are handed to the Tier B
//! thread at each pool reset, timed against its calibration anchors, and come
//! back as `step_dev` (Tier A) and `dev` (a Tier B dump) records plus decode
//! sums. Checks: the steps convert (no drops), causality holds ((a) no stage
//! starts before the host recorded it, (b) none ends after a sync of its
//! stream returned), the anchors succeed on idle GPUs, and each stage's
//! calibrated duration matches a plain `elapsed` twin inside it.
//!
//! GPU, ~10 s: run with the server DOWN.
//!   cargo test --release -p v4flash-kernels --features v41 --test evtrace_dev_gpu -- --ignored --nocapture

use std::collections::HashMap;
use std::time::Duration;

use v4flash_hip::{Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::het::trace::{ctx_step, EventPool};
use v4flash_kernels::het::{evtrace, evtrace_dev, evtrace_ring};

const STEPS: u64 = 40;

/// Every record of `kind` in the `.evt` files under `dir` whose names start
/// with `prefix`, as field maps.
fn read_kind(dir: &std::path::Path, prefix: &str, kind: &str) -> Vec<HashMap<String, f64>> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().filter_map(|e| e.ok()) {
        let p = e.path();
        let name = p.file_name().unwrap().to_string_lossy().to_string();
        if !name.starts_with(prefix) || !name.ends_with(".evt") {
            continue;
        }
        let b = std::fs::read(&p).unwrap();
        let hlen = u32::from_le_bytes(b[4..8].try_into().unwrap()) as usize;
        let h: serde_json::Value = serde_json::from_slice(&b[8..8 + hlen]).unwrap();
        let k = h["kinds"].as_array().unwrap().iter().find(|k| k["name"] == kind).cloned();
        let Some(k) = k else { continue };
        let id = k["id"].as_u64().unwrap() as u16;
        let fields: Vec<String> = k["fields"].as_array().unwrap().iter().map(|f| f.as_str().unwrap().to_string()).collect();
        let mut off = 8 + hlen;
        while off + 4 <= b.len() {
            let kid = u16::from_le_bytes([b[off], b[off + 1]]);
            let n = u16::from_le_bytes([b[off + 2], b[off + 3]]) as usize;
            if off + 4 + 8 * n > b.len() {
                break;
            }
            if kid == id {
                let v: Vec<f64> = (0..n).map(|i| f64::from_le_bytes(b[off + 4 + 8 * i..off + 12 + 8 * i].try_into().unwrap())).collect();
                out.push(fields.iter().cloned().zip(v).collect());
            }
            off += 4 + 8 * n;
        }
    }
    out
}

struct Dev {
    label: &'static str,
    stages: [&'static str; 2],
    stream: Stream,
    pool: EventPool,
    big: DeviceBuffer<u8>,
    small: DeviceBuffer<u8>,
    /// Per step: the plain `elapsed` of the big memset (ms).
    twin: HashMap<u64, f32>,
}

#[test]
#[ignore = "GPU: run with the server down"]
fn tier_b_times_device_stages() {
    let base = std::env::temp_dir().join(format!("evtrace-dev-gpu-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let (dir, ring) = (base.join("evt"), base.join("ring"));
    std::env::set_var("V41_EVTRACE_DIR", &dir);
    std::env::set_var("V41_EVTRACE_RING_DIR", &ring);
    std::env::set_var("V41_EVTRACE_RING_MB", "16");
    std::env::set_var("V41_EVTRACE_SYS_MS", "0");
    evtrace::init("hub", serde_json::json!({}));
    assert!(evtrace_ring::enabled() && evtrace_dev::offload_on());

    let mut devs: Vec<Dev> = Vec::new();
    for id in 0..Device::count().unwrap() {
        let d = Device::new(id);
        d.set_current().unwrap();
        let igpu = d.properties().unwrap().integrated;
        let (label, stages) = if igpu { ("igpu", ["igpu.pair_kwide", "igpu.q2k_down"]) } else { ("dgpu", ["dgpu.q_chain", "dgpu.router"]) };
        let pool = EventPool::new(label, 4096).unwrap();
        pool.set_enabled(true);
        devs.push(Dev { label, stages, stream: Stream::new(id).unwrap(), pool, big: DeviceBuffer::new(id, 256 << 20).unwrap(), small: DeviceBuffer::new(id, 32 << 20).unwrap(), twin: HashMap::new() });
    }
    for step in 0..STEPS {
        for d in devs.iter_mut() {
            d.pool.reset();
            let _ctx = ctx_step(step);
            let (a, b) = (Event::new().unwrap(), Event::new().unwrap());
            {
                let _t = d.pool.stage(d.stages[0], &d.stream).unwrap();
                a.record(&d.stream).unwrap();
                d.big.fill_zero_async(&d.stream).unwrap();
                b.record(&d.stream).unwrap();
            }
            {
                let _t = d.pool.stage(d.stages[1], &d.stream).unwrap();
                d.small.fill_zero_async(&d.stream).unwrap();
            }
            d.stream.synchronize().unwrap();
            d.pool.note_sync(&d.stream);
            d.twin.insert(step, Event::elapsed_ms(&a, &b).unwrap());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    // Hand the last buffers off, let Tier B finish and Tier A flush, fold the sums.
    for d in &devs {
        d.pool.reset();
    }
    std::thread::sleep(Duration::from_secs(3));
    for d in &devs {
        d.pool.reset();
        let (sums, steps) = d.pool.take_dev_sums();
        let skipped = d.pool.take_skipped();
        println!("{}: sums over {steps} steps ({skipped} resets skipped): {sums:?}", d.label);
        // The first handoff finds no spare (Tier B makes them): one step lost.
        assert!(steps >= STEPS - 2, "{}: {steps} steps came back", d.label);
        assert!(sums.iter().any(|s| s.0 == d.stages[0]), "{}: no sums", d.label);
    }

    let step_dev = read_kind(&dir, "hub-", "step_dev");
    let cal = read_kind(&dir, "hub-", "cal");
    for (i, d) in devs.iter().enumerate() {
        let dev_id = evtrace::intern(d.label);
        let recs: Vec<_> = step_dev.iter().filter(|r| r["device"] == dev_id).collect();
        let anchors_ok = cal.iter().filter(|r| r["device"] == dev_id && r["ok"] == 1.0).count();
        let anchors_bad = cal.iter().filter(|r| r["device"] == dev_id && r["ok"] == 0.0).count();
        let sum = |f: &str| recs.iter().map(|r| r[f]).filter(|x| x.is_finite()).sum::<f64>();
        println!(
            "{} (device {i}): {} step_dev, pairs {} dropped {} deferred {} viol_a {} viol_b {} of {}, anchors {anchors_ok} ok / {anchors_bad} discarded, q_us_max {:.1}",
            d.label, recs.len(), sum("pairs"), sum("dropped"), sum("deferred"), sum("viol_a"), sum("viol_b"), sum("checked_b"),
            recs.iter().map(|r| r["q_us_max"]).fold(0.0, f64::max)
        );
        assert!(recs.len() as u64 >= STEPS - 2, "{}: {} step_dev records", d.label, recs.len());
        assert_eq!(sum("dropped"), 0.0, "{}: dropped pairs", d.label);
        assert_eq!(sum("viol_a"), 0.0, "{}: a stage started before the host recorded it", d.label);
        assert_eq!(sum("viol_b"), 0.0, "{}: a stage ended after its stream's sync returned", d.label);
        assert!(sum("checked_b") >= sum("pairs") * 0.9, "{}: rule (b) checked {} of {}", d.label, sum("checked_b"), sum("pairs"));
        assert!(anchors_ok >= 5 && anchors_bad == 0, "{}: anchors {anchors_ok} ok / {anchors_bad} discarded (idle GPU)", d.label);
        // The calibrated stage holds its twin (markers add a few us).
        let short = if d.label == "dgpu" { "d_q_chain" } else { "i_pair_kwide" };
        let mut worst = 0f64;
        for r in &recs {
            let (Some(&tw), ms) = (d.twin.get(&(r["step"] as u64)), r[short]) else { continue };
            if ms.is_finite() {
                let diff = ms - tw as f64;
                worst = worst.max(diff.abs());
                assert!((-0.01..0.1).contains(&diff), "{} step {}: calibrated {ms:.4} ms vs elapsed {tw:.4} ms", d.label, r["step"]);
            }
        }
        println!("{}: calibrated vs elapsed, worst {:.1} us", d.label, worst * 1e3);
    }

    // The dump: one `dev` record per stage.
    std::fs::write(ring.join("dump-request.tmp"), "all").unwrap();
    std::fs::rename(ring.join("dump-request.tmp"), ring.join("dump-request")).unwrap();
    let mut dev = Vec::new();
    for _ in 0..100 {
        std::thread::sleep(Duration::from_millis(100));
        dev = read_kind(&ring, "hub-dump-", "dev");
        if !dev.is_empty() {
            std::thread::sleep(Duration::from_millis(300));
            dev = read_kind(&ring, "hub-dump-", "dev");
            break;
        }
    }
    println!("dump: {} dev records", dev.len());
    assert!(dev.len() as u64 >= (STEPS - 2) * 2 * devs.len() as u64, "{} dev records", dev.len());
    assert!(dev.iter().all(|r| r["t_end"] >= r["t_start"] && r["t_start"] >= r["t_host"] - r["q_us"] * 1e3));
    let _ = std::fs::remove_dir_all(&base);
}
