//! Tier B device timing on the GPUs (docs/v41/EVTRACE_REBUILD_PLAN.md P3), in its
//! own process: stages recorded on every device are handed to the Tier B
//! thread at each pool reset, timed against its calibration anchors, and come
//! back as `step_dev` (Tier A) and `dev` (a Tier B dump) records plus decode
//! sums. Checks: the steps convert (no drops), causality holds ((a) no stage
//! starts before the host recorded it, (b) none ends after a sync of its
//! stream returned), the anchors succeed on idle GPUs, and each stage's
//! calibrated duration matches a plain `elapsed` twin inside it.
//!
//! Also printed (the review's two runtime questions): the stage record
//! calls' host latency early vs late in the run (6 buffers per pool keep ~6x
//! the recorded events alive: does `hipEventRecord` slow down?), and the
//! anchors' bracket bound per device. `EVTRACE_GPU_STEPS=3000` runs long.
//!
//! GPU, ~10 s: run with the server DOWN.
//!   cargo test --release -p v4flash-kernels --features v41 --test evtrace_dev_gpu -- --ignored --nocapture

use std::collections::HashMap;
use std::time::Duration;

use v4flash_hip::{Device, DeviceBuffer, Event, Stream};
use v4flash_kernels::het::trace::{ctx_step, EventPool};
use v4flash_kernels::het::{evtrace, evtrace_dev, evtrace_ring};

fn steps() -> u64 {
    std::env::var("EVTRACE_GPU_STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(40)
}

/// p50 / p99 / max of `v` (us).
fn pcts(v: &mut [f64]) -> (f64, f64, f64) {
    v.sort_by(|a, b| a.total_cmp(b));
    let at = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
    (at(0.5), at(0.99), v[v.len() - 1])
}

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
    /// Per step: host us of the stage calls (both opens and closes).
    rec_us: Vec<f64>,
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
        devs.push(Dev { label, stages, stream: Stream::new(id).unwrap(), pool, big: DeviceBuffer::new(id, 256 << 20).unwrap(), small: DeviceBuffer::new(id, 32 << 20).unwrap(), twin: HashMap::new(), rec_us: Vec::new() });
    }
    const STEPS_FLOOR: u64 = 2;
    let n_steps = steps();
    let (a, b) = (Event::new().unwrap(), Event::new().unwrap());
    for step in 0..n_steps {
        for d in devs.iter_mut() {
            d.pool.reset();
            let _ctx = ctx_step(step);
            let _g = Device::new(d.stream.device_id()).scoped_current().unwrap();
            let (a, b) = (Event::new().unwrap(), Event::new().unwrap());
            let t = std::time::Instant::now();
            let o = d.pool.open(d.stages[0], &d.stream).unwrap();
            let mut us = t.elapsed().as_secs_f64() * 1e6;
            a.record(&d.stream).unwrap();
            d.big.fill_zero_async(&d.stream).unwrap();
            b.record(&d.stream).unwrap();
            let t = std::time::Instant::now();
            d.pool.close(o, &d.stream).unwrap();
            us += t.elapsed().as_secs_f64() * 1e6;
            {
                let t = std::time::Instant::now();
                let _t = d.pool.stage(d.stages[1], &d.stream).unwrap();
                us += t.elapsed().as_secs_f64() * 1e6;
                d.small.fill_zero_async(&d.stream).unwrap();
            }
            d.rec_us.push(us);
            d.stream.synchronize().unwrap();
            d.pool.note_sync(&d.stream);
            d.twin.insert(step, Event::elapsed_ms(&a, &b).unwrap());
        }
        std::thread::sleep(Duration::from_millis(if n_steps > 200 { 2 } else { 10 }));
    }
    drop((a, b));
    for d in devs.iter_mut() {
        let n = d.rec_us.len();
        let k = (n / 10).max(1);
        let (e50, e99, emax) = pcts(&mut d.rec_us[..k].to_vec());
        let (l50, l99, lmax) = pcts(&mut d.rec_us[n - k..].to_vec());
        println!("{}: stage-call host us per step, first {k} steps p50 {e50:.1} p99 {e99:.1} max {emax:.1}; last {k} p50 {l50:.1} p99 {l99:.1} max {lmax:.1}", d.label);
        assert!(l50 < 4.0 * e50.max(5.0), "{}: stage calls slowed down over the run ({e50:.1} -> {l50:.1} us p50)", d.label);
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
        assert!(steps >= n_steps - STEPS_FLOOR, "{}: {steps} steps came back", d.label);
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
        assert!(recs.len() as u64 >= n_steps - STEPS_FLOOR, "{}: {} step_dev records", d.label, recs.len());
        let mut q: Vec<f64> = cal.iter().filter(|r| r["device"] == dev_id && r["ok"] == 1.0).map(|r| r["q_us"]).collect();
        let mut spin: Vec<f64> = cal.iter().filter(|r| r["device"] == dev_id && r["ok"] == 1.0).map(|r| r["spin_us"]).collect();
        if !q.is_empty() {
            let (q50, q99, qmax) = pcts(&mut q);
            let (s50, s99, smax) = pcts(&mut spin);
            println!("{}: anchor bound q_us p50 {q50:.1} p99 {q99:.1} max {qmax:.1}; record-to-seen us p50 {s50:.1} p99 {s99:.1} max {smax:.1}", d.label);
        }
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
    // (A long run's early records may have left the 16 MB ring.)
    assert!(dev.len() as u64 >= (n_steps.min(1000) - STEPS_FLOOR) * 2 * devs.len() as u64, "{} dev records", dev.len());
    assert!(dev.iter().all(|r| r["t_end"] >= r["t_start"] && r["t_start"] >= r["t_host"] - r["q_us"] * 1e3));
    let _ = std::fs::remove_dir_all(&base);
}
