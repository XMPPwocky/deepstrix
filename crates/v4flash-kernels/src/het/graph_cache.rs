//! Captured HIP-graph cache, keyed by (stage_name, layer).
//!
//! Each per-layer forward stage that's purely device-resident with
//! layer-constant scalar parameters can be captured once and replayed
//! per token via `hipGraphLaunch`. The pattern (begin_capture → kernel
//! launches → end_capture → instantiate → launch) is identical for every
//! such stage; this module wraps it so the forward path becomes:
//!
//! ```ignore
//! self.dgpu_graphs.run("mhc_pre_attn", layer, &de.compute, |s| {
//!     de.rms_nw.launch(s, ...)?;
//!     de.f16.matvec(s, ...)?;
//!     // ...
//!     Ok(())
//! })?;
//! ```
//!
//! Each `HeterogeneousEngine` carries one cache per device. On the first
//! call for a (stage, layer) the closure runs under `hipStreamBeginCapture`
//! to build the graph; subsequent calls just launch the cached
//! `GraphExec`. The cache holds `Arc<GraphExec>` so launches don't have
//! to hold the cache mutex.

use color_eyre::eyre;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use v4flash_hip::{sys, GraphExec, Stream};

pub type GraphKey = (&'static str, u64);

/// `V41_MS_GRAPH_RESERVE_MB` (default 400): the device memory a new graph
/// capture must leave free. Each instantiated `GraphExec` holds device memory
/// for the life of the process, and the arena captures one set per (rows,
/// lane buffer) shape: two speculating streams brought new shapes and, on
/// 2026-10-04, ate the headroom the layer-major prefill window needs (4096
/// rows x 82 KB = 336 MB: "store allocation failed; halving the window"),
/// and a capture whose instantiate runs out of memory would fail its step.
/// Below the reserve a NEW shape runs uncaptured; cached shapes still replay.
fn reserve_bytes() -> usize {
    static R: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_MS_GRAPH_RESERVE_MB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(400) << 20
    });
    *R
}

/// `refresh_room` queries the device every this many calls (a driver call;
/// the calls come once per arena step).
const ROOM_EVERY: u64 = 16;

pub struct GraphCache {
    entries: Mutex<HashMap<GraphKey, Arc<GraphExec>>>,
    /// New captures allowed (free device memory above the reserve at the last
    /// `refresh_room`).
    room: AtomicBool,
    calls: AtomicU64,
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            room: AtomicBool::new(true),
            calls: AtomicU64::new(0),
        }
    }

    /// Whether a NEW graph may be captured (`refresh_room`).
    pub fn has_room(&self) -> bool {
        self.room.load(Ordering::Relaxed)
    }

    /// Re-read the CURRENT device's free memory (every `ROOM_EVERY` calls; the
    /// caller makes this cache's device current and has no capture open) and
    /// allow new captures only above `V41_MS_GRAPH_RESERVE_MB`. Logs every
    /// change with the free memory and the number of cached graphs. A failed
    /// query keeps the last answer.
    pub fn refresh_room(&self) {
        if self.calls.fetch_add(1, Ordering::Relaxed) % ROOM_EVERY != 0 {
            return;
        }
        let Ok((free, total)) = v4flash_hip::Device::mem_info_current() else { return };
        let room = free > reserve_bytes();
        if self.room.swap(room, Ordering::Relaxed) != room {
            tracing::warn!(
                room,
                free_mb = free >> 20,
                total_mb = total >> 20,
                reserve_mb = reserve_bytes() >> 20,
                graphs = self.len(),
                "graph cache: new captures {} (device memory)",
                if room { "resumed" } else { "PAUSED: new shapes run uncaptured" }
            );
        }
    }

    /// Run the captured graph for `(stage, layer)` on `stream`. If the
    /// graph hasn't been captured yet, `capture` is called with the
    /// stream in capture mode to record its kernel launches; the
    /// resulting graph is instantiated, stored, and launched.
    ///
    /// Lock discipline: the cache mutex is held across capture +
    /// instantiate on the first call, but released before the launch.
    /// On steady-state calls the mutex is held only long enough to
    /// `Arc::clone` the `GraphExec` out.
    /// The instantiated graph for `(stage, key)`, if captured already.
    pub fn get(&self, stage: &'static str, key: u64) -> Option<Arc<GraphExec>> {
        self.entries.lock().unwrap().get(&(stage, key)).cloned()
    }

    /// Store a graph captured by the caller.
    pub fn insert(&self, stage: &'static str, key: u64, exec: Arc<GraphExec>) {
        self.entries.lock().unwrap().insert((stage, key), exec);
    }

    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    pub fn run<F>(
        &self,
        stage: &'static str,
        layer: u32,
        stream: &Stream,
        capture: F,
    ) -> eyre::Result<()>
    where
        F: FnOnce(&Stream) -> eyre::Result<()>,
    {
        let key: GraphKey = (stage, layer as u64);
        let exec = {
            let mut entries = self.entries.lock().unwrap();
            if let Some(e) = entries.get(&key) {
                e.clone()
            } else {
                stream.begin_capture(sys::HIP_STREAM_CAPTURE_MODE_THREAD_LOCAL)?;
                capture(stream)?;
                let graph = stream.end_capture()?;
                let exec = Arc::new(graph.instantiate()?);
                entries.insert(key, exec.clone());
                exec
            }
        };
        exec.launch(stream)
    }
}
