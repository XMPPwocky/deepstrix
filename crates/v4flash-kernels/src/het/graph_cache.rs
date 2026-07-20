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
use std::sync::{Arc, Mutex};
use v4flash_hip::{sys, GraphExec, Stream};

pub type GraphKey = (&'static str, u32);

#[derive(Default)]
pub struct GraphCache {
    entries: Mutex<HashMap<GraphKey, Arc<GraphExec>>>,
}

impl GraphCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
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
    /// Drop all captured graphs. Needed when the buffers a graph baked in
    /// (e.g. a per-`HetModelState` `kv_cache` pointer, captured inside the
    /// decode `qkv_chain` graph) are replaced by a different allocation —
    /// otherwise a replay writes to the *previous* state's buffers. In
    /// production there is one long-lived state so this is never hit; the
    /// verify correctness harness drives several independent states and must
    /// clear between them.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
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
        let key: GraphKey = (stage, layer);
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
