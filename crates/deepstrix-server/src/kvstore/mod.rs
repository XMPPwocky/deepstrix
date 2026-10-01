//! KV prefix store: chunked, content-addressed
//! (docs/v41/KV_PREFIX_STORE_DESIGN.md, rev 6).
//!
//! The positional KV of a prompt is stored in chunks of C = 1024 positions,
//! each one file keyed by a blake3 chain over the token ids and written once,
//! shared by every prompt with that prefix. The state that is not positional
//! (windows, accumulators, the open rows, the DSpark ring) lives in tails at
//! prompt ends, every K = 8192 positions of a prefill (waypoints), at the
//! system-prefix anchor and on cancel. A lookup walks the chain over the
//! request and restores from the deepest usable tail (6).
//!
//! **M1 (design 13): the store without GPU and without the request path.**
//! Nothing in the server reaches this module yet: `V41_KV_STORE` defaults to
//! `off` and only M2 will read it. M2 adds the device-side capture/restore
//! (`het/kv_capture.rs`), the writes w1-w4, snapping and the shadow mode.
//!
//! Layout:
//! - [`keys`]: the namespace and the chunk / tail key chain (4.3, 4.4);
//! - [`format`]: chunk and tail files, two tail sections, the `anchor` flag;
//! - [`index`]: trie, tails, refcounts, pins, demotion, thinning, eviction,
//!   walk and selection, the invariant checker;
//! - [`io`]: the one IO thread (writes, mutations, unlinks, trash);
//! - [`scan`]: startup (namespace GC, header scan, crash repair, rebuild);
//! - [`knobs`]: the knob classification and the knob hash;
//! - [`Store`] (here): the scheduler-thread facade over all of them.

pub mod format;
pub mod index;
pub mod keys;

/// Positions per chunk (4.2). Equals the production prefill chunk
/// (`V41_MS_CHUNK_ROWS` = `B_MAX`), so cold prefills already end their chunks
/// at multiples of C. Part of the namespace.
pub const C: u32 = 1024;
/// Waypoint spacing (5.2): an encoder tail every K positions of a prefill.
pub const K: u32 = 8192;
/// Bumped only on an explicit decision (4.4.1): a fix that corrected what
/// stored KV contains, or a numerics change measured beyond the M0 null.
/// Bumping it starts a new namespace = cold-starts every conversation.
pub const KV_EPOCH: u32 = 1;
/// Bumped in the same commit as any change that alters prefill bits (4.4).
/// Recorded in headers only, never in the namespace: a bump cold-starts
/// nothing. A diff touching the prefill crates without bumping it says "not
/// bit-changing" in its commit message.
pub const KV_NUMERICS_GEN: u32 = 1;
/// Full tails kept full on one path (8.2): two cover "regenerate the last turn".
pub const DEMOTE_KEEP_FULL: usize = 2;
/// An encoder tail serves only a suffix longer than this (5.3): exactly when
/// `prefill_job_finish` discards the decoder windows anyway. = SWA_WINDOW.
pub const ENC_MIN_SUFFIX: u32 = 128;
/// No restore below this many positions (6.2).
pub const MIN_RESTORE_T: u32 = 64;
/// Eviction score weight of hits (8.3): score = path_last_used + 6 h × log2(1 + hits).
pub const SCORE_HIT_S: u64 = 6 * 3600;
/// How often the invariant checker runs in debug builds and in shadow (9.5).
pub const INVARIANT_CHECK_EVERY_S: u64 = 3600;

const _: () = assert!(K % C == 0);
const _: () = assert!(ENC_MIN_SUFFIX == v4flash_kernels::config::SWA_WINDOW);
