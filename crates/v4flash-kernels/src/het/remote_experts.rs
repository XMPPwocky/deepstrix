//! Expert-parallel remote executor (PLAN.md §4 / §7b): box 2 holds a share of
//! the routed experts resident in its iGPU pool and computes weighted MoE
//! partial sums for the hub over a persistent TCP link.
//!
//! Three pieces live here so the daemon and the hub share one definition of
//! every byte on the wire and every kernel launch:
//!
//! * [`proto`] — length-prefixed, versioned frames (HELLO / REQUEST / RESPONSE /
//!   ERROR). The activation format on the wire is the MoE kernels' NATIVE input,
//!   **Q8_K** (`XQ_BYTES_PER_TOKEN` = 20 super-blocks × 292 B = 5840 B/token):
//!   the hub already quantises `ffn_input_norm` to Q8_K for its own experts, so
//!   sending those bytes makes the remote partial bit-identical to a local
//!   computation by construction (no second rounding), and it is smaller than
//!   f16 (10240 B). The reply is the weighted f16 partial sum `[B × N_EMBD]`
//!   (`REQ_FLAG_RESP_F32` asks for f32 instead; used by the bit-identity tests
//!   and cheap enough at decode sizes). Rows with no remote pick come back as
//!   zeros — the response is always dense, the CLIENT skips the request when no
//!   token of the batch has a remote pick.
//! * [`ExpertShard`] + [`MoeExecutor`] — the daemon side: a packed iGPU pool
//!   (`RoutedExpertWeights`, one contiguous slot range per assigned layer)
//!   filled from the HF checkpoint with the same `read_expert_into` the pager
//!   and `load_experts_packed` use, and the iGPU MoE launched through the
//!   existing het-split kernels (`moe_gate_up_batch_hetsplit` /
//!   `moe_down_batched_hetsplit` at decode sizes, the by-expert kwide chain at
//!   prefill sizes). Picks the remote does NOT own are padded with
//!   [`SENTINEL_EXPERT`] whose remap entry says "the other device takes it", so
//!   every launch has a fixed shape and no zero-fill is needed.
//! * [`RemoteExpertClient`] — the hub side: connect, learn the daemon's
//!   ownership table (HELLO), `submit` a layer's activations + picks from a
//!   writer thread, `wait` for the partial from a reader thread. Requests for
//!   consecutive layers may be in flight at once (responses are FIFO).
//!
//! Why not `ExpertPager` for the pool: its dense windows pin only layers
//! `0..k`, `ensure` pages one expert at a time on one thread, and it keeps a
//! single remap that is refilled per call; the daemon needs an arbitrary
//! `(layer, expert-range)` set loaded once with parallel readers and one
//! pointer-stable remap per layer. The byte layout, geometry checks and the
//! kernel contract are the pager's.
//!
//! Design/measurements: docs/v41/REMOTE_EXPERTS.md.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, ToSocketAddrs};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use color_eyre::eyre::{self, eyre, WrapErr};
use v4flash_core::{gguf::GgufType, V41HfWeights, WeightSrc};
use v4flash_hip::{Device, DeviceBuffer, PinnedBuffer, Stream, HIP_HOST_MALLOC_NON_COHERENT};

use crate::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, N_LAYER,
    SWIGLU_CLAMP_EXP,
};
use crate::model_weights::RoutedExpertWeights;
use crate::mxfp4_repack::Mxfp4Repack;
use crate::q8_k::BLOCK_Q8_K_BYTES;
use crate::weight_contract;
use crate::weights::DeviceWeight;

use super::engine::DeviceEngine;

/// Bytes of one token's Q8_K activation row (the MoE gate/up input).
pub const XQ_BYTES_PER_TOKEN: usize = BLOCKS_Q8K_GATE_IN as usize * BLOCK_Q8_K_BYTES;
/// Bytes of one slot's Q8_K mid row (the down input).
pub const MIDQ_BYTES_PER_SLOT: usize = BLOCKS_Q8K_DOWN_IN as usize * BLOCK_Q8_K_BYTES;
/// Pick id meaning "this slot is not for this executor". Its remap entry is a
/// non-negative dGPU-style slot, so mode-0 het-split launches skip it exactly
/// like a dGPU-resident expert. The remap therefore has `N_EXPERT + 1` entries.
pub const SENTINEL_EXPERT: i32 = N_EXPERT as i32;
pub const REMAP_LEN: usize = N_EXPERT as usize + 1;
/// Pick id on the WIRE meaning "no pick in this slot" (the client masks
/// non-owned picks to this; the daemon turns it into [`SENTINEL_EXPERT`]).
pub const NO_PICK: i32 = -1;
/// Prefill kwide chunk (members per work item), the production value.
pub const CHUNK_SIZE: u32 = 32;
pub const DEFAULT_PORT: u16 = 7431;

// ---------------------------------------------------------------------------
// Socket options (Linux x86_64 constants; declared here rather than pulling in
// `libc` — repo rule: deps need sign-off, and this is four setsockopt calls).
// ---------------------------------------------------------------------------
extern "C" {
    fn setsockopt(fd: i32, level: i32, name: i32, val: *const std::ffi::c_void, len: u32) -> i32;
    fn clock_gettime(clk: i32, tp: *mut Timespec) -> i32;
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
struct Timespec {
    tv_sec: i64,
    tv_nsec: i64,
}

const CLOCK_REALTIME: i32 = 0;
/// The clock the wire timestamps use: unadjusted by NTP/adjtime, so a slew
/// during a request cannot corrupt an offset sample.
const CLOCK_MONOTONIC_RAW: i32 = 4;

fn clock_ns(clk: i32) -> u64 {
    let mut ts = Timespec::default();
    // SAFETY: `ts` is a valid, correctly-sized Timespec; clock ids are constants.
    if unsafe { clock_gettime(clk, &mut ts) } != 0 {
        return 0;
    }
    (ts.tv_sec as u64) * 1_000_000_000 + (ts.tv_nsec as u64)
}

/// CLOCK_MONOTONIC_RAW ns — the clock stamped into every request/response
/// (`proto` t1..t4). Raw rather than CLOCK_MONOTONIC so NTP slew cannot move it
/// under a measurement.
pub fn monotonic_raw_ns() -> u64 {
    clock_ns(CLOCK_MONOTONIC_RAW)
}

/// CLOCK_REALTIME ns — what perfetto timestamps use, so a trace from each box
/// lands on a common (if only NTP-accurate) timeline; `ClockSync` supplies the
/// exact correction.
pub fn realtime_ns() -> u64 {
    clock_ns(CLOCK_REALTIME)
}

/// A back-to-back (monotonic_raw, realtime) pair, so a timestamp on one clock
/// can be mapped to the other on the SAME box. Each side samples one at startup
/// and the daemon ships its pair in HELLO; combined with the measured offset
/// this converts box 2's perfetto REALTIME stamps onto box 1's timeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClockPair {
    pub mono_raw_ns: u64,
    pub realtime_ns: u64,
}

impl ClockPair {
    pub fn sample() -> Self {
        // Sandwich realtime between two monotonic reads and take the midpoint,
        // so the pair is correlated to within one clock read (~20-40 ns).
        let m0 = monotonic_raw_ns();
        let rt = realtime_ns();
        let m1 = monotonic_raw_ns();
        Self { mono_raw_ns: m0 + (m1.saturating_sub(m0)) / 2, realtime_ns: rt }
    }
    /// REALTIME ns corresponding to a monotonic_raw ns on the same box.
    pub fn realtime_for(&self, mono_ns: u64) -> i128 {
        self.realtime_ns as i128 + (mono_ns as i128 - self.mono_raw_ns as i128)
    }
}
const SOL_SOCKET: i32 = 1;
const SO_SNDBUF: i32 = 7;
const SO_RCVBUF: i32 = 8;
const SO_BUSY_POLL: i32 = 46;
const IPPROTO_TCP: i32 = 6;
const TCP_QUICKACK: i32 = 12;

/// Transport recipe measured in docs/v41/SECOND_BOX.md: persistent TCP,
/// TCP_NODELAY, explicit 4 MB socket buffers (a 16 KB initial send buffer
/// splits a 32 KB write and waits for the ACK), SO_BUSY_POLL on the receiving
/// socket. TCP_QUICKACK is OFF by default: re-arming it after every receive
/// (Linux drops it after a few exchanges, so "set once at connect" is
/// effectively off) measured +260 µs per round trip on ≥ 16 KB replies
/// (docs/v41/REMOTE_EXPERTS.md §5.2); `quickack: true` re-arms it per frame.
#[derive(Clone, Debug)]
pub struct SocketOptions {
    pub sndbuf: usize,
    pub rcvbuf: usize,
    pub busy_poll_us: u32,
    pub quickack: bool,
}

impl Default for SocketOptions {
    fn default() -> Self {
        Self { sndbuf: 4 << 20, rcvbuf: 4 << 20, busy_poll_us: 500, quickack: false }
    }
}

fn set_opt_i32(s: &TcpStream, level: i32, name: i32, v: i32) -> bool {
    use std::os::unix::io::AsRawFd;
    let r = unsafe {
        setsockopt(s.as_raw_fd(), level, name, &v as *const i32 as *const _, 4)
    };
    r == 0
}

/// Box 2's reader window, adapted per request (the daemon twin of the hub's
/// `HetEngine::remote_set_phase_busy_poll`). Single-lane decode sends one
/// request per layer, ~3.5-4 ms apart at 1-3 rows, far past the base 500 us
/// window, so the reader sleeps and every request pays an idle wake-up (the
/// ~60 us link measured with both ends hot was ~170-280 us live, 2026-09-25).
/// While the hub says it is decoding (`REQ_FLAG_DECODE`) AND the request frame
/// and its reply each fit one ~64 KB segment (<= 3 rows of f32 replies, <= 11
/// rows of requests), the reader spins `V41_B2_DECODE_BUSY_POLL_US` (default
/// 5000, the sysctl cap; 0 = never adapt). Otherwise it uses `base_us`: a
/// multi-segment message is held on a spinning receiver for about the window,
/// and a spinning daemon reader slowed its own large sends (1.3 MB: 1.9 ->
/// 6.6 ms, LINK_IDLE_LATENCY.md). Calls `setsockopt` only when the value
/// changes; the new window applies from the reader's next `recv`.
pub fn b2_adapt_busy_poll(s: &TcpStream, decode_phase: bool, request_bytes: usize, reply_bytes: usize, base_us: u32) {
    static DECODE_US: std::sync::LazyLock<u32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_DECODE_BUSY_POLL_US").ok().and_then(|v| v.parse().ok()).unwrap_or(5000)
    });
    if *DECODE_US == 0 || base_us == 0 {
        return;
    }
    // One TCP segment on the thunderbolt0 64 KB MTU (MSS ~65,468), with room
    // for the frame header.
    const ONE_SEGMENT: usize = 65_000;
    let small = request_bytes <= ONE_SEGMENT && reply_bytes <= ONE_SEGMENT;
    let want = if decode_phase && small { *DECODE_US } else { base_us };
    if B2_SPIN_CUR.load(std::sync::atomic::Ordering::Relaxed) == want {
        return;
    }
    if set_opt_i32(s, SOL_SOCKET, SO_BUSY_POLL, want as i32) {
        B2_SPIN_CUR.store(want, std::sync::atomic::Ordering::Relaxed);
    } else {
        static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            eprintln!("expertd: SO_BUSY_POLL={want} refused (raise net.core.busy_read, scripts/link_latency_step.sh 4s); reader window not adapted");
        }
    }
}

/// The window `b2_adapt_busy_poll` last set on the connection being served.
/// `serve_connection` resets it for each new connection (whose socket starts
/// at the base window).
static B2_SPIN_CUR: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(u32::MAX);

pub fn apply_socket_options(s: &TcpStream, o: &SocketOptions) -> eyre::Result<()> {
    s.set_nodelay(true)?;
    if o.sndbuf > 0 && !set_opt_i32(s, SOL_SOCKET, SO_SNDBUF, o.sndbuf as i32) {
        eprintln!("remote_experts: SO_SNDBUF={} refused (net.core.wmem_max?)", o.sndbuf);
    }
    if o.rcvbuf > 0 && !set_opt_i32(s, SOL_SOCKET, SO_RCVBUF, o.rcvbuf as i32) {
        eprintln!("remote_experts: SO_RCVBUF={} refused (net.core.rmem_max?)", o.rcvbuf);
    }
    if o.busy_poll_us > 0 && !set_opt_i32(s, SOL_SOCKET, SO_BUSY_POLL, o.busy_poll_us as i32) {
        // Unprivileged processes may only lower it below net.core.busy_read;
        // the sysctl default (500 on both boxes) still applies.
        eprintln!("remote_experts: SO_BUSY_POLL={} refused (needs CAP_NET_ADMIN above busy_read)", o.busy_poll_us);
    }
    quickack(s, o);
    Ok(())
}

/// TCP_QUICKACK is not sticky on Linux; re-arm after each receive.
#[inline]
pub fn quickack(s: &TcpStream, o: &SocketOptions) {
    if o.quickack {
        let _ = set_opt_i32(s, IPPROTO_TCP, TCP_QUICKACK, 1);
    }
}

// ---------------------------------------------------------------------------
// Aligned byte buffer: frames are read straight into this so the f16/f32
// payload can be viewed in place (8-byte aligned).
// ---------------------------------------------------------------------------
pub struct AlignedBuf {
    words: Vec<u64>,
    len: usize,
}

impl AlignedBuf {
    pub fn with_capacity(bytes: usize) -> Self {
        Self { words: Vec::with_capacity(bytes.div_ceil(8)), len: 0 }
    }
    pub fn clear(&mut self) {
        self.len = 0;
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Set the length (contents beyond the old length are unspecified bytes).
    pub fn resize(&mut self, bytes: usize) {
        self.words.resize(bytes.div_ceil(8), 0);
        self.len = bytes;
    }
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `words` holds >= len bytes of initialised u64s.
        unsafe { std::slice::from_raw_parts(self.words.as_ptr() as *const u8, self.len) }
    }
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as above; exclusive borrow.
        unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.len) }
    }
    pub fn extend_from_slice(&mut self, s: &[u8]) {
        let old = self.len;
        self.resize(old + s.len());
        self.as_bytes_mut()[old..].copy_from_slice(s);
    }
    pub fn put_u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u16(&mut self, v: u16) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    pub fn put_u64(&mut self, v: u64) {
        self.extend_from_slice(&v.to_le_bytes());
    }
    /// Typed view of `[off, off + n*size_of::<T>())`; `off` must be aligned for `T`.
    pub fn view<T: Copy>(&self, off: usize, n: usize) -> &[T] {
        let bytes = n * std::mem::size_of::<T>();
        assert!(off + bytes <= self.len, "AlignedBuf::view out of range");
        assert!(off % std::mem::align_of::<T>() == 0, "AlignedBuf::view misaligned");
        // SAFETY: bounds + alignment checked; backing store is u64-aligned and
        // T is a plain-data numeric type at every call site.
        unsafe { std::slice::from_raw_parts(self.as_bytes()[off..].as_ptr() as *const T, n) }
    }
    pub fn view_mut<T: Copy>(&mut self, off: usize, n: usize) -> &mut [T] {
        let bytes = n * std::mem::size_of::<T>();
        assert!(off + bytes <= self.len, "AlignedBuf::view_mut out of range");
        assert!(off % std::mem::align_of::<T>() == 0, "AlignedBuf::view_mut misaligned");
        // SAFETY: as `view`, exclusive borrow.
        unsafe { std::slice::from_raw_parts_mut(self.as_bytes_mut()[off..].as_mut_ptr() as *mut T, n) }
    }
}

// ---------------------------------------------------------------------------
// Wire protocol
// ---------------------------------------------------------------------------
pub mod proto {
    use super::AlignedBuf;
    use color_eyre::eyre::{self, eyre};
    use std::io::Read;

    pub const MAGIC: u32 = 0x5058_5344; // "DSXP"
    /// Bumped to 2 for the two trailing response u32s (`t_page_us`, `n_miss`)
    /// that drive the `remote.pager` perfetto lane. Appended AFTER t1/t2/t3 so
    /// every existing offset is unchanged; the frame LENGTH changes, so both
    /// boxes must be rebuilt together (a v1 daemon fails the length check).
    ///
    /// Bumped to 3 for MXFP4 super-block layout v2. The frame SHAPE is unchanged
    /// here -- what changed is the meaning of the expert bytes each box decodes.
    /// Both boxes repack from raw HF bytes independently, so a rolling deploy
    /// across the bump would have one side reading v1 bytes as v2: no error, no
    /// length mismatch, just silently wrong experts. Tying the layout to the
    /// protocol version makes that combination refuse to connect instead.
    /// Keep this in step with `mxfp4_tables::MXFP4_LAYOUT_VERSION`.
    /// Bumped to 4 for the optional PREFETCH word list after the hints
    /// (`REQ_FLAG_PREFETCH`): the frame length changes, both boxes rebuild.
    pub const VERSION: u16 = 4;
    /// magic u32 | version u16 | kind u16 | seq u32 | payload_len u32
    pub const HDR_LEN: usize = 16;
    pub const KIND_HELLO: u16 = 1;
    pub const KIND_REQUEST: u16 = 2;
    pub const KIND_RESPONSE: u16 = 3;
    pub const KIND_ERROR: u16 = 4;

    /// Request flag: return f32 partials instead of f16.
    pub const REQ_FLAG_RESP_F32: u32 = 1;
    /// RESPONSE-side field packed into the echoed `flags` word: a bitmask over
    /// the request's first 16 `sel` slots, bit i set = pick i MISSED on box 2 and
    /// was paged from its disk.
    ///
    /// Why here and not a new field: the response's 8 u32s are all used and the
    /// clock triple after them must stay 8-aligned, so adding one u32 would
    /// misalign it. `flags` is the request's flags echoed back, and requests only
    /// ever set bits 0-1, so the high half is free.
    ///
    /// Emitted for b==1 (decode) only. A prefill batch's sel is up to 1024x6 and
    /// nothing consumes a miss mask for it.
    pub const RESP_MISS_SHIFT: u32 = 16;
    pub const RESP_MISS_BITS: usize = 16;
    pub const RESP_MISS_MASK: u32 = 0xFFFF << RESP_MISS_SHIFT;

    /// Request flag: take the by-expert (prefill) chain even when
    /// `b <= decode_max_b` — lets the hub choose per request (DSpark verify
    /// batches) and gives the A/B without a daemon restart.
    pub const REQ_FLAG_BATCHED: u32 = 2;
    /// Trailing residency hints after `ew`: u32 n_admit, u32 n_evict, then that
    /// many `layer << 16 | expert` words each. ADMITTED = box 1 now holds it
    /// (box 2 marks its copy evict-first, keeping the tiers exclusive);
    /// EVICTED = box 1 dropped it (v2: box 2 re-pages it in the background).
    pub const REQ_FLAG_HINTS: u32 = 4;
    /// Request flag: after the hints block (present or not), `n u32` then n
    /// words `(layer << 16) | expert` the daemon should PREFETCH into its pool
    /// in the background (look-ahead routing: the hub's prediction of the
    /// NEXT layer's picks that this box owns; ~75% precision on decoder layers,
    /// MEASURED 2026-09-22). Wrong words cost bandwidth only.
    pub const REQ_FLAG_PREFETCH: u32 = 8;
    /// The sender is about to submit ANOTHER request for this SAME layer (the
    /// other hub lane; `forward_step_arena_pipelined` routes the two lanes back
    /// to back). The daemon may therefore WAIT a bounded moment for it and run
    /// both as one MoE pass instead of streaming the layer's experts twice
    /// (`b2_merge`). Without the bit the daemon has to guess, and measured
    /// 2026-09-22 it guessed wrong ~97% of the time: it is usually idle, so it
    /// dequeues lane A before lane B's frame has even arrived, and once a pass
    /// has started merging is impossible. Purely advisory: a daemon that
    /// ignores it is correct, just slower.
    pub const REQ_FLAG_PARTNER: u32 = 16;
    /// Capability: the sender matches replies to tickets by `seq`, so the daemon
    /// may answer this request OUT OF ORDER. Box 2 uses it to PARK a request
    /// that has to page experts in (knob `park`): the resident pass runs, the
    /// misses read in the background, and a request queued behind it is served
    /// and answered in the meantime instead of waiting out the NVMe read.
    pub const REQ_FLAG_OOO: u32 = 32;
    /// Request flag: append box 2's residency map for the request's layer
    /// (`RESID_WORDS` u32s, bit e = expert e resident and landed, AFTER this
    /// request's own paging) behind the partial, for the hub's mirror
    /// (`het::b2_mirror`, box-2 miss substitution). A daemon that appended it
    /// sets `RESP_FLAG_RESID` in the reply. An older daemon echoes the request
    /// flag but not that bit, and appends nothing, so either side can be
    /// deployed first.
    pub const REQ_FLAG_RESID: u32 = 64;
    /// Request flag: the hub is in its DECODE phase (small, frequent
    /// requests, `HetEngine::remote_set_phase_busy_poll(true)`). The daemon
    /// spins its reader long between such requests (`b2_adapt_busy_poll`) and
    /// uses its base window otherwise. An older hub never sets it, so the
    /// daemon keeps the base window.
    pub const REQ_FLAG_DECODE: u32 = 128;
    /// RESPONSE flag: the residency map is appended (see `REQ_FLAG_RESID`).
    /// Bit 15: requests use the low bits and the miss mask the high 16.
    pub const RESP_FLAG_RESID: u32 = 1 << 15;
    pub const RESID_WORDS: usize = (super::N_EXPERT as usize).div_ceil(32);
    /// Request flag (2026-09-26): PIN MODE (`b2_mirror` pinning, hub knob
    /// `V41_B2_PIN`). Box 2 turns pinning on for the connection: an expert
    /// it reports held is never evicted until the hub RELEASES it
    /// (`REQ_FLAG_RELEASE`), so the hub's mirror is exact where it says "held"
    /// (`ExpertShard::pin_report`). The reply's residency map is then the
    /// PINNED set of the layer (never merely resident) and a pin block follows
    /// it (`RESP_FLAG_PIN`). An older daemon ignores the bit and echoes it but
    /// never sets `RESP_FLAG_PIN`, which is how the hub learns to fall back.
    pub const REQ_FLAG_PIN: u32 = 256;
    /// Request flag: after the prefetch block (present or not), `n u32` then n
    /// RELEASE words `(layer << 16) | expert` box 2 unpins, in order, before
    /// serving the request. Sent only once the peer has answered with
    /// `RESP_FLAG_PIN`: an older daemon would fail the frame-length check.
    pub const REQ_FLAG_RELEASE: u32 = 512;
    /// RESPONSE flag: a pin block (`PIN_WORDS` u32s) follows the residency map
    /// (so it is only ever set together with `RESP_FLAG_RESID`):
    /// `epoch` = release words box 2 had applied on this connection when it
    /// built the map, `pinned` / `budget` = its pinned count and cap, a
    /// reserved word, then `RESID_WORDS` words of PAGED bits: bit e = this
    /// request's pass had to page or wait for expert e (not landed when the
    /// pass started). The hub's surprise check is `held at submit & paged`.
    pub const RESP_FLAG_PIN: u32 = 1 << 14;
    pub const PIN_HDR_WORDS: usize = 4;
    pub const PIN_WORDS: usize = PIN_HDR_WORDS + RESID_WORDS;
    /// A REQUEST of at most this many rows is DECODE-SHAPED for pinning: box 2
    /// makes its picks pin-eligible at the reply, the hub counts them for
    /// release ranking. Per request, never per merged pass (two mergeable
    /// 9-32-row lanes are still decode).
    pub const PIN_DECODE_MAX_ROWS: u32 = 16;

    /// Fixed request fields after the header (bytes):
    /// layer, b, flags, n_used, xq_bpt, reserved (6 × u32) then `t1` (u64,
    /// 8-aligned at offset 40). 16 + 32 = 48, so the Q8_K payload stays 8-aligned.
    pub const REQ_FIXED: usize = 32;
    /// Fixed response fields after the header (bytes): 8 × u32 then the clock
    /// triple `t1_echo, t2, t3` (u64 each, 8-aligned at 48/56/64).
    pub const RESP_FIXED: usize = 64;
    /// Byte offset of `t1` inside a REQUEST frame.
    pub const REQ_T1_OFF: usize = HDR_LEN + 24;
    /// Byte offsets of `t1_echo`/`t2`/`t3` inside a RESPONSE frame. The writer
    /// thread patches `t3` in place immediately before `write()`, which is the
    /// only way to stamp "just before the send" when compute and I/O are on
    /// different threads.
    pub const RESP_T1_OFF: usize = HDR_LEN + 32;
    pub const RESP_T2_OFF: usize = HDR_LEN + 40;
    pub const RESP_T3_OFF: usize = HDR_LEN + 48;
    /// Microseconds this request spent PAGING experts in from box 2's own NVMe,
    /// and how many experts missed. Patched by the daemon just before write, like
    /// `t_server_us`. Zero when nothing was paged.
    pub const RESP_PAGE_OFF: usize = HDR_LEN + 56;
    pub const RESP_MISSN_OFF: usize = HDR_LEN + 60;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Header {
        pub kind: u16,
        pub seq: u32,
        pub len: u32,
    }

    pub fn put_header(buf: &mut AlignedBuf, kind: u16, seq: u32, payload_len: u32) {
        buf.put_u32(MAGIC);
        buf.put_u16(VERSION);
        buf.put_u16(kind);
        buf.put_u32(seq);
        buf.put_u32(payload_len);
    }

    /// Patch the payload length once the payload has been appended.
    pub fn patch_len(buf: &mut AlignedBuf) {
        let n = (buf.len() - HDR_LEN) as u32;
        buf.as_bytes_mut()[12..16].copy_from_slice(&n.to_le_bytes());
    }

    /// Append the residency map to a RESPONSE whose payload is complete, and
    /// set `RESP_FLAG_RESID`. Call before `patch_len`.
    pub fn append_residency(buf: &mut AlignedBuf, words: &[u32; RESID_WORDS]) {
        const FLAGS_OFF: usize = HDR_LEN + 8;
        let mut f = [0u8; 4];
        f.copy_from_slice(&buf.as_bytes()[FLAGS_OFF..FLAGS_OFF + 4]);
        let flags = u32::from_le_bytes(f) | RESP_FLAG_RESID;
        buf.as_bytes_mut()[FLAGS_OFF..FLAGS_OFF + 4].copy_from_slice(&flags.to_le_bytes());
        for &w in words {
            buf.put_u32(w);
        }
    }

    /// The residency map appended to a RESPONSE (`RESP_FLAG_RESID`), if any.
    pub fn response_residency<'a>(buf: &'a AlignedBuf, m: &ResponseMeta) -> Option<&'a [u32]> {
        if m.flags & RESP_FLAG_RESID == 0 {
            return None;
        }
        let off = RESP_DATA_OFF + (m.b as usize) * (m.n_embd as usize) * (m.elem_bytes as usize);
        Some(buf.view::<u32>(off, RESID_WORDS))
    }

    /// The pin block of a RESPONSE (`RESP_FLAG_PIN`), parsed.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PinReply {
        pub epoch: u32,
        pub pinned: u32,
        pub budget: u32,
        pub paged: [u32; RESID_WORDS],
    }

    /// Append the pin block after `append_residency` and set `RESP_FLAG_PIN`.
    /// Call before `patch_len`.
    pub fn append_pin(buf: &mut AlignedBuf, epoch: u32, pinned: u32, budget: u32, paged: &[u32; RESID_WORDS]) {
        const FLAGS_OFF: usize = HDR_LEN + 8;
        let mut f = [0u8; 4];
        f.copy_from_slice(&buf.as_bytes()[FLAGS_OFF..FLAGS_OFF + 4]);
        debug_assert!(u32::from_le_bytes(f) & RESP_FLAG_RESID != 0, "pin block without the residency map");
        let flags = u32::from_le_bytes(f) | RESP_FLAG_PIN;
        buf.as_bytes_mut()[FLAGS_OFF..FLAGS_OFF + 4].copy_from_slice(&flags.to_le_bytes());
        for w in [epoch, pinned, budget, 0] {
            buf.put_u32(w);
        }
        for &w in paged {
            buf.put_u32(w);
        }
    }

    /// The pin block appended to a RESPONSE (`RESP_FLAG_PIN`), if any.
    pub fn response_pin(buf: &AlignedBuf, m: &ResponseMeta) -> Option<PinReply> {
        if m.flags & RESP_FLAG_PIN == 0 || m.flags & RESP_FLAG_RESID == 0 {
            return None;
        }
        let off = RESP_DATA_OFF + (m.b as usize) * (m.n_embd as usize) * (m.elem_bytes as usize) + RESID_WORDS * 4;
        let w = buf.view::<u32>(off, PIN_WORDS);
        let mut paged = [0u32; RESID_WORDS];
        paged.copy_from_slice(&w[PIN_HDR_WORDS..]);
        Some(PinReply { epoch: w[0], pinned: w[1], budget: w[2], paged })
    }

    pub fn parse_header(h: &[u8]) -> eyre::Result<Header> {
        if h.len() < HDR_LEN {
            return Err(eyre!("short header"));
        }
        let u32_at = |o: usize| u32::from_le_bytes([h[o], h[o + 1], h[o + 2], h[o + 3]]);
        let u16_at = |o: usize| u16::from_le_bytes([h[o], h[o + 1]]);
        if u32_at(0) != MAGIC {
            return Err(eyre!("bad magic {:#x}", u32_at(0)));
        }
        let ver = u16_at(4);
        if ver != VERSION {
            return Err(eyre!("protocol version {ver} != {VERSION}"));
        }
        Ok(Header { kind: u16_at(6), seq: u32_at(8), len: u32_at(12) })
    }

    /// Read one whole frame (header + payload) into `buf`. Returns the header.
    /// `max_payload` bounds a corrupt length field.
    pub fn read_frame(r: &mut impl Read, buf: &mut AlignedBuf, max_payload: usize) -> eyre::Result<Header> {
        buf.resize(HDR_LEN);
        r.read_exact(buf.as_bytes_mut())?;
        let h = parse_header(buf.as_bytes())?;
        if h.len as usize > max_payload {
            return Err(eyre!("frame payload {} exceeds {max_payload}", h.len));
        }
        buf.resize(HDR_LEN + h.len as usize);
        r.read_exact(&mut buf.as_bytes_mut()[HDR_LEN..])?;
        Ok(h)
    }

    /// Ownership + geometry the daemon announces on connect.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ShardInfo {
        pub n_layer: u32,
        pub n_expert: u32,
        pub n_used: u32,
        pub n_embd: u32,
        pub xq_bytes_per_token: u32,
        pub max_batch: u32,
        pub decode_max_b: u32,
        pub n_resident: u32,
        pub bytes_per_expert: u32,
        /// The daemon's (CLOCK_MONOTONIC_RAW, CLOCK_REALTIME) correlation pair,
        /// sampled at HELLO. With the measured offset this maps the daemon's
        /// wire timestamps — and its perfetto trace — onto the hub's timeline.
        pub clock: super::ClockPair,
        /// `n_layer` bitsets of `ceil(n_expert/32)` words each.
        pub owned: Vec<Vec<u32>>,
    }

    impl ShardInfo {
        pub fn words_per_layer(&self) -> usize {
            (self.n_expert as usize).div_ceil(32)
        }
        pub fn owns(&self, layer: u32, e: u32) -> bool {
            match self.owned.get(layer as usize) {
                Some(bits) => bits.get((e / 32) as usize).is_some_and(|w| (w >> (e % 32)) & 1 == 1),
                None => false,
            }
        }
        pub fn owned_ids(&self, layer: u32) -> Vec<u32> {
            (0..self.n_expert).filter(|&e| self.owns(layer, e)).collect()
        }
        pub fn owned_count(&self, layer: u32) -> usize {
            self.owned.get(layer as usize).map_or(0, |b| b.iter().map(|w| w.count_ones() as usize).sum())
        }
    }

    pub fn encode_hello(buf: &mut AlignedBuf, info: &ShardInfo) {
        buf.clear();
        put_header(buf, KIND_HELLO, 0, 0);
        for v in [
            info.n_layer, info.n_expert, info.n_used, info.n_embd, info.xq_bytes_per_token,
            info.max_batch, info.decode_max_b, info.n_resident, info.bytes_per_expert,
        ] {
            buf.put_u32(v);
        }
        // Clock pair as 4 u32 words so the whole HELLO body stays u32-indexed.
        for v in [
            info.clock.mono_raw_ns as u32,
            (info.clock.mono_raw_ns >> 32) as u32,
            info.clock.realtime_ns as u32,
            (info.clock.realtime_ns >> 32) as u32,
        ] {
            buf.put_u32(v);
        }
        let wpl = info.words_per_layer();
        for l in 0..info.n_layer as usize {
            for w in 0..wpl {
                buf.put_u32(info.owned.get(l).and_then(|b| b.get(w)).copied().unwrap_or(0));
            }
        }
        patch_len(buf);
    }

    pub fn decode_hello(payload: &[u8]) -> eyre::Result<ShardInfo> {
        let u = |i: usize| -> eyre::Result<u32> {
            let o = i * 4;
            payload
                .get(o..o + 4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .ok_or_else(|| eyre!("hello: short payload"))
        };
        let mut info = ShardInfo {
            n_layer: u(0)?,
            n_expert: u(1)?,
            n_used: u(2)?,
            n_embd: u(3)?,
            xq_bytes_per_token: u(4)?,
            max_batch: u(5)?,
            decode_max_b: u(6)?,
            n_resident: u(7)?,
            bytes_per_expert: u(8)?,
            clock: super::ClockPair {
                mono_raw_ns: u(9)? as u64 | ((u(10)? as u64) << 32),
                realtime_ns: u(11)? as u64 | ((u(12)? as u64) << 32),
            },
            owned: Vec::new(),
        };
        let wpl = info.words_per_layer();
        let mut k = 13;
        for _ in 0..info.n_layer {
            let mut bits = Vec::with_capacity(wpl);
            for _ in 0..wpl {
                bits.push(u(k)?);
                k += 1;
            }
            info.owned.push(bits);
        }
        Ok(info)
    }

    /// Append a REQUEST frame. `xq` = `b * xq_bpt` bytes, `sel`/`ew` = `b * n_used`.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_request(
        buf: &mut AlignedBuf,
        seq: u32,
        layer: u32,
        b: u32,
        flags: u32,
        n_used: u32,
        xq_bpt: u32,
        xq: &[u8],
        sel: &[i32],
        ew: &[f32],
        hints: (&[u32], &[u32]),
        prefetch: &[u32],
        release: &[u32],
    ) -> u64 {
        debug_assert_eq!(xq.len(), (b * xq_bpt) as usize);
        debug_assert_eq!(sel.len(), (b * n_used) as usize);
        debug_assert_eq!(ew.len(), (b * n_used) as usize);
        buf.clear();
        put_header(buf, KIND_REQUEST, seq, 0);
        for v in [layer, b, flags, n_used, xq_bpt, 0u32] {
            buf.put_u32(v);
        }
        // Placeholder; the WRITER thread patches the real t1 in immediately
        // before write() (see `patch_u64`), so encode cost is not in the sample.
        buf.put_u64(0);
        buf.extend_from_slice(xq);
        // sel / ew as raw little-endian words (host is LE; the reader parses per word).
        let off = buf.len();
        buf.resize(off + sel.len() * 4 + ew.len() * 4);
        {
            let dst = buf.view_mut::<i32>(off, sel.len());
            dst.copy_from_slice(sel);
        }
        {
            let dst = buf.view_mut::<f32>(off + sel.len() * 4, ew.len());
            dst.copy_from_slice(ew);
        }
        if flags & REQ_FLAG_HINTS != 0 {
            buf.put_u32(hints.0.len() as u32);
            buf.put_u32(hints.1.len() as u32);
            for &w in hints.0.iter().chain(hints.1.iter()) {
                buf.put_u32(w);
            }
        }
        if flags & REQ_FLAG_PREFETCH != 0 {
            buf.put_u32(prefetch.len() as u32);
            for &w in prefetch {
                buf.put_u32(w);
            }
        }
        if flags & REQ_FLAG_RELEASE != 0 {
            buf.put_u32(release.len() as u32);
            for &w in release {
                buf.put_u32(w);
            }
        }
        patch_len(buf);
        0
    }

    /// Overwrite a u64 field in an already-encoded frame (t1 / t3 stamping).
    pub fn patch_u64(buf: &mut AlignedBuf, off: usize, v: u64) {
        buf.as_bytes_mut()[off..off + 8].copy_from_slice(&v.to_le_bytes());
    }

    pub fn read_u64(bytes: &[u8], off: usize) -> u64 {
        let mut a = [0u8; 8];
        a.copy_from_slice(&bytes[off..off + 8]);
        u64::from_le_bytes(a)
    }

    pub struct RequestView<'a> {
        pub layer: u32,
        pub b: u32,
        pub flags: u32,
        pub n_used: u32,
        pub xq_bpt: u32,
        /// Client's CLOCK_MONOTONIC_RAW immediately before its `write()`.
        pub t1: u64,
        pub xq: &'a [u8],
        pub sel: &'a [i32],
        pub ew: &'a [f32],
        /// `REQ_FLAG_HINTS` residency hints (see the flag); empty otherwise.
        pub hint_admit: &'a [u32],
        pub hint_evict: &'a [u32],
        /// `REQ_FLAG_PREFETCH` words; empty otherwise.
        pub prefetch: &'a [u32],
        /// `REQ_FLAG_RELEASE` words; empty otherwise.
        pub release: &'a [u32],
    }

    /// Parse a REQUEST frame held in `buf` (header included).
    pub fn decode_request(buf: &AlignedBuf) -> eyre::Result<RequestView<'_>> {
        let p = buf.as_bytes();
        if p.len() < HDR_LEN + REQ_FIXED {
            return Err(eyre!("request: short frame"));
        }
        let u = |i: usize| {
            let o = HDR_LEN + i * 4;
            u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]])
        };
        let (layer, b, flags, n_used, xq_bpt) = (u(0), u(1), u(2), u(3), u(4));
        let xq_len = (b as usize) * (xq_bpt as usize);
        let n_sel = (b as usize) * (n_used as usize);
        let xq_off = HDR_LEN + REQ_FIXED;
        let sel_off = xq_off + xq_len;
        let ew_off = sel_off + n_sel * 4;
        let hints_off = ew_off + n_sel * 4;
        let (mut hint_admit, mut hint_evict): (&[u32], &[u32]) = (&[], &[]);
        let expect_len = if flags & REQ_FLAG_HINTS != 0 {
            if hints_off + 8 > p.len() {
                return Err(eyre!("request: hints flagged but frame too short"));
            }
            let rd = |o: usize| u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]]) as usize;
            let (na, ne) = (rd(hints_off), rd(hints_off + 4));
            let end = hints_off + 8 + (na + ne) * 4;
            if end > p.len() {
                return Err(eyre!("request: hints n_admit={na} n_evict={ne} overrun frame"));
            }
            hint_admit = buf.view::<u32>(hints_off + 8, na);
            hint_evict = buf.view::<u32>(hints_off + 8 + na * 4, ne);
            end
        } else {
            hints_off
        };
        let mut prefetch: &[u32] = &[];
        let expect_len = if flags & REQ_FLAG_PREFETCH != 0 {
            if expect_len + 4 > p.len() {
                return Err(eyre!("request: prefetch flagged but frame too short"));
            }
            let n = u32::from_le_bytes([p[expect_len], p[expect_len + 1], p[expect_len + 2], p[expect_len + 3]]) as usize;
            let end = expect_len + 4 + n * 4;
            if end > p.len() {
                return Err(eyre!("request: prefetch n={n} overruns frame"));
            }
            prefetch = buf.view::<u32>(expect_len + 4, n);
            end
        } else {
            expect_len
        };
        let mut release: &[u32] = &[];
        let expect_len = if flags & REQ_FLAG_RELEASE != 0 {
            if expect_len + 4 > p.len() {
                return Err(eyre!("request: release flagged but frame too short"));
            }
            let n = u32::from_le_bytes([p[expect_len], p[expect_len + 1], p[expect_len + 2], p[expect_len + 3]]) as usize;
            let end = expect_len + 4 + n * 4;
            if end > p.len() {
                return Err(eyre!("request: release n={n} overruns frame"));
            }
            release = buf.view::<u32>(expect_len + 4, n);
            end
        } else {
            expect_len
        };
        if expect_len != p.len() {
            return Err(eyre!(
                "request: frame len {} != expected {} (b={b}, xq_bpt={xq_bpt}, n_used={n_used})",
                p.len(),
                expect_len
            ));
        }
        // xq_off = 40 (8-aligned); sel_off = 40 + b*5840 is 4-aligned (5840 % 4 == 0).
        Ok(RequestView {
            layer,
            b,
            flags,
            n_used,
            xq_bpt,
            t1: read_u64(p, REQ_T1_OFF),
            xq: &p[xq_off..sel_off],
            sel: buf.view::<i32>(sel_off, n_sel),
            ew: buf.view::<f32>(ew_off, n_sel),
            hint_admit,
            hint_evict,
            prefetch,
            release,
        })
    }

    /// Start a RESPONSE frame; the caller appends `b * n_embd * elem_bytes` of
    /// payload (directly from the device) and calls `patch_len`.
    #[allow(clippy::too_many_arguments)]
    pub fn begin_response(
        buf: &mut AlignedBuf,
        seq: u32,
        layer: u32,
        b: u32,
        flags: u32,
        status: u32,
        t_compute_us: u32,
        t_server_us: u32,
        n_embd: u32,
        elem_bytes: u32,
        t1_echo: u64,
        t2: u64,
    ) {
        buf.clear();
        put_header(buf, KIND_RESPONSE, seq, 0);
        for v in [layer, b, flags, status, t_compute_us, t_server_us, n_embd, elem_bytes] {
            buf.put_u32(v);
        }
        buf.put_u64(t1_echo);
        buf.put_u64(t2);
        buf.put_u64(0); // t3: patched by the writer thread just before write()
        buf.put_u32(0); // t_page_us: patched below, once paging is accounted
        buf.put_u32(0); // n_miss
    }

    /// Byte offset of the response payload inside the frame (8-aligned).
    pub const RESP_DATA_OFF: usize = HDR_LEN + RESP_FIXED;

    #[derive(Clone, Copy, Debug)]
    pub struct ResponseMeta {
        pub layer: u32,
        pub b: u32,
        pub flags: u32,
        pub status: u32,
        pub t_compute_us: u32,
        pub t_server_us: u32,
        pub n_embd: u32,
        pub elem_bytes: u32,
        /// NTP quadruple: `t1` echoed back, `t2` = daemon receive,
        /// `t3` = daemon send (all CLOCK_MONOTONIC_RAW on their own box).
        pub t1: u64,
        pub t2: u64,
        pub t3: u64,
        /// Paging this request did on box 2's own disk (v2+). The hub draws these
        /// on the `remote.pager` lane so a stall can be attributed to box-2 NVMe
        /// rather than to queueing or compute.
        pub t_page_us: u32,
        pub n_miss: u32,
    }

    pub fn decode_response_meta(buf: &AlignedBuf) -> eyre::Result<ResponseMeta> {
        let p = buf.as_bytes();
        if p.len() < RESP_DATA_OFF {
            return Err(eyre!("response: short frame"));
        }
        let u = |i: usize| {
            let o = HDR_LEN + i * 4;
            u32::from_le_bytes([p[o], p[o + 1], p[o + 2], p[o + 3]])
        };
        let m = ResponseMeta {
            layer: u(0),
            b: u(1),
            flags: u(2),
            status: u(3),
            t_compute_us: u(4),
            t_server_us: u(5),
            n_embd: u(6),
            elem_bytes: u(7),
            t1: read_u64(p, RESP_T1_OFF),
            t2: read_u64(p, RESP_T2_OFF),
            t3: read_u64(p, RESP_T3_OFF),
            t_page_us: u32::from_le_bytes([
                p[RESP_PAGE_OFF], p[RESP_PAGE_OFF + 1], p[RESP_PAGE_OFF + 2], p[RESP_PAGE_OFF + 3],
            ]),
            n_miss: u32::from_le_bytes([
                p[RESP_MISSN_OFF], p[RESP_MISSN_OFF + 1], p[RESP_MISSN_OFF + 2], p[RESP_MISSN_OFF + 3],
            ]),
        };
        if m.flags & RESP_FLAG_PIN != 0 && m.flags & RESP_FLAG_RESID == 0 {
            return Err(eyre!("response: pin block without a residency map (flags {:#x})", m.flags));
        }
        let want = RESP_DATA_OFF
            + (m.b as usize) * (m.n_embd as usize) * (m.elem_bytes as usize)
            + if m.flags & RESP_FLAG_RESID != 0 { RESID_WORDS * 4 } else { 0 }
            + if m.flags & RESP_FLAG_PIN != 0 { PIN_WORDS * 4 } else { 0 };
        if p.len() != want {
            return Err(eyre!("response: frame len {} != expected {want}", p.len()));
        }
        Ok(m)
    }

    pub fn encode_error(buf: &mut AlignedBuf, seq: u32, status: u32, msg: &str) {
        buf.clear();
        put_header(buf, KIND_ERROR, seq, 0);
        buf.put_u32(status);
        buf.extend_from_slice(msg.as_bytes());
        patch_len(buf);
    }

    pub fn decode_error(buf: &AlignedBuf) -> (u32, String) {
        let p = &buf.as_bytes()[HDR_LEN..];
        if p.len() < 4 {
            return (u32::MAX, "malformed error frame".into());
        }
        let st = u32::from_le_bytes([p[0], p[1], p[2], p[3]]);
        (st, String::from_utf8_lossy(&p[4..]).into_owned())
    }
}

use proto::ShardInfo;


// ---------------------------------------------------------------------------
// Clock sync: NTP's own algorithm over our own link
// ---------------------------------------------------------------------------

/// One request/response pair's clock quadruple and what it implies.
///
/// `t1` client send, `t2` daemon receive, `t3` daemon send, `t4` client receive
/// (t1/t4 on box 1's CLOCK_MONOTONIC_RAW, t2/t3 on box 2's). Then
/// `offset = ((t2-t1) + (t3-t4))/2` (add to a CLIENT stamp to get a DAEMON
/// stamp) and `delay = ((t4-t1) - (t3-t2))/2` (one-way link time), exactly
/// NTP's estimator. The offset is exact when the two path directions are
/// symmetric; the error is bounded by half the asymmetry, which is why the
/// per-sample `delay` series is kept — a drifting or bimodal delay is how path
/// asymmetry and link queueing show up rather than being assumed absent.
#[derive(Clone, Copy, Debug)]
pub struct ClockSample {
    pub seq: u32,
    pub layer: u32,
    pub b: u32,
    pub t1: u64,
    pub t2: u64,
    pub t3: u64,
    pub t4: u64,
}

impl ClockSample {
    /// ns to ADD to a box-1 CLOCK_MONOTONIC_RAW stamp to get box 2's.
    pub fn offset_ns(&self) -> i64 {
        let a = self.t2 as i128 - self.t1 as i128;
        let b = self.t3 as i128 - self.t4 as i128;
        ((a + b) / 2) as i64
    }
    /// One-way link delay estimate (ns): round trip minus the daemon's own time.
    pub fn delay_ns(&self) -> i64 {
        let rtt = self.t4 as i128 - self.t1 as i128;
        let srv = self.t3 as i128 - self.t2 as i128;
        (((rtt - srv).max(0)) / 2) as i64
    }
    pub fn rtt_ns(&self) -> i64 {
        (self.t4 as i128 - self.t1 as i128) as i64
    }
    pub fn remote_service_ns(&self) -> i64 {
        (self.t3 as i128 - self.t2 as i128) as i64
    }
}

/// Rolling clock-offset estimate over the link, fed by every request.
///
/// One message per layer already flows, so a 40-layer token yields ~40 offset
/// samples at zero extra traffic. The windowed MEDIAN is the published offset
/// (a single queued packet inflates one direction and biases that sample; the
/// median rejects it), while the raw samples stay available for the asymmetry
/// question.
///
/// Measured on the real link (docs/v41/REMOTE_EXPERTS.md §5.5): the two boxes'
/// oscillators differ by ~65 ppm, so the offset moves 65 µs per second and a
/// static estimate is useless within ~50 ms; the published windowed median
/// tracks it to |err| p50 1.6 µs / p90 4.5 µs at window 32 on quiet boxes.
pub struct ClockSync {
    samples: std::collections::VecDeque<ClockSample>,
    window: usize,
    /// Cap on retained samples (0 = unbounded). Raw samples are the point, so
    /// the default keeps a lot: 200k × 40 B ≈ 8 MB.
    capacity: usize,
    /// The daemon's (mono, realtime) pair from HELLO, and ours.
    pub remote_clock: ClockPair,
    pub local_clock: ClockPair,
    /// Only samples with `b <= offset_max_b` feed the OFFSET estimate.
    ///
    /// MEASURED (docs/v41/REMOTE_EXPERTS.md §5.5): NTP's estimator assumes the
    /// two path directions are symmetric, and its error is exactly half the
    /// asymmetry. A B=1024 exchange sends 6.03 MB and receives 10.49 MB — 4.5 MB
    /// of asymmetry — and its offset samples come out biased by −8.25 ms, vs a
    /// ±3 µs residual at B=1. So prefill-sized exchanges are kept in `samples`
    /// (their delay series is exactly how the asymmetry becomes visible) but are
    /// excluded from the published offset. 0 = accept every size.
    pub offset_max_b: u32,
}

impl ClockSync {
    pub fn new(remote_clock: ClockPair, window: usize, capacity: usize) -> Self {
        Self {
            samples: std::collections::VecDeque::new(),
            window: window.max(1),
            capacity,
            remote_clock,
            local_clock: ClockPair::sample(),
            offset_max_b: 4,
        }
    }

    /// Samples eligible for the offset estimate (see `offset_max_b`), newest last.
    fn offset_window(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(self.window);
        for s in self.samples.iter().rev() {
            if self.offset_max_b == 0 || s.b <= self.offset_max_b {
                out.push(s.offset_ns());
                if out.len() >= self.window {
                    break;
                }
            }
        }
        out
    }

    pub fn push(&mut self, s: ClockSample) {
        // O(1). This was `Vec::remove(0)` on a 200,000-entry Vec of 40-byte
        // samples -- an 8 MB memmove per call -- and `push` is on
        // `RemoteExpertClient::wait`, the hub's blocking per-layer call: 80 per
        // decode step. Steady state arrived after 200_000/80 = 2,500 steps and
        // then cost ~640 MB of memmove per step, permanently, on the thread that
        // serialises the whole step: "the box gets slower the longer a session
        // runs" (audit 2026-09-22 A5).
        //
        // A deque rather than a batched drain because every reader here is
        // iterator-based; only `samples()` wants a contiguous slice, and that is
        // the bench binary and the loopback test, never the hub.
        if self.capacity > 0 && self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(s);
    }

    /// Contiguous view, for callers that index or slice. Takes `&mut` because
    /// making a deque contiguous can move elements once; the hub never calls
    /// this, so the cost lands only in the bench/dump paths.
    pub fn samples(&mut self) -> &[ClockSample] {
        self.samples.make_contiguous()
    }

    /// Iterate without requiring `&mut` (oldest first).
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &ClockSample> {
        self.samples.iter()
    }
    pub fn len(&self) -> usize {
        self.samples.len()
    }
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    fn median(mut v: Vec<i64>) -> Option<i64> {
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    /// Median offset over the last `window` ELIGIBLE samples (ns to add to a
    /// box-1 monotonic stamp to get box 2's). `None` before the first one.
    ///
    /// A windowed median, not a lifetime average, for two measured reasons: the
    /// two boxes' oscillators differ by ~65 ppm (the offset genuinely moves 65 µs
    /// every second, so any static estimate is stale within ~50 ms), and a single
    /// queued packet biases one direction of one sample.
    pub fn offset_ns(&self) -> Option<i64> {
        Self::median(self.offset_window())
    }

    /// Relative clock RATE between the boxes (ns of offset per second, i.e. ppm),
    /// from a least-squares fit over the eligible samples. The offset moves at
    /// this rate, so it is what decides how often it must be re-estimated.
    pub fn drift_ppm(&self) -> Option<f64> {
        let pts: Vec<(f64, f64)> = self
            .samples
            .iter()
            .filter(|s| self.offset_max_b == 0 || s.b <= self.offset_max_b)
            .map(|s| (s.t1 as f64 / 1e9, s.offset_ns() as f64))
            .collect();
        if pts.len() < 16 {
            return None;
        }
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
        if sxx <= 0.0 {
            return None;
        }
        let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
        Some(sxy / sxx / 1e3)
    }

    /// Scatter of the offset samples about the fitted drift line (ns): the
    /// estimator's true precision, with the oscillator drift removed. This is
    /// the number that decides whether "did the partial arrive before the
    /// combine wanted it" is answerable at a 32 µs RTT.
    pub fn residual_ns(&self) -> Option<(i64, i64, i64)> {
        let slope = self.drift_ppm()? * 1e3;
        let pts: Vec<(f64, f64)> = self
            .samples
            .iter()
            .filter(|s| self.offset_max_b == 0 || s.b <= self.offset_max_b)
            .map(|s| (s.t1 as f64 / 1e9, s.offset_ns() as f64))
            .collect();
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p.0).sum::<f64>() / n;
        let my = pts.iter().map(|p| p.1).sum::<f64>() / n;
        let mut res: Vec<i64> = pts.iter().map(|p| (p.1 - (my + slope * (p.0 - mx))) as i64).collect();
        res.sort_unstable();
        let at = |q: f64| res[(((res.len() - 1) as f64) * q).round() as usize];
        Some((at(0.1), at(0.5), at(0.9)))
    }

    /// Median one-way delay over the window (ns).
    pub fn delay_ns(&self) -> Option<i64> {
        let skip = self.samples.len().saturating_sub(self.window);
        Self::median(self.samples.iter().skip(skip).map(|s| s.delay_ns()).collect())
    }

    /// (min, p50, p90, p99, max) of a per-sample series over ALL samples.
    pub fn spread(&self, f: impl Fn(&ClockSample) -> i64) -> Option<(i64, i64, i64, i64, i64)> {
        if self.samples.is_empty() {
            return None;
        }
        let mut v: Vec<i64> = self.samples.iter().map(&f).collect();
        v.sort_unstable();
        let at = |q: f64| v[(((v.len() - 1) as f64) * q).round() as usize];
        Some((v[0], at(0.5), at(0.9), at(0.99), v[v.len() - 1]))
    }

    /// ns to ADD to a box-2 perfetto (CLOCK_REALTIME) timestamp to place it on
    /// box 1's REALTIME timeline. Combines the measured monotonic offset with
    /// each box's own (mono, realtime) correlation pair, so it is as accurate as
    /// the offset — microseconds — rather than NTP's 100 µs - 1 ms.
    pub fn perfetto_shift_ns(&self) -> Option<i64> {
        let off = self.offset_ns()?;
        // A daemon monotonic stamp m2 corresponds to local monotonic m1 = m2 - off.
        // Map both to their own REALTIME and take the difference.
        let m2 = self.remote_clock.mono_raw_ns;
        let rt_remote = self.remote_clock.realtime_for(m2);
        let rt_local = self.local_clock.realtime_for((m2 as i128 - off as i128) as u64);
        Some((rt_local - rt_remote) as i64)
    }

    /// One-line summary for a report/log.
    pub fn summary(&self) -> String {
        let n = self.samples.len();
        let o = self.spread(|s| s.offset_ns());
        let d = self.spread(|s| s.delay_ns());
        match (o, d) {
            (Some(o), Some(d)) => format!(
                "clock sync over {n} samples: offset p50 {:.3} us (min {:.3}, p90 {:.3}, p99 {:.3}, max {:.3}; spread {:.3} us) | \
                 one-way delay p50 {:.3} us (min {:.3}, p90 {:.3}, p99 {:.3}) | perfetto shift {:.3} us",
                o.1 as f64 / 1e3, o.0 as f64 / 1e3, o.2 as f64 / 1e3, o.3 as f64 / 1e3, o.4 as f64 / 1e3,
                (o.4 - o.0) as f64 / 1e3,
                d.1 as f64 / 1e3, d.0 as f64 / 1e3, d.2 as f64 / 1e3, d.3 as f64 / 1e3,
                self.perfetto_shift_ns().unwrap_or(0) as f64 / 1e3,
            ) + &match (self.drift_ppm(), self.residual_ns()) {
                (Some(ppm), Some((r10, r50, r90))) => format!(
                    " | drift {ppm:+.1} ppm, detrended residual p10/p50/p90 {:+.3}/{:+.3}/{:+.3} us",
                    r10 as f64 / 1e3, r50 as f64 / 1e3, r90 as f64 / 1e3
                ),
                _ => String::new(),
            },
            _ => format!("clock sync: {n} samples"),
        }
    }
}

// ---------------------------------------------------------------------------
// Assignment: which (layer, expert) pairs a daemon holds
// ---------------------------------------------------------------------------

/// Sorted `(layer, expert ids)` list. Grammar (comma-separated items):
///   `L<a>[-L<b>][:<lo>-<hi>]`  layers a..=b (all experts, or ids lo..=hi)
///   `all[:<lo>-<hi>]`           every layer
///   `<layer>[:<lo>-<hi>]`       a bare layer number
/// e.g. `L20-L39`, `L0-L19:192-383`, `all:0-127`, `3:0-7,7:0-7`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Assignment {
    pub layers: Vec<(u32, Vec<u32>)>,
}

impl Assignment {
    /// Build an assignment from a **frequency-ranked placement file** instead of
    /// an id range.
    ///
    /// Contiguous `lo-hi` ranges ignore the fact that V4.1 routing is strongly
    /// Zipfian: on a 2048-token trace the top 2.4% of (layer, expert) pairs take
    /// 32.6% of all picks. Simulated on the same 6160-expert budget, selecting by
    /// frequency instead of by id range took box 1 from **10.15 to 1.87
    /// misses/token** — 5.4x, for identical RAM. Box 2 pins its whole assignment
    /// at load and never pages, so *which* static set it holds is the only policy
    /// choice available to it, and frequency dominates. (This does NOT contradict
    /// "LRU beats static placement": that compares policies for a tier that can
    /// page, which box 2's cannot.)
    ///
    /// Same file format and the same global-greedy allocator as the dGPU
    /// hot-expert placement file (`weights::parse_hot_expert_file`): one line per
    /// layer, comma-separated `id:count` in descending order, and the budget of
    /// `k_avg * N_LAYER` is taken by GLOBAL count rank — so skewed layers get more
    /// slots than flat ones.
    pub fn from_placement_file(path: &str, k_avg: usize) -> eyre::Result<Self> {
        let per_layer = crate::het::weights::parse_hot_expert_file(path, k_avg)?;
        let layers: Vec<(u32, Vec<u32>)> = per_layer
            .into_iter()
            .enumerate()
            .filter(|(_, ids)| !ids.is_empty())
            .map(|(l, mut ids)| {
                ids.sort_unstable();
                (l as u32, ids)
            })
            .collect();
        if layers.is_empty() {
            return Err(eyre!("placement file {path} selected no experts at k_avg={k_avg}"));
        }
        Ok(Self { layers })
    }

    pub fn parse(spec: &str) -> eyre::Result<Self> {
        let mut per_layer: std::collections::BTreeMap<u32, std::collections::BTreeSet<u32>> = Default::default();
        for item in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
            let (lay, range) = match item.split_once(':') {
                Some((l, r)) => (l, Some(r)),
                None => (item, None),
            };
            let (l0, l1) = if lay == "all" {
                (0u32, N_LAYER as u32 - 1)
            } else {
                let strip = |s: &str| s.strip_prefix('L').unwrap_or(s).to_string();
                match lay.split_once('-') {
                    Some((a, b)) => (
                        strip(a).parse().wrap_err_with(|| format!("bad layer in `{item}`"))?,
                        strip(b).parse().wrap_err_with(|| format!("bad layer in `{item}`"))?,
                    ),
                    None => {
                        let l: u32 = strip(lay).parse().wrap_err_with(|| format!("bad layer in `{item}`"))?;
                        (l, l)
                    }
                }
            };
            if l1 < l0 || l1 >= N_LAYER as u32 {
                return Err(eyre!("`{item}`: layers {l0}..={l1} outside 0..{N_LAYER}"));
            }
            let (e0, e1) = match range {
                None => (0u32, N_EXPERT - 1),
                Some(r) => {
                    let (a, b) = r.split_once('-').ok_or_else(|| eyre!("`{item}`: expert range needs lo-hi"))?;
                    (
                        a.trim().parse().wrap_err_with(|| format!("bad expert in `{item}`"))?,
                        b.trim().parse().wrap_err_with(|| format!("bad expert in `{item}`"))?,
                    )
                }
            };
            if e1 < e0 || e1 >= N_EXPERT {
                return Err(eyre!("`{item}`: experts {e0}..={e1} outside 0..{N_EXPERT}"));
            }
            for l in l0..=l1 {
                per_layer.entry(l).or_default().extend(e0..=e1);
            }
        }
        if per_layer.is_empty() {
            return Err(eyre!("empty expert assignment `{spec}`"));
        }
        Ok(Self { layers: per_layer.into_iter().map(|(l, s)| (l, s.into_iter().collect())).collect() })
    }

    pub fn n_experts(&self) -> usize {
        self.layers.iter().map(|(_, v)| v.len()).sum()
    }

    /// Per-layer ownership bitsets in the HELLO format.
    pub fn bitsets(&self) -> Vec<Vec<u32>> {
        let wpl = (N_EXPERT as usize).div_ceil(32);
        let mut out = vec![vec![0u32; wpl]; N_LAYER as usize];
        for (l, ids) in &self.layers {
            for &e in ids {
                out[*l as usize][(e / 32) as usize] |= 1 << (e % 32);
            }
        }
        out
    }
}

// ---------------------------------------------------------------------------
// ExpertShard: the daemon's resident pool
// ---------------------------------------------------------------------------

struct LayerShard {
    base_slot: u32,
    ids: Vec<u32>,
    /// `REMAP_LEN` entries: owned id -> `-(ABSOLUTE_slot)-1`; everything else
    /// (including the sentinel) -> 0 = "the other device takes it". Absolute
    /// since 2026-09-14 so a layer can hold a slot outside its own region.
    remap_dev: DeviceBuffer<i32>,
    owned: Vec<bool>,
    /// PAGED MODE (catch-all tier). `None` = the classic pinned shard, whose
    /// `ids`/`owned` are fixed at load and never change.
    ///
    /// When present, this layer's contiguous slot region is an LRU cache over
    /// ALL 384 experts rather than a fixed assignment, refilled from this box's
    /// OWN disk. That is what lets the hub stop synchronously paging: a hub miss
    /// is reassigned here (<=6.3 ms on our plaintext NVMe, overlapped with the
    /// hub's compute) instead of read from the hub's dm-crypt NVMe (6.9 ms,
    /// blocking). Weights never cross the USB4 link — at ~724 MB/s an 18.8 MB
    /// expert would cost ~26 ms, worse than either box reading its own disk.
    ///
    /// Per-LAYER regions, not one global pool: `layer_views` hands the executor a
    /// contiguous `[base_slot, base_slot+n)` range, so keeping the region fixed
    /// leaves the executor and the wire format untouched. The cost is that
    /// capacity cannot migrate between layers; a global pool would be strictly
    /// better and is a follow-up.
    page: Option<LayerPager>,
}

/// Per-layer paging COUNTERS. The residency state itself (LRU, slot ownership,
/// remap mirrors) lives on [`ExpertShard`] as a `ShardPool`, so a layer can evict
/// a slot belonging to another layer.
struct LayerPager {
    pub requests: u64,
    pub misses: u64,
    pub read_ns: u64,
    pub h2d_ns: u64,
    /// Sub-terms of `read_ns`, so a regression lands on the right line. The
    /// CODEGEN-FRAGILE note in `hf_v41::read_expert_raw` exists because a change
    /// there once moved 3x of cost from `alloc` into `repack` while tok/s barely
    /// budged: never judge this path on `read_ns` alone.
    pub pread_ns: u64,
    pub repack_cpu_ns: u64,
    /// GPU permute + the stream sync that waits on it (0 on the CPU path).
    pub repack_gpu_ns: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LoadStats {
    pub n_experts: usize,
    pub bytes: u64,
    pub seconds: f64,
}

/// ADMITTED hints that matched a resident slot (daemon side).
pub static HINTS_APPLIED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// One staging set handed to the reader thread by address (the set is owned by
/// exactly one side at a time: the thread from hint to `Done`, the main thread
/// from `Done` to admission, then back). Pinned memory never moves.
#[derive(Clone, Copy)]
struct SetPtr {
    p: [*mut u8; 3],
    n: [usize; 3],
}
unsafe impl Send for SetPtr {}

struct PfDone {
    layer: u32,
    e: u32,
    set: usize,
    /// `PfJob::stage`: lands in the prefill staging band.
    stage: bool,
    /// `PfJob::prefill`: lands prefill-class.
    prefill: bool,
    /// `PfJob::restore`.
    restore: u64,
    offs: [Option<(usize, usize, u32, u32)>; 3],
    coalesced: bool,
    /// Hint sent -> a reader picked it up (queueing behind other reads).
    queue_ns: u64,
    /// The read itself (`read_miss_into` wall).
    read_ns: u64,
    /// `evtrace` (NaN when off): t_hint, t_pop, t_read_start, t_read_end,
    /// yield_ns, pause_ns, certain at pop, certain at read, DEMAND_READS and
    /// running certain / speculative at READ START (after the yield), the
    /// role (start, end) x3, then the drive route (`ExpertRoute::code`).
    ev: [f64; 18],
}

/// One background read for the prefetch readers.
struct PfJob {
    layer: u32,
    e: u32,
    set: usize,
    /// A request needs it now (a parked request's own pick), as opposed to a
    /// guess (look-ahead) or a background admission (box-2 miss substitution).
    certain: bool,
    /// A PREFILL-shaped request's own pick (its early page / park) that lands in
    /// the staging band (cleared when staging is off: see `own_prefill`).
    stage: bool,
    /// A PREFILL-shaped request's own pick, staged or not: reads by
    /// `knobs::prefill_route_split` (striped across both drives, like its
    /// demand reads). Separate from `stage` because staging off (production,
    /// mode-aware eviction) clears `stage` -- and keying the route on it sent
    /// every early-page read mirror-only (60-80% of prefill reads on the SN5000
    /// at 13.8 ms p50 while the E100 idled; MEASURED 2026-10-01).
    own_prefill: bool,
    /// Lands PREFILL-class (evicted first) even outside the staging band: a hub
    /// word carried by a prefill-shaped request (layer-major group prefetch).
    prefill: bool,
    /// Delta restore (`V41_B2_RESTORE`): the decode stamp the expert had when a
    /// prefill phase evicted it (0 = not a restore read).
    restore: u64,
    t_hint: std::time::Instant,
}

/// The readers' job queue. URGENT (`certain`) jobs run before every
/// speculative one, and a speculative job already queued is PROMOTED when a
/// request turns out to need it. Before this it was one FIFO channel: a
/// parked request whose pick was already queued as a background admission was
/// deduped onto that job and waited out the whole queue ahead of it, plus the
/// job's own yield to demand reads (2026-09-25: `box2.page_ms` 37 ms/step of
/// parked waits with ~0 demand misses).
struct PfQueue {
    inner: std::sync::Mutex<PfQueueInner>,
    cv: std::sync::Condvar,
}

struct PfQueueInner {
    jobs: std::collections::VecDeque<PfJob>,
    /// Keys made urgent after a reader had already popped them: a reader that
    /// is still yielding stops yielding. Cleared when the key LANDS
    /// (`admit_prefetched`), since urgency only means something while the key
    /// is pending.
    urgent: std::collections::HashSet<(u32, u32)>,
    /// Readers currently holding a speculative / a certain job.
    running_spec: usize,
    running_certain: usize,
    /// At most this many speculative jobs run at once: the rest of the readers
    /// (`V41_B2_PREFETCH_RESERVE`) are kept for certain ones.
    max_spec: usize,
    /// Reader threads; under urgency routing speculative jobs may use at most
    /// `n_readers - 1` of them, so a certain read always finds one.
    n_readers: usize,
    closed: bool,
    /// `knobs::route_urgency`: keys whose SPECULATIVE read a reader is running
    /// (on the primary drive); a request that needs one does not wait for it.
    spec_keys: std::collections::HashSet<(u32, u32)>,
}

impl PfQueue {
    #[cfg(test)]
    fn new(max_spec: usize) -> Self {
        Self::with_readers(max_spec, max_spec + 1)
    }

    fn with_readers(max_spec: usize, n_readers: usize) -> Self {
        Self {
            inner: std::sync::Mutex::new(PfQueueInner {
                jobs: Default::default(),
                urgent: Default::default(),
                running_spec: 0,
                running_certain: 0,
                max_spec: max_spec.max(1),
                n_readers: n_readers.max(1),
                closed: false,
                spec_keys: Default::default(),
            }),
            cv: std::sync::Condvar::new(),
        }
    }

    /// Urgent jobs go behind the other urgent ones and ahead of every
    /// speculative one; speculative jobs go to the back.
    fn push(&self, job: PfJob) {
        let mut g = self.inner.lock().unwrap();
        if job.certain {
            let at = g.jobs.iter().position(|j| !j.certain).unwrap_or(g.jobs.len());
            g.jobs.insert(at, job);
        } else {
            g.jobs.push_back(job);
        }
        drop(g);
        self.cv.notify_one();
    }

    /// A request needs `(layer, e)`, which is already pending: move its job up
    /// to the urgent section, or, if a reader already has it (or it has been
    /// read but not landed), mark it urgent so a yielding reader stops
    /// yielding. Returns whether anything changed (a job that is already
    /// certain is left alone).
    fn promote(&self, layer: u32, e: u32) -> bool {
        let mut g = self.inner.lock().unwrap();
        let changed = match g.jobs.iter().position(|j| j.layer == layer && j.e == e) {
            Some(i) if g.jobs[i].certain => false,
            Some(i) => {
                let mut job = g.jobs.remove(i).expect("index from position");
                job.certain = true;
                let at = g.jobs.iter().position(|j| !j.certain).unwrap_or(g.jobs.len());
                g.jobs.insert(at, job);
                true
            }
            // A speculative read already running on the primary under urgency
            // routing is not made urgent: the request reads the expert from
            // the mirror instead (see `admit_prefetched`).
            None if g.spec_keys.contains(&(layer, e)) => false,
            None => g.urgent.insert((layer, e)),
        };
        drop(g);
        if changed {
            self.cv.notify_all();
        }
        changed
    }

    /// Should a background read of `(layer, e)` hold off? Yes while an urgent
    /// read is actually RUNNING: a certain job on a reader, or a speculative
    /// one promoted in flight (`urgent`, which `ensure` may be blocked on) --
    /// unless `(layer, e)` is itself urgent. A certain job that is merely
    /// QUEUED does not count: it gets a reserved reader within microseconds,
    /// and with no reader free, pausing the readers it is waiting for would
    /// only idle the drives. One lock per check.
    fn background_should_wait(&self, layer: u32, e: u32) -> bool {
        let g = self.inner.lock().unwrap();
        if g.urgent.contains(&(layer, e)) {
            return false;
        }
        g.running_certain > 0 || !g.urgent.is_empty()
    }

    fn is_urgent(&self, layer: u32, e: u32) -> bool {
        self.inner.lock().unwrap().urgent.contains(&(layer, e))
    }

    /// `evtrace`: running certain, running speculative, queued certain,
    /// queued speculative.
    fn counts(&self) -> [f64; 4] {
        let g = self.inner.lock().unwrap();
        let qc = g.jobs.iter().filter(|j| j.certain).count();
        [g.running_certain as f64, g.running_spec as f64, qc as f64, (g.jobs.len() - qc) as f64]
    }

    fn clear_urgent(&self, layer: u32, e: u32) {
        self.inner.lock().unwrap().urgent.remove(&(layer, e));
    }

    /// Blocks for the next job the caller may run; `None` once closed and
    /// drained. A certain job is always handed out. Under `route=split` a
    /// speculative one only while fewer than `max_spec` run and no certain job
    /// is running or waiting, so the reserved readers, and the drives, are
    /// free for urgent reads; under `route=urgency` see `pop_mode`. The caller
    /// must call `finished` with the returned `certain`.
    #[cfg(test)]
    fn pop(&self) -> Option<PfJob> {
        self.pop_mode(false)
    }

    /// `pop`, under a drive routing. `urgency` (`knobs::route_urgency`):
    /// speculative jobs read from the OTHER drive, so they no longer wait for
    /// running certain ones (still at most `max_spec`), and a speculative job
    /// handed out is recorded in `spec_keys` until `finish_spec_key`.
    fn pop_mode(&self, urgency: bool) -> Option<PfJob> {
        let mut g = self.inner.lock().unwrap();
        loop {
            if g.closed {
                // Shutdown drains everything, ungated.
                let j = g.jobs.pop_front()?;
                if j.certain { g.running_certain += 1 } else { g.running_spec += 1 }
                return Some(j);
            }
            match g.jobs.front() {
                Some(j) if j.certain => {
                    g.running_certain += 1;
                    return g.jobs.pop_front();
                }
                Some(_) if g.running_spec < Self::spec_cap(&g, urgency) && (urgency || g.running_certain == 0) => {
                    g.running_spec += 1;
                    let j = g.jobs.pop_front()?;
                    if urgency {
                        g.spec_keys.insert((j.layer, j.e));
                    }
                    return Some(j);
                }
                _ => {}
            }
            g = self.cv.wait(g).unwrap();
        }
    }

    /// Speculative jobs allowed at once: `max_spec`, and under urgency routing
    /// (where they no longer wait for certain ones) never every reader.
    fn spec_cap(g: &PfQueueInner, urgency: bool) -> usize {
        if urgency { g.max_spec.min(g.n_readers.saturating_sub(1)).max(1) } else { g.max_spec }
    }

    /// A speculative read recorded by `pop_mode(true)` is done.
    fn finish_spec_key(&self, layer: u32, e: u32) {
        self.inner.lock().unwrap().spec_keys.remove(&(layer, e));
    }

    /// The keys whose speculative read is running now (urgency routing).
    fn spec_keys_snapshot(&self) -> std::collections::HashSet<(u32, u32)> {
        self.inner.lock().unwrap().spec_keys.clone()
    }

    /// A job popped as speculative turned out to be needed (`urgent`) before
    /// its read started: count it as certain from now on, so the gate keeps
    /// new speculative reads off the drives while it runs.
    fn reclassify_certain(&self) {
        let mut g = self.inner.lock().unwrap();
        g.running_spec = g.running_spec.saturating_sub(1);
        g.running_certain += 1;
    }

    /// A reader finished a job it popped as `certain` (or not).
    fn finished(&self, certain: bool) {
        let mut g = self.inner.lock().unwrap();
        if certain {
            g.running_certain = g.running_certain.saturating_sub(1);
        } else {
            g.running_spec = g.running_spec.saturating_sub(1);
        }
        drop(g);
        self.cv.notify_all();
    }

    fn close(&self) {
        self.inner.lock().unwrap().closed = true;
        self.cv.notify_all();
    }
}

/// The readers' queue, for the background-read pause hook
/// (`v4flash_core::io_throttle`); set when the readers start (one shard per
/// daemon).
static PF_QUEUE: std::sync::OnceLock<std::sync::Arc<PfQueue>> = std::sync::OnceLock::new();

/// `V41_B2_SPEC_CHUNK_KB` (default 1024; 0 = unchunked): background expert
/// reads (look-ahead, substitution admissions) go to the drive in pieces of
/// this size and pause before each while a demand or urgent read is active,
/// so an urgent read waits for at most the pieces already in flight instead
/// of whole 18.8 MB reads (io_throttle).
fn b2_spec_chunk_bytes() -> usize {
    std::env::var("V41_B2_SPEC_CHUNK_KB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024).saturating_mul(1024)
}

/// Before each chunk of a background read: wait while a demand read runs or an
/// urgent (certain) read is active, unless this very expert has become
/// urgent; at most 20 ms per chunk.
fn b2_background_pause(token: u64) {
    let (layer, e) = ((token >> 16) as u32, (token & 0xFFFF) as u32);
    let Some(q) = PF_QUEUE.get() else { return };
    let t = std::time::Instant::now();
    let mut slept = false;
    while (DEMAND_READS.load(std::sync::atomic::Ordering::Relaxed) > 0 || q.background_should_wait(layer, e))
        && !q.is_urgent(layer, e)
        && t.elapsed() < std::time::Duration::from_millis(20)
    {
        std::thread::sleep(std::time::Duration::from_micros(50));
        slept = true;
    }
    if slept && super::evtrace::enabled() {
        *EV_PAUSE.lock().unwrap().entry(token).or_insert(0) += t.elapsed().as_nanos() as u64;
    }
}

/// Must `admit_prefetched` block for one of `want` (layer `layer`)? Only for
/// a key still pending whose read is NOT a speculative one running on the
/// primary under urgency routing (`skip`, empty under `split`).
fn wanted_in_flight(
    want: &[u32],
    layer: u32,
    pending: &std::collections::HashSet<(u32, u32)>,
    skip: &std::collections::HashSet<(u32, u32)>,
) -> bool {
    want.iter().any(|&e| pending.contains(&(layer, e)) && !skip.contains(&(layer, e)))
}

/// Does the park loop keep waiting for `key`? While pending -- except, under
/// urgency routing, a speculative read running on the primary (the request
/// reads it from the mirror itself) or a key already resident (made so by a
/// mirror read of an interleaved request; its late copy is discarded).
fn park_waits_for(
    key: (u32, u32),
    pending: &std::collections::HashSet<(u32, u32)>,
    spec: &std::collections::HashSet<(u32, u32)>,
    resident: impl Fn(&(u32, u32)) -> bool,
    urgency: bool,
) -> bool {
    pending.contains(&key) && !(urgency && (spec.contains(&key) || resident(&key)))
}

/// Calls `PfQueue::finished` when a reader is done with a job, including by
/// panic, so a failed read can never leave the gate counters raised (which
/// would keep speculative reads off for good).
struct PfFinish<'a> {
    q: &'a PfQueue,
    certain: bool,
    /// A speculative key `pop_mode(true)` recorded, released with the job.
    spec_key: Option<(u32, u32)>,
}

impl Drop for PfFinish<'_> {
    fn drop(&mut self) {
        if let Some((l, e)) = self.spec_key {
            self.q.finish_spec_key(l, e);
        }
        self.q.finished(self.certain);
    }
}

struct B2Prefetch {
    queue: std::sync::Arc<PfQueue>,
    rx_done: std::sync::mpsc::Receiver<Result<PfDone, (usize, u32, u32, String)>>,
    /// Reader threads; they write into `stages` by address, so `Drop` joins
    /// them before the staging is freed.
    readers: Vec<std::thread::JoinHandle<()>>,
    stages: Vec<[PinnedBuffer<u8>; 3]>,
    free: Vec<usize>,
    pending: std::collections::HashSet<(u32, u32)>,
    pub hinted: u64,
    pub admitted: u64,
    pub dropped: u64,
    /// Wanted ids whose prefetch read was still in flight at `ensure`: waited
    /// for instead of re-read (a partial win: the read was already started).
    pub waited: u64,
    /// Urgent requests for a key already pending (promoted instead of deduped).
    pub promoted: u64,
    /// Sums over completed reads: hint->start queueing, read wall, count.
    pub queue_ns: u64,
    pub read_ns: u64,
    pub n_read: u64,
}

impl Drop for B2Prefetch {
    fn drop(&mut self) {
        // Close the queue, so each reader finishes its current read into
        // `stages` and then sees `pop` return None. Fields -- `stages`
        // included -- are dropped only after this returns.
        self.queue.close();
        for h in self.readers.drain(..) {
            let _ = h.join();
        }
    }
}

/// `V41_B2_PREFETCH_PAR`: concurrent prefetch readers (default 4). One reader
/// serialised the hints (3.3 ms each with the mirror split) against a lead of
/// ~6-10 ms from the hint to the next layer's `ensure`, so only the first two
/// or three hints per layer ever landed in time.
/// Demand (miss) reads in flight on this box. Speculative prefetch reads wait
/// for zero before starting: v2 (2026-09-21) raised the per-miss cost from
/// 3.25 to 4.85 ms because the L+1 hints arrive in the same request as layer
/// L's demand misses and raced them on the same two drives. A CERTAIN hint (a
/// queued request's own non-resident picks, see the compute loop) is a demand
/// read that merely started early and never waits.
static DEMAND_READS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// `evtrace`: the request the compute thread is serving (its `seq`), stamped
/// into the `b2_read` / `b2_ensure` records emitted under it.
pub static EV_CUR_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(u64::MAX);

fn ev_cur_seq() -> f64 {
    match EV_CUR_SEQ.load(std::sync::atomic::Ordering::Relaxed) {
        u64::MAX => f64::NAN,
        s => s as f64,
    }
}

/// `a |= b` over residency words.
fn or_words(a: &mut [u32; proto::RESID_WORDS], b: &[u32; proto::RESID_WORDS]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x |= *y;
    }
}

/// `evtrace` `b2_req` pin fields, `pin_on` .. `n_paged` in `B2_REQ` order:
/// the pin state after the request, counter deltas across it (`before` =
/// `ExpertShard::pin_counters` at its start), its release words, and how many
/// of its pass's experts were PAGED.
fn ev_pin_fields(
    before: Option<(PinCounters, u32, u32, u32)>,
    after: Option<(PinCounters, u32, u32, u32)>,
    n_release: usize,
    paged: &[u32; proto::RESID_WORDS],
) -> [f64; 10] {
    let nan = f64::NAN;
    let n_paged = paged.iter().map(|w| w.count_ones()).sum::<u32>() as f64;
    let Some((c1, pinned, budget, epoch)) = after else {
        return [0.0, nan, nan, nan, n_release as f64, nan, nan, nan, nan, n_paged];
    };
    let c0 = before.map(|b| b.0).unwrap_or_default();
    let d = |a: u64, b: u64| a.saturating_sub(b) as f64;
    [
        1.0, f64::from(pinned), f64::from(budget), f64::from(epoch), n_release as f64,
        d(c1.new_pins, c0.new_pins), d(c1.denied, c0.denied), d(c1.no_victim_drops, c0.no_victim_drops),
        d(c1.pinned_evictions, c0.pinned_evictions), n_paged,
    ]
}

/// `evtrace` `b2_req` prefill-staging fields, `stage_claims` .. `stage_spills`
/// in `B2_REQ` order: counter deltas across the request.
fn ev_stage_fields(before: StageCounters, after: StageCounters) -> [f64; 3] {
    let d = |a: u64, b: u64| a.saturating_sub(b) as f64;
    [
        d(after.claims, before.claims),
        d(after.hits, before.hits),
        d(after.spill_in + after.spill_out, before.spill_in + before.spill_out),
    ]
}

thread_local! {
    /// `evtrace`: (start, end) raw ns of the three role reader threads of the
    /// last `read_miss_into` called on THIS thread (NaN = not measured).
    static EV_ROLES: std::cell::Cell<[f64; 6]> = const { std::cell::Cell::new([f64::NAN; 6]) };
}

/// `evtrace`: io_throttle pause ns per background read (`layer << 16 | e`
/// token), taken by the reader when its read ends.
static EV_PAUSE: std::sync::LazyLock<std::sync::Mutex<std::collections::HashMap<u64, u64>>> =
    std::sync::LazyLock::new(Default::default);

/// `V41_B2_EARLY_PAGE=0`: do not start a queued request's misses under the
/// current request's tail (default on).
/// `V41_B2_MERGE` (default ON): when the next queued frame is a request for the
/// SAME layer (the other hub lane), run both as one MoE pass and answer both
/// from its output. Dedups the experts across all rows and pays the fixed
/// per-request cost once; with two lanes each request otherwise re-reads the
/// experts the other lane's rows also picked ("lanes cost bytes", 2026-09-21).
/// `V41_B2_MERGE_WAIT_US` (default 400): how long the daemon may wait for a
/// promised partner (`REQ_FLAG_PARTNER`) before giving up and running alone.
/// Bounded because the promise can go unfulfilled -- the hub's other lane can
/// fail its own route, or the step can end -- and an unfulfilled promise costs
/// exactly this much. 0 disables the wait (plain try_recv).
fn b2_merge_wait_us() -> u64 {
    knobs::merge_wait_us()
}

fn b2_merge() -> bool {
    knobs::merge()
}

/// RUNTIME KNOBS (2026-09-22). A daemon restart costs a 116 GB cold pool and
/// ~10 minutes of warm-up, so A/B-ing two settings used to mean two restarts and
/// two confounded warm-ups. Since 10-01 these are `crate::knobs` (one
/// implementation for every process): seeded from the env, overridden by the
/// knob file -- `V41_KNOBS_FILE`, else `path()` (`V41_B2_KNOBS`, default
/// `~/expertd-knobs.txt`), with the short keys below as aliases -- re-read every
/// second by the watcher (`crate::knobs::start_with`, daemon main) and at once on
/// SIGUSR2. A key REMOVED from the file now reverts its knob to the env/default
/// (until 10-01 the last value stuck).
///
///     printf 'merge=0\nmiss_par=2\n' > ~/expertd-knobs.txt   # (kill -USR2 <pid>: now)
///
/// `miss_par` can only be LOWERED below the startup `V41_B2_MISS_PAR`, which
/// sizes the pinned staging sets.
pub mod knobs {
    use std::sync::atomic::{AtomicBool, Ordering::Relaxed};
    pub static DIRTY: AtomicBool = AtomicBool::new(false);
    /// Every shard's mirror opened (`set_mirror_ok`, at `enable_paging`).
    static MIRROR_OK: AtomicBool = AtomicBool::new(false);
    const MAX: u64 = u32::MAX as u64;
    fn sync_mirror_frac(k: &crate::knobs::Knob) {
        v4flash_core::hf_v41::set_expert_mirror_frac(k.f64() as f32);
    }
    crate::knobs! {
        /// `V41_B2_MERGE` (default on), key `merge`.
        pub static MERGE = Knob::flag("V41_B2_MERGE", true).alias("merge");
        /// `V41_B2_MERGE_WAIT_US` (default 400), key `merge_wait_us`.
        pub static MERGE_WAIT_US = Knob::int("V41_B2_MERGE_WAIT_US", 400, 0, MAX).alias("merge_wait_us");
        /// `V41_B2_MISS_PAR` (default 1, 1..=16), key `miss_par`.
        pub static MISS_PAR = Knob::int("V41_B2_MISS_PAR", 1, 1, 16).alias("miss_par");
        /// `V41_B2_COALESCE` (default off), key `coalesce`.
        pub static COALESCE = Knob::flag("V41_B2_COALESCE", false).alias("coalesce");
        /// `V41_B2_PARK` (default off), key `park`.
        pub static PARK = Knob::flag("V41_B2_PARK", false).alias("park");
        /// `V41_B2_ROUTE`: `split` (default) or `urgency`, key `route`.
        pub static ROUTE = Knob::choice("V41_B2_ROUTE", 0, &[&["split"], &["urgency"]]).alias("route");
        /// `V41_B2_PREFILL_ROUTE`: `split` (default) or `mirror`, key `prefill_route`.
        pub static PREFILL_ROUTE = Knob::choice("V41_B2_PREFILL_ROUTE", 0, &[&["split"], &["mirror"]]).alias("prefill_route");
        /// `V41_B2_FAST_CHAIN` (default on), key `fast_chain`.
        pub static FAST_CHAIN = Knob::flag("V41_B2_FAST_CHAIN", true).alias("fast_chain");
        /// `V41_B2_PREFILL_BUDGET` (default 3500), key `prefill_budget`.
        pub static PREFILL_BUDGET = Knob::int("V41_B2_PREFILL_BUDGET", 3500, 0, MAX).alias("prefill_budget");
        /// `V41_B2_ENCODER_VICTIMS_FIRST` (default on), key `encoder_victims_first`.
        pub static ENCODER_VICTIMS_FIRST = Knob::flag("V41_B2_ENCODER_VICTIMS_FIRST", true).alias("encoder_victims_first");
        /// `V41_EXPERT_MIRROR_FRAC` (default 0.6), key `mirror_frac`: pushed into
        /// `v4flash_core::hf_v41::set_expert_mirror_frac` (the reader's state).
        pub static MIRROR_FRAC = Knob::real("V41_EXPERT_MIRROR_FRAC", 0.6, 0.0, 1.0).alias("mirror_frac").hook(sync_mirror_frac);
    }
    /// `V41_B2_FAST_CHAIN` (default ON; `0` = the exact old chain); file key
    /// `fast_chain`. A batched pass of `b <= FAST_CHAIN_MAX_B` rows (every
    /// box-2 decode request of 2+ rows, verify batches, merged decode pairs)
    /// runs the short chain: ONE upload (xq + sel + ew), ONE fused group +
    /// work-item builder (`b2_moe_group_wi_builder`), no per-request partials
    /// memset (the reduce re-zeroes what it consumed), and for f16 replies the
    /// reduce writes the f16 result straight into pinned host memory (no cast
    /// kernel, no sync, no blocking copy). Bit-identical to the old chain
    /// (tests/remote_experts_fast_chain.rs). Kernel trace of a 3-row request
    /// (tests/remote_experts_chain_trace.rs, 2026-09-27, 1 distinct expert):
    /// 14 GPU commands -> 6, small commands 26.9 -> 12.7 us, gaps inside the
    /// chain 29.3 -> 17.5 us, host readback 25-39 us -> ~1 us.
    pub fn fast_chain() -> bool { FAST_CHAIN.on() }
    /// In-process toggle (tests, A/B harnesses); the daemon uses the env/file.
    pub fn set_fast_chain(on: bool) { FAST_CHAIN.set(if on { "1" } else { "0" }); }
    /// `V41_B2_PREFILL_ROUTE=split` (default) | `mirror`; file key
    /// `prefill_route`. Under `route=urgency` a PREFILL-shaped pass's demand
    /// reads (and its early-page / park reads, `PfJob::stage`) are STRIPED
    /// across both drives (`ExpertRoute::split`) instead of going wholly to
    /// the mirror like decode's demand reads: a prefill chunk's ~200 reads
    /// per layer otherwise monopolise the SN5000 alongside decode's demand
    /// reads while the E100 idles (owner's decision, 2026-09-27). `mirror`
    /// restores the pre-09-27 routing. No effect under `route=split`.
    pub fn prefill_route_split() -> bool { PREFILL_ROUTE.pick() == 0 }
    pub fn merge() -> bool { MERGE.on() }
    pub fn merge_wait_us() -> u64 { MERGE_WAIT_US.get() }
    pub fn miss_par() -> usize { MISS_PAR.usize() }
    /// Two preads per miss instead of eight (the whole 3-role run at once).
    /// ROOT CAUSE of the 2026-09-18 corruption, for the record: the checkpoint
    /// stores w1/w2/w3 but the loader maps gate<-w1, up<-w3, down<-w2, so a
    /// physically-ordered run is gate,down,up and an implementation assuming
    /// gate,up,down swapped two roles on every page-in. It was blessed warm, at
    /// temp 0, where page-ins are rare and the path barely ran.
    /// `read_expert_runs_direct` now derives each role's position from its file
    /// OFFSET, and `V41_B2_COALESCE_CHECK=1` byte-compares every coalesced read
    /// against the per-role one. NOTE it also gives up the mirror split (the
    /// span read only touches the primary drive), so it trades 8 preads at
    /// ~9.9 GB/s for 2 at ~4.5.
    pub fn coalesce() -> bool { COALESCE.on() }
    /// PARK a request that must page (batched hits-first path, sender set
    /// `REQ_FLAG_OOO`): its misses go to the prefetch readers and requests
    /// already queued behind it are served and ANSWERED while they read, on a
    /// second executor. Without it a queued request (the other hub lane) waits
    /// out the whole NVMe read: measured 2026-09-23 at 4 rows, ~131 ms/step of
    /// box-2 queueing, ~as much as box 2's own page time.
    pub fn park() -> bool { PARK.on() }
    /// `route=urgency` (`V41_B2_ROUTE=urgency`; default `split`): which drive
    /// each expert read uses. `split` = every read split across both drives by
    /// `mirror_frac`, scales from the primary. `urgency` = reads a request is
    /// or will soon be waiting on (demand misses, CERTAIN background reads)
    /// come WHOLLY from the mirror (box 2's SN5000), speculative background
    /// reads wholly from the primary (the E100, also the OS disk, whose reads
    /// stall 5-20x under any write burst: evtrace 2026-09-25). The two classes
    /// then share no drive, so speculative reads neither yield to demand reads
    /// nor wait behind certain ones; and a request never waits on a
    /// speculative read already running on the E100 -- it reads the expert
    /// itself from the SN5000 and the late copy is discarded on landing.
    /// Needs a usable mirror on EVERY shard (`set_mirror_ok`); otherwise it
    /// is `split` (a mirror-only read would silently fall back to the primary).
    /// Decode victims one prefill phase may take before it evicts its own
    /// oldest pages (mode-aware eviction, `V41_B2_MODE_EVICT`); read at the
    /// start of each prefill phase. Env `V41_B2_PREFILL_BUDGET`, file key
    /// `prefill_budget`, default 3500 (was 2048 until 2026-10-01). 2048 held one
    /// layer-major WINDOW (0 re-reads within a window) but not a job longer than
    /// one: from a 113K-row job's 4th window on, 96-100% of each window's reads
    /// were experts the same job had read and then evicted itself (40.5K reads
    /// for 3,398 distinct experts, MEASURED 2026-10-01); ~3500 covers a job's
    /// box-2 union (~170 per layer x 20 encoder layers). Its cost -- decode
    /// experts displaced per prefill phase -- is not priced yet.
    pub fn prefill_budget() -> u64 { PREFILL_BUDGET.get() }
    /// When a prefill phase takes DECODE-class victims (mode-aware tiers 2 and
    /// 4), rank them by the prefill's layer SWEEP (`sweep_rank`): pages of
    /// layers this pass has gone past first, then the other region's, last the
    /// layers this pass will reach next. A CED prefill's windows sweep the
    /// encoder layers, and its replay (the last 128 rows through the decoder
    /// layers, right after) needs the decoder layers' experts -- exactly the
    /// decode pages a global-LRU victim choice had evicted: 640-750 box-2 reads
    /// per replay, 2.3-2.9 s, 90% of it waiting on box 2 (MEASURED 2026-10-01).
    /// Decode after the prefill loses as many pages either way, only from other
    /// layers. Env `V41_B2_ENCODER_VICTIMS_FIRST` (`0` = off), file key
    /// `encoder_victims_first`, default on (the name predates the sweep rank:
    /// on = rank decode victims by the sweep, off = plain LRU within a tier).
    pub fn encoder_victims_first() -> bool { ENCODER_VICTIMS_FIRST.on() }
    pub fn route_urgency() -> bool {
        ROUTE.pick() == 1 && MIRROR_OK.load(Relaxed)
    }
    pub fn set_mirror_ok(ok: bool) {
        MIRROR_OK.store(ok, Relaxed);
    }
    pub fn path() -> String {
        std::env::var("V41_B2_KNOBS").unwrap_or_else(|_| {
            format!("{}/expertd-knobs.txt", std::env::var("HOME").unwrap_or_else(|_| "/tmp".into()))
        })
    }
    /// Re-read the knob file now (SIGUSR2; the watcher also does every
    /// second); returns a one-line summary for the log.
    pub fn reload() -> String {
        crate::knobs::step_now();
        format!("knobs reloaded from {:?}: park={} merge={} wait_us={} miss_par={} coalesce={} mirror_frac={:.3} route={} prefill_route={} fast_chain={} prefill_budget={} encoder_victims_first={}",
            crate::knobs::knob_file(), u8::from(park()), u8::from(merge()), merge_wait_us(), miss_par(), u8::from(coalesce()),
            v4flash_core::hf_v41::expert_mirror_frac(), if route_urgency() { "urgency" } else { "split" },
            if prefill_route_split() { "split" } else { "mirror" }, u8::from(fast_chain()), prefill_budget(), u8::from(encoder_victims_first()))
    }
}

extern "C" fn knobs_signal(_sig: i32) {
    knobs::DIRTY.store(true, std::sync::atomic::Ordering::Relaxed);
}

/// Install the SIGUSR2 handler for [`knobs::reload`] (daemon main).
pub fn install_knobs_toggle() -> String {
    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    const SIGUSR2: i32 = 12;
    // The file is read at startup (`crate::knobs::start_with` in the daemon's
    // main, and once more here), not only on SIGUSR2. Without that it was inert
    // until someone signalled, so a launch script that wrote `miss_par=1` into it
    // while passing `V41_B2_MISS_PAR=4` ran at 4 and looked like 1 (2026-09-22
    // audit, B4).
    let _ = knobs::reload();
    let init = format!("knobs: merge={} wait_us={} miss_par={} coalesce={} mirror_frac={:.3} route={} prefill_route={} fast_chain={} prefill_budget={} encoder_victims_first={} (SIGUSR2 reloads {})",
        u8::from(knobs::merge()), knobs::merge_wait_us(), knobs::miss_par(), u8::from(knobs::coalesce()),
        v4flash_core::hf_v41::expert_mirror_frac(), if knobs::route_urgency() { "urgency" } else { "split" },
        if knobs::prefill_route_split() { "split" } else { "mirror" }, u8::from(knobs::fast_chain()), knobs::prefill_budget(), u8::from(knobs::encoder_victims_first()), knobs::path());
    unsafe { signal(SIGUSR2, knobs_signal); }
    init
}

fn b2_early_page() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("V41_B2_EARLY_PAGE").as_deref() != Ok("0"));
    *B
}

fn b2_prefetch_par() -> usize {
    std::env::var("V41_B2_PREFETCH_PAR").ok().and_then(|v| v.parse().ok()).unwrap_or(4usize).clamp(1, 16)
}

/// `V41_B2_PREFETCH_RESERVE` (default 1): prefetch READERS kept for certain
/// reads (a request's own picks); twice as many staging sets are kept free of
/// speculative ones (look-ahead, substitution admissions), which are dropped
/// instead. Clamped so at least one reader stays for speculative work.
/// `RESERVE=0` keeps the certain-first gate (no speculative read starts while a
/// certain one runs or waits) under `route=split`; under `route=urgency` the
/// two classes use different drives, speculative jobs do not wait for certain
/// ones, and (with more than one reader) at least one reader is always left
/// for certain jobs.
/// `RESERVE >= sets / 2` drops every speculative word.
fn b2_prefetch_reserve() -> usize {
    static R: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_PREFETCH_RESERVE").ok().and_then(|v| v.parse().ok()).unwrap_or(1usize).min(8)
    });
    *R
}

/// `V41_B2_PREFETCH_SETS`: staging sets = max prefetch reads in flight (default 8).
fn b2_prefetch_sets() -> usize {
    std::env::var("V41_B2_PREFETCH_SETS").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).clamp(0, 32)
}

pub struct ExpertShard {
    /// Shared with the look-ahead reader threads (which hold their own clone).
    owner: std::sync::Arc<V41HfWeights>,
    device: Device,
    pub routed: RoutedExpertWeights,
    layers: Vec<Option<LayerShard>>,
    info: ShardInfo,
    pub load_stats: LoadStats,
    /// GPU MXFP4 HF->ggml permute, so a miss does not pay a CPU repack. Box 1's
    /// pager has had this since the M7 tier; this side did the repack on the CPU
    /// on EVERY miss, which is why its `read_ns` (8.30 ms) sat 1.84x above the
    /// drive's own 4.52 ms for the same 18.8 MB. `None` = CPU rollback path.
    repack: Option<Mxfp4Repack>,
    repack_stream: Option<Stream>,
    /// Zero-copy O_DIRECT reads are usable: GPU repack on, gate on, staging
    /// 4096-aligned. See [`b2_odirect`].
    direct: bool,
    /// `(layer, expert)` slots a request still computing on the GPU reads:
    /// never a victim while set (the compute loop pins the in-flight request's
    /// picks around a queued request's early paging).
    pub pinned: Vec<(u32, u32)>,
    /// A PARKED request's picks (`knobs::park`): never victims while other
    /// requests are served under its in-flight reads. Separate from `pinned`,
    /// which the early-paging hook sets and clears around each hint.
    pub parked_pins: Vec<(u32, u32)>,
    /// The parked request's non-resident picks as `layer << 16 | expert`, for
    /// the park hook to hand to the prefetch readers.
    pub park_words: Vec<u32>,
    /// ... and whether that request is PREFILL-shaped (its reads then land in
    /// the staging band: `prefetch_words_cls`).
    pub park_prefill: bool,
    /// Mode-aware eviction: the request being served is a PREFILL request by
    /// the hub's flag (no `REQ_FLAG_DECODE`). With the mode on, a claim is a
    /// prefill claim only inside a prefill phase AND for such a request, so a
    /// decode request never ranks in tiers or stamps prefill-class (e.g. the
    /// first requests after a burst, before the hysteresis ends the phase).
    pub req_prefill: bool,
    /// Pin mode: each queued request's picks NOT landed when its frame
    /// ARRIVED (see [`EarlyPaged`]). Empty unless pins are on.
    pub early_paged: EarlyPaged,
    /// Cumulative time the compute thread spent BLOCKED in `admit_prefetched`
    /// waiting for a prefetch read it needs this request (2026-09-22). Also
    /// added to the layer's `read_ns`, see the note there.
    pub prefetch_wait_ns: u64,
    /// `evtrace`: the last `admit_prefetched` call's blocked wait ns, landed
    /// reads, landed reads this request wanted, blocking receives, and wanted
    /// keys NOT waited for (speculative reads running on the primary, urgency).
    pub ev_admit: [u64; 5],
    /// Persistent staging, one per role, in `hipHostMalloc` memory.
    ///
    /// Two things at once. It is persistent, so the miss path no longer
    /// allocates `vec![0u8; bpe]` x3 per fault (18.8 MB of fresh pages, ~4.6k
    /// first-touch faults, on a box already at 118/124 GB). And it is
    /// device-visible: box 2 is an APU, so this IS the RAM the iGPU reads
    /// through GTT. The reader threads pread straight into it and the repack
    /// kernel consumes it in place, so the 18.8 MB H2D a miss used to pay
    /// (1.23 ms measured) is gone — it was copying system RAM to system RAM.
    ///
    /// NON_COHERENT so the iGPU may cache its reads (see `PinnedBuffer`).
    stages: Vec<[PinnedBuffer<u8>; 3]>,
    /// Look-ahead prefetch (`REQ_FLAG_PREFETCH`): background reads into
    /// dedicated staging sets, admitted at the top of the next `ensure`.
    prefetch: Option<B2Prefetch>,
    /// Prefetch staging sets before the reader thread starts (it starts on
    /// the first hint, once the shard sits at its final address).
    pf_stages_spare: Vec<[PinnedBuffer<u8>; 3]>,
    /// Shard-wide paging pool. `None` until `enable_paging`.
    pool: Option<ShardPool>,
}

/// Residency state for the whole shard, so eviction can cross layer regions.
///
/// Per-layer regions were never an executor requirement — `layer_views` hands the
/// kernel the whole pool and `remap` holds ABSOLUTE slots. What they cost is
/// capacity migration: decoder layers get 68 slots against encoder layers' 260,
/// and the routing trace says 92% of decode misses come from those 20 layers. A
/// global victim search fixes that at identical capacity (simulated 20.4 -> 10.4
/// misses/token; the static 164/164 version MEASURED 15.8-17.3 -> 9.65, decode
/// +22%).
///
/// PREFILL must not evict across layers: it sweeps a whole layer's union at once
/// (~203 experts at B=1024) and a global LRU would let one layer's sweep evict
/// another's. So the search is region-restricted whenever the request is
/// prefill-shaped. Decode and prefill never overlap within a request, and slot
/// CONTENTS are untouched by the switch — only which slots are candidates.
struct ShardPool {
    /// absolute slot -> (layer, expert) resident there. `None` = free.
    owner_of: Vec<Option<(u32, u32)>>,
    /// (layer, expert) -> absolute slot.
    slot_of: std::collections::HashMap<(u32, u32), u32>,
    /// Recency per ABSOLUTE slot: the tick of its last use (0 = "evict me
    /// first"). A touch is one store; the victim search is a min-scan over the
    /// candidate slots and runs only on a miss (~2/token). The previous
    /// `VecDeque` LRU cost `iter().position()` + `remove()` on EVERY hit (~3,000
    /// compares + ~1,500 moves, six times per request at 6,160 slots).
    last_use: Vec<u64>,
    tick: u64,
    /// host mirror of each layer's `remap_dev`.
    remap_hosts: Vec<Vec<i32>>,
    /// how many slots each layer currently holds, and the minimum it keeps under
    /// global eviction. See [`b2_pool_floor`].
    held: Vec<u32>,
    floor: Vec<u32>,
    /// this layer's `remap_host` changed and its `remap_dev` is stale. Uploaded
    /// lazily at the START of that layer's next `ensure_layer`, which is what
    /// lets an eviction touch another layer without touching its device buffer.
    dirty: Vec<bool>,
    /// The hub's pins on this connection (`proto::REQ_FLAG_PIN`). Off (and
    /// empty) unless the hub asked for them.
    pins: PinBook,
    /// First slot of the prefill STAGING band `[stage, n)` (`= n` when off).
    /// See "PREFILL STAGING" in the pinning block below.
    stage: u32,
    sc: StageCounters,
    /// Mode-aware eviction (`V41_B2_MODE_EVICT`); off unless enabled at load.
    me: ModeEvict,
}

// ---------------------------------------------------------------------------
// MODE-AWARE EVICTION (2026-09-29, `V41_B2_MODE_EVICT=1`, default OFF; needs
// staging off and the two-class LRU on).
//
// The two-class LRU stamps prefill-class slots older than every decode-class
// one, and the victim search takes the global minimum: right for DECODE (a
// prefill burst must not wipe decode's cold tier), wrong for PREFILL, whose
// claims then evict prefill's OWN pages first -- the prefill class never
// outgrows about one layer's miss union, so every chunk (and every sub-chunk
// of a layer-major group) re-reads what the previous one paged. With staging
// off, prefill's early-page / park landings are also stamped decode-class and
// search the same global minimum, i.e. they evict prefill's demand pages too.
//
// With it on, a PREFILL-mode search (a prefill-shaped claim or a prefill
// landing) ranks candidates in TIERS, least-recently-used within a tier:
//   0 free
//   1 prefill-class, stale (last used before this prefill phase began)
//   2 decode-class, while this phase's BUDGET of decode victims lasts
//   3 prefill-class, this phase (prefill evicts its own oldest)
//   4 decode-class, budget spent
//   5 decode-class slots this prefill phase HIT (it is still using them)
// A decode-mode search is unchanged (prefill-class first, then decode LRU).
// Every decode victim a prefill-mode search takes is recorded with its stamp
// (`decode_delta`) for the delta restore that follows (not yet built).
// Prefill's early-page / park reads land prefill-class. The PHASE is the hub's
// own (`REQ_FLAG_DECODE`, set on every request of its arena decode drivers and
// clear on every prefill-job request, tails and replay included), noted once
// per request before anything is claimed or landed, with hysteresis: a prefill
// phase ends only after `ME_DECODE_STREAK` consecutive decode-flagged requests
// (a decode request served inside a parked prefill chunk must not restart it).
// Inside a prefill phase every claim ranks in tiers and stamps prefill-class,
// whatever its shape (a 6-row prompt tail is prefill); outside one, nothing
// ranks in tiers (a prefill landing then takes the plain LRU: prefill-class
// first). Requires the global pool (tiers across the whole band).
// ---------------------------------------------------------------------------

/// Within the DECODE tiers (2 and 4) of a prefill-mode search for `for_layer`
/// (`knobs::encoder_victims_first`), by the prefill's layer sweep -- an encoder
/// window sweeps layers `< CED_DECODER_START` upward, the replay the decoder
/// layers upward:
///   0  an encoder layer this pass has gone past (`l < for_layer`; untouched,
///      else it would be tier 5): the group's remaining sub-chunks may still
///      revisit it (a sub-chunk's union, ~203 per layer, is smaller than the
///      window's ~255), but nothing sooner needs it -- and the replay needs no
///      encoder layer at all;
///   1  the other region (decoder layers during an encoder window, needed by
///      the replay; decoder layers the replay has gone past);
///   2  a layer this pass will still reach (`l >= for_layer`, same region):
///      evicting it turns a coming hit into a re-read (10-01 review: ~90% of an
///      encoder window's untouched decode pages are about to be touched).
/// 0 everywhere else.
fn sweep_rank(enc_first: bool, tier: u8, owner: Option<(u32, u32)>, for_layer: u32) -> u8 {
    match owner {
        Some((l, _)) if enc_first && (tier == 2 || tier == 4) => {
            let split = crate::config::CED_DECODER_START as u32;
            if l >= for_layer && ((l < split) == (for_layer < split)) {
                2
            } else if l < split && l < for_layer {
                0
            } else {
                1
            }
        }
        _ => 0,
    }
}

/// Per-phase counters of mode-aware eviction (logged at each phase switch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ModeEvictCounters {
    pub took_free: u64,
    pub took_stale: u64,
    pub took_decode: u64,
    pub took_own: u64,
    pub took_decode_over: u64,
    pub took_decode_touched: u64,
}

/// Consecutive decode-flagged requests that end a prefill phase.
const ME_DECODE_STREAK: u32 = 16;

/// Mode-aware eviction state (see the block above).
#[derive(Debug, Default)]
struct ModeEvict {
    on: bool,
    /// Decode-flagged requests seen since the last prefill-flagged one.
    decode_streak: u32,
    /// Decode victims one prefill phase may take before preferring its own.
    budget: u64,
    budget_left: u64,
    /// `budget` follows `knobs::prefill_budget` (re-read at each prefill phase
    /// start); off in tests, which pin a budget.
    budget_live: bool,
    /// The current phase is prefill (else decode).
    prefill_phase: bool,
    /// Prefill phases begun (1-based once the first begins).
    phase: u32,
    /// `tick` when the current prefill phase began.
    phase_start: u64,
    /// Per slot: the prefill phase that last HIT this decode-class slot.
    phase_touch: Vec<u32>,
    /// Decode experts evicted by prefill-mode searches `(layer, e, stamp,
    /// prefill phase)`, oldest first, capped (`MODE_EVICT_DELTA_CAP`).
    decode_delta: std::collections::VecDeque<(u32, u32, u64, u32)>,
    c: ModeEvictCounters,
    /// Delta restore (`V41_B2_RESTORE`, needs the mode on).
    restore_on: bool,
    /// Entries to restore `(layer, e, decode stamp)`, NEWEST stamp first.
    restore: std::collections::VecDeque<(u32, u32, u64)>,
    /// Restore reads started and not yet handled (`RESTORE_INFLIGHT` cap).
    restore_inflight: std::collections::HashSet<(u32, u32)>,
    rc: RestoreCounters,
}

/// Delta-restore counters (cumulative; logged at each phase switch).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RestoreCounters {
    pub queued: u64,
    pub pumped: u64,
    pub landed: u64,
    /// Entries found already resident (decode paged them back itself).
    pub skipped: u64,
    /// Restores stopped: no victim decode would evict before the entry.
    pub stopped: u64,
    /// Landings that arrived inside a prefill phase and went back to the queue.
    pub requeued: u64,
}

/// Restore reads started per decode request (`pump_restore`), at most.
const RESTORE_PUMP: usize = 4;
/// Restore reads in flight at once: the restore is a BACKGROUND share of box
/// 2's readers and staging sets, never all of them (decode's own park /
/// early-page reads and speculative admissions keep the rest). It also only
/// starts while more than half of the staging sets are free.
const RESTORE_INFLIGHT: usize = 2;

/// `V41_B2_RESTORE=1` (needs `V41_B2_MODE_EVICT=1`): after each prefill phase,
/// read back the decode experts it evicted, newest first, each landing only over
/// a victim decode itself would evict before it and keeping its own stamp.
pub fn b2_restore() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("V41_B2_RESTORE").as_deref() == Ok("1"));
    *B
}

const MODE_EVICT_DELTA_CAP: usize = 16384;

/// `V41_B2_MODE_EVICT=1`: mode-aware eviction (default OFF).
pub fn b2_mode_evict() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| std::env::var("V41_B2_MODE_EVICT").as_deref() == Ok("1"));
    *B
}

/// The prefill budget now (`knobs::prefill_budget`: env, knobs file, live).
pub fn b2_prefill_budget() -> u64 {
    knobs::prefill_budget()
}

impl ModeEvict {
    fn enabled(n_slots: usize, budget: u64) -> Self {
        Self { on: true, budget, phase_touch: vec![0; n_slots], ..Default::default() }
    }

    /// Tier of an eligible candidate for a PREFILL-mode search (block above).
    fn tier(&self, slot: u32, last_use: u64, free: bool) -> u8 {
        if free {
            0
        } else if last_use < PREFILL_AGE {
            if last_use < self.phase_start { 1 } else { 3 }
        } else if self.phase_touch[slot as usize] == self.phase {
            5
        } else if self.budget_left > 0 {
            2
        } else {
            4
        }
    }
}

/// Prefill-staging counters (cumulative since `enable_paging`; `b2_req`
/// reports deltas per request). All zero with staging off.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageCounters {
    /// Prefill-shaped claims that took a staging slot.
    pub claims: u64,
    /// Hits on an expert resident in staging (any pass).
    pub hits: u64,
    /// Non-prefill claims that found no main-band victim and took a staging
    /// slot instead (a parked prefill chunk's main hits crowding the band).
    pub spill_in: u64,
    /// Prefill claims that found staging full of their own union and took a
    /// main-band victim (only with `STAGE < N_EXPERT`).
    pub spill_out: u64,
    /// Staged background landings (prefill's early-page / park reads) dropped
    /// for want of a staging victim.
    pub drops: u64,
}

/// Which band a victim search may take from (`ShardPool::pick_victim_any`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Band {
    /// `[0, stage)`: decode-shaped and served-inside claims, decode's
    /// background landings. Region first (non-global), then the whole band.
    Main,
    /// `[stage, n)`: prefill-shaped claims and prefill's background landings.
    Stage,
}

// ---------------------------------------------------------------------------
// Pinning: the hub is never surprised by this pool (2026-09-26)
// ---------------------------------------------------------------------------
//
// THE INVARIANT. When box 2 serves a request, every pick the hub treated as
// HELD when it routed and sent that request is resident here: no read, no
// wait. `hub_held ⊆ pinned ⊆ resident`.
//
// Before this, the hub's mirror was box 2's LANDED map as of the layer's
// previous reply (~1 step), while this pool made ~20 local LRU evictions per
// such interval: decode requests needed 2.27 reads per token where the hub
// expected 1.12, and the difference was paging the substitution never got a
// chance to avoid (91% of it experts evicted here earlier). Measured from
// evtrace 2026-09-26.
//
// * PIN on report (`pin_report`): the reply's residency map is the layer's
//   PINNED set, and reporting is what pins: an ELIGIBLE expert (a pick of a
//   decode-shaped request, or a hub prefetch/admission word) that is landed at
//   reply time is pinned if the budget allows -- the pass's own grants first,
//   then the rest of the layer. A grant refused for budget is dropped (the
//   hub sees "not held" and may grant again); one not landed expires after
//   `PIN_GRANT_TTL` reports of its layer. `pinned ⇒ landed` always.
// * NEVER evicted while pinned: every replacement of a slot's occupant goes
//   through `ShardPool::evict` (the choke point), and every victim search
//   through `ShardPool::pick_victim`, which skips pinned slots. Evicting a
//   pinned expert is a bug: counted, logged, fatal under `V41_B2_ASSERT_PINNED`.
// * RELEASE words from the hub (`REQ_FLAG_RELEASE`) unpin, in wire order, when
//   the request carrying them is SERVED (served order = arrival order: a
//   parked request's words were applied before the requests served inside
//   it). The reply's `epoch` counts the words applied, and the hub masks
//   every release it queued after that epoch out of the map (`b2_mirror`),
//   so an old map can never resurrect a released expert.
// * PREFILL-shaped passes (`b > 16`) grant no eligibility: a 1024-row chunk
//   touches most of a layer once, and pinning its scan admissions would fill
//   the budget with experts decode may never pick (and override the two-class
//   LRU that exists to evict them first). They stay unpinned and unreported,
//   which the hub reads as "not held": it may page them, never be surprised.
//
// BUDGET / RESERVE and why nothing can deadlock. `pinned <= budget = n_slots -
// STAGE - R - sum(floor)` at all times; new pins beyond it are refused
// (`denied`) and reported not held. What a victim search must find is at most
// `|W \ resident|` free-able slots, where W is the pass's distinct picks of one
// layer (<= N_EXPERT = 384, a merged prefill union) and the slots it may not
// take are the hub's pins (<= budget), W's own resident experts, a PARKED
// request's picks P (<= 384, one layer: the hub sets `REQ_FLAG_OOO` on every
// request, so a 1024-row prefill chunk parks like any other), and, under
// `V41_B2_POOL_FLOOR`, at most `floor[l]` slots of each foreign layer. A
// request served inside a park has at most `PARK_MAX_ROWS` = 16 rows (W <= 96).
// So with `R >= max(|W| + |P|) = 384 + 96 = 480` (`PIN_RESERVE_MIN`, the default
// without staging) a demand / certain / prefill claim always finds an unpinned
// victim without any hub action (the claimed expert is a miss, so at most
// `|W| - 1` of W is resident). In-flight background reads hold no slot until
// they land; a landing with no unpinned victim is DROPPED (counted), and
// whoever needs that expert demand-reads it, which the reserve covers.
// Early-page `pinned` is only set around `prefetch_words_ex`, which searches
// no victim. If a reserve below the minimum ever leaves a demand claim with no
// unpinned victim, it REVOKES a pin (an error the counters show) rather than
// failing the request: today's behaviour there was "no evictable slot" and an
// outage.
//
// PREFILL STAGING (2026-09-27, `V41_B2_PREFILL_STAGE`, `ShardPool::stage`).
// Measured 09-27 with ~5300-5600 of 6160 slots pinned: prefill unions and
// speculative admits fought over the few unpinned slots, prefill re-read its
// per-layer unions from disk and box 2 sat at 83-87% busy for 2-minute
// stretches. So the LAST `STAGE` slots `[n - STAGE, n)` are a staging band
// with its own rules, all enforced by `pick_victim_any`'s `Band` (the one
// victim search) and by `PinBook::report`:
// * A PREFILL-shaped pass (`prefill_shaped`, b > 16) claims victims ONLY in
//   staging (LRU within it). Its own wanted ids are excluded as always, so
//   with `STAGE >= N_EXPERT >= |W|` a victim always exists: after k claims the
//   band holds at most `hits + k` wanted experts and the next miss still has
//   `STAGE - |W| + 1 >= 1` candidates. A smaller STAGE (tiny test pools) is
//   allowed: a prefill claim that finds staging full of its own union SPILLS
//   into the main band (`stage_spill_out`, the pre-staging behaviour) rather
//   than failing. Prefill's early-page / park reads (`PfJob::stage`) land in
//   staging too, and are dropped when it has no victim. A prefill pass is
//   never served inside a park (`PARK_MAX_ROWS`), so P is empty for it.
// * Every other victim search (decode-shaped and served-inside claims,
//   background landings of decode's reads) stays in the MAIN band
//   `[0, n - STAGE)`. A pick that HITS an expert resident in staging is a hit
//   (the slot map is global), but `PinBook::report` never PINS an expert whose
//   slot is in staging: it is reported not held, decode re-pages it into the
//   main band once prefill evicts it, and the hub pins it there. So
//   `pinned => landed in main`, and prefill's victim search can never run out
//   of unpinned slots.
// * Floors protect the main band only (`pick_victim`): a staging slot holding
//   layer L's expert still counts in `held[L]`, so a layer with experts in
//   staging is floor-protected in main by that much less. Floors default 0.
// * BUDGET with staging: a non-prefill claim searches main first. Main has at
//   least `n - STAGE - budget - floors = R` unpinned, unfloored slots, of
//   which W's resident (<= |W| - 1) and P's main-resident picks are excluded.
//   With a parked DECODE request `|P| <= 96`, so `R >= 2 * PARK_MAX_ROWS *
//   N_EXPERT_USED = 192` (`PIN_RESERVE_STAGED`, the default with staging)
//   keeps every such claim in main. With a parked PREFILL chunk (`|P| <= 384`,
//   mostly resident in staging, but its hits in main are protected too) main
//   can run dry; the claim then SPILLS into staging (`stage_spill_in`: no
//   pins there, only W and P excluded), and the two bands together always
//   hold a victim when `R + STAGE >= |W| + |P| = 480 = PIN_RESERVE_MIN`
//   (candidates(main) + candidates(staging) >= R + STAGE - (|W| - 1) - |P|
//   >= 1). Defaults: STAGE 384 + R 192 = 576 >= 480, budget = 6160 - 576 -
//   floors = 5584. Only below that bound does a claim revoke a pin, as
//   without staging.

/// No-deadlock minimum of `V41_B2_PIN_RESERVE` WITHOUT staging (see the block
/// above), and the minimum of `reserve + STAGE` with it.
pub const PIN_RESERVE_MIN: usize = N_EXPERT as usize + PARK_MAX_ROWS * N_EXPERT_USED;
/// Default `V41_B2_PIN_RESERVE` WITH staging: a served-inside pass (<= 96
/// picks) never spills into staging while a decode-shaped request (<= 96
/// picks) is parked. Below it a claim may spill; the no-deadlock bound is
/// `reserve + STAGE >= PIN_RESERVE_MIN`.
pub const PIN_RESERVE_STAGED: usize = 2 * PARK_MAX_ROWS * N_EXPERT_USED;
/// Most rows a request served INSIDE a parked one may have (the park
/// executor's size); `PIN_RESERVE_MIN` depends on it.
pub const PARK_MAX_ROWS: usize = 16;

/// `V41_B2_PREFILL_STAGE` (default `N_EXPERT` = 384 = one full layer union;
/// 0 = off = one band, the pre-09-27 behaviour): slots at the END of the
/// pool that prefill-shaped passes claim in and nothing else does (the
/// staging block above `PIN_RESERVE_MIN`). Raw value; `ShardPool::set_stage`
/// clamps it to the pool.
pub fn b2_prefill_stage() -> usize {
    static S: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_PREFILL_STAGE").ok().and_then(|v| v.parse().ok()).unwrap_or(N_EXPERT as usize)
    });
    *S
}

/// `V41_B2_PIN_RESERVE`: slots of the MAIN band box 2 never lets the hub pin.
/// Default `PIN_RESERVE_STAGED` (192) with staging, `PIN_RESERVE_MIN` (480)
/// without. `stage` = the pool's staging slots (0 = off). Below the
/// no-deadlock bound (`reserve + stage >= PIN_RESERVE_MIN`) the argument holds
/// only if the workload's per-layer working set fits (tests with tiny pools);
/// logged. Called once per `pin_enable` (a warning per connection is fine).
pub fn b2_pin_reserve(stage: usize) -> usize {
    static R: std::sync::LazyLock<Option<usize>> =
        std::sync::LazyLock::new(|| std::env::var("V41_B2_PIN_RESERVE").ok().and_then(|v| v.parse().ok()));
    let default = if stage > 0 { PIN_RESERVE_STAGED } else { PIN_RESERVE_MIN };
    let r = R.unwrap_or(default);
    if r + stage < PIN_RESERVE_MIN {
        eprintln!(
            "expertd: WARNING V41_B2_PIN_RESERVE={r} + staging {stage} < {PIN_RESERVE_MIN}: a pass whose layer union + parked \
             picks exceed that will REVOKE pins (hub surprises) instead of finding an unpinned victim"
        );
    } else if stage > 0 && r < PIN_RESERVE_STAGED {
        eprintln!(
            "expertd: note V41_B2_PIN_RESERVE={r} < {PIN_RESERVE_STAGED}: a request served inside a parked one may \
             SPILL its claims into the prefill staging band (counted, no surprise)"
        );
    }
    r
}

/// `V41_B2_ASSERT_PINNED=1`: evicting a pinned expert panics (verification
/// runs). Default: counted and logged. A violation costs the hub one read; a
/// panic here costs a cold 116 GB pool and an outage for every agent.
pub fn b2_assert_pinned() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| matches!(std::env::var("V41_B2_ASSERT_PINNED").as_deref(), Ok("1") | Ok("on")));
    *B
}

/// Pin violations since start (all connections), for the stats line.
pub static PIN_VIOLATIONS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A pinned expert left the pool: count, log (rate-limited), and panic under
/// `V41_B2_ASSERT_PINNED` / in debug builds.
fn pin_violation(what: &str, layer: u32, e: u32) {
    let n = PIN_VIOLATIONS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
    if n <= 20 || n.is_power_of_two() {
        eprintln!("expertd: ERROR pin violation #{n}: {what} L{layer} e{e} (the hub treats it as held)");
    }
    if b2_assert_pinned() {
        panic!("pin violation: {what} L{layer} e{e} (V41_B2_ASSERT_PINNED)");
    }
    debug_assert!(false, "pin violation: {what} L{layer} e{e}");
}

/// Per-connection pin counters (cumulative since the connection's first
/// `REQ_FLAG_PIN` request).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinCounters {
    /// Release words applied (the reply `epoch`, not wrapped).
    pub releases: u64,
    /// Experts pinned at a report.
    pub new_pins: u64,
    /// Eligible, landed experts refused for budget (counted once per
    /// eligibility; reported not held).
    pub denied: u64,
    /// Background landings dropped because every victim was pinned (a victim
    /// existed ignoring the pins).
    pub no_victim_drops: u64,
    /// Pinned experts that left the pool: 0 unless something is wrong.
    pub pinned_evictions: u64,
    /// ... of which a demand claim took on purpose (no unpinned victim left:
    /// the reserve is too small for the workload).
    pub revokes: u64,
}

const PIN_NONE: u8 = 0;
const PIN_HELD: u8 = 1;
/// `PIN_ELIGIBLE_LAST ..= PIN_ELIGIBLE_FIRST`: eligible (granted, not yet
/// landed at a report), counting DOWN one per report of its layer that finds
/// it not landed; `PIN_ELIGIBLE_LAST` is the last report it survives. A grant
/// therefore lives `PIN_GRANT_TTL` reports of its layer (~ as many decode
/// steps): long enough for the background read it usually comes with, short
/// enough that a grant whose read was dropped does not pin the expert when a
/// prefill scan happens to land it much later.
const PIN_ELIGIBLE_LAST: u8 = 2;
const PIN_GRANT_TTL: u8 = 8;
const PIN_ELIGIBLE_FIRST: u8 = PIN_ELIGIBLE_LAST + PIN_GRANT_TTL - 1;

/// Box 2's half of the pin contract, per connection (the block comment above).
/// Pure bookkeeping, no device state: `ShardPool` consults it in every victim
/// search and tells it about every eviction.
#[derive(Clone, Debug)]
struct PinBook {
    on: bool,
    budget: u32,
    pinned: u32,
    /// Release words applied on this connection (wrapping; the reply's
    /// `epoch`).
    epoch: u32,
    /// Per `layer * N_EXPERT + e`: `PIN_*`.
    state: Vec<u8>,
    /// Grants since their layer's last report, as `layer << 16 | e`: a
    /// report pins THESE first (the request's own picks, its admissions),
    /// then scans the layer, so an old eligible expert with a low index cannot
    /// take the last budget slot from the pick the pass just used.
    fresh: Vec<u32>,
    c: PinCounters,
}

impl PinBook {
    fn off() -> Self {
        Self { on: false, budget: 0, pinned: 0, epoch: 0, state: Vec::new(), fresh: Vec::new(), c: PinCounters::default() }
    }

    fn enable(&mut self, budget: u32) {
        if !self.on {
            *self = Self {
                on: true,
                budget,
                pinned: 0,
                epoch: 0,
                state: vec![PIN_NONE; N_LAYER as usize * N_EXPERT as usize],
                fresh: Vec::new(),
                c: PinCounters::default(),
            };
        }
    }

    #[inline]
    fn idx(layer: u32, e: u32) -> Option<usize> {
        (layer < N_LAYER as u32 && e < N_EXPERT).then(|| layer as usize * N_EXPERT as usize + e as usize)
    }

    #[inline]
    fn is_pinned(&self, layer: u32, e: u32) -> bool {
        self.on && Self::idx(layer, e).is_some_and(|i| self.state[i] == PIN_HELD)
    }

    /// The hub wants `(layer, e)` resident (a decode-shaped pick, a prefetch
    /// word): pin it at its layer's next report, budget allowing. A repeat
    /// grant restarts the eligibility clock.
    fn grant(&mut self, layer: u32, e: u32) {
        if !self.on {
            return;
        }
        if let Some(i) = Self::idx(layer, e) {
            if self.state[i] != PIN_HELD {
                self.state[i] = PIN_ELIGIBLE_FIRST;
                self.fresh.push((layer << 16) | e);
            }
        }
    }

    /// `report`'s per-expert step. `landed` = in the pool now; `pinnable` =
    /// landed in the MAIN band (a staging slot is never pinned: it is treated
    /// as not landed for eligibility, and reported not held). Pins an
    /// eligible pinnable expert within budget, else drops the grant
    /// (`denied`: a fresh grant is needed to try again); ages an eligible
    /// unpinnable one unless `age` is false (the fresh pass, so the scan ages
    /// each once); reports a pinned expert that is not landed (a violation).
    #[inline]
    fn visit(&mut self, layer: u32, e: usize, landed: bool, pinnable: bool, age: bool) {
        let i = layer as usize * N_EXPERT as usize + e;
        let s = self.state[i];
        if s >= PIN_ELIGIBLE_LAST {
            if landed && pinnable {
                if self.pinned < self.budget {
                    self.state[i] = PIN_HELD;
                    self.pinned += 1;
                    self.c.new_pins += 1;
                } else {
                    self.state[i] = PIN_NONE;
                    self.c.denied += 1;
                }
            } else if age {
                self.state[i] = if s > PIN_ELIGIBLE_LAST { s - 1 } else { PIN_NONE };
            }
        } else if s == PIN_HELD && !landed {
            self.state[i] = PIN_NONE;
            self.pinned -= 1;
            self.c.pinned_evictions += 1;
            pin_violation("pinned but not landed at report", layer, e as u32);
        }
    }

    /// One RELEASE word (in wire order). Every word advances the epoch, known
    /// key or not, so the hub's count of words sent stays comparable.
    fn release(&mut self, w: u32) {
        if !self.on {
            return;
        }
        self.epoch = self.epoch.wrapping_add(1);
        self.c.releases += 1;
        if let Some(i) = Self::idx(w >> 16, w & 0xFFFF) {
            if self.state[i] == PIN_HELD {
                self.pinned -= 1;
            }
            self.state[i] = PIN_NONE;
        }
    }

    /// `(layer, e)` left the pool. Returns whether it was pinned (a
    /// violation, which the caller reports).
    fn on_evict(&mut self, layer: u32, e: u32) -> bool {
        if !self.on {
            return false;
        }
        let Some(i) = Self::idx(layer, e) else { return false };
        let was = self.state[i] == PIN_HELD;
        if was {
            self.pinned -= 1;
            self.c.pinned_evictions += 1;
        }
        self.state[i] = PIN_NONE;
        was
    }

    /// Pin the layer's FRESH grants that are landed now (`row` = the layer's
    /// `remap_hosts`, nonzero = landed, `-(slot) - 1`) in the MAIN band
    /// (slot < `stage`) first, then every other eligible such expert of
    /// `layer`, budget allowing; age the eligible unlanded (or staged) ones;
    /// return the layer's pinned set as a residency map. A pinned expert
    /// found NOT landed is a violation (it left the pool around the choke
    /// point): unpinned, reported, and absent from the map.
    fn report(&mut self, layer: u32, row: &[i32], stage: u32) -> [u32; proto::RESID_WORDS] {
        let mut w = [0u32; proto::RESID_WORDS];
        if !self.on || layer >= N_LAYER as u32 {
            return w;
        }
        // (landed, pinnable): a staging slot is landed but never pinned.
        let state = |e: usize| -> (bool, bool) {
            match row.get(e) {
                Some(&r) if r != 0 => (true, ((-r - 1) as u32) < stage),
                _ => (false, false),
            }
        };
        // The pass's own grants first, in grant order (no aging: the scan
        // below ages each eligible expert exactly once).
        let mut fresh = std::mem::take(&mut self.fresh);
        for &f in &fresh {
            if f >> 16 == layer {
                let e = (f & 0xFFFF) as usize;
                if e < N_EXPERT as usize {
                    let (landed, pinnable) = state(e);
                    self.visit(layer, e, landed, pinnable, false);
                }
            }
        }
        fresh.retain(|f| f >> 16 != layer);
        self.fresh = fresh;
        let base = layer as usize * N_EXPERT as usize;
        for e in 0..N_EXPERT as usize {
            let (landed, pinnable) = state(e);
            self.visit(layer, e, landed, pinnable, true);
            if self.state[base + e] == PIN_HELD {
                w[e / 32] |= 1 << (e % 32);
            }
        }
        w
    }
}

/// Pin mode: per queued request (by seq), the picks NOT landed when the
/// compute thread first SAW its frame (the early-page hook's `pull`, the merge
/// look-ahead, or the dequeue itself, always before that frame's own reads).
/// OR-ed into the reply's PAGED bits, so a read that hook (or a park) finished
/// before the pass started still counts as "box 2 had to page it" for the
/// hub's surprise check. Not quite the wire arrival: a frame that lands in the
/// socket while a pass is in `ensure` is first seen after that pass's
/// background admissions, so a violation those landings happen to cover is
/// not counted -- the check errs only towards silence, never a false
/// surprise (a held pick is pinned, hence landed, at every instant before its
/// own words are applied). A bounded ring: the reader keeps at most a few
/// frames queued, and a dropped entry only weakens the check.
#[derive(Clone, Debug, Default)]
pub struct EarlyPaged {
    ring: Vec<(u32, [u32; proto::RESID_WORDS])>,
}

impl EarlyPaged {
    /// Most queued frames whose arrival-time bits are kept.
    pub const MAX: usize = 32;

    /// Frame `seq` arrived with `bits` not landed (all-zero bits keep nothing).
    /// Beyond `MAX` entries the OLDEST is dropped.
    pub fn note(&mut self, seq: u32, bits: [u32; proto::RESID_WORDS]) {
        if bits.iter().all(|&w| w == 0) {
            return;
        }
        if self.ring.len() >= Self::MAX {
            self.ring.remove(0);
        }
        self.ring.push((seq, bits));
    }

    /// The arrival-time bits of request `seq` (all zero if none were kept),
    /// consumed.
    pub fn take(&mut self, seq: u32) -> [u32; proto::RESID_WORDS] {
        match self.ring.iter().position(|(s, _)| *s == seq) {
            Some(i) => self.ring.swap_remove(i).1,
            None => [0; proto::RESID_WORDS],
        }
    }

    pub fn clear(&mut self) {
        self.ring.clear();
    }

    pub fn len(&self) -> usize {
        self.ring.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }
}

/// `(layer, e)` pairs a victim search must not take besides the hub's pins
/// and the pass's own wanted ids: a parked request's picks (`parked_pins`)
/// and the early-page `pinned` set.
type ExtraPins<'a> = &'a [(u32, u32)];

impl ShardPool {
    /// A pool of `n_slots`, seeded with `(layer, base_slot, ids)` regions as
    /// `load` placed them (slot `base + i` holds `ids[i]`, all landed, oldest
    /// first), each layer keeping `floor_frac` of its region under global
    /// eviction. Pins off.
    fn seeded(n_slots: usize, layers: &[(u32, u32, &[u32])], floor_frac: f32) -> Self {
        let mut owner_of: Vec<Option<(u32, u32)>> = vec![None; n_slots];
        let mut slot_of = std::collections::HashMap::with_capacity(n_slots);
        let mut last_use = vec![0u64; n_slots];
        let mut tick = 0u64;
        let mut remap_hosts = vec![vec![0i32; REMAP_LEN]; N_LAYER as usize];
        let mut held = vec![0u32; N_LAYER as usize];
        let mut floor = vec![0u32; N_LAYER as usize];
        for &(li, base, ids) in layers {
            for (local, &e) in ids.iter().enumerate() {
                let abs = base + local as u32;
                owner_of[abs as usize] = Some((li, e));
                slot_of.insert((li, e), abs);
                tick += 1;
                last_use[abs as usize] = tick;
                remap_hosts[li as usize][e as usize] = -(abs as i32) - 1;
            }
            held[li as usize] = ids.len() as u32;
            floor[li as usize] = (ids.len() as f32 * floor_frac) as u32;
        }
        Self {
            owner_of,
            slot_of,
            last_use,
            tick,
            remap_hosts,
            dirty: vec![false; N_LAYER as usize],
            held,
            floor,
            pins: PinBook::off(),
            stage: n_slots as u32,
            sc: StageCounters::default(),
            me: ModeEvict::default(),
        }
    }

    /// Turn mode-aware eviction on (load time; also tests).
    fn enable_mode_evict(&mut self, budget: u64) {
        self.me = ModeEvict::enabled(self.owner_of.len(), budget);
    }

    /// `enable_mode_evict` with the budget following the live knob
    /// (`knobs::prefill_budget`, re-read at each prefill phase start).
    fn enable_mode_evict_live(&mut self) {
        self.enable_mode_evict(b2_prefill_budget());
        self.me.budget_live = true;
    }

    /// Note the HUB's phase of the request about to be served (`prefill` =
    /// no `REQ_FLAG_DECODE`), before it claims or lands anything. A prefill
    /// request starts a prefill phase at once; a prefill phase ends after
    /// `ME_DECODE_STREAK` consecutive decode requests.
    fn me_note_request(&mut self, prefill: bool) {
        if !self.me.on {
            return;
        }
        if prefill {
            self.me.decode_streak = 0;
            if !self.me.prefill_phase {
                self.me_switch(true);
            }
        } else if self.me.prefill_phase {
            self.me.decode_streak += 1;
            if self.me.decode_streak >= ME_DECODE_STREAK {
                self.me_switch(false);
            }
        }
    }

    /// A phase switch: log the ending prefill phase, start the new one (a
    /// prefill phase gets a fresh budget and stamps its start).
    fn me_switch(&mut self, prefill: bool) {
        let c = self.me.c;
        if self.me.prefill_phase {
            eprintln!(
                "expertd: mode-evict prefill phase {} ended: victims free {} stale-prefill {} decode {} (of which in use {}; budget {}, {} left) own-prefill {} decode-over-budget {}; decode delta {}",
                self.me.phase, c.took_free, c.took_stale, c.took_decode, c.took_decode_touched, self.me.budget,
                self.me.budget_left, c.took_own, c.took_decode_over, self.me.decode_delta.len()
            );
        }
        if self.me.restore_on && !self.me.prefill_phase && self.me.phase > 0 {
            let r = self.me.rc;
            eprintln!(
                "expertd: mode-evict restore after phase {}: queued {} pumped {} landed {} skipped {} stopped {} requeued {}; {} left",
                self.me.phase, r.queued, r.pumped, r.landed, r.skipped, r.stopped, r.requeued, self.me.restore.len()
            );
        }
        self.me.prefill_phase = prefill;
        self.me.decode_streak = 0;
        self.me.c = ModeEvictCounters::default();
        if prefill {
            self.me.phase += 1;
            self.me.phase_start = self.tick;
            if self.me.budget_live {
                self.me.budget = b2_prefill_budget();
            }
            self.me.budget_left = self.me.budget;
        } else if self.me.restore_on {
            self.me_build_restore();
        }
    }

    /// Prefill -> decode: merge the delta into the restore queue (whatever an
    /// earlier restore left too): one entry per expert at its newest stamp,
    /// minus the resident ones, newest first.
    fn me_build_restore(&mut self) {
        let mut best: std::collections::HashMap<(u32, u32), u64> = std::collections::HashMap::new();
        for &(l, e, t) in self.me.restore.iter() {
            let v = best.entry((l, e)).or_insert(t);
            *v = (*v).max(t);
        }
        for &(l, e, t, _) in self.me.decode_delta.iter() {
            let v = best.entry((l, e)).or_insert(t);
            *v = (*v).max(t);
        }
        self.me.decode_delta.clear();
        let mut ents: Vec<(u32, u32, u64)> = best.into_iter().filter(|(k, _)| !self.slot_of.contains_key(k)).map(|((l, e), t)| (l, e, t)).collect();
        ents.sort_by(|a, b| b.2.cmp(&a.2).then(a.0.cmp(&b.0)).then(a.1.cmp(&b.1)));
        self.me.rc = RestoreCounters { queued: ents.len() as u64, ..Default::default() };
        self.me.restore = ents.into_iter().collect();
        // A completion that never came back must not hold the throttle forever;
        // a late one is harmless (removing an absent key).
        self.me.restore_inflight.clear();
    }

    /// Delta restore: the slot a restore of an expert last stamped `stamp` may
    /// land in -- the plain LRU's victim, provided decode would evict it before
    /// that expert (free, prefill-class, or an older decode stamp). `None` =
    /// nothing is older: this entry and every later (older) one would only be
    /// evicted first, so the restore stops.
    #[allow(clippy::too_many_arguments)]
    fn restore_victim(
        &self,
        region: (u32, u32),
        global: bool,
        want_layer: u32,
        want: &[u32],
        extra: ExtraPins<'_>,
        for_layer: u32,
        stamp: u64,
    ) -> Option<u32> {
        let v = self.pick_victim_any(region, global, Band::Main, want_layer, want, extra, for_layer, false, false)?;
        let lu = self.last_use[v as usize];
        (self.owner_of[v as usize].is_none() || lu < PREFILL_AGE || lu < stamp).then_some(v)
    }

    /// `land`, keeping the expert's own (decode) stamp: a delta restore.
    fn land_stamped(&mut self, slot: u32, key: (u32, u32), stamp: u64) {
        self.land(slot, key, false);
        self.last_use[slot as usize] = stamp;
    }

    /// Mode-aware eviction applies to this search: on, in a prefill phase, and
    /// the search is a prefill claim / landing.
    fn me_tiered(&self, prefill_mode: bool) -> bool {
        prefill_mode && self.me.on && self.me.prefill_phase
    }

    /// Account the victim a search in `prefill_mode` is about to evict (call
    /// BEFORE `evict`, which clears the owner): budget, counters, the delta.
    fn me_account(&mut self, slot: u32, prefill_mode: bool) {
        if !self.me_tiered(prefill_mode) {
            return;
        }
        let lu = self.last_use[slot as usize];
        let Some((l, e)) = self.owner_of[slot as usize] else {
            self.me.c.took_free += 1;
            return;
        };
        if lu >= PREFILL_AGE {
            if self.me.phase_touch[slot as usize] == self.me.phase {
                self.me.c.took_decode_touched += 1;
            }
            if self.me.budget_left > 0 {
                self.me.budget_left -= 1;
                self.me.c.took_decode += 1;
            } else {
                self.me.c.took_decode_over += 1;
            }
            if self.me.decode_delta.len() >= MODE_EVICT_DELTA_CAP {
                self.me.decode_delta.pop_front();
            }
            self.me.decode_delta.push_back((l, e, lu, self.me.phase));
        } else if lu < self.me.phase_start {
            self.me.c.took_stale += 1;
        } else {
            self.me.c.took_own += 1;
        }
    }

    /// Reserve the LAST `want` slots as the prefill staging band (0 = off).
    /// Clamped to half the pool (a decode claim must have a main band);
    /// below `N_EXPERT` a prefill union may not fit and spills (logged).
    /// Returns the staging slots actually reserved. Slot CONTENTS are
    /// untouched: whatever `load` placed there stays until evicted.
    fn set_stage(&mut self, want: usize) -> usize {
        let n = self.owner_of.len();
        let mut stage = want.min(n / 2);
        if stage != want {
            eprintln!("expertd: V41_B2_PREFILL_STAGE={want} clamped to {stage} (half of the {n}-slot pool)");
        }
        if stage > 0 && stage < N_EXPERT as usize {
            eprintln!(
                "expertd: note prefill staging {stage} < N_EXPERT {N_EXPERT}: a layer union larger than the band \
                 spills into the main band (counted as stage_spill_out)"
            );
        }
        if want == 0 {
            stage = 0;
        }
        self.stage = (n - stage) as u32;
        stage
    }

    /// Staging slots (0 = off).
    fn stage_slots(&self) -> usize {
        self.owner_of.len() - self.stage as usize
    }

    /// THE victim search: the least recently used slot in `range` that is
    /// free, or whose occupant is not one of `want` on `want_layer`, not in
    /// `extra`, not pinned by the hub (unless `ignore_hub_pins`), and not a
    /// foreign layer at its floor (`for_layer` is the layer taking the slot;
    /// its own slots are never floor-protected; staging slots are never
    /// floor-protected either -- floors guard the main band).
    #[allow(clippy::too_many_arguments)]
    fn pick_victim(
        &self,
        range: std::ops::Range<u32>,
        want_layer: u32,
        want: &[u32],
        extra: ExtraPins<'_>,
        for_layer: u32,
        ignore_hub_pins: bool,
        prefill_mode: bool,
    ) -> Option<u32> {
        let n = self.owner_of.len() as u32;
        let range = range.start.min(n)..range.end.min(n);
        // Mode-aware eviction: a prefill-mode search in a prefill phase ranks by
        // (tier, sweep position within the decode tiers, recency).
        let tiered = self.me_tiered(prefill_mode);
        let enc_first = tiered && knobs::encoder_victims_first();
        let mut best: Option<((u8, u8, u64), u32)> = None;
        for sl in range {
            let ok = match self.owner_of[sl as usize] {
                Some((ol, oe)) => {
                    if (ol == want_layer && want.contains(&oe))
                        || extra.contains(&(ol, oe))
                        || (!ignore_hub_pins && self.pins.is_pinned(ol, oe))
                    {
                        false
                    } else {
                        // Never take a foreign layer below its floor (main band).
                        ol == for_layer || sl >= self.stage || self.held[ol as usize] > self.floor[ol as usize]
                    }
                }
                None => true,
            };
            if !ok {
                continue;
            }
            let t = self.last_use[sl as usize];
            let key = if tiered {
                let tier = self.me.tier(sl, t, self.owner_of[sl as usize].is_none());
                (tier, sweep_rank(enc_first, tier, self.owner_of[sl as usize], for_layer), t)
            } else {
                (0, 0, t)
            };
            if best.is_none_or(|(bk, _)| key < bk) {
                best = Some((key, sl));
            }
        }
        best.map(|(_, sl)| sl)
    }

    /// `Band::Main`: the region (clipped to the main band) first, then the
    /// whole main band (`global` skips the region). `Band::Stage`: the staging
    /// band only. With staging off `Main` is the whole pool and `Stage` is
    /// empty (and never asked for: every staged path checks `stage_slots`).
    #[allow(clippy::too_many_arguments)]
    fn pick_victim_any(
        &self,
        region: (u32, u32),
        global: bool,
        band: Band,
        want_layer: u32,
        want: &[u32],
        extra: ExtraPins<'_>,
        for_layer: u32,
        ignore_hub_pins: bool,
        prefill_mode: bool,
    ) -> Option<u32> {
        let n = self.owner_of.len() as u32;
        match band {
            Band::Main => {
                let first = if global { 0..self.stage } else { region.0.min(self.stage)..region.1.min(self.stage) };
                self.pick_victim(first, want_layer, want, extra, for_layer, ignore_hub_pins, prefill_mode)
                    .or_else(|| self.pick_victim(0..self.stage, want_layer, want, extra, for_layer, ignore_hub_pins, prefill_mode))
            }
            Band::Stage => self.pick_victim(self.stage..n, want_layer, want, extra, for_layer, ignore_hub_pins, prefill_mode),
        }
    }

    /// THE CHOKE POINT: detach whoever holds `slot` (possibly another layer,
    /// whose device remap is then stale until its next `ensure`) and return
    /// it. Every replacement of a slot's occupant goes through here. Evicting a
    /// hub-pinned expert is a violation of the pin invariant (reported).
    fn evict(&mut self, slot: u32, cur_layer: u32) -> Option<(u32, u32)> {
        let (ol, oe) = self.owner_of[slot as usize].take()?;
        self.slot_of.remove(&(ol, oe));
        self.remap_hosts[ol as usize][oe as usize] = 0;
        self.held[ol as usize] -= 1;
        if ol != cur_layer {
            self.dirty[ol as usize] = true;
        }
        if self.pins.on_evict(ol, oe) {
            pin_violation("evicted", ol, oe);
        }
        Some((ol, oe))
    }

    /// `ensure`: is `(layer, e)` resident? If so, touch its recency (a prefill
    /// hit refreshes only within the prefill class, a decode hit promotes).
    fn touch_hit(&mut self, layer: u32, e: u32, scan_class: bool) -> bool {
        let Some(&slot) = self.slot_of.get(&(layer, e)) else { return false };
        self.tick += 1;
        let staged = slot >= self.stage;
        self.sc.hits += u64::from(staged);
        let lu = &mut self.last_use[slot as usize];
        // A STAGED slot stays prefill-class whoever hits it: staging is an LRU
        // among prefill-class entries, so a decode-class stamp there would
        // make the slot un-evictable while any prefill-class candidate exists
        // -- an expert seeded or spilled into the band would never recycle
        // into main, never pin, and read as "not held" on every reply.
        if scan_class || staged {
            if *lu < PREFILL_AGE {
                *lu = self.tick;
            } else if self.me.on && self.me.prefill_phase && scan_class {
                // Mode-aware eviction: this prefill phase is using decode's slot;
                // its own later claims take it last (tier 5).
                self.me.phase_touch[slot as usize] = self.me.phase;
            }
        } else {
            *lu = self.tick + PREFILL_AGE;
        }
        true
    }

    /// `ensure`: claim a slot for the miss `(layer, e)` of a pass wanting
    /// `want`. Victim, least-recently-used first, never one of `want` on this
    /// layer, never `extra`, never hub-pinned; a PREFILL-shaped pass searches
    /// the staging band, any other the main band, region first (a prefill
    /// sweep should not evict its neighbours as a matter of course), but the
    /// region is a PREFERENCE, not a bound: a layer whose union exceeds its
    /// 154-slot share used to die with "no evictable slot" while thousands of
    /// slots sat evictable elsewhere. A band with no victim SPILLS into the
    /// other (counted; the pinning block says when that can happen), and
    /// with no unpinned victim anywhere (a reserve below the bound) a pin is
    /// REVOKED rather than failing the request (reported by the choke point).
    /// The slot is claimed NOW (so the next pick cannot choose it again); its
    /// remap entry is written only once the data has landed (`commit`).
    /// Returns the slot and whoever was evicted from it.
    #[allow(clippy::too_many_arguments)]
    fn claim_miss(
        &mut self,
        layer: u32,
        e: u32,
        want: &[u32],
        extra: ExtraPins<'_>,
        region: (u32, u32),
        global: bool,
        prefill_shaped: bool,
        scan_class: bool,
    ) -> Option<(u32, Option<(u32, u32)>)> {
        let staged = prefill_shaped && self.stage_slots() > 0;
        let (own, other) = if staged { (Band::Stage, Band::Main) } else { (Band::Main, Band::Stage) };
        let victim = match self.pick_victim_any(region, global, own, layer, want, extra, layer, false, prefill_shaped) {
            Some(v) => {
                self.sc.claims += u64::from(staged);
                v
            }
            None => match (self.stage_slots() > 0).then(|| self.pick_victim_any(region, global, other, layer, want, extra, layer, false, prefill_shaped)).flatten() {
                Some(v) => {
                    if staged { self.sc.spill_out += 1 } else { self.sc.spill_in += 1 }
                    v
                }
                None if self.pins.on => {
                    let v = self.pick_victim_any(region, global, Band::Main, layer, want, extra, layer, true, prefill_shaped)?;
                    self.pins.c.revokes += 1;
                    v
                }
                None => return None,
            },
        };
        self.me_account(victim, prefill_shaped);
        let evicted = self.evict(victim, layer);
        self.owner_of[victim as usize] = Some((layer, e));
        self.slot_of.insert((layer, e), victim);
        self.held[layer as usize] += 1;
        self.tick += 1;
        // A victim in staging (a prefill claim, or a spill-in) is stamped
        // prefill-class whatever the pass: see `touch_hit`.
        self.last_use[victim as usize] = if scan_class || victim >= self.stage { self.tick } else { self.tick + PREFILL_AGE };
        Some((victim, evicted))
    }

    /// A claimed slot's data has landed: map it.
    fn commit(&mut self, layer: u32, e: u32, slot: u32) {
        self.remap_hosts[layer as usize][e as usize] = -(slot as i32) - 1;
    }

    /// A background read of `key` has been repacked into `slot` (already
    /// detached by `evict`): own, map and age it. Background admissions are
    /// stamped as decode-class, like the reads that asked for them, unless
    /// `prefill` (a prefill chunk's early-page / park read: prefill-class,
    /// like its demand claims).
    fn land(&mut self, slot: u32, key: (u32, u32), prefill: bool) {
        self.owner_of[slot as usize] = Some(key);
        self.slot_of.insert(key, slot);
        self.held[key.0 as usize] += 1;
        self.tick += 1;
        self.last_use[slot as usize] = if prefill { self.tick } else { self.tick + PREFILL_AGE };
        self.remap_hosts[key.0 as usize][key.1 as usize] = -(slot as i32) - 1;
        self.dirty[key.0 as usize] = true;
    }

    /// Roll back a claim whose data never landed: free and age the slot.
    /// Never pinned (a pin needs a landed slot at report time).
    fn unclaim(&mut self, layer: u32, e: u32, slot: u32) {
        debug_assert!(!self.pins.is_pinned(layer, e), "a claim is never pinned");
        self.owner_of[slot as usize] = None;
        self.slot_of.remove(&(layer, e));
        self.held[layer as usize] -= 1;
        self.last_use[slot as usize] = 0;
    }
}

/// Recency offset of decode-class slots over prefill-class ones (the two-class
/// LRU, `ensure_layer_inner`).
const PREFILL_AGE: u64 = 1u64 << 40;

/// ONE pool across all layers -- the victim search is a single global LRU and the
/// per-layer regions are only where a layer's experts happen to be loaded, not a
/// residency boundary. `V41_B2_GLOBAL_POOL=0` restores the per-layer search
/// (region preferred, whole pool as fallback), which is what shipped before
/// 2026-09-18 and is useful for A/B.
///
/// Prefill used to be excluded from this on the theory that a layer's sweep would
/// evict its neighbours. That theory cost an outage: a union larger than one
/// layer's 154-slot share had no legal victim and failed the request outright
/// while thousands of slots sat evictable elsewhere.
pub fn b2_global_pool() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_GLOBAL_POOL").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

/// Fraction of its own region a layer is guaranteed to keep, even when a decode
/// request is evicting globally (`V41_B2_POOL_FLOOR`, default 0.90).
///
/// **An earlier note here called floor 0 numerically unsound. THAT WAS WRONG and
/// is retracted.** Floor 0 did produce non-deterministic output, but bisection
/// showed the corruption was `b2_coalesce` (the two-pread expert read), which
/// only misbehaves when page-ins are frequent — i.e. exactly what floor 0
/// causes. With coalescing off, floor 0.00 is BIT-IDENTICAL to floor 0.90
/// (sha 13af380180431910, len 525, three runs each) and materially faster:
///
///     floor 0.90, coalescing off   115-122 ms/tok   hit 0.9433
///     floor 0.00, coalescing off    76-86  ms/tok   hit 0.9726   <- correct AND fast
///
/// So the floor's cost is PREFILL, as originally documented, and nothing else.
/// Verify it with a determinism check anyway; that is what caught the real bug.
///
/// Without a floor the global pool leaks into PREFILL. The phase guard stops a
/// prefill sweep from evicting other layers, but it does not stop DECODE from
/// having already scattered the encoder residency prefill then has to re-page.
///
/// MEASURED frontier (decode 512 tok n=3, prefill 7208 tok cold/warm):
///
///     floor      decode          miss/tok   prefill warm
///     per-layer  4.44             18.50       496
///     0.90       4.92  (+11%)     14.15       526   <- free
///     0.78       5.41  (+22%)     11.26       426   (-14%)
///     none       5.57  (+25%)     10.44       393   (-21%)
///
/// 0.90 protects 234 of an encoder layer's 260 slots — comfortably above the
/// ~203-expert prefill union at B=1024 — and is the only point on the frontier
/// that costs prefill NOTHING while still letting the 20 decoder layers (68
/// slots each, source of 92% of decode misses) borrow the surplus. Lower it only
/// if prefill throughput is expendable.
pub fn b2_pool_floor() -> f32 {
    static F: std::sync::LazyLock<f32> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_POOL_FLOOR")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .map(|v| v.clamp(0.0, 1.0))
            // DEFAULT 0, not 0.90. The measurement in this very doc-comment says
            // floor 0.00 is BIT-IDENTICAL to 0.90 with coalescing off and runs
            // 76-86 ms/tok against 115-122 (hit 0.9726 vs 0.9433) -- the floor's
            // only cost is prefill residency. It stayed at 0.90 because the
            // corruption that motivated it was traced to COALESCING, which is
            // now default off; the floor was never the fix.
            .unwrap_or(0.0)
    });
    *F
}

/// Permute MXFP4 HF->ggml on box 2's iGPU instead of its CPU
/// (`V41_B2_GPU_REPACK=0` reverts to the CPU path). Separate from box 1's
/// `V41_PAGER_GPU_REPACK` so the two sides can be A/B'd independently.
pub fn b2_gpu_repack() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_GPU_REPACK").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

/// Send `REQ_FLAG_BATCHED` on every multi-token remote submit, so a DSpark
/// verify batch takes box 2's by-expert chain instead of its decode chain
/// (`V41_REMOTE_BATCHED_MULTI=0` reverts). Default ON.
///
/// The flag has existed since the protocol was written — its own doc says it is
/// there "to let the hub choose per request (DSpark verify batches)" — but the
/// hub never set it, so every submit with `b <= decode_max_b` (4) took the
/// decode chain by default.
///
/// That chain does not group tokens by expert: it re-reads an expert's weights
/// once per token. MEASURED with the expert set held CONSTANT, so the only
/// variable is the path (`deepstrix-expert-bench --pool 3`, srv p50 us/layer):
///
///     B          1     2     4     5     6     8
///     batched  386   396   444   465   487   527     +20 us per extra token
///     decode   388   738  1387    --    --    --    +333 us per extra token
///
/// 16x on the per-token term, and 3.1x end to end at B=4. `decode_max_b=4` means
/// B>=5 already escapes onto the good path by accident; this makes B=2..4
/// deliberate, which is what a shorter draft window needs.
///
/// B=1 is deliberately left on the decode chain. Batched is marginally faster
/// there too (383 vs 403 us with a realistic pool) but that is 0.8 ms of a 246 ms
/// token, and the two chains sum per-expert partials in a different order, so
/// switching B=1 would perturb today's decode numerics for ~0.3%. Not worth it.
pub fn remote_batched_multi() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_REMOTE_BATCHED_MULTI").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

/// Zero-copy O_DIRECT expert reads on box 2. **OFF by default — MEASURED LOSS.**
/// `V41_B2_ODIRECT=1` enables it. Requires `b2_gpu_repack`, since only the
/// HF-layout reader can land bytes straight in staging.
///
/// This is the THIRD independent rejection of O_DIRECT on this path:
///
///   box 1, `read_range_into_direct` (bouncing)      decode -16%
///   box 2, `read_range_into_direct` (bouncing)      miss 6.60 -> 6.99 ms
///   box 2, `read_range_into_direct_padded` (no copy) miss 6.60 -> 6.75 ms
///
/// The first two were blamed on the bounce buffer — an 18.8 MB aligned alloc
/// plus an 18.8 MB memcpy per fault. Removing it entirely (this path lands bytes
/// straight in GTT staging at the file offset's own 4096-residue) recovered only
/// 0.24 of the 0.39 ms, so the bounce was NOT the main cost.
///
/// What the raw-drive numbers actually say, at the miss's shape:
///
///   3 threads x 6.3 MB  buffered  5.89 ms   O_DIRECT  4.84 ms   box 2
///   3 threads x 6.3 MB  buffered  4.10 ms   O_DIRECT  4.86 ms   box 1
///
/// O_DIRECT wins on box 2 and LOSES on box 1, and in-engine it loses on both.
/// The gap is concurrency: the cached reader splits each tensor across
/// `expert_pread_threads()` preads, while this path issues one pread per tensor
/// — and a role's 0.37 MB scale plane is a poor O_DIRECT read. Anyone retrying
/// this must split the direct reads the same way first; without that the drive's
/// 1.05 ms advantage does not survive contact with the access pattern.
///
/// The code is kept because `read_range_into_direct_padded` is strictly better
/// than the bouncing reader and the measurement is worth preserving.
/// Zero-copy O_DIRECT expert page-ins on box 2. DEFAULT ON since 2026-09-15;
/// `V41_B2_ODIRECT=0` reverts to the buffered path.
///
/// O_DIRECT was rejected TWICE before, and correctly: that was
/// `read_range_into_direct`, which allocates an aligned bounce of the whole
/// extent and memcpys out of it — an 18.8 MB alloc plus an 18.8 MB copy per
/// miss, which cost more than the drive saved. `read_range_into_direct_padded`
/// removes both by exploiting the fact that O_DIRECT only needs the offset, the
/// address and the length to SHARE an alignment, not to be aligned to zero.
///
/// Why it wins here: box 2's expert file is 101 GB and the box has ~5 GB of page
/// cache (123 GB of it is the expert pool), so under 5% can ever be cached and
/// the copy through the cache costs more than the hits save.
///
/// MEASURED on box 2's own page stats, which are a PER-MISS cost and therefore
/// immune to the LRU-warming confound that dominates end-to-end decode A/Bs:
///
///     buffered   ms_per_miss 8.04   (read 7.70  h2d 0.34  repack_gpu 0.32)
///     O_DIRECT   ms_per_miss 7.00   (read 6.06  h2d 0.16  repack_gpu 0.13)
///                             6.22 when comparatively idle
///
/// 13% per miss under load, ~23% idle. At the measured ~20 misses/token that is
/// ~20-36 ms/token. The end-to-end decode delta is BELOW this cluster's
/// measurement resolution (box 2's hit rate drifts more than that between
/// runs), so this is shipped on the mechanism counter, not on a tok/s number.
///
/// Degrades safely: requires GPU repack AND 4096-aligned pinned staging, and
/// falls back to the buffered path when the filesystem refuses the flag.
/// Read all three roles of an expert in TWO preads instead of six.
///
/// **DEFAULT OFF — THIS CORRUPTS UNDER HEAVY EVICTION.** It was defaulted ON on
/// 2026-09-15 and reverted the same day. MEASURED with `V41_T2_CATCHALL=2`
/// (constant partition), same prompt, temperature 0:
///
///     pool floor 0.90, coalescing ON    sha 13af380180431910 (525)  stable
///     pool floor 0.00, coalescing ON    sha 9eee5355594bd024 (517)
///                                       sha 5983106de8523886 (521)  DIFFERS
///     pool floor 0.00, coalescing OFF   sha 13af380180431910 (525)  x3 stable
///
/// The bug only fires when page-ins are FREQUENT: at floor 0.90 evictions are
/// rare, which is why the original bit-identical validation passed — it was run
/// in the regime where this path almost never executes. Validate any change to
/// the read path at floor 0.00, where a decode request pages constantly.
///
/// The two-pread idea is sound (miss 7.11 -> 4.54 ms) and the layout claim holds
/// (an expert's three weight planes are contiguous, and its three scale planes);
/// something in the staging reuse or offset math is wrong under repeated
/// page-ins. Root cause NOT yet found.
///
/// The checkpoint is EXPERT-MAJOR: for every expert the three weight planes are
/// byte-contiguous (3 x 5.625 = 16.875 MB) and so are its three scale planes
/// (3 x 0.352 = 1.055 MB), with the two runs far apart. The per-role path issues
/// SIX preads (~5.6 MB + 0.35 MB each) across three threads and measures
/// 2.96 GB/s aggregate; this drive does 4.47 GB/s on one ~20 MB O_DIRECT read.
/// NVMe strongly prefers one large read.
///
/// MEASURED on box 2's own page stats (a PER-MISS cost, so immune to the
/// LRU-warming confound that dominates end-to-end decode A/Bs):
///
///     buffered            ms_per_miss 8.04   read 7.70   pread 22.13
///     O_DIRECT per-role                7.11        6.94        17.61
///     O_DIRECT coalesced               4.54        4.41         4.41
///
/// 44% cheaper per miss than the original path. `pread == read` is the signature
/// that it is actually engaged: the sum across threads collapses to the wall
/// because it is now two sequential preads instead of six parallel ones.
///
/// VALIDATED BIT-IDENTICAL: same prompt at temperature 0 with
/// `V41_T2_CATCHALL=2`, coalesced vs per-role, produced the same 525-char
/// generation (sha 13af380180431910) twice each.
///
/// Role order is NOT physical order — the loader maps gate<-w1, up<-w3, down<-w2
/// — so `read_expert_runs_direct` derives each role's slot in the run from its
/// FILE OFFSET. Encoding the mapping by hand instead swapped up and down and
/// made generation non-deterministic, which is how this was caught. Per-role
/// GEOMETRY also differs (gate/up are [N_FF_EXP, ...], down is [N_EMBD, ...]);
/// only the byte lengths are uniform, which is what makes one run sliceable.
/// Contiguity and uniform length are CHECKED per expert, falling back to the
/// per-role path rather than trusting the layout.
/// `V41_B2_COALESCE_CHECK=1`: byte-compare every coalesced expert read against
/// the per-role read. Diagnostic only; costs an extra full read per page-in.
pub fn coalesce_check() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_B2_COALESCE_CHECK").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

/// `V41_B2_DECODE_DOWN=1`: batched branch runs the DECODE down kernel per token.
/// Hits-first on the batched path: launch the resident experts, read the misses
/// while they run, launch the missed ones as a second pass into their own partial
/// slots (`MoeExecutor::run_path`). Default OFF; `V41_B2_HITS_FIRST=1` turns it on
/// at start, and `SIGUSR1` flips it at RUNTIME so it can be A/B'd on a warm pool
/// without a restart (a cold restart costs ~116 GB of refills and confounds any
/// comparison). MEASURED 2026-09-21 against a paged 200-slot test daemon: bit-
/// identical to the single-pass path at B=1/4/32 while faulting; timing neutral
/// in a catch-all regime where ~150 serial disk reads dwarf ~4 ms of compute
/// (269 vs 274 ms at B=8, 1045 vs 1175 at B=32, p90 spread larger than the delta).
/// The predicted gain is in the production regime (a few misses per layer against
/// several ms of compute, MULTISTREAM_DECODE_PLAN.md 4.1) — unmeasured there.
static HITS_FIRST: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static HITS_FIRST_INIT: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// Look-ahead prefetch words `(layer << 16) | expert` queued by the batched
/// driver for the NEXT layer (box-2-owned ids of its predicted picks) and
/// drained into the next `submit`.
static PREFETCH_WORDS: std::sync::Mutex<Vec<u32>> = std::sync::Mutex::new(Vec::new());

/// False (nothing queued) when 4096 words are already waiting.
pub fn push_prefetch_words(words: &[u32]) -> bool {
    let mut g = PREFETCH_WORDS.lock().unwrap();
    if g.len() < 4096 {
        g.extend_from_slice(words);
        true
    } else {
        false
    }
}

/// RESTORE words (`b2_mirror::pin_enter_decode`): experts released to open a
/// prefill band, sent back to box 2 as admission words when decode resumes.
/// A separate queue from `PREFETCH_WORDS` so they never go ahead of the cache
/// prior's admissions or the look-ahead words: a request carries at most
/// `b2_mirror::pin_restore_per_request` of them, in the room those leave, and
/// none while a release word is still queued (see `submit_inner`).
static RESTORE_WORDS: std::sync::Mutex<std::collections::VecDeque<u32>> =
    std::sync::Mutex::new(std::collections::VecDeque::new());

/// Most restore words waiting at once.
pub const RESTORE_WORDS_MAX: usize = 8192;

/// Queue restore words (in order); returns how many fit under `RESTORE_WORDS_MAX`.
pub fn push_restore_words(words: &[u32]) -> usize {
    let mut g = RESTORE_WORDS.lock().unwrap_or_else(|p| p.into_inner());
    let n = words.len().min(RESTORE_WORDS_MAX.saturating_sub(g.len()));
    g.extend(&words[..n]);
    n
}

/// Up to `max` restore words, oldest first.
pub fn take_restore_words(max: usize) -> Vec<u32> {
    let mut g = RESTORE_WORDS.lock().unwrap_or_else(|p| p.into_inner());
    let n = g.len().min(max);
    g.drain(..n).collect()
}

/// The pin ledger's view of one request (`b2_mirror::pin_note_submit`).
#[derive(Clone, Copy, Debug, Default)]
pub struct PinNote<'a> {
    /// The router's own picks for these rows (`[b, nu]`, its rank order),
    /// box 2's share (`b2_mirror::wants_for_box2`), when a cache prior or a
    /// mode-2 substitution changed what is sent. `None` ranks by `sel`.
    pub wants: Option<&'a [i32]>,
    /// The rows are DECODE rows (the arena). Only decode picks rank experts
    /// for release: a prefill chunk of <= 16 rows (a prompt's tail, a short
    /// suffix) must not earn credit. `None` = by row count alone.
    pub decode: Option<bool>,
}

pub fn take_prefetch_words(max: usize) -> Vec<u32> {
    let cap = PREFETCH_TAKE_CAP.load(std::sync::atomic::Ordering::Relaxed);
    let mut g = PREFETCH_WORDS.lock().unwrap();
    let n = g.len().min(max).min(cap);
    g.drain(..n).collect()
}

/// Per-request cap on the words `take_prefetch_words` hands a frame (default
/// unlimited). Box 2 starts a speculative word only into a free staging set and
/// drops the rest, so a producer with more words than sets (layer-major group
/// prefetch) paces them out over requests instead of losing them on the first.
static PREFETCH_TAKE_CAP: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(usize::MAX);

/// Set the per-request word cap; returns the previous one (to restore it).
pub fn set_prefetch_take_cap(cap: usize) -> usize {
    PREFETCH_TAKE_CAP.swap(cap.max(1), std::sync::atomic::Ordering::Relaxed)
}

/// Keep only the queued words `keep` accepts (a producer dropping its own stale
/// words before queueing new ones).
pub fn retain_prefetch_words(keep: impl Fn(u32) -> bool) {
    PREFETCH_WORDS.lock().unwrap().retain(|&w| keep(w));
}

/// Remove and return the queued words `take` selects (a producer learning which
/// of its words were NOT sent).
pub fn extract_prefetch_words(take: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut g = PREFETCH_WORDS.lock().unwrap();
    let (out, keep): (Vec<u32>, Vec<u32>) = g.iter().partition(|&&w| take(w));
    *g = keep;
    out
}

pub fn b2_hits_first() -> bool {
    HITS_FIRST_INIT.get_or_init(|| {
        let on = matches!(std::env::var("V41_B2_HITS_FIRST").as_deref(), Ok("1") | Ok("on"));
        HITS_FIRST.store(on, std::sync::atomic::Ordering::Relaxed);
    });
    HITS_FIRST.load(std::sync::atomic::Ordering::Relaxed)
}

extern "C" fn hits_first_toggle(_sig: i32) {
    // Async-signal-safe: one atomic op, nothing else.
    HITS_FIRST.fetch_xor(true, std::sync::atomic::Ordering::Relaxed);
}

/// Install the `SIGUSR1` toggle for [`b2_hits_first`] (daemon main). Returns the
/// initial state so the daemon can log it.
pub fn install_hits_first_toggle() -> bool {
    extern "C" {
        fn signal(sig: i32, handler: extern "C" fn(i32)) -> usize;
    }
    const SIGUSR1: i32 = 10;
    let on = b2_hits_first(); // initialise from the env BEFORE the handler can flip it
    // SAFETY: plain libc signal(2) with an async-signal-safe handler.
    unsafe {
        signal(SIGUSR1, hits_first_toggle);
    }
    on
}

pub fn b2_decode_down() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        matches!(std::env::var("V41_B2_DECODE_DOWN").as_deref(), Ok("1") | Ok("on"))
    });
    *B
}

/// `V41_B2_MISS_PAR=N`: missing experts of one layer read concurrently (default 1: MEASURED
/// 2026-09-20 no gain at 4 -- each expert read is already 8 preads wide and the drive is at its
/// ceiling; per-thread pread doubled instead. Kept for a faster drive.
/// the drive's aggregate random-read peak, see `ensure_layer_inner`). 1 = the
/// old serial loop.
pub fn b2_miss_par() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_MISS_PAR").ok().and_then(|v| v.parse().ok()).unwrap_or(1)
    });
    (*N).clamp(1, 16)
}

/// `V41_B2_SCAN_CLASS=0` disables the two-class LRU (see `ensure_layer_inner`).
pub fn b2_scan_class() -> bool {
    static B: std::sync::LazyLock<bool> =
        std::sync::LazyLock::new(|| std::env::var("V41_B2_SCAN_CLASS").as_deref() != Ok("0"));
    *B
}

pub fn b2_coalesce() -> bool {
    knobs::coalesce()
}

pub fn b2_odirect() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_ODIRECT").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

fn role_kr(which: &str) -> (u64, u64) {
    if which == "down" {
        (N_FF_EXP as u64, N_EMBD as u64)
    } else {
        (N_EMBD as u64, N_FF_EXP as u64)
    }
}

/// Byte geometry of one expert of each role in the checkpoint's engine form.
pub fn expert_geometry(src: &WeightSrc<'_>) -> eyre::Result<[(GgufType, u64, u64, usize); 3]> {
    let geom = |which: &str| -> eyre::Result<(GgufType, u64, u64, usize)> {
        let name = format!("blk.0.ffn_{which}_exps.weight");
        let t = src.tensor(&name).ok_or_else(|| eyre!("expert shard: tensor `{name}` not found"))?;
        let (k, rows) = role_kr(which);
        let bpe = weight_contract::bytes_per_expert(t.dtype, k, rows)?;
        if t.byte_size as usize != N_EXPERT as usize * bpe {
            return Err(eyre!("{name}: byte_size {} != {} × {bpe}", t.byte_size, N_EXPERT));
        }
        Ok((t.dtype, k, rows, bpe))
    };
    Ok([geom("gate")?, geom("up")?, geom("down")?])
}

// SAFETY: the only non-Send/Sync members are `DeviceBuffer`s (raw device
// pointers). HIP allocations are process-wide and the runtime is thread-safe
// with a per-thread `hipSetDevice` (every entry point here pins the device);
// the shard is written once at load and only read afterwards, and the
// executor is used from one thread at a time (`&mut`). Same reasoning as
// `unsafe impl Send for Graph` in v4flash-hip.
unsafe impl Send for ExpertShard {}
unsafe impl Sync for ExpertShard {}
unsafe impl Send for MoeExecutor {}

impl ExpertShard {
    /// Load every expert of `asg` into a packed pool on `igpu`: one contiguous
    /// slot range per layer, ids in ascending order. `threads` parallel
    /// readers, `batch` experts staged per device copy.
    pub fn load(
        owner: V41HfWeights,
        igpu: Device,
        asg: &Assignment,
        threads: usize,
        batch: usize,
        max_batch: u32,
        decode_max_b: u32,
    ) -> eyre::Result<Self> {
        Self::load_traced(owner, igpu, asg, threads, batch, max_batch, decode_max_b, None)
    }

    /// As [`Self::load`], emitting one perfetto span per expert read labelled
    /// `(layer, expert)` so a slow pick is attributable.
    #[allow(clippy::too_many_arguments)]
    pub fn load_traced(
        owner: V41HfWeights,
        igpu: Device,
        asg: &Assignment,
        threads: usize,
        batch: usize,
        max_batch: u32,
        decode_max_b: u32,
        tracer: Option<&ExpertdTracer>,
    ) -> eyre::Result<Self> {
        igpu.set_current()?;
        let t0 = Instant::now();
        let [g, u, d] = expert_geometry(&WeightSrc::from(&owner))?;
        let (gbpe, ubpe, dbpe) = (g.3, u.3, d.3);
        let per_slot = gbpe + ubpe + dbpe;
        let n_slots = asg.n_experts() as u32;
        if n_slots == 0 {
            return Err(eyre!("expert shard: empty assignment"));
        }
        eprintln!(
            "expert shard: {n_slots} experts × {:.2} MB = {:.2} GB pool on device {} ({} layers)",
            per_slot as f64 / 1e6,
            n_slots as f64 * per_slot as f64 / 1e9,
            igpu.id,
            asg.layers.len()
        );
        let make = |(dtype, k, rows, bpe): (GgufType, u64, u64, usize)| -> eyre::Result<DeviceWeight> {
            Ok(DeviceWeight {
                buffer: DeviceBuffer::<u8>::new(igpu.id, n_slots as usize * bpe)?,
                n_elements: n_slots as u64 * k * rows,
                dtype,
                shape: vec![n_slots as u64, rows, k],
            })
        };
        let mut routed = RoutedExpertWeights {
            gate: make(g)?,
            up: make(u)?,
            down: make(d)?,
            gate_bytes_per_expert: gbpe,
            up_bytes_per_expert: ubpe,
            down_bytes_per_expert: dbpe,
            n_slots,
        };

        let threads = threads.max(1);
        let batch = batch.max(1);
        let mut par_gate = vec![0u8; batch * gbpe];
        let mut par_up = vec![0u8; batch * ubpe];
        let mut par_down = vec![0u8; batch * dbpe];
        let mut layers: Vec<Option<LayerShard>> = (0..N_LAYER as usize).map(|_| None).collect();
        let mut next_slot = 0u32;
        let mut bytes = 0u64;
        for (layer, ids) in &asg.layers {
            let tl = Instant::now();
            let names = [
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
                format!("blk.{layer}.ffn_down_exps.weight"),
            ];
            let base = next_slot;
            let mut i0 = 0usize;
            while i0 < ids.len() {
                let n = batch.min(ids.len() - i0);
                {
                    let owner = &owner;
                    let names = &names;
                    let layer_u = *layer as u32;
                    let read_one = move |e: u32, gs: &mut [u8], us: &mut [u8], ds: &mut [u8]| -> eyre::Result<()> {
                        let src = WeightSrc::from(owner);
                        let tg = src.tensor(&names[0]).ok_or_else(|| eyre!("{}", names[0]))?;
                        let tu = src.tensor(&names[1]).ok_or_else(|| eyre!("{}", names[1]))?;
                        let td = src.tensor(&names[2]).ok_or_else(|| eyre!("{}", names[2]))?;
                        // App-level span around the three role reads of ONE expert:
                        // this is the SSD path (pread + MXFP4 repack), so a stall here
                        // attributes to (layer, expert), not to "load".
                        let t0 = tracer.map(|_| super::perfetto::host_now_ns());
                        src.read_expert_into(tg, e as usize, gs)?;
                        src.read_expert_into(tu, e as usize, us)?;
                        src.read_expert_into(td, e as usize, ds)?;
                        if let (Some(tr), Some(t0)) = (tracer, t0) {
                            tr.ssd_read(layer_u, e, t0, super::perfetto::host_now_ns());
                        }
                        Ok(())
                    };
                    let gsl = &mut par_gate[..n * gbpe];
                    let usl = &mut par_up[..n * ubpe];
                    let dsl = &mut par_down[..n * dbpe];
                    let per = n.div_ceil(threads);
                    let err: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);
                    let idsb = &ids[i0..i0 + n];
                    std::thread::scope(|sc| {
                        for (t, ((gc, uc), dc)) in gsl
                            .chunks_mut(per * gbpe)
                            .zip(usl.chunks_mut(per * ubpe))
                            .zip(dsl.chunks_mut(per * dbpe))
                            .enumerate()
                        {
                            let err = &err;
                            sc.spawn(move || {
                                for (i, ((gs, us), ds)) in gc
                                    .chunks_mut(gbpe)
                                    .zip(uc.chunks_mut(ubpe))
                                    .zip(dc.chunks_mut(dbpe))
                                    .enumerate()
                                {
                                    if let Err(e) = read_one(idsb[t * per + i], gs, us, ds) {
                                        *err.lock().unwrap() = Some(format!("{e:#}"));
                                        return;
                                    }
                                }
                            });
                        }
                    });
                    let failed = err.lock().unwrap().take();
                    if let Some(e) = failed {
                        return Err(eyre!("expert shard L{layer}: {e}"));
                    }
                }
                let s0 = (base as usize) + i0;
                routed.gate.buffer.slice_view_mut(s0 * gbpe, n * gbpe).copy_from_host(&par_gate[..n * gbpe])?;
                routed.up.buffer.slice_view_mut(s0 * ubpe, n * ubpe).copy_from_host(&par_up[..n * ubpe])?;
                routed.down.buffer.slice_view_mut(s0 * dbpe, n * dbpe).copy_from_host(&par_down[..n * dbpe])?;
                bytes += (n * per_slot) as u64;
                i0 += n;
            }
            let mut remap = vec![0i32; REMAP_LEN];
            let mut owned = vec![false; N_EXPERT as usize];
            for (local, &e) in ids.iter().enumerate() {
                // ABSOLUTE slot — `layer_views` hands the kernel the WHOLE pool,
                // so a local index here would address another layer's expert.
                // This is uploaded to `remap_dev` immediately and is only
                // re-uploaded from the pager's copy `if dirty`, so a layer that
                // never takes a miss would read wrong weights forever.
                remap[e as usize] = -(base as i32 + local as i32) - 1;
                owned[e as usize] = true;
            }
            let mut remap_dev = DeviceBuffer::<i32>::new(igpu.id, REMAP_LEN)?;
            remap_dev.copy_from_host(&remap)?;
            layers[*layer as usize] = Some(LayerShard { base_slot: base, ids: ids.clone(), remap_dev, owned, page: None });
            next_slot += ids.len() as u32;
            let dt = tl.elapsed().as_secs_f64();
            eprintln!(
                "expert shard: L{layer:>2} {:>3} experts in {:6.2} s ({:.2} GB/s) slots {base}..{next_slot}",
                ids.len(),
                dt,
                ids.len() as f64 * per_slot as f64 / 1e9 / dt.max(1e-9)
            );
        }
        let seconds = t0.elapsed().as_secs_f64();
        let info = ShardInfo {
            n_layer: N_LAYER as u32,
            n_expert: N_EXPERT,
            n_used: N_EXPERT_USED as u32,
            n_embd: N_EMBD,
            xq_bytes_per_token: XQ_BYTES_PER_TOKEN as u32,
            max_batch,
            decode_max_b,
            n_resident: n_slots,
            bytes_per_expert: per_slot as u32,
            clock: ClockPair::sample(),
            owned: asg.bitsets(),
        };
        eprintln!(
            "expert shard: loaded {n_slots} experts, {:.2} GB in {seconds:.1} s ({:.2} GB/s)",
            bytes as f64 / 1e9,
            bytes as f64 / 1e9 / seconds.max(1e-9)
        );
        // Three landing zones sized to the largest role, so a miss's three
        // uploads do not serialise on one buffer (mirrors box 1's pager).
        let bpe3 = [
            routed.gate_bytes_per_expert,
            routed.up_bytes_per_expert,
            routed.down_bytes_per_expert,
        ];
        let (repack, repack_stream) = if b2_gpu_repack() {
            let arch = igpu.properties()?.gcn_arch_name;
            eprintln!(
                "expert shard: GPU MXFP4 repack ON ({arch}), zero-copy pinned staging 3 x {:.1} MB",
                bpe3[0].max(bpe3[1]).max(bpe3[2]) as f64 / 1e6
            );
            (Some(Mxfp4Repack::for_arch(&arch)?), Some(Stream::new(igpu.id)?))
        } else {
            eprintln!("expert shard: GPU MXFP4 repack OFF — CPU repack on every miss");
            (None, None)
        };
        // Staging is oversized so an O_DIRECT read can place each region at its
        // own 4096-residue: one spare block per region, two regions per role.
        // (packed and scale are each 4096-multiples here, so +2 blocks suffices;
        // +4 is slack for a checkpoint whose lengths are not.)
        let mut stages: Vec<[PinnedBuffer<u8>; 3]> = Vec::new();
        for _ in 0..b2_miss_par() { stages.push([
            // Under `V41_B2_COALESCE` staging is REPURPOSED: [0] holds the whole
            // 3-role weight run and [1] the 3-role scale run, so one pread fills
            // each. Sized from bpe3 (which already exceeds packed+scale per role)
            // so it cannot be too small: 3x covers the weight run, 1x the scales.
            // ALWAYS coalesce-sized (3x role 0) so `knobs::coalesce` can be
            // flipped at runtime: the span read bounds-checks its destination and
            // would fail the request otherwise. ~75 MB of extra pinned memory.
            PinnedBuffer::<u8>::new_with_flags(
                3 * bpe3[0] + 4 * 4096,
                HIP_HOST_MALLOC_NON_COHERENT,
            )?,
            PinnedBuffer::<u8>::new_with_flags(bpe3[1] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
            PinnedBuffer::<u8>::new_with_flags(bpe3[2] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
        ]); }
        // Look-ahead prefetch staging + reader thread (see `B2Prefetch`).
        let mut pf_stages: Vec<[PinnedBuffer<u8>; 3]> = Vec::new();
        for _ in 0..b2_prefetch_sets() {
            pf_stages.push([
                PinnedBuffer::<u8>::new_with_flags(3 * bpe3[0] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
                PinnedBuffer::<u8>::new_with_flags(bpe3[1] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
                PinnedBuffer::<u8>::new_with_flags(bpe3[2] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
            ]);
        }
        // O_DIRECT needs a 4096-aligned buffer. hipHostMalloc gives page-aligned
        // memory, but CHECK rather than assume: a misaligned buffer fails pread
        // with EINVAL, which is a confusing way to learn this.
        let aligned = stages.iter().flatten().all(|p| p.as_slice().as_ptr() as usize % 4096 == 0);
        let direct = repack.is_some() && b2_odirect() && aligned;
        if b2_odirect() && !aligned {
            eprintln!("expert shard: O_DIRECT off — pinned staging is not 4096-aligned");
        }
        eprintln!("expert shard: zero-copy O_DIRECT expert reads {}", if direct { "ON" } else { "OFF" });
        Ok(Self {
            owner: std::sync::Arc::new(owner),
            device: igpu,
            routed,
            layers,
            info,
            load_stats: LoadStats { n_experts: n_slots as usize, bytes, seconds },
            repack,
            repack_stream,
            stages,
            prefetch: None,
            pf_stages_spare: pf_stages,
            direct,
            pool: None,
            pinned: Vec::new(),
            parked_pins: Vec::new(),
            park_words: Vec::new(),
            park_prefill: false,
            req_prefill: false,
            early_paged: EarlyPaged::default(),
            prefetch_wait_ns: 0,
            ev_admit: [0; 5],
        })
    }

    /// Queue look-ahead prefetch words `(layer << 16) | expert` (the hub's
    /// prediction of the NEXT layer's picks that this box owns). Non-resident,
    /// not-pending ids are read in the background into a free staging set;
    /// `admit_prefetched` lands them at the next `ensure`. Drops when every
    /// set is busy (bandwidth is the only cost of a wrong hint).
    pub fn prefetch_words(&mut self, words: &[u32]) {
        self.prefetch_words_ex(words, false)
    }

    /// `certain`: the words are a queued request's own picks (not a guess):
    /// read at demand priority, i.e. without waiting for in-flight misses.
    pub fn prefetch_words_ex(&mut self, words: &[u32], certain: bool) {
        self.prefetch_words_cls(words, certain, false)
    }

    /// `stage`: the words are a PREFILL-shaped request's own picks (its early
    /// page or park): the reads land in the staging band, prefill-class, and
    /// route by `knobs::prefill_route_split`. Ignored with staging off.
    pub fn prefetch_words_cls(&mut self, words: &[u32], certain: bool, stage: bool) {
        self.prefetch_words_full(words, certain, stage, false)
    }

    /// Mode-aware eviction: note the hub's phase of a request (`flags`) before
    /// serving it.
    pub fn note_request_phase(&mut self, flags: u32) {
        self.set_request_mode(flags);
        if let Some(pool) = self.pool.as_mut() {
            pool.me_note_request(flags & proto::REQ_FLAG_DECODE == 0);
        }
    }

    /// The request mode only, not the phase state: a frame served INSIDE a
    /// parked request's pass (its caller restores the parked one's mode).
    pub fn set_request_mode(&mut self, flags: u32) {
        self.req_prefill = flags & proto::REQ_FLAG_DECODE == 0;
    }

    /// Mode-aware eviction is on for this shard's pool.
    pub fn mode_evict_on(&self) -> bool {
        self.pool.as_ref().is_some_and(|p| p.me.on)
    }

    /// Hub prefetch words carried by a PREFILL-shaped request (layer-major group
    /// prefetch, `V41_LM_PREFETCH`): speculative, but they land PREFILL-class
    /// (evicted before decode's residents, like the prefill's own demand pages)
    /// instead of decode-class. In the main band when staging is off.
    pub fn prefetch_words_prefill(&mut self, words: &[u32]) {
        self.prefetch_words_full(words, false, false, true)
    }

    fn prefetch_words_full(&mut self, words: &[u32], certain: bool, stage: bool, prefill: bool) {
        let _ = self.prefetch_words_core(words, &[], certain, stage, prefill);
    }

    /// `prefetch_words_full`, with an optional restore stamp per word
    /// (`stamps` empty, or one per word: `PfJob::restore`). Returns the words
    /// DROPPED for want of a free staging set (the delta restore re-queues them).
    fn prefetch_words_core(&mut self, words: &[u32], stamps: &[u64], certain: bool, stage: bool, prefill: bool) -> Vec<u32> {
        let mut dropped_words = Vec::new();
        if words.is_empty() || self.pool.is_none() {
            return dropped_words;
        }
        // Mode-aware eviction: a prefill request's own early-page / park reads
        // land PREFILL-class even with staging off (they used to land
        // decode-class and search the global minimum = prefill's own pages).
        let me_on = self.pool.as_ref().is_some_and(|p| p.me.on);
        let own_prefill = stage;
        let prefill = prefill || (me_on && stage && self.stage_slots() == 0);
        let stage = stage && self.stage_slots() > 0;
        if self.prefetch.is_none() {
            if self.pf_stages_spare.is_empty() {
                return words.to_vec();
            }
            let n_par_q = b2_prefetch_par().min(self.pf_stages_spare.len().max(1));
            let queue = std::sync::Arc::new(PfQueue::with_readers(n_par_q.saturating_sub(b2_prefetch_reserve()), n_par_q));
            let _ = PF_QUEUE.set(std::sync::Arc::clone(&queue));
            v4flash_core::io_throttle::set_chunk_bytes(b2_spec_chunk_bytes());
            v4flash_core::io_throttle::install_pause(b2_background_pause);
            let (tx_done, rx_done) = std::sync::mpsc::channel::<Result<PfDone, (usize, u32, u32, String)>>();
            let stages = std::mem::take(&mut self.pf_stages_spare);
            let ptrs: Vec<SetPtr> = stages.iter().map(|st| SetPtr {
                p: [st[0].as_slice().as_ptr() as *mut u8, st[1].as_slice().as_ptr() as *mut u8, st[2].as_slice().as_ptr() as *mut u8],
                n: [st[0].len(), st[1].len(), st[2].len()],
            }).collect();
            let owner = std::sync::Arc::clone(&self.owner);
            let direct = self.direct;
            let gpu_repack = self.repack.is_some();
            let bpe = [self.routed.gate_bytes_per_expert, self.routed.up_bytes_per_expert, self.routed.down_bytes_per_expert];
            let n_par = b2_prefetch_par().min(stages.len().max(1));
            let mut readers = Vec::with_capacity(n_par);
            for _ in 0..n_par {
            let queue_r = std::sync::Arc::clone(&queue);
            let tx_done = tx_done.clone();
            let ptrs = ptrs.clone();
            let owner = std::sync::Arc::clone(&owner);
            readers.push(std::thread::Builder::new().name("b2-prefetch".into()).spawn(move || {
                let ptrs = ptrs;
                loop {
                    let urgency = knobs::route_urgency();
                    let Some(PfJob { layer, e, set, certain, stage, own_prefill, prefill, restore, t_hint }) = queue_r.pop_mode(urgency) else { break };
                    let ev_on = super::evtrace::enabled();
                    let ev_t_pop = if ev_on { super::evtrace::now() } else { f64::NAN };
                    let mut ev_yield_ns = 0u64;
                    let mut done = PfFinish { q: &queue_r, certain, spec_key: (urgency && !certain).then_some((layer, e)) };
                    // Urgency routing: a speculative read goes to the OTHER
                    // drive from demand reads, so it neither yields nor chunks.
                    if !certain && !urgency {
                        // Yield the drives to demand misses (bounded: a hint that
                        // waits longer than a layer is late anyway) -- unless a
                        // request needs this very expert, in which case it IS the
                        // demand read.
                        let t = std::time::Instant::now();
                        while (DEMAND_READS.load(std::sync::atomic::Ordering::Relaxed) > 0 || queue_r.background_should_wait(layer, e))
                            && t.elapsed() < std::time::Duration::from_millis(20)
                            && !queue_r.is_urgent(layer, e)
                        {
                            std::thread::sleep(std::time::Duration::from_micros(50));
                        }
                        ev_yield_ns = t.elapsed().as_nanos() as u64;
                        if queue_r.is_urgent(layer, e) {
                            queue_r.reclassify_certain();
                            done.certain = true;
                        }
                    }
                    let sp = ptrs[set];
                    // SAFETY: the set is owned by this thread until `Done`, and
                    // `B2Prefetch::drop` joins this thread before its staging
                    // buffers are freed.
                    let (b0, b1, b2) = unsafe {
                        (std::slice::from_raw_parts_mut(sp.p[0], sp.n[0]), std::slice::from_raw_parts_mut(sp.p[1], sp.n[1]), std::slice::from_raw_parts_mut(sp.p[2], sp.n[2]))
                    };
                    let t_read = std::time::Instant::now();
                    let ev_t_read = if ev_on { super::evtrace::now() } else { f64::NAN };
                    // Concurrency as this read STARTS (after any yield).
                    let ev_q = if ev_on { queue_r.counts() } else { [f64::NAN; 4] };
                    let ev_demand = DEMAND_READS.load(std::sync::atomic::Ordering::Relaxed) as f64;
                    let queue_ns = (t_read - t_hint).as_nanos() as u64;
                    // A job still speculative at read start reads in chunks and
                    // yields to urgent reads (io_throttle); certain ones do not.
                    // Under urgency routing: certain -> the mirror, speculative
                    // -> the primary, and neither is throttled; a PREFILL
                    // chunk's own reads (`own_prefill`, staged or not) are
                    // striped across both drives like its demand reads
                    // (`knobs::prefill_route_split`).
                    let route = match (urgency, done.certain, own_prefill && knobs::prefill_route_split()) {
                        (false, _, _) | (true, _, true) => v4flash_core::hf_v41::ExpertRoute::split(),
                        (true, true, false) => v4flash_core::hf_v41::ExpertRoute::mirror_only(),
                        (true, false, false) => v4flash_core::hf_v41::ExpertRoute::primary_only(),
                    };
                    v4flash_core::io_throttle::set_background((!done.certain && !urgency).then_some(((layer as u64) << 16) | e as u64));
                    let r = Self::read_miss_into(&owner, direct, gpu_repack, layer, e, bpe, b0, b1, b2, route);
                    v4flash_core::io_throttle::set_background(None);
                    let read_ns = t_read.elapsed().as_nanos() as u64;
                    let mut ev = [f64::NAN; 18];
                    if ev_on {
                        let roles = EV_ROLES.with(|c| c.replace([f64::NAN; 6]));
                        let pause = EV_PAUSE.lock().unwrap().remove(&(((layer as u64) << 16) | e as u64)).unwrap_or(0);
                        ev[..11].copy_from_slice(&[
                            super::evtrace::inst_to_raw(t_hint), ev_t_pop, ev_t_read, super::evtrace::now(),
                            ev_yield_ns as f64, pause as f64, f64::from(u8::from(certain)), f64::from(u8::from(done.certain)),
                            ev_demand, ev_q[0], ev_q[1],
                        ]);
                        ev[11..17].copy_from_slice(&roles);
                        ev[17] = route.code();
                    }
                    drop(done);
                    let msg = match r {
                        Ok((offs, coalesced)) => Ok(PfDone { layer, e, set, stage, prefill, restore, offs, coalesced, queue_ns, read_ns, ev }),
                        Err(err) => Err((set, layer, e, format!("{err:#}"))),
                    };
                    if tx_done.send(msg).is_err() {
                        break;
                    }
                }
            }).expect("spawn b2-prefetch"));
            }
            let n = stages.len();
            self.prefetch = Some(B2Prefetch { queue, rx_done, readers, stages, free: (0..n).collect(), pending: Default::default(), hinted: 0, admitted: 0, dropped: 0, waited: 0, promoted: 0, queue_ns: 0, read_ns: 0, n_read: 0 });
            eprintln!("expertd: look-ahead prefetch ON ({n} staging sets, {n_par} readers)");
        }
        let pool = self.pool.as_ref().unwrap();
        let pf = self.prefetch.as_mut().unwrap();
        for (i, &w) in words.iter().enumerate() {
            let restore = stamps.get(i).copied().unwrap_or(0);
            let key = ((w >> 16) as u32, (w & 0xFFFF) as u32);
            if key.0 as usize >= self.layers.len() || key.1 >= N_EXPERT || self.layers[key.0 as usize].is_none() {
                continue;
            }
            if pool.slot_of.contains_key(&key) {
                continue;
            }
            if pf.pending.contains(&key) {
                // Already being fetched. If a request needs it now, make that
                // read urgent rather than letting the request wait behind the
                // speculative queue.
                if certain && pf.queue.promote(key.0, key.1) {
                    pf.promoted += 1;
                }
                continue;
            }
            // Keep sets free for certain words: a speculative word that would
            // take one of the last reserved sets is dropped (it is retried by
            // whoever wants it next); a certain word may use any set.
            if !certain && pf.free.len() <= 2 * b2_prefetch_reserve() {
                pf.dropped += 1;
                dropped_words.push(w);
                continue;
            }
            let Some(set) = pf.free.pop() else {
                pf.dropped += 1;
                dropped_words.push(w);
                continue;
            };
            pf.pending.insert(key);
            pf.hinted += 1;
            pf.queue.push(PfJob { layer: key.0, e: key.1, set, certain, stage, own_prefill, prefill, restore, t_hint: std::time::Instant::now() });
        }
        dropped_words
    }

    /// Delta restore (`V41_B2_RESTORE`): in a DECODE phase, start background
    /// reads of the next few restore-queue entries (newest stamp first). Words
    /// dropped for want of a staging set go back to the front, in order; an
    /// entry decode already paged back is skipped. Called per request.
    pub fn pump_restore(&mut self) {
        // Throttle: only while more than half the staging sets are free.
        if self.prefetch.as_ref().is_some_and(|pf| pf.free.len() * 2 <= pf.stages.len()) {
            return;
        }
        let pending_of = |pf: &Option<B2Prefetch>, k: &(u32, u32)| pf.as_ref().is_some_and(|pf| pf.pending.contains(k));
        let Some(pool) = self.pool.as_mut() else { return };
        if !(pool.me.on && pool.me.restore_on && !pool.me.prefill_phase) {
            return;
        }
        let room = RESTORE_PUMP.min(RESTORE_INFLIGHT.saturating_sub(pool.me.restore_inflight.len()));
        let mut batch: Vec<(u32, u32, u64)> = Vec::with_capacity(room);
        while batch.len() < room {
            let Some(ent) = pool.me.restore.pop_front() else { break };
            // Already back (decode paged it), or already being read by another
            // job (it lands as that job's read): nothing to restore.
            if pool.slot_of.contains_key(&(ent.0, ent.1)) || pending_of(&self.prefetch, &(ent.0, ent.1)) {
                pool.me.rc.skipped += 1;
                continue;
            }
            batch.push(ent);
        }
        if batch.is_empty() {
            return;
        }
        let words: Vec<u32> = batch.iter().map(|&(l, e, _)| (l << 16) | e).collect();
        let stamps: Vec<u64> = batch.iter().map(|&(_, _, t)| t).collect();
        let dropped = self.prefetch_words_core(&words, &stamps, false, false, false);
        let pool = self.pool.as_mut().expect("checked above");
        for ent in &batch {
            if !dropped.contains(&((ent.0 << 16) | ent.1)) {
                pool.me.rc.pumped += 1;
                pool.me.restore_inflight.insert((ent.0, ent.1));
            }
        }
        for ent in batch.iter().rev().filter(|ent| dropped.contains(&((ent.0 << 16) | ent.1))) {
            pool.me.restore.push_front(*ent);
        }
    }

    /// Land every prefetch read that has COMPLETED, without waiting for any
    /// (the park loop, `knobs::park`). `layer` is only the caller's current
    /// layer for the dirty bookkeeping; nothing is protected beyond `pinned` /
    /// `parked_pins`.
    pub fn admit_landed(&mut self, layer: u32) -> eyre::Result<()> {
        self.admit_prefetched(layer, &[])
    }

    /// Is any of `words` (`layer << 16 | expert`) still being read by the
    /// prefetch readers (hinted, not yet admitted)?
    pub fn prefetch_pending_any(&self, words: &[u32]) -> bool {
        let Some(pf) = self.prefetch.as_ref() else { return false };
        let urgency = knobs::route_urgency();
        let spec = if urgency { pf.queue.spec_keys_snapshot() } else { Default::default() };
        let resident = |k: &(u32, u32)| self.pool.as_ref().is_some_and(|p| p.slot_of.contains_key(k));
        words.iter().any(|&w| park_waits_for(((w >> 16), (w & 0xFFFF)), &pf.pending, &spec, resident, urgency))
    }

    /// `(queue_ns, read_ns, n_read)` summed over completed prefetch reads.
    pub fn prefetch_read_timing(&self) -> (u64, u64, u64) {
        self.prefetch.as_ref().map_or((0, 0, 0), |p| (p.queue_ns, p.read_ns, p.n_read))
    }

    /// `(hinted, admitted, dropped, waited)` since start.
    pub fn prefetch_stats(&self) -> Option<(u64, u64, u64, u64)> {
        self.prefetch.as_ref().map(|p| (p.hinted, p.admitted, p.dropped, p.waited))
    }

    /// Pending speculative reads made urgent because a request needed them.
    pub fn prefetch_promoted(&self) -> u64 {
        self.prefetch.as_ref().map_or(0, |p| p.promoted)
    }

    /// Land every completed prefetch read into the pool (victim + repack), for
    /// any layer. Called at the top of `ensure_layer_inner`, where nothing
    /// reads the pool. `want` protects the current layer's picks from eviction.
    fn admit_prefetched(&mut self, cur_layer: u32, want: &[u32]) -> eyre::Result<()> {
        let pinned: Vec<(u32, u32)> = self.pinned.iter().chain(self.parked_pins.iter()).copied().collect();
        let Some(pf) = self.prefetch.as_mut() else { return Ok(()) };
        let Some(pool) = self.pool.as_mut() else { return Ok(()) };
        let global = b2_global_pool();
        let repack = self.repack.as_ref();
        let repack_stream = self.repack_stream.as_ref();
        let r = &mut self.routed;
        let bpe = [r.gate_bytes_per_expert, r.up_bytes_per_expert, r.down_bytes_per_expert];
        // A wanted id whose prefetch read is still in flight: wait for it rather
        // than issue a second read of the same bytes (the miss loop below does
        // not know about `pending`, so it would page it again into a NEW slot
        // and the prefetch would land as a duplicate). In flight means a few
        // ms at most.
        // Urgency routing: the wanted keys whose SPECULATIVE read is running on
        // the primary (taken after the promotions below) are not waited for.
        let mut spec_running: std::collections::HashSet<(u32, u32)> = Default::default();
        // TIME the blocking wait (2026-09-22). It happens inside `run_path`, so
        // it was landing in the request's `t_compute_us` while contributing
        // nothing to `t_page_us` -- i.e. the hub's `box2.compute_ms` (service
        // minus page) counted box 2 WAITING FOR ITS DISK as compute. The stats
        // line already hinted at it: `waited` == `admitted` == `hinted`, so
        // every prefetched expert was blocked on. Folded into the layer's page
        // accounting below so "compute" means compute.
        let mut wait_ns = 0u64;
        let ev_on = super::evtrace::enabled();
        let (mut ev_landed, mut ev_wanted, mut ev_blocking) = (0u64, 0u64, 0u64);
        // `b2_read` for a background read the compute thread just handled.
        #[allow(clippy::too_many_arguments)]
        let ev_read = |d: &PfDone, slot: f64, victim: Option<(u32, u32)>, t_recv: f64, t_land0: f64, wanted: bool, blocked: bool, scan_ns: f64, repack_ns: f64, already: bool| {
            let src = if d.ev[6] == 1.0 { 1.0 } else if d.ev[7] == 1.0 { 2.0 } else { 3.0 };
            let (vl, ve) = victim.map_or((f64::NAN, f64::NAN), |(l, e)| (f64::from(l), f64::from(e)));
            let mut v = vec![
                src, ev_cur_seq(), f64::from(d.layer), f64::from(d.e), slot, vl, ve, d.set as f64,
                d.ev[0], d.ev[1], d.ev[2], d.ev[3], t_recv, t_land0, super::evtrace::now(),
                d.ev[4], d.ev[5], d.ev[8], d.ev[9], d.ev[10],
            ];
            v.extend_from_slice(&d.ev[11..17]);
            v.extend_from_slice(&[
                f64::from(u8::from(wanted)), f64::from(u8::from(blocked)), f64::from(u8::from(d.coalesced)),
                scan_ns, repack_ns, f64::NAN, f64::NAN, f64::from(u8::from(already)), d.ev[17],
            ]);
            super::evtrace::emit(&super::evtrace_kinds::B2_READ, &v);
        };
        // About to block on this request's own picks: whatever of them is still
        // pending as a speculative read becomes urgent, so the wait is one read,
        // not the speculative queue ahead of it plus its yield.
        for &e in want {
            if pf.pending.contains(&(cur_layer, e)) && pf.queue.promote(cur_layer, e) {
                pf.promoted += 1;
            }
        }
        // Under urgency routing a request never waits on a speculative read
        // already running on the primary (the E100, which stalls under
        // writes): it reads the expert itself from the mirror (the miss loop
        // below does not know `pending`) and the late copy is discarded when it
        // lands (`slot_of` already holds the key). Queued speculative reads
        // were just promoted to certain, i.e. to the mirror, and are waited for.
        if knobs::route_urgency() {
            spec_running = pf.queue.spec_keys_snapshot();
        }
        let ev_skipped = want.iter().filter(|&&e| pf.pending.contains(&(cur_layer, e)) && spec_running.contains(&(cur_layer, e))).count() as u64;
        let in_flight = |pf: &B2Prefetch| wanted_in_flight(want, cur_layer, &pf.pending, &spec_running);
        loop {
            let must_wait = in_flight(pf);
            let t_w = must_wait.then(std::time::Instant::now);
            let d = match if must_wait { pf.rx_done.recv().map_err(|_| std::sync::mpsc::TryRecvError::Disconnected) } else { pf.rx_done.try_recv() } {
                Ok(Ok(d)) => {
                    if let Some(t) = t_w { wait_ns += t.elapsed().as_nanos() as u64; }
                    if must_wait && d.layer == cur_layer && want.contains(&d.e) { pf.waited += 1; }
                    ev_landed += 1;
                    ev_blocking += u64::from(must_wait);
                    ev_wanted += u64::from(d.layer == cur_layer && want.contains(&d.e));
                    pf.queue_ns += d.queue_ns;
                    pf.read_ns += d.read_ns;
                    pf.n_read += 1;
                    d
                }
                Ok(Err((set, layer, e, msg))) => {
                    if let Some(t) = t_w { wait_ns += t.elapsed().as_nanos() as u64; }
                    eprintln!("expertd: prefetch read failed (L{layer} e{e}): {msg}");
                    pool.me.restore_inflight.remove(&(layer, e));
                    pf.pending.remove(&(layer, e));
                    pf.queue.clear_urgent(layer, e);
                    pf.free.push(set);
                    continue;
                }
                Err(_) => break,
            };
            let key = (d.layer, d.e);
            let ev_t_recv = if ev_on { super::evtrace::now() } else { f64::NAN };
            let ev_wanted_this = d.layer == cur_layer && want.contains(&d.e);
            pool.me.restore_inflight.remove(&key);
            pf.pending.remove(&key);
            pf.queue.clear_urgent(d.layer, d.e);
            if pool.slot_of.contains_key(&key) {
                if ev_on {
                    ev_read(&d, f64::NAN, None, ev_t_recv, ev_t_recv, ev_wanted_this, must_wait && ev_wanted_this, f64::NAN, f64::NAN, true);
                }
                pf.free.push(d.set);
                continue;
            }
            let Some(l) = self.layers.get(d.layer as usize).and_then(|l| l.as_ref()) else { pf.free.push(d.set); continue };
            let region = (l.base_slot as u32, l.base_slot as u32 + l.ids.len() as u32);
            let ev_t_scan = std::time::Instant::now();
            // Never a hub-pinned victim: a background landing is optional, so
            // with every candidate pinned it is DROPPED (whoever needs the
            // expert demand-reads it; the pin reserve covers that claim). A
            // prefill chunk's read (`stage`) lands in the staging band only
            // and is dropped when that has no victim.
            let band = if d.stage { Band::Stage } else { Band::Main };
            // Mode-aware eviction: a prefill landing (staged, or prefill-class)
            // searches in prefill mode.
            let prefill_landing = d.stage || d.prefill;
            // Delta restore (`V41_B2_RESTORE`): lands only over what decode would
            // evict before it, with its own stamp; STOPS the restore when nothing
            // is older; back to the queue if a prefill phase began meanwhile. A
            // restore the current request wants is just a decode landing.
            let restore_stamp = if d.restore != 0 && !ev_wanted_this { d.restore } else { 0 };
            if restore_stamp != 0 && (!pool.me.on || pool.me.prefill_phase) {
                pool.me.restore.push_front((d.layer, d.e, restore_stamp));
                pool.me.rc.requeued += 1;
                pf.free.push(d.set);
                continue;
            }
            let victim = if restore_stamp != 0 {
                match pool.restore_victim(region, global, cur_layer, want, &pinned, d.layer, restore_stamp) {
                    Some(v) => Some(v),
                    None => {
                        pool.me.rc.stopped += 1;
                        pool.me.restore.clear();
                        pf.free.push(d.set);
                        continue;
                    }
                }
            } else {
                pool.pick_victim_any(region, global, band, cur_layer, want, &pinned, d.layer, false, prefill_landing)
            };
            let Some(victim) = victim else {
                if d.stage {
                    pool.sc.drops += 1;
                } else if pool.pins.on && pool.pick_victim_any(region, global, band, cur_layer, want, &pinned, d.layer, true, prefill_landing).is_some() {
                    pool.pins.c.no_victim_drops += 1;
                }
                pf.free.push(d.set);
                continue;
            };
            let ev_scan_ns = ev_t_scan.elapsed().as_nanos() as f64;
            if restore_stamp == 0 {
                pool.me_account(victim, prefill_landing);
            }
            let ev_victim = pool.evict(victim, cur_layer);
            let ev_t_repack = std::time::Instant::now();
            let landed: eyre::Result<()> = match (repack, repack_stream) {
                (Some(rp), Some(rs)) => Self::repack_in_place(rp, rs, r, victim, &pf.stages[d.set], &d.offs, d.coalesced).map(|_| ()),
                _ => (0..3).try_for_each(|i| {
                    let buf = match i { 0 => &mut r.gate.buffer, 1 => &mut r.up.buffer, _ => &mut r.down.buffer };
                    buf.slice_view_mut(victim as usize * bpe[i], bpe[i]).copy_from_host(&pf.stages[d.set][i].as_slice()[..bpe[i]])
                }),
            };
            if let Err(err) = landed {
                // The victim is already detached (free, unowned); give the set
                // back rather than leak it, then report.
                pf.free.push(d.set);
                return Err(err);
            }
            if restore_stamp != 0 {
                pool.land_stamped(victim, key, restore_stamp);
                pool.me.rc.landed += 1;
            } else {
                pool.land(victim, key, d.stage || d.prefill);
            }
            pf.admitted += 1;
            pf.free.push(d.set);
            if ev_on {
                ev_read(&d, f64::from(victim), ev_victim, ev_t_recv, ev_t_recv, ev_wanted_this, must_wait && ev_wanted_this,
                    ev_scan_ns, ev_t_repack.elapsed().as_nanos() as f64, false);
            }
        }
        // Attribute the blocking wait to the LAYER's page accounting, so the
        // hub's `box2.page_ms` covers it and `box2.compute_ms` (service minus
        // page) stops counting disk waits as compute. Also tracked separately
        // in `prefetch_wait_ns` for the stats line.
        self.ev_admit = [wait_ns, ev_landed, ev_wanted, ev_blocking, ev_skipped];
        if wait_ns > 0 {
            self.prefetch_wait_ns += wait_ns;
            if let Some(pg) = self
                .layers
                .get_mut(cur_layer as usize)
                .and_then(|l| l.as_mut())
                .and_then(|l| l.page.as_mut())
            {
                pg.read_ns += wait_ns;
            }
        }
        Ok(())
    }

    /// Cumulative `(misses, page_ns)` for `layer`, or `(0, 0)` when the layer is
    /// not paged. `page_ns` is the wall cost of making experts resident:
    /// `read_ns` (NVMe + any CPU repack) plus `h2d_ns` plus the GPU repack.
    /// Box 1 now holds these `(layer << 16 | expert)`: move our copies to the
    /// FRONT of the LRU so they are the next victims. This is what makes the
    /// two pools exclusive -- without it box 1's L1 was a strict subset of this
    /// pool (both LRUs over the same miss stream) and added zero capacity.
    pub fn hint_evict_first(&mut self, words: &[u32]) {
        let Some(pool) = self.pool.as_mut() else { return };
        let mut n = 0u64;
        for &w in words {
            let key = ((w >> 16) as u32, (w & 0xFFFF) as u32);
            if let Some(&slot) = pool.slot_of.get(&key) {
                pool.last_use[slot as usize] = 0; // front of the LRU: next victim
                n += 1;
            }
        }
        HINTS_APPLIED.fetch_add(n, std::sync::atomic::Ordering::Relaxed);
    }

    /// Snapshot around a request and difference it to get that request's paging.
    /// `evtrace`: the layer's cumulative page stats as separate components:
    /// misses, read_ns, h2d_ns, repack_gpu_ns, pread_ns, repack_cpu_ns.
    pub fn layer_page_detail(&self, layer: u32) -> [u64; 6] {
        match self.layers.get(layer as usize).and_then(|l| l.as_ref()).and_then(|l| l.page.as_ref()) {
            Some(pg) => [pg.misses, pg.read_ns, pg.h2d_ns, pg.repack_gpu_ns, pg.pread_ns, pg.repack_cpu_ns],
            None => [0; 6],
        }
    }

    /// `evtrace`: the background readers (running certain / speculative,
    /// queued certain / speculative, free staging sets, pending keys), their
    /// cumulative hinted / admitted / dropped / waited / promoted, and the
    /// pool's resident count.
    pub fn ev_pf_snapshot(&self) -> [f64; 12] {
        let mut v = [f64::NAN; 12];
        if let Some(p) = self.prefetch.as_ref() {
            v[..4].copy_from_slice(&p.queue.counts());
            v[4] = p.free.len() as f64;
            v[5] = p.pending.len() as f64;
            v[6..11].copy_from_slice(&[p.hinted as f64, p.admitted as f64, p.dropped as f64, p.waited as f64, p.promoted as f64]);
        }
        v[11] = self.pool.as_ref().map_or(f64::NAN, |p| p.slot_of.len() as f64);
        v
    }

    pub fn layer_page_counters(&self, layer: u32) -> (u64, u64) {
        match self.layers.get(layer as usize).and_then(|l| l.as_ref()).and_then(|l| l.page.as_ref()) {
            Some(pg) => (pg.misses, pg.read_ns + pg.h2d_ns + pg.repack_gpu_ns),
            None => (0, 0),
        }
    }

    pub fn info(&self) -> &ShardInfo {
        &self.info
    }
    pub fn device(&self) -> Device {
        self.device
    }
    pub fn owns(&self, layer: u32, e: i32) -> bool {
        (0..N_EXPERT as i32).contains(&e)
            && self.layers.get(layer as usize).and_then(|l| l.as_ref()).is_some_and(|l| l.owned[e as usize])
    }

    /// Turn every resident layer into a CATCH-ALL LRU cache over all 384 experts.
    ///
    /// Called after the normal load, so each layer's region starts warm with the
    /// assigned set rather than empty. From then on `owned` is all-true (this box
    /// can serve any expert) and [`Self::ensure_layer`] pages in whatever the hub
    /// asks for, from this box's own disk.
    pub fn enable_paging(&mut self) -> eyre::Result<()> {
        // `route=urgency` needs the mirror on every shard.
        let mirror_ok = self.owner.mirror_complete();
        knobs::set_mirror_ok(mirror_ok);
        if !mirror_ok && std::env::var_os("V41_EXPERT_MIRROR_DIR").is_some() {
            eprintln!("expertd: expert mirror incomplete: route=urgency will act as route=split");
        }
        for l in self.layers.iter_mut().flatten() {
            let cap = l.ids.len();

            l.owned.iter_mut().for_each(|o| *o = true);
            l.page = Some(LayerPager {
                requests: 0, misses: 0, read_ns: 0, h2d_ns: 0,
                pread_ns: 0, repack_cpu_ns: 0, repack_gpu_ns: 0,
            });
        }
        // Shard-wide pool, seeded from what `load` already placed. Slots are
        // ABSOLUTE throughout; `remap` holds `-(abs_slot)-1` so the kernel can be
        // handed the whole buffer (see `layer_views`).
        let n_slots = self.info.n_resident as usize;
        let frac = b2_pool_floor();
        let seeded: Vec<(u32, u32, &[u32])> = self
            .layers
            .iter()
            .enumerate()
            .filter_map(|(li, l)| l.as_ref().filter(|l| l.page.is_some()).map(|l| (li as u32, l.base_slot, &l.ids[..])))
            .collect();
        eprintln!(
            "expert shard: global pool {} (floor {:.2} = {} slots on a 260-slot layer)",
            if b2_global_pool() { "ON" } else { "OFF" },
            frac,
            (260.0 * frac) as u32
        );
        let mut pool = ShardPool::seeded(n_slots, &seeded, frac);
        // Prefill staging band (the block above `PIN_RESERVE_MIN`): the pin
        // budget a connection will get is stated here once, at startup.
        let stage = pool.set_stage(b2_prefill_stage());
        if b2_mode_evict() {
            if stage == 0 && b2_scan_class() && b2_global_pool() {
                pool.enable_mode_evict_live();
                pool.me.restore_on = b2_restore();
                eprintln!("expertd: mode-aware eviction ON (prefill budget {} decode victims per phase; delta restore {})",
                    b2_prefill_budget(), if pool.me.restore_on { "ON" } else { "off" });
            } else {
                eprintln!("expertd: V41_B2_MODE_EVICT=1 IGNORED: needs V41_B2_PREFILL_STAGE=0, the two-class LRU (V41_B2_SCAN_CLASS != 0) and the global pool (V41_B2_GLOBAL_POOL != 0)");
            }
        }
        let floors: usize = pool.floor.iter().map(|&f| f as usize).sum();
        let reserve = b2_pin_reserve(stage);
        eprintln!(
            "expertd: prefill staging {} ({stage} slots [{}, {n_slots}) of {n_slots}; main band {}; pin budget {} = {n_slots} - {stage} - reserve {reserve} - floors {floors})",
            if stage > 0 { "ON" } else { "OFF" },
            n_slots - stage,
            n_slots - stage,
            n_slots.saturating_sub(stage + reserve + floors),
        );
        self.pool = Some(pool);
        // Do NOT touch the advertised HELLO bitmap. `info.owned` is what the hub's
        // PREFILL path uses for its remote exclusion, and prefill's per-layer union
        // (~203 experts at B=1024) would not fit a catch-all region (154 slots), so
        // advertising all-true makes prefill hand us more than we can hold at once.
        // Catch-all is a DECODE-side decision on the hub ("not resident here =>
        // yours"); all this side needs is `l.owned` all-true above, so the executor
        // accepts an expert outside the advertised set and `ensure_layer` pages it.
        //
        // MEASURED 2026-09-13: advertising all-true also routes ALL prefill MoE
        // here, which was 159 -> 203 tok/s at 6k context (box 1's iGPU freed
        // entirely). That is a real and separate win, but it needs per-layer
        // capacity >= the prefill union, not this flag.
        Ok(())
    }

    pub fn is_paged(&self) -> bool {
        self.layers.iter().flatten().next().is_some_and(|l| l.page.is_some())
    }

    /// Make every id in `ids` resident in `layer`'s region, paging from this box's
    /// own disk. No-op for a pinned (non-paged) shard. MUST be called before
    /// `layer_views` for a paged shard: the MoE kernel reads `remap[e]`, and a
    /// non-resident id still reads 0 = "the other device takes it", which would
    /// silently drop the expert.
    /// As [`Self::ensure_layer`], additionally appending every expert id it had to
    /// PAGE (i.e. that missed) to `missed`.
    ///
    /// The hub uses this to make box 1's decode residency EXCLUSIVE: box 1 caches
    /// what box 2 evicted, so the picks it later serves are ones box 2 would miss
    /// (6.6 ms) rather than hit (87 us). Measured break-even from the stage diff
    /// in `WHY_THE_BIG_POOL_REGRESSED.md`: box 1 costs 140 us/expert, so serving a
    /// box-2 HIT is -53 us and serving a box-2 MISS is +6547 us.
    pub fn ensure_layer_reporting(
        &mut self,
        layer: u32,
        ids: &[i32],
        missed: &mut Vec<u32>,
    ) -> eyre::Result<()> {
        self.ensure_layer_inner(layer, ids, Some(missed), false)
    }

    /// `prefill_shaped`: this request sweeps a layer's union rather than a
    /// token's six picks, so eviction must stay inside the layer's own region.
    pub fn ensure_layer_phased(&mut self, layer: u32, ids: &[i32], prefill_shaped: bool) -> eyre::Result<()> {
        self.ensure_layer_inner(layer, ids, None, prefill_shaped)
    }

    pub fn ensure_layer(&mut self, layer: u32, ids: &[i32]) -> eyre::Result<()> {
        self.ensure_layer_inner(layer, ids, None, false)
    }

    /// Is `layer` a PAGED layer (catch-all pool)? Hits-first only applies there:
    /// an unpaged layer is fully resident by construction.
    /// Resident in the paged pool right now (false for unpaged layers).
    pub fn is_resident_pool(&self, layer: u32, e: u32) -> bool {
        self.pool.as_ref().is_some_and(|p| p.slot_of.contains_key(&(layer, e)))
    }

    /// Prefill staging slots (0 = off / no pool).
    pub fn stage_slots(&self) -> usize {
        self.pool.as_ref().map_or(0, |p| p.stage_slots())
    }

    /// Cumulative prefill-staging counters (zero without a pool).
    pub fn stage_counters(&self) -> StageCounters {
        self.pool.as_ref().map_or_else(StageCounters::default, |p| p.sc)
    }

    pub fn layer_is_paged(&self, layer: u32) -> bool {
        self.pool.is_some()
            && self.layers.get(layer as usize).and_then(|l| l.as_ref()).is_some_and(|l| l.page.is_some())
    }

    /// `REQ_FLAG_RESID`: which experts of `layer` this box could serve RIGHT NOW
    /// without a read, as `RESID_WORDS` u32s (bit e = expert e). Paged layers:
    /// a landed pool slot (`remap_hosts[layer][e] != 0`; a claimed but unlanded
    /// slot is still 0). Unpaged layers: the static assignment.
    pub fn residency_words(&self, layer: u32) -> [u32; proto::RESID_WORDS] {
        let mut w = [0u32; proto::RESID_WORDS];
        let paged = if self.layer_is_paged(layer) {
            self.pool.as_ref().and_then(|p| p.remap_hosts.get(layer as usize))
        } else {
            None
        };
        for e in 0..N_EXPERT as usize {
            let here = match paged {
                Some(r) => r[e] != 0,
                None => self.owns(layer, e as i32),
            };
            if here {
                w[e / 32] |= 1 << (e % 32);
            }
        }
        w
    }

    /// A new connection: the previous hub's pins are void (pinning block
    /// comment above `PinBook`).
    pub fn pin_reset(&mut self) {
        if let Some(p) = self.pool.as_mut() {
            p.pins = PinBook::off();
        }
        self.early_paged.clear();
    }

    /// The hub asked for pins (`REQ_FLAG_PIN`): turn them on for this
    /// connection (idempotent). Budget = pool slots - the prefill staging band
    /// - `b2_pin_reserve()` - the layer floors (the pinning block above
    /// `PIN_RESERVE_MIN`). False for an unpaged shard, which never evicts
    /// anyway.
    pub fn pin_enable(&mut self) -> bool {
        let Some(p) = self.pool.as_mut() else { return false };
        if !p.pins.on {
            let n = p.owner_of.len();
            let stage = p.stage_slots();
            let floors: usize = p.floor.iter().map(|&f| f as usize).sum();
            let r = b2_pin_reserve(stage);
            let budget = n.saturating_sub(stage + r + floors) as u32;
            p.pins.enable(budget);
            eprintln!(
                "expertd: pinning ON for this connection: budget {budget} of {n} slots (staging {stage}, reserve {r}, floors {floors}; \
                 no-deadlock minimum reserve + staging >= {PIN_RESERVE_MIN}){}",
                if b2_assert_pinned() { ", V41_B2_ASSERT_PINNED" } else { "" }
            );
        }
        true
    }

    pub fn pin_on(&self) -> bool {
        self.pool.as_ref().is_some_and(|p| p.pins.on)
    }

    /// A request's RELEASE words (unpin, in order), then its PREFETCH words
    /// (the hub wants them resident: eligible to pin once landed). Call when
    /// the request is served, in arrival order.
    pub fn pin_apply_words(&mut self, release: &[u32], prefetch: &[u32]) {
        let Some(p) = self.pool.as_mut() else { return };
        if !p.pins.on {
            return;
        }
        // Every release word advances the epoch (the hub counts them all).
        for &w in release {
            p.pins.release(w);
        }
        // A grant only for a layer this shard pages and reports: a word for
        // any other layer (unvalidated client input) would sit eligible and
        // on `fresh` for the life of the connection.
        for &w in prefetch {
            let layer = w >> 16;
            if self.layer_is_paged(layer) {
                if let Some(p) = self.pool.as_mut() {
                    p.pins.grant(layer, w & 0xFFFF);
                }
            }
        }
    }

    /// A decode-shaped REQUEST's picks (`proto::PIN_DECODE_MAX_ROWS`, per
    /// request: a merged partner grants its own) become eligible to pin at
    /// the layer's next report. No-op unless pinning is on.
    pub fn pin_grant(&mut self, layer: u32, picks: &[i32]) {
        if !self.pin_on() || !self.layer_is_paged(layer) {
            return;
        }
        let Some(p) = self.pool.as_mut() else { return };
        for &e in picks {
            if (0..N_EXPERT as i32).contains(&e) {
                p.pins.grant(layer, e as u32);
            }
        }
    }

    /// Build the pin-mode reply for `layer` after its pass: a DECODE-shaped
    /// pass's `picks` become eligible (a prefill chunk's do not: see the block
    /// comment; a merged pass grants per request via `pin_grant` and passes
    /// none here), eligible landed experts are pinned within budget, and the
    /// layer's pinned set is returned as the map, with `[epoch, pinned,
    /// budget]`. An unpaged layer is its static assignment (never evicted).
    /// `None` unless pinning is on.
    pub fn pin_report(&mut self, layer: u32, picks: &[i32], decode_shaped: bool) -> Option<([u32; proto::RESID_WORDS], [u32; 3])> {
        let paged = self.layer_is_paged(layer);
        let static_map = if paged { None } else { Some(self.residency_words(layer)) };
        let p = self.pool.as_mut()?;
        if !p.pins.on {
            return None;
        }
        let map = match static_map {
            Some(m) => m,
            None => {
                if decode_shaped {
                    for &e in picks {
                        if (0..N_EXPERT as i32).contains(&e) {
                            p.pins.grant(layer, e as u32);
                        }
                    }
                }
                let row = &p.remap_hosts[layer as usize];
                p.pins.report(layer, row, p.stage)
            }
        };
        Some((map, [p.pins.epoch, p.pins.pinned, p.pins.budget]))
    }

    /// Experts of `layer` among `sel` that are NOT landed right now: the ones
    /// a pass starting now has to page or wait for (the reply's PAGED bits).
    /// Empty for an unpaged layer.
    pub fn paged_bits(&self, layer: u32, sel: &[i32]) -> [u32; proto::RESID_WORDS] {
        let mut w = [0u32; proto::RESID_WORDS];
        if !self.layer_is_paged(layer) {
            return w;
        }
        let Some(row) = self.pool.as_ref().and_then(|p| p.remap_hosts.get(layer as usize)) else { return w };
        for &e in sel {
            if (0..N_EXPERT as i32).contains(&e) && row[e as usize] == 0 {
                w[e as usize / 32] |= 1 << (e % 32);
            }
        }
        w
    }

    /// A request frame arrived (the early-page hook): remember which of its
    /// picks were not landed, for its reply's PAGED bits. Pin mode only.
    pub fn note_early_paged(&mut self, seq: u32, layer: u32, sel: &[i32]) {
        if !self.pin_on() {
            return;
        }
        let bits = self.paged_bits(layer, sel);
        self.early_paged.note(seq, bits);
    }

    /// The arrival-time paged bits of request `seq` (all zero if none were
    /// kept), consumed.
    pub fn take_early_paged(&mut self, seq: u32) -> [u32; proto::RESID_WORDS] {
        self.early_paged.take(seq)
    }

    /// This connection's pin counters and `(pinned, budget, epoch)`.
    pub fn pin_counters(&self) -> Option<(PinCounters, u32, u32, u32)> {
        let p = self.pool.as_ref()?;
        p.pins.on.then(|| (p.pins.c, p.pins.pinned, p.pins.budget, p.pins.epoch))
    }

    /// Is `(layer, e)` pinned by the hub?
    pub fn is_pinned(&self, layer: u32, e: u32) -> bool {
        self.pool.as_ref().is_some_and(|p| p.pins.is_pinned(layer, e))
    }

    /// Which of `ids` are resident on `layer` RIGHT NOW, reading nothing. `NO_PICK`
    /// and out-of-range ids count as resident (nothing to fetch); an unpaged layer
    /// is all-resident. This is the split the hits-first executor launches on
    /// before `ensure_layer*` reads the rest (docs/v41/MULTISTREAM_DECODE_PLAN.md 4.1).
    pub fn resident_mask(&self, layer: u32, ids: &[i32], out: &mut Vec<bool>) {
        out.clear();
        if !self.layer_is_paged(layer) {
            out.resize(ids.len(), true);
            return;
        }
        let pool = self.pool.as_ref().expect("layer_is_paged checked the pool");
        out.extend(ids.iter().map(|&e| {
            !(0..N_EXPERT as i32).contains(&e) || pool.slot_of.contains_key(&(layer, e as u32))
        }));
    }

    /// Re-upload `layer`'s remap if another layer's eviction left it stale — the
    /// lazy upload `ensure_layer_inner` does first. Hits-first launches the
    /// resident experts BEFORE calling ensure, so it needs this on its own.
    pub fn sync_remap(&mut self, layer: u32) -> eyre::Result<()> {
        let Some(l) = self.layers.get_mut(layer as usize).and_then(|l| l.as_mut()) else { return Ok(()) };
        let Some(pool) = self.pool.as_mut() else { return Ok(()) };
        if pool.dirty[layer as usize] {
            l.remap_dev.copy_from_host(&pool.remap_hosts[layer as usize])?;
            pool.dirty[layer as usize] = false;
        }
        Ok(())
    }

    fn ensure_layer_inner(&mut self, layer: u32, ids: &[i32], mut missed: Option<&mut Vec<u32>>, prefill_shaped: bool) -> eyre::Result<()> {
        let pinned: Vec<(u32, u32)> = self.pinned.iter().chain(self.parked_pins.iter()).copied().collect();
        // `evtrace` (`b2_ensure` + one `b2_read` per demand miss): NaN when off.
        let ev_on = super::evtrace::enabled();
        let nan = f64::NAN;
        let ev_t0 = if ev_on { super::evtrace::now() } else { nan };
        let ev_q0 = match (ev_on, self.prefetch.as_ref()) {
            (true, Some(p)) => p.queue.counts(),
            _ => [nan; 4],
        };
        self.ev_admit = [0; 5];
        // Land completed look-ahead prefetches first (any layer): nothing reads
        // the pool here, and this layer's picks are protected from eviction.
        if self.prefetch.is_some() {
            let want_pre: Vec<u32> = ids.iter().filter(|&&e| (0..N_EXPERT as i32).contains(&e)).map(|&e| e as u32).collect();
            self.admit_prefetched(layer, &want_pre)?;
        }
        let ev_admit = self.ev_admit;
        let ev_t_admit = if ev_on { super::evtrace::now() } else { nan };
        let Some(l) = self.layers.get_mut(layer as usize).and_then(|l| l.as_mut()) else {
            // Catch-all needs a region on EVERY layer the hub can send. An
            // encoder-only assignment (e.g. `L0-L19:...`) has none for layers
            // 20-39, so decode dies on the first decoder layer while prefill —
            // which under CED only runs layers 0-19 — looks fine.
            return Err(eyre!(
                "expert shard: layer {layer} has no region. Catch-all (`--paged`) requires an                  assignment spanning ALL {} layers, e.g. `all:lo-hi` or                  `L0-L19:a-383,L20-L39:b-383` — got layers {:?}",
                N_LAYER,
                self.layers.iter().enumerate().filter(|(_, l)| l.is_some()).map(|(i, _)| i).collect::<Vec<_>>(),
            ));
        };
        let Some(pg) = l.page.as_mut() else { return Ok(()) };
        let base = l.base_slot as usize;
        let n_region = l.ids.len();
        let Some(pool) = self.pool.as_mut() else {
            return Err(eyre!("expert shard: paged layer {layer} but no pool"));
        };
        // This layer's remap may be stale because ANOTHER layer evicted one of its
        // slots. Re-upload before anything reads it. Lazy on purpose: an eviction
        // never touches a foreign device buffer, only the host mirror + this flag.
        let ev_dirty_upload = pool.dirty[layer as usize];
        if pool.dirty[layer as usize] {
            l.remap_dev.copy_from_host(&pool.remap_hosts[layer as usize])?;
            pool.dirty[layer as usize] = false;
        }
        let ev_t_dirty = if ev_on { super::evtrace::now() } else { nan };
        // `evtrace`: staging claims / hits / spills across this call.
        let ev_sc0 = pool.sc;
        // ONE POOL for all layers. The per-layer carve was never load-bearing: it
        // is just the ownership count spread evenly across 40 layers, and grouping
        // residency by layer is not what the working set looks like -- a layer that
        // needs more than its equal share should take slots from a layer that needs
        // fewer, which is exactly what a single global LRU does. `V41_B2_GLOBAL_POOL=0`
        // restores the old per-layer search (region first, pool as fallback).
        // TWO-CLASS LRU (2026-09-22): a prefill chunk (`prefill_shaped`, b > 16)
        // is a scan of most of a layer's experts, and with box 1 owning the hot
        // set this pool is decode's COLD TIER, whose working set a 120 s burst
        // of chunks evicted entirely (decode then re-faulted 20-50 experts per
        // step for a minute). Slots paged by prefill are stamped older than
        // every decode-touched slot (their own LRU order kept, so prefill still
        // reuses across chunks); prefill hits do not refresh a decode slot;
        // a decode hit promotes. Victims come from the prefill class first and
        // only then from decode's, so decode's set survives a burst intact.
        // `V41_B2_SCAN_CLASS=0` restores the single LRU.
        // Mode-aware eviction: inside the hub's prefill phase every claim is a
        // prefill claim (a small prompt tail too); otherwise the request shape.
        let prefill_mode = if pool.me.on { pool.me.prefill_phase && self.req_prefill } else { prefill_shaped };
        let scan_class = prefill_mode && b2_scan_class();
        let global = b2_global_pool();
        let r = &mut self.routed;
        // Disjoint field borrows, hoisted: the per-role read closures below must
        // capture `owner` alone, not `&self`, or they collide with `&mut stage`.
        let owner = &self.owner;
        let repack = self.repack.as_ref();
        let repack_stream = self.repack_stream.as_ref();
        let stages = &mut self.stages;
        let direct = self.direct;
        let prefetch_q = self.prefetch.as_ref().map(|p| &*p.queue);
        let mut dirty = false;
        let mut want: Vec<u32> = Vec::with_capacity(ids.len());
        for &e in ids {
            if !(0..N_EXPERT as i32).contains(&e) { continue; }
            let e = e as u32;
            if !want.contains(&e) { want.push(e); }
        }
        let bpe = [r.gate_bytes_per_expert, r.up_bytes_per_expert, r.down_bytes_per_expert];
        let mut pending: Vec<(u32, u32)> = Vec::new();
        // `evtrace`, parallel to `pending`: victim-search ns and the evicted owner.
        let mut ev_miss: Vec<(f64, Option<(u32, u32)>)> = Vec::new();
        let (mut ev_hits, mut ev_scan_ns, mut ev_foreign, mut ev_free) = (0u32, 0f64, 0u32, 0u32);
        // First error of the call. Victim search and reads stop on it, and every
        // slot claimed but not yet landed is rolled back below, so the pool never
        // reports an expert resident that nobody wrote.
        let mut failed: Option<eyre::Report> = None;
        let region = (base as u32, base as u32 + n_region as u32);
        for &e in &want {
            pg.requests += 1;
            if pool.touch_hit(layer, e, scan_class) {
                ev_hits += 1;
                continue;
            }
            pg.misses += 1;
            if let Some(m) = missed.as_deref_mut() {
                m.push(e);
            }
            let ev_t_scan = std::time::Instant::now();
            let claim = pool.claim_miss(layer, e, &want, &pinned, region, global, prefill_mode, scan_class);
            let ev_scan = ev_t_scan.elapsed().as_nanos() as f64;
            ev_scan_ns += ev_scan;
            let Some((victim, ev_victim)) = claim else {
                failed = Some(eyre!(
                    "expert shard: layer {layer} has no evictable slot anywhere \
                     (want {} > region {n_region}, pool {} slots)",
                    want.len(),
                    pool.owner_of.len(),
                ));
                break;
            };
            match ev_victim {
                Some((ol, _)) => ev_foreign += u32::from(ol != layer),
                None => ev_free += 1,
            }
            ev_miss.push((ev_scan, ev_victim));
            pending.push((e, victim));
        }
        // Read the misses `stages.len()` at a time, concurrently (MEASURED on box
        // 2's Crucial E100 2026-09-20: random 18.8 MB O_DIRECT reads aggregate
        // 2.5 GB/s at 1 reader, 3.4 at 4, and fall off beyond; one reader is
        // what the serial loop got). Uploads/repacks follow in order on the
        // repack stream.
        let gpu_repack = repack.is_some();
        let (mut read_ns, mut h2d_ns) = (0u64, 0u64);
        // Runtime-capped (`knobs::miss_par`): the staging sets are sized once at
        // startup by `V41_B2_MISS_PAR`, but the CONCURRENCY can be lowered live.
        // The E100 loses ~25% of its aggregate bandwidth past ~8 outstanding reads
        // and a single miss already issues 3 role reads split across both drives,
        // so 4 concurrent misses put ~24 on the primary drive (2026-09-22). Under
        // `route=urgency` demand misses read wholly from the mirror instead.
        let k = knobs::miss_par().min(stages.len()).max(1);
        let ev_t_victims = if ev_on { super::evtrace::now() } else { nan };
        let mut ev_chunks = 0u32;
        for (ev_ci, chunk) in pending.chunks(k).enumerate() {
            if failed.is_some() {
                break;
            }
            ev_chunks += 1;
            let rp0 = v4flash_core::hf_v41::expert_read_profile();
            let t_r = std::time::Instant::now();
            let ev_t_chunk = if ev_on { super::evtrace::now() } else { nan };
            let ev_q_chunk = match (ev_on, prefetch_q) {
                (true, Some(q)) => q.counts(),
                _ => [nan; 4],
            };
            // (offsets, coalesced, evtrace: read start, read end, role (start, end) x3)
            type R = Result<([Option<(usize, usize, u32, u32)>; 3], bool, [f64; 8]), String>;
            // Demand reads: wholly from the mirror under urgency routing --
            // except a PREFILL-shaped pass's, striped across both drives
            // (`knobs::prefill_route_split`, default): a chunk's ~200 reads
            // per layer otherwise monopolise the SN5000 alongside decode's
            // demand reads while the E100 idles.
            let route = if knobs::route_urgency() && !(prefill_shaped && knobs::prefill_route_split()) {
                v4flash_core::hf_v41::ExpertRoute::mirror_only()
            } else {
                v4flash_core::hf_v41::ExpertRoute::split()
            };
            // Raw pointers into the persistent pinned staging (like the prefetch
            // thread's `SetPtr`): the reader threads must not BORROW `stages`, or
            // the borrow lasts for the whole scope and no expert can repack until
            // every read in the chunk has joined. Each set is touched by exactly
            // one reader until its handle is joined, then only by this thread.
            let ptrs: Vec<SetPtr> = stages.iter().take(chunk.len()).map(|st| SetPtr {
                p: [st[0].as_slice().as_ptr() as *mut u8, st[1].as_slice().as_ptr() as *mut u8, st[2].as_slice().as_ptr() as *mut u8],
                n: [st[0].len(), st[1].len(), st[2].len()],
            }).collect();
            DEMAND_READS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let scoped: eyre::Result<()> = std::thread::scope(|sc| {
                let mut hs = Vec::with_capacity(chunk.len());
                for (&(e, _), &sp) in chunk.iter().zip(ptrs.iter()) {
                    hs.push(sc.spawn(move || -> R {
                        let sp = sp; // capture the whole (Send) wrapper, not its pointer field
                        // SAFETY: see `ptrs`; the set is exclusively this thread's until joined.
                        let (b0, b1, b2) = unsafe {
                            (std::slice::from_raw_parts_mut(sp.p[0], sp.n[0]), std::slice::from_raw_parts_mut(sp.p[1], sp.n[1]), std::slice::from_raw_parts_mut(sp.p[2], sp.n[2]))
                        };
                        let t0 = if ev_on { super::evtrace::now() } else { f64::NAN };
                        let r = Self::read_miss_into(owner, direct, gpu_repack, layer, e, bpe, b0, b1, b2, route);
                        let mut ev = [f64::NAN; 8];
                        if ev_on {
                            ev[0] = t0;
                            ev[1] = super::evtrace::now();
                            ev[2..].copy_from_slice(&EV_ROLES.with(|c| c.replace([f64::NAN; 6])));
                        }
                        r.map(|(offs, coalesced)| (offs, coalesced, ev)).map_err(|err| format!("{err:#}"))
                    }));
                }
                // Join in spawn order and repack each expert as soon as ITS read
                // has landed (2026-09-21): with join-all-then-repack-all, one slow
                // read held every expert's repack in the chunk, and the repacks
                // then ran back to back after the last read instead of under the
                // others.
                for (j, h) in hs.into_iter().enumerate() {
                    let (e, victim) = chunk[j];
                    let res: R = h.join().unwrap_or_else(|_| Err("miss reader panicked".into()));
                    let (offs, coalesced, ev_rd) = res.map_err(|m| eyre!("expert shard: layer {layer} expert {e}: {m}"))?;
                    let st = &stages[j];
                    let t_h = std::time::Instant::now();
                    let ev_t_h = if ev_on { super::evtrace::now() } else { f64::NAN };
                    match (repack, repack_stream) {
                        (Some(rp), Some(rs)) => {
                            pg.repack_gpu_ns += Self::repack_in_place(rp, rs, r, victim, st, &offs, coalesced)?;
                        }
                        _ => {
                            for i in 0..3 {
                                let buf = match i { 0 => &mut r.gate.buffer, 1 => &mut r.up.buffer, _ => &mut r.down.buffer };
                                // `st[i]` is over-allocated by 4 blocks of alignment slack
                                // for the O_DIRECT path, so copy only the expert's bytes.
                                buf.slice_view_mut(victim as usize * bpe[i], bpe[i])
                                    .copy_from_host(&st[i].as_slice()[..bpe[i]])?;
                            }
                        }
                    }
                    let ev_repack_ns = t_h.elapsed().as_nanos() as u64;
                    h2d_ns += ev_repack_ns;
                    pool.commit(layer, e, victim);
                    dirty = true;
                    if ev_on {
                        let (scan, vic) = ev_miss.get(ev_ci * k + j).copied().unwrap_or((f64::NAN, None));
                        let (vl, ve) = vic.map_or((f64::NAN, f64::NAN), |(l, e)| (f64::from(l), f64::from(e)));
                        let mut v = vec![
                            0.0, ev_cur_seq(), f64::from(layer), f64::from(e), f64::from(victim), vl, ve, j as f64,
                            ev_t0, ev_t_chunk, ev_rd[0], ev_rd[1], ev_t_h, ev_t_h, super::evtrace::now(),
                            f64::NAN, f64::NAN, f64::NAN, ev_q_chunk[0], ev_q_chunk[1],
                        ];
                        v.extend_from_slice(&ev_rd[2..8]);
                        v.extend_from_slice(&[
                            1.0, 1.0, f64::from(u8::from(coalesced)), scan, ev_repack_ns as f64,
                            chunk.len() as f64, ev_ci as f64, 0.0, route.code(),
                        ]);
                        super::evtrace::emit(&super::evtrace_kinds::B2_READ, &v);
                    }
                }
                Ok(())
            });
            DEMAND_READS.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            read_ns += t_r.elapsed().as_nanos() as u64;
            let rp1 = v4flash_core::hf_v41::expert_read_profile();
            pg.pread_ns += rp1.2 - rp0.2;
            pg.repack_cpu_ns += rp1.3 - rp0.3;
            if let Err(err) = scoped {
                failed = Some(err);
            }
        }
        pg.read_ns += read_ns;
        pg.h2d_ns += h2d_ns;
        if let Some(err) = failed {
            // Roll back every claim whose data never landed: its remap entry is
            // written only after the read + upload succeed, so an entry that
            // does not point at its victim was never committed. The slot is
            // freed and aged to the front of the LRU (its device bytes may be
            // half-written, but nothing maps it any more).
            for &(e, victim) in &pending {
                if pool.remap_hosts[layer as usize][e as usize] != -(victim as i32) - 1 {
                    pool.unclaim(layer, e, victim);
                }
            }
            // This layer's device remap may still name an evicted victim whose
            // bytes were being overwritten, and the upload below is skipped:
            // force a re-upload before anything reads it.
            pool.dirty[layer as usize] = true;
            return Err(err);
        }
        let ev_t_reads = if ev_on { super::evtrace::now() } else { nan };
        let ev_t_up = std::time::Instant::now();
        if dirty {
            l.remap_dev.copy_from_host(&pool.remap_hosts[layer as usize])?;
            pool.dirty[layer as usize] = false;
        }
        if ev_on {
            let n_miss = pending.len();
            super::evtrace::emit(&super::evtrace_kinds::B2_ENSURE, &[
                ev_cur_seq(), f64::from(layer), ids.len() as f64, want.len() as f64, f64::from(ev_hits), n_miss as f64,
                f64::from(u8::from(prefill_shaped)),
                ev_t0, ev_t_admit, ev_t_dirty, ev_t_victims, ev_t_reads, super::evtrace::now(),
                ev_admit[0] as f64, ev_admit[1] as f64, ev_admit[2] as f64, ev_admit[3] as f64, ev_admit[4] as f64,
                ev_scan_ns, f64::from(ev_foreign), f64::from(ev_free), k as f64, f64::from(ev_chunks),
                f64::from(u8::from(ev_dirty_upload)), if dirty { ev_t_up.elapsed().as_nanos() as f64 } else { 0.0 },
                ev_q0[0], ev_q0[1], ev_q0[2], ev_q0[3],
                (pool.sc.claims - ev_sc0.claims) as f64, (pool.sc.hits - ev_sc0.hits) as f64,
                (pool.sc.spill_in + pool.sc.spill_out - ev_sc0.spill_in - ev_sc0.spill_out) as f64,
            ]);
        }
        Ok(())
    }


    /// Read ONE missing expert's three roles into a staging set (`b0..b2`, the
    /// caller's pinned buffers). Pure I/O + CPU: safe to run for several misses
    /// concurrently (`V41_B2_MISS_PAR`). Returns the per-role staging offsets
    /// and whether the coalesced (two-pread) path was taken.
    #[allow(clippy::too_many_arguments)]
    fn read_miss_into(
        owner: &V41HfWeights,
        direct: bool,
        gpu_repack: bool,
        layer: u32,
        e: u32,
        bpe: [usize; 3],
        b0: &mut [u8],
        b1: &mut [u8],
        b2: &mut [u8],
        route: v4flash_core::hf_v41::ExpertRoute,
    ) -> eyre::Result<([Option<(usize, usize, u32, u32)>; 3], bool)> {
        {
            let names = [
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
                format!("blk.{layer}.ffn_down_exps.weight"),
            ];
            // The three roles are read CONCURRENTLY into PERSISTENT staging, then
            // uploaded — and under `V41_B2_GPU_REPACK` (default on) the MXFP4
            // HF->ggml permute runs on the iGPU instead of this box's CPU.
            //
            // This loop used to be strictly serial — read role i into one staging buffer,
            // blocking H2D, then role i+1 — so ~18.8 MB moved at ~1.9 GB/s and a miss cost
            // **11.09 ms** (box 2's own page stats: read 9.84 + h2d 1.26). At the measured
            // ~41 faults/token that is ~450 ms of the decode token, i.e. the single
            // largest cost in the engine. Box 1 has read experts with parallel preads
            // since the M7 pager (`read_range_into_cached_par`); this side never did.
            //
            // Concurrency got read to 8.30 ms, and there it stuck — because 8.30 was
            // never the drive. MEASURED with O_DIRECT on box 2's own NVMe, the same
            // 20.2 MB random read takes **4.52 ms** (4.47 GB/s) and does NOT improve
            // with queue depth 2..16, so the drive is already saturated by one
            // expert-sized read. The missing 3.8 ms was CPU: a fresh 18.8 MB
            // allocation per fault plus the HF->ggml repack, both of which box 1's
            // pager had already moved off the critical path.
            // Per role: where the packed nibbles and the scale plane actually
            // landed in staging. `None` = contiguous (the non-direct paths).
            let mut offs: [Option<(usize, usize, u32, u32)>; 3] = [None; 3];
            // COALESCED: two preads for all three roles. Falls through to the
            // per-role path when disabled, when this build has no GPU repack /
            // O_DIRECT, or when the run-time contiguity check fails.
            let mut coalesced = false;
            // The coalesced span read is primary-only: never for a read routed
            // to the mirror (split and primary-only reads may coalesce).
            if gpu_repack && direct && b2_coalesce() && route.code() != 1.0 {
                let (dw, ds) = (&mut *b0, &mut *b1);
                let src0 = WeightSrc::from(owner);
                // All three ROLE tensors: the run's physical order is derived from
                // their file offsets, because gate/up/down is NOT w1/w2/w3.
                let t0 = src0.tensor(&names[0]).ok_or_else(|| eyre!("missing {}", names[0]))?;
                let t1 = src0.tensor(&names[1]).ok_or_else(|| eyre!("missing {}", names[1]))?;
                let t2 = src0.tensor(&names[2]).ok_or_else(|| eyre!("missing {}", names[2]))?;
                if let Some(o) = src0
                    .read_expert_runs_direct([&t0, &t1, &t2], e as usize, dw, ds)
                    .map_err(|err| eyre!("expert shard: layer {layer} expert {e}: {err}"))?
                {
                    for i in 0..3 {
                        offs[i] = Some(o[i]);
                    }
                    coalesced = true;
                }
            }
            // `V41_B2_COALESCE_CHECK=1`: verify a coalesced read against the
            // per-role read of the SAME expert, byte for byte. Run it at
            // `V41_B2_POOL_FLOOR=0`, where page-ins are frequent — that is the
            // regime where coalescing was observed to corrupt.
            if coalesced && coalesce_check() {
                let ref_buf = &mut *b2;
                let src0 = WeightSrc::from(owner);
                for r in 0..3 {
                    let name = &names[r];
                    let t = src0.tensor(name).ok_or_else(|| eyre!("missing {name}"))?;
                    let Some((pw2, ps2, out2, nb2)) =
                        src0.read_expert_hf_layout_direct(&t, e as usize, ref_buf)?
                    else {
                        continue;
                    };
                    let (po, so, out1, nb1) = offs[r].expect("coalesced offs");
                    let (plen, slen) = (out2 as usize * nb2 as usize * 16, out2 as usize * nb2 as usize);
                    if (out1, nb1) != (out2, nb2) {
                        eprintln!("COALESCE_CHECK L{layer} e{e} role{r}: geom {:?} != {:?}",
                                  (out1, nb1), (out2, nb2));
                    }
                    let got_p = &b0[po..po + plen];
                    let ref_p = &ref_buf[pw2..pw2 + plen];
                    if got_p != ref_p {
                        let i = got_p.iter().zip(ref_p).position(|(a, b)| a != b).unwrap_or(0);
                        eprintln!("COALESCE_CHECK L{layer} e{e} role{r}: PACKED differs at byte {i}                                    of {plen} (coalesced off {po}, per-role off {pw2})");
                    }
                    let got_s = &b1[so..so + slen];
                    let ref_s = &ref_buf[ps2..ps2 + slen];
                    if got_s != ref_s {
                        let i = got_s.iter().zip(ref_s).position(|(a, b)| a != b).unwrap_or(0);
                        eprintln!("COALESCE_CHECK L{layer} e{e} role{r}: SCALE differs at byte {i}                                    of {slen} (coalesced off {so}, per-role off {ps2})");
                    }
                }
            }
            if !coalesced {
                let bufs: [&mut [u8]; 3] = [b0, b1, b2];
                let mut errs: Vec<String> = Vec::new();
                // The caller's background mark (io_throttle) must reach the role
                // threads: thread-locals do not cross `spawn`.
                let bg = v4flash_core::io_throttle::background();
                // `evtrace`: each role thread's (start, end) raw ns.
                let ev_on = super::evtrace::enabled();
                let ev_roles: [std::sync::atomic::AtomicU64; 6] = Default::default();
                let ev_roles_ref = &ev_roles;
                std::thread::scope(|sc| {
                    let h: Vec<_> = bufs
                        .into_iter()
                        .enumerate()
                        .map(|(i, buf): (usize, &mut [u8])| {
                            let name = names[i].clone();
                            let src = WeightSrc::from(owner);
                            let bi = bpe[i];
                            type R = Result<Option<(usize, usize, u32, u32)>, String>;
                            sc.spawn(move || -> R {
                                if ev_on {
                                    ev_roles_ref[2 * i].store(monotonic_raw_ns(), std::sync::atomic::Ordering::Relaxed);
                                }
                                let r = (move || -> R {
                                v4flash_core::io_throttle::set_background(bg);
                                let t = src.tensor(&name).ok_or(format!("missing {name}"))?;
                                if gpu_repack && direct {
                                    // Zero-copy: O_DIRECT lands each region at its
                                    // own 4096-residue, straight into GTT staging.
                                    if let Some(o) = src
                                        .read_expert_hf_layout_direct_routed(t, e as usize, buf, route)
                                        .map_err(|err| format!("{name}: {err}"))?
                                    {
                                        return Ok(Some(o));
                                    }
                                }
                                if gpu_repack {
                                    // Leave MXFP4 in the HF layout; permuted below.
                                    src.read_expert_hf_layout(t, e as usize, &mut buf[..bi])
                                        .map(|_| None)
                                        .map_err(|err| format!("{name}: {err}"))
                                } else {
                                    src.read_expert_into(t, e as usize, &mut buf[..bi])
                                        .map(|_| None)
                                        .map_err(|err| format!("{name}: {err}"))
                                }
                                })();
                                if ev_on {
                                    ev_roles_ref[2 * i + 1].store(monotonic_raw_ns(), std::sync::atomic::Ordering::Relaxed);
                                }
                                r
                            })
                        })
                        .collect();
                    for (i, j) in h.into_iter().enumerate() {
                        match j.join() {
                            Ok(Ok(o)) => offs[i] = o,
                            Ok(Err(msg)) => errs.push(msg),
                            Err(_) => errs.push(format!("reader thread {i} panicked")),
                        }
                    }
                });
                if ev_on {
                    let mut t = [f64::NAN; 6];
                    for (d, a) in t.iter_mut().zip(ev_roles.iter()) {
                        let v = a.load(std::sync::atomic::Ordering::Relaxed);
                        if v != 0 { *d = v as f64; }
                    }
                    EV_ROLES.with(|c| c.set(t));
                }
                if let Some(msg) = errs.first() {
                    return Err(eyre!("expert shard: layer {layer} expert {e}: {msg}"));
                }
            }
            Ok((offs, coalesced))
        }
    }

    /// Permute the three staged HF-layout roles into `slot` on the iGPU, reading
    /// the staging buffers IN PLACE. Returns the ns spent waiting on the repack
    /// stream. Differs from `ExpertPager::upload_and_repack` on purpose: that one
    /// uploads to device scratch first because box 1's pager also serves the
    /// dGPU, which has its own VRAM. Box 2 is APU-only, so the upload is pure
    /// waste — see `stage`.
    fn repack_in_place(
        rp: &Mxfp4Repack,
        st: &v4flash_hip::Stream,
        routed: &mut RoutedExpertWeights,
        slot: u32,
        stage: &[PinnedBuffer<u8>; 3],
        offs: &[Option<(usize, usize, u32, u32)>; 3],
        // Coalesced staging: packed bytes for EVERY role live in `stage[0]` and
        // scales in `stage[1]`, so the two bases differ from the per-role case.
        coalesced: bool,
    ) -> eyre::Result<u64> {
        // (rows, blocks per row) per role: gate/up are [N_FF_EXP, N_EMBD/32],
        // down is [N_EMBD, N_FF_EXP/32]. Same block count, different shape —
        // exactly the `role_kr` geometry this file already uses.
        let geom = [
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_FF_EXP as u32, (N_EMBD / 32) as u32),
            (N_EMBD as u32, (N_FF_EXP / 32) as u32),
        ];
        let bpe = [
            routed.gate_bytes_per_expert,
            routed.up_bytes_per_expert,
            routed.down_bytes_per_expert,
        ];
        for i in 0..3 {
            let (rows, nb) = geom[i];
            debug_assert_eq!(rows as usize * nb as usize * 17, bpe[i]);
            let dst = match i {
                0 => &mut routed.gate.buffer,
                1 => &mut routed.up.buffer,
                _ => &mut routed.down.buffer,
            };
            // No upload: `stage[i]` is hipHostMalloc memory, which on this APU is
            // the same physical RAM the iGPU reads. The preads above already put
            // the bytes where the kernel wants them.
            let (pbase, sbase) = if coalesced {
                (stage[0].device_ptr() as *mut u8, stage[1].device_ptr() as *mut u8)
            } else {
                (stage[i].device_ptr() as *mut u8, stage[i].device_ptr() as *mut u8)
            };
            match offs[i] {
                // Zero-copy O_DIRECT: the two regions sit at their own residues.
                Some((po, so, o_rows, o_nb)) => {
                    debug_assert_eq!((o_rows, o_nb), (rows, nb));
                    rp.launch_from_ptrs(
                        st, dst, slot as usize * bpe[i],
                        pbase.wrapping_add(po) as v4flash_hip::sys::hipDeviceptr_t,
                        sbase.wrapping_add(so) as v4flash_hip::sys::hipDeviceptr_t,
                        rows, nb,
                    )?;
                }
                // Cached read: scales follow the nibbles contiguously.
                None => rp.launch_from_ptr(
                    st, dst, slot as usize * bpe[i],
                    stage[i].device_ptr(), stage[i].len(), rows, nb,
                )?,
            }
        }
        let t = std::time::Instant::now();
        st.synchronize()?;
        Ok(t.elapsed().as_nanos() as u64)
    }

    /// `(requests, misses, read_ns, h2d_ns)` summed over layers.
    pub fn page_stats(&self) -> (u64, u64, u64, u64) {
        self.layers.iter().flatten().filter_map(|l| l.page.as_ref()).fold(
            (0, 0, 0, 0),
            |a, p| (a.0 + p.requests, a.1 + p.misses, a.2 + p.read_ns, a.3 + p.h2d_ns),
        )
    }

    /// `(pread_ns, repack_cpu_ns, repack_gpu_ns)` summed over layers — the
    /// sub-terms of `read_ns` plus the GPU permute, so a miss can be attributed.
    pub fn page_read_split(&self) -> (u64, u64, u64) {
        self.layers.iter().flatten().filter_map(|l| l.page.as_ref()).fold(
            (0, 0, 0),
            |a, p| (a.0 + p.pread_ns, a.1 + p.repack_cpu_ns, a.2 + p.repack_gpu_ns),
        )
    }
    pub fn has_layer(&self, layer: u32) -> bool {
        self.layers.get(layer as usize).is_some_and(|l| l.is_some())
    }
    pub fn owned_ids(&self, layer: u32) -> &[u32] {
        self.layers.get(layer as usize).and_then(|l| l.as_ref()).map_or(&[], |l| &l.ids)
    }

    /// The layer's gate/up/down buffers (views over its packed slot range) and
    /// its remap, for a mode-0 het-split launch with local slot ids.
    fn layer_views(&self, layer: u32) -> eyre::Result<(DeviceBuffer<u8>, DeviceBuffer<u8>, DeviceBuffer<u8>, &DeviceBuffer<i32>)> {
        let l = self
            .layers
            .get(layer as usize)
            .and_then(|l| l.as_ref())
            .ok_or_else(|| eyre!("expert shard: layer {layer} not resident"))?;
        let r = &self.routed;
        // The WHOLE pool, not this layer's slice: `remap_dev` holds ABSOLUTE slot
        // indices, so the kernel's `e = -remap-1` addresses the full buffer. The
        // old contiguous `[base_slot, base_slot+n)` slice is exactly what
        // REMOTE_EXPERTS.md called "the executor contract" blocking a global
        // pool; it was only ever a host-side convention, and the kernel never
        // cared where the slot lived.
        let _ = l;
        Ok((
            r.gate.buffer.slice_view(0, r.gate.buffer.len()),
            r.up.buffer.slice_view(0, r.up.buffer.len()),
            r.down.buffer.slice_view(0, r.down.buffer.len()),
            &l.remap_dev,
        ))
    }
}

// ---------------------------------------------------------------------------
// MoeExecutor: the iGPU compute for one request
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct ExecTiming {
    pub h2d: Duration,
    pub gpu: Duration,
    pub path_decode: bool,
    /// Work items launched: the builder's count, or its upper bound under
    /// `V41_MOE_WI_DEVCOUNT` (`batched_pass`).
    pub n_work_items: u32,
    /// Bitmask over the request's first `RESP_MISS_BITS` sel slots: bit i set =
    /// pick i had to be paged from this box's disk. Decode (b==1) only; see
    /// `proto::RESP_MISS_SHIFT`.
    pub miss_mask: u32,
    /// Hits-first: distinct experts this request had to page, and whether the
    /// two-pass path ran (misses > 0 on the batched path).
    pub n_missing: u32,
    pub two_pass: bool,
    /// The short batched chain ran (`knobs::fast_chain`, `b <= FAST_CHAIN_MAX_B`).
    pub fast_chain: bool,
    /// Pin mode's PAGED bits (`proto::RESP_FLAG_PIN`): experts of the layer
    /// this pass picked that were not landed when it started, i.e. that it had
    /// to page or wait for. Taken before anything can land or evict.
    pub paged: [u32; proto::RESID_WORDS],
}

/// Largest batched pass (rows) the short chain takes (`knobs::fast_chain`):
/// `b * N_EXPERT_USED` = 96 picks fit the fused builder's one work-group.
pub const FAST_CHAIN_MAX_B: usize = 16;

/// The fast chain's ONE per-request upload, for `b` rows:
/// `[sel i32 | ew f32 | pad to 256 | xq]`. Returns (ew offset, xq offset, total
/// bytes). The sel/ew head alone is what a hits-first pass B / reduce re-uploads.
fn fast_in_layout(b: usize) -> (usize, usize, usize) {
    let n = b * N_EXPERT_USED;
    let xq_off = (n * 8).next_multiple_of(256);
    (n * 4, xq_off, xq_off + b * XQ_BYTES_PER_TOKEN)
}

/// Non-owning views of one request's device inputs: the old chain's own
/// `xq` / `d_selected` / `d_ew` buffers, or the fast chain's single upload.
struct PassIo {
    xq: DeviceBuffer<u8>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
}

/// Per-request kernel geometry shared by the by-expert passes.
struct PassGeo {
    b: usize,
    gbound: u32,
    cap: u32,
    gbpe: u32,
    ubpe: u32,
    dbpe: u32,
    gdt: GgufType,
    ddt: GgufType,
}

pub struct MoeExecutor {
    pub engine: DeviceEngine,
    device: Device,
    rows: usize,
    decode_max_b: usize,
    /// Reused across requests so the decode miss report never allocates.
    missed_scratch: Vec<u32>,
    /// Hits-first scratch: per-pick residency and the distinct missing ids.
    resident_scratch: Vec<bool>,
    missing_scratch: Vec<i32>,
    xq: DeviceBuffer<u8>,
    d_selected: DeviceBuffer<i32>,
    d_ew: DeviceBuffer<f32>,
    /// Pinned host staging for the per-request uploads, so they can be queued
    /// with `hipMemcpyAsync` on the compute stream instead of three blocking
    /// `hipMemcpy`s (measured 33 us for one 5.9 KB request, roadmap item 4).
    /// Safe to reuse per request: the previous request's stream work has
    /// completed before its response was read back.
    xq_pin: PinnedBuffer<u8>,
    sel_pin: PinnedBuffer<i32>,
    ew_pin: PinnedBuffer<f32>,
    group_count: DeviceBuffer<i32>,
    expert_members: DeviceBuffer<i32>,
    /// Group-id space the batched builder is currently sized for.
    ///
    /// **This MUST equal the slot space `remap` encodes, not `N_EXPERT`.** Box 2's
    /// remap has carried ABSOLUTE pool slots since 2026-09-14 (`-(abs_slot)-1`,
    /// see `LayerShard::remap_dev`) so a layer can hold a slot outside its own
    /// region, and `layer_views` hands the kernel the WHOLE pool. But
    /// `moe_group_builder.hip:118` drops any `g >= n_expert`, and this was being
    /// passed `N_EXPERT` (384) against a 6160-slot pool -- so every routed pick
    /// living above slot 383 was SILENTLY DROPPED from the group build while the
    /// reducer still counted it as ours (`remap[e] < 0`) and summed its zeroed
    /// partial row. Net: the expert contributed exactly 0.0, no error anywhere.
    /// B=1 decode was spared (no group builder); every prefill chunk and every
    /// DSpark verify batch was affected.
    group_bound: u32,
    work_items: DeviceBuffer<i32>,
    n_work_items: DeviceBuffer<i32>,
    d_mid_cat: DeviceBuffer<f32>,
    d_midq_cat: DeviceBuffer<u8>,
    partials: DeviceBuffer<f32>,
    pub ffn_moe: DeviceBuffer<f32>,
    out16: DeviceBuffer<u16>,
    sel_host: Vec<i32>,
    ew_host: Vec<f32>,
    xq_host: Vec<u8>,
    warm_in: DeviceBuffer<f32>,
    warm_out: DeviceBuffer<u8>,
    /// Device-time bracket around a request's GPU work, for the perfetto track.
    ev: Option<(v4flash_hip::Event, v4flash_hip::Event)>,
    /// Recorded after a request's last launch; polled (not blocked on) so the
    /// compute thread keeps pulling the next queued frame -- and starting its
    /// misses -- for the whole GPU tail, not just once (2026-09-21: the early
    /// paging caught only 40% of non-resident experts because the next frame
    /// had usually not arrived at the single poll point).
    ev_done: v4flash_hip::Event,
    // --- The short batched chain (`knobs::fast_chain`) ---------------------
    fast_k: crate::b2_fast_chain::B2FastChain,
    /// Rows the fast chain can take on this executor (`FAST_CHAIN_MAX_B` or
    /// fewer on a small executor).
    fast_rows: usize,
    /// ONE device region per request (`fast_in_layout`) instead of three
    /// buffers and three copies; the passes read views into it.
    fin_dev: DeviceBuffer<u8>,
    /// Its pinned staging, three regions of `fast_in_layout(fast_rows).2`
    /// bytes for the same reason as `sel_pin` (pass A / pass B / the reduce).
    fin_pin: PinnedBuffer<u8>,
    /// Host-mapped f16 result: the fast reduce writes it directly when
    /// `reply_f16`, and `run_path`'s final event wait makes it host-visible.
    out16_pin: PinnedBuffer<u16>,
    /// Rows of `out16_pin` that hold the LAST `run_path`'s result (set only
    /// after its final event wait; 0 = `read_f16*` take the device path).
    out16_host_rows: usize,
    /// The next request's reply is f16 (the daemon sets it per request from
    /// `REQ_FLAG_RESP_F32`; default true).
    reply_f16: bool,
    /// `partials` rows `[lo, hi)` that may hold non-zero data (`lo >= hi` =
    /// all zero). The fast chain needs rows `[0, b*nu)` zero on entry and
    /// leaves them zero (its reduce re-zeroes what it consumed); the old chain
    /// memsets on entry and leaves its rows dirty, and a request that errors
    /// mid-chain leaves its rows dirty. Starts all-dirty (hipMalloc).
    p_dirty: (usize, usize),
    /// Partials memsets the fast chain had to issue (tests: 0 in steady state).
    fast_memsets: u64,
    /// Set while a request's GPU work is queued. Still set when the next
    /// `run_path` starts = the last one ERRORED with work possibly in flight
    /// (DMA from the pinned staging, kernels writing `partials`): synchronize
    /// before reusing anything.
    in_flight: bool,
}

impl MoeExecutor {
    /// Scratch for batches up to `rows` tokens (≈ 230 MB at 1024). Batches of
    /// `<= decode_max_b` tokens take the decode kernels one token at a time
    /// (4 launches/token, no host sync); larger ones the by-expert prefill
    /// chain (host readback of the work-item count, as in production).
    pub fn new(igpu: Device, rows: usize, decode_max_b: usize) -> eyre::Result<Self> {
        igpu.set_current()?;
        let arch = igpu.properties()?.gcn_arch_name;
        let engine = DeviceEngine::for_arch(igpu, &arch)?;
        let id = igpu.id;
        let nu = N_EXPERT_USED;
        let wi_len = N_EXPERT as usize + rows * nu;
        let fast_rows = rows.min(FAST_CHAIN_MAX_B);
        let fin_bytes = fast_in_layout(fast_rows).2;
        Ok(Self {
            missed_scratch: Vec::with_capacity(N_EXPERT_USED),
            resident_scratch: Vec::with_capacity(rows * nu),
            missing_scratch: Vec::with_capacity(rows * nu),
            engine,
            device: igpu,
            rows,
            decode_max_b,
            xq: DeviceBuffer::new(id, rows * XQ_BYTES_PER_TOKEN)?,
            d_selected: DeviceBuffer::new(id, rows * nu)?,
            d_ew: DeviceBuffer::new(id, rows * nu)?,
            xq_pin: PinnedBuffer::new(rows * XQ_BYTES_PER_TOKEN)?,
            // Three regions: pass A, pass B and the reduce each upload their own
            // sel/ew from pinned memory, and an earlier region's DMA may still be
            // queued when the next is written.
            sel_pin: PinnedBuffer::new(3 * rows * nu)?,
            ew_pin: PinnedBuffer::new(3 * rows * nu)?,
            group_count: DeviceBuffer::new(id, N_EXPERT as usize)?,
            expert_members: DeviceBuffer::new(id, N_EXPERT as usize * rows)?,
            group_bound: N_EXPERT,
            work_items: DeviceBuffer::new(id, wi_len)?,
            n_work_items: DeviceBuffer::new(id, 1)?,
            d_mid_cat: DeviceBuffer::new(id, rows * nu * N_FF_EXP as usize)?,
            d_midq_cat: DeviceBuffer::new(id, rows * nu * MIDQ_BYTES_PER_SLOT)?,
            partials: DeviceBuffer::new(id, rows * nu * N_EMBD as usize)?,
            ffn_moe: DeviceBuffer::new(id, rows * N_EMBD as usize)?,
            out16: DeviceBuffer::new(id, rows * N_EMBD as usize)?,
            sel_host: vec![SENTINEL_EXPERT; rows * nu],
            ew_host: vec![0.0; rows * nu],
            xq_host: Vec::new(),
            warm_in: DeviceBuffer::new(id, 256)?,
            warm_out: DeviceBuffer::new(id, BLOCK_Q8_K_BYTES)?,
            ev: None,
            ev_done: v4flash_hip::Event::new_no_timing()?,
            fast_k: crate::b2_fast_chain::B2FastChain::for_arch(&arch)?,
            fast_rows,
            fin_dev: DeviceBuffer::new(id, fin_bytes)?,
            fin_pin: PinnedBuffer::new(3 * fin_bytes)?,
            out16_pin: PinnedBuffer::new(fast_rows * N_EMBD as usize)?,
            out16_host_rows: 0,
            reply_f16: true,
            p_dirty: (0, rows * nu),
            fast_memsets: 0,
            in_flight: false,
        })
    }

    /// Reply format of the NEXT `run_path` (true = f16): under the fast chain
    /// an f16 reply is written by the reduce straight into pinned host memory.
    pub fn set_reply_f16(&mut self, on: bool) {
        self.reply_f16 = on;
    }

    /// Partials memsets the fast chain issued so far (after an old-chain pass,
    /// an error, or a larger batch than any before). 0 in steady state.
    pub fn fast_chain_memsets(&self) -> u64 {
        self.fast_memsets
    }

    /// Rows `[0, n)` of `partials` may now be dirty (hull with what already was).
    fn partials_mark_dirty(&mut self, n: usize) {
        let (lo, hi) = self.p_dirty;
        self.p_dirty = if lo >= hi { (0, n) } else { (0, hi.max(n)) };
    }

    /// Make rows `[0, n)` of `partials` zero: a stream-ordered memset of the
    /// part that may be dirty (usually none).
    fn partials_make_clean(&mut self, n: usize) -> eyre::Result<()> {
        let (lo, hi) = self.p_dirty;
        let end = n.min(hi);
        if lo < end {
            let row = N_EMBD as usize;
            self.partials.slice_view_mut(lo * row, (end - lo) * row).fill_zero_async(&self.engine.compute)?;
            self.fast_memsets += 1;
        }
        // Rows [0, end) are zero now: [0, lo) already were.
        self.p_dirty = if end >= hi || lo >= hi { (0, 0) } else { (lo.max(end), hi) };
        Ok(())
    }

    /// One trivial launch + sync (a single Q8_K block) so the iGPU never sits
    /// idle long enough to drop clocks between two layers' requests: measured
    /// 372 → 1043 µs of GPU time per B=1 request after a 1.5 ms idle gap
    /// (docs/v41/REMOTE_EXPERTS.md §5.3).
    pub fn keep_warm(&mut self) -> eyre::Result<()> {
        self.device.set_current()?;
        self.engine.q8k.launch(&self.engine.compute, &mut self.warm_out, &self.warm_in, 1)?;
        self.engine.compute.synchronize()
    }

    /// Allocate the device-time event pair so `run` brackets its GPU work
    /// (enables the daemon's `box2 igpu.compute` perfetto track).
    pub fn enable_device_timing(&mut self) -> eyre::Result<()> {
        self.device.set_current()?;
        self.ev = Some((v4flash_hip::Event::new()?, v4flash_hip::Event::new()?));
        Ok(())
    }

    /// The last request's GPU event bracket (both completed after `run`).
    pub fn device_events(&self) -> Option<(&v4flash_hip::Event, &v4flash_hip::Event)> {
        self.ev.as_ref().map(|(a, b)| (a, b))
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn decode_max_b(&self) -> usize {
        self.decode_max_b
    }
    pub fn device(&self) -> Device {
        self.device
    }

    /// Quantise `x` (`b × N_EMBD` f32) to Q8_K rows on the device — exactly
    /// the hub's `q8k.launch` on `ffn_input_norm`. Test/bench helper.
    pub fn quantize_q8k(&mut self, x: &[f32], out: &mut [u8]) -> eyre::Result<()> {
        let b = x.len() / N_EMBD as usize;
        if b == 0 || b > self.rows || out.len() != b * XQ_BYTES_PER_TOKEN {
            return Err(eyre!("quantize_q8k: bad sizes"));
        }
        self.device.set_current()?;
        let mut xf = DeviceBuffer::<f32>::new(self.device.id, x.len())?;
        xf.copy_from_host(x)?;
        let mut xqv = self.xq.slice_view_mut(0, out.len());
        self.engine.q8k.launch(&self.engine.compute, &mut xqv, &xf, BLOCKS_Q8K_GATE_IN * b as u32)?;
        self.engine.compute.synchronize()?;
        xqv.copy_to_host(out)?;
        Ok(())
    }

    /// Weighted partial sums over the picks of `b` tokens for `layer`:
    /// `sel`/`ew` are `b × N_EXPERT_USED` (wire ids; `NO_PICK` = empty slot,
    /// every real id must be owned by `shard` at `layer`). Result in
    /// `self.ffn_moe[..b*N_EMBD]` (f32) after return.
    pub fn run(
        &mut self,
        shard: &mut ExpertShard,
        layer: u32,
        b: usize,
        xq: &[u8],
        sel: &[i32],
        ew: &[f32],
    ) -> eyre::Result<ExecTiming> {
        self.run_path(shard, layer, b, xq, sel, ew, false, &mut |_, _| Ok(()))
    }

    /// As [`Self::run`]; `force_batched` takes the by-expert chain regardless of `b`.
    #[allow(clippy::too_many_arguments)]
    /// Size the batched group buffers for `n_slots` group ids and return the
    /// bound to pass the builder. See `group_bound`.
    fn ensure_group_bound(&mut self, n_slots: u32) -> eyre::Result<u32> {
        if n_slots <= self.group_bound {
            return Ok(self.group_bound);
        }
        self.device.set_current()?;
        let id = self.device.id;
        self.group_count = DeviceBuffer::new(id, n_slots as usize)?;
        self.expert_members = DeviceBuffer::new(id, n_slots as usize * self.rows)?;
        // `launch_work_items` walks the whole group space, and its output is
        // `n_expert + rows*nu` entries (one chunk header per group, worst case).
        let wi_len = n_slots as usize + self.rows * N_EXPERT_USED;
        if self.work_items.len() < wi_len {
            self.work_items = DeviceBuffer::new(id, wi_len)?;
        }
        self.group_bound = n_slots;
        Ok(n_slots)
    }

    pub fn run_path(
        &mut self,
        shard: &mut ExpertShard,
        layer: u32,
        b: usize,
        xq: &[u8],
        sel: &[i32],
        ew: &[f32],
        force_batched: bool,
        overlap: &mut dyn FnMut(&mut ExpertShard, bool) -> eyre::Result<()>,
    ) -> eyre::Result<ExecTiming> {
        let nu = N_EXPERT_USED;
        // No f16 reply in host memory until this request's final event wait.
        self.out16_host_rows = 0;
        if b == 0 || b > self.rows {
            return Err(eyre!("executor: b={b} outside 1..={}", self.rows));
        }
        if xq.len() != b * XQ_BYTES_PER_TOKEN || sel.len() != b * nu || ew.len() != b * nu {
            return Err(eyre!("executor: payload sizes do not match b={b}"));
        }
        // The previous request errored with GPU work possibly still queued
        // (uploads from the pinned staging this one is about to overwrite,
        // kernels writing `partials`): let it drain first. Its partial rows
        // are already marked dirty.
        if self.in_flight {
            self.device.set_current()?;
            self.engine.compute.synchronize()?;
            self.in_flight = false;
        }
        // Pin mode's surprise evidence: what this pass must page or wait for,
        // judged BEFORE the first `ensure` / admission can change residency.
        // Array lookups only (b * 6 of them).
        let paged = shard.paged_bits(layer, sel);
        // Catch-all tier: make every requested expert resident first. A paged
        // shard's `remap[e]` is 0 ("the other device takes it") until it is,
        // which the kernel would silently honour and drop the expert.
        // Report the misses back to the hub for b==1 (decode). The hub uses them to
        // keep box 1's decode residency EXCLUSIVE of box 2's -- see
        // `ExpertShard::ensure_layer_reporting`. Prefill batches skip it: their sel
        // is up to 1024x6 and nothing consumes the mask.
        let mut miss_mask = 0u32;
        let path_decode = b <= self.decode_max_b && !force_batched;
        // The short batched chain (`knobs::fast_chain`): decode-sized batched
        // passes only; the decode-down diagnostic keeps the old chain.
        let fast = !path_decode && b <= self.fast_rows && knobs::fast_chain() && !b2_decode_down();
        // HITS-FIRST (docs/v41/MULTISTREAM_DECODE_PLAN.md 4.1). On the batched path,
        // split the picks into resident / missing WITHOUT reading anything, launch
        // the resident experts' pass, read the misses while those kernels run, then
        // launch the missed experts as a second pass into their own `partials`
        // slots and reduce once. Numerics are unchanged by construction: a (token,
        // slot) partial is written exactly once, by whichever pass owns it, by the
        // same kernel on the same inputs, and the reduce sums a token's `nu` slots
        // in slot order whichever pass wrote them. The decode path (per token,
        // `ffn_moe` written directly) stays single-pass. `V41_B2_HITS_FIRST=0`
        // restores ensure-then-launch.
        let hits_first = !path_decode && b2_hits_first() && !b2_decode_down() && shard.layer_is_paged(layer);
        self.missing_scratch.clear();
        if hits_first {
            shard.resident_mask(layer, sel, &mut self.resident_scratch);
            for (i, &e) in sel.iter().enumerate() {
                if e != NO_PICK && !self.resident_scratch[i] && !self.missing_scratch.contains(&e) {
                    self.missing_scratch.push(e);
                }
            }
        }
        let two_pass = hits_first && !self.missing_scratch.is_empty();
        if two_pass {
            // Pass A reads this layer's remap: a foreign eviction's pending
            // re-upload must land before anything reads it.
            shard.sync_remap(layer)?;
        } else if b == 1 {
            self.missed_scratch.clear();
            shard.ensure_layer_reporting(layer, sel, &mut self.missed_scratch)?;
            for (i, &sv) in sel.iter().take(proto::RESP_MISS_BITS).enumerate() {
                if sv != NO_PICK && self.missed_scratch.contains(&(sv as u32)) {
                    miss_mask |= 1 << i;
                }
            }
        } else {
            // A verify batch (B<=16) is still decode: six picks per token, not a
            // layer union. Only a real prefill chunk pins a whole layer.
            shard.ensure_layer_phased(layer, sel, b > 16)?;
        }
        // Pass-A sel: a missing pick is SENTINEL (the mode-0 kernels skip it);
        // single-pass: everything is resident by now or the request is malformed.
        for (i, &e) in sel.iter().enumerate() {
            if e == NO_PICK || (two_pass && !self.resident_scratch[i]) {
                self.sel_host[i] = SENTINEL_EXPERT;
                self.ew_host[i] = 0.0;
            } else if shard.owns(layer, e) {
                self.sel_host[i] = e;
                self.ew_host[i] = ew[i];
            } else {
                return Err(eyre!("executor: layer {layer} expert {e} (token {}) is not resident here", i / nu));
            }
        }
        self.device.set_current()?;
        self.in_flight = true;
        let t0 = Instant::now();
        // Uploads: stage through pinned host memory and queue them on the
        // compute stream. The caller's slices are unpinned, and a blocking
        // `hipMemcpy` from unpinned memory cost ~11 us EACH here (three per
        // request = 33 us of a ~380 us decode request). The memcpy into the
        // pinned staging is ~6 KB; the DMA then overlaps whatever the stream is
        // still finishing and orders ahead of this request's kernels.
        // The fast chain stages xq + sel + ew back to back and copies them ONCE
        // (the old chain's three copies were three commands, ~2.5 us apart).
        if fast {
            self.upload_fast(0, b, Some(xq))?;
        } else {
            self.xq_pin.as_mut_slice()[..xq.len()].copy_from_slice(xq);
            self.xq.slice_view_mut(0, xq.len()).copy_from_host_async(&self.xq_pin.as_slice()[..xq.len()], &self.engine.compute)?;
            self.upload_sel(0, b)?;
        }
        let t1 = Instant::now();
        if let Some((a, _)) = self.ev.as_ref() {
            a.record(&self.engine.compute)?;
        }
        // Size the batched group buffers BEFORE borrowing the engine. The
        // group-id space is the POOL SLOT space (box 2's remap encodes absolute
        // pool slots); passing N_EXPERT silently dropped every pick above slot
        // 383 -- see `group_bound`.
        let gbound = self.ensure_group_bound(shard.info.n_resident)?;
        let geo = PassGeo {
            b,
            gbound,
            cap: N_EXPERT_USED as u32,
            gbpe: shard.routed.gate_bytes_per_expert as u32,
            ubpe: shard.routed.up_bytes_per_expert as u32,
            dbpe: shard.routed.down_bytes_per_expert as u32,
            gdt: shard.routed.gate.dtype,
            ddt: shard.routed.down.dtype,
        };
        let mut timing = ExecTiming {
            path_decode,
            miss_mask,
            n_missing: self.missing_scratch.len() as u32,
            two_pass,
            fast_chain: fast,
            paged,
            ..Default::default()
        };
        if path_decode {
            // Decode kernels, one token at a time: q8k(x) is already done on the
            // hub (the wire carries Q8_K), so 3 launches per token.
            let (gate, up, down, remap) = shard.layer_views(layer)?;
            let e = &self.engine;
            let s = &e.compute;
            let (cap, gbpe, ubpe, dbpe, gdt, ddt) = (geo.cap, geo.gbpe, geo.ubpe, geo.dbpe, geo.gdt, geo.ddt);
            for t in 0..b {
                let xq_t = self.xq.slice_view(t * XQ_BYTES_PER_TOKEN, XQ_BYTES_PER_TOKEN);
                let sel_t = self.d_selected.slice_view(t * nu, nu);
                let ew_t = self.d_ew.slice_view(t * nu, nu);
                let mut mid_t = self.d_mid_cat.slice_view_mut(t * nu * N_FF_EXP as usize, nu * N_FF_EXP as usize);
                let mut midq_t = self.d_midq_cat.slice_view_mut(t * nu * MIDQ_BYTES_PER_SLOT, nu * MIDQ_BYTES_PER_SLOT);
                let mut out_t = self.ffn_moe.slice_view_mut(t * N_EMBD as usize, N_EMBD as usize);
                super::dispatch::moe_gate_up_batch_hetsplit(
                    e, gdt, s, &mut mid_t, &gate, &up, &xq_t, &ew_t, &sel_t, remap, 0, cap, gbpe, ubpe,
                    nu as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
                )?;
                e.q8k.launch(s, &mut midq_t, &mid_t, BLOCKS_Q8K_DOWN_IN * nu as u32)?;
                super::dispatch::moe_down_batched_hetsplit(
                    e, ddt, s, &mut out_t, &down, &midq_t, &sel_t, remap, 0, cap, dbpe,
                    MIDQ_BYTES_PER_SLOT as u32, nu as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
                )?;
            }
        } else {
            // Production prefill chain (forward_prefill.rs stage 11, MXFP4 arm):
            // hetsplit group builder -> work items (host readback) -> kwide
            // gate/up -> q8k(mid) -> by-expert kwide2 down -> hetsplit reduce.
            let io = self.pass_io(b, fast);
            let n_prow = b * nu;
            // Partials: the fast chain needs rows [0, n_prow) zero on entry
            // (a memset only after an old-chain pass or an error) and leaves
            // them zero once its reduce is queued; until then they count as
            // dirty, so an error anywhere below leaves the next request a
            // memset. The old chain memsets on entry and leaves them dirty.
            let clean_after = if fast {
                self.partials_make_clean(n_prow)?;
                Some(self.p_dirty)
            } else {
                None
            };
            self.partials_mark_dirty(n_prow);
            let diag_done;
            {
                let (gate, up, down, remap) = shard.layer_views(layer)?;
                let (n_wi, done) = self.batched_pass(&gate, &up, &down, remap, &geo, true, fast, &io)?;
                timing.n_work_items = n_wi;
                diag_done = done;
            }
            if diag_done {
                // `V41_B2_DECODE_DOWN` wrote `ffn_moe` directly; nothing to reduce.
            } else if two_pass {
                // The misses: read them NOW, while pass A's kernels run. `ensure`
                // gets the FULL pick list so its victim search never evicts an
                // expert pass A is reading from (a wanted id is never a victim)
                // and so the hits' recency is touched as usual.
                //
                // PARK first (`knobs::park`): offer the hook the chance to serve
                // requests queued behind this one while these misses read on the
                // prefetch readers. Pass A's kernels keep running on OUR stream;
                // the hook uses its own executor, and every pick of this request
                // is pinned so nothing it serves can evict them. The `ensure`
                // below then admits whatever has landed and waits for the rest.
                if knobs::park() {
                    shard.park_words.clear();
                    for &e in &self.missing_scratch {
                        shard.park_words.push((layer << 16) | e as u32);
                    }
                    shard.park_prefill = if shard.mode_evict_on() { shard.req_prefill } else { b > 16 };
                    shard.parked_pins.clear();
                    for &e in sel.iter().filter(|&&e| e != NO_PICK && (0..N_EXPERT as i32).contains(&e)) {
                        shard.parked_pins.push((layer, e as u32));
                    }
                    let r = overlap(shard, true);
                    shard.parked_pins.clear();
                    shard.park_words.clear();
                    r?;
                }
                shard.ensure_layer_phased(layer, sel, b > 16)?;
                for (i, &e) in sel.iter().enumerate() {
                    let live = e != NO_PICK && !self.resident_scratch[i];
                    self.sel_host[i] = if live { e } else { SENTINEL_EXPERT };
                    self.ew_host[i] = if live { ew[i] } else { 0.0 };
                }
                if fast {
                    self.upload_fast(1, b, None)?;
                } else {
                    self.upload_sel(1, b)?;
                }
                let (gate, up, down, remap) = shard.layer_views(layer)?;
                let (n_wi, _) = self.batched_pass(&gate, &up, &down, remap, &geo, false, fast, &io)?;
                timing.n_work_items += n_wi;
                // Reduce over the FULL pick list: every real slot, whichever pass
                // wrote its partial.
                for (i, &e) in sel.iter().enumerate() {
                    self.sel_host[i] = if e == NO_PICK { SENTINEL_EXPERT } else { e };
                    self.ew_host[i] = if e == NO_PICK { 0.0 } else { ew[i] };
                }
                if fast {
                    self.upload_fast(2, b, None)?;
                } else {
                    self.upload_sel(2, b)?;
                }
                self.reduce_io(remap, b, fast, &io)?;
            } else {
                let (_, _, _, remap) = shard.layer_views(layer)?;
                self.reduce_io(remap, b, fast, &io)?;
            }
            if let Some(st) = clean_after {
                // The fast reduce is queued: rows [0, n_prow) end this request
                // zero, as they started it.
                self.p_dirty = st;
            }
        }
        if let Some((_, ev_b)) = self.ev.as_ref() {
            ev_b.record(&self.engine.compute)?;
        }
        // A queued request's own misses can start reading now, under this
        // request's GPU tail + D2H + reply + the hub's turnaround (see the
        // compute loop). Cheap (decode + a few hash lookups); never waits.
        overlap(shard, false)?;
        self.ev_done.record(&self.engine.compute)?;
        // Poll instead of block: every ~20 us give the overlap hook another
        // chance at the reader queue while the GPU drains. Bounded by the GPU
        // itself; the hook is a no-op once it holds a frame.
        while !self.ev_done.query()? {
            overlap(shard, false)?;
            std::thread::sleep(std::time::Duration::from_micros(20));
        }
        self.in_flight = false;
        if fast && self.reply_f16 {
            // The fast reduce wrote the f16 rows into `out16_pin`; the event
            // above (system-scope release) made them host-visible.
            self.out16_host_rows = b;
        }
        timing.h2d = t1 - t0;
        timing.gpu = t1.elapsed();
        Ok(timing)
    }

    /// The fast chain's upload for `region` (0: pass A / single pass with
    /// `xq`; 1: pass B; 2: the reduce, both sel/ew only): stage
    /// `sel_host`/`ew_host` (+ `xq`) into `fin_pin` in the `fast_in_layout(b)`
    /// layout and queue ONE async copy into `fin_dev`.
    fn upload_fast(&mut self, region: usize, b: usize, xq: Option<&[u8]>) -> eyre::Result<()> {
        let n = b * N_EXPERT_USED;
        let (ew_off, xq_off, total) = fast_in_layout(b);
        let stride = fast_in_layout(self.fast_rows).2;
        let base = region * stride;
        let bytes = if xq.is_some() { total } else { 2 * n * 4 };
        let pin = &mut self.fin_pin.as_mut_slice()[base..base + bytes];
        for (d, v) in pin[..ew_off].chunks_exact_mut(4).zip(&self.sel_host[..n]) {
            d.copy_from_slice(&v.to_le_bytes());
        }
        for (d, v) in pin[ew_off..2 * n * 4].chunks_exact_mut(4).zip(&self.ew_host[..n]) {
            d.copy_from_slice(&v.to_le_bytes());
        }
        if let Some(xq) = xq {
            pin[xq_off..total].copy_from_slice(xq);
        }
        self.fin_dev
            .slice_view_mut(0, bytes)
            .copy_from_host_async(&self.fin_pin.as_slice()[base..base + bytes], &self.engine.compute)
    }

    /// This request's input views: the fast chain's single upload, or the old
    /// chain's three buffers (exactly the views `batched_pass` always used).
    fn pass_io(&self, b: usize, fast: bool) -> PassIo {
        let n = b * N_EXPERT_USED;
        if fast {
            let (ew_off, xq_off, _) = fast_in_layout(b);
            // SAFETY: `fin_dev` holds `fast_in_layout(fast_rows)` bytes and
            // b <= fast_rows; both offsets are 4-byte aligned (xq's is 256).
            unsafe {
                PassIo {
                    sel: self.fin_dev.view_as::<i32>(0, n),
                    ew: self.fin_dev.view_as::<f32>(ew_off, n),
                    xq: self.fin_dev.view_as::<u8>(xq_off, b * XQ_BYTES_PER_TOKEN),
                }
            }
        } else {
            PassIo {
                xq: self.xq.slice_view(0, b * XQ_BYTES_PER_TOKEN),
                sel: self.d_selected.slice_view(0, n),
                ew: self.d_ew.slice_view(0, n),
            }
        }
    }

    /// The reduce of a batched request: the old hetsplit reduce, or (fast) the
    /// zeroing reduce that also writes the f16 reply into `out16_pin`.
    fn reduce_io(&mut self, remap: &DeviceBuffer<i32>, b: usize, fast: bool, io: &PassIo) -> eyre::Result<()> {
        if !fast {
            return self.reduce_partials(remap, b);
        }
        let nu = N_EXPERT_USED;
        let s = &self.engine.compute;
        let mut out_v = self.ffn_moe.slice_view_mut(0, b * N_EMBD as usize);
        let mut part_v = self.partials.slice_view_mut(0, b * nu * N_EMBD as usize);
        let out16 = if self.reply_f16 { Some(&mut self.out16_pin) } else { None };
        self.fast_k.launch_reduce_zero(
            s, &mut out_v, out16, &mut part_v, &io.sel, remap, 0, N_EXPERT_USED as u32, nu as u32, N_EMBD,
            b as u32, N_EXPERT,
        )
    }

    /// Stage `sel_host`/`ew_host` (first `b * nu` entries) through pinned region
    /// `region` (0: pass A / single pass, 1: pass B, 2: the reduce) and queue the
    /// upload on the compute stream. Regions are distinct because an earlier
    /// region's DMA may still be queued when the next is written; the stream sync
    /// at the end of `run_path` retires them all before the next request.
    fn upload_sel(&mut self, region: usize, b: usize) -> eyre::Result<()> {
        let nu = N_EXPERT_USED;
        let n = b * nu;
        let off = region * self.rows * nu;
        self.sel_pin.as_mut_slice()[off..off + n].copy_from_slice(&self.sel_host[..n]);
        self.ew_pin.as_mut_slice()[off..off + n].copy_from_slice(&self.ew_host[..n]);
        self.d_selected.slice_view_mut(0, n).copy_from_host_async(&self.sel_pin.as_slice()[off..off + n], &self.engine.compute)?;
        self.d_ew.slice_view_mut(0, n).copy_from_host_async(&self.ew_pin.as_slice()[off..off + n], &self.engine.compute)?;
        Ok(())
    }

    /// One by-expert pass over whatever `d_selected`/`d_ew` hold right now: group
    /// build -> work items -> gate/up -> q8k(mid) -> down into `partials`.
    ///
    /// The work-item count was read back to size the grids, a GPU-idle host round
    /// trip per pass. Under `V41_MOE_WI_DEVCOUNT` (MXFP4 kwide pair + kwide2) the
    /// grids take `dispatch::moe_wi_upper_bound` and the kernels exit past the
    /// device-side count; the returned count is then that bound. With the
    /// hits-first two passes, pass A's builder may now RUN after the host has
    /// already paged the misses (`ensure_layer_phased`) or served a parked
    /// request that rewrote this layer's remap: still correct, since pass A's
    /// misses are SENTINEL and its hits are wanted/pinned, so their slots cannot
    /// move. `first` zeroes `partials` (once per request; a second pass adds
    /// its own slots beside the first's). Returns (work items, diagnostic-done):
    /// under `V41_B2_DECODE_DOWN` the decode down kernel writes `ffn_moe` directly
    /// and there is nothing to reduce.
    fn batched_pass(
        &mut self,
        gate: &DeviceBuffer<u8>,
        up: &DeviceBuffer<u8>,
        down: &DeviceBuffer<u8>,
        remap: &DeviceBuffer<i32>,
        g: &PassGeo,
        first: bool,
        fast: bool,
        io: &PassIo,
    ) -> eyre::Result<(u32, bool)> {
        let nu = N_EXPERT_USED;
        let b = g.b;
        let bu = b as u32;
        let e = &self.engine;
        let s = &e.compute;
        let max_per_expert = self.rows as u32;
        let (sel_v, ew_v, xq_v) = (&io.sel, &io.ew, &io.xq);
        let max_items = self.work_items.len() as u32;
        if fast {
            // ONE work-group builds the groups THIS pass touches, the work items
            // and their count (b2_fast_chain.hip): no zeroing, no second kernel.
            self.fast_k.launch_builder(
                s, &mut self.group_count, &mut self.expert_members, &mut self.work_items, &mut self.n_work_items,
                sel_v, remap, 0, g.cap, bu, nu as u32, g.gbound, max_per_expert, CHUNK_SIZE, max_items,
            )?;
        } else {
            self.group_count.fill_zero_async(s)?;
            e.moe_group_builder.launch_hetsplit(
                s, &mut self.group_count, &mut self.expert_members, sel_v, remap, 0, g.cap, bu,
                nu as u32, g.gbound, max_per_expert,
            )?;
            self.n_work_items.fill_zero_async(s)?;
            e.moe_group_builder.launch_work_items(
                s, &mut self.work_items, &mut self.n_work_items, &self.group_count, g.gbound, CHUNK_SIZE, max_items,
            )?;
        }
        let devcount = super::forward_prefill::moe_wi_devcount()
            && !b2_decode_down()
            && super::dispatch::moe_wi_devcount_supported(g.gdt, g.ddt);
        let n_wi = if devcount {
            super::dispatch::moe_wi_upper_bound(b * nu, g.gbound, CHUNK_SIZE, max_items as usize)
        } else {
            s.synchronize()?;
            let mut n_wi = [0i32; 1];
            self.n_work_items.copy_to_host(&mut n_wi)?;
            n_wi[0] as u32
        };
        let n_wi_dev: Option<&DeviceBuffer<i32>> = if devcount { Some(&self.n_work_items) } else { None };
        let mut mid_v = self.d_mid_cat.slice_view_mut(0, b * nu * N_FF_EXP as usize);
        // Under the decode-down diagnostic, zero mid first (as the decode
        // gate/up does): the chunked gate/up writes only MEMBER slots, and
        // the decode down sums all nu slots per token, so non-member slots
        // must be 0 for the isolation to be valid rather than summing stale
        // data (which gave KLD 1.6).
        if b2_decode_down() {
            mid_v.fill_zero_async(s)?;
        }
        // `_rows`: MXFP4 at >= 256 rows takes the int8-WMMA arm (`V41_MOE_WMMA_GATEUP`,
        // 2026-09-26 sweep; not bit-exact).
        let handled = super::dispatch::moe_gate_up_chunked_rows(
            e, g.gdt, s, &mut mid_v, gate, up, xq_v, ew_v, &self.group_count, &self.expert_members,
            &self.work_items, n_wi, g.gbpe, g.ubpe, nu as u32, max_per_expert, CHUNK_SIZE, SWIGLU_CLAMP_EXP,
            N_FF_EXP, BLOCKS_Q8K_GATE_IN, n_wi_dev, bu,
        )?;
        if !handled {
            return Err(eyre!("executor: no prefill gate/up kernel for {:?}", g.gdt));
        }
        let mut midq_v = self.d_midq_cat.slice_view_mut(0, b * nu * MIDQ_BYTES_PER_SLOT);
        e.q8k.launch(s, &mut midq_v, &mid_v, BLOCKS_Q8K_DOWN_IN * nu as u32 * bu)?;
        // `V41_B2_DECODE_DOWN=1`: run the DECODE down kernel per token over the
        // batched midq (same derivation the decode branch uses: sel/midq/out
        // sliced by token, NO clobbering of d_selected). Isolates whether the
        // batched-vs-decode 0.276-nat divergence is in the by-expert DOWN
        // kernel (KLD -> ~0 here) or upstream/batch-state (unchanged).
        if b2_decode_down() {
            for t in 0..b {
                let sel_t = self.d_selected.slice_view(t * nu, nu);
                let midq_t = self
                    .d_midq_cat
                    .slice_view(t * nu * MIDQ_BYTES_PER_SLOT, nu * MIDQ_BYTES_PER_SLOT);
                let mut out_t = self.ffn_moe.slice_view_mut(t * N_EMBD as usize, N_EMBD as usize);
                super::dispatch::moe_down_batched_hetsplit(
                    e, g.ddt, s, &mut out_t, down, &midq_t, &sel_t, remap, 0, g.cap, g.dbpe,
                    MIDQ_BYTES_PER_SLOT as u32, nu as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
                )?;
            }
            return Ok((n_wi, true));
        }
        let mut part_v = self.partials.slice_view_mut(0, b * nu * N_EMBD as usize);
        // MUST be zeroed per request. `launch_by_expert_kwide2` writes only
        // the (token, slot) partials whose expert has members THIS pass;
        // every other slot keeps the PREVIOUS request's value, and the
        // reduce below sums `nu` slots per token. `group_count` and
        // `n_work_items` are both zeroed for the same reason -- `partials`
        // was missed once. Only the FIRST pass of a request zeroes: the second
        // pass writes the slots the first left at zero.
        //
        // Only the BATCHED branch accumulates this way, and box 1 sets
        // REQ_FLAG_BATCHED exactly when b > 1, which is why the corruption
        // appeared only at b >= 2: a b=1 request takes the decode branch,
        // which writes `ffn_moe` directly per token.
        //
        // The FAST chain skips it: `run_path` guarantees rows [0, b*nu) are
        // zero on entry (`partials_make_clean`) and its reduce re-zeroes every
        // slot it consumed and every slot holding a real pick.
        if first && !fast {
            part_v.fill_zero_async(s)?;
        }
        match g.ddt {
            // By rows: int8-WMMA arm at >= 128 (`V41_MOE_WMMA_DOWN`, not bit-exact),
            // small-b twin at <= 8 (`V41_MOE_DOWN_DN2`, bit-exact), else kwide2.
            GgufType::MXFP4 => super::dispatch::moe_down_mxfp4(
                e, s, &mut part_v, down, &midq_v, &self.group_count, &self.expert_members, &self.work_items,
                n_wi, g.dbpe, MIDQ_BYTES_PER_SLOT as u32, nu as u32, max_per_expert, CHUNK_SIZE, N_EMBD,
                BLOCKS_Q8K_DOWN_IN, n_wi_dev, bu,
            )?,
            GgufType::IQ3_XXS => e.iq3.launch_by_expert_kwide2(
                s, &mut part_v, down, &midq_v, &self.group_count, &self.expert_members, &self.work_items,
                n_wi, g.dbpe, MIDQ_BYTES_PER_SLOT as u32, nu as u32, max_per_expert, CHUNK_SIZE, N_EMBD,
                BLOCKS_Q8K_DOWN_IN,
            )?,
            other => return Err(eyre!("executor: no prefill down kernel for {other:?}")),
        }
        Ok((n_wi, false))
    }

    /// Sum each token's `nu` partial slots into `ffn_moe`, in slot order, over the
    /// pick list currently in `d_selected` (a sentinel/no-pick slot contributes
    /// nothing: its partial is zero and mode 0 skips it).
    fn reduce_partials(&mut self, remap: &DeviceBuffer<i32>, b: usize) -> eyre::Result<()> {
        let nu = N_EXPERT_USED;
        let e = &self.engine;
        let s = &e.compute;
        let sel_v = self.d_selected.slice_view(0, b * nu);
        let part_v = self.partials.slice_view(0, b * nu * N_EMBD as usize);
        let mut out_v = self.ffn_moe.slice_view_mut(0, b * N_EMBD as usize);
        e.q2k.launch_reduce_partials_hetsplit(
            s, &mut out_v, &part_v, &sel_v, remap, 0, N_EXPERT_USED as u32, nu as u32, N_EMBD, b as u32,
        )
    }

    /// Copy the last result's f32 rows to host.
    pub fn read_f32(&self, b: usize, dst: &mut [f32]) -> eyre::Result<()> {
        self.device.set_current()?;
        // MUST synchronise the compute stream first. `copy_to_host` is a
        // blocking hipMemcpy, which orders against the NULL stream only — it
        // does NOT wait for work queued on `engine.compute`. Without this the
        // copy returns whatever `ffn_moe` happens to hold, i.e. the PREVIOUS
        // request's result.
        //
        // The f16 twin below always synchronised (it has to, for the cast), so
        // DECODE — which asks for f16 — was correct while the speculative
        // verify — which asks for f32, see `resp_f32` — silently got stale
        // partials. MEASURED before the fix, V41_T2_CATCHALL=2, B=2, catch-all
        // on: five consecutive layers returned byte-identical partials
        // (hash 8e3e5912bffb0af5, l2 30.7559, 10240/10240 nonzero) although box
        // 1 sent a DIFFERENT xq and sel for each, and 273 of 680 combines added
        // another layer's partial. The race needs the compute to outlast the
        // copy, which is why it appeared only at >= 2 rows per lane and got far
        // worse under the catch-all (box 2 computes ~22 experts/layer instead
        // of ~3).
        self.engine.compute.synchronize()?;
        self.ffn_moe.slice_view(0, b * N_EMBD as usize).copy_to_host(dst)
    }

    /// Rows `[off_rows, off_rows + b)` of the last result (a coalesced pass
    /// answers two requests from one output buffer).
    pub fn read_f32_at(&self, off_rows: usize, b: usize, dst: &mut [f32]) -> eyre::Result<()> {
        self.device.set_current()?;
        self.engine.compute.synchronize()?;
        self.ffn_moe.slice_view(off_rows * N_EMBD as usize, b * N_EMBD as usize).copy_to_host(dst)
    }

    pub fn read_f16_at(&mut self, off_rows: usize, b: usize, dst: &mut [u16]) -> eyre::Result<()> {
        if self.read_f16_pinned(off_rows, b, dst)? {
            return Ok(());
        }
        self.device.set_current()?;
        let n = b * N_EMBD as usize;
        let src = self.ffn_moe.slice_view(off_rows * N_EMBD as usize, n);
        let mut o = self.out16.slice_view_mut(0, n);
        self.engine.q8k.launch_cast_f16(&self.engine.compute, &mut o, &src, n as u32)?;
        self.engine.compute.synchronize()?;
        o.copy_to_host(dst)
    }

    /// Cast the last result to f16 on the device (`f32_to_f16_cast`, RNE) and
    /// copy to host.
    pub fn read_f16(&mut self, b: usize, dst: &mut [u16]) -> eyre::Result<()> {
        if self.read_f16_pinned(0, b, dst)? {
            return Ok(());
        }
        self.device.set_current()?;
        let n = b * N_EMBD as usize;
        let src = self.ffn_moe.slice_view(0, n);
        let mut o = self.out16.slice_view_mut(0, n);
        self.engine.q8k.launch_cast_f16(&self.engine.compute, &mut o, &src, n as u32)?;
        self.engine.compute.synchronize()?;
        o.copy_to_host(dst)
    }

    /// The fast chain's f16 reply: rows `[off_rows, off_rows + b)` of the last
    /// request, already in host memory (`out16_pin`, written by the reduce and
    /// published by `run_path`'s final event wait). `Ok(false)` = not there
    /// (old chain, decode path, f32 reply requested): take the device path.
    fn read_f16_pinned(&self, off_rows: usize, b: usize, dst: &mut [u16]) -> eyre::Result<bool> {
        if off_rows + b > self.out16_host_rows {
            return Ok(false);
        }
        let row = N_EMBD as usize;
        if dst.len() != b * row {
            return Err(eyre!("read_f16: dst has {} elements, want {}", dst.len(), b * row));
        }
        dst.copy_from_slice(&self.out16_pin.as_slice()[off_rows * row..(off_rows + b) * row]);
        Ok(true)
    }

    /// Host-side staging for a request's activations (the daemon reuses it).
    pub fn xq_host_mut(&mut self, bytes: usize) -> &mut [u8] {
        self.xq_host.resize(bytes, 0);
        &mut self.xq_host[..bytes]
    }
}


// ---------------------------------------------------------------------------
// Perfetto tracing for the daemon
// ---------------------------------------------------------------------------

/// Track uuids for box 2. Distinct from the hub's (`DeviceTimingExporter` uses
/// 0x44504755_* / 0x49504755_*) so the two boxes' trace files merge into one
/// timeline in perfetto's trace processor without a uuid collision.
pub const BOX2_IGPU_COMPUTE_UUID: u64 = 0x424f5832_0000_0001;
pub const BOX2_IGPU_XFER_UUID: u64 = 0x424f5832_0000_0002;
pub const BOX2_REQUEST_UUID: u64 = 0x424f5832_0000_0010;
pub const BOX2_SSD_UUID: u64 = 0x424f5832_0000_0020;
/// Box 2's knobs: every knob at open, then each live change.
pub const BOX2_KNOBS_UUID: u64 = 0x424f5832_0000_0030;

/// The daemon's perfetto exporter: box 2's iGPU compute/xfer device tracks plus
/// host-time application tracks for the per-request phases and the SSD expert
/// reads.
///
/// Timestamps are box 2's CLOCK_REALTIME; `ClockSync::perfetto_shift_ns` on the
/// hub gives the exact ns to shift them onto box 1's timeline (NTP between the
/// boxes is only good to 100 µs - 1 ms, useless against a 32 µs RTT).
pub struct ExpertdTracer {
    exporter: super::perfetto::TrackExporter,
    igpu_compute: std::sync::Mutex<super::perfetto::Track>,
    machine: String,
    /// The last knob change on the `knobs` track (`TrackExporter::emit_knobs`).
    knobs_seen: std::sync::Mutex<u64>,
}

// SAFETY: the only non-Send member is the HIP `Event` inside the device
// track's `Anchor` (a raw `hipEvent_t`). HIP handles are process-wide and the
// runtime is thread-safe; the track is additionally behind a Mutex, and the
// file writer is a Mutex<File>. Same reasoning as `unsafe impl Send for Graph`
// in v4flash-hip. Needed because the SSD read spans are emitted from the
// scoped reader threads during `ExpertShard::load_traced`.
unsafe impl Send for ExpertdTracer {}
unsafe impl Sync for ExpertdTracer {}

impl ExpertdTracer {
    pub fn open(path: impl AsRef<std::path::Path>, igpu: Device, compute: &v4flash_hip::Stream, machine: &str) -> eyre::Result<Self> {
        let exporter = super::perfetto::TrackExporter::open(path, 0x424f5832)?;
        let igpu_compute = exporter.device_track(
            BOX2_IGPU_COMPUTE_UUID,
            &format!("{machine} igpu.compute (device)"),
            compute,
            igpu,
        )?;
        exporter.declare(BOX2_IGPU_XFER_UUID, &format!("{machine} igpu.xfer (device)"))?;
        exporter.declare(BOX2_REQUEST_UUID, &format!("{machine} expertd.request (host)"))?;
        exporter.declare(BOX2_SSD_UUID, &format!("{machine} expertd.ssd (host)"))?;
        exporter.declare(BOX2_KNOBS_UUID, &format!("{machine} knobs"))?;
        let mut seen = 0;
        exporter.emit_knobs(BOX2_KNOBS_UUID, true, &mut seen)?;
        Ok(Self { exporter, igpu_compute: std::sync::Mutex::new(igpu_compute), machine: machine.into(), knobs_seen: std::sync::Mutex::new(seen) })
    }

    pub fn machine(&self) -> &str {
        &self.machine
    }

    /// Host-time span on a track (ns are CLOCK_REALTIME, `super::perfetto::host_now_ns`).
    /// Also drains knob changes onto the `knobs` track (a compare when none).
    pub fn span(&self, uuid: u64, name: &str, start_ns: u64, end_ns: u64) {
        let _ = self.exporter.emit_span(uuid, name, start_ns, end_ns);
        if let Ok(mut seen) = self.knobs_seen.try_lock() {
            let _ = self.exporter.emit_knobs(BOX2_KNOBS_UUID, false, &mut seen);
        }
    }

    /// One SSD expert read, labelled with its (layer, expert) so a stall
    /// attributes to a specific pick rather than to "loading".
    pub fn ssd_read(&self, layer: u32, expert: u32, start_ns: u64, end_ns: u64) {
        self.span(BOX2_SSD_UUID, &format!("ssd L{layer} E{expert}"), start_ns, end_ns);
    }

    /// Device-time slice for one request's GPU work.
    pub fn gpu_slice(&self, name: &str, start: &v4flash_hip::Event, end: &v4flash_hip::Event) {
        if let Ok(t) = self.igpu_compute.lock() {
            let _ = self.exporter.emit_device_slice(&t, name, start, end);
        }
    }

    /// Re-anchor the device track (bounds GPU/host clock drift in a long trace).
    pub fn re_anchor(&self, igpu: Device, compute: &v4flash_hip::Stream) -> eyre::Result<()> {
        if let Ok(mut t) = self.igpu_compute.lock() {
            self.exporter.re_anchor(&mut t, compute, igpu)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Daemon: serve requests over one connection at a time
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ServeOptions {
    pub socket: SocketOptions,
    /// Print one line per request.
    pub verbose: bool,
    /// Print a rolling summary every N requests (0 = only at disconnect).
    pub log_every: usize,
    /// While idle between requests, issue a trivial GPU launch every this many
    /// µs (0 = off) so the iGPU stays clocked up (see `MoeExecutor::keep_warm`).
    pub keep_warm_us: u64,
    /// Re-anchor the device track every N requests (bounds GPU/host drift).
    pub re_anchor_every: usize,
}

impl Default for ServeOptions {
    fn default() -> Self {
        Self { socket: SocketOptions::default(), verbose: false, log_every: 0, keep_warm_us: 250, re_anchor_every: 512 }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct RequestRecord {
    pub seq: u32,
    pub layer: u32,
    pub b: u32,
    pub bytes_in: usize,
    pub bytes_out: usize,
    /// Socket-read duration of the frame (first byte → last byte).
    pub read_us: u32,
    /// Frame complete → compute started (queueing behind the previous request).
    pub queue_us: u32,
    pub h2d_us: u32,
    pub gpu_us: u32,
    pub d2h_us: u32,
    /// Compute done → response fully written to the socket.
    pub write_us: u32,
    pub path_decode: bool,
    /// When the response was handed to the writer thread (write_us base).
    pub t_ready: Instant,
    /// The client's `t1` and our `t2` (CLOCK_MONOTONIC_RAW, different boxes) —
    /// enough for the daemon's own trace to be aligned offline.
    pub t1: u64,
    pub t2: u64,
}

fn pct(sorted: &[u32], p: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let i = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[i.min(sorted.len() - 1)]
}

/// Summarise per-B-class latency percentiles (also used by the client bench).
pub fn summarize(records: &[RequestRecord], label: &str) {
    let mut classes: Vec<u32> = records.iter().map(|r| r.b).collect();
    classes.sort_unstable();
    classes.dedup();
    for b in classes {
        let rs: Vec<&RequestRecord> = records.iter().filter(|r| r.b == b).collect();
        let col = |f: &dyn Fn(&RequestRecord) -> u32| -> (u32, u32, u32) {
            let mut v: Vec<u32> = rs.iter().map(|r| f(r)).collect();
            v.sort_unstable();
            (pct(&v, 0.5), pct(&v, 0.9), pct(&v, 0.99))
        };
        let (rd, qu, h2d, gpu, d2h, wr) = (
            col(&|r| r.read_us),
            col(&|r| r.queue_us),
            col(&|r| r.h2d_us),
            col(&|r| r.gpu_us),
            col(&|r| r.d2h_us),
            col(&|r| r.write_us),
        );
        let bi = rs.iter().map(|r| r.bytes_in).sum::<usize>() as f64 / rs.len() as f64;
        let bo = rs.iter().map(|r| r.bytes_out).sum::<usize>() as f64 / rs.len() as f64;
        eprintln!(
            "{label} B={b:<5} n={:<5} in {:.1} KB out {:.1} KB | us p50/p90/p99: read {}/{}/{} queue {}/{}/{} h2d {}/{}/{} gpu {}/{}/{} d2h {}/{}/{} write {}/{}/{}",
            rs.len(), bi / 1e3, bo / 1e3,
            rd.0, rd.1, rd.2, qu.0, qu.1, qu.2, h2d.0, h2d.1, h2d.2, gpu.0, gpu.1, gpu.2, d2h.0, d2h.1, d2h.2, wr.0, wr.1, wr.2
        );
    }
}

enum Inbound {
    Frame { hdr: proto::Header, buf: AlignedBuf, t_first: Instant, t_done: Instant, t2: u64 },
    Closed(Option<String>),
}

/// Accept connections forever (one at a time) and serve them.
pub fn serve(
    listener: TcpListener,
    shard: &mut ExpertShard,
    exec: &mut MoeExecutor,
    opts: &ServeOptions,
    tracer: Option<&ExpertdTracer>,
) -> eyre::Result<()> {
    eprintln!("expertd: listening on {}", listener.local_addr()?);
    loop {
        let (stream, peer) = listener.accept()?;
        eprintln!("expertd: connection from {peer}");
        match serve_connection(stream, &mut *shard, exec, opts, tracer) {
            Ok((records, n_total)) => {
                eprintln!("expertd: {peer} closed after {n_total} requests");
                summarize(&records, "expertd");
            }
            Err(e) => eprintln!("expertd: {peer} error: {e:#}"),
        }
    }
}

/// Serve one connection: reader thread → compute (this thread) → writer thread.
pub fn serve_connection(
    stream: TcpStream,
    shard: &mut ExpertShard,
    exec: &mut MoeExecutor,
    opts: &ServeOptions,
    tracer: Option<&ExpertdTracer>,
) -> eyre::Result<(Vec<RequestRecord>, u64)> {
    apply_socket_options(&stream, &opts.socket)?;
    B2_SPIN_CUR.store(opts.socket.busy_poll_us, std::sync::atomic::Ordering::Relaxed);
    // Pins belong to the hub that asked for them: a new connection starts with
    // none (the hub resets its mirror on connect too).
    shard.pin_reset();
    // + the word lists (hints 2 x 64, prefetch 128, release 128) at any `b`.
    let max_payload = proto::REQ_FIXED + exec.rows() * (XQ_BYTES_PER_TOKEN + 8 * N_EXPERT_USED) + 4096;
    // HELLO first.
    {
        let mut hello = AlignedBuf::with_capacity(4096);
        // Re-sample the (mono, realtime) pair now rather than reusing the one
        // taken at load: fresher correlation, and REALTIME may have been slewed
        // by NTP during the 30 s expert load.
        let mut info = shard.info().clone();
        info.clock = ClockPair::sample();
        proto::encode_hello(&mut hello, &info);
        (&stream).write_all(hello.as_bytes())?;
    }
    let mut rd = stream.try_clone()?;
    let mut wr = stream.try_clone()?;
    let (tx_in, rx_in) = mpsc::sync_channel::<Inbound>(8);
    let (tx_req_recycle, rx_req_recycle) = mpsc::channel::<AlignedBuf>();
    let (tx_out, rx_out) = mpsc::sync_channel::<(AlignedBuf, Instant, u32)>(8);
    let (tx_resp_recycle, rx_resp_recycle) = mpsc::channel::<AlignedBuf>();
    let (tx_written, rx_written) = mpsc::channel::<(u32, Instant)>();
    let sock_opts = opts.socket.clone();
    let mut records: Vec<RequestRecord> = Vec::new();
    let mut n_total: u64 = 0; // every request served, including those dropped from `records`
    let result: eyre::Result<()> = std::thread::scope(|sc| {
        // Reader.
        sc.spawn(move || {
            loop {
                let mut buf = rx_req_recycle.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(max_payload + proto::HDR_LEN));
                // Time the frame: first byte (header) arrival → payload complete.
                buf.resize(proto::HDR_LEN);
                if let Err(e) = rd.read_exact(buf.as_bytes_mut()) {
                    let _ = tx_in.send(Inbound::Closed(if e.kind() == std::io::ErrorKind::UnexpectedEof { None } else { Some(e.to_string()) }));
                    return;
                }
                let t_first = Instant::now();
                let hdr = match proto::parse_header(buf.as_bytes()) {
                    Ok(h) => h,
                    Err(e) => {
                        let _ = tx_in.send(Inbound::Closed(Some(format!("bad header: {e}"))));
                        return;
                    }
                };
                if hdr.len as usize > max_payload {
                    let _ = tx_in.send(Inbound::Closed(Some(format!("payload {} > max {max_payload}", hdr.len))));
                    return;
                }
                buf.resize(proto::HDR_LEN + hdr.len as usize);
                if let Err(e) = rd.read_exact(&mut buf.as_bytes_mut()[proto::HDR_LEN..]) {
                    let _ = tx_in.send(Inbound::Closed(Some(e.to_string())));
                    return;
                }
                // t2: daemon receive stamp, taken the instant the frame is whole.
                let t2 = monotonic_raw_ns();
                quickack(&rd, &sock_opts);
                let t_done = Instant::now();
                if tx_in.send(Inbound::Frame { hdr, buf, t_first, t_done, t2 }).is_err() {
                    return;
                }
            }
        });
        // Writer. Stamps t3 in place immediately before write(), which is the
        // only accurate place for it once compute and I/O are on different
        // threads (a t3 taken in the compute thread would include the handoff).
        sc.spawn(move || {
            for (mut buf, _t_ready, seq) in rx_out {
                let mut ev_t3 = f64::NAN;
                if buf.len() >= proto::RESP_T3_OFF + 8
                    && matches!(proto::parse_header(buf.as_bytes()), Ok(h) if h.kind == proto::KIND_RESPONSE)
                {
                    let t3 = monotonic_raw_ns();
                    ev_t3 = t3 as f64;
                    proto::patch_u64(&mut buf, proto::RESP_T3_OFF, t3);
                }
                if let Err(e) = wr.write_all(buf.as_bytes()) {
                    eprintln!("expertd: write error: {e}");
                    return;
                }
                super::evtrace::emit(&super::evtrace_kinds::B2_WRITE, &[f64::from(seq), ev_t3, super::evtrace::now(), buf.len() as f64]);
                let _ = tx_written.send((seq, Instant::now()));
                let _ = tx_resp_recycle.send(buf);
            }
        });
        // Compute loop (this thread owns the GPU). Runs in a closure so the
        // channel senders can be dropped and the socket shut down BEFORE the
        // scope joins the reader/writer threads (the writer ends when every
        // `tx_out` sender is gone; the reader when its blocking read fails).
        let mut compute = |tx_out: mpsc::SyncSender<(AlignedBuf, Instant, u32)>, records: &mut Vec<RequestRecord>, n_total: &mut u64| -> eyre::Result<()> {
        let mut n_done = 0usize;
        // PARKING (`knobs::park`): requests served while an earlier one waits on
        // its miss reads run on this second executor, so the parked request's
        // xq / partials / sel regions on `exec` stay intact. Created on first use.
        let mut exec2: Option<MoeExecutor> = None;
        let (mut w_parks, mut w_park_served) = (0u64, 0u64);
        // Park loop time split: serving others vs waiting on own reads (window sums).
        let (mut w_park_wait_ns, mut w_park_serve_ns) = (0u64, 0u64);
        // Prefetch read timing at the last stats line (for window deltas).
        let mut pf_prev = (0u64, 0u64, 0u64);
        // A frame the overlap hook already pulled off the reader (and whose
        // misses it may have started paging) while the previous request ran.
        let mut pending: std::collections::VecDeque<Inbound> = std::collections::VecDeque::new();
        // USE saturation for this thread (profile audit 2026-09-21): how often a
        // request found the queue non-empty, how deep, and how long the GPU sat
        // idle between requests. Windowed with the page-stats print.
        let mut t_prev_ready: Option<Instant> = None;
        let (mut w_idle_ns, mut w_service_ns, mut w_depth_sum, mut w_queued, mut w_n, mut w_merged) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        // Why a merge did NOT happen: no frame available at all vs. one there but
        // not mergeable (different layer / reply format / would exceed max_batch).
        let (mut w_merge_no_frame, mut w_merge_unmergeable, mut w_promised) = (0u64, 0u64, 0u64);
        let mut w_t0 = Instant::now();
        loop {
            let depth_on_take = pending.len() as u64;
            let msg = if let Some(m) = pending.pop_front() {
                m
            } else if opts.keep_warm_us == 0 {
                match rx_in.recv() {
                    Ok(m) => m,
                    Err(_) => break,
                }
            } else {
                // Wait with a timeout; on every timeout poke the GPU.
                let period = Duration::from_micros(opts.keep_warm_us);
                let mut got = None;
                loop {
                    match rx_in.recv_timeout(period) {
                        Ok(m) => {
                            got = Some(m);
                            break;
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => exec.keep_warm()?,
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                match got {
                    Some(m) => m,
                    None => break,
                }
            };
            let (hdr, buf, t_first, t_done, t2) = match msg {
                Inbound::Closed(None) => break,
                Inbound::Closed(Some(e)) => return Err(eyre!("reader: {e}")),
                Inbound::Frame { hdr, buf, t_first, t_done, t2 } => (hdr, buf, t_first, t_done, t2),
            };
            if knobs::DIRTY.swap(false, std::sync::atomic::Ordering::Relaxed) {
                eprintln!("expertd: {}", knobs::reload());
            }
            let t_start = Instant::now();
            // `evtrace` (`b2_req`): stamps + reader state at dequeue.
            let ev_on = super::evtrace::enabled();
            let nan = f64::NAN;
            let ev_t_dequeue = if ev_on { super::evtrace::now() } else { nan };
            let (ev_t_hdr, ev_t_frame) = if ev_on { (super::evtrace::inst_to_raw(t_first), t2 as f64) } else { (nan, nan) };
            let ev_idle_us = t_prev_ready.map_or(nan, |p| t_start.saturating_duration_since(p).as_secs_f64() * 1e6);
            let ev_pf0 = if ev_on { shard.ev_pf_snapshot() } else { [nan; 12] };
            let mut ev_promised = nan;
            if ev_on {
                EV_CUR_SEQ.store(u64::from(hdr.seq), std::sync::atomic::Ordering::Relaxed);
            }
            if let Some(p) = t_prev_ready {
                w_idle_ns += t_start.saturating_duration_since(p).as_nanos() as u64;
            }
            w_n += 1;
            if depth_on_take > 0 {
                w_queued += 1;
                w_depth_sum += depth_on_take;
            }
            if proto::decode_request(&buf).map(|r| r.flags & proto::REQ_FLAG_PARTNER != 0).unwrap_or(false) {
                w_promised += 1;
            }
            let t_start_rt = tracer.map(|_| super::perfetto::host_now_ns());
            let mut resp = rx_resp_recycle.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(proto::RESP_DATA_OFF + exec.rows() * N_EMBD as usize * 4));
            if hdr.kind != proto::KIND_REQUEST {
                proto::encode_error(&mut resp, hdr.seq, 2, &format!("unexpected frame kind {}", hdr.kind));
                let _ = tx_out.send((resp, Instant::now(), hdr.seq));
                return Err(eyre!("unexpected frame kind {}", hdr.kind));
            }
            // Same-layer coalescing (`b2_merge`): the OTHER hub lane's request for
            // this same layer rides along in this pass, so the two lanes stop
            // streaming the same experts twice (at prefill B=512 the union is
            // ~all 384, so a merged 1024-row pass is half the bytes).
            //
            // The partner is looked for in `pending` AND on the socket channel:
            // the loop above pops `pending`'s only entry as THIS request, so with
            // two hub lanes `pending` is empty here and a pending-only check never
            // fired (measured 2026-09-22: "queued 79% ... merged 0%").
            let mut partner: Option<(proto::Header, AlignedBuf, Instant, Instant, u64)> = None;
            if b2_merge() {
                let mergeable = |buf2: &AlignedBuf| -> bool {
                    match (proto::decode_request(&buf), proto::decode_request(buf2)) {
                        (Ok(ra), Ok(rb)) => {
                            let fm = proto::REQ_FLAG_RESP_F32 | proto::REQ_FLAG_BATCHED;
                            ra.layer == rb.layer
                                && (ra.flags & fm) == (rb.flags & fm)
                                && ra.b as usize > exec.decode_max_b()
                                && rb.b as usize > exec.decode_max_b()
                                && (ra.b + rb.b) as usize <= exec.rows()
                        }
                        _ => false,
                    }
                };
                // Does the SENDER say a partner is coming? `REQ_FLAG_PARTNER` is
                // set by the hub on the first of the two lanes' same-layer
                // requests, so an idle daemon knows to hold instead of starting
                // a pass it cannot merge into.
                let partner_promised = proto::decode_request(&buf)
                    .map(|r| r.flags & proto::REQ_FLAG_PARTNER != 0)
                    .unwrap_or(false);
                ev_promised = f64::from(u8::from(partner_promised));
                let from_pending = matches!(pending.front(),
                    Some(Inbound::Frame { hdr: h2, buf: buf2, .. }) if h2.kind == proto::KIND_REQUEST && mergeable(buf2));
                if !pending.is_empty() && !from_pending {
                    w_merge_unmergeable += 1;
                }
                if from_pending {
                    if let Some(Inbound::Frame { hdr: h2, buf: buf2, t_first: tf2, t_done: td2, t2: t22 }) = pending.pop_front() {
                        partner = Some((h2, buf2, tf2, td2, t22));
                    }
                } else if pending.is_empty() {
                    // Nothing queued here: take a look at the socket. A frame that
                    // is NOT mergeable goes to `pending` untouched, so ordering and
                    // the early-paging hook are unaffected.
                    // Bounded wait ONLY when the sender promised a partner; a
                    // plain try_recv loses the race almost always (the reader
                    // thread has not decoded the frame yet, or it is still in the
                    // socket buffer). `V41_B2_MERGE_WAIT_US` (default 400) caps
                    // what an unfulfilled promise can cost; 0 restores try_recv.
                    let got = if partner_promised && b2_merge_wait_us() > 0 {
                        rx_in.recv_timeout(Duration::from_micros(b2_merge_wait_us())).ok()
                    } else {
                        rx_in.try_recv().ok()
                    };
                    match got {
                        Some(m) => {
                            let ok = matches!(&m,
                                Inbound::Frame { hdr: h2, buf: buf2, .. } if h2.kind == proto::KIND_REQUEST && mergeable(buf2));
                            if ok {
                                if let Inbound::Frame { hdr: h2, buf: buf2, t_first: tf2, t_done: td2, t2: t22 } = m {
                                    partner = Some((h2, buf2, tf2, td2, t22));
                                }
                            } else {
                                // Pin mode: note its arrival now. It is served
                                // later from `pending`, and `pull`'s hook only
                                // sees frames it receives itself.
                                if let Inbound::Frame { hdr: h2, buf: buf2, .. } = &m {
                                    if h2.kind == proto::KIND_REQUEST {
                                        if let Ok(nr) = proto::decode_request(buf2) {
                                            shard.note_early_paged(h2.seq, nr.layer, nr.sel);
                                        }
                                    }
                                }
                                w_merge_unmergeable += 1;
                                pending.push_back(m);
                            }
                        }
                        None => w_merge_no_frame += 1,
                    }
                }
            }
            let ev_t_merge = if ev_on { super::evtrace::now() } else { nan };
            let ev_partner = partner.as_ref().map(|p| (f64::from(p.0.seq), p.1.len()));
            let outcome: eyre::Result<(RequestRecord, u32, Option<(RequestRecord, u32, AlignedBuf)>)> = (|| {
                let req = proto::decode_request(&buf)?;
                if req.n_used != N_EXPERT_USED as u32 || req.xq_bpt != XQ_BYTES_PER_TOKEN as u32 {
                    return Err(eyre!("request geometry n_used={} xq_bpt={} != {}/{}", req.n_used, req.xq_bpt, N_EXPERT_USED, XQ_BYTES_PER_TOKEN));
                }
                let b = req.b as usize;
                let reqb = match partner.as_ref() {
                    Some((_, bufb, ..)) => Some(proto::decode_request(bufb)?),
                    None => None,
                };
                let bb = reqb.as_ref().map(|r| r.b as usize).unwrap_or(0);
                // Reader spin window for the next request, by the hub's phase.
                {
                    let elem_out = if req.flags & proto::REQ_FLAG_RESP_F32 != 0 { 4 } else { 2 };
                    let reply = proto::RESP_DATA_OFF + b.max(bb) * N_EMBD as usize * elem_out;
                    b2_adapt_busy_poll(&stream, req.flags & proto::REQ_FLAG_DECODE != 0, buf.len(), reply, opts.socket.busy_poll_us);
                }
                // PINNING (`REQ_FLAG_PIN`): release, then grant, in ARRIVAL
                // order -- this request's words before its partner's -- so the
                // reply epochs count a prefix of the words the hub sent.
                if req.flags & proto::REQ_FLAG_PIN != 0 || reqb.as_ref().is_some_and(|rb| rb.flags & proto::REQ_FLAG_PIN != 0) {
                    shard.pin_enable();
                }
                // Mode-aware eviction: the hub's phase, before this pass (and
                // its partner's) claims or lands anything.
                shard.note_request_phase(req.flags);
                if let Some(rb) = reqb.as_ref() {
                    shard.note_request_phase(rb.flags);
                }
                shard.pump_restore();
                let ev_pin0 = shard.pin_counters();
                let ev_sc0 = shard.stage_counters();
                // A prefill-shaped request's prefetch words are layer-major group
                // prefetch, never pin grants (they would pin prefill experts).
                shard.pin_apply_words(req.release, if req.b > proto::PIN_DECODE_MAX_ROWS { &[][..] } else { req.prefetch });
                if let Some(rb) = reqb.as_ref() {
                    shard.pin_apply_words(rb.release, if rb.b > proto::PIN_DECODE_MAX_ROWS { &[][..] } else { rb.prefetch });
                }
                if let Some(rb) = reqb.as_ref() {
                    if !rb.hint_admit.is_empty() {
                        shard.hint_evict_first(rb.hint_admit);
                    }
                    if !rb.prefetch.is_empty() {
                        if rb.b > proto::PIN_DECODE_MAX_ROWS {
                            shard.prefetch_words_prefill(rb.prefetch);
                        } else {
                            shard.prefetch_words(rb.prefetch);
                        }
                    }
                }
                // Paging THIS request did on our own NVMe. `run_path` calls
                // `ensure_layer*` internally, so bracket it and difference the
                // layer's cumulative counters. Reported back so the hub can draw a
                // `remote.pager` lane and tell a box-2 NVMe stall apart from
                // queueing or compute -- previously indistinguishable from the hub,
                // which only saw one opaque round trip.
                if !req.hint_admit.is_empty() {
                    shard.hint_evict_first(req.hint_admit);
                }
                if !req.prefetch.is_empty() {
                    if req.b > proto::PIN_DECODE_MAX_ROWS {
                        shard.prefetch_words_prefill(req.prefetch);
                    } else {
                        shard.prefetch_words(req.prefetch);
                    }
                }
                let ev_t_hints = if ev_on { super::evtrace::now() } else { nan };
                let ev_pd0 = shard.layer_page_detail(req.layer);
                let ev_pw0 = shard.prefetch_wait_ns;
                let (miss0, page_ns0) = shard.layer_page_counters(req.layer);
                // Early paging for the NEXT queued request: the daemon used to
                // start B's misses only after A's reply, so with two hub lanes
                // B's read latency sat fully exposed behind A's GPU tail, D2H,
                // reply and the hub's turnaround (~1-2 ms per request, 2026-09-21
                // profile: box 2 ~70% busy, ~25% of it waiting). Here A's picks
                // are pinned and B's non-resident picks go to the prefetch
                // readers as CERTAIN hints; B's own `ensure` then finds them in
                // flight and waits for the remainder (`waited`) instead of
                // reading from scratch. `V41_B2_EARLY_PAGE=0` disables.
                let mut cur_pins: Vec<(u32, u32)> = req.sel.iter().filter(|&&e| e >= 0).map(|&e| (req.layer, e as u32)).collect();
                if let Some(rb) = reqb.as_ref() {
                    cur_pins.extend(rb.sel.iter().filter(|&&e| e >= 0).map(|&e| (req.layer, e as u32)));
                }
                let (exec_device, exec_rows, exec_decode_max_b) = (exec.device(), exec.rows(), exec.decode_max_b());
                let rx_in_ref = &rx_in;
                let pending_ref = &mut pending;
                // May THIS request be answered after ones queued behind it?
                let may_park = req.flags & proto::REQ_FLAG_OOO != 0 && reqb.is_none();
                let exec2_ref = &mut exec2;
                let records_ref = &mut *records;
                let n_done_ref = &mut n_done;
                let (parks_ref, park_served_ref) = (&mut w_parks, &mut w_park_served);
                // THIS request's park loop: time spent serving others (not its
                // compute) and waiting on its own reads (its paging). Reported
                // in its reply so the hub's box2.compute/page stay honest.
                let (mut this_park_serve_ns, mut this_park_wait_ns) = (0u64, 0u64);
                let (serve_acc, wait_acc) = (&mut this_park_serve_ns, &mut this_park_wait_ns);
                let tx_out_ref = &tx_out;
                let rx_resp_recycle_ref = &rx_resp_recycle;
                let tx_req_recycle_ref = &tx_req_recycle;
                // Pull EVERY frame the reader has (three hub lanes keep up to
                // two queued); each one's misses start now, in arrival order.
                // With `park`, also serve queued requests while this one's
                // misses read (see `MoeExecutor::run_path`).
                let mut overlap = |shard: &mut ExpertShard, park: bool| -> eyre::Result<()> {
                    // Parking needs the pull below too: it only serves frames
                    // already taken off the reader.
                    if !b2_early_page() {
                        return Ok(());
                    }
                    // A queued request's misses start reading as soon as its frame
                    // is seen, pinned against this request's picks.
                    let early_page = |shard: &mut ExpertShard, m: &Inbound| {
                        let Inbound::Frame { hdr, buf, .. } = m else { return };
                        if hdr.kind != proto::KIND_REQUEST {
                            return;
                        }
                        let Ok(nreq) = proto::decode_request(buf) else { return };
                        if !shard.layer_is_paged(nreq.layer) {
                            return;
                        }
                        shard.note_early_paged(hdr.seq, nreq.layer, nreq.sel);
                        let mut words: Vec<u32> = Vec::with_capacity(nreq.sel.len());
                        for &e in nreq.sel {
                            if (0..N_EXPERT as i32).contains(&e) && !shard.is_resident_pool(nreq.layer, e as u32) {
                                let w = (nreq.layer << 16) | e as u32;
                                if !words.contains(&w) { words.push(w); }
                            }
                        }
                        if !words.is_empty() {
                            shard.pinned = cur_pins.clone();
                            // A prefill chunk's reads land in the staging band.
                            // Mode-aware eviction: the frame's hub flag, not its shape.
                            let pf_class = if shard.mode_evict_on() { nreq.flags & proto::REQ_FLAG_DECODE == 0 } else { nreq.b > proto::PIN_DECODE_MAX_ROWS };
                            shard.prefetch_words_cls(&words, true, pf_class);
                            shard.pinned.clear();
                        }
                    };
                    let pull = |shard: &mut ExpertShard, pending: &mut std::collections::VecDeque<Inbound>| {
                        while let Ok(m) = rx_in_ref.try_recv() {
                            early_page(shard, &m);
                            let stop = matches!(m, Inbound::Closed(_));
                            pending.push_back(m);
                            if stop { break; }
                        }
                    };
                    pull(shard, pending_ref);
                    if !(park && may_park) {
                        return Ok(());
                    }
                    // `PARK_MAX_ROWS`: the pin reserve counts on it.
                    let rows2 = PARK_MAX_ROWS.min(exec_rows);
                    let servable = |m: &Inbound| -> bool {
                        match m {
                            Inbound::Frame { hdr, buf, .. } if hdr.kind == proto::KIND_REQUEST => {
                                proto::decode_request(buf)
                                    .map(|r| r.flags & proto::REQ_FLAG_OOO != 0 && (r.b as usize) <= rows2)
                                    .unwrap_or(false)
                            }
                            _ => false,
                        }
                    };
                    // Hand this request's misses to the prefetch readers NOW,
                    // whether or not anything is queued yet: the other lane's
                    // frame usually lands DURING the read (checking once, before
                    // it, parked 2-5 of ~2000 requests on 2026-09-23). Then, until
                    // the reads have landed, serve every servable frame that
                    // arrives. `ensure` afterwards admits what is left.
                    let words = std::mem::take(&mut shard.park_words);
                    let park_prefill = shard.park_prefill;
                    shard.prefetch_words_cls(&words, true, park_prefill);
                    static PARK_LOG: std::sync::LazyLock<bool> =
                        std::sync::LazyLock::new(|| std::env::var("V41_B2_PARK_LOG").as_deref() == Ok("1"));
                    // Bound: a lost read (dropped hint, failed prefetch) falls
                    // through to `ensure`, which reads it itself.
                    const PARK_MAX: Duration = Duration::from_millis(50);
                    let t_park = Instant::now();
                    let mut counted = false;
                    loop {
                        while pending_ref.front().is_some_and(|m| servable(m)) {
                            if exec2_ref.is_none() {
                                *exec2_ref = Some(MoeExecutor::new(exec_device, rows2, exec_decode_max_b)?);
                                eprintln!("expertd: park executor ready ({rows2} rows)");
                            }
                            if !counted {
                                counted = true;
                                *parks_ref += 1;
                                if *PARK_LOG {
                                    eprintln!("expertd: park L{} misses={} queued={} after {} us", req.layer, words.len(), pending_ref.len(), t_park.elapsed().as_micros());
                                }
                            }
                            let ex2 = exec2_ref.as_mut().unwrap();
                            let Some(Inbound::Frame { hdr, buf, t_first, t_done, t2 }) = pending_ref.pop_front() else { unreachable!() };
                            let resp = rx_resp_recycle_ref.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(proto::RESP_DATA_OFF + exec_rows * N_EMBD as usize * 4));
                            let t_serve = Instant::now();
                            if let Ok(r) = proto::decode_request(&buf) {
                                let elem_out = if r.flags & proto::REQ_FLAG_RESP_F32 != 0 { 4 } else { 2 };
                                let reply = proto::RESP_DATA_OFF + r.b as usize * N_EMBD as usize * elem_out;
                                b2_adapt_busy_poll(&stream, r.flags & proto::REQ_FLAG_DECODE != 0, buf.len(), reply, opts.socket.busy_poll_us);
                            }
                            // The served-inside frame sets its own request mode;
                            // the parked request's is restored for its pass.
                            let parked_mode = shard.req_prefill;
                            let out = serve_interleaved(ex2, shard, &hdr, &buf, t_first, t_done, t2, resp);
                            shard.req_prefill = parked_mode;
                            let _ = tx_req_recycle_ref.send(buf);
                            *serve_acc += t_serve.elapsed().as_nanos() as u64;
                            match out {
                                Ok((rec, resp)) => {
                                    let t_ready = rec.t_ready;
                                    records_ref.push(rec);
                                    if tx_out_ref.send((resp, t_ready, hdr.seq)).is_err() {
                                        return Err(eyre!("writer thread gone"));
                                    }
                                    *n_done_ref += 1;
                                    *park_served_ref += 1;
                                }
                                Err(e) => {
                                    let msg = format!("{e:#}");
                                    eprintln!("expertd: interleaved request seq {} failed: {msg}", hdr.seq);
                                    let mut eb = AlignedBuf::with_capacity(4096);
                                    proto::encode_error(&mut eb, hdr.seq, 1, &msg);
                                    let _ = tx_out_ref.send((eb, Instant::now(), hdr.seq));
                                    return Err(eyre!("interleaved request failed: {msg}"));
                                }
                            }
                            pull(shard, pending_ref);
                        }
                        // Land finished reads (never blocks: nothing is `want`ed);
                        // this request's picks are pinned, so none is a victim.
                        shard.admit_landed(req.layer)?;
                        if !shard.prefetch_pending_any(&words) || t_park.elapsed() >= PARK_MAX {
                            break;
                        }
                        // Anything already queued that is NOT servable waits its
                        // turn behind us anyway; stop polling for it.
                        if pending_ref.front().is_some_and(|m| !servable(m)) {
                            break;
                        }
                        match rx_in_ref.recv_timeout(Duration::from_micros(100)) {
                            Ok(m) => {
                                early_page(shard, &m);
                                let stop = matches!(m, Inbound::Closed(_));
                                pending_ref.push_back(m);
                                if stop {
                                    break;
                                }
                            }
                            Err(mpsc::RecvTimeoutError::Timeout) => {}
                            Err(mpsc::RecvTimeoutError::Disconnected) => break,
                        }
                    }
                    *wait_acc += (t_park.elapsed().as_nanos() as u64).saturating_sub(*serve_acc);
                    shard.park_words = words;
                    Ok(())
                };
                let (xq_m, sel_m, ew_m);
                let (xq_run, sel_run, ew_run): (&[u8], &[i32], &[f32]) = match reqb.as_ref() {
                    Some(rb) => {
                        xq_m = [req.xq, rb.xq].concat();
                        sel_m = [req.sel, rb.sel].concat();
                        ew_m = [req.ew, rb.ew].concat();
                        (&xq_m, &sel_m, &ew_m)
                    }
                    None => (req.xq, req.sel, req.ew),
                };
                let ev_t_run0 = if ev_on { super::evtrace::now() } else { nan };
                // A merged partner has the same reply format (`mergeable`).
                exec.set_reply_f16(req.flags & proto::REQ_FLAG_RESP_F32 == 0);
                let timing = exec.run_path(shard, req.layer, b + bb, xq_run, sel_run, ew_run, req.flags & proto::REQ_FLAG_BATCHED != 0, &mut overlap)?;
                let ev_t_run1 = if ev_on { super::evtrace::now() } else { nan };
                drop(overlap);
                w_park_wait_ns += this_park_wait_ns;
                w_park_serve_ns += this_park_serve_ns;
                let (miss1, page_ns1) = shard.layer_page_counters(req.layer);
                // Own park-loop wait IS this request's paging (its reads, landing).
                let t_page_us = ((page_ns1.saturating_sub(page_ns0) + this_park_wait_ns) / 1000).min(u32::MAX as u64) as u32;
                let n_miss_req = miss1.saturating_sub(miss0).min(u32::MAX as u64) as u32;
                let t_d2h0 = Instant::now();
                let f32_out = req.flags & proto::REQ_FLAG_RESP_F32 != 0;
                let elem = if f32_out { 4 } else { 2 };
                let n = b * N_EMBD as usize;
                // Time spent serving OTHER requests while parked is not ours.
                let t_compute_us = ((t_d2h0 - t_start).as_micros() as u64).saturating_sub(this_park_serve_ns / 1000) as u32;
                let resp_flags = (req.flags & !proto::RESP_MISS_MASK)
                    | ((timing.miss_mask << proto::RESP_MISS_SHIFT) & proto::RESP_MISS_MASK);
                proto::begin_response(&mut resp, hdr.seq, req.layer, req.b, resp_flags, 0, t_compute_us, 0, N_EMBD, elem, req.t1, t2);
                resp.resize(proto::RESP_DATA_OFF + n * elem as usize);
                if f32_out {
                    exec.read_f32_at(0, b, resp.view_mut::<f32>(proto::RESP_DATA_OFF, n))?;
                } else {
                    exec.read_f16_at(0, b, resp.view_mut::<u16>(proto::RESP_DATA_OFF, n))?;
                }
                let ev_t_d2h = if ev_on { super::evtrace::now() } else { nan };
                // Pin mode: the map is the PINNED set (reporting pins), then the
                // pin block with this pass's paged bits. Each decode-shaped
                // REQUEST's picks become pinnable here (per request, like the
                // hub's release ranking: two mergeable 9-32-row lanes make a
                // pass of up to 64 rows that is still decode), before either
                // report so both maps carry them.
                if req.flags & proto::REQ_FLAG_PIN != 0 && req.b <= proto::PIN_DECODE_MAX_ROWS {
                    shard.pin_grant(req.layer, req.sel);
                }
                if let Some(rb) = reqb.as_ref() {
                    if rb.flags & proto::REQ_FLAG_PIN != 0 && rb.b <= proto::PIN_DECODE_MAX_ROWS {
                        shard.pin_grant(rb.layer, rb.sel);
                    }
                }
                // PAGED bits: what the pass found not landed when it started,
                // plus what the early-page hook found not landed when the
                // frame(s) arrived (a read it finished before the pass began
                // was still a page the hub did not expect).
                // Per REQUEST: the pass's bits are shared, each request's
                // arrival bits are its own (a merged partner is scored against
                // what was missing when ITS frame arrived, not its lane mate's).
                let mut paged = timing.paged;
                let mut paged_b = timing.paged;
                if req.flags & proto::REQ_FLAG_PIN != 0 {
                    or_words(&mut paged, &shard.take_early_paged(hdr.seq));
                }
                if let (Some(rb), Some((hb, ..))) = (reqb.as_ref(), partner.as_ref()) {
                    if rb.flags & proto::REQ_FLAG_PIN != 0 {
                        or_words(&mut paged_b, &shard.take_early_paged(hb.seq));
                    }
                }
                // One report per pass: the partner (same layer, `mergeable`)
                // reuses it, so a merged pass ages the layer's grants once and
                // scans it once.
                let pin_a = if req.flags & proto::REQ_FLAG_PIN != 0 { shard.pin_report(req.layer, &[], false) } else { None };
                match pin_a {
                    Some((map, [epoch, pinned, budget])) => {
                        proto::append_residency(&mut resp, &map);
                        proto::append_pin(&mut resp, epoch, pinned, budget, &paged);
                    }
                    None if req.flags & proto::REQ_FLAG_RESID != 0 => {
                        proto::append_residency(&mut resp, &shard.residency_words(req.layer));
                    }
                    None => {}
                }
                // The partner's reply: rows [b, b + bb) of the same pass. Page
                // time and miss count are reported on THIS request only, so the
                // hub's per-step sums are unchanged.
                let extra: Option<(RequestRecord, u32, AlignedBuf)> = match (reqb.as_ref(), partner.as_ref()) {
                    (Some(rb), Some((hb, bufb, tfb, tdb, t2b))) => {
                        let nb = bb * N_EMBD as usize;
                        let mut resp_b = rx_resp_recycle.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(proto::RESP_DATA_OFF + exec.rows() * N_EMBD as usize * 4));
                        proto::begin_response(&mut resp_b, hb.seq, rb.layer, rb.b, resp_flags, 0, t_compute_us, 0, N_EMBD, elem, rb.t1, *t2b);
                        resp_b.resize(proto::RESP_DATA_OFF + nb * elem as usize);
                        if f32_out {
                            exec.read_f32_at(b, bb, resp_b.view_mut::<f32>(proto::RESP_DATA_OFF, nb))?;
                        } else {
                            exec.read_f16_at(b, bb, resp_b.view_mut::<u16>(proto::RESP_DATA_OFF, nb))?;
                        }
                        let pin_b = if rb.flags & proto::REQ_FLAG_PIN != 0 {
                            pin_a.or_else(|| shard.pin_report(rb.layer, &[], false))
                        } else {
                            None
                        };
                        match pin_b {
                            Some((map, [epoch, pinned, budget])) => {
                                proto::append_residency(&mut resp_b, &map);
                                proto::append_pin(&mut resp_b, epoch, pinned, budget, &paged_b);
                            }
                            None if rb.flags & proto::REQ_FLAG_RESID != 0 => {
                                proto::append_residency(&mut resp_b, &shard.residency_words(rb.layer));
                            }
                            None => {}
                        }
                        proto::patch_len(&mut resp_b);
                        let t_ready_b = Instant::now();
                        let t_server_us_b = (t_ready_b - *tdb).as_micros() as u32;
                        resp_b.as_bytes_mut()[proto::HDR_LEN + 20..proto::HDR_LEN + 24].copy_from_slice(&t_server_us_b.to_le_bytes());
                        resp_b.as_bytes_mut()[proto::RESP_PAGE_OFF..proto::RESP_PAGE_OFF + 4].copy_from_slice(&0u32.to_le_bytes());
                        resp_b.as_bytes_mut()[proto::RESP_MISSN_OFF..proto::RESP_MISSN_OFF + 4].copy_from_slice(&0u32.to_le_bytes());
                        let rec_b = RequestRecord {
                            seq: hb.seq,
                            layer: rb.layer,
                            b: rb.b,
                            bytes_in: bufb.len(),
                            bytes_out: resp_b.len(),
                            read_us: (*tdb - *tfb).as_micros() as u32,
                            queue_us: (t_start - *tdb).as_micros() as u32,
                            h2d_us: 0,
                            gpu_us: timing.gpu.as_micros() as u32,
                            d2h_us: (t_ready_b - t_d2h0).as_micros() as u32,
                            write_us: 0,
                            path_decode: timing.path_decode,
                            t_ready: t_ready_b,
                            t1: rb.t1,
                            t2: *t2b,
                        };
                        Some((rec_b, hb.seq, resp_b))
                    }
                    _ => None,
                };
                // `V41_B2_DBG=1`: hash the payload box 2 is about to SEND. Same
                // FNV over every 97th f32 as box 1's [partial-src], so the two
                // sides' duplicate structure is directly comparable: duplicates
                // here mean box 2 computed/read the same bytes twice; duplicates
                // only on box 1 mean the wire or client buffer lifecycle.
                static B2_DBG: std::sync::LazyLock<bool> =
                    std::sync::LazyLock::new(|| std::env::var("V41_B2_DBG").is_ok());
                if *B2_DBG && f32_out {
                    let payload = resp.view::<f32>(proto::RESP_DATA_OFF, n);
                    let mut h: u64 = 0xcbf29ce484222325;
                    for &v in payload.iter().step_by(97) {
                        h ^= v.to_bits() as u64;
                        h = h.wrapping_mul(0x100000001b3);
                    }
                    eprintln!(
                        "[b2-send] L{} seq={} b={} compute_us={} hash={h:016x}",
                        req.layer, hdr.seq, b, t_compute_us
                    );
                }
                proto::patch_len(&mut resp);
                let t_ready = Instant::now();
                // t_server = frame complete → response handed to the writer.
                let t_server_us = (t_ready - t_done).as_micros() as u32;
                resp.as_bytes_mut()[proto::HDR_LEN + 20..proto::HDR_LEN + 24].copy_from_slice(&t_server_us.to_le_bytes());
                resp.as_bytes_mut()[proto::RESP_PAGE_OFF..proto::RESP_PAGE_OFF + 4]
                    .copy_from_slice(&t_page_us.to_le_bytes());
                resp.as_bytes_mut()[proto::RESP_MISSN_OFF..proto::RESP_MISSN_OFF + 4]
                    .copy_from_slice(&n_miss_req.to_le_bytes());
                let rec = RequestRecord {
                    seq: hdr.seq,
                    layer: req.layer,
                    b: req.b,
                    bytes_in: buf.len(),
                    bytes_out: resp.len(),
                    read_us: (t_done - t_first).as_micros() as u32,
                    queue_us: (t_start - t_done).as_micros() as u32,
                    h2d_us: timing.h2d.as_micros() as u32,
                    gpu_us: timing.gpu.as_micros() as u32,
                    d2h_us: (t_ready - t_d2h0).as_micros() as u32,
                    write_us: 0,
                    path_decode: timing.path_decode,
                    t_ready,
                    t1: req.t1,
                    t2,
                };
                if ev_on {
                    let pd1 = shard.layer_page_detail(req.layer);
                    let pf1 = shard.ev_pf_snapshot();
                    let d = |i: usize| pd1[i].saturating_sub(ev_pd0[i]) as f64;
                    let mut sel: Vec<i32> = req.sel.iter().copied().filter(|&e| e >= 0).collect();
                    let n_sel = sel.len();
                    sel.sort_unstable();
                    sel.dedup();
                    let (pseq, pb) = ev_partner.map_or((nan, nan), |(s, _)| (s, bb as f64));
                    let mut v = vec![
                        f64::from(hdr.seq), f64::from(req.layer), f64::from(req.b), f64::from(req.flags),
                        f64::from(u8::from(ev_partner.is_some())), pseq, pb, ev_promised, nan,
                        ev_t_hdr, ev_t_frame, ev_t_dequeue, ev_t_merge, ev_t_hints, ev_t_run0, ev_t_run1, ev_t_d2h,
                        super::evtrace::inst_to_raw(t_ready),
                        depth_on_take as f64, pending_ref.len() as f64, ev_idle_us,
                        n_sel as f64, sel.len() as f64, req.hint_admit.len() as f64, req.prefetch.len() as f64,
                        f64::from(n_miss_req), f64::from(t_page_us), f64::from(t_compute_us), f64::from(t_server_us),
                        d(0), d(1), d(2), d(3), d(4), d(5), shard.prefetch_wait_ns.saturating_sub(ev_pw0) as f64,
                        this_park_wait_ns as f64, this_park_serve_ns as f64,
                        f64::from(u8::from(timing.path_decode)), f64::from(u8::from(timing.two_pass)),
                        f64::from(timing.n_work_items), f64::from(timing.n_missing),
                        timing.h2d.as_secs_f64() * 1e6, timing.gpu.as_secs_f64() * 1e6,
                    ];
                    v.extend_from_slice(&ev_pf0[..6]);
                    for i in 6..11 {
                        v.push(pf1[i] - ev_pf0[i]);
                    }
                    v.push(pf1[11]);
                    v.extend_from_slice(&ev_pin_fields(ev_pin0, shard.pin_counters(), req.release.len(), &paged));
                    v.extend_from_slice(&ev_stage_fields(ev_sc0, shard.stage_counters()));
                    super::evtrace::emit(&super::evtrace_kinds::B2_REQ, &v);
                    // The merged partner: same pass, its own identity and arrival.
                    if let (Some(rb), Some((hb, _, tfb, _, t2b))) = (reqb.as_ref(), partner.as_ref()) {
                        let k = &super::evtrace_kinds::B2_REQ;
                        let mut sel_b: Vec<i32> = rb.sel.iter().copied().filter(|&e| e >= 0).collect();
                        let n_sel_b = sel_b.len();
                        sel_b.sort_unstable();
                        sel_b.dedup();
                        for (name, x) in [
                            ("seq", f64::from(hb.seq)), ("b", f64::from(rb.b)), ("flags", f64::from(rb.flags)), ("merged", 2.0),
                            ("partner_seq", f64::from(hdr.seq)), ("partner_b", b as f64),
                            ("t_hdr", super::evtrace::inst_to_raw(*tfb)), ("t_frame", *t2b as f64),
                            ("n_sel", n_sel_b as f64), ("n_distinct", sel_b.len() as f64),
                            ("n_hint_admit", rb.hint_admit.len() as f64), ("n_prefetch_words", rb.prefetch.len() as f64),
                            ("n_miss", 0.0), ("page_us", 0.0), ("server_us", nan),
                            // Per-PASS quantities are the carrier's: NaN here so
                            // a sum over rows counts each pass once.
                            ("d_misses", nan), ("d_read_ns", nan), ("d_h2d_ns", nan), ("d_repack_gpu_ns", nan),
                            ("d_pread_ns", nan), ("d_repack_cpu_ns", nan), ("d_prefetch_wait_ns", nan),
                            ("park_wait_ns", nan), ("park_serve_ns", nan), ("n_work_items", nan), ("n_missing", nan),
                            ("exec_h2d_us", nan), ("exec_gpu_us", nan),
                            ("pf_d_hinted", nan), ("pf_d_admitted", nan), ("pf_d_dropped", nan), ("pf_d_waited", nan),
                            ("pf_d_promoted", nan),
                            ("pin_release_words", rb.release.len() as f64), ("pin_new", nan), ("pin_denied", nan),
                            ("pin_drops_no_victim", nan), ("pin_evictions", nan),
                            ("n_paged", paged_b.iter().map(|w| w.count_ones()).sum::<u32>() as f64),
                            ("stage_claims", nan), ("stage_hits", nan), ("stage_spills", nan),
                        ] {
                            super::evtrace::set_named(k, &mut v, name, x);
                        }
                        super::evtrace::emit(k, &v);
                    }
                }
                Ok((rec, hdr.seq, extra))
            })();
            let _ = tx_req_recycle.send(buf);
            if let Some((_, bufb, ..)) = partner.take() {
                let _ = tx_req_recycle.send(bufb);
            }
            match outcome {
                Ok((rec, seq, extra)) => {
                    if let (Some(tr), Some(t0)) = (tracer, t_start_rt) {
                        let now = super::perfetto::host_now_ns();
                        tr.span(
                            BOX2_REQUEST_UUID,
                            &format!("L{} B={} {}", rec.layer, rec.b, if rec.path_decode { "decode" } else { "batched" }),
                            t0,
                            now,
                        );
                        if let Some((a, b)) = exec.device_events() {
                            tr.gpu_slice(&format!("moe L{} B={}", rec.layer, rec.b), a, b);
                        }
                        if opts.re_anchor_every > 0 && (n_done + 1) % opts.re_anchor_every == 0 {
                            let _ = tr.re_anchor(exec.device(), &exec.engine.compute);
                        }
                    }
                    t_prev_ready = Some(rec.t_ready);
                    w_service_ns += rec.t_ready.saturating_duration_since(t_start).as_nanos() as u64;
                    records.push(rec);
                    // Bound the per-connection history: it reached 1,088,120
                    // entries (~90 MB) on a day-long link. Keep the newest
                    // RECORDS_CAP for `summarize`; fold in the writer's
                    // completion stamps before dropping the older half.
                    const RECORDS_CAP: usize = 100_000;
                    if records.len() >= 2 * RECORDS_CAP {
                        apply_written(records, &rx_written);
                        records.drain(..records.len() - RECORDS_CAP);
                    }
                    if tx_out.send((resp, rec.t_ready, seq)).is_err() {
                        return Err(eyre!("writer thread gone"));
                    }
                    n_done += 1;
                    *n_total = n_done as u64;
                    if let Some((rec_b, seq_b, resp_b)) = extra {
                        // Coalesced partner: its reply follows A's on the wire
                        // (the hub waits FIFO by seq).
                        t_prev_ready = Some(rec_b.t_ready);
                        records.push(rec_b);
                        if tx_out.send((resp_b, rec_b.t_ready, seq_b)).is_err() {
                            return Err(eyre!("writer thread gone"));
                        }
                        n_done += 1;
                        *n_total = n_done as u64;
                        w_n += 1;
                        w_merged += 1;
                    }
                    // Catch-all tier: box 2 now owns ALL the paging, so its miss
                    // rate is the number that matters and the hub cannot see it.
                    // One line per 2000 requests (= per ~50 tokens at 40 layers).
                    if shard.is_paged() && n_done % 2000 == 0 {
                        let (req, miss, read_ns, h2d_ns) = shard.page_stats();
                        if miss > 0 {
                            let (pread_ns, rcpu_ns, rgpu_ns) = shard.page_read_split();
                            let per = |ns: u64| ns as f64 / miss as f64 / 1e6;
                            let mut pfs = shard.prefetch_stats().map(|(h, a, d, w)| format!(" prefetch hinted={h} admitted={a} dropped={d} waited={w} promoted={} wait_ms={:.0}", shard.prefetch_promoted(), shard.prefetch_wait_ns as f64 / 1e6)).unwrap_or_default();
                            if let Some((c, pinned, budget, _)) = shard.pin_counters() {
                                pfs.push_str(&format!(
                                    " | pins {pinned}/{budget} released={} new={} denied={} drops_no_victim={} pinned_evictions={} revokes={}",
                                    c.releases, c.new_pins, c.denied, c.no_victim_drops, c.pinned_evictions, c.revokes
                                ));
                            }
                            if shard.stage_slots() > 0 {
                                let s = shard.stage_counters();
                                pfs.push_str(&format!(
                                    " | stage {} claims={} hits={} spill_in={} spill_out={} drops={}",
                                    shard.stage_slots(), s.claims, s.hits, s.spill_in, s.spill_out, s.drops
                                ));
                            }
                            // `pread` here is the PROCESS-WIDE read counter differenced
                            // around demand chunks, so concurrent prefetch reads inflate
                            // it; read `read` (per-miss wall) instead.
                            let wall = w_t0.elapsed().as_secs_f64().max(1e-9);
                            let pf_now = shard.prefetch_read_timing();
                            let win = format!(
                                " | window {:.1}s: busy {:.0}% idle/req {:.2} ms queued {:.0}% depth {:.2} \
merged {:.0}% (promised {:.0}%, miss: no-frame {} unmergeable {}) parks {} served-under-park {} \
park wait {:.2} serve {:.2} ms/park | pf reads {} queue {:.2} read {:.2} ms/read",
                                wall, 100.0 * w_service_ns as f64 / 1e9 / wall,
                                w_idle_ns as f64 / 1e6 / w_n.max(1) as f64,
                                100.0 * w_queued as f64 / w_n.max(1) as f64,
                                w_depth_sum as f64 / w_queued.max(1) as f64,
                                100.0 * w_merged as f64 / w_n.max(1) as f64,
                                100.0 * w_promised as f64 / w_n.max(1) as f64,
                                w_merge_no_frame, w_merge_unmergeable, w_parks, w_park_served,
                                w_park_wait_ns as f64 / 1e6 / w_parks.max(1) as f64,
                                w_park_serve_ns as f64 / 1e6 / w_parks.max(1) as f64,
                                pf_now.2 - pf_prev.2,
                                (pf_now.0 - pf_prev.0) as f64 / 1e6 / (pf_now.2 - pf_prev.2).max(1) as f64,
                                (pf_now.1 - pf_prev.1) as f64 / 1e6 / (pf_now.2 - pf_prev.2).max(1) as f64,
                            );
                            pf_prev = pf_now;
                            eprintln!(
                                "expertd: page stats requests={req} misses={miss} hit={:.4} \
ms_per_miss={:.2} (read {:.2} [pread {:.2} repack_cpu {:.2}] h2d {:.2} repack_gpu {:.2}){pfs}{win}",
                                1.0 - miss as f64 / req.max(1) as f64,
                                (read_ns + h2d_ns) as f64 / miss as f64 / 1e6,
                                per(read_ns), per(pread_ns), per(rcpu_ns),
                                per(h2d_ns), per(rgpu_ns),
                            );
                            w_t0 = Instant::now();
                            w_idle_ns = 0; w_service_ns = 0; w_depth_sum = 0; w_queued = 0; w_n = 0; w_merged = 0;
                            w_merge_no_frame = 0; w_merge_unmergeable = 0; w_promised = 0;
                            w_parks = 0; w_park_served = 0; w_park_wait_ns = 0; w_park_serve_ns = 0;
                        }
                    }
                    if opts.verbose {
                        let r = records.last().unwrap();
                        eprintln!(
                            "expertd: seq {seq} L{} B={} {} in {} B out {} B | read {} us queue {} us h2d {} us gpu {} us d2h {} us",
                            r.layer, r.b, if r.path_decode { "decode" } else { "batched" }, r.bytes_in, r.bytes_out,
                            r.read_us, r.queue_us, r.h2d_us, r.gpu_us, r.d2h_us
                        );
                    }
                    if opts.log_every > 0 && n_done % opts.log_every == 0 {
                        summarize(&records[records.len().saturating_sub(opts.log_every)..], "expertd");
                    }
                }
                Err(e) => {
                    let msg = format!("{e:#}");
                    eprintln!("expertd: request seq {} failed: {msg}", hdr.seq);
                    proto::encode_error(&mut resp, hdr.seq, 1, &msg);
                    let _ = tx_out.send((resp, Instant::now(), hdr.seq));
                    return Err(eyre!("request failed: {msg}"));
                }
            }
        }
        Ok(())
        };
        let res = compute(tx_out, &mut records, &mut n_total);
        // Senders are gone (tx_out moved into `compute`); unblock the reader
        // (it may sit in read_exact) so the scope can join both threads.
        let _ = stream.shutdown(std::net::Shutdown::Both);
        res
    });
    // The writer has exited (scope joined it): fill in write completion times.
    apply_written(&mut records, &rx_written);
    result?;
    Ok((records, n_total))
}

/// Serve ONE plain request start to finish on `exec` (the park executor) and
/// build its reply: the non-merged half of `serve_connection`'s compute body.
/// Runs INSIDE a parked request's `run_path`, so it gets no hook of its own: a
/// miss here is read synchronously (it shares the drives with the parked
/// request's reads either way).
#[allow(clippy::too_many_arguments)]
fn serve_interleaved(
    exec: &mut MoeExecutor,
    shard: &mut ExpertShard,
    hdr: &proto::Header,
    buf: &AlignedBuf,
    t_first: Instant,
    t_done: Instant,
    t2: u64,
    mut resp: AlignedBuf,
) -> eyre::Result<(RequestRecord, AlignedBuf)> {
    let t_start = Instant::now();
    // `evtrace`: records under this request carry ITS seq; the parked
    // request's is restored on every exit (the guard), incl. errors.
    struct SeqGuard(u64);
    impl Drop for SeqGuard {
        fn drop(&mut self) {
            EV_CUR_SEQ.store(self.0, std::sync::atomic::Ordering::Relaxed);
        }
    }
    let ev_on = super::evtrace::enabled();
    let ev_t_dequeue = if ev_on { super::evtrace::now() } else { f64::NAN };
    let ev_guard = ev_on.then(|| SeqGuard(EV_CUR_SEQ.swap(u64::from(hdr.seq), std::sync::atomic::Ordering::Relaxed)));
    let req = proto::decode_request(buf)?;
    if req.n_used != N_EXPERT_USED as u32 || req.xq_bpt != XQ_BYTES_PER_TOKEN as u32 {
        return Err(eyre!("request geometry n_used={} xq_bpt={} != {}/{}", req.n_used, req.xq_bpt, N_EXPERT_USED, XQ_BYTES_PER_TOKEN));
    }
    let b = req.b as usize;
    // Pin words: served in arrival order right after the parked request, whose
    // own words were applied before it parked.
    if req.flags & proto::REQ_FLAG_PIN != 0 {
        shard.pin_enable();
    }
    // A frame served inside a parked pass: its own claim mode, but it does not
    // move the phase (a burst of decode frames inside one park must not end a
    // prefill phase under the parked chunk).
    shard.set_request_mode(req.flags);
    let ev_pin0 = shard.pin_counters();
    let ev_sc0 = shard.stage_counters();
    shard.pin_apply_words(req.release, if req.b > proto::PIN_DECODE_MAX_ROWS { &[][..] } else { req.prefetch });
    if !req.hint_admit.is_empty() {
        shard.hint_evict_first(req.hint_admit);
    }
    if !req.prefetch.is_empty() {
        if req.b > proto::PIN_DECODE_MAX_ROWS {
            shard.prefetch_words_prefill(req.prefetch);
        } else {
            shard.prefetch_words(req.prefetch);
        }
    }
    let ev_pd0 = shard.layer_page_detail(req.layer);
    let ev_pw0 = shard.prefetch_wait_ns;
    let ev_t_run0 = if ev_on { super::evtrace::now() } else { f64::NAN };
    let (miss0, page_ns0) = shard.layer_page_counters(req.layer);
    exec.set_reply_f16(req.flags & proto::REQ_FLAG_RESP_F32 == 0);
    let timing = exec.run_path(shard, req.layer, b, req.xq, req.sel, req.ew, req.flags & proto::REQ_FLAG_BATCHED != 0, &mut |_, _| Ok(()))?;
    let ev_t_run1 = if ev_on { super::evtrace::now() } else { f64::NAN };
    let (miss1, page_ns1) = shard.layer_page_counters(req.layer);
    let t_page_us = (page_ns1.saturating_sub(page_ns0) / 1000).min(u32::MAX as u64) as u32;
    let n_miss_req = miss1.saturating_sub(miss0).min(u32::MAX as u64) as u32;
    let t_d2h0 = Instant::now();
    let f32_out = req.flags & proto::REQ_FLAG_RESP_F32 != 0;
    let elem = if f32_out { 4 } else { 2 };
    let n = b * N_EMBD as usize;
    let t_compute_us = (t_d2h0 - t_start).as_micros() as u32;
    let resp_flags = (req.flags & !proto::RESP_MISS_MASK)
        | ((timing.miss_mask << proto::RESP_MISS_SHIFT) & proto::RESP_MISS_MASK);
    proto::begin_response(&mut resp, hdr.seq, req.layer, req.b, resp_flags, 0, t_compute_us, 0, N_EMBD, elem, req.t1, t2);
    resp.resize(proto::RESP_DATA_OFF + n * elem as usize);
    if f32_out {
        exec.read_f32_at(0, b, resp.view_mut::<f32>(proto::RESP_DATA_OFF, n))?;
    } else {
        exec.read_f16_at(0, b, resp.view_mut::<u16>(proto::RESP_DATA_OFF, n))?;
    }
    let ev_t_d2h = if ev_on { super::evtrace::now() } else { f64::NAN };
    // PAGED bits: the pass's own plus the early-page hook's at arrival (see
    // `serve_connection`).
    let mut paged = timing.paged;
    if req.flags & proto::REQ_FLAG_PIN != 0 {
        or_words(&mut paged, &shard.take_early_paged(hdr.seq));
    }
    let pin = if req.flags & proto::REQ_FLAG_PIN != 0 { shard.pin_report(req.layer, req.sel, req.b <= proto::PIN_DECODE_MAX_ROWS) } else { None };
    match pin {
        Some((map, [epoch, pinned, budget])) => {
            proto::append_residency(&mut resp, &map);
            proto::append_pin(&mut resp, epoch, pinned, budget, &paged);
        }
        None if req.flags & proto::REQ_FLAG_RESID != 0 => {
            proto::append_residency(&mut resp, &shard.residency_words(req.layer));
        }
        None => {}
    }
    proto::patch_len(&mut resp);
    let t_ready = Instant::now();
    let t_server_us = (t_ready - t_done).as_micros() as u32;
    if let Some(g) = ev_guard.as_ref() {
        let pd1 = shard.layer_page_detail(req.layer);
        let d = |i: usize| pd1[i].saturating_sub(ev_pd0[i]) as f64;
        let mut sel: Vec<i32> = req.sel.iter().copied().filter(|&e| e >= 0).collect();
        let n_sel = sel.len();
        sel.sort_unstable();
        sel.dedup();
        let under = if g.0 == u64::MAX { f64::NAN } else { g.0 as f64 };
        let pf = ev_pin_fields(ev_pin0, shard.pin_counters(), req.release.len(), &paged);
        let sf = ev_stage_fields(ev_sc0, shard.stage_counters());
        super::evtrace::emit_named(&super::evtrace_kinds::B2_REQ, &[
            ("pin_on", pf[0]), ("pin_pinned", pf[1]), ("pin_budget", pf[2]), ("pin_epoch", pf[3]),
            ("pin_release_words", pf[4]), ("pin_new", pf[5]), ("pin_denied", pf[6]), ("pin_drops_no_victim", pf[7]),
            ("pin_evictions", pf[8]), ("n_paged", pf[9]),
            ("stage_claims", sf[0]), ("stage_hits", sf[1]), ("stage_spills", sf[2]),
            ("seq", f64::from(hdr.seq)), ("layer", f64::from(req.layer)), ("b", f64::from(req.b)), ("flags", f64::from(req.flags)),
            ("merged", 0.0), ("served_under", under),
            ("t_hdr", super::evtrace::inst_to_raw(t_first)), ("t_frame", t2 as f64), ("t_dequeue", ev_t_dequeue),
            ("t_run_start", ev_t_run0), ("t_run_end", ev_t_run1), ("t_d2h_end", ev_t_d2h), ("t_ready", super::evtrace::inst_to_raw(t_ready)),
            ("n_sel", n_sel as f64), ("n_distinct", sel.len() as f64),
            ("n_hint_admit", req.hint_admit.len() as f64), ("n_prefetch_words", req.prefetch.len() as f64),
            ("n_miss", f64::from(n_miss_req)), ("page_us", f64::from(t_page_us)), ("compute_us", f64::from(t_compute_us)),
            ("server_us", f64::from(t_server_us)),
            ("d_misses", d(0)), ("d_read_ns", d(1)), ("d_h2d_ns", d(2)), ("d_repack_gpu_ns", d(3)), ("d_pread_ns", d(4)),
            ("d_repack_cpu_ns", d(5)), ("d_prefetch_wait_ns", shard.prefetch_wait_ns.saturating_sub(ev_pw0) as f64),
            ("path_decode", f64::from(u8::from(timing.path_decode))), ("two_pass", f64::from(u8::from(timing.two_pass))),
            ("n_work_items", f64::from(timing.n_work_items)), ("n_missing", f64::from(timing.n_missing)),
            ("exec_h2d_us", timing.h2d.as_secs_f64() * 1e6), ("exec_gpu_us", timing.gpu.as_secs_f64() * 1e6),
        ]);
    }
    resp.as_bytes_mut()[proto::HDR_LEN + 20..proto::HDR_LEN + 24].copy_from_slice(&t_server_us.to_le_bytes());
    resp.as_bytes_mut()[proto::RESP_PAGE_OFF..proto::RESP_PAGE_OFF + 4].copy_from_slice(&t_page_us.to_le_bytes());
    resp.as_bytes_mut()[proto::RESP_MISSN_OFF..proto::RESP_MISSN_OFF + 4].copy_from_slice(&n_miss_req.to_le_bytes());
    let rec = RequestRecord {
        seq: hdr.seq,
        layer: req.layer,
        b: req.b,
        bytes_in: buf.len(),
        bytes_out: resp.len(),
        read_us: (t_done - t_first).as_micros() as u32,
        queue_us: (t_start - t_done).as_micros() as u32,
        h2d_us: timing.h2d.as_micros() as u32,
        gpu_us: timing.gpu.as_micros() as u32,
        d2h_us: (t_ready - t_d2h0).as_micros() as u32,
        write_us: 0,
        path_decode: timing.path_decode,
        t_ready,
        t1: req.t1,
        t2,
    };
    Ok((rec, resp))
}

/// Drain the writer's `(seq, written_at)` stamps into `write_us` of the records
/// still held; stamps for records already dropped are discarded.
fn apply_written(records: &mut Vec<RequestRecord>, rx_written: &mpsc::Receiver<(u32, Instant)>) {
    let mut by_seq: std::collections::HashMap<u32, usize> =
        records.iter().enumerate().map(|(i, r)| (r.seq, i)).collect();
    while let Ok((wseq, t_w)) = rx_written.try_recv() {
        if let Some(i) = by_seq.remove(&wseq) {
            records[i].write_us = (t_w - records[i].t_ready).as_micros().min(u32::MAX as u128) as u32;
        }
    }
}

// ---------------------------------------------------------------------------
// Hub-side client
// ---------------------------------------------------------------------------

/// Handle of a request in flight. Responses arrive in submission order.
/// Box-1's own two thread handoffs per remote call, which `link_us` cannot see.
///
/// A request crosses caller -> writer thread -> wire -> reader thread -> caller.
/// `rtt_us = t_recv - ticket.t_submit` INCLUDES the submit->writer hop and
/// EXCLUDES the reader->caller hop, so neither is attributable from the
/// `link_us = 205us + bytes/785MBps` regression alone. Worth measuring because a
/// futex wake is 5-30 us against a MEASURED 16.6 us raw TCP round trip on this
/// link (64B echo, same socket options), and 80 calls/token makes each hop
/// ~1 ms/token. Plain relaxed atomics: two adds on a path that already does a
/// syscall.
pub static HOP_SUBMIT_TO_WRITE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// True reader->caller wakeup: only counted when the caller was ALREADY blocked
/// in `recv()` when the frame landed.
pub static HOP_WAIT_WAKE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
/// How EARLY a reply arrived, when the caller was still busy. This is slack, not
/// cost -- it means the remote leg was hidden behind local work for that call.
pub static HOP_SLACK_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Calls where the caller was already blocked, i.e. the link WAS exposed.
pub static HOP_N_BLOCKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static HOP_N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Drain as `(submit_to_write_us, wake_us, slack_us, n_blocked, n)`.
///
/// `wake_us` averages over BLOCKED calls only and `slack_us` over early ones, so
/// neither is diluted by the other. The first version of this averaged one
/// number over everything and reported 2715 us -- which was not a handoff at
/// all, but replies sitting in the channel while the caller was still doing GPU
/// work between `wait(laneA)` and `wait(laneB)`. Slack read as cost.
pub fn take_hop_stats() -> (f64, f64, f64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let n = HOP_N.swap(0, Relaxed);
    let nb = HOP_N_BLOCKED.swap(0, Relaxed);
    let a = HOP_SUBMIT_TO_WRITE_NS.swap(0, Relaxed);
    let w = HOP_WAIT_WAKE_NS.swap(0, Relaxed);
    let sl = HOP_SLACK_NS.swap(0, Relaxed);
    if n == 0 {
        return (0.0, 0.0, 0.0, 0, 0);
    }
    let early = n.saturating_sub(nb).max(1);
    (
        a as f64 / n as f64 / 1000.0,
        w as f64 / nb.max(1) as f64 / 1000.0,
        sl as f64 / early as f64 / 1000.0,
        nb,
        n,
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Ticket {
    pub seq: u32,
    pub layer: u32,
    pub b: u32,
    pub bytes_out: usize,
    pub t_submit: Instant,
    /// The flags as SENT (incl. HINTS / PREFETCH / OOO / DECODE), and how many
    /// residency hints and prefetch words rode on the frame (`evtrace`).
    pub flags: u32,
    pub n_hints: u32,
    pub n_pf_words: u32,
    /// Pin mode (`REQ_FLAG_PIN`): the sent picks the mirror HELD at submit
    /// (bit e = expert e of `layer`), for the surprise check on the reply.
    pub held: [u32; proto::RESID_WORDS],
    pub n_held: u32,
}

/// One layer's remote partial sums: `b × N_EMBD` elements of f16 (default) or
/// f32 (`REQ_FLAG_RESP_F32`), row `t` = token `t` of the request; rows without
/// a remote pick are zero.
pub struct RemotePartial {
    pub layer: u32,
    pub b: u32,
    pub is_f32: bool,
    /// Bit i = sel slot i MISSED on box 2 (decode only). See
    /// `proto::RESP_MISS_SHIFT`. 0 for prefill batches, which do not report.
    pub miss_mask: u32,
    pub rtt_us: u32,
    pub t_remote_compute_us: u32,
    /// Box-2 paging for this request (v2+): microseconds on its NVMe, and how
    /// many experts missed. Drives the `remote.pager` perfetto lane.
    pub t_remote_page_us: u32,
    pub n_remote_miss: u32,
    pub t_remote_server_us: u32,
    pub bytes_in: usize,
    pub bytes_out: usize,
    /// This exchange's clock quadruple (`None` if a peer did not stamp).
    pub clock: Option<ClockSample>,
    /// Pin mode: box 2's `(epoch, pinned, budget)` from the reply's pin block.
    pub pin: Option<(u32, u32, u32)>,
    /// Pin mode: distinct sent picks the mirror held at submit, how many of
    /// them box 2 paged anyway (SURPRISES, `b2_mirror::check_surprises`), and
    /// how many experts the pass paged.
    pub n_held: u32,
    pub n_surprise: u32,
    pub n_paged: u32,
    frame: AlignedBuf,
}

impl RemotePartial {
    pub fn f16(&self) -> &[u16] {
        assert!(!self.is_f32, "partial is f32");
        self.frame.view::<u16>(proto::RESP_DATA_OFF, self.b as usize * N_EMBD as usize)
    }
    pub fn f32(&self) -> &[f32] {
        assert!(self.is_f32, "partial is f16");
        self.frame.view::<f32>(proto::RESP_DATA_OFF, self.b as usize * N_EMBD as usize)
    }
    /// The partial's payload bytes only. A `RESP_FLAG_RESID` residency map
    /// may follow them in the frame.
    pub fn bytes(&self) -> &[u8] {
        let elem = if self.is_f32 { 4 } else { 2 };
        let n = self.b as usize * N_EMBD as usize * elem;
        &self.frame.as_bytes()[proto::RESP_DATA_OFF..proto::RESP_DATA_OFF + n]
    }
    /// Link time = round trip minus the daemon's own frame→response time.
    pub fn link_us(&self) -> u32 {
        self.rtt_us.saturating_sub(self.t_remote_server_us)
    }

    /// Fold this request into the per-B link statistics.
    ///
    /// The aggregate `remote_link_us` in `TokenTiming` cannot answer "how much
    /// of the link cost is LATENCY and how much is BANDWIDTH", and worse, it is
    /// not even a valid link time: it subtracts a SUM of server times taken
    /// across CONCURRENT requests from a single exposed wait, so it routinely
    /// clamps to zero (observed srv 28171 us > rtt 21031 us). This is
    /// per-request, where the subtraction is sound, and it keeps the payload
    /// size alongside -- so link_us regressed on bytes gives latency as the
    /// intercept and 1/bandwidth as the slope.
    pub fn record_link_stats(&self) {
        link_stats::record(
            self.b,
            self.link_us() as u64,
            (self.bytes_in + self.bytes_out) as u64,
            self.t_remote_page_us as u64,
            self.n_remote_miss as u64,
        );
    }
}

/// Per-batch-size link statistics, for splitting link cost into latency and
/// bandwidth. Indexed by `b` (clamped); b=1 is decode, b=3 a verify lane.
pub mod link_stats {
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
    pub const MAX_B: usize = 9;
    static US: [AtomicU64; MAX_B] = [const { AtomicU64::new(0) }; MAX_B];
    static BYTES: [AtomicU64; MAX_B] = [const { AtomicU64::new(0) }; MAX_B];
    static N: [AtomicU64; MAX_B] = [const { AtomicU64::new(0) }; MAX_B];
    // Box 2's OWN paging, which box 1's pager counters cannot see: a run can
    // read `decode_hit 1.0000` on box 1 while box 2 misses on every request.
    // MEASURED cold vs warm: server time 255 ms -> 28 ms, a 9x swing that is
    // invisible without these.
    static PAGE_US: AtomicU64 = AtomicU64::new(0);
    static MISS: AtomicU64 = AtomicU64::new(0);
    // Split by phase: b == 1 is a decode token, anything wider is a prefill
    // chunk or a verify lane. Prefill forcing box 2's STATIC encoder share
    // back through the same LRU that decode fills is a suspected source of
    // decode misses; this is what tells the two apart.
    static PAGE_US_B1: AtomicU64 = AtomicU64::new(0);
    static MISS_B1: AtomicU64 = AtomicU64::new(0);

    pub fn record(b: u32, us: u64, bytes: u64, page_us: u64, miss: u64) {
        let i = (b as usize).min(MAX_B - 1);
        US[i].fetch_add(us, Relaxed);
        BYTES[i].fetch_add(bytes, Relaxed);
        N[i].fetch_add(1, Relaxed);
        PAGE_US.fetch_add(page_us, Relaxed);
        MISS.fetch_add(miss, Relaxed);
        if b == 1 {
            PAGE_US_B1.fetch_add(page_us, Relaxed);
            MISS_B1.fetch_add(miss, Relaxed);
        }
    }

    /// (box-2 page microseconds, box-2 expert misses) since the last call.
    pub fn take_paging() -> (u64, u64) {
        (PAGE_US.swap(0, Relaxed), MISS.swap(0, Relaxed))
    }

    /// Decode-only (b == 1) share of the above, drained together with it.
    pub fn take_paging_decode() -> (u64, u64) {
        (PAGE_US_B1.swap(0, Relaxed), MISS_B1.swap(0, Relaxed))
    }

    /// (b, calls, mean link us, mean bytes) for every b that saw traffic.
    pub fn take() -> Vec<(u32, u64, f64, f64)> {
        (0..MAX_B)
            .filter_map(|i| {
                let n = N[i].swap(0, Relaxed);
                let (us, by) = (US[i].swap(0, Relaxed), BYTES[i].swap(0, Relaxed));
                (n > 0).then(|| (i as u32, n, us as f64 / n as f64, by as f64 / n as f64))
            })
            .collect()
    }
}

enum ClientInbound {
    Resp { buf: AlignedBuf, t_recv: Instant, t4: u64 },
    Err(String),
}

pub struct RemoteExpertClient {
    /// Redial material. The link is the ONLY path to half the experts, so losing
    /// it used to end the process's usefulness: the writer thread exits on a
    /// broken pipe, `tx_req` closes, and every later submit fails forever. One
    /// box-2 error became a permanent outage needing a manual restart.
    addr: String,
    opts: SocketOptions,
    /// The `SO_BUSY_POLL` window currently on `stream`, so the per-phase switch
    /// (`HetEngine::remote_set_phase_busy_poll`) is a no-op when unchanged.
    busy_poll_now: u32,
    /// The hub's phase (`HetEngine::remote_set_phase_busy_poll`): requests
    /// carry `REQ_FLAG_DECODE` while true.
    decode_phase: bool,
    /// Set when the link is known broken. The request that discovers it still
    /// fails -- its in-flight tickets can never be answered -- but the NEXT
    /// request redials instead of inheriting the corpse.
    dead: bool,
    info: ShardInfo,
    stream: TcpStream,
    tx_req: Option<mpsc::SyncSender<(AlignedBuf, u64)>>,
    rx_resp: mpsc::Receiver<ClientInbound>,
    /// Reply frames taken off `rx_resp` (by `ready` or by a `wait` for another
    /// ticket) but not yet consumed, keyed by `seq`. Replies may arrive out of
    /// order (`REQ_FLAG_OOO`: box 2 answers a parked request after the one
    /// queued behind it), so every consumer matches by seq.
    stash: Vec<(u32, AlignedBuf, Instant, u64)>,
    rx_req_recycle: mpsc::Receiver<AlignedBuf>,
    tx_resp_recycle: mpsc::Sender<AlignedBuf>,
    next_seq: u32,
    in_flight: std::collections::VecDeque<Ticket>,
    sel_scratch: Vec<i32>,
    ew_scratch: Vec<f32>,
    clock: ClockSync,
    writer: Option<std::thread::JoinHandle<()>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl RemoteExpertClient {
    /// Connect, apply the transport recipe, read the daemon's HELLO.
    /// Redial if the link is known broken. Call at the START of a request, when
    /// nothing is in flight — never mid-request, since the old socket's in-flight
    /// tickets are unanswerable and a fresh HELLO may report different ownership.
    ///
    /// Rebuilding the whole client is deliberate: a new connection means a new
    /// HELLO, so ownership, geometry and the clock pair must all be re-read
    /// rather than carried over.
    pub fn ensure_connected(&mut self) -> eyre::Result<()> {
        if !self.dead {
            return Ok(());
        }
        let fresh = Self::connect(&self.addr, &self.opts.clone())?;
        // The hub's phase survives the redial: a fresh client starts "not
        // decoding", which would flag decode requests as prefill until the next
        // decode driver sets it (box 2's mode-aware eviction reads the flag).
        let decode_phase = self.decode_phase;
        // Dropping the old value closes its channel and joins its threads.
        *self = fresh;
        self.decode_phase = decode_phase;
        eprintln!("remote_experts: reconnected to {}", self.addr);
        Ok(())
    }

    /// Is the link known broken? (Next request will redial.)
    pub fn is_dead(&self) -> bool {
        self.dead
    }

    /// Record the hub's phase; requests carry `REQ_FLAG_DECODE` while decoding.
    pub fn set_decode_phase(&mut self, decode: bool) {
        self.decode_phase = decode;
    }

    /// Change the socket's `SO_BUSY_POLL` window (microseconds) in place; a
    /// plain `setsockopt`, safe from any thread while the reader spins. Returns
    /// whether the kernel accepted it. Above `net.core.busy_read` it is refused
    /// for an unprivileged process: logged ONCE, and the window stays as it was.
    /// Why the window is phase-dependent: `HetEngine::remote_set_phase_busy_poll`.
    pub fn set_busy_poll_us(&mut self, us: u32) -> bool {
        if us == self.busy_poll_now {
            return true;
        }
        if set_opt_i32(&self.stream, SOL_SOCKET, SO_BUSY_POLL, us as i32) {
            self.busy_poll_now = us;
            true
        } else {
            static WARNED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                eprintln!(
                    "remote_experts: SO_BUSY_POLL={us} refused (net.core.busy_read is the cap for an unprivileged process; \
                     see scripts/link_latency_step.sh 4s) -- keeping {}",
                    self.busy_poll_now
                );
            }
            false
        }
    }

    pub fn connect(addr: &str, opts: &SocketOptions) -> eyre::Result<Self> {
        let stream = TcpStream::connect(addr)?;
        apply_socket_options(&stream, opts)?;
        let mut rd = stream.try_clone()?;
        let mut hello = AlignedBuf::with_capacity(8192);
        let h = proto::read_frame(&mut rd, &mut hello, 1 << 20)?;
        if h.kind != proto::KIND_HELLO {
            return Err(eyre!("expected HELLO, got kind {}", h.kind));
        }
        let info = proto::decode_hello(&hello.as_bytes()[proto::HDR_LEN..])?;
        if info.n_expert != N_EXPERT || info.n_embd != N_EMBD || info.n_used != N_EXPERT_USED as u32
            || info.xq_bytes_per_token != XQ_BYTES_PER_TOKEN as u32
        {
            return Err(eyre!(
                "remote geometry mismatch: n_expert {} n_embd {} n_used {} xq_bpt {} (ours {N_EXPERT}/{N_EMBD}/{N_EXPERT_USED}/{XQ_BYTES_PER_TOKEN})",
                info.n_expert, info.n_embd, info.n_used, info.xq_bytes_per_token
            ));
        }
        let mut wr = stream.try_clone()?;
        let (tx_req, rx_req) = mpsc::sync_channel::<(AlignedBuf, u64)>(16);
        let (tx_req_recycle, rx_req_recycle) = mpsc::channel::<AlignedBuf>();
        let (tx_resp, rx_resp) = mpsc::sync_channel::<ClientInbound>(16);
        let (tx_resp_recycle, rx_resp_recycle) = mpsc::channel::<AlignedBuf>();
        let writer = std::thread::Builder::new().name("rexp-writer".into()).spawn(move || {
            for (mut buf, t_submit_raw) in rx_req {
                // t1 immediately before write(), so frame encoding is outside the sample.
                let t1 = monotonic_raw_ns();
                if buf.len() >= proto::REQ_T1_OFF + 8 {
                    proto::patch_u64(&mut buf, proto::REQ_T1_OFF, t1);
                }
                // Everything between `submit` returning the buffer and this
                // instant is channel + scheduler, not work.
                HOP_SUBMIT_TO_WRITE_NS.fetch_add(
                    t1.saturating_sub(t_submit_raw),
                    std::sync::atomic::Ordering::Relaxed,
                );
                if let Err(e) = wr.write_all(buf.as_bytes()) {
                    eprintln!("remote_experts: write error: {e}");
                    return;
                }
                let _ = tx_req_recycle.send(buf);
            }
        })?;
        // + the optional residency map (`REQ_FLAG_RESID`) and pin block
        // (`REQ_FLAG_PIN`) behind the partial.
        let max_resp = proto::RESP_DATA_OFF + info.max_batch as usize * N_EMBD as usize * 4 + proto::RESID_WORDS * 4 + proto::PIN_WORDS * 4;
        let sock_opts = opts.clone();
        let reader = std::thread::Builder::new().name("rexp-reader".into()).spawn(move || {
            loop {
                let mut buf = rx_resp_recycle.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(max_resp));
                match proto::read_frame(&mut rd, &mut buf, max_resp) {
                    Ok(_) => {}
                    Err(e) => {
                        let _ = tx_resp.send(ClientInbound::Err(format!("{e}")));
                        return;
                    }
                }
                // t4: client receive stamp, the instant the reply is whole.
                let t4 = monotonic_raw_ns();
                quickack(&rd, &sock_opts);
                if tx_resp.send(ClientInbound::Resp { buf, t_recv: Instant::now(), t4 }).is_err() {
                    return;
                }
            }
        })?;
        let nu = N_EXPERT_USED;
        // Window 32 (~one decode token of layers). MEASURED sweep, quiet boxes
        // (docs/v41/REMOTE_EXPERTS.md §5.5): a trailing median lags by ~half a
        // window, and at the boxes' 65 ppm relative clock rate that lag IS the
        // error floor — |err| p50 is 1.1 / 1.6 / 3.1 / 12.3 / 49 µs at window
        // 16 / 32 / 64 / 256 / 1024. 32 keeps the median error at 1.6 µs while
        // still rejecting a queued outlier.
        let clock = ClockSync::new(info.clock, 32, 200_000);
        // A new connection has no pins and no maps: the mirror starts over
        // (box 2 dropped the old connection's pins with it).
        super::b2_mirror::on_connect();
        Ok(Self {
            addr: addr.to_string(),
            opts: opts.clone(),
            dead: false,
            busy_poll_now: opts.busy_poll_us,
            decode_phase: false,
            sel_scratch: vec![NO_PICK; info.max_batch as usize * nu],
            ew_scratch: vec![0.0; info.max_batch as usize * nu],
            clock,
            info,
            stream,
            tx_req: Some(tx_req),
            rx_resp,
            rx_req_recycle,
            tx_resp_recycle,
            next_seq: 1,
            in_flight: Default::default(),
            stash: Vec::new(),
            writer: Some(writer),
            reader: Some(reader),
        })
    }

    pub fn info(&self) -> &ShardInfo {
        &self.info
    }
    pub fn owns(&self, layer: u32, e: i32) -> bool {
        e >= 0 && self.info.owns(layer, e as u32)
    }
    pub fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// Rolling clock-offset estimate over this link, fed by every request
    /// (`ClockSync::offset_ns` / `delay_ns` / `samples` / `perfetto_shift_ns`).
    pub fn clock(&self) -> &ClockSync {
        &self.clock
    }

    /// ns to ADD to a box-1 CLOCK_MONOTONIC_RAW stamp to get the daemon's;
    /// `None` until the first response lands.
    pub fn clock_offset_ns(&self) -> Option<i64> {
        self.clock.offset_ns()
    }

    /// Enqueue one layer's request. `xq` = `b` Q8_K rows (the hub's
    /// `d_xq_q8k` bytes), `sel`/`ew` = the router's `b × N_EXPERT_USED` picks
    /// and weights. Picks the remote does not own are masked out here; if no
    /// token has a remote pick nothing is sent and `None` is returned. Returns
    /// as soon as the frame is handed to the writer thread.
    /// Submit WITHOUT the advertised-bitmap mask — the caller has already decided the
    /// partition and the daemon accepts out-of-set experts.
    ///
    /// This exists because the two-box design intends exactly that: `enable_paging`
    /// sets the daemon's `l.owned` all-true "so the executor accepts an expert outside
    /// the advertised set and `ensure_layer` pages it", and keeps the ADVERTISED bitmap
    /// static only so PREFILL's exclusion mask stays correct. But `submit_flags` masked
    /// by that same advertised bitmap, so the hub's decode-side decision never reached
    /// the wire.
    ///
    /// MEASURED 2026-09-14 over a 10-token generation before this fix: 1,858 live picks
    /// replaced by NO_PICK (~186 of 240 routed experts per token, 77%) and 119 layers
    /// where every pick was masked — `Ok(None)`, so the hub did not even wait. Those
    /// experts were computed by NOBODY. `verify_routing_exactly_once` could not see it:
    /// it validates the hub's INTENT (the remap encoding), not the outcome.
    pub fn submit_unmasked(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool) -> eyre::Result<Option<Ticket>> {
        let flags = if resp_f32 { proto::REQ_FLAG_RESP_F32 } else { 0 };
        self.submit_inner(layer, b, xq, sel, ew, flags, false, PinNote::default())
    }

    /// As [`Self::submit_unmasked`] with explicit `proto::REQ_FLAG_*` bits.
    pub fn submit_unmasked_flags(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32) -> eyre::Result<Option<Ticket>> {
        self.submit_inner(layer, b, xq, sel, ew, flags, false, PinNote::default())
    }

    /// `submit` or `submit_unmasked`, chosen by whether box 1 is computing any
    /// of this layer's experts.
    ///
    /// MASKED (`unmasked = false`) filters the picks down to what box 2
    /// ADVERTISED it owns and leaves the rest for box 1 — correct whenever box 1
    /// pages its own share. Under the small-B offload box 1 pages NOTHING, so a
    /// masked submit leaves every unadvertised pick computed by nobody. That is
    /// silent: `verify_routing_exactly_once` validates the hub's own `owns_eff`,
    /// not box 2's advertised table. It halved DSpark's acceptance
    /// (E 2.12 -> 1.10) before it was caught.
    ///
    /// Getting this backwards is equally silent in the other direction: an
    /// unmasked submit while box 1 still computes its share DOUBLE-COUNTS every
    /// expert both devices claim.
    ///
    /// `pin`: what the pin ledger should learn from this request (`PinNote`).
    #[allow(clippy::too_many_arguments)]
    pub fn submit_dispatch(&mut self, unmasked: bool, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool, partner: bool, pin: PinNote<'_>) -> eyre::Result<Option<Ticket>> {
        let extra = if partner { proto::REQ_FLAG_PARTNER } else { 0 };
        let resid = if super::b2_mirror::wanted() {
            // These picks will be resident on box 2 by the time the other lane's
            // request for this layer is served: overlay them until the reply.
            super::b2_mirror::note_submitted(layer, sel);
            proto::REQ_FLAG_RESID | super::b2_mirror::pin_request_flag()
        } else {
            0
        };
        let f = if resp_f32 { proto::REQ_FLAG_RESP_F32 } else { 0 } | extra | resid;
        self.submit_inner(layer, b, xq, sel, ew, f, !unmasked, pin)
    }

    pub fn submit(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool) -> eyre::Result<Option<Ticket>> {
        self.submit_flags(layer, b, xq, sel, ew, if resp_f32 { proto::REQ_FLAG_RESP_F32 } else { 0 })
    }

    /// As [`Self::submit`] with explicit `proto::REQ_FLAG_*` bits.
    pub fn submit_flags(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32) -> eyre::Result<Option<Ticket>> {
        self.submit_inner(layer, b, xq, sel, ew, flags, true, PinNote::default())
    }

    /// [`Self::submit_flags`] without the advertised-ownership mask — i.e. what
    /// the hub does under T2 catch-all, where "not resident on box 1" is the
    /// routing rule and box 2 pages anything it is handed. Needed by the bench to
    /// exercise the miss path at all: masked submits can only ever request
    /// resident experts, so they never fault.
    pub fn submit_flags_unmasked(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32) -> eyre::Result<Option<Ticket>> {
        self.submit_inner(layer, b, xq, sel, ew, flags, false, PinNote::default())
    }

    /// `pin`: the pin ledger's view of the request (`PinNote`).
    #[allow(clippy::too_many_arguments)]
    fn submit_inner(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32, mask: bool, pin: PinNote<'_>) -> eyre::Result<Option<Ticket>> {
        let nu = N_EXPERT_USED;
        if b == 0 || b > self.info.max_batch as usize {
            return Err(eyre!("remote submit: b={b} outside 1..={}", self.info.max_batch));
        }
        // A multi-token request is a DSpark verify batch; take the by-expert
        // chain, which reads each expert's weights ONCE for the whole batch
        // instead of once per token. See `remote_batched_multi`.
        let flags = if b > 1 && remote_batched_multi() {
            flags | proto::REQ_FLAG_BATCHED
        } else {
            flags
        };
        if xq.len() != b * XQ_BYTES_PER_TOKEN || sel.len() != b * nu || ew.len() != b * nu {
            return Err(eyre!("remote submit: payload sizes do not match b={b}"));
        }
        let mut any = false;
        // `V41_MASK_DBG=1`: count LIVE picks this mask drops. The mask is the STATIC
        // HELLO bitmap (`self.owns` -> `asg.bitsets()`), NOT the hub's residency-derived
        // `owns_remote`. Under `V41_T2_CATCHALL=1` the hub marks a pick remote because it
        // does not hold it; if box 2 does not statically own it either, it is silently
        // replaced by NO_PICK here and computed by NOBODY —
        // `verify_routing_exactly_once` cannot see it because it validates the HUB's view.
        let mut masked_live = 0usize;
        for i in 0..b * nu {
            if !mask || self.owns(layer, sel[i]) {
                self.sel_scratch[i] = sel[i];
                self.ew_scratch[i] = ew[i];
                any = true;
            } else {
                if sel[i] >= 0 && sel[i] < N_EXPERT as i32 {
                    masked_live += 1;
                }
                self.sel_scratch[i] = NO_PICK;
                self.ew_scratch[i] = 0.0;
            }
        }
        if masked_live > 0
            && std::env::var("V41_MASK_DBG").as_deref() == Ok("1")
        {
            tracing::warn!(
                layer, masked_live, live_sent = b * nu - masked_live,
                "remote submit MASKED live picks (static HELLO bitmap, not owns_remote)"
            );
        }
        if !any {
            if std::env::var("V41_MASK_DBG").as_deref() == Ok("1") {
                tracing::warn!(layer, "remote submit DROPPED ENTIRE LAYER (no pick survived the mask)");
            }
            return Ok(None);
        }
        let mut buf = self.rx_req_recycle.try_recv().unwrap_or_else(|_| {
            AlignedBuf::with_capacity(proto::HDR_LEN + proto::REQ_FIXED + self.info.max_batch as usize * (XQ_BYTES_PER_TOKEN + 8 * nu))
        });
        let seq = self.next_seq;
        self.next_seq = self.next_seq.wrapping_add(1);
        // Residency hints ride on the next frame (box 1 = L1, box 2 = victim).
        let (ha, he) = if super::expert_pager::b1_prefetch() {
            super::expert_pager::take_residency_hints(64)
        } else {
            (Vec::new(), Vec::new())
        };
        let flags = if ha.is_empty() && he.is_empty() { flags } else { flags | proto::REQ_FLAG_HINTS };
        // Pin mode: the queued RELEASE words, which only go out once box 2 has
        // shown it understands them; what the mirror HOLDS among the picks
        // actually sent (the surprise check on the reply).
        let pin_req = flags & proto::REQ_FLAG_PIN != 0;
        let rel = if pin_req { super::b2_mirror::take_release_words(128) } else { Vec::new() };
        let mut pf = take_prefetch_words(128);
        // RESTORE words fill the room the admission / look-ahead words leave,
        // a few per request, and only once every release is on the wire: box
        // 2 applies a request's releases before its prefetch grants, so a
        // restore can never reach it before the release it undoes.
        if pin_req && pf.len() < 128 && super::b2_mirror::releases_queued() == 0 {
            pf.extend(take_restore_words((128 - pf.len()).min(super::b2_mirror::pin_restore_per_request())));
        }
        let flags = if pf.is_empty() { flags } else { flags | proto::REQ_FLAG_PREFETCH };
        let (held, n_held) = if pin_req {
            let decode_shaped = b as u32 <= proto::PIN_DECODE_MAX_ROWS && pin.decode.unwrap_or(true);
            super::b2_mirror::pin_note_submit(layer, &self.sel_scratch[..b * nu], pin.wants, decode_shaped)
        } else {
            ([0u32; proto::RESID_WORDS], 0)
        };
        let flags = if rel.is_empty() { flags } else { flags | proto::REQ_FLAG_RELEASE };
        // `wait` matches by seq, so any reply order is fine from here.
        let flags = flags | proto::REQ_FLAG_OOO;
        let flags = if self.decode_phase { flags | proto::REQ_FLAG_DECODE } else { flags };
        proto::encode_request(
            &mut buf, seq, layer, b as u32, flags, nu as u32, XQ_BYTES_PER_TOKEN as u32, xq,
            &self.sel_scratch[..b * nu], &self.ew_scratch[..b * nu], (&ha, &he), &pf, &rel,
        );
        let ticket = Ticket {
            seq, layer, b: b as u32, bytes_out: buf.len(), t_submit: Instant::now(),
            flags, n_hints: (ha.len() + he.len()) as u32, n_pf_words: pf.len() as u32,
            held, n_held,
        };
        let sent = match self.tx_req.as_ref() {
            Some(tx) => tx
                .send((buf, monotonic_raw_ns()))
                .map_err(|_| eyre!("writer thread gone")),
            None => Err(eyre!("client closed")),
        };
        if let Err(e) = sent {
            // Do NOT redial here: this request's earlier layers are already in
            // flight on the old socket and can never be reconciled. Fail it, and
            // let the next request start clean.
            self.dead = true;
            return Err(e);
        }
        self.in_flight.push_back(ticket);
        Ok(Some(ticket))
    }

    /// `seq` of the oldest in-flight request. `None` when nothing is in flight.
    pub fn head_seq(&self) -> Option<u32> {
        self.in_flight.front().map(|t| t.seq)
    }

    /// Move every frame the reader has delivered into `stash`. A closed channel
    /// or a socket error is returned so the caller can surface it.
    fn drain_into_stash(&mut self) -> Result<(), String> {
        loop {
            match self.rx_resp.try_recv() {
                Ok(ClientInbound::Resp { buf, t_recv, t4 }) => {
                    let seq = proto::parse_header(buf.as_bytes()).map(|h| h.seq).unwrap_or(u32::MAX);
                    self.stash.push((seq, buf, t_recv, t4));
                }
                Ok(ClientInbound::Err(e)) => return Err(e),
                Err(mpsc::TryRecvError::Empty) => return Ok(()),
                Err(mpsc::TryRecvError::Disconnected) => return Err("reader thread gone".into()),
            }
        }
    }

    /// Non-blocking: has request `seq`'s reply arrived? Takes frames off the
    /// channel into `stash` without consuming them, so `wait` still does all
    /// the accounting. A broken link reports READY so the caller's `wait`
    /// surfaces the error instead of spinning. A scheduling HINT for the
    /// ready-first lane driver only -- correctness never depends on it,
    /// because `wait` blocks for real either way.
    pub fn ready(&mut self, seq: u32) -> bool {
        if self.drain_into_stash().is_err() {
            return true;
        }
        self.stash.iter().any(|f| f.0 == seq)
    }

    /// `ready` for the oldest in-flight request (the FIFO-era API).
    pub fn head_ready(&mut self) -> bool {
        match self.head_seq() {
            Some(s) => self.ready(s),
            None => true,
        }
    }

    /// Block until request `ticket` has answered. Any in-flight ticket may be
    /// waited for, in any order: replies are matched by `seq`.
    pub fn wait(&mut self, ticket: Ticket) -> eyre::Result<RemotePartial> {
        let pos = self.in_flight.iter().position(|t| t.seq == ticket.seq)
            .ok_or_else(|| eyre!("wait: ticket seq {} is not in flight", ticket.seq))?;
        self.in_flight.remove(pos);
        // Stamped BEFORE the blocking recv, so we can tell "we waited for the
        // reply" apart from "the reply waited for us".
        let t_wait_enter = Instant::now();
        let mut found = self.stash.iter().position(|f| f.0 == ticket.seq).map(|i| self.stash.swap_remove(i));
        while found.is_none() {
            match self.rx_resp.recv() {
                Ok(ClientInbound::Resp { buf, t_recv, t4 }) => {
                    let seq = proto::parse_header(buf.as_bytes()).map(|h| h.seq).unwrap_or(u32::MAX);
                    if seq == ticket.seq {
                        found = Some((seq, buf, t_recv, t4));
                    } else {
                        self.stash.push((seq, buf, t_recv, t4));
                    }
                }
                // Both arms mean the socket is gone, not that this reply was bad, so
                // the link must be redialed before the next request rather than
                // inherited: see `ensure_connected`.
                Ok(ClientInbound::Err(e)) => {
                    self.dead = true;
                    return Err(eyre!("remote connection: {e}"));
                }
                Err(_) => {
                    self.dead = true;
                    return Err(eyre!("reader thread gone"));
                }
            }
        }
        let (_, buf, t_recv, t4) = found.expect("loop exits with a frame");
        {
            use std::sync::atomic::Ordering::Relaxed;
            let now = Instant::now();
            if t_recv >= t_wait_enter {
                // We were already parked in recv() when the frame landed:
                // this is a genuine wakeup, and the remote leg was EXPOSED.
                HOP_WAIT_WAKE_NS.fetch_add((now - t_recv).as_nanos() as u64, Relaxed);
                HOP_N_BLOCKED.fetch_add(1, Relaxed);
            } else {
                // The frame was already sitting in the stash/channel: the remote
                // leg finished behind local work and cost us nothing.
                HOP_SLACK_NS.fetch_add((t_wait_enter - t_recv).as_nanos() as u64, Relaxed);
            }
            HOP_N.fetch_add(1, Relaxed);
        }
        let h = proto::parse_header(buf.as_bytes())?;
        if h.kind == proto::KIND_ERROR {
            let (st, msg) = proto::decode_error(&buf);
            return Err(eyre!("remote error (status {st}) on seq {}: {msg}", h.seq));
        }
        if h.kind != proto::KIND_RESPONSE {
            return Err(eyre!("unexpected frame kind {}", h.kind));
        }
        if h.seq != ticket.seq {
            return Err(eyre!("response seq {} != ticket seq {}", h.seq, ticket.seq));
        }
        let m = proto::decode_response_meta(&buf)?;
        if m.status != 0 {
            return Err(eyre!("remote status {} on seq {}", m.status, h.seq));
        }
        if m.layer != ticket.layer || m.b != ticket.b {
            return Err(eyre!("response (L{} B{}) does not match ticket (L{} B{})", m.layer, m.b, ticket.layer, ticket.b));
        }
        // Pin mode: the map is box 2's PINNED set as of `epoch` release words;
        // the reply also says what the pass had to page, which must not
        // include anything the mirror held when this request was sent.
        let pin = proto::response_pin(&buf, &m);
        if let Some(words) = proto::response_residency(&buf, &m) {
            match pin.as_ref() {
                Some(p) => super::b2_mirror::update_pinned(m.layer, words, p.epoch, p.pinned, p.budget),
                None => super::b2_mirror::update(m.layer, words),
            }
        }
        let mut n_surprise = 0;
        if ticket.flags & proto::REQ_FLAG_PIN != 0 {
            super::b2_mirror::pin_reply_seen(pin.is_some());
            if let Some(p) = pin.as_ref() {
                n_surprise = super::b2_mirror::check_surprises(m.layer, h.seq, &ticket.held, &p.paged);
            }
        }
        // NTP quadruple for this exchange. t1 is what the WRITER stamped (echoed
        // back by the daemon), not the submit time, so encoding is excluded.
        let sample = ClockSample { seq: h.seq, layer: m.layer, b: m.b, t1: m.t1, t2: m.t2, t3: m.t3, t4 };
        let valid = m.t1 != 0 && m.t2 != 0 && m.t3 != 0 && t4 != 0;
        if valid {
            self.clock.push(sample);
        }
        let partial = RemotePartial {
            layer: m.layer,
            b: m.b,
            is_f32: m.elem_bytes == 4,
            miss_mask: (m.flags & proto::RESP_MISS_MASK) >> proto::RESP_MISS_SHIFT,
            rtt_us: (t_recv - ticket.t_submit).as_micros().min(u32::MAX as u128) as u32,
            t_remote_compute_us: m.t_compute_us,
            t_remote_page_us: m.t_page_us,
            n_remote_miss: m.n_miss,
            t_remote_server_us: m.t_server_us,
            bytes_in: buf.len(),
            bytes_out: ticket.bytes_out,
            clock: valid.then_some(sample),
            pin: pin.as_ref().map(|p| (p.epoch, p.pinned, p.budget)),
            n_held: ticket.n_held,
            n_surprise,
            n_paged: pin.as_ref().map_or(0, |p| p.paged.iter().map(|w| w.count_ones()).sum()),
            frame: buf,
        };
        // Every partial passes through here, so this is the one place the link
        // statistics can be complete. The perfetto site below is gated on the
        // exporter being attached, and the exporter itself perturbs the run.
        partial.record_link_stats();
        Ok(partial)
    }

    /// Hand a consumed partial's buffer back for reuse.
    pub fn recycle(&self, p: RemotePartial) {
        let _ = self.tx_resp_recycle.send(p.frame);
    }

    /// Consume every outstanding response. A request that errors out mid-layer
    /// leaves tickets in flight; the next request's first `wait` then fails
    /// with "ticket seq X but oldest in flight is Y" and EVERY later request
    /// fails the same way (observed 2026-09-17 after one prompt overflowed the
    /// prefill window stride). Call from the request error path.
    pub fn drain_in_flight(&mut self) -> usize {
        let mut n = 0;
        // Replies already stashed (out-of-order arrivals) answer in-flight
        // tickets too; receiving for them would block forever.
        for (seq, buf, ..) in std::mem::take(&mut self.stash) {
            if let Some(pos) = self.in_flight.iter().position(|t| t.seq == seq) {
                self.in_flight.remove(pos);
                n += 1;
            }
            let _ = self.tx_resp_recycle.send(buf);
        }
        while self.in_flight.pop_front().is_some() {
            match self.rx_resp.recv() {
                Ok(ClientInbound::Resp { buf, .. }) => {
                    let _ = self.tx_resp_recycle.send(buf);
                }
                Ok(ClientInbound::Err(_)) | Err(_) => break,
            }
            n += 1;
        }
        n
    }

    /// Convenience: submit + wait.
    pub fn call(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool) -> eyre::Result<Option<RemotePartial>> {
        match self.submit(layer, b, xq, sel, ew, resp_f32)? {
            Some(t) => Ok(Some(self.wait(t)?)),
            None => Ok(None),
        }
    }
}

impl Drop for RemoteExpertClient {
    fn drop(&mut self) {
        // Close the request channel so the writer exits, then shut the socket
        // so the reader's blocking read returns.
        self.tx_req.take();
        let _ = self.stream.shutdown(std::net::Shutdown::Both);
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

/// f32 → f16 bits with the same rounding as the device cast (`__float2half`,
/// round-to-nearest-even); for checking f16 responses against f32 references.
pub fn f32_to_f16_bits(f: f32) -> u16 {
    weight_contract::f32_to_f16_bits(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Box 2's knob file and env as deployed on 2026-10-01 resolve to what the
    /// daemon ran with before the move to `crate::knobs` (short keys as aliases;
    /// the file over the env: `miss_par=1` beats `V41_B2_MISS_PAR=4`).
    #[test]
    fn box2s_deployed_knob_file_resolves_as_before() {
        let file = crate::knobs::parse_file("merge=1\nmerge_wait_us=400\nmiss_par=1\ncoalesce=0\nmirror_frac=0.70\npark=1\nroute=urgency\n");
        let env = |k: &str| match k {
            "V41_EXPERT_MIRROR_FRAC" => Some("0.70".to_string()),
            "V41_B2_MISS_PAR" => Some("4".to_string()),
            _ => None,
        };
        let mut warned = std::collections::HashSet::new();
        let p = crate::knobs::pass(knobs::ALL, &file, &env, &|_| None, &mut warned);
        assert!(p.warnings.is_empty(), "{:?}", p.warnings);
        assert!(knobs::merge() && knobs::park() && !knobs::coalesce());
        assert_eq!((knobs::merge_wait_us(), knobs::miss_par()), (400, 1));
        assert_eq!(knobs::ROUTE.pick(), 1, "urgency");
        assert!((knobs::MIRROR_FRAC.f64() - 0.70).abs() < 1e-12);
        assert!(knobs::prefill_route_split() && knobs::fast_chain() && knobs::encoder_victims_first());
        assert_eq!(knobs::prefill_budget(), 3500);
        assert!(p.hooks.iter().any(|k| k.name == "V41_EXPERT_MIRROR_FRAC"), "mirror_frac is pushed to v4flash_core");
    }

    #[test]
    fn assignment_grammar() {
        let a = Assignment::parse("L3:0-7,7:0-7").unwrap();
        assert_eq!(a.layers.len(), 2);
        assert_eq!(a.layers[0], (3, (0..8).collect()));
        assert_eq!(a.layers[1], (7, (0..8).collect()));
        assert_eq!(a.n_experts(), 16);
        let b = Assignment::parse("L20-L21").unwrap();
        assert_eq!(b.layers[0].1.len(), N_EXPERT as usize);
        assert_eq!(b.layers[1].0, 21);
        let c = Assignment::parse("all:0-1").unwrap();
        assert_eq!(c.layers.len(), N_LAYER as usize);
        assert_eq!(c.n_experts(), 2 * N_LAYER as usize);
        // duplicates merge
        let d = Assignment::parse("L2:0-3,L2:2-5").unwrap();
        assert_eq!(d.layers[0].1, vec![0, 1, 2, 3, 4, 5]);
        assert!(Assignment::parse("L99").is_err());
        assert!(Assignment::parse(&format!("L0:0-{N_EXPERT}")).is_err());
        assert!(Assignment::parse("").is_err());
        let bits = a.bitsets();
        assert_eq!(bits[3][0], 0xff);
        assert_eq!(bits[4][0], 0);
    }

    #[test]
    fn hello_roundtrip() {
        let a = Assignment::parse("L3:0-7,7:100-131").unwrap();
        let info = ShardInfo {
            n_layer: N_LAYER as u32,
            n_expert: N_EXPERT,
            n_used: N_EXPERT_USED as u32,
            n_embd: N_EMBD,
            xq_bytes_per_token: XQ_BYTES_PER_TOKEN as u32,
            max_batch: 1024,
            decode_max_b: 4,
            n_resident: 40,
            bytes_per_expert: 18_800_000,
            clock: ClockPair { mono_raw_ns: 123_456_789, realtime_ns: 1_700_000_000_000_000_000 },
            owned: a.bitsets(),
        };
        let mut buf = AlignedBuf::with_capacity(4096);
        proto::encode_hello(&mut buf, &info);
        let h = proto::parse_header(buf.as_bytes()).unwrap();
        assert_eq!(h.kind, proto::KIND_HELLO);
        assert_eq!(h.len as usize, buf.len() - proto::HDR_LEN);
        let back = proto::decode_hello(&buf.as_bytes()[proto::HDR_LEN..]).unwrap();
        assert_eq!(back, info);
        assert_eq!(back.clock.mono_raw_ns, 123_456_789, "64-bit clock pair survives the u32 packing");
        assert_eq!(back.clock.realtime_ns, 1_700_000_000_000_000_000);
        assert!(back.owns(7, 100) && back.owns(7, 131) && !back.owns(7, 132) && !back.owns(3, 8));
        assert_eq!(back.owned_ids(3), (0..8).collect::<Vec<_>>());
        assert_eq!(back.owned_count(7), 32);
    }

    #[test]
    fn request_response_roundtrip() {
        let b = 3usize;
        let nu = N_EXPERT_USED;
        let xq: Vec<u8> = (0..b * XQ_BYTES_PER_TOKEN).map(|i| (i * 7 % 251) as u8).collect();
        let sel: Vec<i32> = (0..b * nu).map(|i| if i % 4 == 0 { NO_PICK } else { (i * 13 % 384) as i32 }).collect();
        let ew: Vec<f32> = (0..b * nu).map(|i| i as f32 * 0.125).collect();
        let mut buf = AlignedBuf::with_capacity(1 << 16);
        proto::encode_request(&mut buf, 42, 17, b as u32, proto::REQ_FLAG_RESP_F32, nu as u32, XQ_BYTES_PER_TOKEN as u32, &xq, &sel, &ew, (&[], &[]), &[], &[]);
        proto::patch_u64(&mut buf, proto::REQ_T1_OFF, 111_222_333);
        let h = proto::parse_header(buf.as_bytes()).unwrap();
        assert_eq!((h.kind, h.seq), (proto::KIND_REQUEST, 42));
        let r = proto::decode_request(&buf).unwrap();
        assert_eq!((r.layer, r.b, r.flags), (17, 3, proto::REQ_FLAG_RESP_F32));
        assert_eq!(r.t1, 111_222_333, "t1 survives the round trip at REQ_T1_OFF");
        assert_eq!(r.xq, &xq[..]);
        assert_eq!(r.sel, &sel[..]);
        assert_eq!(r.ew, &ew[..]);
        // Read it back through the stream reader.
        let mut cursor = std::io::Cursor::new(buf.as_bytes().to_vec());
        let mut rb = AlignedBuf::with_capacity(16);
        let h2 = proto::read_frame(&mut cursor, &mut rb, 1 << 20).unwrap();
        assert_eq!(h2, h);
        assert_eq!(rb.as_bytes(), buf.as_bytes());

        let n = b * N_EMBD as usize;
        let mut resp = AlignedBuf::with_capacity(1 << 16);
        proto::begin_response(&mut resp, 42, 17, b as u32, 0, 0, 123, 456, N_EMBD, 2, 111_222_333, 444_555_666);
        resp.resize(proto::RESP_DATA_OFF + n * 2);
        for (i, v) in resp.view_mut::<u16>(proto::RESP_DATA_OFF, n).iter_mut().enumerate() {
            *v = i as u16;
        }
        proto::patch_len(&mut resp);
        proto::patch_u64(&mut resp, proto::RESP_T3_OFF, 777_888_999);
        let m = proto::decode_response_meta(&resp).unwrap();
        assert_eq!((m.layer, m.b, m.t_compute_us, m.t_server_us, m.elem_bytes), (17, 3, 123, 456, 2));
        assert_eq!((m.t1, m.t2, m.t3), (111_222_333, 444_555_666, 777_888_999));
        assert_eq!(resp.view::<u16>(proto::RESP_DATA_OFF, n)[n - 1], (n - 1) as u16);

        let mut err = AlignedBuf::with_capacity(64);
        proto::encode_error(&mut err, 7, 9, "nope");
        assert_eq!(proto::decode_error(&err), (9, "nope".to_string()));
    }

    /// The box-2 prefetch readers' queue: certain jobs first, a speculative job
    /// only within `max_spec` and never while a certain one runs or waits,
    /// promotion of a queued job vs. urgency for a popped one, and `close`
    /// draining everything before `None`.
    #[test]
    fn prefetch_queue_priority_and_reservation() {
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        let job = |e: u32, certain: bool| PfJob { layer: 3, e, set: e as usize, certain, stage: false, own_prefill: false, prefill: false, restore: 0, t_hint: Instant::now() };
        let q = Arc::new(PfQueue::new(1));
        q.push(job(1, false));
        q.push(job(2, false));
        q.push(job(3, true));
        // Certain first.
        let a = q.pop().unwrap();
        assert_eq!((a.e, a.certain), (3, true));
        // A speculative job may not start while a certain one runs.
        let q2 = Arc::clone(&q);
        let h = std::thread::spawn(move || q2.pop().map(|j| j.e));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!h.is_finished(), "speculative job handed out while a certain one runs");
        q.finished(true);
        assert_eq!(h.join().unwrap(), Some(1));
        // max_spec = 1 and one speculative job runs: the next one waits...
        let q3 = Arc::clone(&q);
        let h = std::thread::spawn(move || q3.pop().map(|j| (j.e, j.certain)));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!h.is_finished(), "second speculative job exceeded max_spec");
        // ...but a PROMOTED job is certain and goes at once.
        assert!(q.promote(3, 2), "promote a queued speculative job");
        assert_eq!(h.join().unwrap(), Some((2, true)));
        assert!(!q.promote(3, 2) || q.is_urgent(3, 2), "promote of a popped job marks it urgent");
        assert!(q.is_urgent(3, 2));
        q.clear_urgent(3, 2);
        assert!(!q.is_urgent(3, 2));
        q.finished(true);
        q.finished(false);
        // Already-certain jobs are left alone.
        q.push(job(7, true));
        assert!(!q.promote(3, 7));
        // Close drains remaining jobs, ungated, then None.
        q.push(job(8, false));
        q.push(job(9, false));
        q.close();
        let mut drained: Vec<u32> = std::iter::from_fn(|| q.pop().map(|j| j.e)).collect();
        drained.sort();
        assert_eq!(drained, vec![7, 8, 9]);
        assert!(q.pop().is_none());
    }

    /// The wait decisions under urgency routing, as pure functions.
    #[test]
    fn urgency_wait_decisions() {
        use std::collections::HashSet;
        let pending: HashSet<(u32, u32)> = [(4, 1), (4, 2), (4, 3)].into_iter().collect();
        let none: HashSet<(u32, u32)> = HashSet::new();
        let spec: HashSet<(u32, u32)> = [(4, 2)].into_iter().collect();
        // admit: split (empty skip) waits for any pending wanted key.
        assert!(wanted_in_flight(&[2], 4, &pending, &none));
        // urgency: not for one running speculatively on the primary...
        assert!(!wanted_in_flight(&[2], 4, &pending, &spec));
        // ...but still for any other pending one, and never for non-pending.
        assert!(wanted_in_flight(&[2, 3], 4, &pending, &spec));
        assert!(!wanted_in_flight(&[9], 4, &pending, &spec));
        assert!(!wanted_in_flight(&[1], 5, &pending, &none), "other layer");
        // park: split waits for every pending key, resident or not.
        let resident = |k: &(u32, u32)| *k == (4, 3);
        assert!(park_waits_for((4, 2), &pending, &spec, resident, false));
        assert!(park_waits_for((4, 3), &pending, &spec, resident, false));
        // urgency: not for a running speculative one, nor an already-resident one.
        assert!(!park_waits_for((4, 2), &pending, &spec, resident, true));
        assert!(!park_waits_for((4, 3), &pending, &spec, resident, true));
        assert!(park_waits_for((4, 1), &pending, &spec, resident, true));
        assert!(!park_waits_for((4, 9), &pending, &spec, resident, true));
    }

    /// `PfFinish` releases a recorded speculative key and the reader slot on
    /// drop, and `promote` leaves a running speculative key un-urgent.
    #[test]
    fn prefetch_finish_releases_spec_key() {
        use std::time::Instant;
        let q = PfQueue::with_readers(2, 3);
        q.push(PfJob { layer: 6, e: 1, set: 0, certain: false, stage: false, own_prefill: false, prefill: false, restore: 0, t_hint: Instant::now() });
        let j = q.pop_mode(true).unwrap();
        assert!(q.spec_keys_snapshot().contains(&(6, 1)));
        assert!(!q.promote(6, 1), "a running speculative key must not be promoted/urgent");
        assert!(!q.is_urgent(6, 1));
        assert_eq!(q.counts()[1], 1.0, "one speculative job running");
        {
            let _f = PfFinish { q: &q, certain: j.certain, spec_key: Some((j.layer, j.e)) };
        }
        assert!(q.spec_keys_snapshot().is_empty());
        assert_eq!(q.counts()[1], 0.0, "PfFinish released the reader slot");
        // Urgency cap: max_spec 2 but 3 readers -> 2 may run; with 2 readers -> 1.
        let q2 = PfQueue::with_readers(2, 2);
        for e in 0..3 {
            q2.push(PfJob { layer: 6, e, set: 0, certain: false, stage: false, own_prefill: false, prefill: false, restore: 0, t_hint: Instant::now() });
        }
        let _a = q2.pop_mode(true).unwrap();
        let g = q2.inner.lock().unwrap();
        assert_eq!(PfQueue::spec_cap(&g, true), 1, "urgency must leave one reader for certain jobs");
        assert_eq!(PfQueue::spec_cap(&g, false), 2);
    }

    /// Urgency routing (`pop_mode(true)`): a speculative job reads from the
    /// OTHER drive, so it is handed out while a certain job runs (still capped
    /// at `max_spec`), and its key is recorded until `finish_spec_key`;
    /// split mode's gate is unchanged.
    #[test]
    fn prefetch_queue_urgency_routing() {
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        let job = |e: u32, certain: bool| PfJob { layer: 5, e, set: e as usize, certain, stage: false, own_prefill: false, prefill: false, restore: 0, t_hint: Instant::now() };
        let q = Arc::new(PfQueue::new(1));
        q.push(job(1, true));
        q.push(job(2, false));
        q.push(job(3, false));
        let a = q.pop_mode(true).unwrap();
        assert_eq!((a.e, a.certain), (1, true));
        assert!(q.spec_keys_snapshot().is_empty(), "a certain job is not a speculative key");
        // Certain job running: speculative still handed out under urgency.
        let b = q.pop_mode(true).unwrap();
        assert_eq!((b.e, b.certain), (2, false));
        assert!(q.spec_keys_snapshot().contains(&(5, 2)));
        // max_spec = 1 still caps speculative jobs.
        let q2 = Arc::clone(&q);
        let h = std::thread::spawn(move || q2.pop_mode(true).map(|j| j.e));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!h.is_finished(), "second speculative job exceeded max_spec under urgency");
        // Releasing the first (as PfFinish does) lets the next one go and records it.
        q.finish_spec_key(5, 2);
        q.finished(false);
        assert_eq!(h.join().unwrap(), Some(3));
        let keys = q.spec_keys_snapshot();
        assert!(!keys.contains(&(5, 2)) && keys.contains(&(5, 3)), "{keys:?}");
        q.finish_spec_key(5, 3);
        q.finished(false);
        assert!(q.spec_keys_snapshot().is_empty());
        // Split mode: a speculative job waits for the running certain one and
        // records no key.
        q.push(job(4, false));
        let q3 = Arc::clone(&q);
        let h = std::thread::spawn(move || q3.pop_mode(false).map(|j| j.e));
        std::thread::sleep(Duration::from_millis(50));
        assert!(!h.is_finished(), "split mode handed out a speculative job beside a certain one");
        q.finished(true);
        assert_eq!(h.join().unwrap(), Some(4));
        assert!(q.spec_keys_snapshot().is_empty(), "split mode recorded a speculative key");
        q.finished(false);
    }

    /// `REQ_FLAG_RESID`: the residency map rides behind the partial, flagged by
    /// `RESP_FLAG_RESID`. The payload and the length check are unaffected, and
    /// a frame WITHOUT the bit but with the echoed request flag (an older
    /// daemon) still parses.
    #[test]
    fn response_residency_roundtrip() {
        let (b, n) = (2usize, 2 * N_EMBD as usize);
        let mut resp = AlignedBuf::with_capacity(proto::RESP_DATA_OFF + n * 2 + 64);
        proto::begin_response(&mut resp, 5, 21, b as u32, proto::REQ_FLAG_RESID, 0, 1, 2, N_EMBD, 2, 3, 4);
        resp.resize(proto::RESP_DATA_OFF + n * 2);
        for (i, v) in resp.view_mut::<u16>(proto::RESP_DATA_OFF, n).iter_mut().enumerate() {
            *v = i as u16;
        }
        // Older daemon: echoes the flag, appends nothing.
        let mut old = AlignedBuf::with_capacity(resp.len());
        old.extend_from_slice(resp.as_bytes());
        proto::patch_len(&mut old);
        let m = proto::decode_response_meta(&old).unwrap();
        assert!(proto::response_residency(&old, &m).is_none());

        let mut words = [0u32; proto::RESID_WORDS];
        words[0] = 0b1010;
        words[proto::RESID_WORDS - 1] = 1 << 31;
        proto::append_residency(&mut resp, &words);
        proto::patch_len(&mut resp);
        let m = proto::decode_response_meta(&resp).unwrap();
        assert_ne!(m.flags & proto::RESP_FLAG_RESID, 0);
        assert_eq!(proto::response_residency(&resp, &m).unwrap(), &words[..]);
        assert_eq!(resp.view::<u16>(proto::RESP_DATA_OFF, n)[n - 1], (n - 1) as u16);
    }

    /// Pin mode on the wire: RELEASE words after the prefetch block, the pin
    /// block after the residency map, and an older daemon's reply (the
    /// request's PIN bit echoed, no pin block) parsing as "no pin support".
    #[test]
    fn pin_proto_roundtrip() {
        let nu = N_EXPERT_USED;
        let xq = vec![7u8; XQ_BYTES_PER_TOKEN];
        let sel: Vec<i32> = (0..nu as i32).collect();
        let ew = vec![0.25f32; nu];
        let (pf, rel) = ([(3u32 << 16) | 5], [(3u32 << 16) | 7, (4u32 << 16) | 9]);
        let mut buf = AlignedBuf::with_capacity(1 << 16);
        let f = proto::REQ_FLAG_PIN | proto::REQ_FLAG_PREFETCH | proto::REQ_FLAG_RELEASE;
        proto::encode_request(&mut buf, 1, 3, 1, f, nu as u32, XQ_BYTES_PER_TOKEN as u32, &xq, &sel, &ew, (&[], &[]), &pf, &rel);
        let r = proto::decode_request(&buf).unwrap();
        assert_eq!((r.prefetch, r.release), (&pf[..], &rel[..]));
        // Release without prefetch; and neither (an older hub's frame).
        proto::encode_request(&mut buf, 2, 3, 1, proto::REQ_FLAG_RELEASE, nu as u32, XQ_BYTES_PER_TOKEN as u32, &xq, &sel, &ew, (&[], &[]), &[], &rel);
        let r = proto::decode_request(&buf).unwrap();
        assert!(r.prefetch.is_empty() && r.release == &rel[..]);
        proto::encode_request(&mut buf, 3, 3, 1, proto::REQ_FLAG_PIN, nu as u32, XQ_BYTES_PER_TOKEN as u32, &xq, &sel, &ew, (&[], &[]), &[], &rel);
        assert!(proto::decode_request(&buf).unwrap().release.is_empty(), "no RELEASE flag, no words");

        let (b, n) = (1usize, N_EMBD as usize);
        let mut resp = AlignedBuf::with_capacity(proto::RESP_DATA_OFF + n * 2 + 256);
        proto::begin_response(&mut resp, 1, 3, b as u32, proto::REQ_FLAG_PIN | proto::REQ_FLAG_RESID, 0, 1, 2, N_EMBD, 2, 3, 4);
        resp.resize(proto::RESP_DATA_OFF + n * 2);
        let mut old = AlignedBuf::with_capacity(resp.len() + 64);
        old.extend_from_slice(resp.as_bytes());
        let mut map = [0u32; proto::RESID_WORDS];
        map[0] = 0b1001;
        let mut paged = [0u32; proto::RESID_WORDS];
        paged[proto::RESID_WORDS - 1] = 1 << 31;
        proto::append_residency(&mut resp, &map);
        proto::append_pin(&mut resp, 5, 7, 9, &paged);
        proto::patch_len(&mut resp);
        let m = proto::decode_response_meta(&resp).unwrap();
        assert_eq!(proto::response_residency(&resp, &m).unwrap(), &map[..]);
        assert_eq!(proto::response_pin(&resp, &m), Some(proto::PinReply { epoch: 5, pinned: 7, budget: 9, paged }));
        // Older daemon: residency map only, PIN request bit merely echoed.
        proto::append_residency(&mut old, &map);
        proto::patch_len(&mut old);
        let m = proto::decode_response_meta(&old).unwrap();
        assert_ne!(m.flags & proto::REQ_FLAG_PIN, 0, "echoed");
        assert!(proto::response_pin(&old, &m).is_none());
        assert!(proto::response_residency(&old, &m).is_some());
    }

    /// `PinBook`: eligibility, pin on report within budget (the pass's own
    /// grants first), a denial drops the grant, a grant expires after
    /// `PIN_GRANT_TTL` reports, release (every word advances the epoch),
    /// eviction.
    #[test]
    fn pin_book_budget_release_eligibility() {
        let mut p = PinBook::off();
        p.grant(1, 2);
        assert!(!p.is_pinned(1, 2), "off: nothing pins");
        p.enable(2);
        let mut row = vec![0i32; REMAP_LEN];
        for (e, s) in [(2usize, 0i32), (3, 1), (4, 2)] {
            row[e] = -s - 1;
        }
        for e in [2, 3, 4, 5] {
            p.grant(1, e);
        }
        let w = p.report(1, &row, u32::MAX);
        assert_eq!(w[0], (1 << 2) | (1 << 3), "budget 2: the first two landed grants, in grant order");
        assert_eq!((p.pinned, p.c.new_pins, p.c.denied), (2, 2, 1));
        assert!(!p.is_pinned(1, 5), "5 is eligible but not landed");
        let _ = p.report(1, &row, u32::MAX);
        assert_eq!(p.c.denied, 1, "a denial drops the grant: counted once per grant");
        // Resident but never used / granted: never pinned.
        row[6] = -7;
        assert_eq!(p.report(1, &row, u32::MAX)[0] & (1 << 6), 0);
        // Release 2: epoch 1; 4 was denied (grant dropped) so the freed budget
        // stays free until it is granted again.
        p.release((1 << 16) | 2);
        assert_eq!((p.epoch, p.pinned), (1, 1));
        assert_eq!(p.report(1, &row, u32::MAX)[0], 1 << 3);
        p.grant(1, 4);
        assert_eq!(p.report(1, &row, u32::MAX)[0], (1 << 3) | (1 << 4));
        assert!(!p.is_pinned(1, 2), "a released expert needs a new grant to pin again");
        // Words for unknown / unpinned keys still advance the epoch.
        p.release((1 << 16) | 300);
        p.release((99 << 16) | 1);
        assert_eq!(p.epoch, 3);
        // 5 lands (still within its TTL): budget full, so its grant is dropped.
        row[5] = -9;
        let _ = p.report(1, &row, u32::MAX);
        assert!(!p.is_pinned(1, 5));
        assert_eq!(p.c.denied, 2);
        assert!(p.on_evict(1, 3), "evicting a pinned expert is reported");
        assert_eq!((p.pinned, p.c.pinned_evictions), (1, 1));
        row[3] = 0;
        assert_eq!(p.report(1, &row, u32::MAX)[0], 1 << 4, "5 needs a fresh grant");
        p.grant(1, 5);
        assert_eq!(p.report(1, &row, u32::MAX)[0], (1 << 4) | (1 << 5));
        assert!(!p.on_evict(1, 6));
        assert_eq!(p.pinned, 2);
        // Enabling again keeps the state; `off` resets it.
        p.enable(100);
        assert_eq!(p.budget, 2);

        // TTL: a grant survives PIN_GRANT_TTL - 1 reports without landing and
        // expires on the next; a repeat grant restarts the clock.
        let mut q = PinBook::off();
        q.enable(10);
        let mut row = vec![0i32; REMAP_LEN];
        q.grant(2, 7);
        for _ in 0..PIN_GRANT_TTL - 1 {
            let _ = q.report(2, &row, u32::MAX);
        }
        row[7] = -1;
        assert_eq!(q.report(2, &row, u32::MAX)[0], 1 << 7, "landed on its last report: pinned");
        q.grant(2, 8);
        for _ in 0..PIN_GRANT_TTL {
            let _ = q.report(2, &row, u32::MAX);
        }
        row[8] = -2;
        assert_eq!(q.report(2, &row, u32::MAX)[0] & (1 << 8), 0, "expired: never pinned");
        q.grant(2, 9);
        for _ in 0..PIN_GRANT_TTL - 1 {
            let _ = q.report(2, &row, u32::MAX);
        }
        q.grant(2, 9);
        for _ in 0..PIN_GRANT_TTL - 1 {
            let _ = q.report(2, &row, u32::MAX);
        }
        row[9] = -3;
        assert_eq!(q.report(2, &row, u32::MAX)[0] & (1 << 9), 1 << 9, "re-granted: clock restarted");
        assert_eq!(q.c.denied, 0);

        // Fresh first: with one budget slot, the pass's own pick (300) beats an
        // older eligible expert with a lower index (10) that landed meanwhile.
        let mut f = PinBook::off();
        f.enable(1);
        let mut row = vec![0i32; REMAP_LEN];
        f.grant(3, 10);
        let _ = f.report(3, &row, u32::MAX);
        row[10] = -1;
        row[300] = -2;
        f.grant(3, 300);
        let w = f.report(3, &row, u32::MAX);
        assert!(f.is_pinned(3, 300) && !f.is_pinned(3, 10), "{w:?}");
        assert_eq!((f.pinned, f.c.denied), (1, 1));
    }

    /// The arrival-time PAGED ring: zero bits keep nothing, entries leave by
    /// seq in any order, the oldest is dropped past `MAX`, `clear` empties.
    #[test]
    fn early_paged_ring() {
        let bits = |e: u32| {
            let mut w = [0u32; proto::RESID_WORDS];
            w[(e / 32) as usize] |= 1 << (e % 32);
            w
        };
        let mut r = EarlyPaged::default();
        r.note(1, [0; proto::RESID_WORDS]);
        assert!(r.is_empty(), "all-zero bits keep nothing");
        for seq in 1..=EarlyPaged::MAX as u32 + 8 {
            r.note(seq, bits(seq % 384));
        }
        assert_eq!(r.len(), EarlyPaged::MAX);
        for seq in 1..=8u32 {
            assert_eq!(r.take(seq), [0; proto::RESID_WORDS], "seq {seq} was the oldest: dropped");
        }
        assert_eq!(r.take(20), bits(20));
        assert_eq!(r.take(20), [0; proto::RESID_WORDS], "consumed");
        assert_eq!(r.take(EarlyPaged::MAX as u32 + 8), bits((EarlyPaged::MAX as u32 + 8) % 384));
        assert_eq!(r.len(), EarlyPaged::MAX - 2);
        r.clear();
        assert!(r.is_empty());
        assert_eq!(r.take(9), [0; proto::RESID_WORDS]);
    }

    /// The victim search never takes a hub-pinned slot, a wanted one, or a
    /// parked one; the choke point detaches exactly what it takes; a
    /// background landing with only pinned candidates is dropped.
    #[test]
    fn pool_victims_skip_pins() {
        let ids: Vec<u32> = (0..4).collect();
        let mut pool = ShardPool::seeded(8, &[(1, 0, &ids), (2, 4, &ids)], 0.0);
        pool.pins.enable(3);
        for e in 0..3 {
            pool.pins.grant(1, e);
        }
        let row = pool.remap_hosts[1].clone();
        let _ = pool.pins.report(1, &row, pool.stage);
        assert!(pool.pins.is_pinned(1, 0) && pool.pins.is_pinned(1, 2));
        // Oldest slots are layer 1's (seeded first); 0-2 pinned, 3 is next.
        let (slot, ev) = pool.claim_miss(2, 10, &[10], &[], (4, 8), true, false, false).unwrap();
        assert_eq!((slot, ev), (3, Some((1, 3))));
        pool.commit(2, 10, slot);
        // Then layer 2's, oldest first, minus wanted and parked ones.
        let (slot, ev) = pool.claim_miss(2, 11, &[11, 0], &[(2, 1)], (4, 8), true, false, false).unwrap();
        assert_eq!((slot, ev), (6, Some((2, 2))), "skips wanted 2/0 and parked 2/1");
        pool.commit(2, 11, slot);
        assert_eq!(pool.pins.c.pinned_evictions, 0);
        // Everything left unpinned is wanted or parked: nothing to take
        // without a revoke; a background landing would be dropped.
        let want = [10u32, 11, 0, 3];
        let extra = [(2u32, 1u32)];
        assert!(pool.pick_victim_any((4, 8), true, Band::Main, 2, &want, &extra, 2, false, false).is_none());
        assert!(pool.pick_victim_any((4, 8), true, Band::Main, 2, &want, &extra, 2, true, false).is_some(), "only pins stand in the way");
        for (l, e) in [(1u32, 0u32), (1, 1), (1, 2)] {
            assert!(pool.slot_of.contains_key(&(l, e)) && pool.remap_hosts[l as usize][e as usize] != 0);
        }
    }

    /// With a reserve below the workload (misconfiguration), a demand claim
    /// REVOKES a pin rather than failing; the choke point reports it. Release
    /// builds only: in debug the violation is a `debug_assert`.
    #[cfg(not(debug_assertions))]
    #[test]
    fn pool_revokes_rather_than_fails() {
        let ids: Vec<u32> = (0..2).collect();
        let mut pool = ShardPool::seeded(2, &[(1, 0, &ids)], 0.0);
        pool.pins.enable(2);
        pool.pins.grant(1, 0);
        pool.pins.grant(1, 1);
        let row = pool.remap_hosts[1].clone();
        let _ = pool.pins.report(1, &row, pool.stage);
        let v0 = PIN_VIOLATIONS.load(std::sync::atomic::Ordering::Relaxed);
        let (_, ev) = pool.claim_miss(1, 5, &[5], &[], (0, 2), true, false, false).expect("revoke, not fail");
        assert!(ev.is_some());
        assert_eq!((pool.pins.c.revokes, pool.pins.c.pinned_evictions, pool.pins.pinned), (1, 1, 1));
        assert!(PIN_VIOLATIONS.load(std::sync::atomic::Ordering::Relaxed) > v0);
    }

    /// MODE-AWARE EVICTION (the block above `ModeEvictCounters`): a prefill
    /// phase takes stale prefill slots first, then decode's LRU within its
    /// budget (recording each in the delta), then its own oldest pages; a
    /// decode claim is unchanged; with the mode off a prefill search is the
    /// plain global LRU.
    #[test]
    fn pool_mode_evict_tiers_budget_and_delta() {
        let ids: Vec<u32> = (0..4).collect();
        // Off: a prefill-mode search is the plain LRU (slot 0 is the oldest).
        let pool = ShardPool::seeded(12, &[(1, 0, &ids), (2, 4, &ids), (3, 8, &ids)], 0.0);
        assert_eq!(pool.pick_victim_any((0, 12), true, Band::Main, 9, &[], &[], 9, false, true), Some(0));

        let mut pool = ShardPool::seeded(12, &[(1, 0, &ids), (2, 4, &ids), (3, 8, &ids)], 0.0);
        pool.enable_mode_evict(2);
        // Decode touches layers 2 and 3 (decode-class), in the order
        // (2,0) (3,0) (2,1) (3,1) ...; layer 1 stays prefill-class (seeded).
        for e in 0..4 {
            assert!(pool.touch_hit(2, e, false));
            assert!(pool.touch_hit(3, e, false));
        }
        pool.me_note_request(true);
        assert_eq!((pool.me.phase, pool.me.budget_left), (1, 2));
        let region = (0, 12);
        let want5: Vec<u32> = (20..27).collect();
        // Tier 1: layer 1's stale prefill slots first.
        for e in 20..24 {
            let (slot, ev) = pool.claim_miss(5, e, &want5, &[], region, true, true, true).unwrap();
            assert_eq!(ev.map(|x| x.0), Some(1), "claim of {e} takes a stale prefill slot");
            pool.commit(5, e, slot);
        }
        // Tier 2: decode's LRU while the budget lasts (layer 5's own pages are
        // wanted by this pass, so they are not candidates here anyway).
        let (s1, ev1) = pool.claim_miss(5, 24, &want5, &[], region, true, true, true).unwrap();
        assert_eq!(ev1, Some((2, 0)));
        pool.commit(5, 24, s1);
        let (s2, ev2) = pool.claim_miss(5, 25, &want5, &[], region, true, true, true).unwrap();
        assert_eq!(ev2, Some((3, 0)));
        pool.commit(5, 25, s2);
        assert_eq!(pool.me.budget_left, 0);
        // Tier 3: budget spent, another layer's claim takes prefill's OWN
        // oldest page, not decode's.
        let (s3, ev3) = pool.claim_miss(6, 30, &[30], &[], region, true, true, true).unwrap();
        assert_eq!(ev3, Some((5, 20)));
        pool.commit(6, 30, s3);
        let c = pool.me.c;
        assert_eq!((c.took_stale, c.took_decode, c.took_own, c.took_decode_over), (4, 2, 1, 0));
        let delta: Vec<(u32, u32)> = pool.me.decode_delta.iter().map(|&(l, e, _, _)| (l, e)).collect();
        assert_eq!(delta, vec![(2, 0), (3, 0)]);
        assert!(pool.me.decode_delta.iter().all(|&(_, _, t, ph)| t >= PREFILL_AGE && ph == 1), "the delta keeps decode stamps");
        // Decode mode is unchanged: prefill-class (layer 5/6 pages) first.
        for _ in 0..ME_DECODE_STREAK {
            pool.me_note_request(false);
        }
        assert!(!pool.me.prefill_phase);
        let (s4, ev4) = pool.claim_miss(8, 40, &[40], &[], region, true, false, false).unwrap();
        assert_eq!(ev4.map(|x| x.0), Some(5), "a decode claim takes prefill-class first");
        pool.commit(8, 40, s4);
        // The next prefill phase gets a fresh budget.
        pool.me_note_request(true);
        assert_eq!((pool.me.phase, pool.me.budget_left, pool.me.c), (2, 2, ModeEvictCounters::default()));
    }

    /// SF1 regression: a few decode-flagged requests inside a prefill burst (a
    /// decode request served inside a parked chunk) must not restart the phase
    /// -- that would reset the budget and turn the ongoing prefill's pages
    /// stale. A full streak ends it; outside a prefill phase nothing is tiered.
    #[test]
    fn pool_mode_evict_ranks_decode_victims_by_the_sweep() {
        // Decoder layer 25's decode pages are OLDER than encoder layer 3's, yet
        // a layer-7 prefill claim in the decode tier takes layer 3's first (the
        // sweep has passed it; the replay right after needs the decoder layers),
        // LRU within each (`knobs::encoder_victims_first`, default on).
        // Plus encoder layer 12, AHEAD of a layer-7 claim's sweep: last, even
        // though its pages are the oldest of all.
        let ids: Vec<u32> = (0..2).collect();
        let mut pool = ShardPool::seeded(6, &[(12, 0, &ids), (25, 2, &ids), (3, 4, &ids)], 0.0);
        pool.enable_mode_evict(8);
        for l in [12, 25, 3] {
            for e in 0..2 {
                assert!(pool.touch_hit(l, e, false));
            }
        }
        pool.me_note_request(true);
        let want: Vec<u32> = (10..16).collect();
        let mut order = Vec::new();
        for e in 10..16 {
            let (slot, ev) = pool.claim_miss(7, e, &want, &[], (0, 6), true, true, true).unwrap();
            order.push(ev.unwrap());
            pool.commit(7, e, slot);
        }
        assert_eq!(order, vec![(3, 0), (3, 1), (25, 0), (25, 1), (12, 0), (12, 1)]);
        assert_eq!(pool.me.c.took_decode, 6);
    }

    #[test]
    fn pool_mode_evict_phase_does_not_flap() {
        let ids: Vec<u32> = (0..4).collect();
        let mut pool = ShardPool::seeded(8, &[(1, 0, &ids), (2, 4, &ids)], 0.0);
        pool.enable_mode_evict(3);
        for e in 0..4 {
            assert!(pool.touch_hit(2, e, false)); // decode-class
        }
        pool.me_note_request(true);
        let (phase, start) = (pool.me.phase, pool.me.phase_start);
        // Spend one unit of budget: stale prefill first (layer 1), then decode.
        for e in 10..15 {
            let (slot, _) = pool.claim_miss(3, e, &[10, 11, 12, 13, 14], &[], (0, 8), true, true, true).unwrap();
            pool.commit(3, e, slot);
        }
        assert_eq!((pool.me.c.took_stale, pool.me.c.took_decode, pool.me.budget_left), (4, 1, 2));
        for _ in 0..ME_DECODE_STREAK - 1 {
            pool.me_note_request(false);
        }
        pool.me_note_request(true);
        assert_eq!((pool.me.phase, pool.me.phase_start, pool.me.budget_left), (phase, start, 2), "no restart");
        assert!(pool.me.prefill_phase && pool.me_tiered(true));
        for _ in 0..ME_DECODE_STREAK {
            pool.me_note_request(false);
        }
        assert!(!pool.me.prefill_phase && !pool.me_tiered(true), "a full streak ends it; untiered outside");
        pool.me_note_request(true);
        assert_eq!((pool.me.phase, pool.me.budget_left), (phase + 1, 3));
    }

    /// Randomized: prefill and decode phases, hits, claims and landings through
    /// the real claim / landing calls; every victim is checked against the
    /// tier rule computed independently, plus budget, delta and map invariants.
    #[test]
    fn pool_mode_evict_randomized_against_oracle() {
        const LAYERS: u32 = 4;
        // Real layer ids on both sides of CED_DECODER_START (the sweep rank).
        const LAYER_IDS: [u32; LAYERS as usize] = [3, 9, 21, 30];
        const PER: u32 = 12;
        const IDS: u32 = 20;
        let n = (LAYERS * PER) as usize;
        let ids: Vec<u32> = (0..PER).collect();
        let regions: Vec<(u32, u32, &[u32])> = (0..LAYERS).map(|i| (LAYER_IDS[i as usize], i * PER, &ids[..])).collect();
        for seed in 1..=8u64 {
            let mut rng = SimRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
            let mut pool = ShardPool::seeded(n, &regions, 0.0);
            pool.enable_mode_evict(6);
            let (mut prefill_victims, mut checked) = (0u64, 0u64);
            let mut phase_decode_victims = 0u64;
            let mut last_phase = 0u32;
            for step in 0..4000u32 {
                // Phases in runs of requests (with occasional stray decode ones).
                let prefill_req = (step / 200) % 2 == 1 && rng.below(10) != 0;
                pool.me_note_request(prefill_req);
                if pool.me.phase != last_phase {
                    last_phase = pool.me.phase;
                    phase_decode_victims = 0;
                }
                // As `ensure_layer_inner` with the mode on: a prefill claim only
                // inside a prefill phase AND for a prefill-flagged request.
                let prefill_mode = pool.me.prefill_phase && prefill_req;
                let li = rng.below(LAYERS as u64) as u32;
                let layer = LAYER_IDS[li as usize];
                let want: Vec<u32> = (0..1 + rng.below(6)).map(|_| rng.below(IDS as u64) as u32).collect();
                let landing = rng.below(8) == 0;
                for &e in &want {
                    if pool.touch_hit(layer, e, prefill_mode) {
                        continue;
                    }
                    // Oracle: eligible = free, or not one of `want` on `layer`;
                    // rank (tier, last_use) in a tiered search, else last_use.
                    let tiered = pool.me.on && pool.me.prefill_phase && prefill_mode;
                    let rank = |sl: usize| -> (u8, u8, u64) {
                        let lu = pool.last_use[sl];
                        let t = match pool.owner_of[sl] {
                            _ if !tiered => 0,
                            None => 0,
                            Some(_) if lu < PREFILL_AGE => if lu < pool.me.phase_start { 1 } else { 3 },
                            Some(_) if pool.me.phase_touch[sl] == pool.me.phase => 5,
                            Some(_) if pool.me.budget_left > 0 => 2,
                            Some(_) => 4,
                        };
                        // The sweep rank (knob default on): encoder layers behind
                        // the claim's layer, then the other region, then layers
                        // the sweep still reaches.
                        let split = crate::config::CED_DECODER_START as u32;
                        let dl = match pool.owner_of[sl] {
                            Some((l, _)) if tiered && (t == 2 || t == 4) => {
                                if l >= layer && ((l < split) == (layer < split)) { 2 } else if l < split && l < layer { 0 } else { 1 }
                            }
                            _ => 0,
                        };
                        (t, dl, lu)
                    };
                    let expect = (0..n)
                        .filter(|&sl| !matches!(pool.owner_of[sl], Some((ol, oe)) if ol == layer && want.contains(&oe)))
                        .min_by_key(|&sl| (rank(sl), sl))
                        .map(|sl| sl as u32);
                    let exp_tier = expect.map(|v| rank(v as usize).0);
                    let exp_decode = expect.is_some_and(|v| pool.owner_of[v as usize].is_some() && pool.last_use[v as usize] >= PREFILL_AGE);
                    let got = if landing {
                        let v = pool.pick_victim_any((0, n as u32), true, Band::Main, layer, &want, &[], layer, false, prefill_mode);
                        if let Some(v) = v {
                            pool.me_account(v, prefill_mode);
                            pool.evict(v, layer);
                            pool.land(v, (layer, e), prefill_mode);
                        }
                        v
                    } else {
                        let c = pool.claim_miss(layer, e, &want, &[], (li * PER, (li + 1) * PER), true, prefill_mode, prefill_mode);
                        c.map(|(v, _)| {
                            pool.commit(layer, e, v);
                            v
                        })
                    };
                    assert_eq!(got, expect, "seed {seed} step {step}: victim");
                    checked += 1;
                    if got.is_some() && tiered {
                        prefill_victims += 1;
                        if exp_decode && exp_tier == Some(2) {
                            phase_decode_victims += 1;
                        }
                    }
                    assert!(phase_decode_victims <= pool.me.budget, "seed {seed}: budget overrun");
                }
                // Map invariants.
                for (sl, o) in pool.owner_of.iter().enumerate() {
                    if let Some(k) = o {
                        assert_eq!(pool.slot_of.get(k), Some(&(sl as u32)), "seed {seed}: maps disagree");
                    }
                }
                assert_eq!(pool.slot_of.len(), pool.owner_of.iter().filter(|o| o.is_some()).count());
            }
            assert!(pool.me.decode_delta.iter().all(|&(_, _, t, ph)| t >= PREFILL_AGE && ph >= 1), "seed {seed}: delta stamps");
            assert!(checked > 2000 && prefill_victims > 300 && pool.me.phase >= 5, "seed {seed}: not exercised ({checked}, {prefill_victims}, {})", pool.me.phase);
        }
    }

    /// DELTA RESTORE: after a prefill phase the decode experts it evicted come
    /// back newest-first, each landing only over what decode would evict before
    /// it (prefill-class first, then decode slots OLDER than its own stamp) and
    /// keeping that stamp; what decode demand-paged meanwhile (fresh stamps) is
    /// never a victim, and the restore stops when nothing is older.
    #[test]
    fn pool_restore_lands_only_over_older_and_keeps_its_stamp() {
        let ids: Vec<u32> = (0..4).collect();
        let mut pool = ShardPool::seeded(8, &[(1, 0, &ids), (2, 4, &ids)], 0.0);
        pool.enable_mode_evict(8);
        pool.me.restore_on = true;
        for l in [1u32, 2] {
            for e in 0..4 {
                assert!(pool.touch_hit(l, e, false)); // all decode-class
            }
        }
        let stamp_of = |p: &ShardPool, k: (u32, u32)| p.last_use[p.slot_of[&k] as usize];
        // Prefill phase: evicts (1,0) and (1,1) (the two oldest decode slots).
        pool.me_note_request(true);
        for e in 20..22 {
            let (slot, _) = pool.claim_miss(3, e, &[20, 21], &[], (0, 8), true, true, true).unwrap();
            pool.commit(3, e, slot);
        }
        let delta: Vec<(u32, u32, u64)> = pool.me.decode_delta.iter().map(|&(l, e, t, _)| (l, e, t)).collect();
        assert_eq!(delta.iter().map(|&(l, e, _)| (l, e)).collect::<Vec<_>>(), vec![(1, 0), (1, 1)]);
        let (t10, t11) = (delta[0].2, delta[1].2);
        // Back to decode: the queue is the delta, newest stamp first.
        for _ in 0..ME_DECODE_STREAK {
            pool.me_note_request(false);
        }
        assert_eq!(pool.me.restore.iter().map(|&(l, e, _)| (l, e)).collect::<Vec<_>>(), vec![(1, 1), (1, 0)]);
        assert_eq!(pool.me.rc.queued, 2);
        // Decode demand-pages (4,0) before the restore lands: fresh stamp.
        let (s40, _) = pool.claim_miss(4, 0, &[0], &[], (0, 8), true, false, false).unwrap();
        pool.commit(4, 0, s40);
        let t40 = pool.last_use[s40 as usize];
        assert!(t40 > t11, "a demand page during the restore is younger than every entry");
        // (1,1): the victim is the remaining PREFILL-class page (layer 3).
        let v = pool.restore_victim((0, 8), true, 9, &[], &[], 1, t11).expect("prefill-class page first");
        assert_eq!(pool.owner_of[v as usize].map(|k| k.0), Some(3));
        pool.evict(v, 9);
        pool.land_stamped(v, (1, 1), t11);
        assert_eq!(stamp_of(&pool, (1, 1)), t11, "keeps its own stamp");
        // (1,0), stamp t10: the plain LRU victim is now (1,1) at t11 > t10 --
        // nothing is older than the entry, so it would only be evicted first:
        // STOP. (4,0) was never a candidate either.
        assert!(pool.restore_victim((0, 8), true, 9, &[], &[], 1, t10).is_none());
        assert_eq!(stamp_of(&pool, (4, 0)), t40);
        // Merge: an old queue entry and a new delta entry for the same expert
        // keep the newer stamp; resident experts are dropped.
        pool.me.restore.clear();
        pool.me.restore.push_back((2, 9, 5 + PREFILL_AGE));
        pool.me.decode_delta.push_back((2, 9, 7 + PREFILL_AGE, 3));
        pool.me.decode_delta.push_back((1, 1, 9 + PREFILL_AGE, 3)); // resident again
        pool.me.decode_delta.push_back((2, 8, 6 + PREFILL_AGE, 3));
        pool.me_build_restore();
        assert_eq!(pool.me.restore.iter().copied().collect::<Vec<_>>(), vec![(2, 9, 7 + PREFILL_AGE), (2, 8, 6 + PREFILL_AGE)]);
        assert!(pool.me.decode_delta.is_empty());
    }

    /// Randomized DELTA RESTORE: prefill phases build the delta, decode phases
    /// interleave demand claims / hits with OUT-OF-ORDER landings of in-flight
    /// restores (up to 4 popped from the queue, landed in random order). A
    /// restore never evicts a decode slot at or above its own stamp (so never a
    /// page decode made during the restore), keeps its stamp, and stops only
    /// when no eligible slot is older; the queue is duplicate-free, newest
    /// first, bounded, and never holds a resident expert when built.
    #[test]
    fn pool_restore_randomized_never_evicts_younger() {
        const LAYERS: u32 = 4;
        const PER: u32 = 12;
        let n = (LAYERS * PER) as usize;
        let ids: Vec<u32> = (0..PER).collect();
        let regions: Vec<(u32, u32, &[u32])> = (0..LAYERS).map(|l| (l, l * PER, &ids[..])).collect();
        let (mut landed, mut stops, mut younger_seen) = (0u64, 0u64, 0u64);
        for seed in 1..=8u64 {
            let mut rng = SimRng(seed.wrapping_mul(0xD1B5_4A32_D192_ED03) | 1);
            let mut pool = ShardPool::seeded(n, &regions, 0.0);
            pool.enable_mode_evict(10);
            pool.me.restore_on = true;
            for l in 0..LAYERS {
                for e in 0..PER {
                    assert!(pool.touch_hit(l, e, false));
                }
            }
            for cycle in 0..40u32 {
                // Prefill phase: claims for prefill-only experts (ids 30..60).
                pool.me_note_request(true);
                for _ in 0..(5 + rng.below(25)) {
                    let (l, e) = (rng.below(LAYERS as u64) as u32, 30 + rng.below(30) as u32);
                    if !pool.touch_hit(l, e, true) {
                        let (v, _) = pool.claim_miss(l, e, &[e], &[], (l * PER, (l + 1) * PER), true, true, true).unwrap();
                        pool.commit(l, e, v);
                    }
                }
                // Decode phase: the switch builds the queue.
                for _ in 0..ME_DECODE_STREAK {
                    pool.me_note_request(false);
                }
                assert!(!pool.me.prefill_phase);
                let q: Vec<(u32, u32, u64)> = pool.me.restore.iter().copied().collect();
                let keys: std::collections::HashSet<(u32, u32)> = q.iter().map(|&(l, e, _)| (l, e)).collect();
                assert_eq!(keys.len(), q.len(), "seed {seed} cycle {cycle}: duplicate keys");
                assert!(q.len() <= N_LAYER as usize * N_EXPERT as usize);
                assert!(q.windows(2).all(|w| w[0].2 >= w[1].2), "newest first");
                assert!(q.iter().all(|&(l, e, _)| !pool.slot_of.contains_key(&(l, e))), "none resident");
                let restore_start = pool.tick;
                let mut inflight: Vec<(u32, u32, u64)> = Vec::new();
                for _ in 0..(20 + rng.below(60)) {
                    if rng.below(2) == 0 {
                        // Decode demand: a hit or a claim (fresh stamp).
                        let (l, e) = (rng.below(LAYERS as u64) as u32, rng.below(20) as u32);
                        if !pool.touch_hit(l, e, false) {
                            let (v, _) = pool.claim_miss(l, e, &[e], &[], (l * PER, (l + 1) * PER), true, false, false).unwrap();
                            pool.commit(l, e, v);
                        }
                        continue;
                    }
                    while inflight.len() < 4 {
                        let Some(ent) = pool.me.restore.pop_front() else { break };
                        inflight.push(ent);
                    }
                    if inflight.is_empty() {
                        continue;
                    }
                    let (l, e, stamp) = inflight.swap_remove(rng.below(inflight.len() as u64) as usize);
                    if pool.slot_of.contains_key(&(l, e)) {
                        continue;
                    }
                    match pool.restore_victim((0, n as u32), true, l, &[], &[], l, stamp) {
                        None => {
                            // Stop only when nothing eligible is older.
                            assert!(pool.owner_of.iter().enumerate().all(|(sl, o)| o.is_some() && pool.last_use[sl] >= PREFILL_AGE && pool.last_use[sl] >= stamp),
                                "seed {seed}: a stop with an older / prefill / free slot available");
                            pool.me.restore.clear();
                            inflight.clear();
                            stops += 1;
                        }
                        Some(v) => {
                            let lu = pool.last_use[v as usize];
                            let occupied = pool.owner_of[v as usize].is_some();
                            assert!(!occupied || lu < PREFILL_AGE || lu < stamp, "seed {seed}: restore evicts a younger decode slot");
                            assert!(!(occupied && lu >= PREFILL_AGE && lu - PREFILL_AGE > restore_start), "seed {seed}: evicts a page decode made during the restore");
                            younger_seen += u64::from(pool.last_use.iter().any(|&t| t >= PREFILL_AGE && t - PREFILL_AGE > restore_start));
                            pool.evict(v, l);
                            pool.land_stamped(v, (l, e), stamp);
                            assert_eq!(pool.last_use[v as usize], stamp);
                            landed += 1;
                        }
                    }
                }
                for (sl, o) in pool.owner_of.iter().enumerate() {
                    if let Some(k) = o {
                        assert_eq!(pool.slot_of.get(k), Some(&(sl as u32)), "seed {seed}: maps disagree");
                    }
                }
            }
        }
        assert!(landed > 300 && stops > 5 && younger_seen > 50, "not exercised: landed {landed} stops {stops} younger {younger_seen}");
    }

    /// Past its budget a prefill phase takes decode slots it is NOT using
    /// (tier 4) before the ones it hit this phase (tier 5).
    #[test]
    fn pool_mode_evict_spares_decode_slots_prefill_is_using() {
        let ids: Vec<u32> = (0..3).collect();
        let mut pool = ShardPool::seeded(3, &[(2, 0, &ids)], 0.0);
        pool.enable_mode_evict(0);
        for e in 0..3 {
            assert!(pool.touch_hit(2, e, false)); // all decode-class, (2,0) oldest
        }
        pool.me_note_request(true);
        assert!(pool.touch_hit(2, 0, true), "a prefill hit on decode's slot");
        assert!(pool.last_use[0] >= PREFILL_AGE, "it stays decode-class");
        let (_, ev) = pool.claim_miss(7, 50, &[50], &[], (0, 3), true, true, true).unwrap();
        assert_eq!(ev, Some((2, 1)), "the oldest decode slot prefill is NOT using");
        assert_eq!(pool.me.c.took_decode_over, 1);
    }

    /// PREFILL STAGING on a seeded pool (the block above `PIN_RESERVE_MIN`):
    /// a prefill-shaped claim takes a staging slot and nothing else does; a
    /// staged landing lands inside and a decode landing outside; `report`
    /// never pins an expert resident in staging (it is reported not held);
    /// the two spill directions are counted and never revoke a pin.
    #[test]
    fn pool_prefill_stage_bands() {
        let ids: Vec<u32> = (0..4).collect();
        // 12 slots: layer 1 at 0..4, layer 2 at 4..8, layer 3 at 8..12.
        let mut pool = ShardPool::seeded(12, &[(1, 0, &ids), (2, 4, &ids), (3, 8, &ids)], 0.0);
        assert_eq!(pool.set_stage(4), 4, "the last 4 slots");
        assert_eq!((pool.stage, pool.stage_slots()), (8, 4));
        assert_eq!(pool.set_stage(100), 6, "clamped to half the pool");
        assert_eq!(pool.set_stage(4), 4);
        let region = (0, 4);
        // Prefill claims: LRU within staging, never outside, whatever the LRU
        // says about the main band (layer 1's slots are the oldest).
        for (k, e) in (20..23).enumerate() {
            let (slot, ev) = pool.claim_miss(1, e, &[20, 21, 22, 23], &[], region, true, true, true).unwrap();
            assert_eq!((slot, ev), (8 + k as u32, Some((3, k as u32))), "prefill claim {k} in staging");
            pool.commit(1, e, slot);
        }
        assert_eq!((pool.sc.claims, pool.sc.spill_out), (3, 0));
        // A decode claim: the main band only (the free-est staging slot, 11,
        // holds 3/3 and is older than everything in main, but is off limits).
        let (slot, ev) = pool.claim_miss(2, 30, &[30], &[], (4, 8), true, false, false).unwrap();
        assert_eq!((slot, ev), (0, Some((1, 0))), "decode claim in main");
        pool.commit(2, 30, slot);
        assert_eq!(pool.sc.claims, 3);
        // Landings: a staged one inside, a decode one outside.
        let v = pool.pick_victim_any(region, true, Band::Stage, 2, &[], &[], 1, false, false).unwrap();
        assert_eq!(v, 11);
        pool.evict(v, 2);
        pool.land(v, (1, 40), true);
        assert!(pool.last_use[11] < PREFILL_AGE, "a staged landing is prefill-class");
        let v = pool.pick_victim_any(region, true, Band::Main, 2, &[], &[], 1, false, false).unwrap();
        assert!(v < 8, "a decode landing stays in main (got {v})");
        pool.evict(v, 2);
        pool.land(v, (1, 41), false);
        assert!(pool.last_use[v as usize] >= PREFILL_AGE);
        // Hits on staged experts are counted; pins never land in staging.
        assert!(pool.touch_hit(1, 40, false) && pool.touch_hit(1, 41, false));
        assert_eq!(pool.sc.hits, 1);
        // A decode hit keeps a STAGED slot prefill-class (staging's LRU must be
        // able to recycle it into main), and a main slot decode-class.
        assert!(pool.last_use[11] < PREFILL_AGE, "staged slot stays prefill-class after a decode hit");
        assert!(pool.last_use[v as usize] >= PREFILL_AGE);
        pool.pins.enable(8);
        pool.pins.grant(1, 40); // staged
        pool.pins.grant(1, 41); // main
        pool.pins.grant(1, 20); // staged (a prefill claim)
        let row = pool.remap_hosts[1].clone();
        let map = pool.pins.report(1, &row, pool.stage);
        assert!(pool.pins.is_pinned(1, 41) && map[1] & (1 << 9) != 0);
        assert!(!pool.pins.is_pinned(1, 40) && !pool.pins.is_pinned(1, 20), "never pinned in staging");
        assert_eq!(map[1] & (1 << 8), 0);
        assert_eq!((pool.pins.pinned, pool.pins.c.denied), (1, 0), "a staged grant ages, it is not denied");
        for sl in pool.stage..12 {
            let (l, e) = pool.owner_of[sl as usize].unwrap();
            assert!(!pool.pins.is_pinned(l, e));
        }
        // Spill OUT: a prefill union that fills staging (4 wanted, all
        // resident there) claims its 5th expert in main, counted, no revoke.
        for sl in 8..12 {
            pool.evict(sl, 1);
            pool.owner_of[sl as usize] = Some((1, 50 + sl - 8));
            pool.slot_of.insert((1, 50 + sl - 8), sl);
            pool.held[1] += 1;
        }
        let (slot, _) = pool.claim_miss(1, 54, &[50, 51, 52, 53, 54], &[], region, true, true, true).unwrap();
        assert!(slot < 8, "spilled into main (got {slot})");
        pool.commit(1, 54, slot);
        assert_eq!((pool.sc.spill_out, pool.pins.c.revokes), (1, 0));
        // Spill IN: every main slot pinned, wanted or parked -> a decode claim
        // takes a staging slot rather than revoking a pin.
        for sl in 0..8u32 {
            if let Some((l, e)) = pool.owner_of[sl as usize] {
                pool.pins.grant(l, e);
            }
        }
        for l in 1..=3u32 {
            let row = pool.remap_hosts[l as usize].clone();
            let _ = pool.pins.report(l, &row, pool.stage);
        }
        assert_eq!(pool.pins.pinned, 8);
        let (slot, ev) = pool.claim_miss(2, 60, &[60], &[], (4, 8), true, false, false).unwrap();
        assert!(slot >= 8 && ev.is_some(), "spilled into staging (got {slot})");
        assert_eq!((pool.sc.spill_in, pool.pins.c.revokes, pool.pins.c.pinned_evictions), (1, 0, 0));
        // Staging off: one band again, every search sees the whole pool.
        assert_eq!(pool.set_stage(0), 0);
        assert_eq!(pool.stage, 12);
        let sc = pool.sc;
        let (slot, _) = pool.claim_miss(2, 61, &[61], &[], (4, 8), true, true, true).unwrap();
        assert!(slot >= 8, "with staging off the LRU (a staging slot) wins: {slot}");
        assert_eq!(pool.sc, sc, "no staging counters move with staging off");
    }

    /// THE PIN PROTOCOL, randomized: box 2's real pool, victim search, choke
    /// point and `PinBook` (driven in `serve_connection`'s order: words at
    /// service in arrival order, merged partners, parked requests with others
    /// served inside, background reads landing in random order and dropped
    /// when every victim is pinned, early-page reads, prefill-shaped chunks),
    /// against the hub's real `PinLedger` (releases at step start, replies
    /// consumed in random order, Zipf decode picks, admission words). After
    /// every event: `held ⊆ pinned ⊆ landed` and `pinned <= budget`; every
    /// reply: zero surprises; every claim: a victim without revoking. Then the
    /// same stream with a hub that forgets the epoch mask must be CAUGHT.
    #[test]
    fn pin_protocol_randomized() {
        for seed in 1..=6u64 {
            let s = pin_sim(seed, false);
            assert_eq!(s.surprises, 0, "seed {seed}: {s:?}");
            assert_eq!(s.subset_violations, 0, "seed {seed}: {s:?}");
            assert_eq!(s.c.pinned_evictions + s.c.revokes, 0, "seed {seed}: {s:?}");
            // Not vacuous: the protocol was exercised hard.
            assert!(s.held_checked > 2_000, "seed {seed}: {s:?}");
            assert!(s.c.releases > 100 && s.c.new_pins > 300 && s.evictions > 500, "seed {seed}: {s:?}");
            assert!(s.merged > 5 && s.parked > 20 && s.served_inside > 20 && s.prefill > 20, "seed {seed}: {s:?}");
            assert!(s.bg_landed > 100 && s.stale_maps > 50, "seed {seed}: {s:?}");
            // Prefill staging was exercised: prefill chunks claimed in the
            // band and hit there across chunks, never spilled out (STAGE >=
            // the union); the per-event invariants live in `Box2Sim`.
            assert!(s.sc.claims > 100 && s.sc.hits > 50 && s.sc.spill_out == 0, "seed {seed}: {s:?}");
            eprintln!("pin sim seed {seed}: {s:?}");
        }
        // Mutation: a hub that applies maps without masking later releases.
        let caught = (1..=6u64).map(|seed| pin_sim(seed, true)).map(|s| s.subset_violations + s.surprises as u64).sum::<u64>();
        assert!(caught > 0, "the checks did not catch a hub that ignores the release epoch");
    }

    #[derive(Debug, Default)]
    struct PinSimStats {
        c: PinCounters,
        /// Merged passes where the partner held an expert its lane mate found
        /// missing at arrival (a false surprise if the bits were shared).
        partner_split: u64,
        surprises: u32,
        subset_violations: u64,
        held_checked: u64,
        evictions: u64,
        merged: u64,
        parked: u64,
        served_inside: u64,
        prefill: u64,
        bg_landed: u64,
        bg_dropped: u64,
        stale_maps: u64,
        max_pinned: u32,
        /// The pool's prefill-staging counters at the end.
        sc: StageCounters,
    }

    /// xorshift64.
    struct SimRng(u64);
    impl SimRng {
        fn below(&mut self, n: u64) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0 % n.max(1)
        }
    }

    #[derive(Clone)]
    struct SimReq {
        seq: u32,
        layer: u32,
        b: usize,
        sel: Vec<i32>,
        prefetch: Vec<u32>,
        release: Vec<u32>,
        /// Hub side: the sent picks the ledger held at submit.
        held: [u32; proto::RESID_WORDS],
        /// Box 2 has pulled this frame off the wire (its arrival bits noted).
        pulled: bool,
    }

    struct SimReply {
        seq: u32,
        layer: u32,
        map: [u32; proto::RESID_WORDS],
        epoch: u32,
        pinned: u32,
        budget: u32,
        paged: [u32; proto::RESID_WORDS],
    }

    /// Box 2 for `pin_sim`: the REAL `ShardPool` (victim search, choke point,
    /// claims, landings) and `PinBook`, driven in `serve_connection`'s order.
    /// Only the reads are instant and the kernels absent.
    struct Box2Sim {
        pool: ShardPool,
        /// Background reads in flight (prefetch words, early page, park), and
        /// whether each is a PREFILL-shaped request's own (`PfJob::stage`).
        bg: Vec<((u32, u32), bool)>,
        budget: u32,
        per: u32,
        global: bool,
        /// Arrival-time paged bits per queued frame (the early-page hook).
        early: EarlyPaged,
        /// Deterministic: always merge a same-layer front frame, never park,
        /// no early background reads (the scripted scenario tests).
        scripted: bool,
        st: PinSimStats,
    }

    impl Box2Sim {
        const BG_SETS: usize = 8;

        fn region(&self, l: u32) -> (u32, u32) {
            (l * self.per, (l + 1) * self.per)
        }

        fn distinct(sel: &[i32]) -> Vec<u32> {
            let mut v: Vec<u32> = Vec::new();
            for &e in sel {
                if e >= 0 && !v.contains(&(e as u32)) {
                    v.push(e as u32);
                }
            }
            v
        }

        /// `prefetch_words_cls`: skip resident / in flight; bounded sets.
        fn queue_bg(&mut self, words: &[u32], stage: bool) {
            let stage = stage && self.pool.stage_slots() > 0;
            for &w in words {
                let key = (w >> 16, w & 0xFFFF);
                if !self.pool.slot_of.contains_key(&key) && !self.bg.iter().any(|b| b.0 == key) && self.bg.len() < Self::BG_SETS {
                    self.bg.push((key, stage));
                }
            }
        }

        /// `admit_prefetched`: a random subset has completed; land it in random
        /// order, protecting `want` of `layer` and `extra`; drop the landing
        /// when every victim is pinned (or, staged, when staging has none).
        fn admit(&mut self, rng: &mut SimRng, layer: u32, want: &[u32], extra: &[(u32, u32)]) {
            let mut i = 0;
            while i < self.bg.len() {
                if rng.below(2) == 0 {
                    i += 1;
                    continue;
                }
                let j = i + rng.below((self.bg.len() - i) as u64) as usize;
                let (key, stage) = self.bg.swap_remove(j);
                if self.pool.slot_of.contains_key(&key) {
                    continue;
                }
                let region = self.region(key.0);
                let band = if stage { Band::Stage } else { Band::Main };
                match self.pool.pick_victim_any(region, self.global, band, layer, want, extra, key.0, false, false) {
                    Some(v) => {
                        // STAGING INVARIANT: a landing takes a slot of its own band.
                        assert_eq!(v >= self.pool.stage, stage, "landing of {key:?} (stage {stage}) at slot {v}");
                        self.st.evictions += u64::from(self.pool.evict(v, layer).is_some());
                        self.pool.land(v, key, stage);
                        self.st.bg_landed += 1;
                    }
                    None => {
                        if stage {
                            self.pool.sc.drops += 1;
                        } else if self.pool.pick_victim_any(region, self.global, band, layer, want, extra, key.0, true, false).is_some() {
                            self.pool.pins.c.no_victim_drops += 1;
                        }
                        self.st.bg_dropped += 1;
                    }
                }
            }
        }

        /// `ensure_layer_inner`: admit, then hit or claim every wanted id, then
        /// the demand reads land.
        fn ensure(&mut self, rng: &mut SimRng, layer: u32, sel: &[i32], extra: &[(u32, u32)], prefill: bool) {
            let want = Self::distinct(sel);
            self.admit(rng, layer, &want, extra);
            let region = self.region(layer);
            let mut claims = Vec::new();
            for &e in &want {
                if self.pool.touch_hit(layer, e, prefill) {
                    continue;
                }
                let sc0 = self.pool.sc;
                let (slot, ev) = self
                    .pool
                    .claim_miss(layer, e, &want, extra, region, self.global, prefill, prefill)
                    .expect("the reserve guarantees an unpinned victim");
                // STAGING INVARIANTS (with staging on): a prefill claim takes a
                // staging slot; a decode claim takes a main slot unless it
                // SPILLED (main exhausted: counted, and only ever while a
                // request is parked -- W alone never exhausts the reserve).
                if self.pool.stage_slots() > 0 {
                    let staged = slot >= self.pool.stage;
                    if prefill {
                        assert!(staged, "prefill claim L{layer} e{e} took main slot {slot}");
                        assert_eq!(self.pool.sc.spill_out, sc0.spill_out, "STAGE >= the union: no spill out");
                    } else if staged {
                        assert_eq!(self.pool.sc.spill_in, sc0.spill_in + 1, "decode claim L{layer} e{e} in staging without a spill");
                        assert!(!extra.is_empty(), "a decode claim spilled with nothing parked");
                    }
                }
                self.st.evictions += u64::from(ev.is_some());
                claims.push((e, slot));
            }
            for (e, slot) in claims {
                self.pool.commit(layer, e, slot);
            }
        }

        fn paged_bits(&self, layer: u32, sel: &[i32]) -> [u32; proto::RESID_WORDS] {
            let mut w = [0u32; proto::RESID_WORDS];
            for &e in sel {
                if e >= 0 && self.pool.remap_hosts[layer as usize][e as usize] == 0 {
                    w[e as usize / 32] |= 1 << (e % 32);
                }
            }
            w
        }

        /// A request's words at service: `pin_enable`, `pin_apply_words`, then
        /// the prefetch readers.
        fn words_in(&mut self, r: &SimReq) {
            self.pool.pins.enable(self.budget);
            for &w in &r.release {
                self.pool.pins.release(w);
            }
            for &w in &r.prefetch {
                self.pool.pins.grant(w >> 16, w & 0xFFFF);
            }
            self.queue_bg(&r.prefetch, false);
        }

        /// `pin_grant` for each decode-shaped request of a pass, then ONE
        /// `report` of the layer: `(map, epoch, pinned, budget)` for every
        /// reply of the pass.
        fn report_pass(&mut self, layer: u32, reqs: &[(&SimReq, bool)]) -> ([u32; proto::RESID_WORDS], u32, u32, u32) {
            for &(r, decode_shaped) in reqs {
                if decode_shaped {
                    for &e in &r.sel {
                        if e >= 0 {
                            self.pool.pins.grant(layer, e as u32);
                        }
                    }
                }
            }
            let row = self.pool.remap_hosts[layer as usize].clone();
            let map = self.pool.pins.report(layer, &row, self.pool.stage);
            let p = &self.pool.pins;
            (map, p.epoch, p.pinned, p.budget)
        }

        /// `pin_report` + the reply's pin block (a pass of one request).
        fn reply(&mut self, r: &SimReq, picks: &[i32], decode_shaped: bool, paged: [u32; proto::RESID_WORDS]) -> SimReply {
            if decode_shaped {
                for &e in picks {
                    if e >= 0 {
                        self.pool.pins.grant(r.layer, e as u32);
                    }
                }
            }
            let row = self.pool.remap_hosts[r.layer as usize].clone();
            let map = self.pool.pins.report(r.layer, &row, self.pool.stage);
            let p = &self.pool.pins;
            SimReply { seq: r.seq, layer: r.layer, map, epoch: p.epoch, pinned: p.pinned, budget: p.budget, paged }
        }

        /// Serve the wire's front request as `serve_connection` does: words in
        /// arrival order, a same-layer partner merged, early page for the next
        /// frame, or PARK (misses to the readers, queued servable requests
        /// served and answered inside with this one's picks pinned).
        fn serve(&mut self, rng: &mut SimRng, wire: &mut std::collections::VecDeque<SimReq>, replies: &mut Vec<SimReply>) {
            // `pull`: every frame on the wire is seen once as it arrives, and
            // its not-landed picks are noted for its reply's PAGED bits. The
            // sim notes them before this serve's admissions; the daemon may
            // first see a frame after them (`EarlyPaged` doc), so this is the
            // check at its most sensitive.
            for r in wire.iter_mut() {
                if !r.pulled {
                    r.pulled = true;
                    let bits = self.paged_bits(r.layer, &r.sel);
                    self.early.note(r.seq, bits);
                }
            }
            let Some(a) = wire.pop_front() else { return };
            self.words_in(&a);
            let partner = match wire.front() {
                Some(nb) if (self.scripted || rng.below(2) == 0) && nb.layer == a.layer && a.b > 4 && nb.b > 4 && a.b + nb.b <= 64 => wire.pop_front(),
                _ => None,
            };
            if let Some(b) = partner.as_ref() {
                self.words_in(b);
                self.st.merged += 1;
            }
            if let Some(nx) = wire.front() {
                if !self.scripted && rng.below(2) == 0 {
                    let w: Vec<u32> = Self::distinct(&nx.sel).into_iter().map(|e| (nx.layer << 16) | e).collect();
                    // The early-page hook: a prefill frame's reads are staged.
                    self.queue_bg(&w, nx.b > 16);
                }
            }
            let mut sel = a.sel.clone();
            if let Some(b) = partner.as_ref() {
                sel.extend_from_slice(&b.sel);
            }
            let bt = a.b + partner.as_ref().map_or(0, |b| b.b);
            let paged = self.paged_bits(a.layer, &sel);
            if !self.scripted && partner.is_none() && paged.iter().any(|&w| w != 0) && rng.below(3) == 0 {
                self.st.parked += 1;
                let parked: Vec<(u32, u32)> = Self::distinct(&sel).into_iter().map(|e| (a.layer, e)).collect();
                let w: Vec<u32> = parked.iter().filter(|k| !self.pool.slot_of.contains_key(k)).map(|k| (k.0 << 16) | k.1).collect();
                // The park hook: a parked prefill chunk's reads are staged.
                self.queue_bg(&w, a.b > 16);
                for _ in 0..1 + rng.below(3) {
                    if wire.front().is_some_and(|nx| nx.b <= PARK_MAX_ROWS) {
                        let c = wire.pop_front().unwrap();
                        self.words_in(&c);
                        let mut p = self.paged_bits(c.layer, &c.sel);
                        or_words(&mut p, &self.early.take(c.seq));
                        self.ensure(rng, c.layer, &c.sel, &parked, c.b > 16);
                        let r = self.reply(&c, &c.sel, c.b as u32 <= proto::PIN_DECODE_MAX_ROWS, p);
                        replies.push(r);
                        self.st.served_inside += 1;
                    }
                    self.admit(rng, a.layer, &[], &parked);
                }
            }
            if bt > 16 {
                self.st.prefill += 1;
            }
            self.ensure(rng, a.layer, &sel, &[], bt > 16);
            // Per REQUEST shape (`PIN_DECODE_MAX_ROWS`), as `serve_connection`.
            let dec = |r: &SimReq| r.b as u32 <= proto::PIN_DECODE_MAX_ROWS;
            // Per request, as `serve_connection`: the pass's bits are shared,
            // each request's arrival bits are its own.
            let ea = self.early.take(a.seq);
            let mut paged_a = paged;
            or_words(&mut paged_a, &ea);
            let mut paged_b = paged;
            if let Some(b) = partner.as_ref() {
                let eb = self.early.take(b.seq);
                or_words(&mut paged_b, &eb);
                // The partner held something that was missing when its lane
                // mate's frame arrived (and had landed by the pass): charged
                // to the mate only. Counted so the case is known to occur.
                if crate::het::b2_mirror::surprise_count(&b.held, &ea) > 0
                    && crate::het::b2_mirror::surprise_count(&b.held, &paged) == 0
                {
                    self.st.partner_split += 1;
                }
            }
            // ONE report per pass, as `serve_connection`: both requests'
            // grants first, then the layer's report, reused by the partner.
            if let Some(b) = partner.as_ref() {
                let (map, epoch, pinned, budget) = self.report_pass(a.layer, &[(&a, dec(&a)), (b, dec(b))]);
                replies.push(SimReply { seq: a.seq, layer: a.layer, map, epoch, pinned, budget, paged: paged_a });
                replies.push(SimReply { seq: b.seq, layer: b.layer, map, epoch, pinned, budget, paged: paged_b });
            } else {
                let r = self.reply(&a, &a.sel, dec(&a), paged_a);
                replies.push(r);
            }
        }
    }

    /// Review round 3's interleaving, scripted through the same sim: R(L) is
    /// served while A(L) waits with hot expert e not landed (A's arrival bit
    /// e set); R's pass lands e and its report pins it; the hub applies R's
    /// map and holds e; B(L) is submitted with e held and MERGES with A. B's
    /// reply must be scored against ITS OWN arrival (e was landed: no bit),
    /// not its lane mate's: with one shared word the hub would count a
    /// surprise that never happened (and panic under the assert knob).
    #[test]
    fn merged_partner_arrival_bits_are_per_request() {
        use crate::het::b2_mirror::{surprise_count, PinLedger};
        const PER: u32 = 8;
        let region_ids: Vec<u32> = (0..PER).collect();
        let regions: Vec<(u32, u32, &[u32])> = (0..2).map(|l| (l, l * PER, &region_ids[..])).collect();
        let mut b2 = Box2Sim {
            pool: ShardPool::seeded(2 * PER as usize, &regions, 0.0),
            bg: Vec::new(),
            budget: 4,
            per: PER,
            global: true,
            early: EarlyPaged::default(),
            scripted: true,
            st: PinSimStats::default(),
        };
        let mut rng = SimRng(7);
        let mut ledger = PinLedger::new();
        let e = 20i32;
        let row = |first: i32| -> Vec<i32> { vec![first, 0, 1, 2, 3, 4] };
        let req = |seq: u32, b: usize, sel: Vec<i32>, held: [u32; proto::RESID_WORDS]| SimReq {
            seq, layer: 0, b, sel, prefetch: Vec::new(), release: Vec::new(), held, pulled: false,
        };
        let none = [0u32; proto::RESID_WORDS];
        let r = req(1, 1, vec![e, 21, 22, 23, 24, 25], none);
        let a = req(2, 6, (0..6).flat_map(|_| row(e)).collect(), none);
        let mut wire: std::collections::VecDeque<SimReq> = [r, a].into_iter().collect();
        let mut replies = Vec::new();
        // R served alone (A is noted at pull with e missing, and left queued).
        b2.serve(&mut rng, &mut wire, &mut replies);
        assert_eq!((replies.len(), wire.len()), (1, 1));
        let rr = replies.remove(0);
        assert_eq!(rr.seq, 1);
        assert!(rr.map[0] & (1 << e) != 0, "R's pass landed e and its report pinned it");
        assert!(b2.pool.pins.is_pinned(0, e as u32));
        // The hub applies R's map: e is held from now on; B is submitted with it.
        ledger.apply_map(0, &rr.map, rr.epoch);
        assert!(ledger.held(0, e as u32));
        let mut held_b = none;
        held_b[0] |= 1 << e;
        let b = req(3, 6, (0..6).flat_map(|_| row(e)).collect(), held_b);
        wire.push_back(b);
        b2.serve(&mut rng, &mut wire, &mut replies);
        assert_eq!(b2.st.merged, 1, "A and B were served as one pass");
        assert_eq!(replies.len(), 2);
        let ra = replies.iter().find(|r| r.seq == 2).unwrap();
        let rb = replies.iter().find(|r| r.seq == 3).unwrap();
        assert!(ra.paged[0] & (1 << e) != 0, "A found e missing when its frame arrived");
        assert_eq!(rb.paged[0] & (1 << e), 0, "B did not: e had landed before B arrived");
        assert_eq!(surprise_count(&held_b, &rb.paged), 0, "no surprise for B");
        assert_eq!(surprise_count(&held_b, &ra.paged), 1, "one shared word would have charged B with A's miss");
        assert_eq!(b2.st.partner_split, 1);
    }

    /// One randomized run of the pin protocol (see `pin_protocol_randomized`).
    /// `forget_mask`: the hub applies every map as if it post-dated every
    /// release (the bug the epoch exists to prevent).
    fn pin_sim(seed: u64, forget_mask: bool) -> PinSimStats {
        use crate::het::b2_mirror::{surprise_count, PinLedger};
        const L: u32 = 6; // layers 0..L
        const RANGE: u32 = 32; // experts 0..RANGE per layer, Zipf-picked
        const PER: u32 = 20; // seeded slots per layer
        const N: usize = (L * PER) as usize;
        // The no-deadlock bound for this workload: a pass wants <= RANGE
        // experts of one layer, a parked request's picks are <= RANGE more,
        // so `R + STAGE >= 2 * RANGE`. Split as the production defaults are
        // (staging = one union, the rest as reserve): prefill claims never
        // spill out (STAGE >= RANGE), served-inside claims may spill IN under
        // a park (R < 2 * RANGE), nothing ever revokes.
        const STAGE: usize = RANGE as usize;
        const R: usize = RANGE as usize;
        const HEADROOM: u32 = 8;
        const MAX_IN_FLIGHT: usize = 6;
        let mut rng = SimRng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);
        // Zipf(1.1) over 0..RANGE by inverse CDF, hot ids permuted per layer.
        let cdf: Vec<f64> = {
            let w: Vec<f64> = (1..=RANGE).map(|k| 1.0 / (k as f64).powf(1.1)).collect();
            let s: f64 = w.iter().sum();
            let mut acc = 0.0;
            w.iter().map(|x| { acc += x / s; acc }).collect()
        };
        let zipf = |u: u64, l: u32| -> u32 {
            let x = (u % 1_000_000) as f64 / 1e6;
            let rank = cdf.iter().position(|&c| x < c).unwrap_or(RANGE as usize - 1) as u32;
            (rank * 7 + l * 5) % RANGE
        };
        let region_ids: Vec<u32> = (0..PER).collect();
        let regions: Vec<(u32, u32, &[u32])> = (0..L).map(|l| (l, l * PER, &region_ids[..])).collect();
        let mut pool = ShardPool::seeded(N, &regions, 0.0);
        assert_eq!(pool.set_stage(STAGE), STAGE);
        let mut b2 = Box2Sim {
            pool,
            bg: Vec::new(),
            budget: (N - STAGE - R) as u32,
            per: PER,
            global: seed % 2 == 1,
            early: EarlyPaged::default(),
            scripted: false,
            st: PinSimStats::default(),
        };
        let mut ledger = PinLedger::new();
        let mut wire: std::collections::VecDeque<SimReq> = Default::default();
        let mut replies: Vec<SimReply> = Vec::new();
        let mut sent: Vec<SimReq> = Vec::new();
        let (mut pin_active, mut seq) = (false, 0u32);
        let mut surprises = 0u32;
        let mut subset_violations = 0u64;
        let mut held_checked = 0u64;
        let mut stale_maps = 0u64;

        // held ⊆ pinned ⊆ landed, and the pinned count / budget.
        let check = |b2: &mut Box2Sim, ledger: &PinLedger, subset_violations: &mut u64| {
            let mut n_pinned = 0u32;
            for l in 0..L {
                for e in 0..RANGE {
                    let pinned = b2.pool.pins.is_pinned(l, e);
                    n_pinned += u32::from(pinned);
                    if ledger.held(l, e) && !pinned {
                        *subset_violations += 1;
                    }
                    let r = b2.pool.remap_hosts[l as usize][e as usize];
                    assert!(!pinned || r != 0, "pinned but not landed: L{l} e{e}");
                    // STAGING INVARIANT: pinned => landed in the MAIN band.
                    assert!(!pinned || ((-r - 1) as u32) < b2.pool.stage, "pinned in staging: L{l} e{e} slot {}", -r - 1);
                }
            }
            if b2.pool.pins.on {
                assert_eq!(n_pinned, b2.pool.pins.pinned, "pinned count drifted");
                assert!(b2.pool.pins.pinned <= b2.pool.pins.budget);
            }
            b2.st.max_pinned = b2.st.max_pinned.max(b2.pool.pins.pinned);
        };

        for _step in 0..300 {
            // Releases at step start, when nothing is routed but unsent.
            if pin_active {
                let _ = ledger.step(HEADROOM, 64, 512);
            }
            check(&mut b2, &ledger, &mut subset_violations);
            for layer in 0..L {
                // Two lanes, sometimes a third (`V41_MS_LANES=3`): a third
                // same-layer request can then queue behind an unserved one
                // and merge with it.
                let lanes = 2 + usize::from(rng.below(3) == 0);
                for _lane in 0..lanes {
                    let b = match rng.below(20) {
                        0..=12 => 1,
                        13..=14 => 2 + rng.below(3) as usize,
                        15..=17 => 6,
                        _ => 17 + rng.below(24) as usize,
                    };
                    let mut sel = Vec::with_capacity(b * N_EXPERT_USED);
                    for _ in 0..b {
                        let mut row: Vec<i32> = Vec::new();
                        while row.len() < N_EXPERT_USED {
                            let e = zipf(rng.below(u64::MAX), layer) as i32;
                            if !row.contains(&e) {
                                row.push(e);
                            }
                        }
                        sel.extend(row);
                    }
                    let mut held = [0u32; proto::RESID_WORDS];
                    for &e in &sel {
                        if ledger.held(layer, e as u32) {
                            held[e as usize / 32] |= 1 << (e % 32);
                        }
                        if b <= 16 {
                            ledger.note_pick(layer, e as u32);
                        }
                    }
                    held_checked += held.iter().map(|w| w.count_ones() as u64).sum::<u64>();
                    // Admission words for a few experts the hub does not hold.
                    let n_pf = rng.below(3);
                    let prefetch: Vec<u32> = (0..n_pf)
                        .map(|_| zipf(rng.below(u64::MAX), layer))
                        .filter(|&e| !ledger.held(layer, e))
                        .map(|e| (layer << 16) | e)
                        .collect();
                    let release = if pin_active { ledger.take_words(128) } else { Vec::new() };
                    seq += 1;
                    let r = SimReq { seq, layer, b, sel, prefetch, release, held, pulled: false };
                    sent.push(r.clone());
                    wire.push_back(r);
                    // Box 2 and the hub's reply consumption interleave at random;
                    // at most MAX_IN_FLIGHT requests stay unanswered.
                    loop {
                        let must = sent.len() > MAX_IN_FLIGHT;
                        let act = if must { 1 + rng.below(2) } else { rng.below(4) };
                        if act == 1 && !wire.is_empty() {
                            b2.serve(&mut rng, &mut wire, &mut replies);
                        } else if act == 2 && !replies.is_empty() {
                            let rp = replies.swap_remove(rng.below(replies.len() as u64) as usize);
                            let i = sent.iter().position(|q| q.seq == rp.seq).expect("a reply to a sent request");
                            let q = sent.swap_remove(i);
                            stale_maps += u64::from(rp.epoch != b2.pool.pins.epoch);
                            assert!(rp.pinned <= rp.budget);
                            ledger.note_reply(rp.epoch, rp.pinned, rp.budget);
                            let epoch = if forget_mask { u32::MAX / 2 } else { rp.epoch };
                            let _ = ledger.apply_map(rp.layer, &rp.map, epoch);
                            surprises += surprise_count(&q.held, &rp.paged);
                            pin_active = true;
                        } else if act == 0 && !must {
                            break;
                        }
                        check(&mut b2, &ledger, &mut subset_violations);
                    }
                }
            }
        }
        PinSimStats {
            c: b2.pool.pins.c,
            surprises,
            subset_violations,
            held_checked,
            stale_maps,
            sc: b2.pool.sc,
            ..b2.st
        }
    }

    /// The NTP estimator: a synthetic exchange with a KNOWN offset and a known
    /// one-way delay must be recovered exactly when the path is symmetric, and
    /// the error must be bounded by half the asymmetry when it is not.
    #[test]
    fn clock_sync_recovers_offset_and_delay() {
        let mk = |t1: u64, up: u64, srv: u64, down: u64, offset: i64| ClockSample {
            seq: 1,
            layer: 0,
            b: 1,
            t1,
            t2: (t1 as i64 + up as i64 + offset) as u64,
            t3: (t1 as i64 + up as i64 + srv as i64 + offset) as u64,
            t4: t1 + up + srv + down,
        };
        // Symmetric path: 45 us each way, 400 us of service, +12.345 ms offset.
        let s = mk(1_000_000, 45_000, 400_000, 45_000, 12_345_000);
        assert_eq!(s.offset_ns(), 12_345_000, "symmetric path recovers the offset exactly");
        assert_eq!(s.delay_ns(), 45_000);
        assert_eq!(s.remote_service_ns(), 400_000);
        assert_eq!(s.rtt_ns(), 490_000);
        // Asymmetric path (90 us up, 10 us down): the offset error is exactly
        // half the asymmetry — the reason the raw delay series is kept.
        let a = mk(1_000_000, 90_000, 400_000, 10_000, 12_345_000);
        assert_eq!(a.offset_ns() - 12_345_000, 40_000);
        assert_eq!(a.delay_ns(), 50_000, "delay is the mean of the two directions");

        // The windowed median rejects a single queued outlier.
        let mut cs = ClockSync::new(ClockPair::default(), 16, 0);
        for i in 0..15u64 {
            cs.push(mk(1_000_000 + i * 1_000_000, 45_000, 400_000, 45_000, 12_345_000));
        }
        cs.push(mk(99_000_000, 45_000, 400_000, 3_000_000, 12_345_000)); // one stalled reply
        assert_eq!(cs.offset_ns(), Some(12_345_000), "median ignores the outlier");
        assert_eq!(cs.len(), 16);
        let (min, p50, _p90, _p99, max) = cs.spread(|s| s.delay_ns()).unwrap();
        assert_eq!((min, p50), (45_000, 45_000));
        assert_eq!(max, 1_522_500, "the outlier is still visible in the raw series");
    }

    /// A ClockPair maps a monotonic stamp to realtime on the same box.
    #[test]
    fn clock_pair_maps_mono_to_realtime() {
        let p = ClockPair { mono_raw_ns: 1_000, realtime_ns: 1_700_000_000_000_000_000 };
        assert_eq!(p.realtime_for(1_000), 1_700_000_000_000_000_000);
        assert_eq!(p.realtime_for(2_000), 1_700_000_000_000_001_000);
        assert_eq!(p.realtime_for(0), 1_699_999_999_999_999_000);
        // A real sample: the two clocks must both be moving and correlated.
        let s = ClockPair::sample();
        assert!(s.mono_raw_ns > 0 && s.realtime_ns > 1_600_000_000_000_000_000);
        let t = ClockPair::sample();
        assert!(t.mono_raw_ns >= s.mono_raw_ns);
    }

    #[test]
    fn f16_matches_reference_rounding() {
        for &x in &[0.0f32, 1.0, -2.5, 65504.0, 1e-8, 3.14159, -0.1] {
            let bits = f32_to_f16_bits(x);
            // Halfway/RNE spot checks against known encodings.
            if x == 1.0 { assert_eq!(bits, 0x3c00); }
            if x == -2.5 { assert_eq!(bits, 0xc100); }
            if x == 65504.0 { assert_eq!(bits, 0x7bff); }
        }
    }
}
