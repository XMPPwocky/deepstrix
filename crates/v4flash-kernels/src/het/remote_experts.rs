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
    pub const VERSION: u16 = 1;
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

    /// Fixed request fields after the header (bytes):
    /// layer, b, flags, n_used, xq_bpt, reserved (6 × u32) then `t1` (u64,
    /// 8-aligned at offset 40). 16 + 32 = 48, so the Q8_K payload stays 8-aligned.
    pub const REQ_FIXED: usize = 32;
    /// Fixed response fields after the header (bytes): 8 × u32 then the clock
    /// triple `t1_echo, t2, t3` (u64 each, 8-aligned at 48/56/64).
    pub const RESP_FIXED: usize = 56;
    /// Byte offset of `t1` inside a REQUEST frame.
    pub const REQ_T1_OFF: usize = HDR_LEN + 24;
    /// Byte offsets of `t1_echo`/`t2`/`t3` inside a RESPONSE frame. The writer
    /// thread patches `t3` in place immediately before `write()`, which is the
    /// only way to stamp "just before the send" when compute and I/O are on
    /// different threads.
    pub const RESP_T1_OFF: usize = HDR_LEN + 32;
    pub const RESP_T2_OFF: usize = HDR_LEN + 40;
    pub const RESP_T3_OFF: usize = HDR_LEN + 48;

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
        if ew_off + n_sel * 4 != p.len() {
            return Err(eyre!(
                "request: frame len {} != expected {} (b={b}, xq_bpt={xq_bpt}, n_used={n_used})",
                p.len(),
                ew_off + n_sel * 4
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
        };
        let want = RESP_DATA_OFF + (m.b as usize) * (m.n_embd as usize) * (m.elem_bytes as usize);
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
    samples: Vec<ClockSample>,
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
            samples: Vec::new(),
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
        if self.capacity > 0 && self.samples.len() >= self.capacity {
            self.samples.remove(0);
        }
        self.samples.push(s);
    }

    pub fn samples(&self) -> &[ClockSample] {
        &self.samples
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
        let tail = &self.samples[self.samples.len().saturating_sub(self.window)..];
        Self::median(tail.iter().map(|s| s.delay_ns()).collect())
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

pub struct ExpertShard {
    #[allow(dead_code)]
    owner: V41HfWeights,
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
    stage: [PinnedBuffer<u8>; 3],
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
    /// eviction order over ABSOLUTE slots, front = least recently used.
    lru: std::collections::VecDeque<u32>,
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
}

/// Widen the victim search to the whole pool on decode-shaped requests
/// (`V41_B2_GLOBAL_POOL=0` keeps the per-layer search, which is byte-identical
/// to the pre-pool behaviour). Prefill-shaped requests are always
/// region-restricted regardless.
pub fn b2_global_pool() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_GLOBAL_POOL").map(|v| v != "0").unwrap_or(true)
    });
    *B
}

/// Fraction of its own region a layer is guaranteed to keep, even when a decode
/// request is evicting globally (`V41_B2_POOL_FLOOR`, default 0.90).
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
            .unwrap_or(0.90)
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
/// DEFAULT ON since 2026-09-15; `V41_B2_COALESCE=0` reverts.
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
pub fn b2_coalesce() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| {
        std::env::var("V41_B2_COALESCE").map(|v| v != "0").unwrap_or(true)
    });
    *B
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
        let stage = [
            // Under `V41_B2_COALESCE` staging is REPURPOSED: [0] holds the whole
            // 3-role weight run and [1] the 3-role scale run, so one pread fills
            // each. Sized from bpe3 (which already exceeds packed+scale per role)
            // so it cannot be too small: 3x covers the weight run, 1x the scales.
            PinnedBuffer::<u8>::new_with_flags(
                if b2_coalesce() { 3 * bpe3[0] } else { bpe3[0] } + 4 * 4096,
                HIP_HOST_MALLOC_NON_COHERENT,
            )?,
            PinnedBuffer::<u8>::new_with_flags(bpe3[1] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
            PinnedBuffer::<u8>::new_with_flags(bpe3[2] + 4 * 4096, HIP_HOST_MALLOC_NON_COHERENT)?,
        ];
        // O_DIRECT needs a 4096-aligned buffer. hipHostMalloc gives page-aligned
        // memory, but CHECK rather than assume: a misaligned buffer fails pread
        // with EINVAL, which is a confusing way to learn this.
        let aligned = stage.iter().all(|p| p.as_slice().as_ptr() as usize % 4096 == 0);
        let direct = repack.is_some() && b2_odirect() && aligned;
        if b2_odirect() && !aligned {
            eprintln!("expert shard: O_DIRECT off — pinned staging is not 4096-aligned");
        }
        eprintln!("expert shard: zero-copy O_DIRECT expert reads {}", if direct { "ON" } else { "OFF" });
        Ok(Self {
            owner,
            device: igpu,
            routed,
            layers,
            info,
            load_stats: LoadStats { n_experts: n_slots as usize, bytes, seconds },
            repack,
            repack_stream,
            stage,
            direct,
            pool: None,
        })
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
        let mut owner_of: Vec<Option<(u32, u32)>> = vec![None; n_slots];
        let mut slot_of = std::collections::HashMap::with_capacity(n_slots);
        let mut lru = std::collections::VecDeque::with_capacity(n_slots);
        let mut remap_hosts = vec![vec![0i32; REMAP_LEN]; N_LAYER as usize];
        for (li, l) in self.layers.iter().enumerate() {
            let Some(l) = l.as_ref() else { continue };
            if l.page.is_none() {
                continue;
            }
            let base = l.base_slot;
            for (local, &e) in l.ids.iter().enumerate() {
                let abs = base + local as u32;
                owner_of[abs as usize] = Some((li as u32, e));
                slot_of.insert((li as u32, e), abs);
                lru.push_back(abs);
                remap_hosts[li][e as usize] = -(abs as i32) - 1;
            }
        }
        let frac = b2_pool_floor();
        let mut held = vec![0u32; N_LAYER as usize];
        let mut floor = vec![0u32; N_LAYER as usize];
        for (li, l) in self.layers.iter().enumerate() {
            if let Some(l) = l.as_ref() {
                if l.page.is_some() {
                    held[li] = l.ids.len() as u32;
                    floor[li] = (l.ids.len() as f32 * frac) as u32;
                }
            }
        }
        eprintln!(
            "expert shard: global pool {} (floor {:.2} = {} slots on a 260-slot layer)",
            if b2_global_pool() { "ON" } else { "OFF" },
            frac,
            (260.0 * frac) as u32
        );
        self.pool = Some(ShardPool {
            owner_of,
            slot_of,
            lru,
            remap_hosts,
            dirty: vec![false; N_LAYER as usize],
            held,
            floor,
        });
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

    fn ensure_layer_inner(&mut self, layer: u32, ids: &[i32], mut missed: Option<&mut Vec<u32>>, prefill_shaped: bool) -> eyre::Result<()> {
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
        if pool.dirty[layer as usize] {
            l.remap_dev.copy_from_host(&pool.remap_hosts[layer as usize])?;
            pool.dirty[layer as usize] = false;
        }
        // Prefill sweeps a whole layer's union, so it must stay inside its own
        // region or one layer's sweep evicts another's. Decode may roam.
        let global = b2_global_pool() && !prefill_shaped;
        let r = &mut self.routed;
        // Disjoint field borrows, hoisted: the per-role read closures below must
        // capture `owner` alone, not `&self`, or they collide with `&mut stage`.
        let owner = &self.owner;
        let repack = self.repack.as_ref();
        let repack_stream = self.repack_stream.as_ref();
        let stage = &mut self.stage;
        let direct = self.direct;
        let mut dirty = false;
        let mut want: Vec<u32> = Vec::with_capacity(ids.len());
        for &e in ids {
            if !(0..N_EXPERT as i32).contains(&e) { continue; }
            let e = e as u32;
            if !want.contains(&e) { want.push(e); }
        }
        for &e in &want {
            pg.requests += 1;
            if let Some(&slot) = pool.slot_of.get(&(layer, e)) {
                if let Some(p) = pool.lru.iter().position(|&s| s == slot) { pool.lru.remove(p); }
                pool.lru.push_back(slot);
                continue;
            }
            pg.misses += 1;
            if let Some(m) = missed.as_deref_mut() {
                m.push(e);
            }
            // Victim, least-recently-used first. Never a slot holding an id we are
            // about to need on THIS layer in THIS call. Region-restricted unless
            // the request is decode-shaped and the global pool is enabled.
            let lo = base as u32;
            let hi = lo + n_region as u32;
            let victim = pool
                .lru
                .iter()
                .copied()
                .find(|&sl| {
                    if !global && !(lo..hi).contains(&sl) {
                        return false;
                    }
                    match pool.owner_of[sl as usize] {
                        Some((ol, oe)) => {
                            if ol == layer && want.contains(&oe) {
                                return false;
                            }
                            // Never take a foreign layer below its floor.
                            ol == layer || pool.held[ol as usize] > pool.floor[ol as usize]
                        }
                        None => true,
                    }
                })
                .ok_or_else(|| eyre!("expert shard: layer {layer} has no evictable slot"))?;
            if let Some(p) = pool.lru.iter().position(|&s| s == victim) { pool.lru.remove(p); }
            // Detach from whoever held it — possibly a DIFFERENT layer, whose
            // device remap is then stale until its next `ensure_layer`.
            if let Some((ol, oe)) = pool.owner_of[victim as usize].take() {
                pool.slot_of.remove(&(ol, oe));
                pool.remap_hosts[ol as usize][oe as usize] = 0;
                pool.held[ol as usize] -= 1;
                if ol != layer {
                    pool.dirty[ol as usize] = true;
                }
            }
            let names = [
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
                format!("blk.{layer}.ffn_down_exps.weight"),
            ];
            let bpe = [r.gate_bytes_per_expert, r.up_bytes_per_expert, r.down_bytes_per_expert];
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
            let (mut read_ns, mut h2d_ns) = (0u64, 0u64);
            let gpu_repack = repack.is_some();
            let rp0 = v4flash_core::hf_v41::expert_read_profile();
            let t_r = std::time::Instant::now();
            // Per role: where the packed nibbles and the scale plane actually
            // landed in staging. `None` = contiguous (the non-direct paths).
            let mut offs: [Option<(usize, usize, u32, u32)>; 3] = [None; 3];
            // COALESCED: two preads for all three roles. Falls through to the
            // per-role path when disabled, when this build has no GPU repack /
            // O_DIRECT, or when the run-time contiguity check fails.
            let mut coalesced = false;
            if gpu_repack && direct && b2_coalesce() {
                let [p0, p1, _] = &mut *stage;
                let (dw, ds) = (p0.as_mut_slice(), p1.as_mut_slice());
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
            if !coalesced {
                let [p0, p1, p2] = &mut *stage;
                let bufs: [&mut [u8]; 3] =
                    [p0.as_mut_slice(), p1.as_mut_slice(), p2.as_mut_slice()];
                let mut errs: Vec<String> = Vec::new();
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
                                let t = src.tensor(&name).ok_or(format!("missing {name}"))?;
                                if gpu_repack && direct {
                                    // Zero-copy: O_DIRECT lands each region at its
                                    // own 4096-residue, straight into GTT staging.
                                    if let Some(o) = src
                                        .read_expert_hf_layout_direct(t, e as usize, buf)
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
                if let Some(msg) = errs.first() {
                    return Err(eyre!("expert shard: layer {layer} expert {e}: {msg}"));
                }
            }
            read_ns += t_r.elapsed().as_nanos() as u64;
            let rp1 = v4flash_core::hf_v41::expert_read_profile();
            pg.pread_ns += rp1.2 - rp0.2;
            pg.repack_cpu_ns += rp1.3 - rp0.3;
            let t_h = std::time::Instant::now();
            match (repack, repack_stream) {
                (Some(rp), Some(st)) => {
                    pg.repack_gpu_ns += Self::repack_in_place(
                        rp, st, r, victim, stage, &offs, coalesced,
                    )?;
                }
                _ => {
                    for i in 0..3 {
                        let buf = match i { 0 => &mut r.gate.buffer, 1 => &mut r.up.buffer, _ => &mut r.down.buffer };
                        buf.slice_view_mut(victim as usize * bpe[i], bpe[i])
                            .copy_from_host(stage[i].as_slice())?;
                    }
                }
            }
            h2d_ns += t_h.elapsed().as_nanos() as u64;
            pg.read_ns += read_ns;
            pg.h2d_ns += h2d_ns;
            pool.owner_of[victim as usize] = Some((layer, e));
            pool.slot_of.insert((layer, e), victim);
            pool.held[layer as usize] += 1;
            pool.lru.push_back(victim);
            pool.remap_hosts[layer as usize][e as usize] = -(victim as i32) - 1;
            dirty = true;
        }
        if dirty {
            l.remap_dev.copy_from_host(&pool.remap_hosts[layer as usize])?;
            pool.dirty[layer as usize] = false;
        }
        Ok(())
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
    pub n_work_items: u32,
    /// Bitmask over the request's first `RESP_MISS_BITS` sel slots: bit i set =
    /// pick i had to be paged from this box's disk. Decode (b==1) only; see
    /// `proto::RESP_MISS_SHIFT`.
    pub miss_mask: u32,
}

pub struct MoeExecutor {
    pub engine: DeviceEngine,
    device: Device,
    rows: usize,
    decode_max_b: usize,
    /// Reused across requests so the decode miss report never allocates.
    missed_scratch: Vec<u32>,
    xq: DeviceBuffer<u8>,
    d_selected: DeviceBuffer<i32>,
    d_ew: DeviceBuffer<f32>,
    group_count: DeviceBuffer<i32>,
    expert_members: DeviceBuffer<i32>,
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
        Ok(Self {
            missed_scratch: Vec::with_capacity(N_EXPERT_USED),
            engine,
            device: igpu,
            rows,
            decode_max_b,
            xq: DeviceBuffer::new(id, rows * XQ_BYTES_PER_TOKEN)?,
            d_selected: DeviceBuffer::new(id, rows * nu)?,
            d_ew: DeviceBuffer::new(id, rows * nu)?,
            group_count: DeviceBuffer::new(id, N_EXPERT as usize)?,
            expert_members: DeviceBuffer::new(id, N_EXPERT as usize * rows)?,
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
        })
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
        self.run_path(shard, layer, b, xq, sel, ew, false)
    }

    /// As [`Self::run`]; `force_batched` takes the by-expert chain regardless of `b`.
    #[allow(clippy::too_many_arguments)]
    pub fn run_path(
        &mut self,
        shard: &mut ExpertShard,
        layer: u32,
        b: usize,
        xq: &[u8],
        sel: &[i32],
        ew: &[f32],
        force_batched: bool,
    ) -> eyre::Result<ExecTiming> {
        let nu = N_EXPERT_USED;
        if b == 0 || b > self.rows {
            return Err(eyre!("executor: b={b} outside 1..={}", self.rows));
        }
        if xq.len() != b * XQ_BYTES_PER_TOKEN || sel.len() != b * nu || ew.len() != b * nu {
            return Err(eyre!("executor: payload sizes do not match b={b}"));
        }
        // Catch-all tier: make every requested expert resident first. A paged
        // shard's `remap[e]` is 0 ("the other device takes it") until it is,
        // which the kernel would silently honour and drop the expert.
        // Report the misses back to the hub for b==1 (decode). The hub uses them to
        // keep box 1's decode residency EXCLUSIVE of box 2's -- see
        // `ExpertShard::ensure_layer_reporting`. Prefill batches skip it: their sel
        // is up to 1024x6 and nothing consumes the mask.
        let mut miss_mask = 0u32;
        if b == 1 {
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
        let (gate, up, down, remap) = shard.layer_views(layer)?;
        for (i, &e) in sel.iter().enumerate() {
            if e == NO_PICK {
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
        let t0 = Instant::now();
        // Uploads: the caller's slices are ordinary (unpinned) host memory, so
        // these are synchronous memcpys into GTT.
        self.xq.slice_view_mut(0, xq.len()).copy_from_host(xq)?;
        self.d_selected.slice_view_mut(0, b * nu).copy_from_host(&self.sel_host[..b * nu])?;
        self.d_ew.slice_view_mut(0, b * nu).copy_from_host(&self.ew_host[..b * nu])?;
        let t1 = Instant::now();
        if let Some((a, _)) = self.ev.as_ref() {
            a.record(&self.engine.compute)?;
        }
        let e = &self.engine;
        let s = &e.compute;
        let cap = N_EXPERT_USED as u32;
        let gbpe = shard.routed.gate_bytes_per_expert as u32;
        let ubpe = shard.routed.up_bytes_per_expert as u32;
        let dbpe = shard.routed.down_bytes_per_expert as u32;
        let gdt = shard.routed.gate.dtype;
        let ddt = shard.routed.down.dtype;
        let mut timing = ExecTiming { path_decode: b <= self.decode_max_b && !force_batched, ..Default::default() };
        timing.miss_mask = miss_mask;
        if timing.path_decode {
            // Decode kernels, one token at a time: q8k(x) is already done on the
            // hub (the wire carries Q8_K), so 3 launches per token.
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
            // hetsplit group builder → work items (host readback) → kwide
            // gate/up → q8k(mid) → by-expert kwide2 down → hetsplit reduce.
            let bu = b as u32;
            let max_per_expert = self.rows as u32;
            let sel_v = self.d_selected.slice_view(0, b * nu);
            let ew_v = self.d_ew.slice_view(0, b * nu);
            let xq_v = self.xq.slice_view(0, b * XQ_BYTES_PER_TOKEN);
            self.group_count.fill_zero_async(s)?;
            e.moe_group_builder.launch_hetsplit(
                s, &mut self.group_count, &mut self.expert_members, &sel_v, remap, 0, cap, bu,
                nu as u32, N_EXPERT, max_per_expert,
            )?;
            self.n_work_items.fill_zero_async(s)?;
            let max_items = self.work_items.len() as u32;
            e.moe_group_builder.launch_work_items(
                s, &mut self.work_items, &mut self.n_work_items, &self.group_count, N_EXPERT, CHUNK_SIZE, max_items,
            )?;
            s.synchronize()?;
            let mut n_wi = [0i32; 1];
            self.n_work_items.copy_to_host(&mut n_wi)?;
            let n_wi = n_wi[0] as u32;
            timing.n_work_items = n_wi;
            let mut mid_v = self.d_mid_cat.slice_view_mut(0, b * nu * N_FF_EXP as usize);
            let handled = super::dispatch::moe_gate_up_chunked(
                e, gdt, s, &mut mid_v, &gate, &up, &xq_v, &ew_v, &self.group_count, &self.expert_members,
                &self.work_items, n_wi, gbpe, ubpe, nu as u32, max_per_expert, CHUNK_SIZE, SWIGLU_CLAMP_EXP,
                N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            )?;
            if !handled {
                return Err(eyre!("executor: no prefill gate/up kernel for {gdt:?}"));
            }
            let mut midq_v = self.d_midq_cat.slice_view_mut(0, b * nu * MIDQ_BYTES_PER_SLOT);
            e.q8k.launch(s, &mut midq_v, &mid_v, BLOCKS_Q8K_DOWN_IN * nu as u32 * bu)?;
            let mut part_v = self.partials.slice_view_mut(0, b * nu * N_EMBD as usize);
            match ddt {
                GgufType::MXFP4 => e.mxfp4.launch_by_expert_kwide2(
                    s, &mut part_v, &down, &midq_v, &self.group_count, &self.expert_members, &self.work_items,
                    n_wi, dbpe, MIDQ_BYTES_PER_SLOT as u32, nu as u32, max_per_expert, CHUNK_SIZE, N_EMBD,
                    BLOCKS_Q8K_DOWN_IN,
                )?,
                GgufType::IQ3_XXS => e.iq3.launch_by_expert_kwide2(
                    s, &mut part_v, &down, &midq_v, &self.group_count, &self.expert_members, &self.work_items,
                    n_wi, dbpe, MIDQ_BYTES_PER_SLOT as u32, nu as u32, max_per_expert, CHUNK_SIZE, N_EMBD,
                    BLOCKS_Q8K_DOWN_IN,
                )?,
                other => return Err(eyre!("executor: no prefill down kernel for {other:?}")),
            }
            let mut out_v = self.ffn_moe.slice_view_mut(0, b * N_EMBD as usize);
            e.q2k.launch_reduce_partials_hetsplit(
                s, &mut out_v, &part_v, &sel_v, remap, 0, cap, nu as u32, N_EMBD, bu,
            )?;
        }
        if let Some((_, b)) = self.ev.as_ref() {
            b.record(s)?;
        }
        s.synchronize()?;
        timing.h2d = t1 - t0;
        timing.gpu = t1.elapsed();
        Ok(timing)
    }

    /// Copy the last result's f32 rows to host.
    pub fn read_f32(&self, b: usize, dst: &mut [f32]) -> eyre::Result<()> {
        self.device.set_current()?;
        self.ffn_moe.slice_view(0, b * N_EMBD as usize).copy_to_host(dst)
    }

    /// Cast the last result to f16 on the device (`f32_to_f16_cast`, RNE) and
    /// copy to host.
    pub fn read_f16(&mut self, b: usize, dst: &mut [u16]) -> eyre::Result<()> {
        self.device.set_current()?;
        let n = b * N_EMBD as usize;
        let src = self.ffn_moe.slice_view(0, n);
        let mut o = self.out16.slice_view_mut(0, n);
        self.engine.q8k.launch_cast_f16(&self.engine.compute, &mut o, &src, n as u32)?;
        self.engine.compute.synchronize()?;
        o.copy_to_host(dst)
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
        Ok(Self { exporter, igpu_compute: std::sync::Mutex::new(igpu_compute), machine: machine.into() })
    }

    pub fn machine(&self) -> &str {
        &self.machine
    }

    /// Host-time span on a track (ns are CLOCK_REALTIME, `super::perfetto::host_now_ns`).
    pub fn span(&self, uuid: u64, name: &str, start_ns: u64, end_ns: u64) {
        let _ = self.exporter.emit_span(uuid, name, start_ns, end_ns);
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
            Ok(records) => {
                eprintln!("expertd: {peer} closed after {} requests", records.len());
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
) -> eyre::Result<Vec<RequestRecord>> {
    apply_socket_options(&stream, &opts.socket)?;
    let max_payload = proto::REQ_FIXED + exec.rows() * (XQ_BYTES_PER_TOKEN + 8 * N_EXPERT_USED) + 64;
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
                if buf.len() >= proto::RESP_T3_OFF + 8
                    && matches!(proto::parse_header(buf.as_bytes()), Ok(h) if h.kind == proto::KIND_RESPONSE)
                {
                    proto::patch_u64(&mut buf, proto::RESP_T3_OFF, monotonic_raw_ns());
                }
                if let Err(e) = wr.write_all(buf.as_bytes()) {
                    eprintln!("expertd: write error: {e}");
                    return;
                }
                let _ = tx_written.send((seq, Instant::now()));
                let _ = tx_resp_recycle.send(buf);
            }
        });
        // Compute loop (this thread owns the GPU). Runs in a closure so the
        // channel senders can be dropped and the socket shut down BEFORE the
        // scope joins the reader/writer threads (the writer ends when every
        // `tx_out` sender is gone; the reader when its blocking read fails).
        let mut compute = |tx_out: mpsc::SyncSender<(AlignedBuf, Instant, u32)>, records: &mut Vec<RequestRecord>| -> eyre::Result<()> {
        let mut n_done = 0usize;
        loop {
            let msg = if opts.keep_warm_us == 0 {
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
            let t_start = Instant::now();
            let t_start_rt = tracer.map(|_| super::perfetto::host_now_ns());
            let mut resp = rx_resp_recycle.try_recv().unwrap_or_else(|_| AlignedBuf::with_capacity(proto::RESP_DATA_OFF + exec.rows() * N_EMBD as usize * 4));
            if hdr.kind != proto::KIND_REQUEST {
                proto::encode_error(&mut resp, hdr.seq, 2, &format!("unexpected frame kind {}", hdr.kind));
                let _ = tx_out.send((resp, Instant::now(), hdr.seq));
                return Err(eyre!("unexpected frame kind {}", hdr.kind));
            }
            let outcome: eyre::Result<(RequestRecord, u32)> = (|| {
                let req = proto::decode_request(&buf)?;
                if req.n_used != N_EXPERT_USED as u32 || req.xq_bpt != XQ_BYTES_PER_TOKEN as u32 {
                    return Err(eyre!("request geometry n_used={} xq_bpt={} != {}/{}", req.n_used, req.xq_bpt, N_EXPERT_USED, XQ_BYTES_PER_TOKEN));
                }
                let b = req.b as usize;
                let timing = exec.run_path(shard, req.layer, b, req.xq, req.sel, req.ew, req.flags & proto::REQ_FLAG_BATCHED != 0)?;
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
                    exec.read_f32(b, resp.view_mut::<f32>(proto::RESP_DATA_OFF, n))?;
                } else {
                    exec.read_f16(b, resp.view_mut::<u16>(proto::RESP_DATA_OFF, n))?;
                }
                proto::patch_len(&mut resp);
                let t_ready = Instant::now();
                // t_server = frame complete → response handed to the writer.
                let t_server_us = (t_ready - t_done).as_micros() as u32;
                resp.as_bytes_mut()[proto::HDR_LEN + 20..proto::HDR_LEN + 24].copy_from_slice(&t_server_us.to_le_bytes());
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
                Ok((rec, hdr.seq))
            })();
            let _ = tx_req_recycle.send(buf);
            match outcome {
                Ok((rec, seq)) => {
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
                    records.push(rec);
                    if tx_out.send((resp, rec.t_ready, seq)).is_err() {
                        return Err(eyre!("writer thread gone"));
                    }
                    n_done += 1;
                    // Catch-all tier: box 2 now owns ALL the paging, so its miss
                    // rate is the number that matters and the hub cannot see it.
                    // One line per 2000 requests (= per ~50 tokens at 40 layers).
                    if shard.is_paged() && n_done % 2000 == 0 {
                        let (req, miss, read_ns, h2d_ns) = shard.page_stats();
                        if miss > 0 {
                            let (pread_ns, rcpu_ns, rgpu_ns) = shard.page_read_split();
                            let per = |ns: u64| ns as f64 / miss as f64 / 1e6;
                            eprintln!(
                                "expertd: page stats requests={req} misses={miss} hit={:.4} \
ms_per_miss={:.2} (read {:.2} [pread {:.2} repack_cpu {:.2}] h2d {:.2} repack_gpu {:.2})",
                                1.0 - miss as f64 / req.max(1) as f64,
                                (read_ns + h2d_ns) as f64 / miss as f64 / 1e6,
                                per(read_ns), per(pread_ns), per(rcpu_ns),
                                per(h2d_ns), per(rgpu_ns),
                            );
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
        let res = compute(tx_out, &mut records);
        // Senders are gone (tx_out moved into `compute`); unblock the reader
        // (it may sit in read_exact) so the scope can join both threads.
        let _ = stream.shutdown(std::net::Shutdown::Both);
        res
    });
    // The writer has exited (scope joined it): fill in write completion times.
    let mut by_seq: std::collections::HashMap<u32, usize> =
        records.iter().enumerate().map(|(i, r)| (r.seq, i)).collect();
    while let Ok((wseq, t_w)) = rx_written.try_recv() {
        if let Some(i) = by_seq.remove(&wseq) {
            records[i].write_us = (t_w - records[i].t_ready).as_micros().min(u32::MAX as u128) as u32;
        }
    }
    result?;
    Ok(records)
}

// ---------------------------------------------------------------------------
// Hub-side client
// ---------------------------------------------------------------------------

/// Handle of a request in flight. Responses arrive in submission order.
#[derive(Clone, Copy, Debug)]
pub struct Ticket {
    pub seq: u32,
    pub layer: u32,
    pub b: u32,
    pub bytes_out: usize,
    pub t_submit: Instant,
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
    pub t_remote_server_us: u32,
    pub bytes_in: usize,
    pub bytes_out: usize,
    /// This exchange's clock quadruple (`None` if a peer did not stamp).
    pub clock: Option<ClockSample>,
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
    pub fn bytes(&self) -> &[u8] {
        &self.frame.as_bytes()[proto::RESP_DATA_OFF..]
    }
    /// Link time = round trip minus the daemon's own frame→response time.
    pub fn link_us(&self) -> u32 {
        self.rtt_us.saturating_sub(self.t_remote_server_us)
    }
}

enum ClientInbound {
    Resp { buf: AlignedBuf, t_recv: Instant, t4: u64 },
    Err(String),
}

pub struct RemoteExpertClient {
    info: ShardInfo,
    stream: TcpStream,
    tx_req: Option<mpsc::SyncSender<AlignedBuf>>,
    rx_resp: mpsc::Receiver<ClientInbound>,
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
    pub fn connect(addr: impl ToSocketAddrs, opts: &SocketOptions) -> eyre::Result<Self> {
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
        let (tx_req, rx_req) = mpsc::sync_channel::<AlignedBuf>(16);
        let (tx_req_recycle, rx_req_recycle) = mpsc::channel::<AlignedBuf>();
        let (tx_resp, rx_resp) = mpsc::sync_channel::<ClientInbound>(16);
        let (tx_resp_recycle, rx_resp_recycle) = mpsc::channel::<AlignedBuf>();
        let writer = std::thread::Builder::new().name("rexp-writer".into()).spawn(move || {
            for mut buf in rx_req {
                // t1 immediately before write(), so frame encoding is outside the sample.
                if buf.len() >= proto::REQ_T1_OFF + 8 {
                    proto::patch_u64(&mut buf, proto::REQ_T1_OFF, monotonic_raw_ns());
                }
                if let Err(e) = wr.write_all(buf.as_bytes()) {
                    eprintln!("remote_experts: write error: {e}");
                    return;
                }
                let _ = tx_req_recycle.send(buf);
            }
        })?;
        let max_resp = proto::RESP_DATA_OFF + info.max_batch as usize * N_EMBD as usize * 4;
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
        Ok(Self {
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
        self.submit_inner(layer, b, xq, sel, ew, flags, false)
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
    #[allow(clippy::too_many_arguments)]
    pub fn submit_dispatch(&mut self, unmasked: bool, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool) -> eyre::Result<Option<Ticket>> {
        if unmasked {
            self.submit_unmasked(layer, b, xq, sel, ew, resp_f32)
        } else {
            self.submit(layer, b, xq, sel, ew, resp_f32)
        }
    }

    pub fn submit(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], resp_f32: bool) -> eyre::Result<Option<Ticket>> {
        self.submit_flags(layer, b, xq, sel, ew, if resp_f32 { proto::REQ_FLAG_RESP_F32 } else { 0 })
    }

    /// As [`Self::submit`] with explicit `proto::REQ_FLAG_*` bits.
    pub fn submit_flags(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32) -> eyre::Result<Option<Ticket>> {
        self.submit_inner(layer, b, xq, sel, ew, flags, true)
    }

    /// [`Self::submit_flags`] without the advertised-ownership mask — i.e. what
    /// the hub does under T2 catch-all, where "not resident on box 1" is the
    /// routing rule and box 2 pages anything it is handed. Needed by the bench to
    /// exercise the miss path at all: masked submits can only ever request
    /// resident experts, so they never fault.
    pub fn submit_flags_unmasked(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32) -> eyre::Result<Option<Ticket>> {
        self.submit_inner(layer, b, xq, sel, ew, flags, false)
    }

    fn submit_inner(&mut self, layer: u32, b: usize, xq: &[u8], sel: &[i32], ew: &[f32], flags: u32, mask: bool) -> eyre::Result<Option<Ticket>> {
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
        proto::encode_request(
            &mut buf, seq, layer, b as u32, flags, nu as u32, XQ_BYTES_PER_TOKEN as u32, xq,
            &self.sel_scratch[..b * nu], &self.ew_scratch[..b * nu],
        );
        let ticket = Ticket { seq, layer, b: b as u32, bytes_out: buf.len(), t_submit: Instant::now() };
        self.tx_req.as_ref().ok_or_else(|| eyre!("client closed"))?.send(buf).map_err(|_| eyre!("writer thread gone"))?;
        self.in_flight.push_back(ticket);
        Ok(Some(ticket))
    }

    /// Block until the oldest in-flight request has answered. `ticket` must be
    /// that request (FIFO).
    pub fn wait(&mut self, ticket: Ticket) -> eyre::Result<RemotePartial> {
        let head = self.in_flight.pop_front().ok_or_else(|| eyre!("wait: nothing in flight"))?;
        if head.seq != ticket.seq {
            return Err(eyre!("wait: ticket seq {} but oldest in flight is {}", ticket.seq, head.seq));
        }
        let (buf, t_recv, t4) = match self.rx_resp.recv() {
            Ok(ClientInbound::Resp { buf, t_recv, t4 }) => (buf, t_recv, t4),
            Ok(ClientInbound::Err(e)) => return Err(eyre!("remote connection: {e}")),
            Err(_) => return Err(eyre!("reader thread gone")),
        };
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
        // NTP quadruple for this exchange. t1 is what the WRITER stamped (echoed
        // back by the daemon), not the submit time, so encoding is excluded.
        let sample = ClockSample { seq: h.seq, layer: m.layer, b: m.b, t1: m.t1, t2: m.t2, t3: m.t3, t4 };
        let valid = m.t1 != 0 && m.t2 != 0 && m.t3 != 0 && t4 != 0;
        if valid {
            self.clock.push(sample);
        }
        Ok(RemotePartial {
            layer: m.layer,
            b: m.b,
            is_f32: m.elem_bytes == 4,
            miss_mask: (m.flags & proto::RESP_MISS_MASK) >> proto::RESP_MISS_SHIFT,
            rtt_us: (t_recv - ticket.t_submit).as_micros().min(u32::MAX as u128) as u32,
            t_remote_compute_us: m.t_compute_us,
            t_remote_server_us: m.t_server_us,
            bytes_in: buf.len(),
            bytes_out: ticket.bytes_out,
            clock: valid.then_some(sample),
            frame: buf,
        })
    }

    /// Hand a consumed partial's buffer back for reuse.
    pub fn recycle(&self, p: RemotePartial) {
        let _ = self.tx_resp_recycle.send(p.frame);
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
        proto::encode_request(&mut buf, 42, 17, b as u32, proto::REQ_FLAG_RESP_F32, nu as u32, XQ_BYTES_PER_TOKEN as u32, &xq, &sel, &ew);
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
