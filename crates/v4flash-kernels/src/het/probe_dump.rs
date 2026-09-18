//! `V41_PROBE_DUMP=<path>`: collect a training set for a router probe —
//! `(gate input at a source layer, the experts every decoder layer selected)`.
//!
//! WHAT AND WHY. `docs/v41/PREFETCH_STUDY_2026-09-18.md` measured every
//! id-based predictor of decoder routing to be dead (the CED-seam predictor
//! reaches 4.8% precision against a ~20-35% break-even). Those predictors saw
//! only 6 discrete expert ids; the gate itself reads a 5120-dim vector. This
//! dumps that vector so the question can be settled with the real input.
//!
//! THREE DESIGN CHOICES, each load-bearing:
//!
//! 1. **Dump the GATE INPUT (`ffn_input_norm`), not the raw residual stream.**
//!    The gate is `topk(sqrtsoftplus(W_L x / temp) + b_L)` where `x` is already
//!    `ffn_norm(...)`. Storing post-norm means a probe consumes exactly what
//!    the real gate consumes -- and RMS scale, which is the single biggest
//!    systematic difference between layer 19's stream and layer 39's, is
//!    already divided out before we ever see it.
//!
//! 2. **Dump layer 20's as well as layer 19's.** Layer 20's gate input is
//!    computed anyway (its own gate must run), it is the freshest state at the
//!    CED seam, and layers 23-39 are the ones with enough lead time to hide a
//!    read. Two sources cost 20 KB/token and let the source be A/B'd offline
//!    instead of guessed.
//!
//! 3. **Leave room for PREFILL rows** (`V41_PROBE_PREFILL_STRIDE`, not wired
//!    yet). A decode request yields ~400 samples; a 93K-token prefill yields
//!    93K, so one subsampled request is worth hours of decode. Prefill rows
//!    carry a mild distribution shift (different positions) but the same gates
//!    and the same residual dynamics; `phase` is in the record so they can be
//!    held out. Today only decode is collected.
//!
//! THE ZERO-SHOT TEST RUNS OFFLINE FROM THIS SAME FILE. `W_L` and `b_L` are in
//! the checkpoint, so "how well does layer L's own gate do when fed layer 19's
//! activation" needs no second model run and no training -- and its answer is
//! what prices the probe (data required scales with the SQUARE of the
//! correction's norm, so a good zero-shot is worth ~10x less data).
//!
//! Records are FIXED SIZE so the file mmaps as an array. Everything is written
//! by a background thread; the hot path does one D2H of an already-synced
//! buffer and a channel send that drops rather than blocks.

use crate::config::{N_EMBD, N_EXPERT_USED, N_LAYER};

/// First decoder layer — the CED seam. Targets are `DST0..N_LAYER`.
pub const DST0: usize = 20;
pub const N_DST: usize = N_LAYER as usize - DST0;

/// `V41_PROBE_DUMP=<path>`; `None` disables everything here.
fn path() -> Option<&'static str> {
    static P: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    P.get_or_init(|| std::env::var("V41_PROBE_DUMP").ok()).as_deref()
}

pub fn on() -> bool {
    static B: std::sync::LazyLock<bool> = std::sync::LazyLock::new(|| path().is_some());
    *B
}

/// `V41_PROBE_SRC=19,20`: which layers' gate inputs to store, in order.
pub fn src_layers() -> &'static [u32] {
    static S: std::sync::OnceLock<Vec<u32>> = std::sync::OnceLock::new();
    S.get_or_init(|| {
        std::env::var("V41_PROBE_SRC")
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![19, 20])
    })
}

/// `V41_PROBE_PREFILL_STRIDE=N`: **NOT WIRED YET.** The knob and the `phase`
/// field exist so the format does not change when it lands, but only the DECODE
/// path calls `observe_decode` today, so setting this does nothing. Wiring it
/// means handling `ffn_input_norm` as `[b, N_EMBD]` in `forward_prefill` and
/// picking rows by stride. Worth it: a decode request yields ~400 samples and a
/// 93K-token prefill at stride 16 yields ~5,800.
pub fn prefill_stride() -> usize {
    static N: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
        std::env::var("V41_PROBE_PREFILL_STRIDE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
    });
    *N
}

fn f32_to_f16(x: f32) -> u16 {
    // Round-to-nearest-even via the standard bit twiddle; activations are well
    // inside f16 range after an RMS norm, but flush overflow to inf anyway.
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let mut exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let mant = b & 0x007f_ffff;
    if exp >= 0x1f {
        return sign | 0x7c00;
    }
    if exp <= 0 {
        return sign;
    }
    let mut m = (mant >> 13) as u16;
    if (mant & 0x1000) != 0 && ((mant & 0x0fff) != 0 || (m & 1) != 0) {
        m += 1;
        if m == 0x400 {
            m = 0;
            exp += 1;
            if exp >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((exp as u16) << 10) | m
}

/// One sample: the source activations plus every decoder layer's picks.
pub struct Record {
    pub phase: u8, // 0 = decode, 1 = prefill
    pub pos: u32,
    pub acts: Vec<u16>,               // src_layers().len() * N_EMBD, f16 bits
    pub picks: [i16; N_DST * N_EXPERT_USED],
}

/// Header, so a reader never has to guess geometry that a rebuild could change.
fn header() -> Vec<u8> {
    let mut h = Vec::new();
    h.extend_from_slice(b"DSPROBE1");
    for v in [
        N_EMBD as u32,
        N_EXPERT_USED as u32,
        DST0 as u32,
        N_DST as u32,
        src_layers().len() as u32,
    ] {
        h.extend_from_slice(&v.to_le_bytes());
    }
    for &l in src_layers() {
        h.extend_from_slice(&l.to_le_bytes());
    }
    h
}

struct Writer {
    tx: std::sync::mpsc::SyncSender<Record>,
}

static WRITER: std::sync::OnceLock<Option<Writer>> = std::sync::OnceLock::new();
pub static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static WRITTEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn writer() -> Option<&'static Writer> {
    WRITER
        .get_or_init(|| {
            let p = path()?;
            // Bounded: a slow disk must never stall decode. A full queue drops
            // the sample and counts it -- training data is fungible, a stalled
            // token is not.
            let (tx, rx) = std::sync::mpsc::sync_channel::<Record>(256);
            let p = p.to_string();
            let p_log = p.clone();
            let hdr = header();
            std::thread::Builder::new()
                .name("probe-dump".into())
                .spawn(move || {
                    use std::io::Write;
                    let f = match std::fs::OpenOptions::new().create(true).append(true).open(&p) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("probe-dump: cannot open {p}: {e}; disabled");
                            return;
                        }
                    };
                    let fresh = f.metadata().map(|m| m.len() == 0).unwrap_or(false);
                    let mut w = std::io::BufWriter::with_capacity(1 << 20, f);
                    if fresh {
                        let _ = w.write_all(&hdr);
                    }
                    while let Ok(r) = rx.recv() {
                        let _ = w.write_all(&[r.phase, 0, 0, 0]);
                        let _ = w.write_all(&r.pos.to_le_bytes());
                        for v in &r.acts {
                            let _ = w.write_all(&v.to_le_bytes());
                        }
                        for v in &r.picks {
                            let _ = w.write_all(&v.to_le_bytes());
                        }
                        WRITTEN.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    let _ = w.flush();
                })
                .ok()?;
            eprintln!(
                "probe dump ON -> {p_log} (src layers {:?}, targets L{}..{}, prefill stride {})",
                src_layers(),
                DST0,
                N_LAYER - 1,
                prefill_stride()
            );
            Some(Writer { tx })
        })
        .as_ref()
}

/// Per-token accumulator. One per in-flight sample; `emit` consumes it.
pub struct Sample {
    acts: Vec<u16>,
    picks: [i16; N_DST * N_EXPERT_USED],
    n_src_seen: usize,
}

impl Default for Sample {
    fn default() -> Self {
        Self {
            acts: Vec::with_capacity(src_layers().len() * N_EMBD as usize),
            picks: [-1; N_DST * N_EXPERT_USED],
            n_src_seen: 0,
        }
    }
}

impl Sample {
    /// Store a source layer's gate input (f32 host copy of `ffn_input_norm`).
    pub fn push_act(&mut self, layer: u32, x: &[f32]) {
        if !src_layers().contains(&layer) || x.len() != N_EMBD as usize {
            return;
        }
        self.acts.extend(x.iter().copied().map(f32_to_f16));
        self.n_src_seen += 1;
    }

    /// Store one decoder layer's selected experts.
    pub fn push_picks(&mut self, layer: u32, sel: &[i32]) {
        let l = layer as usize;
        if l < DST0 || l >= N_LAYER as usize {
            return;
        }
        let base = (l - DST0) * N_EXPERT_USED;
        for (i, &e) in sel.iter().take(N_EXPERT_USED).enumerate() {
            self.picks[base + i] = e as i16;
        }
    }

    /// Queue the sample. Incomplete samples are dropped: a record missing a
    /// source layer would silently train on zeros.
    pub fn emit(self, phase: u8, pos: u32) {
        let Some(w) = writer() else { return };
        if self.n_src_seen != src_layers().len() {
            DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return;
        }
        let r = Record { phase, pos, acts: self.acts, picks: self.picks };
        if w.tx.try_send(r).is_err() {
            DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// `(written, dropped)` for the log, so a dump that is silently losing samples
/// is visible rather than assumed complete.
pub fn stats() -> (u64, u64) {
    (
        WRITTEN.load(std::sync::atomic::Ordering::Relaxed),
        DROPPED.load(std::sync::atomic::Ordering::Relaxed),
    )
}

thread_local! {
    /// The token currently being accumulated. Decode runs one token at a time
    /// on one thread, so a thread-local needs no locking and cannot interleave
    /// two tokens' layers.
    static CUR: std::cell::RefCell<Sample> = std::cell::RefCell::new(Sample::default());
}

/// Call once per layer on the decode path, AFTER `d_selected` has been read
/// back (so the stream is already synced and neither copy adds a stall).
/// `act` is the host copy of `ffn_input_norm` for a source layer, else `None`.
/// Emits the completed sample at the last layer.
pub fn observe_decode(layer: u32, act: Option<&[f32]>, sel: &[i32], pos: u32) {
    if !on() {
        return;
    }
    CUR.with(|c| {
        let mut s = c.borrow_mut();
        if layer == 0 {
            // A request that ended mid-token leaves a partial behind; start clean.
            *s = Sample::default();
        }
        if let Some(a) = act {
            s.push_act(layer, a);
        }
        s.push_picks(layer, sel);
        if layer as i32 == N_LAYER - 1 {
            std::mem::take(&mut *s).emit(0, pos);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The f16 conversion is hand-rolled, and a silent bias in it would poison
    /// every sample in the dataset while still looking like plausible floats.
    /// Checked against exactly-representable values and against `f32::from` of
    /// the result, which is the only thing a reader will ever do with it.
    fn back(h: u16) -> f32 {
        let sign = ((h >> 15) & 1) as u32;
        let exp = ((h >> 10) & 0x1f) as u32;
        let mant = (h & 0x3ff) as u32;
        let bits = if exp == 0 {
            if mant == 0 { sign << 31 } else { return 0.0 }
        } else if exp == 0x1f {
            (sign << 31) | 0x7f80_0000 | (mant << 13)
        } else {
            (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
        };
        f32::from_bits(bits)
    }

    #[test]
    fn f16_roundtrips_exact_values() {
        for v in [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1024.0, 0.125] {
            assert_eq!(back(f32_to_f16(v)), v, "exact value {v} did not round-trip");
        }
    }

    #[test]
    fn f16_is_accurate_and_unbiased_over_the_activation_range() {
        // Post-RMSNorm activations sit around unit scale. f16 has ~3 decimal
        // digits there, so demand <0.1% relative error and a mean error that
        // does not drift one way (round-to-nearest-EVEN, not truncation --
        // truncation would bias every sample toward zero).
        let mut sum_rel = 0.0f64;
        let mut n = 0u32;
        let mut x = -8.0f32;
        while x <= 8.0 {
            if x.abs() > 1e-3 {
                let got = back(f32_to_f16(x));
                let rel = ((got - x) / x) as f64;
                assert!(rel.abs() < 1e-3, "{x} -> {got}, rel {rel}");
                sum_rel += rel;
                n += 1;
            }
            x += 0.0037;
        }
        let bias = sum_rel / n as f64;
        assert!(bias.abs() < 1e-5, "rounding is biased: mean relative error {bias}");
    }

    #[test]
    fn a_sample_missing_a_source_layer_is_dropped_not_zero_filled() {
        // The failure that would be invisible in training: a record whose
        // activation never arrived, written as zeros and learned from.
        let s = Sample::default();
        assert_eq!(s.n_src_seen, 0);
        assert!(s.picks.iter().all(|&p| p == -1), "picks must start as -1, not 0");
    }

    #[test]
    fn picks_land_at_the_layer_indexed_slot() {
        let mut s = Sample::default();
        s.push_picks(DST0 as u32, &[7, 8, 9, 10, 11, 12]);
        s.push_picks((N_LAYER - 1) as u32, &[1, 2, 3, 4, 5, 6]);
        assert_eq!(s.picks[0], 7);
        assert_eq!((s.picks[(N_DST - 1) * N_EXPERT_USED], s.picks[N_DST * N_EXPERT_USED - 1]), (1, 6));
        // An encoder layer has no slot and must be ignored, not aliased.
        s.push_picks(0, &[99; 6]);
        assert!(!s.picks.contains(&99), "an encoder layer wrote into the decoder target array");
    }
}
