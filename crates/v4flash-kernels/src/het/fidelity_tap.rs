//! Taps for the fidelity gate (`tests/v41_golden_gate.rs`).
//!
//! - The pick SINK records every expert pick the engine makes, so the gate can
//!   compare each routing decision against the CPU reference's.
//! - The residual SINK records the residual stream (the mHC state, `HC_DIM`)
//!   after every layer of a serial decode step, so the gate can name the first
//!   layer that departs from the reference instead of only seeing the logits.
//! - The PIN forces the engine's picks to the reference's. Top-k selection is
//!   discontinuous: a 1e-4 difference in a router score near the boundary swaps
//!   an expert, and the swap compounds through every later layer and position.
//!   Pinned, a small numeric divergence stays small, so the gate can price the
//!   engine's numerics separately from its routing flips.
//!
//! Both are off unless a test turns them on. Off, each costs one relaxed atomic
//! load per layer at sites that already exist; nothing is read back or synced.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

use color_eyre::eyre;
use v4flash_hip::{DeviceBuffer, Stream};

use crate::config::{N_EXPERT, N_EXPERT_USED};

/// One router pick as read back by the pager path, for [`pick_sink_take`].
#[derive(Clone, Debug)]
pub struct PickRecord {
    /// `false` = serial decode (`D` in the text trace), `true` = a batched call
    /// row (prefill chunk, CED replay or arena step; `P`).
    pub batched: bool,
    pub layer: u16,
    /// Row within the call and the call's row count (always 0 / 1 for decode).
    pub row: u32,
    pub rows: u32,
    /// Routed expert ids in the router's rank order.
    pub ids: [i32; N_EXPERT_USED],
}

static PICK_SINK_ON: AtomicBool = AtomicBool::new(false);
static PICK_SINK: Mutex<Vec<PickRecord>> = Mutex::new(Vec::new());

/// In-process pick sink. Recorded at the two sites that already read the picks
/// back for paging, so it adds no readback and no sync.
pub fn pick_sink_enable(on: bool) {
    PICK_SINK_ON.store(on, Ordering::Relaxed);
}
#[inline]
pub fn pick_sink_on() -> bool {
    PICK_SINK_ON.load(Ordering::Relaxed)
}
pub fn pick_sink_push(batched: bool, layer: usize, row: u32, rows: u32, ids: &[i32]) {
    let mut rec = PickRecord { batched, layer: layer as u16, row, rows, ids: [-1; N_EXPERT_USED] };
    let n = ids.len().min(rec.ids.len());
    rec.ids[..n].copy_from_slice(&ids[..n]);
    PICK_SINK.lock().unwrap().push(rec);
}
/// Everything recorded since the last take, in call order.
pub fn pick_sink_take() -> Vec<PickRecord> {
    std::mem::take(&mut *PICK_SINK.lock().unwrap())
}

static RESIDUAL_SINK_ON: AtomicBool = AtomicBool::new(false);
static RESIDUAL_SINK: Mutex<Vec<(u16, Vec<f32>)>> = Mutex::new(Vec::new());

/// Residual sink, serial decode only (`forward_token_impl`). On, it costs a
/// stream drain and an 80 KB readback per layer.
pub fn residual_sink_enable(on: bool) {
    RESIDUAL_SINK_ON.store(on, Ordering::Relaxed);
}
#[inline]
pub fn residual_sink_on() -> bool {
    RESIDUAL_SINK_ON.load(Ordering::Relaxed)
}
/// Record `layer`'s output residual; drains `stream` first.
pub fn residual_sink_push(stream: &Stream, layer: usize, residual: &DeviceBuffer<f32>) -> eyre::Result<()> {
    stream.synchronize()?;
    let mut h = vec![0f32; crate::config::HC_DIM as usize];
    residual.slice_view(0, h.len()).copy_to_host(&mut h)?;
    RESIDUAL_SINK.lock().unwrap().push((layer as u16, h));
    Ok(())
}
/// `(layer, residual after that layer)` since the last take, in call order.
pub fn residual_sink_take() -> Vec<(u16, Vec<f32>)> {
    std::mem::take(&mut *RESIDUAL_SINK.lock().unwrap())
}

/// The reference's picks, `[layer][pos][k]` for absolute positions `0..n_pos`
/// (the fixture's `topk_ids.i32`). One sequence: the table is keyed by position
/// alone, so it must not be installed while two streams decode.
pub struct PinTable {
    n_layer: usize,
    n_pos: usize,
    ids: Vec<i32>,
}

impl PinTable {
    pub fn new(n_layer: usize, n_pos: usize, ids: Vec<i32>) -> eyre::Result<Self> {
        eyre::ensure!(
            ids.len() == n_layer * n_pos * N_EXPERT_USED,
            "pin table: {} ids for {n_layer} layers x {n_pos} positions x {N_EXPERT_USED}",
            ids.len()
        );
        eyre::ensure!(
            ids.iter().all(|&e| (0..N_EXPERT as i32).contains(&e)),
            "pin table: expert id out of range"
        );
        Ok(Self { n_layer, n_pos, ids })
    }

    fn get(&self, layer: usize, pos: u32) -> Option<&[i32]> {
        let pos = pos as usize;
        if layer >= self.n_layer || pos >= self.n_pos {
            return None;
        }
        let o = (layer * self.n_pos + pos) * N_EXPERT_USED;
        Some(&self.ids[o..o + N_EXPERT_USED])
    }
}

static PIN_ON: AtomicBool = AtomicBool::new(false);
static PIN: Mutex<Option<PinTable>> = Mutex::new(None);
static PIN_OVERRIDES: AtomicU64 = AtomicU64::new(0);

/// Install (or with `None`, remove) the pin. Resets the override count.
pub fn pin_set(table: Option<PinTable>) {
    let mut g = PIN.lock().unwrap();
    PIN_ON.store(table.is_some(), Ordering::Relaxed);
    *g = table;
    PIN_OVERRIDES.store(0, Ordering::Relaxed);
}
#[inline]
pub fn pin_on() -> bool {
    PIN_ON.load(Ordering::Relaxed)
}
/// Rows whose pick set the pin replaced since the last take (the routing flips
/// a free run would have taken).
pub fn pin_overrides_take() -> u64 {
    PIN_OVERRIDES.swap(0, Ordering::Relaxed)
}

/// `router_topk.hip`'s `softplus_stable`.
fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// Apply the pin to `rows` router outputs in place: `logits` `[rows][N_EXPERT]`
/// are the router's raw logits, `sel` / `ew` `[rows][k]` its picks and weights.
/// A row whose pick SET already equals the reference's is left untouched, so a
/// pinned run is bit-identical to a free one wherever routing agreed. Any other
/// row gets the reference ids in reference order, weighted as `router_topk`
/// weights its own picks (the bias only selects; it never weights). The weights
/// use the host's `exp`/`ln_1p`, so they can differ from the kernel's by an ulp.
/// Rows past the table's positions stay free. Returns the rows changed.
pub fn pin_apply(
    layer: usize,
    pos_of_row: impl Fn(u32) -> u32,
    logits: &[f32],
    sel: &mut [i32],
    ew: &mut [f32],
    scale: f32,
    eps: f32,
) -> usize {
    let (k, ne) = (N_EXPERT_USED, N_EXPERT as usize);
    let g = PIN.lock().unwrap();
    let Some(t) = g.as_ref() else { return 0 };
    let mut changed = 0;
    for r in 0..sel.len() / k {
        let Some(want) = t.get(layer, pos_of_row(r as u32)) else { continue };
        let got = &mut sel[r * k..(r + 1) * k];
        let (mut a, mut b) = ([0i32; N_EXPERT_USED], [0i32; N_EXPERT_USED]);
        a.copy_from_slice(got);
        b.copy_from_slice(want);
        a.sort_unstable();
        b.sort_unstable();
        if a == b {
            continue;
        }
        let lg = &logits[r * ne..(r + 1) * ne];
        let p: Vec<f32> = want.iter().map(|&e| softplus_stable(lg[e as usize]).sqrt()).collect();
        let sum = p.iter().sum::<f32>().max(eps);
        got.copy_from_slice(want);
        for (w, pj) in ew[r * k..(r + 1) * k].iter_mut().zip(&p) {
            *w = pj / sum * scale;
        }
        changed += 1;
    }
    PIN_OVERRIDES.fetch_add(changed as u64, Ordering::Relaxed);
    changed
}

/// [`pin_apply`] on device buffers: drains `stream` (the router's), reads the
/// first `rows` rows back, and writes them again only if a row changed. A test
/// hook: the sync is paid per layer while the pin is installed.
pub fn pin_device_rows(
    stream: &Stream,
    layer: usize,
    pos_of_row: impl Fn(u32) -> u32,
    rows: usize,
    logits: &DeviceBuffer<f32>,
    sel: &mut DeviceBuffer<i32>,
    ew: &mut DeviceBuffer<f32>,
    scale: f32,
    eps: f32,
) -> eyre::Result<()> {
    let (k, ne) = (N_EXPERT_USED, N_EXPERT as usize);
    stream.synchronize()?;
    let mut lg = vec![0f32; rows * ne];
    let mut s = vec![0i32; rows * k];
    let mut w = vec![0f32; rows * k];
    logits.slice_view(0, rows * ne).copy_to_host(&mut lg)?;
    sel.slice_view(0, rows * k).copy_to_host(&mut s)?;
    ew.slice_view(0, rows * k).copy_to_host(&mut w)?;
    if pin_apply(layer, pos_of_row, &lg, &mut s, &mut w, scale, eps) > 0 {
        sel.slice_view_mut(0, rows * k).copy_from_host(&s)?;
        ew.slice_view_mut(0, rows * k).copy_from_host(&w)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test: the pin is process-global state.
    #[test]
    fn pin_apply_replaces_only_differing_sets() {
        let (k, ne) = (N_EXPERT_USED, N_EXPERT as usize);
        let (n_layer, n_pos) = (2, 3);
        let mut ids = vec![0i32; n_layer * n_pos * k];
        for (i, v) in ids.iter_mut().enumerate() {
            *v = (i % k) as i32 * 7 + 1; // 1, 8, 15, 22, 29, 36
        }
        pin_set(Some(PinTable::new(n_layer, n_pos, ids).unwrap()));
        let logits: Vec<f32> = (0..3 * ne).map(|i| ((i % ne) as f32 * 0.37).sin()).collect();
        // Row 0: the reference set in another order -> untouched.
        // Row 1: one expert swapped -> replaced.  Row 2: position 3, past the table -> free.
        let mut sel = vec![36, 29, 22, 15, 8, 1, 1, 8, 15, 22, 29, 99, 5, 6, 7, 8, 9, 10];
        let mut ew = vec![0.25f32; 3 * k];
        let (sel0, ew0) = (sel.clone(), ew.clone());
        let n = pin_apply(1, |r| r + 1, &logits, &mut sel, &mut ew, 1.5, 6.103515625e-5);
        assert_eq!(n, 1);
        assert_eq!(pin_overrides_take(), 1);
        assert_eq!(&sel[..k], &sel0[..k]);
        assert_eq!(&ew[..k], &ew0[..k]);
        assert_eq!(&sel[k..2 * k], &[1, 8, 15, 22, 29, 36]);
        let p: Vec<f32> = [1, 8, 15, 22, 29, 36].iter().map(|&e| softplus_stable(logits[ne + e]).sqrt()).collect();
        let sum: f32 = p.iter().sum();
        for (w, pj) in ew[k..2 * k].iter().zip(&p) {
            assert_eq!(*w, pj / sum * 1.5);
        }
        assert_eq!(&sel[2 * k..], &sel0[2 * k..]);
        assert_eq!(&ew[2 * k..], &ew0[2 * k..]);
        pin_set(None);
        assert!(!pin_on());
        assert_eq!(pin_apply(1, |r| r, &logits, &mut sel, &mut ew, 1.5, 6.103515625e-5), 0);
    }
}
