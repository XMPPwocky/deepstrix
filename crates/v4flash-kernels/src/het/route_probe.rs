//! ROUTE PROBE (`V41_ROUTE_PROBE=<dir>`): on the arena (decode) path, log per
//! row and layer the real expert picks, a one-layer LOOK-AHEAD proxy (layer
//! L+1's router applied to layer L's router input) with its online precision,
//! and the layer-20 router-input activation, for training a layer-20 ->
//! decoder-picks predictor offline (init from the decoder routers, regularised
//! toward them). Files (append, raw little-endian):
//!   picks.i32   [rows][N_LAYER][N_EXPERT_USED]   real picks (-1 = none)
//!   look.i32    [rows][N_LAYER][N_EXPERT_USED]   proxy for layer L computed at L-1 (layer 0: -1)
//!   act20.f32   [rows][N_EMBD]                   ffn_input_norm at CANDIDATE_SOURCE_LAYER (20)
//!   meta.txt    one line per flush: rows so far
//! Costs a device sync per layer per lane: PROBE ONLY.
use crate::config::{CANDIDATE_SOURCE_LAYER, N_EMBD, N_EXPERT, N_EXPERT_USED, N_LAYER};
use color_eyre::eyre;
use std::io::Write;
use std::sync::Mutex;
use v4flash_hip::DeviceBuffer;

const NU: usize = N_EXPERT_USED;
const NL: usize = N_LAYER as usize;

struct Lane {
    b: usize,
    picks: Vec<i32>, // b * NL * NU
    look: Vec<i32>,
    act20: Vec<f32>, // b * N_EMBD
}

struct State {
    lanes: [Option<Lane>; 4],
    rows: u64,
    hit: [u64; NL],
    tot: [u64; NL],
    last_report: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);

pub fn dir() -> Option<&'static std::path::Path> {
    static D: std::sync::LazyLock<Option<std::path::PathBuf>> = std::sync::LazyLock::new(|| {
        let p = std::env::var_os("V41_ROUTE_PROBE").map(std::path::PathBuf::from)?;
        std::fs::create_dir_all(&p).ok()?;
        Some(p)
    });
    D.as_deref()
}

pub fn enabled() -> bool { dir().is_some() }

/// Scratch for the look-ahead router (allocated on first use, per process).
pub struct Scratch {
    pub logits: DeviceBuffer<f32>,
    pub sel: DeviceBuffer<i32>,
    pub ew: DeviceBuffer<f32>,
}

/// Record layer `layer`'s real picks (`d_selected`, b*NU), the look-ahead
/// proxy for layer+1 (`look_sel`, b*NU, already computed by the caller) and, at
/// the candidate source layer, the activation (`act`, b*N_EMBD).
pub fn note(lane: usize, layer: usize, b: usize, picks: &[i32], look_next: Option<&[i32]>, act: Option<&[f32]>) -> eyre::Result<()> {
    let Some(d) = dir() else { return Ok(()) };
    let mut g = STATE.lock().unwrap();
    let st = g.get_or_insert_with(|| State { lanes: [None, None, None, None], rows: 0, hit: [0; NL], tot: [0; NL], last_report: 0 });
    let lane = lane.min(st.lanes.len() - 1);
    if layer == 0 {
        st.lanes[lane] = Some(Lane { b, picks: vec![-1; b * NL * NU], look: vec![-1; b * NL * NU], act20: vec![0.0; b * N_EMBD as usize] });
    }
    let Some(l) = st.lanes[lane].as_mut() else { return Ok(()) };
    if l.b != b { st.lanes[lane] = None; return Ok(()); }
    for r in 0..b {
        l.picks[(r * NL + layer) * NU..(r * NL + layer + 1) * NU].copy_from_slice(&picks[r * NU..(r + 1) * NU]);
        if let Some(lk) = look_next {
            if layer + 1 < NL {
                l.look[(r * NL + layer + 1) * NU..(r * NL + layer + 2) * NU].copy_from_slice(&lk[r * NU..(r + 1) * NU]);
            }
        }
        if layer > 0 {
            // precision of the proxy predicted at layer-1 for this layer
            let pred = &l.look[(r * NL + layer) * NU..(r * NL + layer + 1) * NU];
            let real = &l.picks[(r * NL + layer) * NU..(r * NL + layer + 1) * NU];
            if pred[0] >= 0 {
                st.tot[layer] += NU as u64;
                st.hit[layer] += pred.iter().filter(|p| real.contains(p)).count() as u64;
            }
        }
    }
    if let Some(a) = act {
        if layer == CANDIDATE_SOURCE_LAYER as usize {
            l.act20.copy_from_slice(&a[..b * N_EMBD as usize]);
        }
    }
    if layer + 1 == NL {
        let l = st.lanes[lane].take().unwrap();
        let mut fp = std::fs::OpenOptions::new().create(true).append(true).open(d.join("picks.i32"))?;
        let mut fl = std::fs::OpenOptions::new().create(true).append(true).open(d.join("look.i32"))?;
        let mut fa = std::fs::OpenOptions::new().create(true).append(true).open(d.join("act20.f32"))?;
        fp.write_all(bytemuck_i32(&l.picks))?;
        fl.write_all(bytemuck_i32(&l.look))?;
        fa.write_all(bytemuck_f32(&l.act20))?;
        st.rows += b as u64;
        if st.rows - st.last_report >= 500 {
            st.last_report = st.rows;
            let enc: (u64, u64) = (1..CANDIDATE_SOURCE_LAYER as usize).map(|i| (st.hit[i], st.tot[i])).fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
            let dec: (u64, u64) = (CANDIDATE_SOURCE_LAYER as usize..NL).map(|i| (st.hit[i], st.tot[i])).fold((0, 0), |a, x| (a.0 + x.0, a.1 + x.1));
            let per: Vec<String> = (1..NL).step_by(4).map(|i| format!("L{i}:{:.2}", st.hit[i] as f64 / st.tot[i].max(1) as f64)).collect();
            eprintln!("route-probe: rows={} lookahead precision enc={:.3} dec={:.3} [{}]", st.rows, enc.0 as f64 / enc.1.max(1) as f64, dec.0 as f64 / dec.1.max(1) as f64, per.join(" "));
            let mut fm = std::fs::OpenOptions::new().create(true).append(true).open(d.join("meta.txt"))?;
            writeln!(fm, "rows={} enc={:.4} dec={:.4}", st.rows, enc.0 as f64 / enc.1.max(1) as f64, dec.0 as f64 / dec.1.max(1) as f64)?;
        }
    }
    Ok(())
}

fn bytemuck_i32(v: &[i32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) } }
fn bytemuck_f32(v: &[f32]) -> &[u8] { unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) } }

pub const N_EXPERT_U32: u32 = N_EXPERT;
