//! KNOWN_BUGS #57: a drafter ring FILLED on the transition to drafting
//! (`ms_dspark::Backlog` / `MsDspark::fill`, deepstrix-server) must hold what a
//! ring written every step holds (`V41_MS_DSPARK_RING=all`). Three arms run the
//! same script -- prefill seeds, steps with 1..3+ live streams, kept rows, drafts
//! (the real drafter forward with its rewind) -- through the real kernels:
//!
//! - `Every`: every kept row and the seed written at once;
//! - `Fill`: the production schedule (seed held; a ring-writing step writes a
//!   backlog of one step's rows; a bigger backlog waits for the draft, which
//!   writes it -- the latest window, up to 16 chunks back to back, unsynced --
//!   then syncs);
//! - `Drop`: the pre-#57 `RING=solo` (rows of 3+ live steps never written): the
//!   negative control, which must differ.
//!
//! Scenario A has only gaps within a window: `Fill` must equal `Every` bit for
//! bit (every drafter output and the ring bytes). Scenario B has a gap longer
//! than a window: the rows land at other ring slots, so the valid window must
//! match BY POSITION (same rows, same bytes) and the drafter outputs to float
//! reassociation, far closer than `Drop`.
//!
//! Loads the drafter only (iGPU, ~8.7 GB): hub DOWN. Run:
//! ```text
//! CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p v4flash-kernels \
//!   --release --features v41 --test mtp_ring_fill -- --ignored --nocapture
//! ```
#![cfg(feature = "v41")]

use std::collections::{BTreeMap, VecDeque};

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{HC_DIM, N_EMBD, N_HEAD_DIM};
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::mtp::{mtp_rope, MtpState, MTP_BLOCK, MTP_SRC_LAYERS, MTP_WINDOW, RING_ROWS_MAX};
use v4flash_kernels::het::weights::MtpWeights;

const HF_DIR_DEFAULT: &str =
    "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277";

const K: usize = MTP_SRC_LAYERS.len() * N_EMBD as usize;
const ROW: usize = N_HEAD_DIM as usize;

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

/// Deterministic residual-like rows.
fn rows(seed: u64, n: usize, len: usize) -> Vec<f32> {
    let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n * len)
        .map(|_| {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((x >> 40) as f32 / (1u64 << 24) as f32) - 0.5) * 4.0
        })
        .collect()
}

/// The main model's residual of position `p` (the same in every arm).
fn hidden_of(p: u32) -> Vec<f32> {
    rows(p as u64 + 17, 1, K)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    Every,
    Fill,
    Drop,
}

#[derive(Clone, Copy, Debug)]
enum Ev {
    /// A prefill's seed: the `n` rows ending at `last` (the stream restarts).
    Seed { n: u32, last: u32 },
    /// One step: `live` streams, this stream drafts first (only when `live <= 2`),
    /// then keeps `keep` rows.
    Step { live: usize, keep: u32, draft: bool },
}

/// A ring's valid window by position: per layer, the row bytes of each of the
/// latest `n_valid` writes.
type Window = BTreeMap<u32, Vec<Vec<u16>>>;

struct Arm<'a> {
    e: &'a DeviceEngine,
    w: &'a MtpWeights,
    mode: Mode,
    st: MtpState,
    /// `MsDspark`'s per-slot bookkeeping.
    last_ring_pos: Option<u32>,
    backlog: VecDeque<(u32, Vec<f32>)>,
    /// Position of each ring slot's latest write.
    slot_pos: Vec<Option<u32>>,
    /// The stream's last kept position.
    pos: u32,
    outs: Vec<Vec<f32>>,
    windows: Vec<Window>,
    written: usize,
    /// Extra older rows on the FIRST seed (`Every` only): the rotation control.
    seed_extra: u32,
}

impl<'a> Arm<'a> {
    fn new(e: &'a DeviceEngine, w: &'a MtpWeights, dev: &Device, mode: Mode) -> eyre::Result<Self> {
        let mut st = MtpState::alloc(dev.id)?;
        for r in &mut st.rings {
            r.fill_zero()?;
        }
        Ok(Self {
            e, w, mode, st, last_ring_pos: None, backlog: VecDeque::new(), slot_pos: vec![None; MTP_WINDOW],
            pos: 0, outs: Vec::new(), windows: Vec::new(), written: 0, seed_extra: 0,
        })
    }

    fn reset(&mut self) {
        self.st.set_ring_writes(0);
        self.last_ring_pos = None;
        self.backlog.clear();
        self.slot_pos = vec![None; MTP_WINDOW];
    }

    /// `write_backlog` of an explicit run: rows the ring holds and rows older
    /// than the run's last window skipped (`ms_dspark::ring_skip`), then ONE
    /// `ring_write_rows` per `RING_ROWS_MAX` chunk, back to back, no sync.
    fn write_run(&mut self, pos0: u32, run: &[Vec<f32>]) -> eyre::Result<()> {
        let n = run.len();
        let held = match self.last_ring_pos {
            Some(p) if p >= pos0 => (p - pos0) as usize + 1,
            _ => 0,
        };
        let skip = held.max(n.saturating_sub(MTP_WINDOW)).min(n);
        let rope = mtp_rope();
        for (c, chunk) in run[skip..].chunks(RING_ROWS_MAX).enumerate() {
            let p = pos0 + (skip + c * RING_ROWS_MAX) as u32;
            let w0 = self.st.ring_writes();
            self.st.ring_write_rows(self.e, &self.e.compute, self.w, &rope, p, &chunk.concat())?;
            for j in 0..chunk.len() {
                self.slot_pos[(w0 + j) % MTP_WINDOW] = Some(p + j as u32);
            }
            self.written += chunk.len();
        }
        if skip < n {
            self.last_ring_pos = Some(pos0 + n as u32 - 1);
        }
        Ok(())
    }

    fn write_backlog(&mut self) -> eyre::Result<()> {
        if self.backlog.is_empty() {
            return Ok(());
        }
        let pos0 = self.backlog[0].0;
        let run: Vec<Vec<f32>> = std::mem::take(&mut self.backlog).into_iter().map(|(_, h)| h).collect();
        self.write_run(pos0, &run)
    }

    /// `Backlog::hold`.
    fn hold(&mut self, pos0: u32, run: Vec<Vec<f32>>) {
        if self.backlog.back().is_some_and(|(p, _)| p + 1 != pos0) {
            panic!("backlog run does not continue (pos0 {pos0})");
        }
        for (j, h) in run.into_iter().enumerate() {
            self.backlog.push_back((pos0 + j as u32, h));
        }
        while self.backlog.len() > MTP_WINDOW {
            self.backlog.pop_front();
        }
    }

    /// `MsDspark::keep_rows` (and the pre-#57 / `RING=all` variants).
    fn keep(&mut self, pos0: u32, run: Vec<Vec<f32>>, write_ring: bool) -> eyre::Result<()> {
        match self.mode {
            Mode::Every => self.write_run(pos0, &run)?,
            Mode::Drop => {
                if write_ring {
                    self.write_run(pos0, &run)?;
                }
            }
            Mode::Fill => {
                self.hold(pos0, run);
                if write_ring && self.backlog.len() <= MTP_BLOCK + 1 {
                    self.write_backlog()?;
                }
            }
        }
        Ok(())
    }

    fn window(&self) -> eyre::Result<Window> {
        let writes = self.st.ring_writes();
        // The rows written so far, at most a window (`ring_geom`'s n_valid
        // before the next draft's own row).
        let n_valid = MTP_WINDOW.min(writes);
        let host: Vec<Vec<u16>> = self
            .st
            .rings
            .iter()
            .map(|r| {
                let mut v = vec![0u16; r.len()];
                r.copy_to_host(&mut v).map(|()| v)
            })
            .collect::<eyre::Result<_>>()?;
        let mut out = Window::new();
        for back in 1..=n_valid {
            let slot = (writes - back) % MTP_WINDOW;
            let p = self.slot_pos[slot].ok_or_else(|| eyre!("valid slot {slot} never written"))?;
            out.insert(p, host.iter().map(|l| l[slot * ROW..(slot + 1) * ROW].to_vec()).collect());
        }
        Ok(out)
    }

    /// `MsDspark::draft`: (`Fill`) fill + sync, then the drafter forward from
    /// the last kept position, rewinding when the ring already holds it.
    fn draft(&mut self, token_row: &[f32], noise_row: &[f32]) -> eyre::Result<()> {
        if self.mode == Mode::Fill {
            self.write_backlog()?;
        }
        self.e.compute.synchronize()?;
        let pos = self.pos;
        let rewind = self.last_ring_pos == Some(pos);
        let w = self.st.ring_writes();
        self.st.set_ring_writes(w - usize::from(rewind && w > 0));
        let slot = self.st.ring_writes() % MTP_WINDOW;
        self.st.inject_main_hidden(&hidden_of(pos))?;
        self.st.forward(self.e, &self.e.compute, self.w, &mtp_rope(), pos, token_row, noise_row)?;
        self.e.compute.synchronize()?;
        self.slot_pos[slot] = Some(pos);
        self.last_ring_pos = Some(pos);
        let mut h = vec![0f32; self.st.h.len()];
        self.st.h.copy_to_host(&mut h)?;
        self.outs.push(h);
        self.windows.push(self.window()?);
        Ok(())
    }

    fn run(&mut self, script: &[Ev]) -> eyre::Result<()> {
        let token_row = rows(7, 1, HC_DIM as usize);
        let noise_row = rows(11, 1, HC_DIM as usize);
        for ev in script {
            match *ev {
                Ev::Seed { n, last } => {
                    self.e.compute.synchronize()?;
                    self.reset();
                    let pos0 = last + 1 - n - std::mem::take(&mut self.seed_extra);
                    let run: Vec<Vec<f32>> = (pos0..=last).map(hidden_of).collect();
                    // Production seeds through keep_rows with write = !fill.
                    self.keep(pos0, run, self.mode != Mode::Fill)?;
                    self.pos = last;
                }
                Ev::Step { live, keep, draft } => {
                    let write_ring = live <= 2;
                    if draft {
                        assert!(write_ring, "a stream drafts only when <= 2 are live");
                        self.draft(&token_row, &noise_row)?;
                    }
                    let pos0 = self.pos + 1;
                    let run: Vec<Vec<f32>> = (pos0..pos0 + keep).map(hidden_of).collect();
                    self.keep(pos0, run, write_ring)?;
                    self.pos += keep;
                }
            }
        }
        self.e.compute.synchronize()?;
        Ok(())
    }

    fn ring_bytes(&self) -> eyre::Result<Vec<Vec<u16>>> {
        self.st
            .rings
            .iter()
            .map(|r| {
                let mut v = vec![0u16; r.len()];
                r.copy_to_host(&mut v).map(|()| v)
            })
            .collect()
    }
}

fn steps(n: usize, live: usize, keep: u32, draft: bool) -> Vec<Ev> {
    vec![Ev::Step { live, keep, draft }; n]
}

/// Gaps within a window only.
fn scenario_a() -> Vec<Ev> {
    let mut s = vec![Ev::Seed { n: 100, last: 2000 }];
    s.extend(steps(5, 3, 1, false)); // admitted into a 3+ stretch: the seed waits (105 rows, within a window)
    s.push(Ev::Step { live: 2, keep: 4, draft: true }); // fill = seed + 5 rows
    for (k, live) in [(3, 1), (1, 2), (6, 1), (2, 2), (5, 1)] {
        s.push(Ev::Step { live, keep: k, draft: true });
    }
    s.extend(steps(40, 4, 1, false)); // a 40-row gap
    s.extend(steps(3, 1, 1, false)); // back-off at 1 live: the backlog (> 6) waits
    s.push(Ev::Step { live: 1, keep: 3, draft: true });
    s.extend(steps(4, 1, 2, true));
    s.extend(steps(10, 3, 1, false)); // a 10-row gap
    s.push(Ev::Step { live: 2, keep: 1, draft: false }); // a write step, backlog 11: waits
    s.push(Ev::Step { live: 2, keep: 5, draft: true });
    s.push(Ev::Step { live: 3, keep: 1, draft: false }); // one-step stretch
    s.push(Ev::Step { live: 2, keep: 2, draft: true });
    s.extend(steps(3, 1, 3, true));
    s.push(Ev::Seed { n: 1, last: 5000 }); // a whole-prompt hit: 1-row seed
    s.push(Ev::Step { live: 1, keep: 2, draft: true });
    s.extend(steps(6, 2, 4, true));
    s.extend(steps(7, 3, 1, false));
    s.push(Ev::Step { live: 2, keep: 3, draft: true });
    s
}

/// A gap longer than a window (and a seed shorter than one).
fn scenario_b() -> Vec<Ev> {
    let mut s = vec![Ev::Seed { n: 100, last: 3000 }];
    s.extend(steps(3, 1, 3, true));
    s.extend(steps(200, 3, 1, false));
    s.extend(steps(6, 1, 2, true));
    s.extend(steps(150, 5, 1, false));
    s.extend(steps(2, 1, 1, false));
    s.extend(steps(5, 2, 3, true));
    s
}

fn bits_equal(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// ||a - b|| / ||a||.
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a.iter().zip(b).map(|(x, y)| ((x - y) as f64).powi(2)).sum();
    let n: f64 = a.iter().map(|x| (*x as f64).powi(2)).sum();
    (d / n.max(1e-30)).sqrt()
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

#[test]
#[ignore]
fn filled_ring_matches_an_every_step_ring() -> eyre::Result<()> {
    install_panic_handler()?;
    let dir = std::env::var("V41_HF_DIR").unwrap_or_else(|_| HF_DIR_DEFAULT.to_string());
    let hf = V41HfWeights::open(&dir, None)?;
    let dev = pick_igpu()?;
    dev.set_current()?;
    let arch = dev.properties()?.gcn_arch_name;
    let e = DeviceEngine::for_arch(dev, &arch)?;
    let w = MtpWeights::load(&hf, dev, 40)?;
    let mut failures = Vec::new();

    for (name, script, exact) in [("A (gaps <= window)", scenario_a(), true), ("B (gap > window)", scenario_b(), false)] {
        let mut arms = Vec::new();
        for (mode, extra) in [(Mode::Every, 0), (Mode::Fill, 0), (Mode::Drop, 0), (Mode::Every, 28)] {
            let mut a = Arm::new(&e, &w, &dev, mode)?;
            a.seed_extra = extra;
            a.run(&script)?;
            arms.push(a);
        }
        let (every, fill, drop, rot) = (&arms[0], &arms[1], &arms[2], &arms[3]);
        let n = every.outs.len();
        assert_eq!(fill.outs.len(), n);
        assert_eq!(drop.outs.len(), n);
        let fill_bits = (0..n).filter(|&i| !bits_equal(&every.outs[i], &fill.outs[i])).count();
        let drop_bits = (0..n).filter(|&i| !bits_equal(&every.outs[i], &drop.outs[i])).count();
        let fill_max = (0..n).map(|i| max_abs(&every.outs[i], &fill.outs[i])).fold(0.0, f32::max);
        let drop_max = (0..n).map(|i| max_abs(&every.outs[i], &drop.outs[i])).fold(0.0, f32::max);
        let fill_win = (0..n).filter(|&i| every.windows[i] != fill.windows[i]).count();
        let drop_win = (0..n).filter(|&i| every.windows[i] != drop.windows[i]).count();
        let ring_same = every.ring_bytes()? == fill.ring_bytes()?;
        println!(
            "mtp ring fill {name}: {n} drafts; rows written every/fill/drop = {}/{}/{}\n  \
             fill vs every: {fill_bits} drafter outputs differ (max |dh| {fill_max:.3e}), {fill_win} windows differ by position, final ring bytes {}\n  \
             drop vs every: {drop_bits} drafter outputs differ (max |dh| {drop_max:.3e}), {drop_win} windows differ by position",
            every.written, fill.written, drop.written, if ring_same { "identical" } else { "differ" },
        );
        if fill_win > 0 {
            failures.push(format!("{name}: {fill_win} filled windows differ from the every-step ring by position"));
        }
        if exact && (fill_bits > 0 || !ring_same) {
            failures.push(format!("{name}: fill is not bit-identical ({fill_bits} outputs, ring bytes same = {ring_same})"));
        }
        // The rotation control: the same rows by position at slots shifted by
        // a bigger first seed, as production's own history shifts them. Its
        // deviation from `Every` is the noise floor of slot order alone.
        let same_rows: Vec<usize> = (0..n).filter(|&i| rot.windows[i] == every.windows[i]).collect();
        let fmt = |v: &[f64]| v.iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>().join(" ");
        let rel_fill: Vec<f64> = (0..n).map(|i| rel(&every.outs[i], &fill.outs[i])).collect();
        let rel_drop: Vec<f64> = (0..n).map(|i| rel(&every.outs[i], &drop.outs[i])).collect();
        let rel_rot: Vec<f64> = (0..n).map(|i| rel(&every.outs[i], &rot.outs[i])).collect();
        println!("  per-draft ||dh||/||h||\n    fill: {}\n    rot:  {}\n    drop: {}\n    rot holds the same rows as every at drafts {same_rows:?}",
            fmt(&rel_fill), fmt(&rel_rot), fmt(&rel_drop));
        if !exact {
            let fill_m = same_rows.iter().map(|&i| rel_fill[i]).fold(0.0, f64::max);
            let rot_m = same_rows.iter().map(|&i| rel_rot[i]).fold(0.0, f64::max);
            let drop_m = same_rows.iter().map(|&i| rel_drop[i]).fold(0.0, f64::max);
            println!("  over those drafts: max rel fill {fill_m:.2e}, rotation-only {rot_m:.2e}, drop {drop_m:.2e}");
            if same_rows.is_empty() || !(fill_m <= 3.0 * rot_m + 1e-6) || !(fill_m < 0.5 * drop_m) {
                failures.push(format!("{name}: fill deviates beyond slot-order noise (fill {fill_m:.2e}, rotation {rot_m:.2e}, drop {drop_m:.2e})"));
            }
        }
        if drop_win == 0 || drop_bits == 0 {
            failures.push(format!("{name}: the negative control (pre-#57 drop) does not differ -- the script has no gap"));
        }
    }
    if !failures.is_empty() {
        return Err(eyre!("{}", failures.join("; ")));
    }
    Ok(())
}
