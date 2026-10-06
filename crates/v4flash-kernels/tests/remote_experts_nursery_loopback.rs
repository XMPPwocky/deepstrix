//! Box-2 NURSERY end to end on the gfx1151 iGPU
//! (docs/v41/B2_PREDICTED_MISS_PREFETCH_DESIGN.md 3.2, section 7 gates; the
//! sibling of `remote_experts_pin_loopback`). The REAL daemon loop
//! (`serve_connection`: LIKELY words applied at arrival, the LIKELY reader
//! class, nursery landings, promote-on-use inside `touch_hit`, the NURSERY
//! block) and the REAL hub side (`RemoteExpertClient` + `het::lookahead` +
//! `b2_mirror`: the hint queue, `REQ_FLAG_LIKELY`, the capability probe, the
//! NURSERY mirror bits, the budget and the bars) in one process over loopback,
//! on a tiny paged pool with a 4-slot nursery that the stream overflows.
//!
//! One daemon (`nursery=4`, pinning on), two connections, the same
//! pre-generated stream: hints OFF (`V41_B2_MISS_PREFETCH=off`), then ON
//! (`k1`, an ORACLE look-ahead: the hint words are the next layer's actual
//! picks plus one wrong expert per lane-layer, queued through
//! `queue_hint_words` as the Route would). Asserts: every partial
//! BIT-IDENTICAL between the runs (I1: a nursery entry is served from its slot
//! like any other), zero surprises (fatal via `V41_B2_ASSERT_NO_SURPRISE`),
//! zero pinned evictions / revokes (fatal via `V41_B2_ASSERT_PINNED`), the
//! capability detected (`nursery_supported`), hints on the wire
//! (`lh2_hints_sent > 0`), landings AND promotions on box 2, `lands = hits +
//! recycled + delta(occupied)`, every reported NURSERY entry never held.
//!
//! Sizing: 4 layers x 6 seeded experts = 24 pool slots (451 MB of GTT); picks
//! over 8 experts per layer (32 > 24, so it evicts); staging 6 (one union),
//! nursery 4 (a quarter of the 18-slot main band), reserve 6: budget 24 - 6 -
//! 6 - 4 = 8, and reserve + staging = 12 >= max wants 6 + parked 6.
//!
//! Run (box 1, one GPU test process at a time, NEVER beside the live hub):
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test remote_experts_nursery_loopback -- --ignored --nocapture

use std::net::TcpListener;
use std::sync::mpsc;

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED};
use v4flash_kernels::het::b2_mirror;
use v4flash_kernels::het::lookahead::{self, Pred};
use v4flash_kernels::het::remote_experts::{
    queue_hint_words, serve_connection, Assignment, ExpertShard, MoeExecutor, NurseryCounters, PartialCounters, PinCounters,
    RemoteExpertClient, ServeOptions, SocketOptions, NO_PICK, XQ_BYTES_PER_TOKEN,
};

const LAYERS: [u32; 4] = [3, 7, 11, 15];
const SEEDED: u32 = 6;
const RANGE: u32 = 8;
const NURSERY: usize = 4;
const ROWS: usize = 64;
const STEPS: usize = 30;

fn model_dir() -> String {
    std::env::var("V41_MODEL")
        .unwrap_or_else(|_| format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap()))
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// A valid Q8_K row on the CPU (the layout the hub sends).
fn q8k_row(rng: &mut Rng) -> Vec<u8> {
    let mut out = Vec::with_capacity(XQ_BYTES_PER_TOKEN);
    for _ in 0..BLOCKS_Q8K_GATE_IN {
        let d = 0.002f32 + (rng.below(1000) as f32) * 1e-5;
        out.extend_from_slice(&d.to_le_bytes());
        let qs: Vec<i8> = (0..256).map(|_| (rng.below(255) as i32 - 127) as i8).collect();
        out.extend(qs.iter().map(|&q| q as u8));
        for g in 0..16 {
            let s: i16 = qs[g * 16..(g + 1) * 16].iter().map(|&q| q as i16).sum();
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    assert_eq!(out.len(), XQ_BYTES_PER_TOKEN);
    out
}

/// One request of the stream.
#[derive(Clone)]
struct Spec {
    layer: u32,
    b: usize,
    xq: Vec<u8>,
    sel: Vec<i32>,
    ew: Vec<f32>,
    partner: bool,
    /// A wrong hint for this lane-layer (an expert the next layer does not pick).
    wrong: u32,
}

/// `STEPS` decode steps; per layer the two lanes submit back to back (lane
/// A promises a partner), then both replies are awaited, in either order.
/// Decode-shaped only (<= 16 rows): the nursery is a decode mechanism.
fn gen_stream(seed: u64) -> Vec<Vec<(Spec, Spec, bool)>> {
    let mut rng = Rng(seed);
    let w: Vec<f64> = (1..=RANGE).map(|k| 1.0 / (k as f64).powf(1.1)).collect();
    let tot: f64 = w.iter().sum();
    let mut acc = 0.0;
    let cdf: Vec<f64> = w.iter().map(|x| { acc += x / tot; acc }).collect();
    let zipf = |rng: &mut Rng, l: u32| -> u32 {
        let x = rng.below(1_000_000) as f64 / 1e6;
        let r = cdf.iter().position(|&c| x < c).unwrap_or(RANGE as usize - 1) as u32;
        (r + l) % RANGE
    };
    let mut steps = Vec::with_capacity(STEPS);
    for _ in 0..STEPS {
        let mut layers = Vec::new();
        for &layer in &LAYERS {
            let lane = |rng: &mut Rng, partner: bool| -> Spec {
                let b = match rng.below(10) {
                    0..=5 => 1,
                    6 | 7 => 3,
                    _ => 6,
                };
                let nu = N_EXPERT_USED;
                let mut sel = vec![NO_PICK; b * nu];
                let mut ew = vec![0f32; b * nu];
                for t in 0..b {
                    let k = 1 + rng.below(3) as usize;
                    let mut row: Vec<u32> = Vec::new();
                    while row.len() < k {
                        let e = zipf(rng, layer);
                        if !row.contains(&e) {
                            row.push(e);
                        }
                    }
                    for (i, &e) in row.iter().enumerate() {
                        let s = (i * 2 + t) % nu;
                        sel[t * nu + s] = e as i32;
                        ew[t * nu + s] = 0.1 + (rng.below(900) as f32) / 1000.0;
                    }
                }
                let xq: Vec<u8> = (0..b).flat_map(|_| q8k_row(rng)).collect();
                Spec { layer, b, xq, sel, ew, partner, wrong: zipf(rng, layer) }
            };
            let a = lane(&mut rng, true);
            let b = lane(&mut rng, false);
            layers.push((a, b, rng.below(2) == 0));
        }
        steps.push(layers);
    }
    steps
}

#[derive(Debug, Default)]
struct RunStats {
    partials: Vec<Vec<u32>>,
    surprises: u64,
    paged: u64,
    hints_sent: u64,
    nursery_words: u64,
    nursery_held: u64,
}

/// The ORACLE look-ahead of lane-layer `(step, i)`: the next layer's picks of
/// both lanes as rank-1 non-resident predictions (the mirror's `held` says
/// which are non-resident), plus the lane's wrong expert.
fn oracle_preds(next: &(Spec, Spec, bool), wrong: u32) -> Vec<Pred> {
    let nl = next.0.layer as i32;
    let mut preds: Vec<Pred> = Vec::new();
    for &e in next.0.sel.iter().chain(next.1.sel.iter()).chain(std::iter::once(&(wrong as i32))) {
        if e == NO_PICK || preds.iter().any(|p| i32::from(p.e) == e) {
            continue;
        }
        let nonres = b2_mirror::lookup(nl, e as u32).is_none_or(|r| !r.held && !r.pending && !r.incoming);
        preds.push(Pred { e: e as u16, rank: 1, nonres, margin: f32::NAN });
    }
    preds
}

fn run(addr: &str, stream: &[Vec<(Spec, Spec, bool)>], hints: bool) -> eyre::Result<RunStats> {
    b2_mirror::set_pin_wanted(true);
    assert!(v4flash_kernels::knobs::B2_MISS_PREFETCH.set(if hints { "k1" } else { "off" }));
    let mut client = RemoteExpertClient::connect(addr, &SocketOptions::default())?;
    client.set_decode_phase(true);
    let mut st = RunStats::default();
    let _ = lookahead::take_stats();
    for step in stream {
        b2_mirror::begin_step();
        let rows: usize = step.iter().map(|(a, b, _)| a.b + b.b).sum();
        lookahead::begin_step(b2_mirror::step(), rows);
        let mp = lookahead::cfg();
        for (i, (a, b, b_first)) in step.iter().enumerate() {
            // The Route of this lane-layer: hint the NEXT layer's picks.
            if hints {
                if let Some(next) = step.get(i + 1) {
                    for lane in [a, b] {
                        let preds = oracle_preds(next, lane.wrong);
                        lookahead::count_preds(&preds);
                        let _ = queue_hint_words(mp.step, next.0.layer as i32, &preds, mp.rank, mp.cap as usize, mp.margin());
                    }
                }
            }
            let mut tickets = Vec::new();
            for s in [a, b] {
                let t = client
                    .submit_dispatch(true, s.layer, s.b, &s.xq, &s.sel, &s.ew, true, s.partner, Default::default())?
                    .ok_or_else(|| eyre!("request with picks not sent"))?;
                tickets.push(t);
            }
            if *b_first {
                tickets.swap(0, 1);
            }
            let mut got = Vec::new();
            for t in tickets {
                let p = client.wait(t)?;
                let (_, pinned, budget) = p.pin.ok_or_else(|| eyre!("pin mode: reply without a pin block"))?;
                assert!(pinned <= budget, "pinned {pinned} > budget {budget}");
                st.surprises += u64::from(p.n_surprise);
                st.paged += u64::from(p.n_paged);
                got.push((t.seq, p.f32().iter().map(|x| x.to_bits()).collect::<Vec<u32>>()));
                client.recycle(p);
            }
            got.sort_by_key(|g| g.0);
            st.partials.extend(got.into_iter().map(|g| g.1));
            // The mirror's NURSERY bits are dedup state, never held (I2).
            for &l in &LAYERS {
                for e in 0..RANGE {
                    if b2_mirror::nursery(l as i32, e) {
                        st.nursery_words += 1;
                        st.nursery_held += u64::from(b2_mirror::lookup(l as i32, e).is_some_and(|r| r.held));
                    }
                }
            }
        }
        let s = lookahead::take_stats();
        st.hints_sent += s[lookahead::Stat::HintsSent as usize];
    }
    drop(client);
    Ok(st)
}

#[test]
#[ignore]
fn remote_experts_nursery_loopback() -> eyre::Result<()> {
    install_panic_handler()?;
    // Knobs are read once, before anything below touches them.
    for (k, v) in [
        ("V41_B2_PIN_RESERVE", "6"),
        ("V41_B2_PREFILL_STAGE", "6"),
        ("V41_B2_PIN_HEADROOM", "2"),
        ("V41_B2_PIN_DECAY_STEPS", "8"),
        ("V41_B2_PREFETCH_SETS", "4"),
        ("V41_B2_NURSERY", "4"),
        ("V41_B2_PARK", "1"),
        ("V41_B2_HITS_FIRST", "1"),
        ("V41_B2_ASSERT_PINNED", "1"),
        ("V41_B2_ASSERT_NO_SURPRISE", "1"),
    ] {
        std::env::set_var(k, v);
    }
    std::env::remove_var("V41_SUB");
    let (tx_done, rx_done) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        if rx_done.recv_timeout(std::time::Duration::from_secs(1200)).is_err() {
            eprintln!("remote_experts_nursery_loopback: WATCHDOG: no progress in 20 min (deadlock?)");
            std::process::exit(3);
        }
    });

    let dir = model_dir();
    let igpu = pick_igpu()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?.to_string();
    let (tx_ready, rx_ready) = mpsc::channel::<eyre::Result<()>>();
    // Per connection: the pin counters, the nursery counters + occupied level,
    // and the two-phase landing stats (`[gate/up, down]` wait ns, partial
    // counters, partial slots now).
    #[allow(clippy::type_complexity)]
    let (tx_stats, rx_stats) = mpsc::channel::<(Option<(PinCounters, u32, u32, u32)>, (NurseryCounters, u32), bool, ([u64; 2], PartialCounters, u32, [u64; 2]))>();
    let daemon = std::thread::spawn(move || -> eyre::Result<()> {
        let setup = (|| -> eyre::Result<(ExpertShard, MoeExecutor)> {
            let hf = V41HfWeights::open(&dir, None)?;
            let spec: Vec<String> = LAYERS.iter().map(|l| format!("L{l}:0-{}", SEEDED - 1)).collect();
            let asg = Assignment::parse(&spec.join(","))?;
            let mut shard = ExpertShard::load(hf, igpu, &asg, 4, 5, ROWS as u32, 4)?;
            shard.enable_paging()?;
            let exec = MoeExecutor::new(igpu, ROWS, 4)?;
            Ok((shard, exec))
        })();
        let (mut shard, mut exec) = match setup {
            Ok(v) => {
                tx_ready.send(Ok(())).unwrap();
                v
            }
            Err(e) => {
                tx_ready.send(Err(eyre!("{e:#}"))).unwrap();
                return Err(e);
            }
        };
        let opts = ServeOptions { socket: SocketOptions::default(), verbose: false, log_every: 0, keep_warm_us: 250, re_anchor_every: 512 };
        for run in 0..3 {
            // Run 3: the two-phase landing with gate/up-only hints (design 3.4);
            // the knobs are live, flipped in-process before the connection.
            let two_phase = run == 2;
            v4flash_kernels::het::remote_experts::knobs::LAND_TWO_PHASE.set(if two_phase { "1" } else { "0" });
            v4flash_kernels::het::remote_experts::knobs::LIKELY_GATEUP_ONLY.set(if two_phase { "1" } else { "0" });
            let (stream, _) = listener.accept()?;
            serve_connection(stream, &mut shard, &mut exec, &opts, None)?;
            tx_stats.send((shard.pin_counters(), shard.nursery_counters(), shard.nursery_on(), shard.two_phase_stats())).unwrap();
        }
        Ok(())
    });
    rx_ready.recv()??;

    let stream = gen_stream(0x5eed_0002_b2b2_0002);
    let n_req: usize = stream.iter().map(|s| 2 * s.len()).sum();
    let off = run(&addr, &stream, false)?;
    let (pins_off, (nc_off, occ_off), nursery_on, _) = rx_stats.recv()?;
    let supported_before = b2_mirror::nursery_supported();
    let on = run(&addr, &stream, true)?;
    let (pins_on, (nc_on, occ_on), _, tp_on) = rx_stats.recv()?;
    let supported_after = b2_mirror::nursery_supported();
    // Run 3: two-phase landing + gate/up-only hints (design 3.4).
    let two = run(&addr, &stream, true)?;
    let (pins_two, (nc_two, _), _, tp_two) = rx_stats.recv()?;
    daemon.join().map_err(|_| eyre!("daemon panicked"))??;
    let _ = tx_done.send(());
    eprintln!("two-phase: pins {pins_two:?}; nursery {nc_two:?}; partial {:?} waits gate/up {:.1} ms down {:.1} ms (exposed {}/{}); paged {} surprises {}",
        tp_two.1, tp_two.0[0] as f64 / 1e6, tp_two.0[1] as f64 / 1e6, tp_two.3[0], tp_two.3[0] + tp_two.3[1], two.paged, two.surprises);

    eprintln!("hints OFF: pins {pins_off:?}; nursery {nc_off:?} occupied {occ_off}; paged {} surprises {}", off.paged, off.surprises);
    eprintln!(
        "hints ON : pins {pins_on:?}; nursery {nc_on:?} occupied {occ_on}; paged {} surprises {} hints_sent {} mirror nursery words {} (held {})",
        on.paged, on.surprises, on.hints_sent, on.nursery_words, on.nursery_held
    );
    assert!(nursery_on, "the daemon's nursery is on");
    assert_eq!(off.partials.len(), n_req);
    assert_eq!(on.partials.len(), n_req);
    let mismatched = off.partials.iter().zip(&on.partials).filter(|(a, b)| a != b).count();
    assert_eq!(mismatched, 0, "hints changed {mismatched} of {n_req} partials (I1)");
    assert!(off.partials.iter().all(|p| p.len() % N_EMBD as usize == 0));
    // Hints off: `k1` was not asked, so the capability stayed unknown and no
    // LIKELY word went; the nursery saw no landing.
    assert!(!supported_before, "the capability is only probed under k1/k2");
    assert_eq!(off.hints_sent, 0);
    assert_eq!(nc_off.lands, 0, "{nc_off:?}");
    // Hints on: detected, sent, landed, promoted; the invariant across the run.
    assert!(supported_after, "box 2 answered REQ_FLAG_LIKELY with a NURSERY block");
    assert!(on.hints_sent > 20, "hints on the wire: {}", on.hints_sent);
    let d = |a: u64, b: u64| a - b;
    let (lands, hits, recycled) = (d(nc_on.lands, nc_off.lands), d(nc_on.hits, nc_off.hits), d(nc_on.recycled, nc_off.recycled));
    assert!(lands > 10 && hits > 0, "landed {lands}, promoted {hits}");
    assert_eq!(lands as i64, hits as i64 + recycled as i64 + (i64::from(occ_on) - i64::from(occ_off)), "lands = hits + recycled + delta(occupied): {nc_on:?}");
    assert_eq!(nc_on.drops, nc_off.drops, "no LIKELY word dropped ({} sets)", 4);
    assert!(on.nursery_words > 0, "the mirror saw NURSERY entries");
    assert_eq!(on.nursery_held, 0, "a nursery entry is never held (I2)");
    // The pin contract held with hints on: zero surprises, zero pinned
    // evictions / revokes, budget minus the nursery.
    let (c, _pinned, budget, _epoch) = pins_on.ok_or_else(|| eyre!("pinning never turned on"))?;
    assert_eq!(budget, 24 - 6 - 6 - NURSERY as u32);
    assert_eq!((on.surprises, off.surprises), (0, 0));
    assert_eq!((c.pinned_evictions, c.revokes), (0, 0), "{c:?}");
    // Two-phase landing (I1 again): every partial bit-identical to the plain
    // run; gate/up-only hints landed PARTIAL and were promoted + completed by
    // the pass; no partial slot left dangling; the pin contract held.
    assert_eq!(tp_on.1, PartialCounters::default(), "run 2 made no partial entry");
    let mismatched = off.partials.iter().zip(&two.partials).filter(|(a, b)| a != b).count();
    assert_eq!(mismatched, 0, "the two-phase landing changed {mismatched} of {n_req} partials (I1)");
    let pc = tp_two.1;
    assert!(pc.partial_lands > 10 && pc.partial_promotions > 0, "{pc:?}");
    assert!(pc.partial_promotions <= pc.partial_lands && pc.completions + pc.partial_evicted <= pc.partial_lands, "{pc:?}");
    assert_eq!(two.surprises, 0);
    let c2 = pins_two.ok_or_else(|| eyre!("pinning never turned on"))?.0;
    assert_eq!((c2.pinned_evictions, c2.revokes), (0, 0), "{c2:?}");
    assert!(tp_two.0[1] > 0, "the down phase ran (its wait is accounted)");
    Ok(())
}
