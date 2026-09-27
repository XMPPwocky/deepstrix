//! Box-2 PINNING end to end on the gfx1151 iGPU (`het::b2_mirror` module doc,
//! PINNING; the block above `remote_experts::PinBook`). The REAL daemon loop
//! (`serve_connection`: merge, park + hits-first, early page, prefetch
//! readers, the victim search and choke point) and the REAL hub side
//! (`RemoteExpertClient` + `b2_mirror`: negotiation, held-at-submit, release
//! words at `begin_step`, epoch masking, the surprise check) in one process
//! over loopback, on a tiny paged pool that the stream overflows.
//!
//! One daemon, two connections, the same pre-generated stream: pinning OFF,
//! then ON. Asserts: every partial BIT-IDENTICAL between the runs (pinning
//! only changes which slot box 2 evicts, never what it computes), zero
//! surprises (also fatal via `V41_B2_ASSERT_NO_SURPRISE`), zero pinned
//! evictions / revokes on box 2 (fatal via `V41_B2_ASSERT_PINNED`), `pinned <=
//! budget` on every reply, releases actually flowing, and no deadlock (a
//! watchdog ends the process).
//!
//! Sizing (why these numbers): 4 layers x 5 seeded experts = 20 pool slots
//! (376 MB of GTT); picks over 6 experts per layer (24 > 20, so it evicts).
//! The no-deadlock reserve is max(pass wants) + max(parked picks) = 6 + 6 = 12
//! here, so the budget is 8; the production default (480) is sized the same
//! way for 384 experts per layer and 16-row park passes. Each miss reads one
//! 18.8 MB expert from this box's disk (O_DIRECT): a few hundred per run.
//!
//! Run (box 1, one GPU test process at a time):
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test remote_experts_pin_loopback -- --ignored --nocapture

use std::net::TcpListener;
use std::sync::mpsc;

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED};
use v4flash_kernels::het::b2_mirror;
use v4flash_kernels::het::remote_experts::{
    push_prefetch_words, serve_connection, Assignment, ExpertShard, MoeExecutor, PinCounters, RemoteExpertClient,
    ServeOptions, SocketOptions, NO_PICK, XQ_BYTES_PER_TOKEN,
};

const LAYERS: [u32; 4] = [3, 7, 11, 15];
const SEEDED: u32 = 5;
const RANGE: u32 = 6;
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

/// A valid Q8_K row on the CPU: per 256-block `d` (f32), 256 int8 `qs`, 16
/// int16 `bsums` (sums of 16 consecutive `qs`), the layout the hub sends.
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
    /// Admission words pushed before the submit (substitution admits).
    admit: Vec<u32>,
}

/// `STEPS` decode steps; per layer the two lanes submit back to back (lane
/// A promises a partner), then both replies are awaited, in either order.
fn gen_stream(seed: u64) -> Vec<Vec<(Spec, Spec, bool)>> {
    let mut rng = Rng(seed);
    // Zipf(1.1) ranks over 0..RANGE, hot ids rotated per layer.
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
                    6 => 3,
                    7 | 8 => 6,
                    _ => 20, // prefill-shaped (> 16 rows)
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
                let admit = if rng.below(8) == 0 { vec![(layer << 16) | zipf(rng, layer)] } else { Vec::new() };
                Spec { layer, b, xq, sel, ew, partner, admit }
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
    pin_replies: u64,
    held: u64,
    surprises: u64,
    paged: u64,
    max_pinned: u32,
    budget: u32,
}

fn run(addr: &str, stream: &[Vec<(Spec, Spec, bool)>], pin: bool) -> eyre::Result<RunStats> {
    b2_mirror::set_pin_wanted(pin);
    let mut client = RemoteExpertClient::connect(addr, &SocketOptions::default())?;
    client.set_decode_phase(true);
    let mut st = RunStats::default();
    for step in stream {
        b2_mirror::begin_step();
        for (a, b, b_first) in step {
            let mut tickets = Vec::new();
            for s in [a, b] {
                if !s.admit.is_empty() {
                    assert!(push_prefetch_words(&s.admit));
                }
                let t = client
                    .submit_dispatch(true, s.layer, s.b, &s.xq, &s.sel, &s.ew, true, s.partner)?
                    .ok_or_else(|| eyre!("request with picks not sent"))?;
                tickets.push(t);
            }
            if *b_first {
                tickets.swap(0, 1);
            }
            let mut got = Vec::new();
            for t in tickets {
                let p = client.wait(t)?;
                if pin {
                    let (_, pinned, budget) = p.pin.ok_or_else(|| eyre!("pin mode: reply without a pin block"))?;
                    assert!(pinned <= budget, "pinned {pinned} > budget {budget}");
                    st.max_pinned = st.max_pinned.max(pinned);
                    st.budget = budget;
                    st.pin_replies += 1;
                    st.held += u64::from(p.n_held);
                    st.surprises += u64::from(p.n_surprise);
                    st.paged += u64::from(p.n_paged);
                } else {
                    assert!(p.pin.is_none(), "pin block without asking");
                }
                got.push((t.seq, p.f32().iter().map(|x| x.to_bits()).collect::<Vec<u32>>()));
                client.recycle(p);
            }
            // Stream order (lane A, lane B), whatever the wait order.
            got.sort_by_key(|g| g.0);
            st.partials.extend(got.into_iter().map(|g| g.1));
        }
    }
    drop(client);
    Ok(st)
}

#[test]
#[ignore]
fn remote_experts_pin_loopback() -> eyre::Result<()> {
    install_panic_handler()?;
    // Knobs are read once, before anything below touches them.
    for (k, v) in [
        ("V41_B2_PIN_RESERVE", "12"),
        ("V41_B2_PIN_HEADROOM", "2"),
        ("V41_B2_PIN_DECAY_STEPS", "8"),
        ("V41_B2_PREFETCH_SETS", "4"),
        ("V41_B2_PARK", "1"),
        ("V41_B2_HITS_FIRST", "1"),
        ("V41_B2_ASSERT_PINNED", "1"),
        ("V41_B2_ASSERT_NO_SURPRISE", "1"),
    ] {
        std::env::set_var(k, v);
    }
    std::env::remove_var("V41_SUB");
    // No deadlock: a hung run ends the process instead of the CI slot.
    let (tx_done, rx_done) = mpsc::channel::<()>();
    std::thread::spawn(move || {
        if rx_done.recv_timeout(std::time::Duration::from_secs(1200)).is_err() {
            eprintln!("remote_experts_pin_loopback: WATCHDOG: no progress in 20 min (deadlock?)");
            std::process::exit(3);
        }
    });

    let dir = model_dir();
    let igpu = pick_igpu()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?.to_string();
    let (tx_ready, rx_ready) = mpsc::channel::<eyre::Result<()>>();
    let (tx_pins, rx_pins) = mpsc::channel::<Option<(PinCounters, u32, u32, u32)>>();
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
        for _ in 0..2 {
            let (stream, _) = listener.accept()?;
            serve_connection(stream, &mut shard, &mut exec, &opts, None)?;
            tx_pins.send(shard.pin_counters()).unwrap();
        }
        Ok(())
    });
    rx_ready.recv()??;

    let stream = gen_stream(0x5eed_0001_b2b2_0001);
    let n_req: usize = stream.iter().map(|s| 2 * s.len()).sum();
    let off = run(&addr, &stream, false)?;
    let pins_off = rx_pins.recv()?;
    let (s0, h0, r0) = b2_mirror::pin_totals();
    let on = run(&addr, &stream, true)?;
    let pins_on = rx_pins.recv()?;
    let (s1, h1, r1) = b2_mirror::pin_totals();
    daemon.join().map_err(|_| eyre!("daemon panicked"))??;
    let _ = tx_done.send(());

    eprintln!("pin OFF: box 2 pins {pins_off:?}");
    eprintln!(
        "pin ON : {} replies, held picks {} surprises {} paged {} max pinned {}/{}; mirror totals: surprises {} held {} released {}; box 2 {pins_on:?}",
        on.pin_replies, on.held, on.surprises, on.paged, on.max_pinned, on.budget, s1 - s0, h1 - h0, r1 - r0
    );
    assert!(pins_off.is_none(), "pinning must stay off without the flag");
    assert_eq!(off.partials.len(), n_req);
    assert_eq!(on.partials.len(), n_req);
    let mismatched = off.partials.iter().zip(&on.partials).filter(|(a, b)| a != b).count();
    assert_eq!(mismatched, 0, "pinning changed {mismatched} of {n_req} partials");
    assert!(off.partials.iter().all(|p| p.len() % N_EMBD as usize == 0));
    let (c, _pinned, budget, _epoch) = pins_on.ok_or_else(|| eyre!("pinning never turned on"))?;
    assert_eq!(budget, 20 - 12);
    assert_eq!(on.surprises, 0);
    assert_eq!(s1 - s0, 0, "mirror counted surprises");
    assert_eq!((c.pinned_evictions, c.revokes), (0, 0), "{c:?}");
    // Not vacuous: held picks were checked, pins and releases flowed, box 2
    // paged (the pool overflowed).
    assert_eq!(on.pin_replies as usize, n_req);
    assert!(on.held > 50 && on.paged > 10, "held {} paged {}", on.held, on.paged);
    assert!(c.new_pins > 10 && c.releases > 0 && r1 - r0 > 0, "{c:?} released {}", r1 - r0);
    assert_eq!(c.releases, r1 - r0 - b2_mirror_unsent(), "box 2 applied every release word the hub sent");
    Ok(())
}

/// Release words still queued on the hub when the run ended (never sent).
fn b2_mirror_unsent() -> u64 {
    b2_mirror::take_release_words(1 << 20).len() as u64
}
