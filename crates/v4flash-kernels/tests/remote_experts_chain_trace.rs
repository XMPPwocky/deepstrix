//! Box-2 decode CHAIN cost on the gfx1151 iGPU, for a kernel trace (2026-09-27).
//!
//! Production box 2 (3-row decode, one request per layer) measures
//! `run - page = 162 us fixed + 89 us per distinct expert` (the per-expert term
//! is 211 GB/s, at the roof). This drives the REAL `MoeExecutor::run_path` +
//! result readback in-process on a small paged shard of one layer, so that
//! `rocprofv3 --kernel-trace --memory-copy-trace` can split the fixed cost into
//! its dispatches and the gaps between them. The daemon itself cannot be traced:
//! it has no clean shutdown, and the profiler flushes at process exit.
//!
//! Phases (in this order, each after a warm-up): `one` = every row picks the
//! same single expert (distinct 1: fixed cost + one expert), `prod` = 3 rows,
//! 1-2 picks per row from 48 experts (distinct ~4.3, the production mix).
//! Prints per-phase wall p50 for the run and the readback.
//!
//! Run (box 1, one GPU test process at a time; ~0.4 GB of GTT, well under a
//! second of iGPU time):
//!   rocprofv3 --kernel-trace --memory-copy-trace -d DIR -o kt --output-format csv -- \
//!     cargo test --release -p v4flash-kernels --features v41 --test remote_experts_chain_trace -- --ignored --nocapture

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT_USED};
use v4flash_kernels::het::remote_experts::{Assignment, ExpertShard, MoeExecutor, NO_PICK, XQ_BYTES_PER_TOKEN};

const LAYER: u32 = 20;
const N_EXP: u32 = 48;
const ROWS: usize = 16;
const B: usize = 3;

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

/// A valid Q8_K row (per 256-block `d` f32, 256 int8 `qs`, 16 int16 `bsums`).
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

/// `B` rows; row t picks `k(t)` distinct experts from `pool` in random slots.
fn picks(rng: &mut Rng, pool: &[u32], per_row: &dyn Fn(&mut Rng) -> usize) -> (Vec<i32>, Vec<f32>) {
    let nu = N_EXPERT_USED;
    let mut sel = vec![NO_PICK; B * nu];
    let mut ew = vec![0f32; B * nu];
    for t in 0..B {
        let k = per_row(rng).min(pool.len());
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < k {
            let e = pool[rng.below(pool.len() as u64) as usize];
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        for (i, &e) in chosen.iter().enumerate() {
            let s = (i * 2 + t) % nu;
            sel[t * nu + s] = e as i32;
            ew[t * nu + s] = 0.1 + rng.below(900) as f32 / 1000.0;
        }
    }
    (sel, ew)
}

fn pct(v: &mut [f64], q: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * q) as usize]
}

#[test]
#[ignore]
fn remote_experts_chain_trace() -> eyre::Result<()> {
    install_panic_handler()?;
    let igpu = pick_igpu()?;
    let hf = V41HfWeights::open(&model_dir(), None)?;
    let asg = Assignment::parse(&format!("L{LAYER}:0-{}", N_EXP - 1))?;
    let mut shard = ExpertShard::load(hf, igpu, &asg, 4, 5, ROWS as u32, 1)?;
    shard.enable_paging()?;
    let mut exec = MoeExecutor::new(igpu, ROWS, 1)?;
    let pool: Vec<u32> = shard.owned_ids(LAYER).to_vec();
    assert_eq!(pool.len(), N_EXP as usize, "all experts resident");
    let mut rng = Rng(0x0b2c_4a17_7e57_0001);
    let xq: Vec<u8> = (0..B).flat_map(|_| q8k_row(&mut rng)).collect();
    let mut out = vec![0u16; B * N_EMBD as usize];
    let mut noop = |_: &mut ExpertShard, _: bool| -> eyre::Result<()> { Ok(()) };
    let one = [pool[7]];
    let phases: [(&str, &[u32], fn(&mut Rng) -> usize, usize); 2] = [
        ("one", &one, |_| 1, 300),
        ("prod", &pool, |r| 1 + r.below(2) as usize, 600),
    ];
    for (name, p, k, n) in phases {
        let mut run_us = Vec::with_capacity(n);
        let mut rb_us = Vec::with_capacity(n);
        let mut distinct = 0usize;
        for i in 0..(n + 50) {
            let (sel, ew) = picks(&mut rng, p, &k);
            let t0 = std::time::Instant::now();
            exec.run_path(&mut shard, LAYER, B, &xq, &sel, &ew, true, &mut noop)?;
            let t1 = std::time::Instant::now();
            exec.read_f16_at(0, B, &mut out)?;
            let t2 = std::time::Instant::now();
            if i >= 50 {
                run_us.push((t1 - t0).as_secs_f64() * 1e6);
                rb_us.push((t2 - t1).as_secs_f64() * 1e6);
                let mut d: Vec<i32> = sel.iter().copied().filter(|&e| e >= 0).collect();
                d.sort_unstable();
                d.dedup();
                distinct += d.len();
            }
        }
        eprintln!(
            "phase {name}: {n} requests, distinct mean {:.2}; host wall p50 run {:.1} us (launch only), readback incl. sync {:.1} us",
            distinct as f64 / n as f64,
            pct(&mut run_us, 0.5),
            pct(&mut rb_us, 0.5)
        );
        // Phase separator in the kernel trace: a long idle gap.
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
    Ok(())
}
