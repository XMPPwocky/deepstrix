//! Remote expert executor — loopback test on one box (docs/v41/REMOTE_EXPERTS.md).
//!
//! A daemon thread loads a TINY shard (2 layers × 8 experts ≈ 300 MB) of the
//! V4.1 checkpoint on the gfx1151 iGPU and serves 127.0.0.1; the test thread
//! connects through `RemoteExpertClient` and compares every remote partial
//! against an INDEPENDENT local computation: the same het-split kernels driven
//! over a dense slot==id buffer with the test's own remap and its own launch
//! sequence (decode kernels at B=1, the by-expert kwide chain at B>4). Then it
//! checks that remote(A) + local(B) over a disjoint split of a token's six
//! picks reproduces the plain (non-split) MoE within fp32 reassociation, and
//! times the loopback round trips at B = 1, 4, 1024.
//!
//! Memory on box 1 (the production server owns most of the RAM): two shard
//! copies (2 × 0.3 GB) + two executors' scratch (2 × 0.25 GB at rows=1024)
//! ≈ 1.1 GB total. Never loads more than 16 experts.
//!
//! Run (box 1):
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test --release -p v4flash-kernels \
//!     --features v41 --test remote_experts_loopback -- --ignored --nocapture

use std::net::TcpListener;
use std::sync::mpsc;
use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{V41HfWeights, WeightSrc};
use v4flash_hip::{install_panic_handler, Device, DeviceBuffer};
use v4flash_kernels::config::{
    BLOCKS_Q8K_DOWN_IN, BLOCKS_Q8K_GATE_IN, N_EMBD, N_EXPERT, N_EXPERT_USED, N_FF_EXP, SWIGLU_CLAMP_EXP,
};
use v4flash_kernels::het::dispatch;
use v4flash_kernels::het::engine::DeviceEngine;
use v4flash_kernels::het::remote_experts::{
    expert_geometry, f32_to_f16_bits, Assignment, ExpertShard, MoeExecutor, RemoteExpertClient,
    ServeOptions, SocketOptions, CHUNK_SIZE, MIDQ_BYTES_PER_SLOT, NO_PICK, REMAP_LEN, SENTINEL_EXPERT,
    XQ_BYTES_PER_TOKEN,
};

const LAYER_A: u32 = 3;
const LAYER_B: u32 = 7;
const N_SUB: usize = 8; // experts 0..8 of each layer
const ROWS: usize = 1024;

fn model_dir() -> String {
    std::env::var("V41_MODEL").unwrap_or_else(|_| {
        format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap())
    })
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
    fn f32(&mut self) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

/// Dense slot==id copy of experts `0..N_SUB` of one layer, loaded by the test
/// itself (independent of `ExpertShard`), plus a remap that owns exactly those.
struct DenseRef {
    gate: DeviceBuffer<u8>,
    up: DeviceBuffer<u8>,
    down: DeviceBuffer<u8>,
    remap: DeviceBuffer<i32>,
    gbpe: usize,
    ubpe: usize,
    dbpe: usize,
    gdt: v4flash_core::gguf::GgufType,
    ddt: v4flash_core::gguf::GgufType,
}

impl DenseRef {
    fn load(hf: &V41HfWeights, igpu: Device, layer: u32) -> eyre::Result<Self> {
        igpu.set_current()?;
        let src = WeightSrc::from(hf);
        let [g, u, d] = expert_geometry(&src)?;
        let load_role = |which: &str, bpe: usize| -> eyre::Result<DeviceBuffer<u8>> {
            let name = format!("blk.{layer}.ffn_{which}_exps.weight");
            let t = src.tensor(&name).ok_or_else(|| eyre!("{name}"))?;
            let mut host = vec![0u8; N_SUB * bpe];
            for e in 0..N_SUB {
                src.read_expert_into(t, e, &mut host[e * bpe..(e + 1) * bpe])?;
            }
            let mut buf = DeviceBuffer::<u8>::new(igpu.id, host.len())?;
            buf.copy_from_host(&host)?;
            Ok(buf)
        };
        let mut remap_h = vec![0i32; REMAP_LEN];
        for e in 0..N_SUB {
            remap_h[e] = -(e as i32) - 1;
        }
        let mut remap = DeviceBuffer::<i32>::new(igpu.id, REMAP_LEN)?;
        remap.copy_from_host(&remap_h)?;
        Ok(Self {
            gate: load_role("gate", g.3)?,
            up: load_role("up", u.3)?,
            down: load_role("down", d.3)?,
            remap,
            gbpe: g.3,
            ubpe: u.3,
            dbpe: d.3,
            gdt: g.0,
            ddt: d.0,
        })
    }
}

/// Independent reference scratch + launch sequences (mirrors the hub's paths).
struct RefExec {
    e: DeviceEngine,
    xq: DeviceBuffer<u8>,
    sel: DeviceBuffer<i32>,
    ew: DeviceBuffer<f32>,
    mid: DeviceBuffer<f32>,
    midq: DeviceBuffer<u8>,
    out: DeviceBuffer<f32>,
    group_count: DeviceBuffer<i32>,
    expert_members: DeviceBuffer<i32>,
    work_items: DeviceBuffer<i32>,
    n_work_items: DeviceBuffer<i32>,
    partials: DeviceBuffer<f32>,
    out16: DeviceBuffer<u16>,
}

impl RefExec {
    fn new(igpu: Device) -> eyre::Result<Self> {
        igpu.set_current()?;
        let arch = igpu.properties()?.gcn_arch_name;
        let id = igpu.id;
        let nu = N_EXPERT_USED;
        Ok(Self {
            e: DeviceEngine::for_arch(igpu, &arch)?,
            xq: DeviceBuffer::new(id, ROWS * XQ_BYTES_PER_TOKEN)?,
            sel: DeviceBuffer::new(id, ROWS * nu)?,
            ew: DeviceBuffer::new(id, ROWS * nu)?,
            mid: DeviceBuffer::new(id, ROWS * nu * N_FF_EXP as usize)?,
            midq: DeviceBuffer::new(id, ROWS * nu * MIDQ_BYTES_PER_SLOT)?,
            out: DeviceBuffer::new(id, ROWS * N_EMBD as usize)?,
            group_count: DeviceBuffer::new(id, N_EXPERT as usize)?,
            expert_members: DeviceBuffer::new(id, N_EXPERT as usize * ROWS)?,
            work_items: DeviceBuffer::new(id, N_EXPERT as usize + ROWS * nu)?,
            n_work_items: DeviceBuffer::new(id, 1)?,
            partials: DeviceBuffer::new(id, ROWS * nu * N_EMBD as usize)?,
            out16: DeviceBuffer::new(id, ROWS * N_EMBD as usize)?,
        })
    }

    fn upload(&mut self, b: usize, xq: &[u8], sel: &[i32], ew: &[f32]) -> eyre::Result<()> {
        let nu = N_EXPERT_USED;
        let sel_s: Vec<i32> = sel.iter().map(|&e| if e == NO_PICK { SENTINEL_EXPERT } else { e }).collect();
        self.xq.slice_view_mut(0, b * XQ_BYTES_PER_TOKEN).copy_from_host(xq)?;
        self.sel.slice_view_mut(0, b * nu).copy_from_host(&sel_s)?;
        self.ew.slice_view_mut(0, b * nu).copy_from_host(ew)?;
        Ok(())
    }

    /// Het-split partial over the owned picks (sentinel-padded), decode kernels.
    fn decode_partial(&mut self, r: &DenseRef, b: usize, xq: &[u8], sel: &[i32], ew: &[f32]) -> eyre::Result<Vec<f32>> {
        let nu = N_EXPERT_USED;
        self.upload(b, xq, sel, ew)?;
        let s = &self.e.compute;
        for t in 0..b {
            let xq_t = self.xq.slice_view(t * XQ_BYTES_PER_TOKEN, XQ_BYTES_PER_TOKEN);
            let sel_t = self.sel.slice_view(t * nu, nu);
            let ew_t = self.ew.slice_view(t * nu, nu);
            let mut mid_t = self.mid.slice_view_mut(t * nu * N_FF_EXP as usize, nu * N_FF_EXP as usize);
            let mut midq_t = self.midq.slice_view_mut(t * nu * MIDQ_BYTES_PER_SLOT, nu * MIDQ_BYTES_PER_SLOT);
            let mut out_t = self.out.slice_view_mut(t * N_EMBD as usize, N_EMBD as usize);
            dispatch::moe_gate_up_batch_hetsplit(
                &self.e, r.gdt, s, &mut mid_t, &r.gate, &r.up, &xq_t, &ew_t, &sel_t, &r.remap, 0, nu as u32,
                r.gbpe as u32, r.ubpe as u32, nu as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
            )?;
            self.e.q8k.launch(s, &mut midq_t, &mid_t, BLOCKS_Q8K_DOWN_IN * nu as u32)?;
            dispatch::moe_down_batched_hetsplit(
                &self.e, r.ddt, s, &mut out_t, &r.down, &midq_t, &sel_t, &r.remap, 0, nu as u32, r.dbpe as u32,
                MIDQ_BYTES_PER_SLOT as u32, nu as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
            )?;
        }
        s.synchronize()?;
        let mut v = vec![0f32; b * N_EMBD as usize];
        self.out.slice_view(0, v.len()).copy_to_host(&mut v)?;
        Ok(v)
    }

    /// Plain (non-split) decode MoE over all six picks (all must be in 0..N_SUB).
    fn decode_full(&mut self, r: &DenseRef, xq: &[u8], sel: &[i32], ew: &[f32]) -> eyre::Result<Vec<f32>> {
        let nu = N_EXPERT_USED;
        assert!(sel.iter().all(|&e| (0..N_SUB as i32).contains(&e)));
        self.upload(1, xq, sel, ew)?;
        let s = &self.e.compute;
        let xq_t = self.xq.slice_view(0, XQ_BYTES_PER_TOKEN);
        let sel_t = self.sel.slice_view(0, nu);
        let ew_t = self.ew.slice_view(0, nu);
        let mut mid_t = self.mid.slice_view_mut(0, nu * N_FF_EXP as usize);
        let mut midq_t = self.midq.slice_view_mut(0, nu * MIDQ_BYTES_PER_SLOT);
        let mut out_t = self.out.slice_view_mut(0, N_EMBD as usize);
        dispatch::moe_gate_up_batch(
            &self.e, r.gdt, s, &mut mid_t, &r.gate, &r.up, &xq_t, &ew_t, &sel_t, r.gbpe as u32, r.ubpe as u32,
            nu as u32, SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
        )?;
        self.e.q8k.launch(s, &mut midq_t, &mid_t, BLOCKS_Q8K_DOWN_IN * nu as u32)?;
        dispatch::moe_down_batched(
            &self.e, r.ddt, s, &mut out_t, &r.down, &midq_t, &sel_t, r.dbpe as u32, MIDQ_BYTES_PER_SLOT as u32,
            nu as u32, N_EMBD, BLOCKS_Q8K_DOWN_IN,
        )?;
        s.synchronize()?;
        let mut v = vec![0f32; N_EMBD as usize];
        out_t.copy_to_host(&mut v)?;
        Ok(v)
    }

    /// Het-split partial over the owned picks, by-expert prefill chain.
    fn batched_partial(&mut self, r: &DenseRef, b: usize, xq: &[u8], sel: &[i32], ew: &[f32]) -> eyre::Result<Vec<f32>> {
        let nu = N_EXPERT_USED;
        self.upload(b, xq, sel, ew)?;
        let s = &self.e.compute;
        let sel_v = self.sel.slice_view(0, b * nu);
        let ew_v = self.ew.slice_view(0, b * nu);
        let xq_v = self.xq.slice_view(0, b * XQ_BYTES_PER_TOKEN);
        self.group_count.fill_zero_async(s)?;
        self.e.moe_group_builder.launch_hetsplit(
            s, &mut self.group_count, &mut self.expert_members, &sel_v, &r.remap, 0, nu as u32, b as u32,
            nu as u32, N_EXPERT, ROWS as u32,
        )?;
        self.n_work_items.fill_zero_async(s)?;
        let max_items = self.work_items.len() as u32;
        self.e.moe_group_builder.launch_work_items(
            s, &mut self.work_items, &mut self.n_work_items, &self.group_count, N_EXPERT, CHUNK_SIZE, max_items,
        )?;
        s.synchronize()?;
        let mut n_wi = [0i32; 1];
        self.n_work_items.copy_to_host(&mut n_wi)?;
        let n_wi = n_wi[0] as u32;
        let mut mid_v = self.mid.slice_view_mut(0, b * nu * N_FF_EXP as usize);
        assert!(dispatch::moe_gate_up_chunked(
            &self.e, r.gdt, s, &mut mid_v, &r.gate, &r.up, &xq_v, &ew_v, &self.group_count, &self.expert_members,
            &self.work_items, n_wi, r.gbpe as u32, r.ubpe as u32, nu as u32, ROWS as u32, CHUNK_SIZE,
            SWIGLU_CLAMP_EXP, N_FF_EXP, BLOCKS_Q8K_GATE_IN,
        )?);
        let mut midq_v = self.midq.slice_view_mut(0, b * nu * MIDQ_BYTES_PER_SLOT);
        self.e.q8k.launch(s, &mut midq_v, &mid_v, BLOCKS_Q8K_DOWN_IN * nu as u32 * b as u32)?;
        let mut part_v = self.partials.slice_view_mut(0, b * nu * N_EMBD as usize);
        self.e.mxfp4.launch_by_expert_kwide2(
            s, &mut part_v, &r.down, &midq_v, &self.group_count, &self.expert_members, &self.work_items, n_wi,
            r.dbpe as u32, MIDQ_BYTES_PER_SLOT as u32, nu as u32, ROWS as u32, CHUNK_SIZE, N_EMBD, BLOCKS_Q8K_DOWN_IN,
        )?;
        let mut out_v = self.out.slice_view_mut(0, b * N_EMBD as usize);
        self.e.q2k.launch_reduce_partials_hetsplit(
            s, &mut out_v, &part_v, &sel_v, &r.remap, 0, nu as u32, nu as u32, N_EMBD, b as u32,
        )?;
        s.synchronize()?;
        let mut v = vec![0f32; b * N_EMBD as usize];
        out_v.copy_to_host(&mut v)?;
        Ok(v)
    }

    fn cast_f16(&mut self, v: &[f32]) -> eyre::Result<Vec<u16>> {
        let n = v.len();
        let mut src = self.out.slice_view_mut(0, n);
        src.copy_from_host(v)?;
        let mut o = self.out16.slice_view_mut(0, n);
        self.e.q8k.launch_cast_f16(&self.e.compute, &mut o, &src, n as u32)?;
        self.e.compute.synchronize()?;
        let mut out = vec![0u16; n];
        o.copy_to_host(&mut out)?;
        Ok(out)
    }
}

fn make_picks(rng: &mut Rng, b: usize, k: usize, pool: &[u32]) -> (Vec<i32>, Vec<f32>) {
    let nu = N_EXPERT_USED;
    let mut sel = vec![NO_PICK; b * nu];
    let mut ew = vec![0f32; b * nu];
    for t in 0..b {
        let mut chosen: Vec<u32> = Vec::new();
        while chosen.len() < k.min(pool.len()) {
            let e = pool[rng.below(pool.len())];
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        let mut slots: Vec<usize> = (0..nu).collect();
        for i in (1..nu).rev() {
            let j = rng.below(i + 1);
            slots.swap(i, j);
        }
        for (i, &e) in chosen.iter().enumerate() {
            sel[t * nu + slots[i]] = e as i32;
            ew[t * nu + slots[i]] = 0.05 + 0.95 * (rng.f32() * 0.5 + 0.5);
        }
    }
    (sel, ew)
}

fn count_mismatch_f32(a: &[f32], b: &[f32]) -> usize {
    a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count()
}

#[test]
#[ignore]
fn remote_experts_loopback() -> eyre::Result<()> {
    install_panic_handler()?;
    let dir = model_dir();
    let igpu = pick_igpu()?;
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let addr = listener.local_addr()?;
    let (tx_ready, rx_ready) = mpsc::channel::<eyre::Result<()>>();

    // ---- daemon thread: tiny shard + executor, serves exactly one connection ----
    let dir_d = dir.clone();
    let daemon = std::thread::spawn(move || -> eyre::Result<Vec<v4flash_kernels::het::remote_experts::RequestRecord>> {
        let setup = (|| -> eyre::Result<(ExpertShard, MoeExecutor)> {
            let hf = V41HfWeights::open(&dir_d, None)?;
            let asg = Assignment::parse(&format!("L{LAYER_A}:0-{},L{LAYER_B}:0-{}", N_SUB - 1, N_SUB - 1))?;
            let shard = ExpertShard::load(hf, igpu, &asg, 4, 8, ROWS as u32, 4)?;
            let exec = MoeExecutor::new(igpu, ROWS, 4)?;
            Ok((shard, exec))
        })();
        // `serve_connection` takes `&mut ExpertShard` since the paging re-thread;
        // this test stopped compiling then, so its bit-identity assertions have
        // not run since. Keep it building.
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
        let (stream, _) = listener.accept()?;
        let opts = ServeOptions { socket: SocketOptions::default(), verbose: false, log_every: 0, keep_warm_us: 250, re_anchor_every: 512 };
        v4flash_kernels::het::remote_experts::serve_connection(stream, &mut shard, &mut exec, &opts, None)
    });
    rx_ready.recv()??;

    // ---- test thread: client + independent reference ----
    let hf = V41HfWeights::open(&dir, None)?;
    let ref_a = DenseRef::load(&hf, igpu, LAYER_A)?;
    let ref_b = DenseRef::load(&hf, igpu, LAYER_B)?;
    let mut rx = RefExec::new(igpu)?;
    let mut client = RemoteExpertClient::connect(addr, &SocketOptions::default())?;
    let info = client.info().clone();
    assert_eq!(info.n_resident, 2 * N_SUB as u32);
    assert_eq!(info.owned_ids(LAYER_A), (0..N_SUB as u32).collect::<Vec<_>>());
    assert_eq!(info.owned_ids(LAYER_B), (0..N_SUB as u32).collect::<Vec<_>>());
    assert!(!info.owns(LAYER_A, N_SUB as u32) && !info.owns(0, 0));

    let mut rng = Rng(0x1234_5678_9abc_def1);
    let pool: Vec<u32> = (0..N_SUB as u32).collect();

    // Activations: random f32 quantised on the device exactly as the hub does.
    let quant = |rx: &mut RefExec, b: usize, rng: &mut Rng| -> eyre::Result<Vec<u8>> {
        let x: Vec<f32> = (0..b * N_EMBD as usize).map(|_| rng.f32()).collect();
        let mut xf = DeviceBuffer::<f32>::new(igpu.id, x.len())?;
        xf.copy_from_host(&x)?;
        let mut xqv = rx.xq.slice_view_mut(0, b * XQ_BYTES_PER_TOKEN);
        rx.e.q8k.launch(&rx.e.compute, &mut xqv, &xf, BLOCKS_Q8K_GATE_IN * b as u32)?;
        rx.e.compute.synchronize()?;
        let mut out = vec![0u8; b * XQ_BYTES_PER_TOKEN];
        xqv.copy_to_host(&mut out)?;
        Ok(out)
    };

    // 0. THE DECISIVE ISOLATION (run first, before the strict bit-identity asserts): decode chain vs batched chain, same expert
    //     weights and same inputs, computed by the two REFERENCE paths (no
    //     client, no server). If these diverge, box 2's kernels are the source
    //     of the verify's batched-vs-decode KLD; if they match, the divergence
    //     is server-level (membership/remap/catch-all), not the kernels.
    for &b in &[1usize, 6] {
        let xq = quant(&mut rx, b, &mut rng)?;
        let (sel, ew) = make_picks(&mut rng, b, 3, &pool);
        let dec = rx.decode_partial(&ref_a, b, &xq, &sel, &ew)?;
        let bat = rx.batched_partial(&ref_a, b, &xq, &sel, &ew)?;
        let scale = dec.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-9);
        let mut max_abs = 0f32;
        let mut sse = 0f64;
        for i in 0..dec.len() {
            let d = (dec[i] - bat[i]).abs();
            max_abs = max_abs.max(d);
            sse += (d as f64) * (d as f64);
        }
        let rmse = (sse / dec.len() as f64).sqrt();
        eprintln!(
            "CHAIN-DIFF L{LAYER_A} B={b}: decode vs batched  max|diff|={max_abs:.4e}  \
             max|ref|={scale:.4e}  rel={:.3e}  rmse={rmse:.4e}",
            max_abs / scale
        );
    }


    // 1. Decode path (B = 1 and 4), both layers, k = 1..=6 remote picks.
    for (layer, r) in [(LAYER_A, &ref_a), (LAYER_B, &ref_b)] {
        for &b in &[1usize, 4] {
            for k in [1usize, 3, 6] {
                let xq = quant(&mut rx, b, &mut rng)?;
                let (sel, ew) = make_picks(&mut rng, b, k, &pool);
                let want = rx.decode_partial(r, b, &xq, &sel, &ew)?;
                let want16 = rx.cast_f16(&want)?;
                let got32 = client.call(layer, b, &xq, &sel, &ew, true)?.expect("remote picks present");
                let got16 = client.call(layer, b, &xq, &sel, &ew, false)?.expect("remote picks present");
                let m32 = count_mismatch_f32(got32.f32(), &want);
                let m16 = got16.f16().iter().zip(&want16).filter(|(a, b)| a != b).count();
                let m16cpu = got16.f16().iter().zip(&want).filter(|(a, b)| **a != f32_to_f16_bits(**b)).count();
                let nz = want.iter().filter(|v| **v != 0.0).count();
                eprintln!(
                    "decode  L{layer} B={b} k={k}: f32 mismatches {m32}, f16 {m16} (vs CPU RNE {m16cpu}), nonzero {nz}/{}, rtt {}/{} us, remote gpu {} us",
                    want.len(), got32.rtt_us, got16.rtt_us, got32.t_remote_compute_us
                );
                assert_eq!(m32, 0, "decode f32 partial not bit-identical (L{layer} B={b} k={k})");
                assert_eq!(m16, 0, "decode f16 partial not bit-identical (L{layer} B={b} k={k})");
                assert!(nz > 0);
                client.recycle(got32);
                client.recycle(got16);
            }
        }
    }

    // 2. Disjoint split: remote(A) + local(complement) ≈ plain full MoE (fp32 reassociation only).
    {
        let xq = quant(&mut rx, 1, &mut rng)?;
        let (sel_full, ew_full) = make_picks(&mut rng, 1, 6, &pool);
        let full = rx.decode_full(&ref_b, &xq, &sel_full, &ew_full)?;
        let mut sel_a = sel_full.clone();
        let mut sel_b = sel_full.clone();
        for i in 0..N_EXPERT_USED {
            if i % 2 == 0 { sel_b[i] = NO_PICK } else { sel_a[i] = NO_PICK }
        }
        let remote = client.call(LAYER_B, 1, &xq, &sel_a, &ew_full, true)?.expect("picks");
        let local = rx.decode_partial(&ref_b, 1, &xq, &sel_b, &ew_full)?;
        let mut max_rel = 0f32;
        let scale = full.iter().fold(0f32, |m, v| m.max(v.abs()));
        for i in 0..full.len() {
            let s = remote.f32()[i] + local[i];
            max_rel = max_rel.max((s - full[i]).abs() / scale);
        }
        eprintln!("split   L{LAYER_B}: remote(3 picks) + local(3 picks) vs plain 6-pick MoE: max |diff|/max|full| = {max_rel:.3e}");
        assert!(max_rel < 1e-5, "disjoint partials do not sum to the full MoE: {max_rel}");
        client.recycle(remote);
    }

    // 3. Batched path (B = 64 and 1024), bit-identical to the by-expert reference chain.
    for &b in &[64usize, ROWS] {
        let xq = quant(&mut rx, b, &mut rng)?;
        let (sel, ew) = make_picks(&mut rng, b, 3, &pool);
        let want = rx.batched_partial(&ref_a, b, &xq, &sel, &ew)?;
        let want16 = rx.cast_f16(&want)?;
        let got32 = client.call(LAYER_A, b, &xq, &sel, &ew, true)?.expect("picks");
        let got16 = client.call(LAYER_A, b, &xq, &sel, &ew, false)?.expect("picks");
        let m32 = count_mismatch_f32(got32.f32(), &want);
        let m16 = got16.f16().iter().zip(&want16).filter(|(a, b)| a != b).count();
        let nz = want.iter().filter(|v| **v != 0.0).count();
        eprintln!(
            "batched L{LAYER_A} B={b}: f32 mismatches {m32}, f16 {m16}, nonzero {nz}/{}, rtt {}/{} us, remote gpu {} us, out {} KB in {} KB",
            want.len(), got32.rtt_us, got16.rtt_us, got32.t_remote_compute_us, got32.bytes_out / 1000, got16.bytes_in / 1000
        );
        assert_eq!(m32, 0, "batched f32 partial not bit-identical (B={b})");
        assert_eq!(m16, 0, "batched f16 partial not bit-identical (B={b})");
        client.recycle(got32);
        client.recycle(got16);
    }

    // 4. Masking: picks the daemon does not own are dropped client-side; a
    //    batch with no remote pick sends nothing.
    {
        let xq = quant(&mut rx, 1, &mut rng)?;
        let sel = vec![100i32, 200, 300, 8, 9, 10];
        let ew = vec![1.0f32; 6];
        assert!(client.call(LAYER_A, 1, &xq, &sel, &ew, false)?.is_none());
        assert!(client.call(0, 1, &xq, &[0, 1, 2, 3, 4, 5], &ew, false)?.is_none(), "layer 0 not owned");
        let mixed = vec![0i32, 200, 1, 300, 2, 3]; // 0,1,2,3 owned, 200/300 not
        let masked = vec![0i32, NO_PICK, 1, NO_PICK, 2, 3];
        let want = rx.decode_partial(&ref_a, 1, &xq, &masked, &ew)?;
        let got = client.call(LAYER_A, 1, &xq, &mixed, &ew, true)?.expect("owned picks");
        assert_eq!(count_mismatch_f32(got.f32(), &want), 0, "client-side masking");
        client.recycle(got);
        eprintln!("masking: ok");
    }

    // 5. Loopback timing (not the link, but the executor + framing cost).
    for &(b, depth) in &[(1usize, 1usize), (4, 1), (1024, 1), (1024, 2)] {
        let xq = quant(&mut rx, b, &mut rng)?;
        let (sel, ew) = make_picks(&mut rng, b, 3, &pool);
        let iters = if b >= 256 { 20 } else { 200 };
        for _ in 0..3 {
            let p = client.call(LAYER_B, b, &xq, &sel, &ew, false)?.unwrap();
            client.recycle(p);
        }
        let mut rtts = Vec::new();
        let mut gpus = Vec::new();
        let t0 = Instant::now();
        let mut tickets = std::collections::VecDeque::new();
        let (mut sub, mut done) = (0, 0);
        while done < iters {
            while sub < iters && tickets.len() < depth {
                tickets.push_back(client.submit(LAYER_B, b, &xq, &sel, &ew, false)?.unwrap());
                sub += 1;
            }
            let p = client.wait(tickets.pop_front().unwrap())?;
            rtts.push(p.rtt_us);
            gpus.push(p.t_remote_compute_us);
            client.recycle(p);
            done += 1;
        }
        let wall = t0.elapsed().as_secs_f64();
        rtts.sort_unstable();
        gpus.sort_unstable();
        eprintln!(
            "loopback B={b:<4} depth={depth}: rtt p50 {} us p90 {} us | remote compute p50 {} us | {:.3} ms/layer",
            rtts[rtts.len() / 2], rtts[rtts.len() * 9 / 10], gpus[gpus.len() / 2], wall * 1e3 / iters as f64
        );
    }

    // 6. Clock sync: every exchange must have carried a valid NTP quadruple.
    //    On loopback the true offset is ZERO (one box, one clock), so this is a
    //    real check of the estimator and the stamping points, not just plumbing.
    {
        let cs = client.clock();
        assert!(cs.len() > 100, "expected a clock sample per request, got {}", cs.len());
        let (omin, o50, _o90, o99, omax) = cs.spread(|s| s.offset_ns()).unwrap();
        let (dmin, d50, _d90, d99, dmax) = cs.spread(|s| s.delay_ns()).unwrap();
        eprintln!(
            "clock: {} samples | offset p50 {:.3} us (min {:.3} max {:.3}) | delay p50 {:.3} us (min {:.3} p99 {:.3} max {:.3})",
            cs.len(), o50 as f64 / 1e3, omin as f64 / 1e3, omax as f64 / 1e3,
            d50 as f64 / 1e3, dmin as f64 / 1e3, d99 as f64 / 1e3, dmax as f64 / 1e3
        );
        // Same machine ⇒ same clock ⇒ the estimator must report ~0 offset. The
        // loopback path is symmetric, so the only error is the stamping points.
        assert!(o50.abs() < 100_000, "loopback median offset should be ~0, got {o50} ns");
        assert!(d50 >= 0 && d50 < 500_000, "loopback one-way delay implausible: {d50} ns");
        assert!(o99.abs() < 5_000_000, "loopback offset p99 {o99} ns");
        for s in cs.samples() {
            assert!(s.t2 >= s.t1 && s.t3 >= s.t2 && s.t4 >= s.t3, "timestamps out of order: {s:?}");
        }
    }

    drop(client);
    let records = daemon.join().map_err(|_| eyre!("daemon thread panicked"))??;
    v4flash_kernels::het::remote_experts::summarize(&records, "daemon");
    assert!(records.len() > 10);
    Ok(())
}
