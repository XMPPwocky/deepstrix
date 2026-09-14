//! `deepstrix-expert-bench` — hub-side client bench + cross-box bit-identity check
//! for `deepstrix-expertd` (docs/v41/REMOTE_EXPERTS.md).
//!
//!   deepstrix-expert-bench --connect 10.99.0.2:7431 [--model DIR] \
//!       [--batches 1,4,1024] [--iters 200] [--depth 1] [--picks 3] \
//!       [--check-layer L --check-n 4] [--pool N] [--f32] [--batched] [--busy-poll 500]
//!       [--gap-us N]   (spin N µs between a reply and the next request: the hub's
//!                       per-layer attention time; exposes the daemon's wake-up cost)
//!       [--clock-dump FILE]  (write every raw NTP quadruple as CSV)
//!
//! Every request/response carries the NTP quadruple (t1..t4, CLOCK_MONOTONIC_RAW),
//! so each run also reports the measured clock offset between the boxes, its
//! spread, the per-sample one-way delay series, and the ns to shift box 2's
//! perfetto trace by.
//!
//! * `--check-layer L --check-n K`: load K experts of layer L (which the daemon
//!   must own) into a local shard on THIS box's gfx1151 and compare the remote
//!   partial (f32 and f16) bit for bit at B = 1, 4 and 64.
//! * batches: per-layer round trip at each B (`--iters` requests, `--picks`
//!   remote picks per token drawn from the first `--pool` owned experts of the
//!   layer — default all, i.e. a B=1024 request touches every resident expert
//!   of the layer; a small pool isolates compute from weight bandwidth —
//!   `--depth` requests in flight), reporting rtt p50/p90/p99, the daemon's own
//!   time, link time and achieved GB/s.

use std::time::Instant;

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{N_EMBD, N_EXPERT, N_EXPERT_USED};
use v4flash_kernels::het::remote_experts::{
    f32_to_f16_bits, proto, Assignment, ExpertShard, MoeExecutor, RemoteExpertClient, SocketOptions,
    NO_PICK, XQ_BYTES_PER_TOKEN,
};

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

/// Tiny deterministic PRNG (xorshift64*).
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

/// `b` tokens × N_EXPERT_USED picks: `k` distinct ids drawn from `pool` per
/// token (the rest NO_PICK), weights in (0, 1].
fn make_picks(rng: &mut Rng, b: usize, k: usize, pool: &[u32]) -> (Vec<i32>, Vec<f32>) {
    let nu = N_EXPERT_USED;
    let mut sel = vec![NO_PICK; b * nu];
    let mut ew = vec![0f32; b * nu];
    for t in 0..b {
        let mut chosen: Vec<u32> = Vec::with_capacity(k);
        while chosen.len() < k.min(pool.len()) {
            let e = pool[rng.below(pool.len())];
            if !chosen.contains(&e) {
                chosen.push(e);
            }
        }
        // Scatter into random slots so the sentinel padding is exercised at every position.
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

/// Catch-all submits bypass the advertised-ownership mask so the daemon actually
/// faults; masked submits can only request resident experts and never miss.
#[allow(clippy::too_many_arguments)]
fn submit_maybe_unmasked(
    c: &mut RemoteExpertClient,
    unmasked: bool,
    layer: u32,
    b: usize,
    xq: &[u8],
    sel: &[i32],
    ew: &[f32],
    flags: u32,
) -> eyre::Result<Option<v4flash_kernels::het::remote_experts::Ticket>> {
    if unmasked {
        c.submit_flags_unmasked(layer, b, xq, sel, ew, flags)
    } else {
        c.submit_flags(layer, b, xq, sel, ew, flags)
    }
}

fn pct(v: &mut [u32], p: f64) -> u32 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[(((v.len() - 1) as f64) * p).round() as usize]
}

struct Args {
    connect: String,
    model: String,
    batches: Vec<usize>,
    iters: usize,
    depth: usize,
    picks: usize,
    pool: usize,
    check_layer: Option<u32>,
    check_n: usize,
    f32_resp: bool,
    batched: bool,
    catchall: bool,
    gap_us: u64,
    clock_dump: Option<String>,
    socket: SocketOptions,
}

fn parse_args() -> eyre::Result<Args> {
    let mut a = Args {
        connect: String::new(),
        model: std::env::var("V41_MODEL").unwrap_or_else(|_| {
            format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap_or_default())
        }),
        batches: vec![1, 4, 1024],
        iters: 200,
        depth: 1,
        picks: 3,
        pool: usize::MAX,
        check_layer: None,
        check_n: 4,
        f32_resp: false,
        batched: false,
        catchall: false,
        gap_us: 0,
        clock_dump: None,
        socket: SocketOptions::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| eyre!("{k} needs a value"));
        match k.as_str() {
            "--connect" => a.connect = val()?,
            "--model" => a.model = val()?,
            "--batches" => a.batches = val()?.split(',').map(|s| s.trim().parse()).collect::<Result<_, _>>()?,
            "--iters" => a.iters = val()?.parse()?,
            "--depth" => a.depth = val()?.parse::<usize>()?.max(1),
            "--picks" => a.picks = val()?.parse::<usize>()?.clamp(1, N_EXPERT_USED),
            "--pool" => a.pool = val()?.parse::<usize>()?.max(1),
            "--check-layer" => a.check_layer = Some(val()?.parse()?),
            "--check-n" => a.check_n = val()?.parse()?,
            "--f32" => a.f32_resp = true,
            "--batched" => a.batched = true,
            "--catchall" => a.catchall = true,
            "--gap-us" => a.gap_us = val()?.parse()?,
            "--clock-dump" => a.clock_dump = Some(val()?),
            "--busy-poll" => a.socket.busy_poll_us = val()?.parse()?,
            "--sndbuf" => a.socket.sndbuf = val()?.parse()?,
            "--rcvbuf" => a.socket.rcvbuf = val()?.parse()?,
            "--no-quickack" => a.socket.quickack = false,
            "--quickack" => a.socket.quickack = true,
            other => return Err(eyre!("unknown argument {other}")),
        }
    }
    if a.connect.is_empty() {
        return Err(eyre!("--connect host:port is required"));
    }
    Ok(a)
}

fn main() -> eyre::Result<()> {
    install_panic_handler()?;
    let args = parse_args()?;
    let mut client = RemoteExpertClient::connect(&args.connect, &args.socket)?;
    let info = client.info().clone();
    let owned_layers: Vec<u32> = (0..info.n_layer).filter(|&l| info.owned_count(l) > 0).collect();
    eprintln!(
        "bench: connected to {} — {} resident experts ({:.1} GB), {} layers owned {:?}, max_batch {}, decode_max_b {}",
        args.connect,
        info.n_resident,
        info.n_resident as f64 * info.bytes_per_expert as f64 / 1e9,
        owned_layers.len(),
        owned_layers.iter().map(|&l| format!("L{l}:{}", info.owned_count(l))).collect::<Vec<_>>(),
        info.max_batch,
        info.decode_max_b
    );
    if owned_layers.is_empty() {
        return Err(eyre!("daemon owns nothing"));
    }
    let igpu = pick_igpu()?;
    let rows = args.batches.iter().copied().max().unwrap_or(1).max(64).min(info.max_batch as usize);
    let mut exec = MoeExecutor::new(igpu, rows, info.decode_max_b as usize)?;
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);

    // ---- bit-identity check against a local shard of a few of the daemon's experts ----
    if let Some(layer) = args.check_layer {
        let ids = info.owned_ids(layer);
        if ids.is_empty() {
            return Err(eyre!("check: daemon does not own layer {layer}"));
        }
        let ids: Vec<u32> = ids.into_iter().take(args.check_n).collect();
        eprintln!("check: loading local shard L{layer}:{ids:?} from {}", args.model);
        let hf = V41HfWeights::open(&args.model, None)?;
        let asg = Assignment { layers: vec![(layer, ids.clone())] };
        let mut local = ExpertShard::load(hf, igpu, &asg, 4, 8, rows as u32, info.decode_max_b as u32)?;
        let mut all_ok = true;
        for &b in &[1usize, 4, 64] {
            let b = b.min(rows);
            let x: Vec<f32> = (0..b * N_EMBD as usize).map(|_| rng.f32()).collect();
            let mut xq = vec![0u8; b * XQ_BYTES_PER_TOKEN];
            exec.quantize_q8k(&x, &mut xq)?;
            let (sel, ew) = make_picks(&mut rng, b, args.picks.min(ids.len()), &ids);
            exec.run(&mut local, layer, b, &xq, &sel, &ew)?;
            let mut ref32 = vec![0f32; b * N_EMBD as usize];
            exec.read_f32(b, &mut ref32)?;
            let mut ref16 = vec![0u16; b * N_EMBD as usize];
            exec.read_f16(b, &mut ref16)?;
            let r32 = client.call(layer, b, &xq, &sel, &ew, true)?.ok_or_else(|| eyre!("no remote picks?"))?;
            let r16 = client.call(layer, b, &xq, &sel, &ew, false)?.ok_or_else(|| eyre!("no remote picks?"))?;
            let n32 = r32.f32().iter().zip(&ref32).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
            let n16 = r16.f16().iter().zip(&ref16).filter(|(a, b)| a != b).count();
            let n16cpu = r16.f16().iter().zip(&ref32).filter(|(a, b)| **a != f32_to_f16_bits(**b)).count();
            let nz = ref32.iter().filter(|v| **v != 0.0).count();
            let ok = n32 == 0 && n16 == 0;
            all_ok &= ok;
            eprintln!(
                "check: L{layer} B={b:<3} {} — f32 mismatches {n32}, f16 mismatches {n16} (vs CPU RNE {n16cpu}), {nz}/{} nonzero, rtt {} us / {} us",
                if ok { "BIT-IDENTICAL" } else { "MISMATCH" },
                ref32.len(),
                r32.rtt_us,
                r16.rtt_us
            );
            client.recycle(r32);
            client.recycle(r16);
        }
        if !all_ok {
            return Err(eyre!("check FAILED"));
        }
    }

    // ---- timing sweeps ----
    println!(
        "{:>6} {:>6} {:>5} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>9}",
        "B", "iters", "depth", "out KB", "in KB", "rtt p50", "rtt p90", "rtt p99", "srv p50", "gpu p50", "link p50", "ms/layer", "GB/s"
    );
    println!("(picks/token {}, pool {}, response {}{})", args.picks, if args.pool == usize::MAX { "all".to_string() } else { args.pool.to_string() }, if args.f32_resp { "f32" } else { "f16" }, if args.batched { ", forced batched path" } else { "" });
    if args.catchall { println!("(CATCH-ALL: fresh picks/iter drawn from all {N_EXPERT} ids — the daemon pages from disk)"); }
    if args.gap_us > 0 {
        println!("(gap {} us between reply and next request; ms/layer and GB/s exclude the gap)", args.gap_us);
    }
    let flags = (if args.f32_resp { proto::REQ_FLAG_RESP_F32 } else { 0 }) | (if args.batched { proto::REQ_FLAG_BATCHED } else { 0 });
    for &b in &args.batches {
        if b > rows {
            eprintln!("bench: skipping B={b} > rows {rows}");
            continue;
        }
        let x: Vec<f32> = (0..b * N_EMBD as usize).map(|_| rng.f32()).collect();
        let mut xq = vec![0u8; b * XQ_BYTES_PER_TOKEN];
        exec.quantize_q8k(&x, &mut xq)?;
        // Per-layer pick sets (layers round-robin over what the daemon owns).
        let n_layers_used = owned_layers.len().min(8);
        // `--catchall`: draw from ALL N_EXPERT ids, not just the ones the daemon
        // owns, and draw FRESH picks for every iteration. Both matter. The default
        // path builds one pick set per layer and replays it, so after the warm-up
        // every expert is resident and the miss path is never exercised; and a pool
        // of owned ids can only ever hit. This arm makes the daemon page from its
        // own disk on nearly every request, which is what a real decode does under
        // T2 catch-all — and it measures that cost with NO box-1 weight load.
        let picks: Vec<(u32, Vec<i32>, Vec<f32>)> = if args.catchall {
            let all: Vec<u32> = (0..N_EXPERT).collect();
            (0..args.iters.max(n_layers_used))
                .map(|i| {
                    let l = owned_layers[i % n_layers_used];
                    let (s, w) = make_picks(&mut rng, b, args.picks, &all);
                    (l, s, w)
                })
                .collect()
        } else {
            (0..n_layers_used)
                .map(|i| {
                    let l = owned_layers[i];
                    let mut pool = info.owned_ids(l);
                    pool.truncate(args.pool);
                    let (s, w) = make_picks(&mut rng, b, args.picks, &pool);
                    (l, s, w)
                })
                .collect()
        };
        // Warm-up.
        for i in 0..4 {
            let (l, s, w) = &picks[i % picks.len()];
            if let Some(t) = submit_maybe_unmasked(&mut client, args.catchall, *l, b, &xq, s, w, flags)? {
                let p = client.wait(t)?;
                client.recycle(p);
            }
        }
        let mut rtts = Vec::with_capacity(args.iters);
        let mut srv = Vec::with_capacity(args.iters);
        let mut gpu = Vec::with_capacity(args.iters);
        let mut link = Vec::with_capacity(args.iters);
        let (mut bytes_out, mut bytes_in) = (0usize, 0usize);
        let t_wall = Instant::now();
        let mut tickets = std::collections::VecDeque::new();
        let mut submitted = 0usize;
        let mut completed = 0usize;
        while completed < args.iters {
            while submitted < args.iters && tickets.len() < args.depth {
                let (l, s, w) = &picks[submitted % picks.len()];
                let t = submit_maybe_unmasked(&mut client, args.catchall, *l, b, &xq, s, w, flags)?.ok_or_else(|| eyre!("no remote picks"))?;
                tickets.push_back(t);
                submitted += 1;
            }
            let t = tickets.pop_front().unwrap();
            let p = client.wait(t)?;
            if args.gap_us > 0 {
                let t_gap = Instant::now();
                while t_gap.elapsed().as_micros() < args.gap_us as u128 {
                    std::hint::spin_loop();
                }
            }
            rtts.push(p.rtt_us);
            srv.push(p.t_remote_server_us);
            gpu.push(p.t_remote_compute_us);
            link.push(p.link_us());
            bytes_out += p.bytes_out;
            bytes_in += p.bytes_in;
            client.recycle(p);
            completed += 1;
        }
        let wall = t_wall.elapsed().as_secs_f64() - args.gap_us as f64 * 1e-6 * args.iters as f64;
        let ms_per_layer = wall * 1e3 / args.iters as f64;
        let gbs = (bytes_out + bytes_in) as f64 / 1e9 / wall;
        println!(
            "{:>6} {:>6} {:>5} {:>9.1} {:>9.1} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8.3} {:>9.3}",
            b,
            args.iters,
            args.depth,
            bytes_out as f64 / args.iters as f64 / 1e3,
            bytes_in as f64 / args.iters as f64 / 1e3,
            pct(&mut rtts, 0.5),
            pct(&mut rtts, 0.9),
            pct(&mut rtts, 0.99),
            pct(&mut srv, 0.5),
            pct(&mut gpu, 0.5),
            pct(&mut link, 0.5),
            ms_per_layer,
            gbs
        );
    }

    // ---- clock sync over the link (NTP's estimator, one sample per request) ----
    let cs = client.clock();
    if cs.is_empty() {
        eprintln!("clock: no samples (peer did not stamp t1..t3)");
        return Ok(());
    }
    eprintln!("\n{}", cs.summary());
    if let (Some((omin, o50, o90, o99, omax)), Some((dmin, d50, d90, d99, dmax))) =
        (cs.spread(|s| s.offset_ns()), cs.spread(|s| s.delay_ns()))
    {
        println!(
            "\nclock sync: {} samples over the link\n  \
             offset (box1 -> box2, us): p50 {:.3}  min {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}  | full spread {:.3} us\n  \
             one-way delay      (us): p50 {:.3}  min {:.3}  p90 {:.3}  p99 {:.3}  max {:.3}\n  \
             perfetto shift for box 2's trace: {:.3} us  ({} ns)",
            cs.len(),
            o50 as f64 / 1e3, omin as f64 / 1e3, o90 as f64 / 1e3, o99 as f64 / 1e3, omax as f64 / 1e3,
            (omax - omin) as f64 / 1e3,
            d50 as f64 / 1e3, dmin as f64 / 1e3, d90 as f64 / 1e3, d99 as f64 / 1e3, dmax as f64 / 1e3,
            cs.perfetto_shift_ns().unwrap_or(0) as f64 / 1e3,
            cs.perfetto_shift_ns().unwrap_or(0),
        );
        if let (Some(ppm), Some((r10, r50, r90))) = (cs.drift_ppm(), cs.residual_ns()) {
            println!(
                "  relative clock rate: {ppm:+.1} ppm ({:+.1} us of offset per second)\n  \
                 detrended residual (true estimator precision, offset samples with B<={}): \
                 p10 {:+.3} us  p50 {:+.3} us  p90 {:+.3} us",
                ppm, cs.offset_max_b, r10 as f64 / 1e3, r50 as f64 / 1e3, r90 as f64 / 1e3
            );
        }
        // Drift: compare the median offset of the first and last tenth.
        let n = cs.len();
        if n >= 200 {
            let k = n / 10;
            let med = |sl: &[v4flash_kernels::het::remote_experts::ClockSample]| {
                let mut v: Vec<i64> = sl.iter().map(|s| s.offset_ns()).collect();
                v.sort_unstable();
                v[v.len() / 2]
            };
            let first = med(&cs.samples()[..k]);
            let last = med(&cs.samples()[n - k..]);
            let span_s = (cs.samples()[n - 1].t1 - cs.samples()[0].t1) as f64 / 1e9;
            println!(
                "  drift: first-tenth median {:.3} us -> last-tenth {:.3} us over {:.1} s = {:.1} ppm",
                first as f64 / 1e3, last as f64 / 1e3, span_s,
                if span_s > 0.0 { (last - first) as f64 / 1e3 / span_s } else { 0.0 }
            );
        }
    }
    if let Some(path) = args.clock_dump.as_deref() {
        use std::io::Write;
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        writeln!(f, "seq,layer,b,t1,t2,t3,t4,offset_ns,delay_ns,rtt_ns,remote_service_ns")?;
        for s in cs.samples() {
            writeln!(
                f, "{},{},{},{},{},{},{},{},{},{},{}",
                s.seq, s.layer, s.b, s.t1, s.t2, s.t3, s.t4,
                s.offset_ns(), s.delay_ns(), s.rtt_ns(), s.remote_service_ns()
            )?;
        }
        eprintln!("clock: {} raw samples -> {path}", cs.len());
    }
    Ok(())
}
