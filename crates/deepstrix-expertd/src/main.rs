//! `deepstrix-expertd` — remote expert executor daemon (PLAN.md §4, docs/v41/REMOTE_EXPERTS.md).
//!
//! Loads an assigned set of routed experts from the HF checkpoint into a
//! resident iGPU pool and serves weighted MoE partial sums over TCP.
//!
//!   deepstrix-expertd --model /weights/dsv4.1f --experts L20-L39 \
//!       [--listen 0.0.0.0:7431] [--max-batch 1024] [--decode-max-b 4] \
//!       [--load-threads 8] [--load-batch 32] [--verbose] [--log-every N] \
//!       [--busy-poll 500] [--sndbuf 4194304] [--rcvbuf 4194304] [--quickack]
//!       [--keep-warm-us 250]   (0 = let the iGPU idle between requests)
//!       [--trace box2.perfetto-trace] [--machine lumi-brain2]
//!
//! Assignment grammar: `L<a>[-L<b>][:<lo>-<hi>]`, `all[:<lo>-<hi>]`, `<layer>[:<lo>-<hi>]`,
//! comma-separated (see `Assignment::parse`).

use std::net::TcpListener;

use color_eyre::eyre::{self, eyre};
use v4flash_core::V41HfWeights;
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::het::remote_experts::{
    Assignment, ExpertShard, ExpertdTracer, MoeExecutor, ServeOptions, SocketOptions, DEFAULT_PORT,
};

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 device"))
}

fn rss_gb() -> f64 {
    std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines().find(|l| l.starts_with("VmRSS:")).and_then(|l| {
                l.split_whitespace().nth(1).and_then(|k| k.parse::<f64>().ok())
            })
        })
        .map_or(0.0, |kb| kb / 1e6)
}

fn gtt_used_gb() -> f64 {
    let mut tot = 0f64;
    if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
        for e in rd.flatten() {
            let p = e.path().join("device/mem_info_gtt_used");
            if let Ok(s) = std::fs::read_to_string(&p) {
                if let Ok(v) = s.trim().parse::<f64>() {
                    tot = tot.max(v / 1e9);
                }
            }
        }
    }
    tot
}

struct Args {
    model: String,
    experts: String,
    /// Frequency-ranked placement file (see `Assignment::from_placement_file`).
    /// Takes precedence over `--experts`. Pair with `--experts-k` (average
    /// experts per layer; the real budget is `k * N_LAYER`, allocated by global
    /// count rank, so per-layer counts vary).
    experts_file: Option<String>,
    experts_k: usize,
    /// Catch-all tier: after loading, turn every layer's region into an LRU over
    /// all 384 experts, refilled from THIS box's disk. Lets the hub stop paging.
    paged: bool,
    listen: String,
    max_batch: usize,
    decode_max_b: usize,
    load_threads: usize,
    load_batch: usize,
    verbose: bool,
    log_every: usize,
    keep_warm_us: u64,
    trace: Option<String>,
    machine: String,
    socket: SocketOptions,
}

fn parse_args() -> eyre::Result<Args> {
    let mut a = Args {
        model: std::env::var("V41_MODEL").unwrap_or_else(|_| {
            format!("{}/.cache/deepstrix/models/dsv4.1f", std::env::var("HOME").unwrap_or_default())
        }),
        experts: String::new(),
        experts_file: None,
        experts_k: 0,
        paged: false,
        listen: format!("0.0.0.0:{DEFAULT_PORT}"),
        max_batch: v4flash_kernels::het::B_MAX,
        decode_max_b: 4,
        load_threads: 8,
        load_batch: 32,
        verbose: false,
        log_every: 0,
        keep_warm_us: 250,
        trace: None,
        machine: std::fs::read_to_string("/proc/sys/kernel/hostname")
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|_| "box2".into()),
        socket: SocketOptions::default(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(k) = it.next() {
        let mut val = || it.next().ok_or_else(|| eyre!("{k} needs a value"));
        match k.as_str() {
            "--model" => a.model = val()?,
            "--experts" => a.experts = val()?,
            "--experts-file" => a.experts_file = Some(val()?),
            "--paged" => a.paged = true,
            "--experts-k" => a.experts_k = val()?.parse()?,
            "--listen" => a.listen = val()?,
            "--max-batch" => a.max_batch = val()?.parse()?,
            "--decode-max-b" => a.decode_max_b = val()?.parse()?,
            "--load-threads" => a.load_threads = val()?.parse()?,
            "--load-batch" => a.load_batch = val()?.parse()?,
            "--log-every" => a.log_every = val()?.parse()?,
            "--keep-warm-us" => a.keep_warm_us = val()?.parse()?,
            "--trace" => a.trace = Some(val()?),
            "--machine" => a.machine = val()?,
            "--busy-poll" => a.socket.busy_poll_us = val()?.parse()?,
            "--sndbuf" => a.socket.sndbuf = val()?.parse()?,
            "--rcvbuf" => a.socket.rcvbuf = val()?.parse()?,
            "--no-quickack" => a.socket.quickack = false,
            "--quickack" => a.socket.quickack = true,
            "--verbose" | "-v" => a.verbose = true,
            "-h" | "--help" => {
                eprintln!("{}", include_str!("main.rs").lines().take(16).map(|l| l.trim_start_matches("//! ")).collect::<Vec<_>>().join("\n"));
                std::process::exit(0);
            }
            other => return Err(eyre!("unknown argument {other}")),
        }
    }
    if a.experts.is_empty() && a.experts_file.is_none() {
        return Err(eyre!(
            "--experts <spec> is required (e.g. L20-L39), or --experts-file <path> --experts-k <k>"
        ));
    }
    if a.experts_file.is_some() && a.experts_k == 0 {
        return Err(eyre!("--experts-file needs --experts-k <avg experts per layer>"));
    }
    if a.max_batch == 0 || a.max_batch > v4flash_kernels::het::B_MAX {
        return Err(eyre!("--max-batch must be in 1..={}", v4flash_kernels::het::B_MAX));
    }
    Ok(a)
}

fn main() -> eyre::Result<()> {
    install_panic_handler()?;
    if !cfg!(feature = "v41") {
        return Err(eyre!("deepstrix-expertd must be built with --features v41 (V4.1 experts from the HF checkpoint)"));
    }
    let args = parse_args()?;
    // A frequency-ranked placement file beats a contiguous id range by a wide
    // margin on V4.1 (Zipfian routing); see `Assignment::from_placement_file`.
    let asg = match args.experts_file.as_deref() {
        Some(path) => {
            eprintln!("expertd: placement file {path}, k_avg={}", args.experts_k);
            Assignment::from_placement_file(path, args.experts_k)?
        }
        None => Assignment::parse(&args.experts)?,
    };
    eprintln!(
        "expertd: {} experts over {} layers from {}",
        asg.n_experts(),
        asg.layers.len(),
        args.model
    );
    let igpu = pick_igpu()?;
    igpu.set_current()?;
    let t0 = std::time::Instant::now();
    let hf = V41HfWeights::open(&args.model, None)?;
    eprintln!("expertd: checkpoint opened in {:.2} s", t0.elapsed().as_secs_f64());
    // The tracer must exist before the load so the SSD expert reads are traced.
    // Its device track anchors to the executor's compute stream, so build the
    // executor first (cheap: scratch only).
    let mut exec = MoeExecutor::new(igpu, args.max_batch, args.decode_max_b)?;
    let tracer = match args.trace.as_deref() {
        Some(path) => {
            exec.enable_device_timing()?;
            let t = ExpertdTracer::open(path, igpu, &exec.engine.compute, &args.machine)?;
            eprintln!("expertd: perfetto trace -> {path} (machine {})", args.machine);
            Some(t)
        }
        None => None,
    };
    let mut shard = ExpertShard::load_traced(
        hf,
        igpu,
        &asg,
        args.load_threads,
        args.load_batch,
        args.max_batch as u32,
        args.decode_max_b as u32,
        tracer.as_ref(),
    )?;
    if args.paged {
        shard.enable_paging()?;
        eprintln!(
            "expertd: CATCH-ALL paged mode — {} slots/layer (avg) as an LRU over all {} experts, \
refilled from this box's own disk. The ADVERTISED assignment is unchanged (prefill still \
uses it); the hub decides at decode time to send us anything it does not hold.",
            shard.load_stats.n_experts / (v4flash_kernels::config::N_LAYER as usize).max(1),
            v4flash_kernels::config::N_EXPERT,
        );
    }
    eprintln!(
        "expertd: ready — {} experts ({:.2} GB) loaded in {:.1} s; RSS {:.2} GB, GTT used {:.2} GB; max_batch {} decode_max_b {}",
        shard.load_stats.n_experts,
        shard.load_stats.bytes as f64 / 1e9,
        shard.load_stats.seconds,
        rss_gb(),
        gtt_used_gb(),
        args.max_batch,
        args.decode_max_b
    );
    let hits_first = v4flash_kernels::het::remote_experts::install_hits_first_toggle();
    eprintln!("expertd: hits-first batched MoE {} (V41_B2_HITS_FIRST=1 at start; `kill -USR1 {}` flips it at runtime)",
        if hits_first { "ON" } else { "OFF" }, std::process::id());
    let listener = TcpListener::bind(&args.listen)?;
    let opts = ServeOptions {
        socket: args.socket,
        verbose: args.verbose,
        log_every: args.log_every,
        keep_warm_us: args.keep_warm_us,
        re_anchor_every: 512,
    };
    v4flash_kernels::het::remote_experts::serve(listener, &mut shard, &mut exec, &opts, tracer.as_ref())
}
