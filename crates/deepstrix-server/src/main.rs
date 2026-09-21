//! deepstrix-server entry point.
//!
//! Usage:
//!   deepstrix-server --gguf <path> [--addr 127.0.0.1:8080] [--ctx 8192]
//!                    [--snapshot-dir ~/.cache/deepstrix/snapshots]
//!                    [--disk-cap-gb 100] [--default-top-p 0.95]
//!                    [--default-reasoning-effort low]
//!
//! Loads the V4-Flash model into a dedicated engine worker thread,
//! then serves an OpenAI-compatible `/v1/chat/completions` endpoint
//! over HTTP. On-disk snapshot cache for cross-restart KV reuse.

use std::net::SocketAddr;
use std::path::PathBuf;

use axum::routing::{get, post};
use axum::Router;
use clap::Parser;
use color_eyre::eyre::{self, eyre};
use v4flash_hip::install_panic_handler;

use deepstrix_server::engine_worker::{run_watchdog, spawn, WorkerConfig};
use deepstrix_server::openai::error::log_error_responses;
use deepstrix_server::openai::handler::{chat_completions, healthz, list_models, lmstudio_models, readyz};

#[derive(Parser, Debug)]
#[command(version, about = "OpenAI-compatible HTTP server for deepstrix V4-Flash")]
struct Args {
    /// Path to the V4-Flash GGUF file.
    #[arg(long)]
    gguf: String,
    /// HTTP bind address (host:port).
    /// Repeatable: pass `--addr` once per address to listen on, e.g.
    /// `--addr 127.0.0.1:18080 --addr 100.79.4.101:18080` to serve loopback
    /// and the tailnet. An address that cannot be bound is warned about and
    /// skipped — only failing to bind ALL of them is fatal.
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: Vec<SocketAddr>,
    /// KV cache capacity (tokens).
    #[arg(long, default_value_t = 8192)]
    ctx: u32,
    /// Model name reported back in OpenAI responses.
    #[arg(long, default_value = "deepseek-v4-flash")]
    model_name: String,
    /// Root directory for on-disk KV snapshots. Defaults to
    /// `$XDG_CACHE_HOME/deepstrix/snapshots`, falling back to
    /// `~/.cache/deepstrix/snapshots`.
    #[arg(long)]
    snapshot_dir: Option<PathBuf>,
    /// Soft cap for the on-disk snapshot cache, in GB. LRU evict kicks
    /// in above this.
    #[arg(long, default_value_t = 100)]
    disk_cap_gb: u64,
    /// Forward-progress deadline (ms). If a request is in-flight and
    /// no token sample / prefill chunk completes within this window,
    /// the watchdog aborts the process for supervisor restart. Default
    /// 60s — comfortably above the worst-case chunk wall-clock (~20s
    /// at depth 64K) but short enough to detect a wedged GPU quickly.
    /// Override with env `DEEPSTRIX_HANG_DEADLINE_MS` (env wins).
    #[arg(long, default_value_t = 60_000)]
    hang_deadline_ms: i64,
    /// Vision tower (ViT + aligner). Enables image parts in
    /// `/v1/chat/completions`; the tower is loaded onto the iGPU at
    /// startup and the text vocab must carry the `<｜deepseek_image｜>`
    /// token. V4-Flash: the Vision-Exp `mmproj-F16.gguf`. V4.1 build
    /// (`--features v41`): the HF snapshot directory (normally the same as
    /// `--gguf`) — the `vision.*` / `aligner.*` safetensors are read from it
    /// directly, and the `bias_vl` routing sidecar is derived from it on
    /// first use. Env `DEEPSTRIX_MMPROJ` is used when the flag is absent.
    /// Without either, requests with images get HTTP 400.
    #[arg(long)]
    mmproj: Option<PathBuf>,
    /// Allow image parts to reference absolute local file paths under
    /// this directory (repeatable). OFF by default: without it the
    /// server only accepts `data:<type>;base64,...` image URLs.
    ///
    /// SECURITY: this endpoint has no authentication, so every client
    /// that can reach `--addr` can read any image file under the
    /// directories listed here. Only point it at a directory you are
    /// happy to serve. Env `DEEPSTRIX_ALLOW_IMAGE_DIRS` (`:`-separated)
    /// is used when the flag is absent.
    #[arg(long = "allow-image-dir")]
    allow_image_dir: Vec<PathBuf>,
    /// Nucleus (top-p) cutoff applied when a request omits `top_p`.
    /// Default 0.95 = DeepSeek's agentic recipe for this model. Pass 1.0
    /// to disable truncation and restore the pre-top_p sampler exactly.
    /// Sampling-only: this does NOT change the rendered prompt, so on-disk
    /// KV snapshots stay valid when you change it.
    ///
    /// Must be in (0, 1]; an out-of-range value is a startup error, NOT a
    /// clamp. (A per-request `top_p` is what gets clamped instead — see
    /// `openai::handler::resolve_top_p`.)
    #[arg(long = "default-top-p", default_value_t = deepstrix_server::openai::handler::DEFAULT_TOP_P)]
    default_top_p: f32,
    /// Reasoning effort applied when a request sends neither `reasoning`
    /// nor `reasoning_effort`. One of none/off/disabled/false, minimal/low,
    /// medium/high, xhigh/max/ultra. Default "low" — the server's
    /// historical behaviour.
    ///
    /// WARNING: raising this changes the RENDERED PROMPT (high/max prepend
    /// a preamble to the system block), so every KV prefix cached under the
    /// old default stops matching and the whole snapshot cache has to be
    /// re-prefilled. Opt in deliberately.
    #[arg(long = "default-reasoning-effort", default_value = "low")]
    default_reasoning_effort: String,
}

fn default_snapshot_dir() -> eyre::Result<PathBuf> {
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .ok_or_else(|| eyre!("cannot determine cache dir (set HOME or XDG_CACHE_HOME)"))?;
    Ok(base.join("deepstrix").join("snapshots"))
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> eyre::Result<()> {
    install_panic_handler()?;
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "deepstrix_server=info,info".into()),
        )
        .init();

    let args = Args::parse();
    let snapshot_root = match args.snapshot_dir {
        Some(p) => p,
        None => default_snapshot_dir()?,
    };
    let disk_cap_bytes = args.disk_cap_gb.saturating_mul(1024 * 1024 * 1024);
    let default_reasoning_effort =
        deepstrix_server::prompt::ReasoningEffort::parse_str(&args.default_reasoning_effort)
            .map_err(|e| eyre!("--default-reasoning-effort: {e}"))?;
    if default_reasoning_effort != deepstrix_server::prompt::DEFAULT_EFFORT {
        tracing::warn!(
            effort = ?default_reasoning_effort,
            "--default-reasoning-effort is not the compiled-in default: requests that omit \
             reasoning_effort now render a DIFFERENT prompt, so previously cached KV prefixes \
             will not match and must be re-prefilled"
        );
    }
    if !(args.default_top_p > 0.0 && args.default_top_p <= 1.0) {
        return Err(eyre!(
            "--default-top-p must be in (0, 1] (got {})",
            args.default_top_p
        ));
    }
    let mmproj_path = args.mmproj.or_else(|| {
        std::env::var_os("DEEPSTRIX_MMPROJ")
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    });
    let allow_image_dirs = if args.allow_image_dir.is_empty() {
        std::env::var("DEEPSTRIX_ALLOW_IMAGE_DIRS")
            .ok()
            .filter(|v| !v.is_empty())
            .map(|v| v.split(':').filter(|p| !p.is_empty()).map(PathBuf::from).collect())
            .unwrap_or_default()
    } else {
        args.allow_image_dir
    };
    tracing::info!(
        addrs = ?args.addr,
        ctx = args.ctx,
        gguf = %args.gguf,
        mmproj = ?mmproj_path,
        allow_image_dirs = ?allow_image_dirs,
        snapshot_dir = %snapshot_root.display(),
        disk_cap_gb = args.disk_cap_gb,
        default_top_p = args.default_top_p,
        default_reasoning_effort = ?default_reasoning_effort,
        "starting deepstrix-server"
    );

    let engine = spawn(WorkerConfig {
        gguf_path: args.gguf,
        n_kv_max: args.ctx,
        model_name: args.model_name,
        snapshot_root,
        snapshot_cap_bytes: disk_cap_bytes,
        mmproj_path,
        allow_image_dirs,
        default_top_p: args.default_top_p,
        default_reasoning_effort,
    })?;

    // Forward-progress watchdog. Env override > CLI flag > default.
    let hang_deadline_ms = std::env::var("DEEPSTRIX_HANG_DEADLINE_MS")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(args.hang_deadline_ms);
    let watchdog_progress = engine.progress.clone();
    std::thread::Builder::new()
        .name("deepstrix-watchdog".into())
        .spawn(move || run_watchdog(watchdog_progress, hang_deadline_ms, 2000))
        .map_err(|e| eyre!("failed to spawn watchdog thread: {e}"))?;
    tracing::info!(
        hang_deadline_ms,
        "watchdog armed; on stall the process will abort() for supervisor restart"
    );

    // axum's DefaultBodyLimit is 2 MiB, which caps a base64 `data:` image
    // at ~1.4 MiB and rejects anything larger with an opaque 413 that
    // never mentions images. The limit is derived from MAX_IMAGE_BYTES so
    // the two cannot drift; only the chat route is raised.
    let app = Router::new()
        .route(
            "/v1/chat/completions",
            post(chat_completions).layer(axum::extract::DefaultBodyLimit::max(
                deepstrix_server::vision_prompt::MAX_REQUEST_BODY_BYTES,
            )),
        )
        .route("/v1/models", get(list_models))
        .route("/api/v1/models", get(lmstudio_models))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        // Every 4xx/5xx we hand back is otherwise silent server-side.
        .layer(axum::middleware::from_fn(log_error_responses))
        .with_state(engine.clone());

    // A listener that will not bind is a warning, not a fatal error: the
    // usual cause is an interface that is not up yet (tailscale0 after a
    // cold boot), and refusing to start would take the loopback endpoint
    // down with it. Binding NONE of them is still fatal.
    let mut listeners = Vec::with_capacity(args.addr.len());
    for addr in &args.addr {
        match tokio::net::TcpListener::bind(addr).await {
            Ok(l) => {
                tracing::info!("listening on http://{addr}");
                listeners.push(l);
            }
            Err(e) => tracing::warn!(%addr, error = %e, "could not bind this address; skipping"),
        }
    }
    if listeners.is_empty() {
        return Err(eyre!("no listen address could be bound (tried {:?})", args.addr));
    }

    // Serve the same router on every listener. `Router` is cheap to clone
    // (an Arc inside), and they share one engine handle, so the single
    // worker thread still serializes generation across all of them.
    let mut serves = tokio::task::JoinSet::new();
    for listener in listeners {
        let app = app.clone();
        serves.spawn(async move { axum::serve(listener, app).await });
    }

    // Race the listeners against a shutdown signal — when SIGINT/SIGTERM
    // arrives, ask the worker to save its dirty live state before we exit.
    // A serve error is stashed rather than returned, so the engine still
    // gets its chance to flush; dropping the JoinSet aborts the rest.
    let mut serve_err: Option<eyre::Report> = None;
    tokio::select! {
        Some(joined) = serves.join_next() => {
            match joined {
                Ok(Ok(())) => tracing::warn!("an HTTP listener stopped on its own"),
                Ok(Err(e)) => serve_err = Some(e.into()),
                Err(e) => serve_err = Some(eyre!("HTTP listener task failed: {e}")),
            }
        }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received");
            // The engine worker's blocking loop does not observe signals, so an
            // idle server used to survive SIGTERM until SIGKILL (which skips the
            // atexit handlers an attached rocprofv3 needs to flush its trace).
            // Guarantee a real `exit()` after a grace period.
            let grace = std::env::var("V41_EXIT_GRACE_S").ok().and_then(|v| v.parse().ok()).unwrap_or(5u64);
            std::thread::spawn(move || {
                std::thread::sleep(std::time::Duration::from_secs(grace));
                tracing::warn!(grace_s = grace, "shutdown grace elapsed; exiting the process");
                std::process::exit(0);
            });
        }
    }
    if let Err(e) = engine.shutdown().await {
        tracing::warn!(error = %e, "engine shutdown returned error");
    }
    if let Some(e) = serve_err {
        return Err(e);
    }
    tracing::info!("deepstrix-server exited cleanly");
    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigint = signal(SignalKind::interrupt()).ok();
    let mut sigterm = signal(SignalKind::terminate()).ok();
    tokio::select! {
        _ = async { if let Some(s) = sigint.as_mut() { s.recv().await; } else { std::future::pending::<()>().await; } } => {}
        _ = async { if let Some(s) = sigterm.as_mut() { s.recv().await; } else { std::future::pending::<()>().await; } } => {}
    }
}
