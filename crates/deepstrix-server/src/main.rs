//! deepstrix-server entry point.
//!
//! Usage:
//!   deepstrix-server --gguf <path> [--addr 127.0.0.1:8080] [--ctx 8192]
//!                    [--snapshot-dir ~/.cache/deepstrix/snapshots]
//!                    [--disk-cap-gb 100]
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
use deepstrix_server::openai::handler::{chat_completions, healthz, list_models, lmstudio_models, readyz};

#[derive(Parser, Debug)]
#[command(version, about = "OpenAI-compatible HTTP server for deepstrix V4-Flash")]
struct Args {
    /// Path to the V4-Flash GGUF file.
    #[arg(long)]
    gguf: String,
    /// HTTP bind address (host:port).
    #[arg(long, default_value = "127.0.0.1:8080")]
    addr: SocketAddr,
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
    /// Vision-Exp mmproj GGUF (ViT + aligner). Enables image parts in
    /// `/v1/chat/completions`; the tower is loaded onto the iGPU at
    /// startup and the text GGUF must carry the `<｜deepseek_image｜>`
    /// token. Env `DEEPSTRIX_MMPROJ` is used when the flag is absent.
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
        addr = %args.addr,
        ctx = args.ctx,
        gguf = %args.gguf,
        mmproj = ?mmproj_path,
        allow_image_dirs = ?allow_image_dirs,
        snapshot_dir = %snapshot_root.display(),
        disk_cap_gb = args.disk_cap_gb,
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
        .with_state(engine.clone());

    let listener = tokio::net::TcpListener::bind(args.addr).await?;
    tracing::info!("listening on http://{}", args.addr);

    // Bind the serve task and a shutdown signal in parallel — when
    // SIGINT/SIGTERM arrives, ask the worker to save its dirty live
    // state to disk before we exit.
    let serve = axum::serve(listener, app);
    tokio::select! {
        r = serve => { r?; }
        _ = shutdown_signal() => {
            tracing::info!("shutdown signal received");
        }
    }
    if let Err(e) = engine.shutdown().await {
        tracing::warn!(error = %e, "engine shutdown returned error");
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
