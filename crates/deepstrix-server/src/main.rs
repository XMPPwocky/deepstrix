//! deepstrix-server entry point.
//!
//! Usage:
//!   deepstrix-server --gguf <path> [--addr 127.0.0.1:8080] [--ctx 8192]
//!
//! Loads the V4-Flash model into a dedicated engine worker thread,
//! then serves an OpenAI-compatible `/v1/chat/completions` endpoint
//! over HTTP. Phase 1: non-streaming chat only, no tools, no caching.

use std::net::SocketAddr;

use axum::routing::post;
use axum::Router;
use clap::Parser;
use color_eyre::eyre;
use v4flash_hip::install_panic_handler;

use deepstrix_server::engine_worker::{spawn, WorkerConfig};
use deepstrix_server::openai::handler::chat_completions;

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
    tracing::info!(addr = %args.addr, ctx = args.ctx, gguf = %args.gguf, "starting deepstrix-server");

    let engine = spawn(WorkerConfig {
        gguf_path: args.gguf,
        n_kv_max: args.ctx,
        model_name: args.model_name,
    })?;

    let app = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(engine);

    let listener = tokio::net::TcpListener::bind(args.addr).await?;
    tracing::info!("listening on http://{}", args.addr);
    axum::serve(listener, app).await?;
    Ok(())
}
