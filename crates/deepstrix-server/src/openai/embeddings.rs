//! `POST /v1/embeddings` (OpenAI): Qwen3-Embedding through the hub's embed
//! phase (docs/v41/EMBED_PHASE_DESIGN.md §7).
//!
//! The handler tokenizes with the EMBEDDING model's vocab, appends
//! `<|endoftext|>`, checks every limit (so the worker never sees an input it
//! cannot run), queues the job and awaits the worker's reply. A client that
//! goes away drops the reply channel; the worker then skips the job.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::{Deserialize, Serialize};
use tokio::sync::oneshot;

use crate::embed_phase::{EmbedInfo, EmbedRequest};
use crate::engine_worker::{EngineHandle, SubmitError};
use crate::openai::error::ApiError;

/// OpenAI's own cap on inputs per request.
pub const MAX_INPUTS: usize = 2048;

/// Request body cap for `/v1/embeddings` (axum's default is 2 MiB).
pub const EMBEDDINGS_BODY_LIMIT: usize = 64 << 20;

#[derive(Debug, Deserialize)]
pub struct EmbeddingsRequest {
    pub input: EmbeddingInput,
    #[serde(default)]
    pub model: Option<String>,
    /// `float` (default) or `base64` (little-endian f32).
    #[serde(default)]
    pub encoding_format: Option<String>,
    /// Matryoshka truncation: the first `dimensions` components, L2-normalized.
    #[serde(default)]
    pub dimensions: Option<u32>,
    #[serde(default)]
    pub user: Option<String>,
}

/// The four shapes OpenAI accepts. Untagged: tried in order, and an empty
/// array parses as `Texts(vec![])` (refused as empty below).
#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum EmbeddingInput {
    Text(String),
    Texts(Vec<String>),
    Tokens(Vec<u32>),
    TokenLists(Vec<Vec<u32>>),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingsResponse {
    pub object: &'static str,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbeddingUsage,
}

#[derive(Debug, Serialize)]
pub struct EmbeddingData {
    pub object: &'static str,
    pub index: usize,
    pub embedding: EmbeddingValue,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
pub enum EmbeddingValue {
    Float(Vec<f32>),
    Base64(String),
}

#[derive(Debug, Serialize)]
pub struct EmbeddingUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}

fn bad(msg: impl Into<String>) -> ApiError {
    ApiError::BadRequest(msg.into())
}

/// Standard base64 (RFC 4648, padded) of `bytes`.
pub fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(A[(n >> 18) as usize & 63] as char);
        out.push(A[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
    }
    out
}

/// Every input as token ids with EOS appended, every limit checked.
pub fn tokenize_inputs(info: &EmbedInfo, input: EmbeddingInput) -> Result<Vec<Vec<u32>>, ApiError> {
    let mut out: Vec<Vec<u32>> = match input {
        EmbeddingInput::Text(s) => vec![text_ids(info, &s)?],
        EmbeddingInput::Texts(v) => {
            check_count(v.len())?;
            v.iter().map(|s| text_ids(info, s)).collect::<Result<_, _>>()?
        }
        EmbeddingInput::Tokens(t) => vec![token_ids(info, t)?],
        EmbeddingInput::TokenLists(v) => {
            check_count(v.len())?;
            v.into_iter().map(|t| token_ids(info, t)).collect::<Result<_, _>>()?
        }
    };
    let mut total = 0usize;
    for (i, ids) in out.iter_mut().enumerate() {
        ids.push(info.eos_id);
        if ids.len() > info.max_input_tokens {
            return Err(bad(format!(
                "input {i} is {} tokens (with the appended <|endoftext|>); the limit is {} (V41_EMBED_MAX_INPUT_TOKENS)",
                ids.len(),
                info.max_input_tokens
            )));
        }
        total += ids.len();
    }
    if total > info.max_request_tokens {
        return Err(bad(format!(
            "the request is {total} tokens; the limit is {} (V41_EMBED_MAX_REQUEST_TOKENS)",
            info.max_request_tokens
        )));
    }
    Ok(out)
}

fn check_count(n: usize) -> Result<(), ApiError> {
    if n == 0 {
        return Err(bad("`input` must not be empty"));
    }
    if n > MAX_INPUTS {
        return Err(bad(format!("`input` has {n} items; at most {MAX_INPUTS}")));
    }
    Ok(())
}

fn text_ids(info: &EmbedInfo, s: &str) -> Result<Vec<u32>, ApiError> {
    if s.is_empty() {
        return Err(bad("`input` must not contain empty strings"));
    }
    let ids: Vec<u32> = info.vocab.encode_qwen2(s).into_iter().map(|t| t as u32).collect();
    // Startup checks the vocab fits the embedding table; this keeps one bad
    // id from failing a whole phase if that ever changes.
    if let Some(&id) = ids.iter().find(|&&id| id as usize >= info.n_vocab) {
        return Err(ApiError::EngineFailed(color_eyre::eyre::eyre!("tokenizer produced id {id} >= vocab {}", info.n_vocab)));
    }
    Ok(ids)
}

fn token_ids(info: &EmbedInfo, t: Vec<u32>) -> Result<Vec<u32>, ApiError> {
    if t.is_empty() {
        return Err(bad("`input` must not contain empty token arrays"));
    }
    if let Some(&bad_id) = t.iter().find(|&&id| id as usize >= info.n_vocab) {
        return Err(bad(format!("token id {bad_id} is out of range (vocab {})", info.n_vocab)));
    }
    Ok(t)
}

/// The response body for finished embeddings.
pub fn response(info: &EmbedInfo, embeddings: Vec<Vec<f32>>, prompt_tokens: u32, base64_out: bool) -> EmbeddingsResponse {
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, v)| EmbeddingData {
            object: "embedding",
            index,
            embedding: if base64_out {
                EmbeddingValue::Base64(base64(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>()))
            } else {
                EmbeddingValue::Float(v)
            },
        })
        .collect();
    EmbeddingsResponse {
        object: "list",
        data,
        model: info.model_name.clone(),
        usage: EmbeddingUsage { prompt_tokens, total_tokens: prompt_tokens },
    }
}

pub async fn embeddings(
    State(engine): State<EngineHandle>,
    payload: Result<Json<EmbeddingsRequest>, axum::extract::rejection::JsonRejection>,
) -> Result<Response, ApiError> {
    let Some(info) = engine.embed.clone() else {
        return Err(ApiError::Rejection {
            status: StatusCode::NOT_FOUND,
            code: "embeddings_disabled",
            message: "embeddings are not enabled on this server (start it with --embed-gguf)".into(),
        });
    };
    let Json(req) = payload.map_err(ApiError::from_json_rejection)?;
    let base64_out = match req.encoding_format.as_deref() {
        None | Some("float") => false,
        Some("base64") => true,
        Some(other) => return Err(bad(format!("encoding_format {other:?} is not supported (float | base64)"))),
    };
    let dims = match req.dimensions {
        None => None,
        Some(d) if (32..=info.n_embd as u32).contains(&d) => Some(d as usize),
        Some(d) => return Err(bad(format!("dimensions {d} is out of range (32..={})", info.n_embd))),
    };
    // Tokenizing a few hundred thousand tokens of text is milliseconds to tens
    // of milliseconds: off the async workers.
    let info2: Arc<EmbedInfo> = info.clone();
    let inputs = tokio::task::spawn_blocking(move || tokenize_inputs(&info2, req.input))
        .await
        .map_err(|e| ApiError::EngineFailed(color_eyre::eyre::eyre!("tokenizer task: {e}")))??;
    let n_inputs = inputs.len();
    let prompt_tokens: u32 = inputs.iter().map(|v| v.len() as u32).sum();
    let (tx, rx) = oneshot::channel();
    let t0 = Instant::now();
    engine
        .submit_embed(EmbedRequest::new(inputs, dims, tx))
        .map_err(|e| match e {
            SubmitError::Busy => ApiError::Busy("embedding queue is full; retry shortly".into()),
            SubmitError::WorkerDead => ApiError::EngineFailed(color_eyre::eyre::eyre!("engine worker is gone")),
        })?;
    // Wait for the reply; a request queued before the engine thread died
    // keeps its sender alive in the queue, so watch the worker too.
    let mut rx = rx;
    let reply = loop {
        tokio::select! {
            r = &mut rx => break r,
            _ = tokio::time::sleep(std::time::Duration::from_secs(5)) => {
                if engine.worker_gone() {
                    return Err(ApiError::EngineFailed(color_eyre::eyre::eyre!("the engine worker is gone")));
                }
            }
        }
    };
    let out = reply
        .map_err(|_| ApiError::EngineFailed(color_eyre::eyre::eyre!("the engine dropped the embedding request")))?
        .map_err(|e| ApiError::EngineFailed(color_eyre::eyre::eyre!("{e}")))?;
    tracing::info!(
        model = req.model.as_deref().unwrap_or(""),
        inputs = n_inputs,
        prompt_tokens,
        dims = ?dims,
        ms = t0.elapsed().as_millis() as u64,
        "embeddings served"
    );
    Ok(Json(response(&info, out, prompt_tokens, base64_out)).into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
        assert_eq!(base64(&1.0f32.to_le_bytes()), "AACAPw==");
    }

    #[test]
    fn input_shapes_parse() {
        let p = |s: &str| serde_json::from_str::<EmbeddingsRequest>(s).unwrap().input;
        assert!(matches!(p(r#"{"input":"hi"}"#), EmbeddingInput::Text(_)));
        assert!(matches!(p(r#"{"input":["a","b"]}"#), EmbeddingInput::Texts(v) if v.len() == 2));
        assert!(matches!(p(r#"{"input":[1,2,3]}"#), EmbeddingInput::Tokens(v) if v == [1, 2, 3]));
        assert!(matches!(p(r#"{"input":[[1],[2,3]]}"#), EmbeddingInput::TokenLists(v) if v.len() == 2));
        assert!(matches!(p(r#"{"input":[]}"#), EmbeddingInput::Texts(v) if v.is_empty()));
        assert!(serde_json::from_str::<EmbeddingsRequest>(r#"{"input":[-1]}"#).is_err());
    }
}
