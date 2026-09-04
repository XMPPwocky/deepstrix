//! `/v1/chat/completions` handler.
//!
//! Two response paths:
//!   * `stream:false` (default) → JSON `chat.completion` after the model
//!     finishes. Accumulates worker events; runs them through the DSML
//!     scanner; assembles tool_calls + text into the `Choice::message`.
//!   * `stream:true` → SSE stream of `chat.completion.chunk` events,
//!     emitting text deltas as they arrive and synthesizing
//!     `delta.tool_calls` events for each DSML invoke block.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::poll_fn;

/// Drop guard: flips the engine's cancel bool when the HTTP response
/// future is dropped (i.e. when the client disconnects mid-decode).
/// The worker's decode loop polls the bool and breaks out of the
/// generation as soon as it sees true.
struct CancelOnDrop(Arc<AtomicBool>);
impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

use crate::dsml::{DsmlEvent, DsmlScanner};
use crate::engine_worker::{accumulate, EngineHandle, GenerateReq, SubmitError, WorkerEvent};
use crate::openai::error::ApiError;
use crate::openai::sse::{
    encode_chunk, reasoning_delta, role_delta, text_delta, tool_call_args_delta,
    tool_call_start_delta, ChunkDelta,
};
use crate::openai::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Role, ToolCall,
    ToolCallFunction, Usage,
};
use crate::prompt::{render_prompt, ReasoningEffort};
use crate::vision_prompt::{expand_images, VlPrompt};

const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_MIN_P_REL: f32 = 0.0;
// Per-turn completion cap applied when the request omits `max_tokens`.
// Letta-code's pi-ai only sends max_tokens when the registered model
// has it set explicitly (pi-stream-adapter.ts:486-487); when it
// doesn't, we apply this default. 2048 (the OpenAI-historical default)
// is far too low for think-mode + a non-trivial Write/Edit tool call:
// `<think>` typically eats 500–2000 tokens at long ctx, leaving
// almost nothing for the actual response, and the model truncates
// mid-DSML — surfaces to letta as "Empty LLM response, retrying"
// (now caught by `DsmlScanner::finish()`'s mid-markup flush) or as
// malformed tool args even when caught. 16K accommodates think + a
// multi-kB file Write comfortably while still bounding runaway
// generation. Clients can override either direction by sending
// max_tokens explicitly in the request.
const DEFAULT_MAX_NEW: usize = 16384;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let stream = req.stream.unwrap_or(false);

    // V4-Flash 0731 reasoning effort. Both request fields map to one
    // 4-state level (Off/Low/High/Max); `reasoning_effort` wins when
    // both are set; absent → Low (thinking on, no preamble — same as
    // the historical think-mode default). Unknown strings are an
    // invalid parameter → HTTP 400, matching the render_prompt
    // BadRequest convention below.
    let effort = ReasoningEffort::from_request_fields(
        req.reasoning.as_deref(),
        req.reasoning_effort.as_deref(),
    )
    .map_err(ApiError::BadRequest)?;
    // Vision-Exp: image parts are allowed only when the worker loaded a
    // tower (`--mmproj`) AND the vocab has the placeholder token. Reject
    // up front with a clear 400 rather than failing inside the worker.
    let has_images = req.messages.iter().any(|m| m.has_images());
    if has_images && !engine.vision_enabled {
        return Err(ApiError::BadRequest(
            "request contains image parts but this server has no vision tower loaded \
             (start with --mmproj <mmproj-F16.gguf> on a Vision-Exp GGUF)"
                .to_string(),
        ));
    }
    let tokens = render_prompt(
        &engine.vocab,
        &req.messages,
        req.tools.as_deref(),
        effort,
        engine.image_placeholder_id,
    )
    .map_err(|e| ApiError::BadRequest(format!("{e:#}")))?;
    // Expand each placeholder into its synthetic image block (decode +
    // resize + layout; CPU-only, ~10-100 ms per image). Also validates the
    // image sources: http(s) URLs / bad base64 / denied paths → 400.
    //
    // PNG/JPEG decode plus a scalar bicubic resample is tens of ms of
    // pure CPU, so it must not run inline on the async task —
    // `block_in_place` hands this worker's other tasks to a sibling
    // thread for the duration (needs the multi-thread runtime, which
    // `main`'s `#[tokio::main]` gives us). Cheaper than `spawn_blocking`
    // here because it borrows `req` instead of cloning every payload.
    let vl = if has_images {
        let inputs: Vec<&crate::openai::types::ImageInput> =
            req.messages.iter().flat_map(|m| m.images()).collect();
        let placeholder = engine
            .image_placeholder_id
            .expect("vision_enabled implies placeholder id");
        let policy = engine.image_policy.clone();
        tokio::task::block_in_place(|| expand_images(tokens, &inputs, placeholder, &policy))
            .map_err(|e| ApiError::BadRequest(format!("{e:#}")))?
    } else {
        VlPrompt { tokens, images: Vec::new(), spans: Vec::new() }
    };
    let VlPrompt { tokens, images, spans: image_spans } = vl;
    let prompt_tokens_count = tokens.len() as u32;

    // Up-front ctx check. Without this, an over-long prompt
    // surfaces as `ERROR engine error during stream` mid-SSE —
    // letta treats that as an opaque model failure. A proper
    // HTTP 400 with a clear message lets letta drop/summarize
    // history and retry cleanly.
    if prompt_tokens_count >= engine.n_kv_max {
        // Include the literal marker `context_length_exceeded` so
        // letta's `isContextWindowOverflowError` matcher recognises
        // it and triggers conversation compaction instead of bailing
        // the agent loop. See letta-code
        // `src/backend/dev/context-window-overflow.ts:44`.
        return Err(ApiError::ContextExhausted(format!(
            "context_length_exceeded: prompt length {} >= maximum context length {}",
            prompt_tokens_count, engine.n_kv_max
        )));
    }

    let temperature = req.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    let max_new = req
        .max_tokens
        .map(|m| m as usize)
        .unwrap_or(DEFAULT_MAX_NEW);
    let seed = req.seed.unwrap_or_else(default_seed);

    let gen_req = GenerateReq {
        tokens,
        images,
        image_spans,
        max_new,
        temperature,
        min_p_rel: DEFAULT_MIN_P_REL,
        seed,
    };

    let id = format!("chatcmpl-{}", uuid::Uuid::now_v7().simple());
    let model = engine.model_name.as_str().to_string();

    let include_usage = req
        .stream_options
        .as_ref()
        .and_then(|s| s.include_usage)
        .unwrap_or(false);

    let (rx, cancel) = engine.submit(gen_req, req.session_id.clone()).map_err(|e| match e {
        SubmitError::Busy => ApiError::Busy(format!("{e}")),
        SubmitError::WorkerDead => {
            ApiError::EngineFailed(color_eyre::eyre::eyre!("{e}"))
        }
    })?;

    let tok_dsml = engine.vocab.dsml_id;
    if stream {
        // The cancel guard moves into the SSE driver. When the
        // SSE response (and its task) is dropped — e.g. client
        // disconnect — the guard fires and flips the cancel bool.
        let guard = CancelOnDrop(cancel);
        let (tx, mut sse_rx) =
            tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::spawn(drive_sse_stream(
            id.clone(),
            model.clone(),
            rx,
            tx,
            guard,
            include_usage,
            prompt_tokens_count,
            tok_dsml,
        ));
        let stream = poll_fn::<Result<Event, Infallible>, _>(move |cx| sse_rx.poll_recv(cx));
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let guard = CancelOnDrop(cancel);
        let result = accumulate(rx, tok_dsml).await.map_err(ApiError::from)?;
        drop(guard);
        // Malformed DSML wins over everything else — the turn is
        // corrupted, don't pretend it succeeded as text/tool_calls.
        // Drop both content and tool_calls so letta records an empty
        // assistant turn with finish_reason="internal_error" rather
        // than storing partial garbage that taints subsequent turns.
        //
        // Why "internal_error" and not "error": pi-ai forwards unknown
        // finish_reasons verbatim as "Provider finish_reason: <reason>"
        // and letta's retryable-error detector substring-matches that
        // string against a fixed pattern list. "internal_error" is in
        // the list (local-provider-errors.ts); bare "error" is not, so
        // letta would surface a non-retryable hard error.
        let (finish_reason, content, tool_calls) = if result.saw_malformed {
            tracing::warn!(
                "non-streaming: scanner saw malformed DSML; reporting finish_reason=internal_error"
            );
            ("internal_error", None, Vec::new())
        } else if result.saw_tool {
            ("tool_calls", if result.text.is_empty() {
                None
            } else {
                Some(result.text)
            }, result.tool_calls)
        } else {
            (result.finish_reason.as_openai(), if result.text.is_empty() {
                None
            } else {
                Some(result.text)
            }, result.tool_calls)
        };
        let resp = ChatCompletionResponse {
            id,
            object: "chat.completion",
            created: unix_now(),
            model,
            choices: vec![Choice {
                index: 0,
                message: ChatMessage {
                    role: Role::Assistant,
                    content,
                    parts: Vec::new(),
                    tool_calls,
                    tool_call_id: None,
                    name: None,
                },
                finish_reason,
            }],
            usage: Usage {
                prompt_tokens: prompt_tokens_count,
                completion_tokens: result.completion_tokens,
                total_tokens: prompt_tokens_count + result.completion_tokens,
            },
        };
        Ok(Json(resp).into_response())
    }
}

/// GET /v1/models — minimal listing for letta's provider auth flow.
/// We surface `max_context_length` and `loaded_context_length` to
/// match LM Studio's API shape — letta's pi-ai stream adapter reads
/// these to know how much prompt the model can take. Both values are
/// our `--ctx` (= `n_kv_max`) since deepstrix-server doesn't
/// distinguish "supported" from "currently-loaded" ctx.
pub async fn list_models(State(engine): State<EngineHandle>) -> Json<serde_json::Value> {
    let id = engine.model_name.as_str().to_string();
    let ctx = engine.n_kv_max as u64;
    Json(serde_json::json!({
        "object": "list",
        "data": [{
            "id": id,
            "object": "model",
            "type": "llm",
            "created": 0,
            "owned_by": "deepstrix",
            "max_context_length": ctx,
            "loaded_context_length": ctx,
            "context_window": ctx,
            "context_length": ctx,
            "max_completion_tokens": DEFAULT_MAX_NEW as u64,
            "max_output_tokens": DEFAULT_MAX_NEW as u64,
            "state": "loaded"
        }]
    }))
}

/// GET /api/v1/models — LM Studio-compatible model listing.
///
/// Exists so Hermes can discover that this server supports reasoning:
/// Hermes hard-codes `supports_reasoning = false` for custom OpenAI
/// endpoints, but when its provider is set to `lmstudio` it probes
/// `{server_root}/api/v1/models` for a `models` list (NOT OpenAI's
/// `data` envelope) and reads `capabilities.reasoning.allowed_options`
/// per model (matched on `key` or `id`). Any option besides "off"
/// enables reasoning, after which Hermes sends a flat top-level
/// `reasoning_effort` string — our native format.
///
/// The advertised options are LM Studio's vocabulary (its ladder tops
/// out at "xhigh"; Hermes statically clamps its max/ultra to xhigh
/// before the allowed-set check, which is why our parse_str maps
/// xhigh -> Max). Advertising exactly what we accept means Hermes
/// clamps rather than sending something we would 400 on.
pub async fn lmstudio_models(State(engine): State<EngineHandle>) -> Json<serde_json::Value> {
    let id = engine.model_name.as_str().to_string();
    let ctx = engine.n_kv_max as u64;
    Json(serde_json::json!({
        "models": [{
            "key": id,
            "id": id,
            "object": "model",
            "type": "llm",
            "state": "loaded",
            "max_context_length": ctx,
            "loaded_context_length": ctx,
            "capabilities": {
                "reasoning": {
                    "allowed_options": ["off", "minimal", "low", "medium", "high", "xhigh"]
                }
            }
        }]
    }))
}

/// GET /healthz — process-level liveness. Always 200 once the HTTP
/// stack is up. Does NOT detect a wedged engine; use /readyz for that.
pub async fn healthz() -> &'static str {
    "ok\n"
}

/// GET /readyz — engine-level readiness. 200 if the worker is either
/// idle or making forward progress; 503 if a request has been in-
/// flight for longer than the watchdog deadline (engine likely wedged,
/// abort imminent).
///
/// Body is a one-line summary so an operator curling the endpoint can
/// see why a 503 fired.
pub async fn readyz(
    State(engine): State<crate::engine_worker::EngineHandle>,
) -> Response {
    use axum::http::StatusCode;
    let p = &engine.progress;
    let inflight = p.inflight.load(std::sync::atomic::Ordering::Relaxed);
    let stale_ms = p.stale_ms();
    let deadline_ms = std::env::var("DEEPSTRIX_HANG_DEADLINE_MS")
        .ok()
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(60_000);
    if inflight && stale_ms > deadline_ms {
        (
            StatusCode::SERVICE_UNAVAILABLE,
            format!(
                "engine stalled: inflight=true stale_ms={stale_ms} deadline_ms={deadline_ms}\n"
            ),
        )
            .into_response()
    } else {
        (
            StatusCode::OK,
            format!("ok inflight={inflight} stale_ms={stale_ms}\n"),
        )
            .into_response()
    }
}


/// Drives the SSE stream forward: pulls `WorkerEvent`s, runs them
/// through the DSML scanner, and pushes encoded `Event`s into `out`.
async fn drive_sse_stream(
    id: String,
    model: String,
    mut rx: tokio::sync::mpsc::Receiver<WorkerEvent>,
    out: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    _cancel_guard: CancelOnDrop,
    include_usage: bool,
    prompt_tokens: u32,
    tok_dsml: Option<i32>,
) {
    let created = unix_now();
    let send = |delta: ChunkDelta,
                finish_reason: Option<&'static str>|
     -> Result<Event, Infallible> {
        let s = encode_chunk(&id, &model, created, delta, finish_reason);
        // axum's Event::data adds back the "data: " + "\n\n" framing.
        let payload = s
            .trim_start_matches("data: ")
            .trim_end_matches("\n\n")
            .to_string();
        Ok(Event::default().data(payload))
    };

    if out.send(send(role_delta(), None)).await.is_err() {
        return;
    }

    // The scanner is driven by TOK_DSML (vocab-dependent). If the
    // vocab didn't expose a dsml_id (non-V4-Flash model), fall back
    // to a sentinel that never matches — effectively disabling DSML
    // detection.
    let scanner_tok_dsml = tok_dsml.unwrap_or(-1);
    let mut scanner = DsmlScanner::new(scanner_tok_dsml);
    let mut sent_tool_index: u32 = 0;
    let mut saw_tool = false;
    let mut finish_reason: &'static str = "stop";
    let mut completion_tokens: u32 = 0;
    // UTF-8 buffers per channel. Applied to *bytes the scanner emits
    // as Text*, not to raw worker bytes — so the scanner sees the
    // raw byte stream including TOK_DSML's bytes, which it discards.
    let mut content_pending: Vec<u8> = Vec::new();
    let mut reasoning_pending: Vec<u8> = Vec::new();

    fn drain_valid_utf8(pending: &mut Vec<u8>, chunk: &[u8]) -> String {
        pending.extend_from_slice(chunk);
        let valid_to = match std::str::from_utf8(pending) {
            Ok(_) => pending.len(),
            Err(e) => e.valid_up_to(),
        };
        if valid_to == 0 {
            return String::new();
        }
        let drained: Vec<u8> = pending.drain(..valid_to).collect();
        String::from_utf8(drained).unwrap()
    }

    // Race rx.recv() against out.closed(). Without this we'd park
    // on rx.recv() during long prefills — the worker emits zero
    // events while batching — and never observe the client
    // disconnect. CancelOnDrop only fires when this task exits, so
    // a parked task means cancel stays false even though the SSE
    // consumer (sse_rx) was dropped seconds ago. Exiting on
    // out.closed() drops _cancel_guard, flips the bool, and the
    // worker's per-chunk cancel check picks it up at the next
    // prefill chunk boundary.
    loop {
        let ev = tokio::select! {
            biased;
            _ = out.closed() => {
                tracing::debug!("sse: downstream closed during generation; cancelling");
                return;
            }
            ev = rx.recv() => ev,
        };
        let Some(ev) = ev else { break };
        match ev {
            WorkerEvent::Chunk {
                token_id,
                bytes,
                reasoning,
            } => {
                if reasoning {
                    let s = drain_valid_utf8(&mut reasoning_pending, &bytes);
                    if !s.is_empty()
                        && out.send(send(reasoning_delta(s), None)).await.is_err()
                    {
                        return;
                    }
                    continue;
                }
                // Drive the scanner with the actual token-id + bytes;
                // it suppresses TOK_DSML bytes regardless and emits
                // clean Text/ToolCall events.
                for de in scanner.push_token(token_id, &bytes) {
                    match de {
                        DsmlEvent::Text(t) => {
                            let s = drain_valid_utf8(&mut content_pending, &t);
                            if !s.is_empty()
                                && out.send(send(text_delta(s), None)).await.is_err()
                            {
                                return;
                            }
                        }
                        DsmlEvent::ToolCall {
                            id: tid,
                            name,
                            arguments,
                            ..
                        } => {
                            saw_tool = true;
                            let tc = ToolCall {
                                id: tid,
                                kind: "function".into(),
                                function: ToolCallFunction { name, arguments },
                            };
                            if out
                                .send(send(
                                    tool_call_start_delta(&tc, sent_tool_index),
                                    None,
                                ))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            if out
                                .send(send(
                                    tool_call_args_delta(&tc, sent_tool_index),
                                    None,
                                ))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            sent_tool_index += 1;
                        }
                        DsmlEvent::ToolCallsEnd => {
                            saw_tool = true;
                        }
                    }
                }
            }
            WorkerEvent::Done {
                finish,
                completion_tokens: ct,
                ..
            } => {
                completion_tokens = ct;
                // Drain anything still buffered in the scanner. Since
                // the DSML-repair port, finish() can also emit
                // ToolCall / ToolCallsEnd events (truncated tool-call
                // blocks recovered by appending the missing closing
                // tags) — stream them exactly like live ones.
                for de in scanner.finish() {
                    match de {
                        DsmlEvent::Text(t) => {
                            let s = drain_valid_utf8(&mut content_pending, &t);
                            if !s.is_empty()
                                && out.send(send(text_delta(s), None)).await.is_err()
                            {
                                return;
                            }
                        }
                        DsmlEvent::ToolCall {
                            id: tid,
                            name,
                            arguments,
                            ..
                        } => {
                            saw_tool = true;
                            let tc = ToolCall {
                                id: tid,
                                kind: "function".into(),
                                function: ToolCallFunction { name, arguments },
                            };
                            if out
                                .send(send(
                                    tool_call_start_delta(&tc, sent_tool_index),
                                    None,
                                ))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            if out
                                .send(send(
                                    tool_call_args_delta(&tc, sent_tool_index),
                                    None,
                                ))
                                .await
                                .is_err()
                            {
                                return;
                            }
                            sent_tool_index += 1;
                        }
                        DsmlEvent::ToolCallsEnd => {
                            saw_tool = true;
                        }
                    }
                }
                // Lossy-flush any trailing partial bytes.
                if !content_pending.is_empty() {
                    let s = String::from_utf8_lossy(&content_pending).into_owned();
                    content_pending.clear();
                    if !s.is_empty()
                        && out.send(send(text_delta(s), None)).await.is_err()
                    {
                        return;
                    }
                }
                if !reasoning_pending.is_empty() {
                    let s = String::from_utf8_lossy(&reasoning_pending).into_owned();
                    reasoning_pending.clear();
                    if !s.is_empty()
                        && out.send(send(reasoning_delta(s), None)).await.is_err()
                    {
                        return;
                    }
                }
                finish_reason = if saw_tool {
                    "tool_calls"
                } else {
                    finish.as_openai()
                };
            }
            WorkerEvent::Error(e) => {
                tracing::error!(error=%e, "engine error during stream");
                // "internal_error" — see retryable-naming note on the
                // malformed-DSML branch below.
                finish_reason = "internal_error";
            }
        }
    }

    // Malformed DSML overrides whatever finish we picked above. We
    // can't unsend already-emitted text chunks (streaming), but at
    // least the terminating finish_reason tells letta the turn is
    // broken. "internal_error" rather than "error" so letta's
    // substring-based retryable-error classifier matches — see the
    // longer note on the non-streaming branch above.
    //
    // Caveat: letta also gates retry on whether ANY model output
    // (text_delta / thinking_delta / toolcall_end) was emitted before
    // the error (pi-stream-adapter.ts: `emittedModelOutput`). For
    // bad-DSML *before* any text/tool deltas the retry will fire; for
    // bad-DSML *after* we've already streamed text, letta will surface
    // the error without retrying even though we mark it retryable.
    if scanner.saw_malformed() {
        tracing::warn!(
            "stream: scanner saw malformed DSML; reporting finish_reason=internal_error"
        );
        finish_reason = "internal_error";
    }

    // Final terminating chunk with finish_reason.
    let _ = out
        .send(send(ChunkDelta::default(), Some(finish_reason)))
        .await;
    // Optional usage chunk (OpenAI sends this after the finish chunk
    // when stream_options.include_usage=true).
    if include_usage {
        let total = prompt_tokens + completion_tokens;
        let payload = serde_json::json!({
            "id": id,
            "object": "chat.completion.chunk",
            "created": unix_now(),
            "model": model,
            "choices": [],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total,
            }
        });
        let _ = out
            .send(Ok(Event::default().data(payload.to_string())))
            .await;
    }
    // [DONE] sentinel.
    let _ = out
        .send(Ok(Event::default().data("[DONE]".to_string())))
        .await;
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn default_seed() -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xD5C0DE);
    nanos ^ 0xD5C0DE
}
