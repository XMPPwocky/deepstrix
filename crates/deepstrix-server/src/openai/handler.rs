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
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::stream::poll_fn;

use crate::dsml::{DsmlEvent, DsmlScanner};
use crate::engine_worker::{accumulate, EngineHandle, FinishReason, GenerateReq, WorkerEvent};
use crate::openai::error::ApiError;
use crate::openai::sse::{
    encode_chunk, role_delta, text_delta, tool_call_args_delta, tool_call_start_delta,
    ChunkDelta,
};
use crate::openai::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Role, ToolCall,
    ToolCallFunction, Usage,
};
use crate::prompt::render_prompt;

const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_MIN_P_REL: f32 = 0.0;
const DEFAULT_MAX_NEW: usize = 2048;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Response, ApiError> {
    let stream = req.stream.unwrap_or(false);

    let tokens = render_prompt(
        &engine.vocab,
        &req.messages,
        req.tools.as_deref(),
    )
    .map_err(|e| ApiError::BadRequest(format!("{e:#}")))?;
    let prompt_tokens_count = tokens.len() as u32;

    let temperature = req.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    let max_new = req
        .max_tokens
        .map(|m| m as usize)
        .unwrap_or(DEFAULT_MAX_NEW);
    let seed = req.seed.unwrap_or_else(default_seed);

    let gen_req = GenerateReq {
        tokens,
        max_new,
        temperature,
        min_p_rel: DEFAULT_MIN_P_REL,
        seed,
    };

    let id = format!("chatcmpl-{}", uuid::Uuid::now_v7().simple());
    let model = engine.model_name.as_str().to_string();

    let rx = engine
        .submit(gen_req, req.session_id.clone())
        .map_err(ApiError::from)?;

    if stream {
        // Spawn a task that drives the worker stream and pushes SSE
        // events into a channel. The HTTP response wraps the channel
        // as a Stream via futures_util::poll_fn.
        let (tx, mut sse_rx) =
            tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
        tokio::spawn(drive_sse_stream(id.clone(), model.clone(), rx, tx));
        let stream = poll_fn::<Result<Event, Infallible>, _>(move |cx| sse_rx.poll_recv(cx));
        Ok(Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response())
    } else {
        let result = accumulate(rx).await.map_err(ApiError::from)?;
        let _ = prompt_tokens_count; // re-counted below from result
        let blocking = build_blocking_response(
            id,
            model,
            result.text,
            prompt_tokens_count,
            result.completion_tokens,
            result.finish_reason,
        );
        Ok(Json(blocking).into_response())
    }
}

fn build_blocking_response(
    id: String,
    model: String,
    raw_text: String,
    prompt_tokens: u32,
    completion_tokens: u32,
    finish: FinishReason,
) -> ChatCompletionResponse {
    // Pass the full text through the scanner to separate tool calls
    // from plain text.
    let mut scanner = DsmlScanner::new();
    let mut events = scanner.push_text(&raw_text);
    events.extend(scanner.finish());

    let mut content_text = String::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();
    let mut saw_tool = false;
    for ev in events {
        match ev {
            DsmlEvent::Text(s) => content_text.push_str(&s),
            DsmlEvent::ToolCall {
                id, name, arguments, ..
            } => {
                tool_calls.push(ToolCall {
                    id,
                    kind: "function".into(),
                    function: ToolCallFunction { name, arguments },
                });
                saw_tool = true;
            }
            DsmlEvent::ToolCallsEnd => saw_tool = true,
        }
    }

    let finish_reason = if saw_tool {
        "tool_calls"
    } else {
        finish.as_openai()
    };

    ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_now(),
        model,
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: Role::Assistant,
                content: if content_text.is_empty() {
                    None
                } else {
                    Some(content_text)
                },
                tool_calls,
                tool_call_id: None,
                name: None,
            },
            finish_reason,
        }],
        usage: Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
        },
    }
}

/// Drives the SSE stream forward: pulls `WorkerEvent`s, runs them
/// through the DSML scanner, and pushes encoded `Event`s into `out`.
async fn drive_sse_stream(
    id: String,
    model: String,
    mut rx: tokio::sync::mpsc::Receiver<WorkerEvent>,
    out: tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
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

    let mut scanner = DsmlScanner::new();
    let mut sent_tool_index: u32 = 0;
    let mut saw_tool = false;
    let mut finish_reason: &'static str = "stop";

    while let Some(ev) = rx.recv().await {
        match ev {
            WorkerEvent::Chunk(s) => {
                let events = scanner.push_text(&s);
                for de in events {
                    match de {
                        DsmlEvent::Text(t) => {
                            if !t.is_empty()
                                && out.send(send(text_delta(t), None)).await.is_err()
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
            WorkerEvent::Done { finish, .. } => {
                let tail = scanner.finish();
                for de in tail {
                    if let DsmlEvent::Text(t) = de {
                        if !t.is_empty()
                            && out.send(send(text_delta(t), None)).await.is_err()
                        {
                            return;
                        }
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
                finish_reason = "error";
            }
        }
    }

    // Final terminating chunk with finish_reason.
    let _ = out
        .send(send(ChunkDelta::default(), Some(finish_reason)))
        .await;
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
