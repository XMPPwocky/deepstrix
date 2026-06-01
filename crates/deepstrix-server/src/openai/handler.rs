//! `/v1/chat/completions` handler — non-streaming Phase 1 path.

use axum::extract::State;
use axum::Json;

use crate::engine_worker::{EngineHandle, GenerateReq};
use crate::openai::error::ApiError;
use crate::openai::types::{
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Role, Usage,
};
use crate::prompt::render_prompt;

/// Default sampler parameters when the request omits them. Matches the
/// DeepSeek V4-Flash recommended recipe (`temperature=1.0`, `top_p=1.0`).
const DEFAULT_TEMPERATURE: f32 = 1.0;
const DEFAULT_MIN_P_REL: f32 = 0.0;
/// Default max tokens per request. Letta typically overrides this. Cap
/// chosen to avoid runaway generation if the client forgets to set it.
const DEFAULT_MAX_NEW: usize = 2048;

pub async fn chat_completions(
    State(engine): State<EngineHandle>,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<ChatCompletionResponse>, ApiError> {
    if req.stream.unwrap_or(false) {
        return Err(ApiError::BadRequest(
            "streaming responses are not supported in Phase 1; omit stream:true".into(),
        ));
    }
    if req.tools.as_ref().map(|t| !t.is_empty()).unwrap_or(false) {
        return Err(ApiError::BadRequest(
            "tool definitions are not supported in Phase 1; omit tools".into(),
        ));
    }

    // Extract a merged system prompt from any leading role:"system"
    // messages, and the remaining (user/assistant) tail.
    let (system_prompt, body) = split_system(req.messages);

    let tokens = render_prompt(&engine.vocab, &body, system_prompt.as_deref())
        .map_err(|e| ApiError::BadRequest(format!("{e:#}")))?;
    let prompt_tokens_count = tokens.len() as u32;

    let temperature = req.temperature.unwrap_or(DEFAULT_TEMPERATURE);
    let max_new = req.max_tokens.map(|m| m as usize).unwrap_or(DEFAULT_MAX_NEW);
    let seed = req.seed.unwrap_or_else(default_seed);

    let gen_req = GenerateReq {
        tokens,
        max_new,
        temperature,
        min_p_rel: DEFAULT_MIN_P_REL,
        seed,
    };

    let result = engine.generate(gen_req).await.map_err(ApiError::from)?;

    let id = format!("chatcmpl-{}", uuid_like(seed));
    let resp = ChatCompletionResponse {
        id,
        object: "chat.completion",
        created: unix_now(),
        model: engine.model_name.as_str().to_string(),
        choices: vec![Choice {
            index: 0,
            message: ChatMessage {
                role: Role::Assistant,
                content: Some(result.text),
                tool_calls: Vec::new(),
                tool_call_id: None,
                name: None,
            },
            finish_reason: result.finish_reason.as_openai(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens_count,
            completion_tokens: result.completion_tokens,
            total_tokens: prompt_tokens_count + result.completion_tokens,
        },
    };
    Ok(Json(resp))
}

fn split_system(messages: Vec<ChatMessage>) -> (Option<String>, Vec<ChatMessage>) {
    let mut system_chunks: Vec<String> = Vec::new();
    let mut body: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut seen_non_system = false;
    for m in messages {
        if matches!(m.role, Role::System) && !seen_non_system {
            if let Some(c) = m.content {
                if !c.is_empty() {
                    system_chunks.push(c);
                }
            }
        } else {
            seen_non_system = true;
            body.push(m);
        }
    }
    let merged = if system_chunks.is_empty() {
        None
    } else {
        Some(system_chunks.join("\n\n"))
    };
    (merged, body)
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn default_seed() -> u64 {
    // Per-request derived from wall clock — deterministic per request,
    // not deterministic across requests. Clients that want repro pass
    // the `seed` field explicitly.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0xD5C0DE);
    nanos ^ 0xD5C0DE
}

fn uuid_like(seed: u64) -> String {
    // Cheap pseudo-uuid from the seed; Phase 2 swaps to uuid::Uuid::now_v7.
    format!("{:016x}", seed)
}
