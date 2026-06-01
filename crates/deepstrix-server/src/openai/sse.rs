//! OpenAI-style SSE chunk encoding.
//!
//! Wire format (per OpenAI Chat Completions streaming spec):
//!   data: {"id":"...","object":"chat.completion.chunk",...}\n\n
//!   ...
//!   data: [DONE]\n\n

use serde::Serialize;

use crate::openai::types::ToolCall;

#[derive(Debug, Default, Serialize)]
pub struct ChunkDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCallDelta>,
}

#[derive(Debug, Serialize)]
pub struct ToolCallDelta {
    pub index: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub kind: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub function: Option<ToolCallFunctionDelta>,
}

#[derive(Debug, Default, Serialize)]
pub struct ToolCallFunctionDelta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

#[derive(Debug, Serialize)]
struct ChunkEnvelope<'a> {
    id: &'a str,
    object: &'static str,
    created: u64,
    model: &'a str,
    choices: Vec<ChunkChoice>,
}

#[derive(Debug, Serialize)]
struct ChunkChoice {
    index: u32,
    delta: ChunkDelta,
    #[serde(skip_serializing_if = "Option::is_none")]
    finish_reason: Option<&'static str>,
}

/// Encode one chunk as a `data: {...}\n\n` SSE event string. Caller
/// converts to a `Bytes` or `Sse::Event` as appropriate.
pub fn encode_chunk(
    id: &str,
    model: &str,
    created: u64,
    delta: ChunkDelta,
    finish_reason: Option<&'static str>,
) -> String {
    let env = ChunkEnvelope {
        id,
        object: "chat.completion.chunk",
        created,
        model,
        choices: vec![ChunkChoice {
            index: 0,
            delta,
            finish_reason,
        }],
    };
    let payload = serde_json::to_string(&env).unwrap_or_else(|_| "{}".into());
    format!("data: {payload}\n\n")
}

pub fn encode_done() -> &'static str {
    "data: [DONE]\n\n"
}

/// Build a "tool call start" delta: emits id, name, type, empty arguments.
pub fn tool_call_start_delta(tc: &ToolCall, index: u32) -> ChunkDelta {
    ChunkDelta {
        tool_calls: vec![ToolCallDelta {
            index,
            id: Some(tc.id.clone()),
            kind: Some("function"),
            function: Some(ToolCallFunctionDelta {
                name: Some(tc.function.name.clone()),
                arguments: Some(String::new()),
            }),
        }],
        ..Default::default()
    }
}

/// Build a "tool call arguments" delta: emits the full JSON arguments in
/// one delta (Phase 2 buffers the entire call; later phases can stream).
pub fn tool_call_args_delta(tc: &ToolCall, index: u32) -> ChunkDelta {
    ChunkDelta {
        tool_calls: vec![ToolCallDelta {
            index,
            id: None,
            kind: None,
            function: Some(ToolCallFunctionDelta {
                name: None,
                arguments: Some(tc.function.arguments.clone()),
            }),
        }],
        ..Default::default()
    }
}

pub fn text_delta(s: String) -> ChunkDelta {
    ChunkDelta {
        content: Some(s),
        ..Default::default()
    }
}

pub fn role_delta() -> ChunkDelta {
    ChunkDelta {
        role: Some("assistant"),
        ..Default::default()
    }
}
