//! OpenAI Chat Completions wire types.
//!
//! Mirrors the public OpenAI API shape closely enough that letta-code's
//! `lmstudio_openai` provider routes through unchanged. Fields not used
//! in Phase 1 (`tool_calls`, `tool_call_id`) are present in the structs
//! so we can deserialize requests that include them without erroring —
//! Phase 2 wires the rendering / emission.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation. `content` is text-only in v1
/// (we don't yet decode the array-of-parts variant that OpenAI also
/// permits). Tool-related fields are tolerated but not yet rendered.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolCallFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    /// JSON-encoded arguments string (OpenAI convention).
    pub arguments: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    pub function: ToolDefFunction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDefFunction {
    pub name: String,
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema describing the parameters object.
    #[serde(default)]
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub tools: Option<Vec<ToolDef>>,
    #[serde(default)]
    pub stream: Option<bool>,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Letta passes this through; we treat it as a hot-cache hint in
    /// Phase 4 and ignore it for now.
    #[serde(default)]
    pub session_id: Option<String>,
    /// Optional seed for the host PRNG that drives multinomial sampling.
    /// Letta doesn't typically set this; default = a process-stable
    /// per-request value derived from request arrival time.
    #[serde(default)]
    pub seed: Option<u64>,
    /// OpenAI stream options. Currently we honor only
    /// `include_usage` (emit a final usage chunk in SSE responses).
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StreamOptions {
    #[serde(default)]
    pub include_usage: Option<bool>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str, // "chat.completion"
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ChatMessage,
    pub finish_reason: &'static str, // "stop" | "length" | "tool_calls"
}

#[derive(Debug, Clone, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}
