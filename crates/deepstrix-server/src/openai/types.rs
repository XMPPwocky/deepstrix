//! OpenAI Chat Completions wire types.
//!
//! Mirrors the public OpenAI API shape closely enough that letta-code's
//! `lmstudio_openai` provider routes through unchanged. Fields not used
//! in Phase 1 (`tool_calls`, `tool_call_id`) are present in the structs
//! so we can deserialize requests that include them without erroring —
//! Phase 2 wires the rendering / emission.

use serde::{Deserialize, Deserializer, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation. OpenAI permits `content` to be
/// either a plain string OR an array of typed parts (text / image_url /
/// input_audio …). We accept both; non-text parts are dropped at deserialize
/// time and a single concatenated string is exposed to the rest of the
/// pipeline.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    #[serde(default, deserialize_with = "deserialize_content_flexible")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// Custom deserializer that collapses OpenAI's flexible `content` field
/// into a plain `Option<String>`. Accepts:
///   * null → None
///   * "..." → Some("...")
///   * [{"type":"text","text":"..."}, ...] → joined text parts; non-text
///     parts (images, audio, etc.) are dropped with a debug log
fn deserialize_content_flexible<'de, D>(d: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    use serde::de::Error as _;
    let v = serde_json::Value::deserialize(d)?;
    match v {
        serde_json::Value::Null => Ok(None),
        serde_json::Value::String(s) => Ok(Some(s)),
        serde_json::Value::Array(parts) => {
            let mut out = String::new();
            for p in parts {
                match &p {
                    serde_json::Value::String(s) => out.push_str(s),
                    serde_json::Value::Object(obj) => {
                        let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                        match kind {
                            "text" | "input_text" => {
                                if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                                    out.push_str(t);
                                }
                            }
                            other => {
                                tracing::debug!(
                                    part_type = other,
                                    "dropping non-text content part"
                                );
                            }
                        }
                    }
                    _ => {
                        return Err(D::Error::custom(format!(
                            "content array element is not a string or object: {p}"
                        )))
                    }
                }
            }
            if out.is_empty() {
                Ok(None)
            } else {
                Ok(Some(out))
            }
        }
        other => Err(D::Error::custom(format!(
            "content must be null, a string, or an array; got {other}"
        ))),
    }
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
    /// OpenAI o1/o3-style reasoning toggle: "low" | "medium" | "high".
    /// Any non-"none" value enables think mode (the assistant turn
    /// opens with `<think>` instead of `</think>`). Letta sends this
    /// as `reasoning` in pi-ai's stream adapter; OpenAI clients send
    /// it as `reasoning_effort`. We accept either.
    #[serde(default)]
    pub reasoning: Option<String>,
    #[serde(default)]
    pub reasoning_effort: Option<String>,
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
