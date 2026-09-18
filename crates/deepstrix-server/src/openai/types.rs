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
    /// `"developer"` deserializes here too. OpenAI renamed the system role to
    /// `developer` for o1-era models and clients now send either; prime-agent
    /// sends `developer` and was getting a 422 that killed the request. The two
    /// mean the same thing to us, and serde `alias` keeps ONE variant so every
    /// `match` on Role stays exhaustive and the system-merge path is unchanged.
    /// We always SERIALIZE it back as `system`.
    #[serde(alias = "developer")]
    System,
    User,
    Assistant,
    Tool,
}

/// A single message in the conversation. OpenAI permits `content` to be
/// either a plain string OR an array of typed parts (text / image_url /
/// input_audio …). We accept both:
///
/// * `content` is the TEXT VIEW — the string as sent, or (for an array)
///   the text parts joined with `"\n\n"` (the Vision-Exp chat template's
///   `dsv4_media` join rule). Image parts are NOT in this string; every
///   pre-vision code path (system merge, tool results, diagnostics,
///   assistant history) keeps working on it unchanged.
/// * `parts` is the ordered part list, populated ONLY when the array
///   contained at least one image part. `prompt::render_prompt` renders a
///   user message from `parts` when non-empty (image part → the
///   `<｜deepseek_image｜>` placeholder id), else from `content`.
///
/// Non-text, non-image parts (audio …) are still dropped with a debug log.
#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: Option<String>,
    /// See the struct docs. Never serialized (image payloads would bloat
    /// the transcript dumps; the text view is what those need).
    #[serde(skip_serializing)]
    pub parts: Vec<ContentPart>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Reasoning the assistant emitted on this turn, as sent back by a
    /// client that stores it (letta / OpenAI `reasoning_content`). The chat
    /// template replays it inside `<think>…</think>` on history turns that
    /// keep reasoning (`ds4_server.c:1957` replays `m->reasoning`); without
    /// this field the replay could never happen and every historical
    /// assistant turn rendered as an empty think block.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl ChatMessage {
    /// Text-only message constructor (the shape every pre-vision caller built by hand).
    pub fn text(role: Role, content: impl Into<String>) -> Self {
        ChatMessage {
            role,
            content: Some(content.into()),
            parts: Vec::new(),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
            reasoning_content: None,
        }
    }

    /// Image parts of this message, in order.
    pub fn images(&self) -> impl Iterator<Item = &ImageInput> {
        self.parts.iter().filter_map(|p| match p {
            ContentPart::Image(img) => Some(img),
            ContentPart::Text(_) => None,
        })
    }

    pub fn has_images(&self) -> bool {
        self.images().next().is_some()
    }
}

/// One element of an array-form `content`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
    Image(ImageInput),
}

/// An `image_url` part as received. The URL is classified at parse time
/// but NOT fetched/decoded until `vision_prompt::load_image_bytes` — so a
/// bad payload surfaces as a clean HTTP 400 from the handler rather than
/// as axum's generic 422 JSON rejection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageInput {
    pub source: ImageSource,
    /// OpenAI `detail` hint (`auto` / `low` / `high`); accepted, ignored.
    pub detail: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ImageSource {
    /// `data:<media_type>;base64,<payload>` — payload kept undecoded.
    DataBase64 { media_type: String, payload: String },
    /// An absolute local path (`/abs/file.png` or `file:///abs/file.png`).
    LocalFile(std::path::PathBuf),
    /// Anything else (http(s) URLs, relative paths …). Rejected with a
    /// clear error when the request is prepared — the server never
    /// fetches remote URLs.
    Unsupported(String),
}

impl ImageSource {
    /// Classify an `image_url` string. Never fails; see [`ImageSource::Unsupported`].
    pub fn classify(url: &str) -> ImageSource {
        let trimmed = url.trim();
        if let Some(rest) = trimmed.strip_prefix("data:") {
            // RFC 2397: data:[<mediatype>][;base64],<data>
            if let Some((head, payload)) = rest.split_once(',') {
                let mut params = head.split(';');
                let media_type = params.next().unwrap_or("").to_string();
                let is_b64 = params.any(|p| p.eq_ignore_ascii_case("base64"));
                if is_b64 {
                    return ImageSource::DataBase64 {
                        media_type,
                        payload: payload.to_string(),
                    };
                }
            }
            return ImageSource::Unsupported(trimmed.to_string());
        }
        if let Some(path) = trimmed.strip_prefix("file://") {
            if path.starts_with('/') {
                return ImageSource::LocalFile(std::path::PathBuf::from(path));
            }
            return ImageSource::Unsupported(trimmed.to_string());
        }
        if trimmed.starts_with('/') {
            return ImageSource::LocalFile(std::path::PathBuf::from(trimmed));
        }
        ImageSource::Unsupported(trimmed.to_string())
    }
}

/// Raw wire shape; `content` is kept as a JSON value and normalised in
/// `ChatMessage`'s `Deserialize` impl below.
#[derive(Deserialize)]
struct ChatMessageWire {
    role: Role,
    #[serde(default)]
    content: Option<serde_json::Value>,
    #[serde(default)]
    tool_calls: Vec<ToolCall>,
    #[serde(default)]
    tool_call_id: Option<String>,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    reasoning_content: Option<String>,
}

impl<'de> Deserialize<'de> for ChatMessage {
    fn deserialize<D>(d: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::Error as _;
        let w = ChatMessageWire::deserialize(d)?;
        let (content, parts) = match w.content {
            None | Some(serde_json::Value::Null) => (None, Vec::new()),
            Some(serde_json::Value::String(s)) => (Some(s), Vec::new()),
            Some(serde_json::Value::Array(arr)) => {
                parse_content_parts(&arr).map_err(D::Error::custom)?
            }
            Some(other) => {
                return Err(D::Error::custom(format!(
                    "content must be null, a string, or an array; got {other}"
                )))
            }
        };
        Ok(ChatMessage {
            role: w.role,
            content,
            parts,
            tool_calls: w.tool_calls,
            tool_call_id: w.tool_call_id,
            name: w.name,
            reasoning_content: w.reasoning_content,
        })
    }
}

/// Normalise an array-form `content`:
///   * `"..."` / `{"type":"text"|"input_text","text":"..."}` → text part
///   * `{"type":"image_url","image_url":"<url>"|{"url":"<url>","detail":..}}`,
///     `{"type":"input_image","image_url":"<url>"}`, and the Anthropic
///     `{"type":"image","source":{"type":"base64","media_type":..,"data":..}}`
///     → image part
///   * any other `type` containing "image" → hard error (never a silent drop)
///   * anything else → dropped with a debug log (pre-vision behaviour)
///
/// Returns `(text_view, parts)`; `parts` is empty unless an image was
/// present. The text view joins text parts with `"\n\n"` (dsv4_media)
/// when the array carried an image, and with `""` otherwise — see the
/// JOIN RULE comment in the body for why the no-image path must keep the
/// pre-vision concatenation.
fn parse_content_parts(
    arr: &[serde_json::Value],
) -> Result<(Option<String>, Vec<ContentPart>), String> {
    let mut parts: Vec<ContentPart> = Vec::new();
    let mut any_image = false;
    for p in arr {
        match p {
            serde_json::Value::String(s) => parts.push(ContentPart::Text(s.clone())),
            serde_json::Value::Object(obj) => {
                let kind = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
                match kind {
                    "text" | "input_text" => {
                        if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
                            parts.push(ContentPart::Text(t.to_string()));
                        }
                    }
                    // `image` is the Anthropic spelling; it used to fall
                    // into the debug-log drop below, which answered the
                    // request as text-only with no error and no
                    // placeholder — the model confidently describing an
                    // image it never saw.
                    "image" | "image_url" | "input_image" => {
                        let (url, detail) = if let Some(src) = obj.get("source") {
                            // Anthropic: {"type":"image","source":{"type":"base64",
                            //             "media_type":"image/png","data":"..."}}
                            let o = src.as_object().ok_or_else(|| {
                                "image part: source must be an object".to_string()
                            })?;
                            let sty = o.get("type").and_then(|v| v.as_str()).unwrap_or("");
                            let url = match sty {
                                "base64" => {
                                    let mt = o
                                        .get("media_type")
                                        .and_then(|v| v.as_str())
                                        .unwrap_or("application/octet-stream");
                                    let data =
                                        o.get("data").and_then(|v| v.as_str()).ok_or_else(|| {
                                            "image part: source.data missing or not a string"
                                                .to_string()
                                        })?;
                                    format!("data:{mt};base64,{data}")
                                }
                                "url" => o
                                    .get("url")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| {
                                        "image part: source.url missing or not a string".to_string()
                                    })?
                                    .to_string(),
                                other => {
                                    return Err(format!(
                                        "image part: unsupported source.type {other:?} \
                                         (expected \"base64\" or \"url\")"
                                    ))
                                }
                            };
                            (url, o.get("detail").and_then(|v| v.as_str()).map(String::from))
                        } else {
                            match obj.get("image_url") {
                                Some(serde_json::Value::String(s)) => (s.clone(), None),
                                Some(serde_json::Value::Object(o)) => (
                                    o.get("url")
                                        .and_then(|v| v.as_str())
                                        .ok_or_else(|| {
                                            "image_url part: image_url.url missing or not a string"
                                                .to_string()
                                        })?
                                        .to_string(),
                                    o.get("detail").and_then(|v| v.as_str()).map(String::from),
                                ),
                                _ => {
                                    return Err(
                                        "image part: expected `image_url` (string or {url,detail}) \
                                         or an Anthropic `source` object"
                                            .to_string(),
                                    )
                                }
                            }
                        };
                        any_image = true;
                        parts.push(ContentPart::Image(ImageInput {
                            source: ImageSource::classify(&url),
                            detail,
                        }));
                    }
                    // Fail loudly on anything else that calls itself an
                    // image: dropping it silently answers the request as
                    // text-only, which is worse than a 400.
                    other if other.contains("image") => {
                        return Err(format!(
                            "unsupported image content part type {other:?}; use \"image_url\", \
                             \"input_image\" or \"image\""
                        ))
                    }
                    other => {
                        tracing::debug!(part_type = other, "dropping unsupported content part");
                    }
                }
            }
            _ => {
                return Err(format!(
                    "content array element is not a string or object: {p}"
                ))
            }
        }
    }
    let text_view = {
        let texts: Vec<&str> = parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                ContentPart::Image(_) => None,
            })
            .collect();
        // JOIN RULE. `dsv4_media` (template line 25) joins EVERY part of
        // an array-form `content` with "\n\n", unconditionally — it never
        // looks at whether a part is an image. So this join is "\n\n" in
        // both branches:
        //
        //   * with an image, `render_prompt` renders from `parts` and
        //     `push_user_parts` inserts the separators itself; this text
        //     view only has to agree with it;
        //   * with NO image, `parts` is dropped and the message renders
        //     from this string, so the separators have to be in here.
        //
        // The pre-vision deserializer concatenated text parts with no
        // separator at all, which silently diverged from the template for
        // any client sending multi-block text content (Anthropic-shaped
        // clients, multi-block tool text). Fixing it costs those
        // conversations a one-off KV-snapshot miss, the same price every
        // other template-fidelity fix in this change pays. Pinned by the
        // `user_turn_with_text_parts_only` golden case.
        let joined = texts.join("\n\n");
        if joined.is_empty() {
            None
        } else {
            Some(joined)
        }
    };
    Ok((text_view, if any_image { parts } else { Vec::new() }))
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

/// One entry of the request's `tools` array.
///
/// `function` is kept as the **raw JSON value the client sent**, not a
/// destructured struct. The chat template renders it with `tool['function'] |
/// tojson`, so every field the client included has to survive — including
/// ones we have no name for (`strict`, vendor `x-*` hints) — in the client's
/// own key order (`serde_json`'s `preserve_order` feature, enabled workspace
/// wide, keeps `Value::Object` insertion-ordered). Destructuring here used to
/// drop unknown fields, force the key order to name/description/parameters,
/// and emit `"description": null` for tools that never had one.
/// `function` is `#[serde(default)]` (→ `Value::Null` when absent) because the
/// template only ever reads it for entries whose `type` is `"function"`
/// (lines 78-80) and silently skips every other entry — a client that declares
/// a provider builtin as `{"type": "web_search"}` with no `function` key must
/// not get a 400. An entry that DOES claim `type: "function"` but carries no
/// object there is rejected by [`crate::prompt::build_segments`] instead of
/// being rendered as the literal line `null` into the schema block. (Under
/// jinja that case raises out of `tojson`, so a 400 is the faithful answer.)
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolDef {
    #[serde(rename = "type")]
    pub kind: String, // "function"
    #[serde(default)]
    pub function: serde_json::Value,
}

impl ToolDef {
    /// The tool's `function.name`, when it has one. Diagnostics only —
    /// rendering never goes through this.
    pub fn name(&self) -> Option<&str> {
        self.function.get("name").and_then(|v| v.as_str())
    }
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
    /// V4-Flash 0731 reasoning-effort level. Mapped by
    /// `prompt::ReasoningEffort::from_request_fields_with_default`:
    ///   ""|"none"|"off"|"disabled"|"false" → Off (no `<think>` phase)
    ///   absent/null                        → the operator default
    ///                                        (`prompt::DEFAULT_EFFORT` =
    ///                                        Low, overridable with
    ///                                        `--default-reasoning-effort`)
    ///   "low"                              → Low (think on, no preamble)
    ///   "medium"|"high"|"xhigh"            → High (0731 "high" preamble)
    ///   "max"                              → Max  (0731 "max" preamble)
    ///   anything else                      → HTTP 400
    /// Letta sends this as `reasoning` in pi-ai's stream adapter; OpenAI
    /// clients send it as `reasoning_effort`. We accept either;
    /// `reasoning_effort` wins if both are set.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(json: &str) -> ChatMessage {
        serde_json::from_str::<ChatMessage>(json).expect("parse")
    }

    #[test]
    fn content_string_is_text_view_without_parts() {
        let m = parse(r#"{"role":"user","content":"hello"}"#);
        assert_eq!(m.content.as_deref(), Some("hello"));
        assert!(m.parts.is_empty());
        assert!(!m.has_images());
        let m = parse(r#"{"role":"assistant","content":null}"#);
        assert_eq!(m.content, None);
        let m = parse(r#"{"role":"assistant"}"#);
        assert_eq!(m.content, None);
    }

    #[test]
    fn text_only_array_joins_parts_the_template_way() {
        // `dsv4_media` (template line 25) joins every part of an array-form
        // content with "\n\n" whether or not an image is present. The
        // pre-vision deserializer concatenated with no separator, which
        // diverged from the template for multi-block text content; this now
        // matches. Pinned end-to-end by the `user_turn_with_text_parts_only`
        // golden case.
        let m = parse(
            r#"{"role":"user","content":[{"type":"text","text":"a"},"b",{"type":"input_text","text":"c"}]}"#,
        );
        assert_eq!(m.content.as_deref(), Some("a\n\nb\n\nc"));
        // Text-only arrays still keep the legacy single-string shape (no
        // `parts`), so `build_segments` renders them from `content`.
        assert!(m.parts.is_empty());
        // A single text block is unchanged — no trailing/leading separator.
        let one = parse(r#"{"role":"user","content":[{"type":"text","text":"only"}]}"#);
        assert_eq!(one.content.as_deref(), Some("only"));
    }

    #[test]
    fn anthropic_image_part_is_accepted() {
        let m = parse(
            r#"{"role":"user","content":[
                {"type":"image","source":{"type":"base64","media_type":"image/png","data":"iVBORw0KGgo="}},
                {"type":"text","text":"what is this?"}
            ]}"#,
        );
        assert!(m.has_images());
        assert_eq!(
            m.images().next().unwrap().source,
            ImageSource::DataBase64 {
                media_type: "image/png".into(),
                payload: "iVBORw0KGgo=".into()
            }
        );
    }

    #[test]
    fn unknown_image_shaped_part_is_an_error_not_a_silent_drop() {
        assert!(serde_json::from_str::<ChatMessage>(
            r#"{"role":"user","content":[{"type":"image_file","image_file":{"file_id":"x"}}]}"#
        )
        .is_err());
        // Non-image unknown parts still drop quietly (pre-vision behaviour).
        let m = parse(r#"{"role":"user","content":[{"type":"audio","audio":{}},{"type":"text","text":"hi"}]}"#);
        assert_eq!(m.content.as_deref(), Some("hi"));
    }

    #[test]
    fn content_array_with_data_url_image() {
        let m = parse(
            r#"{"role":"user","content":[
                {"type":"text","text":"what is this?"},
                {"type":"image_url","image_url":{"url":"data:image/png;base64,iVBORw0KGgo=","detail":"high"}},
                {"type":"text","text":"answer briefly"}
            ]}"#,
        );
        assert_eq!(m.content.as_deref(), Some("what is this?\n\nanswer briefly"));
        assert_eq!(m.parts.len(), 3);
        assert!(m.has_images());
        let img = m.images().next().unwrap();
        assert_eq!(img.detail.as_deref(), Some("high"));
        assert_eq!(
            img.source,
            ImageSource::DataBase64 {
                media_type: "image/png".into(),
                payload: "iVBORw0KGgo=".into()
            }
        );
        assert!(matches!(m.parts[0], ContentPart::Text(ref t) if t == "what is this?"));
        assert!(matches!(m.parts[2], ContentPart::Text(ref t) if t == "answer briefly"));
    }

    #[test]
    fn image_url_string_form_and_local_paths() {
        let m = parse(
            r#"{"role":"user","content":[{"type":"image_url","image_url":"/tmp/x.png"},{"type":"input_image","image_url":"file:///tmp/y.jpg"}]}"#,
        );
        assert_eq!(m.content, None);
        let srcs: Vec<_> = m.images().map(|i| i.source.clone()).collect();
        assert_eq!(
            srcs,
            vec![
                ImageSource::LocalFile("/tmp/x.png".into()),
                ImageSource::LocalFile("/tmp/y.jpg".into())
            ]
        );
    }

    #[test]
    fn http_urls_and_relative_paths_are_classified_unsupported() {
        for u in [
            "http://example.com/a.png",
            "https://example.com/a.png",
            "relative/a.png",
            "data:image/png,notbase64",
            "file://relative",
            "",
        ] {
            assert!(
                matches!(ImageSource::classify(u), ImageSource::Unsupported(_)),
                "{u}"
            );
        }
        // Parsing does NOT fail — rejection is deferred to request prep
        // so it becomes a clear HTTP 400 (see vision_prompt::load_image_bytes).
        let m = parse(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"url":"https://example.com/a.png"}}]}"#,
        );
        assert!(matches!(
            m.images().next().unwrap().source,
            ImageSource::Unsupported(ref s) if s == "https://example.com/a.png"
        ));
    }

    #[test]
    fn malformed_content_is_a_parse_error() {
        assert!(serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":42}"#).is_err());
        assert!(serde_json::from_str::<ChatMessage>(r#"{"role":"user","content":[42]}"#).is_err());
        assert!(serde_json::from_str::<ChatMessage>(
            r#"{"role":"user","content":[{"type":"image_url","image_url":{"detail":"low"}}]}"#
        )
        .is_err());
    }

    #[test]
    fn full_request_with_image_parses() {
        let r: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,/9j/"}},{"type":"text","text":"hi"}]}]}"#,
        )
        .unwrap();
        assert_eq!(r.messages.len(), 2);
        assert!(r.messages[1].has_images());
        assert_eq!(r.messages[1].content.as_deref(), Some("hi"));
    }
}

#[cfg(test)]
mod role_alias_tests {
    use super::*;

    /// prime-agent sends OpenAI's `developer` role; rejecting it 422'd the whole
    /// request ("unknown variant `developer`") and the client's retry gave up.
    #[test]
    fn developer_role_deserializes_as_system() {
        let m: ChatMessage = serde_json::from_str(
            r#"{"role":"developer","content":"you are a helpful agent"}"#,
        )
        .expect("developer role must deserialize");
        assert_eq!(m.role, Role::System);
    }

    #[test]
    fn system_still_works_and_round_trips_as_system() {
        let m: ChatMessage = serde_json::from_str(r#"{"role":"system","content":"x"}"#).unwrap();
        assert_eq!(m.role, Role::System);
        // We normalise on the way out: one spelling downstream.
        assert_eq!(serde_json::to_value(Role::System).unwrap(), serde_json::json!("system"));
    }

    #[test]
    fn the_other_roles_are_untouched() {
        for (s, want) in [
            ("user", Role::User),
            ("assistant", Role::Assistant),
            ("tool", Role::Tool),
        ] {
            let m: ChatMessage =
                serde_json::from_str(&format!(r#"{{"role":"{s}","content":"x"}}"#)).unwrap();
            assert_eq!(m.role, want);
        }
    }
}
