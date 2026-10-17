//! DeepSeek-V4.1 chat encoding for `/v1/chat/completions` — a port of
//! DeepSeek's official reference implementation, `deepseek-recipe`
//! (github.com/deepseek-ai/deepseek-recipe, released with
//! DeepSeek-V4.1-Flash-1017): the OpenAI request conversion
//! (`protocol/openai/chat_completion/request/convert.rs`) and the V4.1 prompt
//! rendering (`deepseek-recipe-encoding/src/v4/{mod,dsv41}.rs`). Identity is
//! checked by `tests/oracle_prompt.rs`, which diffs this module against the
//! reference crates on property-generated requests (bytes and token ids).
//!
//! The checkpoint's `encoding/encoding.py` is a second reference; it and the
//! recipe disagree in places (reasoning of completed turns is DROPPED by
//! encoding.py's default `drop_thinking=True` and KEPT by the recipe; empty
//! system messages; tool-schema normalization; float exponents; ...). This
//! module follows the recipe; `tests/oracle_prompt.rs::
//! secondary_oracle_encoding_py_vs_recipe` lists the classes.
//!
//! Shape of a prompt: BOS; for message 0 a `<｜System｜>` prefix when it is a
//! system message or thinking is on, then (thinking only) the effort line
//! "Reasoning Effort: N (range 1-100, …)\n\n"; system messages
//! `<｜System｜>content` (the request's tools and `json_object` format appended
//! to the first one, which is created empty when missing); a user message
//! `<｜User｜>content`, joined to a preceding user / tool message by "\n\n"
//! instead; a tool result `<tool_result>…</tool_result>` the same way;
//! `<｜latest_reminder｜>content`; every assistant message
//! `<｜Assistant｜>` + (`<think>reasoning</think>` in thinking mode except at
//! index 0, else `</think>`) + content + DSML tool calls + EOS; then ALWAYS the
//! generation prompt `<｜Assistant｜><think>` / `</think>` (plus
//! "\n\n<｜DSML｜ calls>\n" for a required tool choice).
//!
//! Segments: `Seg::Ours` = markup and special tokens (tokenised with the
//! specials), `Seg::Client` = text supplied by the client (never a control
//! token) — the same contract as `prompt.rs`. The reference tokenizes the
//! whole string with specials enabled, so a client string spelling a special
//! token is a control token there and text here (the one deliberate
//! token-level difference; the prompt BYTES are identical).
//!
//! Images: every image part becomes the `<｜deepseek_image｜>` placeholder
//! (`Seg::Ours`, id 129264 — the recipe's bundled tokenizer spells the same id
//! `<｜image｜>`), joined with the message's other parts by "\n\n";
//! `vision_prompt::expand_images` replaces each placeholder id with the image
//! span. Images on system / assistant messages are rejected, as is client
//! text spelling the recipe's placeholder `<｜image｜>`.

use color_eyre::eyre::{self, eyre};
use v4flash_core::BpeVocab;

use crate::openai::types::{ChatCompletionRequest, ChatMessage, ContentPart, Role, ToolCall};
use crate::prompt::{
    encode_segments, ReasoningEffort, Seg, ASSISTANT_TEXT, BOS_TEXT, EOS_TEXT, THINK_BEGIN_TEXT, THINK_END_TEXT, USER_TEXT,
};
use crate::prompt_v41_templates::{
    REASONING_EFFORT_FMT_PREFIX, REASONING_EFFORT_FMT_SUFFIX, RESPONSE_FORMAT_PREFIX, TOOLS_HEAD, TOOLS_TAIL,
};

pub const SYSTEM_TEXT: &str = "<\u{ff5c}System\u{ff5c}>";
pub const LATEST_REMINDER_TEXT: &str = "<\u{ff5c}latest_reminder\u{ff5c}>";
const DSML: &str = "\u{ff5c}DSML\u{ff5c}";
/// The recipe's image placeholder spelling; client text carrying it is
/// rejected (`protocol/validation.rs::image_special_token_not_allowed`).
const RECIPE_IMAGE_TOKEN: &str = "<\u{ff5c}image\u{ff5c}>";

/// V4.1 reasoning effort: an integer budget in 1..=100. Named levels map to
/// 25 (minimal) / 50 / 75 / 100; the reference default is "high".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Effort(pub u8);

impl V41Effort {
    pub const DEFAULT: V41Effort = V41Effort(75);

    /// Case-insensitive. "minimal" (OpenAI / Hermes ladders) is 25 (owner 10-11): below "low", thinking still on.
    /// (KNOWN DEVIATION: the recipe maps "minimal" to low = 50.)
    pub fn from_name(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "minimal" => Some(V41Effort(25)),
            "low" => Some(V41Effort(50)),
            "high" => Some(V41Effort(75)),
            "max" => Some(V41Effort(100)),
            _ => None,
        }
    }

    /// Accepts an integer 1..=100 or one of low/high/max (as a JSON string).
    pub fn from_json(v: &serde_json::Value) -> eyre::Result<Self> {
        if let Some(n) = v.as_i64() {
            return if (1..=100).contains(&n) {
                Ok(V41Effort(n as u8))
            } else {
                Err(eyre!("reasoning_effort {n} outside 1..=100"))
            };
        }
        if let Some(s) = v.as_str() {
            // Numeric strings too ("88"): clients that map levels to numbers
            // often serialise them as strings.
            if let Ok(n) = s.trim().parse::<i64>() {
                return if (1..=100).contains(&n) {
                    Ok(V41Effort(n as u8))
                } else {
                    Err(eyre!("reasoning_effort {n} outside 1..=100"))
                };
            }
            return Self::from_name(s).ok_or_else(|| eyre!("reasoning_effort {s:?}: expected low|high|max or 1..100"));
        }
        Err(eyre!("reasoning_effort must be an integer or low|high|max"))
    }
}

// ----- JSON as the reference writes it -----

/// `deepseek_recipe_core::util::json_formatter::stringify_python_style`:
/// `json.dumps`-like spaced separators (", " and ": "), non-ASCII unescaped,
/// serde_json's float digits with single-digit negative exponents
/// re-packaged the Python way ("1e-05", "1e-06"). The digits are whatever
/// serde_json's formatter writes (zmij since serde_json 1.0.14x: "1e+30",
/// as `json.dumps`); the reference links the same serde_json, so this
/// formats through it rather than through `dsml::to_json_hf`'s own repr.
pub fn to_json_recipe(v: &serde_json::Value) -> String {
    use serde_json::ser::{CompactFormatter, Formatter};
    use std::io::{self, Write};
    struct RecipeFormatter;
    impl Formatter for RecipeFormatter {
        fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
            w.write_all(if first { b"" } else { b", " })
        }
        fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> io::Result<()> {
            w.write_all(if first { b"" } else { b", " })
        }
        fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> io::Result<()> {
            w.write_all(b": ")
        }
        fn write_f64<W: ?Sized + Write>(&mut self, w: &mut W, value: f64) -> io::Result<()> {
            let mut buf = Vec::with_capacity(32);
            CompactFormatter.write_f64(&mut buf, value)?;
            if !buf.contains(&b'e') {
                let (sign, rest): (&[u8], &[u8]) = match buf.strip_prefix(b"-") {
                    Some(rest) => (b"-", rest),
                    None => (b"", &buf),
                };
                if let Some(fraction) = rest.strip_prefix(b"0.") {
                    let zeros = fraction.iter().take_while(|&&b| b == b'0').count();
                    if zeros >= 4 {
                        if let [first_digit, rest_digits @ ..] = &fraction[zeros..] {
                            w.write_all(sign)?;
                            w.write_all(std::slice::from_ref(first_digit))?;
                            if !rest_digits.is_empty() {
                                w.write_all(b".")?;
                                w.write_all(rest_digits)?;
                            }
                            return write!(w, "e-{:02}", zeros + 1);
                        }
                    }
                }
            } else if let [mantissa @ .., b'e', b'-', digit] = buf.as_slice() {
                w.write_all(mantissa)?;
                return write!(w, "e-0{}", *digit as char);
            }
            w.write_all(&buf)
        }
    }
    let mut buf = Vec::with_capacity(128);
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, RecipeFormatter);
    match serde::Serialize::serialize(v, &mut ser) {
        Ok(()) => String::from_utf8(buf).unwrap_or_else(|_| "null".into()),
        Err(_) => "null".into(),
    }
}

// ----- the reference's conversation (deepseek_recipe_core::messages::InputMessage) -----

#[derive(Clone, Debug)]
enum Im {
    System(Vec<Seg>),
    /// `images`: each image's ordinal among the request messages' images in
    /// message order (the order the handler collects their pixels in).
    User { segs: Vec<Seg>, images: Vec<usize> },
    Assistant { content: String, reasoning: Option<String>, tool_calls: Vec<ToolCall> },
    Tool { segs: Vec<Seg>, images: Vec<usize>, id: String },
    Reminder(String),
}

/// A function tool after `convert_tools` (`deepseek_recipe_core::tools::ToolDefinition`).
#[derive(Clone, Debug)]
struct ToolDefinition {
    name: String,
    description: Option<String>,
    parameters: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ToolChoice {
    Auto,
    None,
    Required,
}

/// The reference's `Conversation`: everything rendering needs.
#[derive(Clone, Debug)]
struct Conversation {
    messages: Vec<Im>,
    thinking: bool,
    effort: V41Effort,
    tools: Vec<ToolDefinition>,
    tool_choice: ToolChoice,
    json_object: bool,
}

fn has_recipe_image_token(s: &str) -> bool {
    s.contains(RECIPE_IMAGE_TOKEN)
}

fn image_token_error() -> eyre::Report {
    eyre!(
        "The sub-string \"{RECIPE_IMAGE_TOKEN}\" in your input is not allowed for this API. Please remove \
         \"{RECIPE_IMAGE_TOKEN}\" from your input and try again."
    )
}

/// `ChatCompletionRequestContent::into_content`: a string as is; content
/// parts joined by "\n\n" with every image as the placeholder. (Our
/// deserializer keeps `parts` only when there is an image; otherwise
/// `content` is already the "\n\n"-joined text view.)
fn content_segs(m: &ChatMessage, next_img: &mut usize) -> (Vec<Seg>, Vec<usize>) {
    if m.parts.is_empty() {
        return (vec![Seg::Client(m.content.clone().unwrap_or_default())], Vec::new());
    }
    let mut segs = Vec::new();
    let mut images = Vec::new();
    for (i, p) in m.parts.iter().enumerate() {
        if i > 0 {
            segs.push(Seg::Ours("\n\n".into()));
        }
        match p {
            ContentPart::Text(t) => segs.push(Seg::Client(t.clone())),
            ContentPart::Image(_) => {
                segs.push(Seg::Ours(v4flash_vision::IMAGE_PLACEHOLDER.to_string()));
                images.push(*next_img);
                *next_img += 1;
            }
        }
    }
    (segs, images)
}

fn segs_text(segs: &[Seg]) -> String {
    segs.iter().map(Seg::text).collect()
}

/// `transform_messages` + the image-token check. The reference also rejects
/// tool messages that do not answer the immediately preceding assistant's
/// calls one-to-one, assistant messages with neither content nor tool calls,
/// and empty `tool_calls` arrays; the server renders those (KNOWN DEVIATION
/// TOOL_RESULT_STRUCTURE_NOT_ENFORCED, see tests/oracle_prompt.rs).
fn transform_messages(messages: &[ChatMessage]) -> eyre::Result<Vec<Im>> {
    if messages.is_empty() {
        return Err(eyre!("Empty input messages"));
    }
    let carries_token = messages.iter().any(|m| {
        m.content.as_deref().is_some_and(has_recipe_image_token)
            || m.parts.iter().any(|p| matches!(p, ContentPart::Text(t) if has_recipe_image_token(t)))
            || m.reasoning_content.as_deref().is_some_and(has_recipe_image_token)
            || m.tool_calls.iter().any(|tc| has_recipe_image_token(&tc.function.name) || has_recipe_image_token(&tc.function.arguments))
    });
    if carries_token {
        return Err(image_token_error());
    }
    let mut out = Vec::with_capacity(messages.len());
    let mut next_img = 0usize;
    for m in messages {
        match m.role {
            Role::System => {
                if m.has_images() {
                    return Err(eyre!("Image in system message is unsupported"));
                }
                let (segs, _) = content_segs(m, &mut next_img);
                if !segs_text(&segs).is_empty() {
                    out.push(Im::System(segs));
                }
            }
            Role::User => {
                let (segs, images) = content_segs(m, &mut next_img);
                out.push(Im::User { segs, images });
            }
            Role::Assistant => {
                if m.has_images() {
                    return Err(eyre!("Image in assistant message is unsupported"));
                }
                out.push(Im::Assistant {
                    content: m.content.clone().unwrap_or_default(),
                    reasoning: m.reasoning_content.clone(),
                    tool_calls: m.tool_calls.clone(),
                });
            }
            Role::Tool => {
                let (segs, images) = content_segs(m, &mut next_img);
                out.push(Im::Tool { segs, images, id: m.tool_call_id.clone().unwrap_or_default() });
            }
            Role::LatestReminder => {
                if m.has_images() {
                    return Err(eyre!("Image in latest_reminder message is unsupported"));
                }
                out.push(Im::Reminder(m.content.clone().unwrap_or_default()));
            }
        }
    }
    Ok(out)
}

/// `normalize_messages` (V4.1: mid-conversation system messages kept):
/// consecutive user messages merge, joined by "\n\n"; then
/// `sort_tool_results_by_call_order`.
fn normalize_messages(messages: Vec<Im>) -> Vec<Im> {
    let mut normalized: Vec<Im> = Vec::with_capacity(messages.len());
    for m in messages {
        match (m, normalized.last_mut()) {
            (Im::User { segs, images }, Some(Im::User { segs: prev, images: prev_images })) => {
                prev.push(Seg::Ours("\n\n".into()));
                prev.extend(segs);
                prev_images.extend(images);
            }
            (m, _) => normalized.push(m),
        }
    }
    sort_tool_results_by_call_order(&mut normalized);
    normalized
}

/// Within each run of user / tool messages, order the tool messages by the
/// position of their call id in the last assistant's tool calls (unknown
/// ids first, stable).
fn sort_tool_results_by_call_order(messages: &mut [Im]) {
    let mut order: Vec<String> = Vec::new();
    let mut idx = 0;
    while idx < messages.len() {
        match &messages[idx] {
            Im::Assistant { tool_calls, .. } if !tool_calls.is_empty() => {
                order = tool_calls.iter().map(|tc| tc.id.clone()).collect();
                idx += 1;
            }
            Im::User { .. } | Im::Tool { .. } => {
                let start = idx;
                while idx < messages.len() && matches!(messages[idx], Im::User { .. } | Im::Tool { .. }) {
                    idx += 1;
                }
                let tool_idxs: Vec<usize> = (start..idx).filter(|&i| matches!(messages[i], Im::Tool { .. })).collect();
                if tool_idxs.len() > 1 && !order.is_empty() {
                    let mut tools: Vec<Im> = tool_idxs.iter().map(|&i| messages[i].clone()).collect();
                    tools.sort_by_key(|m| match m {
                        Im::Tool { id, .. } => order.iter().position(|o| o == id).unwrap_or(0),
                        _ => 0,
                    });
                    for (&i, t) in tool_idxs.iter().zip(tools) {
                        messages[i] = t;
                    }
                }
            }
            _ => idx += 1,
        }
    }
}

/// `convert_tools`: function tools as `{name, description, parameters}`,
/// the tool choice applied. `tool_choice: "none"` drops every tool; a named
/// choice keeps only that tool and requires it (as "required" does), which
/// thinking mode does not support.
///
/// KNOWN DEVIATIONS (accepted here, rejected by the reference): tool names
/// outside `[A-Za-z0-9_-]{1,128}`, `parameters` that are not a JSON Schema of
/// type object, and non-`function` entries (skipped, as before).
fn convert_tools(req: &ChatCompletionRequest, thinking: bool) -> eyre::Result<(Vec<ToolDefinition>, ToolChoice)> {
    enum Choice {
        Absent,
        Mode(ToolChoice),
        Named(String),
    }
    let choice = match &req.tool_choice {
        None | Some(serde_json::Value::Null) => Choice::Absent,
        Some(serde_json::Value::String(s)) => match s.as_str() {
            "none" => Choice::Mode(ToolChoice::None),
            "auto" => Choice::Mode(ToolChoice::Auto),
            "required" => Choice::Mode(ToolChoice::Required),
            other => return Err(eyre!("tool_choice: unknown mode {other:?} (expected none, auto or required)")),
        },
        Some(serde_json::Value::Object(o)) => {
            let name = o.get("function").and_then(|f| f.get("name")).and_then(|n| n.as_str());
            match (o.get("type").and_then(|t| t.as_str()), name) {
                (Some("function"), Some(n)) => Choice::Named(n.to_string()),
                _ => return Err(eyre!("tool_choice: expected {{\"type\": \"function\", \"function\": {{\"name\": …}}}}")),
            }
        }
        Some(other) => return Err(eyre!("tool_choice: unsupported value {other}")),
    };
    if matches!(choice, Choice::Mode(ToolChoice::None)) {
        return Ok((Vec::new(), ToolChoice::None));
    }
    let tools: Vec<&crate::openai::types::ToolDef> =
        req.tools.as_deref().unwrap_or(&[]).iter().filter(|t| t.kind == "function").collect();
    if tools.is_empty() {
        return Ok((Vec::new(), ToolChoice::Auto));
    }
    let mut names = std::collections::HashSet::new();
    let mut definitions = Vec::with_capacity(tools.len());
    for (i, t) in tools.iter().enumerate() {
        let f = t.function.as_object().ok_or_else(|| eyre!("tools.{i}.function: expected an object"))?;
        let name = f
            .get("name")
            .and_then(|n| n.as_str())
            .ok_or_else(|| eyre!("tools.{i}.function.name: expected a string"))?
            .to_string();
        if !names.insert(name.clone()) {
            return Err(eyre!("Tool names must be unique"));
        }
        let description = match f.get("description") {
            None | Some(serde_json::Value::Null) => None,
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(_) => return Err(eyre!("tools.{i}.function.description: expected a string")),
        };
        let parameters = match f.get("parameters") {
            None | Some(serde_json::Value::Null) => serde_json::json!({}),
            Some(p) => p.clone(),
        };
        definitions.push(ToolDefinition { name, description, parameters });
    }
    let choice = match choice {
        Choice::Named(name) => {
            if !names.contains(&name) {
                return Err(eyre!("tool_choice: no tool named '{name}' was specified"));
            }
            definitions.retain(|t| t.name == name);
            ToolChoice::Required
        }
        Choice::Mode(ToolChoice::Required) => ToolChoice::Required,
        _ => ToolChoice::Auto,
    };
    if choice == ToolChoice::Required && thinking {
        return Err(eyre!("Thinking mode does not support this tool_choice"));
    }
    if definitions.iter().any(|t| {
        has_recipe_image_token(&t.name)
            || t.description.as_deref().is_some_and(has_recipe_image_token)
            || has_recipe_image_token(&to_json_recipe(&t.parameters))
    }) {
        return Err(image_token_error());
    }
    Ok((definitions, choice))
}

/// `convert_response_format`: `json_object` (the conversation must mention
/// "json"), `text` / `json_schema` (accepted, ignored), `regex` rejected.
fn convert_response_format(req: &ChatCompletionRequest, messages: &[Im]) -> eyre::Result<bool> {
    let Some(rf) = req.response_format.as_ref().filter(|v| !v.is_null()) else { return Ok(false) };
    match rf.get("type").and_then(|t| t.as_str()) {
        Some("json_object") => {
            let mentions_json = messages.iter().any(|m| {
                let text = match m {
                    Im::System(s) | Im::User { segs: s, .. } | Im::Tool { segs: s, .. } => segs_text(s),
                    Im::Assistant { content, .. } => content.clone(),
                    Im::Reminder(s) => s.clone(),
                };
                text.as_bytes().windows(4).any(|w| w.eq_ignore_ascii_case(b"json"))
            });
            if !mentions_json {
                return Err(eyre!(
                    "Prompt must contain the word 'json' in some form to use 'response_format' of type 'json_object'."
                ));
            }
            Ok(true)
        }
        Some("text") | Some("json_schema") => Ok(false),
        Some("regex") => Err(eyre!("response_format: only text and json_object are supported")),
        _ => Err(eyre!("response_format: expected a type of text, json_object or json_schema")),
    }
}

// ----- effort / thinking -----

/// The 4-state effort and the explicit numeric budget (if any) a request
/// asks for. `default_effort` is the operator default
/// (`--default-reasoning-effort`) for a request that names none.
///
/// A NUMERIC `reasoning_effort` ("88" or 88) is V4.1's native 1..=100
/// budget: thinking on, budget as given (0 = off). Clients that spread their
/// effort ladder over numbers (prime-agent's thinkingLevelMap) get finer
/// grain than the three named tiers. (KNOWN DEVIATION: the reference accepts
/// only none/minimal/low/medium/high/xhigh/max.)
pub fn request_effort(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> Result<(ReasoningEffort, Option<u8>), String> {
    let numeric_budget: Option<u8> = req
        .reasoning_effort
        .as_deref()
        .and_then(|s| s.trim().parse::<i64>().ok())
        .map(|n| if (0..=100).contains(&n) { Ok(n as u8) } else { Err(format!("reasoning_effort {n} outside 0..=100")) })
        .transpose()?;
    let effort = match numeric_budget {
        Some(0) => ReasoningEffort::Off,
        Some(n) if n >= 100 => ReasoningEffort::Max,
        Some(n) if n > 50 => ReasoningEffort::High,
        Some(_) => ReasoningEffort::Low,
        None => ReasoningEffort::from_request_fields_with_default(
            req.reasoning.as_deref(),
            req.reasoning_effort.as_deref(),
            default_effort,
        )?,
    };
    Ok((effort, numeric_budget))
}

/// The V4.1 thinking flag and 1..=100 budget for a request
/// (`ChatCompletionRequest::resolve_thinking`): an explicit `thinking`
/// (`enabled` / `adaptive` / `disabled`) wins over `reasoning_effort`, which
/// wins over the default; thinking without a named effort is "high" (75).
pub fn request_thinking_v41(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> Result<(bool, V41Effort), String> {
    let (effort, numeric_budget) = request_effort(req, default_effort)?;
    let thinking = match req.thinking.as_ref() {
        Some(t) => t.enabled()?,
        None => effort.thinking_enabled(),
    };
    // An explicit integer in the request wins; then a named level ("minimal"
    // = 25, ...: `reasoning_effort` wins, else letta's `reasoning`); then the
    // 4-state level, where Off (thinking forced on by `thinking`) is the
    // reference's default "high".
    let budget = numeric_budget
        .filter(|&n| n >= 1)
        .map(V41Effort)
        .or_else(|| match req.reasoning_effort.as_deref() {
            Some(s) => V41Effort::from_name(s),
            None => req.reasoning.as_deref().and_then(V41Effort::from_name),
        })
        .unwrap_or(match effort {
            ReasoningEffort::Low => V41Effort(50),
            ReasoningEffort::Off | ReasoningEffort::High => V41Effort(75),
            ReasoningEffort::Max => V41Effort(100),
        });
    Ok((thinking, budget))
}

// ----- rendering (`EncodingV4::render_conversation`) -----

fn tools_segs(tools: &[ToolDefinition], out: &mut Vec<Seg>) {
    // `render_tool_prompt`: each tool as {"name", "description" ("" when
    // absent), "parameters"}, one per line. TOOLS_HEAD/TOOLS_TAIL are OUR
    // markup (their `｜DSML｜` must tokenise to the special id); only the
    // schemas are client text.
    let schemas: Vec<String> = tools
        .iter()
        .map(|t| {
            to_json_recipe(&serde_json::json!({
                "name": t.name,
                "description": t.description.as_deref().unwrap_or_default(),
                "parameters": t.parameters,
            }))
        })
        .collect();
    out.push(Seg::Ours(TOOLS_HEAD.to_string()));
    out.push(Seg::Client(schemas.join("\n")));
    out.push(Seg::Ours(TOOLS_TAIL.to_string()));
}

/// `render_tool_arguments`: the arguments as a JSON object (anything else
/// becomes `{"arguments": <raw string>}`); string values raw with
/// `string="true"`, the rest as JSON with `string="false"`.
fn push_tool_call(out: &mut Vec<Seg>, tc: &ToolCall) {
    let args = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&tc.function.arguments).unwrap_or_else(|_| {
        let mut o = serde_json::Map::new();
        o.insert("arguments".to_string(), serde_json::Value::String(tc.function.arguments.clone()));
        o
    });
    out.push(Seg::Ours(format!("<{DSML} invoke name=\"")));
    out.push(Seg::Client(tc.function.name.clone()));
    out.push(Seg::Ours("\">\n".to_string()));
    for (i, (k, v)) in args.iter().enumerate() {
        if i > 0 {
            out.push(Seg::Ours("\n".to_string()));
        }
        let (is_str, body) = match v {
            serde_json::Value::String(s) => (true, s.clone()),
            other => (false, to_json_recipe(other)),
        };
        out.push(Seg::Ours(format!("<{DSML} parameter name=\"")));
        out.push(Seg::Client(k.clone()));
        out.push(Seg::Ours(format!("\" string=\"{}\">", if is_str { "true" } else { "false" })));
        out.push(Seg::Client(body));
        out.push(Seg::Ours(format!("</{DSML} parameter>")));
    }
    out.push(Seg::Ours(format!("\n</{DSML} invoke>")));
}

/// `render_message` for message `index`. `images` receives each emitted
/// placeholder's image ordinal, in prompt order.
fn render_message(out: &mut Vec<Seg>, images: &mut Vec<usize>, messages: &[Im], index: usize, conv: &Conversation) {
    let m = &messages[index];
    let prev = index.checked_sub(1).map(|i| &messages[i]);
    let effort_prompt = (index == 0 && conv.thinking)
        .then(|| format!("{REASONING_EFFORT_FMT_PREFIX}{}{REASONING_EFFORT_FMT_SUFFIX}", conv.effort.0));
    if index == 0 && (effort_prompt.is_some() || matches!(m, Im::System(_))) {
        out.push(Seg::Ours(SYSTEM_TEXT.to_string()));
    }
    if let Some(e) = effort_prompt {
        out.push(Seg::Ours(e));
    }
    let after_user = matches!(prev, Some(Im::User { .. } | Im::Tool { .. }));
    match m {
        Im::System(segs) => {
            if index > 0 {
                out.push(Seg::Ours(SYSTEM_TEXT.to_string()));
            }
            out.extend(segs.iter().cloned());
        }
        Im::User { segs, images: imgs } => {
            out.push(Seg::Ours(if after_user { "\n\n" } else { USER_TEXT }.to_string()));
            out.extend(segs.iter().cloned());
            images.extend(imgs);
        }
        Im::Reminder(s) => {
            out.push(Seg::Ours(LATEST_REMINDER_TEXT.to_string()));
            out.push(Seg::Client(s.clone()));
        }
        Im::Tool { segs, images: imgs, .. } => {
            out.push(Seg::Ours(if after_user { "\n\n" } else { USER_TEXT }.to_string()));
            out.push(Seg::Ours("<tool_result>".to_string()));
            out.extend(segs.iter().cloned());
            out.push(Seg::Ours("</tool_result>".to_string()));
            images.extend(imgs);
        }
        Im::Assistant { content, reasoning, tool_calls } => {
            out.push(Seg::Ours(ASSISTANT_TEXT.to_string()));
            if conv.thinking && index > 0 {
                out.push(Seg::Ours(THINK_BEGIN_TEXT.to_string()));
                out.push(Seg::Client(reasoning.clone().unwrap_or_default()));
                out.push(Seg::Ours(THINK_END_TEXT.to_string()));
            } else {
                out.push(Seg::Ours(THINK_END_TEXT.to_string()));
            }
            out.push(Seg::Client(content.clone()));
            if !tool_calls.is_empty() {
                out.push(Seg::Ours(format!("\n\n<{DSML} calls>\n")));
                for (i, tc) in tool_calls.iter().enumerate() {
                    if i > 0 {
                        out.push(Seg::Ours("\n".to_string()));
                    }
                    push_tool_call(out, tc);
                }
                out.push(Seg::Ours(format!("\n</{DSML} calls>")));
            }
            out.push(Seg::Ours(EOS_TEXT.to_string()));
        }
    }
}

fn render_conversation(mut conv: Conversation) -> (Vec<Seg>, Vec<usize>) {
    let mut messages = std::mem::take(&mut conv.messages);
    let has_tools = conv.tool_choice != ToolChoice::None && !conv.tools.is_empty();
    if has_tools || conv.json_object {
        if !matches!(messages.first(), Some(Im::System(_))) {
            messages.insert(0, Im::System(Vec::new()));
        }
        if let Some(Im::System(segs)) = messages.first_mut() {
            if has_tools {
                segs.push(Seg::Ours("\n\n".to_string()));
                tools_segs(&conv.tools, segs);
            }
            if conv.json_object {
                segs.push(Seg::Ours(format!("\n\n{RESPONSE_FORMAT_PREFIX}")));
                segs.push(Seg::Ours(to_json_recipe(&serde_json::json!({"type": "json_object"}))));
            }
        }
    }
    let mut out = vec![Seg::Ours(BOS_TEXT.to_string())];
    let mut images = Vec::new();
    for index in 0..messages.len() {
        render_message(&mut out, &mut images, &messages, index, &conv);
    }
    out.push(Seg::Ours(ASSISTANT_TEXT.to_string()));
    out.push(Seg::Ours(if conv.thinking { THINK_BEGIN_TEXT } else { THINK_END_TEXT }.to_string()));
    if conv.tool_choice == ToolChoice::Required && !conv.tools.is_empty() {
        out.push(Seg::Ours(format!("\n\n<{DSML} calls>\n")));
    }
    (out, images)
}

/// `ChatCompletionRequest::convert`, the parts that shape the prompt.
fn convert(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> eyre::Result<Conversation> {
    let (thinking, effort) = request_thinking_v41(req, default_effort).map_err(|e| eyre!(e))?;
    let (tools, tool_choice) = convert_tools(req, thinking)?;
    let messages = transform_messages(&req.messages)?;
    let json_object = convert_response_format(req, &messages)?;
    Ok(Conversation { messages: normalize_messages(messages), thinking, effort, tools, tool_choice, json_object })
}

// ----- request-level entry points (what `handler::chat_completions` renders) -----

/// Whether the model's output starts INSIDE a `<｜DSML｜ calls>` block (a
/// required / named tool choice: the prompt ends with the opening tag), so
/// the output scanner must start in that frame.
pub fn request_starts_in_tool_calls(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> bool {
    let Ok((thinking, _)) = request_thinking_v41(req, default_effort) else { return false };
    convert_tools(req, thinking).is_ok_and(|(tools, choice)| choice == ToolChoice::Required && !tools.is_empty())
}

/// The request's prompt segments and, per emitted image placeholder in
/// prompt order, the ordinal of its image among the request messages' images
/// in message order — exactly what the server renders for
/// `/v1/chat/completions`, from the request as deserialized.
/// `tests/oracle_prompt.rs` diffs this against DeepSeek's reference.
pub fn request_segments_v41(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> eyre::Result<(Vec<Seg>, Vec<usize>)> {
    Ok(render_conversation(convert(req, default_effort)?))
}

/// [`request_segments_v41`] as text.
pub fn render_request_text_v41(req: &ChatCompletionRequest, default_effort: ReasoningEffort) -> eyre::Result<String> {
    Ok(request_segments_v41(req, default_effort)?.0.iter().map(Seg::text).collect())
}

/// [`request_segments_v41`] as token ids: markup/specials via the vocab's
/// special ids, client text via BPE. `image_placeholder` is the vocab id of
/// `<｜deepseek_image｜>` (129264; `EngineHandle::image_placeholder_id`);
/// every image part renders as exactly one placeholder id, and the caller
/// must hand `vision_prompt::expand_images` the images in the returned order.
/// A request with images and no placeholder id is an error.
pub fn render_request_v41(
    vocab: &BpeVocab,
    req: &ChatCompletionRequest,
    default_effort: ReasoningEffort,
    image_placeholder: Option<i32>,
) -> eyre::Result<(Vec<i32>, Vec<usize>)> {
    if image_placeholder.is_none() && req.messages.iter().any(|m| m.has_images()) {
        return Err(eyre!(
            "V4.1 prompt: a message has image parts but the loaded vocab has no `{}` token",
            v4flash_vision::IMAGE_PLACEHOLDER
        ));
    }
    let (segs, image_order) = request_segments_v41(req, default_effort)?;
    Ok((encode_segments(vocab, &segs, image_placeholder), image_order))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(v: serde_json::Value) -> ChatCompletionRequest {
        let mut v = v;
        v["model"] = serde_json::json!("m");
        serde_json::from_value(v).unwrap()
    }

    fn render(v: serde_json::Value) -> String {
        render_request_text_v41(&req(v), ReasoningEffort::High).unwrap()
    }

    /// Named effort levels -> V4.1 budgets ("minimal" = 25, owner 10-11).
    #[test]
    fn effort_names_map_to_budgets() {
        for (s, n) in [("minimal", 25), ("Minimal", 25), ("low", 50), ("LOW", 50), ("high", 75), ("max", 100)] {
            assert_eq!(V41Effort::from_name(s), Some(V41Effort(n)), "{s}");
        }
        assert_eq!(V41Effort::from_name("medium"), None, "medium falls back to the 4-state effort (High -> 75)");
        assert_eq!(V41Effort::from_json(&serde_json::json!("minimal")).unwrap(), V41Effort(25));
    }

    /// The `encoding/README.md` "OpenAI-style messages" example: image parts
    /// render as the placeholder, parts joined by "\n\n".
    #[test]
    fn user_image_part_renders_readme_example() {
        let s = render(serde_json::json!({"thinking": {"type": "disabled"}, "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "有什么内容？"}
            ]
        }]}));
        assert_eq!(
            s,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>第一张图\n\n<\u{ff5c}deepseek_image\u{ff5c}>\n\n有什么内容？<\u{ff5c}Assistant\u{ff5c}></think>"
        );
        // The placeholder is OURS (a control segment); client text spelling it is not.
        let (segs, _) = request_segments_v41(
            &req(serde_json::json!({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}]})),
            ReasoningEffort::High,
        )
        .unwrap();
        assert!(segs.iter().any(|s| matches!(s, Seg::Ours(t) if t == v4flash_vision::IMAGE_PLACEHOLDER)));
        let (segs, _) =
            request_segments_v41(&req(serde_json::json!({"messages": [{"role": "user", "content": "x <｜deepseek_image｜> y"}]})), ReasoningEffort::High)
                .unwrap();
        assert!(segs.iter().all(|s| !matches!(s, Seg::Ours(t) if t == v4flash_vision::IMAGE_PLACEHOLDER)));
        // The recipe's own spelling in client text is rejected, as there.
        assert!(render_request_text_v41(&req(serde_json::json!({"messages": [{"role": "user", "content": "x <｜image｜>"}]})), ReasoningEffort::High).is_err());
    }

    /// Tool results that arrive out of call order are sorted into call order,
    /// images included; the reported image order must follow the prompt, not
    /// the messages, or the handler pairs each placeholder with the wrong
    /// picture.
    #[test]
    fn image_order_follows_sorted_tool_results() {
        let img = |u: &str| serde_json::json!({"type": "image_url", "image_url": {"url": u}});
        let r = req(serde_json::json!({"messages": [
            {"role": "user", "content": [{"type": "text", "text": "first"}, img("data:image/png;base64,AA==")]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "a", "type": "function", "function": {"name": "shot", "arguments": "{}"}},
                {"id": "b", "type": "function", "function": {"name": "shot", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "b", "content": [{"type": "text", "text": "B"}, img("data:image/png;base64,Qg==")]},
            {"role": "tool", "tool_call_id": "a", "content": [{"type": "text", "text": "A"}, img("data:image/png;base64,QQ==")]}
        ]}));
        let (segs, order) = request_segments_v41(&r, ReasoningEffort::High).unwrap();
        let s: String = segs.iter().map(Seg::text).collect();
        assert!(s.find("<tool_result>A").unwrap() < s.find("<tool_result>B").unwrap(), "{s}");
        assert!(s.contains("<tool_result>A\n\n<\u{ff5c}deepseek_image\u{ff5c}></tool_result>"), "{s}");
        // Message order: 0 = user's, 1 = B's, 2 = A's. Prompt order: user, A, B.
        assert_eq!(order, vec![0, 2, 1]);
    }

    #[test]
    fn images_on_assistant_are_rejected() {
        let r = req(serde_json::json!({"messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}
        ]}));
        assert!(render_request_text_v41(&r, ReasoningEffort::High).is_err());
    }

    /// The reference's float spelling: Python-style small negative
    /// exponents, serde_json's positive ones.
    /// The reference's float spelling, which with serde_json's current
    /// formatter (zmij) is also `json.dumps`'s.
    #[test]
    fn recipe_json_floats() {
        let cases = [
            (1e-6, "1e-06"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (2.5e-3, "0.0025"),
            (1e15, "1000000000000000.0"),
            (1e16, "1e+16"),
            (1e30, "1e+30"),
            (6.02e23, "6.02e+23"),
            (3.0, "3.0"),
            (-0.0, "-0.0"),
            (1e-10, "1e-10"),
            (5e-324, "5e-324"),
        ];
        for (v, want) in cases {
            assert_eq!(to_json_recipe(&serde_json::json!(v)), want, "{v:e}");
            assert_eq!(crate::dsml::to_json_hf(&serde_json::json!(v)), want, "{v:e} (json.dumps)");
        }
        assert_eq!(to_json_recipe(&serde_json::json!({"a": [1, "é"], "b": null})), "{\"a\": [1, \"é\"], \"b\": null}");
    }

    /// Synthetic agent request (system prompt, tool schemas, `rounds` of
    /// reasoning + tool call + tool result with code/log/JSON output, a user
    /// message every 10 rounds). Deterministic; no real traffic.
    pub(crate) fn synth_agent_request(rounds: usize) -> ChatCompletionRequest {
        let prose = "The agent reads the repository, runs the tests, and reports what changed. \
                     Keep answers short; cite file:line; never invent output. ";
        let code = "    fn step(&mut self, rows: usize) -> eyre::Result<()> {\n        let t = Instant::now();\n        \
                    for (i, r) in self.rows.iter_mut().enumerate().take(rows) {\n            r.x += 0.5 * i as f32; // update\n        }\n        \
                    self.ms += t.elapsed().as_secs_f64() * 1e3;\n        Ok(())\n    }\n";
        let log = "2026-10-03T21:07:45.123456Z  INFO deepstrix_server::multistream: stream done slot=0 prompt_tokens=262026 \
                   completion_tokens=347 tok_per_s=\"21.49\" finish=Stop sha=3f9a1c0b77de\n";
        let json = "{\"files\": [{\"path\": \"crates/v4flash-kernels/src/het/mtp.rs\", \"lines\": 2251, \"changed\": true}, \
                    {\"path\": \"docs/v41/KNOWN_BUGS.md\", \"lines\": 1460, \"changed\": false}], \"ok\": true}\n";
        let tools: Vec<serde_json::Value> = (0..12)
            .map(|i| {
                serde_json::json!({"type": "function", "function": {
                    "name": format!("tool_{i}"),
                    "description": prose.repeat(4),
                    "parameters": {"type": "object", "properties": {
                        "command": {"type": "string", "description": "The shell command to run."},
                        "timeout": {"type": "integer", "description": "Seconds."},
                        "path": {"type": "string", "description": "An absolute path."}
                    }, "required": ["command"]}
                }})
            })
            .collect();
        let mut v = vec![serde_json::json!({"role": "system", "content": prose.repeat(60)})];
        for r in 0..rounds {
            if r % 10 == 0 {
                v.push(serde_json::json!({"role": "user", "content": format!("round {r}: {}", prose.repeat(2))}));
            }
            let id = format!("call_{r}");
            v.push(serde_json::json!({
                "role": "assistant", "content": "",
                "reasoning_content": format!("Round {r}. {}", prose.repeat(10)),
                "tool_calls": [{"id": id, "type": "function", "function": {
                    "name": format!("tool_{}", r % 12),
                    "arguments": serde_json::json!({"command": format!("cargo test -p v4flash-kernels --release -- round_{r} 2>&1 | tail -40"), "timeout": 600}).to_string()
                }}]
            }));
            v.push(serde_json::json!({"role": "tool", "tool_call_id": id,
                "content": format!("{}{}{}", code.repeat(4), log.repeat(8), json.repeat(4))}));
        }
        req(serde_json::json!({"messages": v, "tools": tools}))
    }

    /// The BPE cost of ONE long pre-tokenizer piece (a symbol / whitespace /
    /// letter run): the greedy merge loop rescans every pair after each merge.
    ///   cargo test --release -p deepstrix-server --features v41 --lib -- --ignored --nocapture bpe_long_piece_cost
    #[test]
    #[ignore]
    fn bpe_long_piece_cost() {
        let tj = std::env::var("TOKENIZER_JSON").unwrap_or_else(|_| {
            "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/tokenizer.json".into()
        });
        let vocab = BpeVocab::from_tokenizer_json(&tj, Some("joyai-llm".to_string())).unwrap();
        for unit in ["\u{2500}", "=", "-", " ", "\n", "\u{7684}", "a", "\u{2588}", "\u{1F600}"] {
            let mut row = format!("{unit:?}:");
            for n in [100usize, 1000, 4000] {
                let s = unit.repeat(n);
                let t = std::time::Instant::now();
                let ids = vocab.encode(&s);
                let longest = v4flash_core::tokenizer::joyai_pre_tokenize(s.as_bytes()).iter().map(|p| p.len()).max().unwrap_or(0);
                row += &format!("  n={n}: {:.1} ms ({} ids, longest piece {} B)", t.elapsed().as_secs_f64() * 1e3, ids.len(), longest);
            }
            eprintln!("{row}");
        }
    }

    /// Where the per-turn prompt time goes (PR #2 stall, 2026-10-03): build vs
    /// encode on a synthetic ~1 MB agent conversation. Needs the V4.1
    /// tokenizer.json (`TOKENIZER_JSON`, default the hub's model dir).
    ///   cargo test --release -p deepstrix-server --features v41 --lib -- --ignored --nocapture prompt_cost_breakdown
    #[test]
    #[ignore]
    fn prompt_cost_breakdown() {
        let tj = std::env::var("TOKENIZER_JSON").unwrap_or_else(|_| {
            "/persist/hf_cache/models--deepseek-ai--DeepSeek-V4.1-Flash/snapshots/dba1be0a40aa45a94ad051997016db3960a90277/tokenizer.json".into()
        });
        let vocab = BpeVocab::from_tokenizer_json(&tj, Some("joyai-llm".to_string())).unwrap();
        let rounds: usize = std::env::var("SYNTH_ROUNDS").ok().and_then(|s| s.parse().ok()).unwrap_or(250);
        let r = synth_agent_request(rounds);
        let t = std::time::Instant::now();
        let (segs, _) = request_segments_v41(&r, ReasoningEffort::High).unwrap();
        let t_build = t.elapsed();
        let text: String = segs.iter().map(Seg::text).collect();
        let t = std::time::Instant::now();
        let ids = crate::prompt::encode_segments(&vocab, &segs, None);
        let t_enc = t.elapsed();
        let t = std::time::Instant::now();
        let pieces = v4flash_core::tokenizer::joyai_pre_tokenize(text.as_bytes());
        let t_pre = t.elapsed();
        let maxp = pieces.iter().map(|p| p.len()).max().unwrap_or(0);
        let t = std::time::Instant::now();
        let ids2 = vocab.encode(&text);
        let t_plain = t.elapsed();
        eprintln!(
            "rounds {rounds}: {} messages, {} bytes, {} tokens | build {:.1} ms | encode_segments {:.1} ms | \
             pre-tokenize {:.1} ms ({} pieces, longest {} bytes) | plain encode {:.1} ms ({} tokens)",
            r.messages.len(), text.len(), ids.len(), t_build.as_secs_f64() * 1e3, t_enc.as_secs_f64() * 1e3,
            t_pre.as_secs_f64() * 1e3, pieces.len(), maxp, t_plain.as_secs_f64() * 1e3, ids2.len()
        );
    }
}
