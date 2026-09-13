//! DeepSeek-V4.1 chat encoding — a port of the text path of the checkpoint's
//! `encoding/encoding.py` (`encode_messages`), which is a Python encoder, not
//! a Jinja template. Byte-exactness is checked by
//! `tests/v41_prompt_vectors.rs` against vectors the reference produced
//! (`scripts/v41_oracle/gen_prompt_vectors.py`).
//!
//! V4.1 vs V4-Flash (`prompt.rs`): DSML tag names carry a leading space and the
//! block is ` calls` (`<｜DSML｜ calls>`, `<｜DSML｜ invoke …>`, `<｜DSML｜ parameter …>`);
//! reasoning effort is a numeric budget line ("Reasoning Effort: N (range
//! 1-100, …)", named low/high/max → 50/75/100, thinking mode only, before the
//! first message and preceded by `<｜System｜>`); mid-conversation system
//! messages are allowed and count as user turns for the assistant header; tool
//! results are merged into the user turn as `<tool_result>…</tool_result>`
//! blocks; the assistant turn is reasoning + `</think>` + content + tool calls +
//! EOS; reasoning before the last user turn is dropped unless any message
//! carries tools; an optional context prefix suppresses the BOS.
//!
//! Segments: `Seg::Ours` = markup and special tokens (tokenised with the
//! specials), `Seg::Client` = text supplied by the client (never a control
//! token) — the same contract as `prompt.rs`.
//!
//! Images (`encoding.py::_process_image_blocks`): every image content block
//! becomes the `<｜deepseek_image｜>` placeholder — inline among a user
//! message's parts, or inline inside a `<tool_result>` body — joined with
//! the surrounding text by "\n\n" like any other block. The placeholder is
//! always a `Seg::Ours` so it tokenises to the special id (`image_token_id`
//! 129264); client text spelling it out BPE-splits. `vision_prompt::
//! expand_images` then replaces each placeholder id with the image span.
//! Images on system / assistant messages are rejected (as in `prompt.rs`).
//!
//! Not supported (no request field for them): the `latest_reminder` role, the
//! message-level `response_format`, `task` special tokens.

use color_eyre::eyre::{self, eyre};
use v4flash_core::BpeVocab;

use crate::dsml::to_json_hf;
use crate::openai::types::{ChatMessage, ContentPart, Role, ToolCall, ToolDef};
use crate::prompt::{
    encode_segments, Seg, ASSISTANT_TEXT, BOS_TEXT, EOS_TEXT, THINK_BEGIN_TEXT, THINK_END_TEXT, USER_TEXT,
};
use crate::prompt_v41_templates::{
    REASONING_EFFORT_FMT_PREFIX, REASONING_EFFORT_FMT_SUFFIX, TOOLS_HEAD, TOOLS_TAIL,
};

pub const SYSTEM_TEXT: &str = "<\u{ff5c}System\u{ff5c}>";
const DSML: &str = "\u{ff5c}DSML\u{ff5c}";

/// V4.1 reasoning effort: an integer budget in 1..=100. Named levels map to
/// 50 / 75 / 100; the reference default is "high".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct V41Effort(pub u8);

impl V41Effort {
    pub const DEFAULT: V41Effort = V41Effort(75);

    pub fn from_name(s: &str) -> Option<Self> {
        match s {
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
            return Self::from_name(s).ok_or_else(|| eyre!("reasoning_effort {s:?}: expected low|high|max or 1..100"));
        }
        Err(eyre!("reasoning_effort must be an integer or low|high|max"))
    }
}

// ----- internal message form (the reference's post-merge messages) -----

#[derive(Clone, Debug, PartialEq, Eq)]
enum R {
    System,
    User,
    Assistant,
}

/// One piece of a tool-result body: text, or an image part (rendered as the
/// placeholder inline, where `_process_image_blocks` puts it).
#[derive(Clone, Debug)]
enum Part {
    Text(String),
    Image,
}

#[derive(Clone, Debug)]
enum Block {
    Text(String),
    /// An image part of a user message → the placeholder.
    Image,
    ToolResult(Vec<Part>),
}

#[derive(Clone, Debug)]
struct Msg {
    role: R,
    /// Plain content (system / assistant / user without blocks).
    content: Option<String>,
    /// User content blocks after `merge_tool_messages`.
    blocks: Option<Vec<Block>>,
    /// Tool result blocks carry the call id for `sort_tool_results_by_call_order`.
    block_ids: Vec<Option<String>>,
    tool_calls: Vec<ToolCall>,
    reasoning: Option<String>,
    /// Tools attached to this message (the reference attaches the request's
    /// tools to message 0 and renders them only on a system message).
    tools: Option<Vec<ToolDef>>,
}

fn text_of(m: &ChatMessage) -> eyre::Result<Option<String>> {
    if m.parts.iter().any(|p| matches!(p, ContentPart::Image(_))) {
        return Err(eyre!(
            "V4.1 prompt: message ({:?}) has image parts; images are only supported in user and \
             tool-result messages",
            m.role
        ));
    }
    Ok(m.content.clone())
}

/// A user message's content blocks (`parts` when it carries images — the
/// deserializer keeps `parts` only then — else the text view).
fn user_blocks(m: &ChatMessage) -> Vec<Block> {
    if m.parts.is_empty() {
        return vec![Block::Text(m.content.clone().unwrap_or_default())];
    }
    m.parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(t) => Block::Text(t.clone()),
            ContentPart::Image(_) => Block::Image,
        })
        .collect()
}

/// A tool message's result body as parts.
fn tool_parts(m: &ChatMessage) -> Vec<Part> {
    if m.parts.is_empty() {
        return vec![Part::Text(m.content.clone().unwrap_or_default())];
    }
    m.parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(t) => Part::Text(t.clone()),
            ContentPart::Image(_) => Part::Image,
        })
        .collect()
}

/// `merge_tool_messages`: tool messages become `<tool_result>` blocks in the
/// preceding user turn (or a new one); consecutive user turns merge.
fn merge(messages: &[ChatMessage], tools: Option<&[ToolDef]>) -> eyre::Result<Vec<Msg>> {
    let mut out: Vec<Msg> = Vec::new();
    for (i, m) in messages.iter().enumerate() {
        match m.role {
            Role::Tool => {
                let block = Block::ToolResult(tool_parts(m));
                let id = m.tool_call_id.clone();
                match out.last_mut() {
                    Some(last) if last.role == R::User && last.blocks.is_some() => {
                        last.blocks.as_mut().unwrap().push(block);
                        last.block_ids.push(id);
                    }
                    _ => out.push(Msg {
                        role: R::User,
                        content: None,
                        blocks: Some(vec![block]),
                        block_ids: vec![id],
                        tool_calls: vec![],
                        reasoning: None,
                        tools: None,
                    }),
                }
            }
            Role::User => {
                let blocks = user_blocks(m);
                let n = blocks.len();
                match out.last_mut() {
                    Some(last) if last.role == R::User && last.blocks.is_some() => {
                        last.blocks.as_mut().unwrap().extend(blocks);
                        last.block_ids.extend(std::iter::repeat_n(None, n));
                    }
                    _ => out.push(Msg {
                        role: R::User,
                        content: None,
                        blocks: Some(blocks),
                        block_ids: vec![None; n],
                        tool_calls: vec![],
                        reasoning: None,
                        tools: if i == 0 { tools.map(|t| t.to_vec()) } else { None },
                    }),
                }
            }
            Role::System | Role::Assistant => out.push(Msg {
                role: if m.role == Role::System { R::System } else { R::Assistant },
                content: text_of(m)?,
                blocks: None,
                block_ids: vec![],
                tool_calls: m.tool_calls.clone(),
                reasoning: m.reasoning_content.clone(),
                tools: if i == 0 { tools.map(|t| t.to_vec()) } else { None },
            }),
        }
    }
    Ok(out)
}

/// `sort_tool_results_by_call_order`: within a user turn, order the tool
/// result blocks by the preceding assistant's tool-call order.
fn sort_tool_results(msgs: &mut [Msg]) {
    let mut order: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for m in msgs.iter_mut() {
        match m.role {
            R::Assistant if !m.tool_calls.is_empty() => {
                order.clear();
                for (i, tc) in m.tool_calls.iter().enumerate() {
                    if !tc.id.is_empty() {
                        order.insert(tc.id.clone(), i);
                    }
                }
            }
            R::User => {
                let Some(blocks) = m.blocks.as_mut() else { continue };
                let idxs: Vec<usize> = (0..blocks.len()).filter(|&i| matches!(blocks[i], Block::ToolResult(_))).collect();
                if idxs.len() > 1 && !order.is_empty() {
                    let mut tool_blocks: Vec<(usize, Block, Option<String>)> =
                        idxs.iter().map(|&i| (i, blocks[i].clone(), m.block_ids[i].clone())).collect();
                    // Python `sorted` is stable; key = order of the call id (0 if unknown).
                    tool_blocks.sort_by_key(|(_, _, id)| id.as_deref().and_then(|s| order.get(s)).copied().unwrap_or(0));
                    for (slot, (_, b, id)) in idxs.iter().zip(tool_blocks.into_iter()) {
                        blocks[*slot] = b;
                        m.block_ids[*slot] = id;
                    }
                }
            }
            _ => {}
        }
    }
}

/// The reference's last-user definition: a user, or a system message that is
/// not the first message.
fn last_user_index(msgs: &[Msg]) -> isize {
    for i in (0..msgs.len()).rev() {
        if msgs[i].role == R::User || (msgs[i].role == R::System && i > 0) {
            return i as isize;
        }
    }
    -1
}

/// `_drop_thinking_messages`: keep everything, but strip reasoning from
/// assistant turns before the last user turn.
fn drop_thinking(msgs: &[Msg]) -> Vec<Msg> {
    let lu = last_user_index(msgs);
    let mut out = Vec::with_capacity(msgs.len());
    for (i, m) in msgs.iter().enumerate() {
        if m.role == R::Assistant && (i as isize) < lu {
            let mut c = m.clone();
            c.reasoning = None;
            out.push(c);
        } else {
            out.push(m.clone());
        }
    }
    out
}

fn tools_text(tools: &[ToolDef]) -> String {
    // tools_from_openai_format: the function object as given (name re-assigned
    // to itself; namespaces are not supported here), json.dumps(ensure_ascii=False).
    let schemas: Vec<String> = tools.iter().map(|t| to_json_hf(&t.function)).collect();
    format!("{TOOLS_HEAD}{}{TOOLS_TAIL}", schemas.join("\n"))
}

/// `encode_arguments_to_dsml`: the tool call's arguments (a JSON string,
/// possibly double-encoded) → parameter tags.
fn push_tool_call(out: &mut Vec<Seg>, tc: &ToolCall) {
    let mut args: serde_json::Value = serde_json::Value::String(tc.function.arguments.clone());
    for _ in 0..2 {
        match &args {
            serde_json::Value::String(s) => match serde_json::from_str::<serde_json::Value>(s) {
                Ok(v) => args = v,
                Err(_) => break,
            },
            _ => break,
        }
    }
    let obj = match args {
        serde_json::Value::Object(o) => o,
        _ => {
            let mut o = serde_json::Map::new();
            o.insert("arguments".to_string(), serde_json::Value::String(tc.function.arguments.clone()));
            o
        }
    };
    out.push(Seg::Ours(format!("<{DSML} invoke name=\"")));
    out.push(Seg::Client(tc.function.name.clone()));
    out.push(Seg::Ours("\">\n".to_string()));
    let n = obj.len();
    for (i, (k, v)) in obj.iter().enumerate() {
        let (is_str, body) = match v {
            serde_json::Value::String(s) => (true, s.clone()),
            other => (false, to_json_hf(other)),
        };
        out.push(Seg::Ours(format!("<{DSML} parameter name=\"")));
        out.push(Seg::Client(k.clone()));
        out.push(Seg::Ours(format!("\" string=\"{}\">", if is_str { "true" } else { "false" })));
        out.push(Seg::Client(body));
        out.push(Seg::Ours(format!("</{DSML} parameter>")));
        if i + 1 < n {
            out.push(Seg::Ours("\n".to_string()));
        }
    }
    out.push(Seg::Ours(format!("\n</{DSML} invoke>")));
}

/// `render_message`.
fn render_message(out: &mut Vec<Seg>, index: usize, msgs: &[Msg], thinking: bool, drop: bool, effort: V41Effort) {
    let m = &msgs[index];
    let lu = last_user_index(msgs);
    let effort_prompt = if index == 0 && thinking {
        Some(format!("{REASONING_EFFORT_FMT_PREFIX}{}{REASONING_EFFORT_FMT_SUFFIX}", effort.0))
    } else {
        None
    };
    if index == 0 && (effort_prompt.is_some() || m.role == R::System) {
        out.push(Seg::Ours(SYSTEM_TEXT.to_string()));
    }
    if let Some(e) = effort_prompt {
        out.push(Seg::Ours(e));
    }
    match m.role {
        R::System => {
            if index > 0 {
                out.push(Seg::Ours(SYSTEM_TEXT.to_string()));
            }
            out.push(Seg::Client(m.content.clone().unwrap_or_default()));
            if let Some(tools) = m.tools.as_ref().filter(|t| !t.is_empty()) {
                out.push(Seg::Ours("\n\n".to_string()));
                out.push(Seg::Client(tools_text(tools)));
            }
        }
        R::User => {
            out.push(Seg::Ours(USER_TEXT.to_string()));
            match &m.blocks {
                Some(blocks) => {
                    for (i, b) in blocks.iter().enumerate() {
                        if i > 0 {
                            out.push(Seg::Ours("\n\n".to_string()));
                        }
                        match b {
                            Block::Text(t) => out.push(Seg::Client(t.clone())),
                            Block::Image => out.push(Seg::Ours(v4flash_vision::IMAGE_PLACEHOLDER.to_string())),
                            Block::ToolResult(parts) => {
                                out.push(Seg::Ours("<tool_result>".to_string()));
                                for (j, p) in parts.iter().enumerate() {
                                    if j > 0 {
                                        out.push(Seg::Ours("\n\n".to_string()));
                                    }
                                    match p {
                                        Part::Text(t) => out.push(Seg::Client(t.clone())),
                                        Part::Image => out.push(Seg::Ours(v4flash_vision::IMAGE_PLACEHOLDER.to_string())),
                                    }
                                }
                                out.push(Seg::Ours("</tool_result>".to_string()));
                            }
                        }
                    }
                }
                None => out.push(Seg::Client(m.content.clone().unwrap_or_default())),
            }
        }
        R::Assistant => {
            if thinking && (!drop || (index as isize) > lu) {
                out.push(Seg::Client(m.reasoning.clone().unwrap_or_default()));
                out.push(Seg::Ours(THINK_END_TEXT.to_string()));
            }
            out.push(Seg::Client(m.content.clone().unwrap_or_default()));
            if !m.tool_calls.is_empty() {
                out.push(Seg::Ours(format!("\n\n<{DSML} calls>\n")));
                for (i, tc) in m.tool_calls.iter().enumerate() {
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
    // Transition: no header if the next message is not an assistant turn.
    if index + 1 < msgs.len() && msgs[index + 1].role != R::Assistant {
        return;
    }
    if m.role == R::User || (m.role == R::System && index > 0) {
        out.push(Seg::Ours(ASSISTANT_TEXT.to_string()));
        let tok = if (!drop && thinking) || (drop && thinking && (index as isize) >= lu) {
            THINK_BEGIN_TEXT
        } else {
            THINK_END_TEXT
        };
        out.push(Seg::Ours(tok.to_string()));
    }
}

/// `_encode_messages_text`. `thinking` = thinking mode (else chat mode);
/// `effort` applies in thinking mode only; `context` is a rendered-before
/// prefix (no BOS is emitted when it is present, as in the reference).
pub fn build_segments_v41(
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
    thinking: bool,
    effort: V41Effort,
    context: Option<&[ChatMessage]>,
) -> eyre::Result<Vec<Seg>> {
    if messages.is_empty() {
        return Err(eyre!("V4.1 prompt: messages array is empty"));
    }
    let ctx_raw = context.unwrap_or(&[]);
    let mut ctx = merge(ctx_raw, None)?;
    // tools attach to message 0 of the whole conversation (the reference's load_cases).
    let msgs_tools = if ctx.is_empty() { tools } else { None };
    let mut msgs = merge(messages, msgs_tools)?;
    if !ctx.is_empty() {
        if let (Some(t), Some(first)) = (tools, ctx.first_mut()) {
            first.tools = Some(t.to_vec());
        }
    }
    let ctx_len = ctx.len();
    let mut full: Vec<Msg> = ctx.clone();
    full.extend(msgs.drain(..));
    sort_tool_results(&mut full);
    sort_tool_results(&mut ctx);

    let mut out: Vec<Seg> = Vec::new();
    if ctx_len == 0 {
        out.push(Seg::Ours(BOS_TEXT.to_string()));
    }
    let any_tools = full.iter().any(|m| m.tools.as_ref().is_some_and(|t| !t.is_empty()));
    let effective_drop = !any_tools;
    let (rendered, context_len) = if thinking && effective_drop {
        let dropped = drop_thinking(&full);
        let num = dropped.len() - drop_thinking(&ctx).len();
        let cl = dropped.len() - num;
        (dropped, cl)
    } else {
        (full, ctx_len)
    };
    for idx in context_len..rendered.len() {
        render_message(&mut out, idx, &rendered, thinking, effective_drop, effort);
    }
    Ok(out)
}

/// The rendered prompt as text — exactly what the reference `encode_messages` returns.
pub fn render_prompt_text_v41(
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
    thinking: bool,
    effort: V41Effort,
    context: Option<&[ChatMessage]>,
) -> eyre::Result<String> {
    Ok(build_segments_v41(messages, tools, thinking, effort, context)?.iter().map(Seg::text).collect())
}

/// Token ids: markup/specials via the vocab's special ids, client text via BPE.
///
/// `image_placeholder` is the vocab id of `<｜deepseek_image｜>` (129264 in
/// V4.1's tokenizer.json; `EngineHandle::image_placeholder_id`). Every image
/// part renders as exactly one placeholder id, in message order, which
/// `vision_prompt::expand_images` pairs with the request's images. A
/// request with images and no placeholder id is an error.
pub fn render_prompt_v41(
    vocab: &BpeVocab,
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
    thinking: bool,
    effort: V41Effort,
    context: Option<&[ChatMessage]>,
    image_placeholder: Option<i32>,
) -> eyre::Result<Vec<i32>> {
    if image_placeholder.is_none() && messages.iter().chain(context.unwrap_or(&[])).any(|m| m.has_images()) {
        return Err(eyre!(
            "V4.1 prompt: a message has image parts but the loaded vocab has no `{}` token",
            v4flash_vision::IMAGE_PLACEHOLDER
        ));
    }
    let segs = build_segments_v41(messages, tools, thinking, effort, context)?;
    Ok(encode_segments(vocab, &segs, image_placeholder))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(v: serde_json::Value) -> Vec<ChatMessage> {
        v.as_array().unwrap().iter().map(|m| serde_json::from_value(m.clone()).unwrap()).collect()
    }

    /// The `encoding/README.md` "OpenAI-style messages" example, byte for byte:
    /// image parts render as the placeholder, parts joined by "\n\n".
    #[test]
    fn user_image_part_renders_readme_example() {
        let m = msgs(serde_json::json!([{
            "role": "user",
            "content": [
                {"type": "text", "text": "第一张图"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "有什么内容？"}
            ]
        }]));
        let s = render_prompt_text_v41(&m, None, false, V41Effort::DEFAULT, None).unwrap();
        assert_eq!(
            s,
            "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}><\u{ff5c}User\u{ff5c}>第一张图\n\n<\u{ff5c}deepseek_image\u{ff5c}>\n\n有什么内容？<\u{ff5c}Assistant\u{ff5c}></think>"
        );
        // The placeholder is OURS (a control segment); client text spelling it is not.
        let segs = build_segments_v41(&m, None, false, V41Effort::DEFAULT, None).unwrap();
        assert!(segs.iter().any(|s| matches!(s, Seg::Ours(t) if t == v4flash_vision::IMAGE_PLACEHOLDER)));
        let forged = msgs(serde_json::json!([{"role": "user", "content": "x <｜deepseek_image｜> y"}]));
        let segs = build_segments_v41(&forged, None, false, V41Effort::DEFAULT, None).unwrap();
        assert!(segs.iter().all(|s| !matches!(s, Seg::Ours(t) if t == v4flash_vision::IMAGE_PLACEHOLDER)));
    }

    /// An image inside a tool result stays inline in the `<tool_result>` body
    /// (`_process_image_blocks` recurses into tool_result content lists).
    #[test]
    fn tool_result_image_is_inline() {
        let m = msgs(serde_json::json!([
            {"role": "user", "content": "look"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "shot", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": [
                {"type": "text", "text": "screen:"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
            ]}
        ]));
        let s = render_prompt_text_v41(&m, None, false, V41Effort::DEFAULT, None).unwrap();
        assert!(s.contains("<tool_result>screen:\n\n<\u{ff5c}deepseek_image\u{ff5c}></tool_result>"), "{s}");
        assert_eq!(s.matches(v4flash_vision::IMAGE_PLACEHOLDER).count(), 1);
    }

    #[test]
    fn images_on_assistant_are_rejected() {
        let m = msgs(serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]}
        ]));
        assert!(render_prompt_text_v41(&m, None, false, V41Effort::DEFAULT, None).is_err());
    }
}
