//! Render OpenAI `messages[]` → V4-Flash token-id sequence.
//!
//! Mirrors `external/ds4/ds4_server.c:render_chat_prompt_text` (1901-1978)
//! — the canonical V4-Flash chat template used in production. Supports:
//!   * system / user / assistant / tool roles
//!   * tool definitions (rendered into the system prompt via DSML schema block)
//!   * assistant history turns that contained tool calls (re-rendered as DSML)
//!   * tool-result messages wrapped as `<tool_result>…</tool_result>` text
//!
//! Template structure (after rendering):
//!   <BOS>
//!   [merged system prompt text]
//!   [tool schemas block if tools provided]
//!   For each turn in history:
//!     For each user/tool-result message in a contiguous run:
//!       <User> [content or <tool_result>...</tool_result>]
//!     <Assistant> </think> [content][optional DSML tool_calls] <EOS>
//!   Trailing open turn (after last user-like message):
//!     <Assistant> </think>      (no-think mode default)

use color_eyre::eyre::{self, eyre};
use v4flash_core::tokenizer::BpeVocab;

use crate::dsml::{render_tool_calls_in_history, render_tools_prompt};
use crate::openai::types::{ChatMessage, Role, ToolDef};
use crate::tokens::{TOK_ASSISTANT, TOK_BOS, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END, TOK_USER};

/// The literal bytes of the `｜DSML｜` special token. The V4-Flash BPE
/// does NOT auto-merge these bytes to the single special-token id at
/// `vocab.encode()` time — it splits the chunk into 4 regular tokens
/// (`｜DS ML｜`). To emit the real special token (`vocab.dsml_id`,
/// typically 128825) we have to scan our rendered DSML markup for this
/// literal and substitute the token-id push in place. Anything around
/// the marker still goes through `vocab.encode`. See
/// `encode_with_special_marker`.
const DSML_MARKER: &str = "\u{ff5c}DSML\u{ff5c}";

/// Tokenize a string, but emit `marker_id` whenever the literal `marker`
/// appears in the input — splitting around it and BPE-encoding the
/// surrounding segments. Used to materialize special-token IDs that BPE
/// would otherwise split into regular tokens.
fn encode_with_special_marker(
    vocab: &BpeVocab,
    text: &str,
    marker: &str,
    marker_id: i32,
) -> Vec<i32> {
    let mut out = Vec::new();
    let mut remaining = text;
    while let Some(pos) = remaining.find(marker) {
        if pos > 0 {
            out.extend(vocab.encode(&remaining[..pos]));
        }
        out.push(marker_id);
        remaining = &remaining[pos + marker.len()..];
    }
    if !remaining.is_empty() {
        out.extend(vocab.encode(remaining));
    }
    out
}

/// Tokenize text into the prompt, substituting the DSML marker for its
/// special-token id if the vocab has one. Falls back to plain
/// `vocab.encode` when no DSML id is known — the model is robust enough
/// to recognize the textual form, so this is a safe degradation.
fn encode_text(vocab: &BpeVocab, text: &str) -> Vec<i32> {
    match vocab.dsml_id {
        Some(id) => encode_with_special_marker(vocab, text, DSML_MARKER, id),
        None => vocab.encode(text),
    }
}

pub fn render_prompt(
    vocab: &BpeVocab,
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
    think_mode: bool,
) -> eyre::Result<Vec<i32>> {
    if messages.is_empty() {
        return Err(eyre!("render_prompt: messages array is empty"));
    }

    // System block. We encode the user-provided portion (role:"system"
    // messages) and our generated tools-schema portion separately —
    // user text goes through plain `vocab.encode` to avoid letting a
    // user inject special-token text (`｜DSML｜`) into the prompt; our
    // own schema goes through `encode_text` which materialises the
    // DSML special-token id.
    let mut user_system_text = String::new();
    for m in messages {
        if matches!(m.role, Role::System) {
            if let Some(c) = m.content.as_ref() {
                if !c.is_empty() {
                    if !user_system_text.is_empty() {
                        user_system_text.push_str("\n\n");
                    }
                    user_system_text.push_str(c);
                }
            }
        }
    }
    let tools_block = tools
        .filter(|t| !t.is_empty())
        .map(|t| render_tools_prompt(t))
        .unwrap_or_default();

    let mut out: Vec<i32> = Vec::new();
    out.push(TOK_BOS);
    if !user_system_text.is_empty() {
        out.extend(vocab.encode(&user_system_text));
    }
    if !tools_block.is_empty() {
        if !user_system_text.is_empty() {
            // Visual separation — encoded as plain text, not special.
            out.extend(vocab.encode("\n\n"));
        }
        out.extend(encode_text(vocab, &tools_block));
    }

    // Walk turns, tracking lazily-opened <User> and pending <Assistant>.
    let mut pending_assistant = false;
    let mut user_tool_block_open = false; // we've already emitted <User> for a
                                          // contiguous run of tool-result messages

    for m in messages {
        match m.role {
            Role::System => continue,
            Role::User => {
                out.push(TOK_USER);
                if let Some(c) = m.content.as_ref() {
                    if !c.is_empty() {
                        // Plain encode — don't let user content forge special tokens.
                        out.extend(vocab.encode(c));
                    }
                }
                pending_assistant = true;
                user_tool_block_open = false;
            }
            Role::Tool => {
                if !user_tool_block_open {
                    out.push(TOK_USER);
                    user_tool_block_open = true;
                }
                let body = build_tool_result_text(m);
                // Tool-result bodies come from external command output —
                // treat as untrusted, no special-token substitution.
                out.extend(vocab.encode(&body));
                pending_assistant = true;
            }
            Role::Assistant => {
                if !pending_assistant {
                    return Err(eyre!(
                        "render_prompt: assistant message with no prior user/tool turn"
                    ));
                }
                out.push(TOK_ASSISTANT);
                out.push(TOK_THINK_END);
                if let Some(c) = m.content.as_ref() {
                    if !c.is_empty() {
                        // If a DSML tool_calls block follows, strip the
                        // content's trailing whitespace. The model's
                        // natural output is `<content text>\n\n<｜DSML｜...>`
                        // (often as a single token like ".\n\n"), and
                        // `render_tool_calls_in_history` ALSO prefixes
                        // "\n\n" before the DSML block — so without the
                        // strip we'd emit four newlines vs. the live
                        // cache's two and break byte-aligned LCP.
                        let text = if !m.tool_calls.is_empty() {
                            c.trim_end_matches(['\n', '\r', '\t', ' '])
                        } else {
                            c.as_str()
                        };
                        if !text.is_empty() {
                            out.extend(vocab.encode(text));
                        }
                    }
                }
                if !m.tool_calls.is_empty() {
                    let dsml = render_tool_calls_in_history(&m.tool_calls);
                    out.extend(encode_text(vocab, &dsml));
                }
                out.push(TOK_EOS);
                pending_assistant = false;
                user_tool_block_open = false;
            }
        }
    }

    // Final open assistant turn. think_mode=true opens with `<think>`
    // so the model emits reasoning until it samples `</think>`
    // (TOK_THINK_END) as a proper special token; the SSE handler
    // routes reasoning tokens to `delta.reasoning_content` until then.
    if pending_assistant {
        out.push(TOK_ASSISTANT);
        out.push(if think_mode {
            TOK_THINK_BEGIN
        } else {
            TOK_THINK_END
        });
    }
    Ok(out)
}

fn build_tool_result_text(m: &ChatMessage) -> String {
    // ds4_server.c:1942-1945 — `<tool_result>` + DSML-text-escaped content
    // + `</tool_result>`. No special token for `tool_result`; the model
    // sees it as regular text.
    let mut s = String::new();
    s.push_str("<tool_result>");
    if let Some(content) = m.content.as_ref() {
        // DSML text escape (`<`, `>`, `&` → entities).
        for ch in content.chars() {
            match ch {
                '&' => s.push_str("&amp;"),
                '<' => s.push_str("&lt;"),
                '>' => s.push_str("&gt;"),
                other => s.push(other),
            }
        }
    }
    s.push_str("</tool_result>");
    s
}
