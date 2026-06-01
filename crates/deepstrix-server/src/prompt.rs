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
use crate::tokens::{TOK_ASSISTANT, TOK_BOS, TOK_EOS, TOK_THINK_END, TOK_USER};

pub fn render_prompt(
    vocab: &BpeVocab,
    messages: &[ChatMessage],
    tools: Option<&[ToolDef]>,
) -> eyre::Result<Vec<i32>> {
    if messages.is_empty() {
        return Err(eyre!("render_prompt: messages array is empty"));
    }

    // System block: concatenate all role:"system" messages (anywhere in
    // the array — ds4 collects them by walking the entire list, not just
    // the prefix). Then append the tools schema if any.
    let mut system_text = String::new();
    for m in messages {
        if matches!(m.role, Role::System) {
            if let Some(c) = m.content.as_ref() {
                if !c.is_empty() {
                    if !system_text.is_empty() {
                        system_text.push_str("\n\n");
                    }
                    system_text.push_str(c);
                }
            }
        }
    }
    if let Some(t) = tools {
        if !t.is_empty() {
            let block = render_tools_prompt(t);
            if !block.is_empty() {
                if !system_text.is_empty() {
                    system_text.push_str("\n\n");
                }
                system_text.push_str(&block);
            }
        }
    }

    let mut out: Vec<i32> = Vec::new();
    out.push(TOK_BOS);
    if !system_text.is_empty() {
        out.extend(vocab.encode(&system_text));
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
                        out.extend(vocab.encode(c));
                    }
                }
                if !m.tool_calls.is_empty() {
                    let dsml = render_tool_calls_in_history(&m.tool_calls);
                    out.extend(vocab.encode(&dsml));
                }
                out.push(TOK_EOS);
                pending_assistant = false;
                user_tool_block_open = false;
            }
        }
    }

    // Final open assistant turn.
    if pending_assistant {
        out.push(TOK_ASSISTANT);
        out.push(TOK_THINK_END);
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
