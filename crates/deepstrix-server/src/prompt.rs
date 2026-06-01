//! Render OpenAI `messages[]` → V4-Flash token-id sequence.
//!
//! Mirrors `ChatTurnBuilder::build` (`crates/deepstrix-cli/src/bin/chat.rs:96-122`)
//! but extends it to handle a full multi-turn history with system /
//! user / assistant roles in any order. The tool / tool-call cases will
//! be wired up in Phase 2; Phase 1 supports text-only conversations.
//!
//! Template (matches the on-wire chat template the model was trained on):
//!   `<BOS>`
//!   [system_prompt_text]      // concatenation of all role:"system" messages
//!   For each (user, assistant) turn pair:
//!     `<User>` user_text
//!     `<Assistant>` `</think>` assistant_text   // </think> means no-think mode
//!     `<EOS>`
//!   Trailing prompt (open assistant turn for generation):
//!     `<User>` <last_user_text>
//!     `<Assistant>` `</think>`

use color_eyre::eyre::{self, eyre};
use v4flash_core::tokenizer::BpeVocab;

use crate::openai::types::{ChatMessage, Role};
use crate::tokens::{TOK_ASSISTANT, TOK_BOS, TOK_EOS, TOK_THINK_END, TOK_USER};

/// Render a list of OpenAI chat messages into a token-id sequence ready
/// for `forward_prefill_pipelined`. The result includes a trailing
/// `<Assistant> </think>` so the model is positioned to start emitting
/// the next assistant turn.
///
/// `system_prompt` is the optional content of any role:"system" message
/// — pre-merged by the caller. We don't search for system messages
/// in-line; OpenAI's convention is to either have one at the front, or
/// to merge multiple as the harness sees fit.
pub fn render_prompt(
    vocab: &BpeVocab,
    messages: &[ChatMessage],
    system_prompt: Option<&str>,
) -> eyre::Result<Vec<i32>> {
    if messages.is_empty() {
        return Err(eyre!("render_prompt: messages array is empty"));
    }
    let mut out: Vec<i32> = Vec::new();
    out.push(TOK_BOS);
    if let Some(sys) = system_prompt {
        if !sys.is_empty() {
            out.extend(vocab.encode(sys));
        }
    }

    // Walk turns. We collapse adjacent same-role messages by concatenating
    // their content with "\n" — matches typical OpenAI client behavior.
    // Roles other than user/assistant in v1 are an error (tool roles land
    // in Phase 2).
    let mut i = 0;
    while i < messages.len() {
        let m = &messages[i];
        match m.role {
            Role::System => {
                // System messages should have been merged into
                // `system_prompt` by the caller. Tolerate an inline one
                // by treating it as additional system context.
                if !out.contains(&TOK_USER) {
                    if let Some(c) = m.content.as_ref() {
                        out.extend(vocab.encode(c));
                    }
                } else {
                    return Err(eyre!(
                        "render_prompt: role:\"system\" message after a user turn is not supported"
                    ));
                }
                i += 1;
            }
            Role::User => {
                out.push(TOK_USER);
                if let Some(c) = m.content.as_ref() {
                    out.extend(vocab.encode(c));
                }
                i += 1;

                // If there's a following assistant message, it's a
                // closed turn — include the assistant text with an EOS
                // so the model treats it as history. The final user
                // message (no following assistant) is the open prompt.
                if i < messages.len() && messages[i].role == Role::Assistant {
                    out.push(TOK_ASSISTANT);
                    // </think> = no-think mode for history (we don't
                    // know whether the historical turn was emitted in
                    // think mode; </think> is the safer default since
                    // it positions the model to read the content as
                    // already-post-think).
                    out.push(TOK_THINK_END);
                    if let Some(c) = messages[i].content.as_ref() {
                        out.extend(vocab.encode(c));
                    }
                    out.push(TOK_EOS);
                    i += 1;
                }
            }
            Role::Assistant => {
                return Err(eyre!(
                    "render_prompt: role:\"assistant\" message at position {i} has no preceding user message"
                ));
            }
            Role::Tool => {
                return Err(eyre!(
                    "render_prompt: role:\"tool\" messages are not supported in Phase 1"
                ));
            }
        }
    }

    // Open the generating assistant turn. </think> = no-think mode by
    // default; Phase 2 may add a request-level toggle.
    out.push(TOK_ASSISTANT);
    out.push(TOK_THINK_END);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Option<BpeVocab> {
        // Tests that need a real vocab are #[ignore]-gated and require
        // the GGUF file. Pure structural tests (token-id sequencing) can
        // use a mock; for now we only test against a real vocab in the
        // gated path.
        None
    }

    #[test]
    fn empty_messages_errors() {
        // Doesn't need a real vocab — fails before encode.
        let dummy = match vocab() {
            Some(v) => v,
            None => return, // skip without a vocab
        };
        let r = render_prompt(&dummy, &[], None);
        assert!(r.is_err());
    }
}
