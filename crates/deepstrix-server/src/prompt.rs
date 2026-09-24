//! Prompt building blocks shared by the chat template: [`Seg`] (text we
//! authored vs text the client supplied, so a client can never forge a
//! special token), [`encode_segments`], the special-token literals, and
//! [`ReasoningEffort`]. The V4.1 chat template itself is `prompt_v41.rs`
//! (the V4-Flash template was removed with V4-Flash, 2026-09-24).

use v4flash_core::tokenizer::BpeVocab;

use crate::tokens::{TOK_ASSISTANT, TOK_BOS, TOK_EOS, TOK_THINK_BEGIN, TOK_THINK_END, TOK_USER};



/// Reasoning effort for a request, per the V4-Flash 0731 spec
/// (`REASONING_EFFORT_PROMPTS` in `encoding_dsv4.py`) plus an explicit
/// Off state for thinking disabled entirely.
///
///   * `Off`  — no `<think>` phase (assistant turn opens with `</think>`).
///   * `Low`  — thinking on, no preamble. 0731's default level; matches
///     the server's historical think-mode default exactly.
///   * `High` — thinking on + [`REASONING_HIGH_PREFIX`] after BOS.
///   * `Max`  — thinking on + [`REASONING_MAX_PREFIX`] after BOS.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningEffort {
    Off,
    Low,
    High,
    Max,
}

impl ReasoningEffort {
    /// True when the assistant turn should open with `<think>`.
    pub fn thinking_enabled(self) -> bool {
        !matches!(self, ReasoningEffort::Off)
    }


    /// Map one request-field string to a level. Case-insensitive.
    ///
    ///   "" | "none" | "off" | "disabled" | "false" → Off
    ///   "low"                                      → Low
    ///   "medium" | "high" | "xhigh"                → High  (lenient
    ///       toward legacy OpenAI `reasoning_effort` values)
    ///   "max"                                      → Max
    ///   anything else                              → Err (caller maps this
    ///       to the invalid-parameter HTTP 400 convention)
    pub fn parse_str(s: &str) -> Result<Self, String> {
        // Only low/high/max exist in the 0731 spec; the rest are lenient
        // aliases for other clients' effort ladders (OpenAI: minimal..xhigh;
        // Hermes: minimal..ultra, and its LM Studio route statically clamps
        // max/ultra -> xhigh, so xhigh maps to Max here or Hermes users
        // could never reach our top tier).
        match s.to_ascii_lowercase().as_str() {
            "" | "none" | "off" | "disabled" | "false" => Ok(ReasoningEffort::Off),
            "minimal" | "low" => Ok(ReasoningEffort::Low),
            "medium" | "high" => Ok(ReasoningEffort::High),
            "xhigh" | "max" | "ultra" => Ok(ReasoningEffort::Max),
            other => Err(format!(
                "invalid reasoning effort {other:?}: expected one of \
                 \"none\", \"off\", \"disabled\", \"false\", \"minimal\", \
                 \"low\", \"medium\", \"high\", \"xhigh\", \"max\", \"ultra\""
            )),
        }
    }

    /// Resolve the effort from the request's `reasoning` (letta / pi-ai)
    /// and `reasoning_effort` (OpenAI) fields. If both are set,
    /// `reasoning_effort` wins. Both absent/null → `Low` (thinking on,
    /// no preamble — the server's historical default behavior).
    pub fn from_request_fields(
        reasoning: Option<&str>,
        reasoning_effort: Option<&str>,
    ) -> Result<Self, String> {
        Self::from_request_fields_with_default(reasoning, reasoning_effort, DEFAULT_EFFORT)
    }

    /// As [`Self::from_request_fields`], but the operator picks what an
    /// absent field means (`--default-reasoning-effort`).
    ///
    /// NOTE: raising this changes the RENDERED PROMPT for every request
    /// that omits the field (High/Max prepend a preamble to the system
    /// block), so every cached KV prefix built under the old default stops
    /// matching and has to be re-prefilled. That is why the compiled-in
    /// default stays `Low` and this is an explicit operator opt-in.
    pub fn from_request_fields_with_default(
        reasoning: Option<&str>,
        reasoning_effort: Option<&str>,
        default: ReasoningEffort,
    ) -> Result<Self, String> {
        match reasoning_effort.or(reasoning) {
            None => Ok(default),
            Some(s) => Self::parse_str(s),
        }
    }
}

/// What an absent `reasoning` / `reasoning_effort` field means. Do NOT
/// change this constant — see `from_request_fields_with_default`; use
/// `--default-reasoning-effort` instead.
pub const DEFAULT_EFFORT: ReasoningEffort = ReasoningEffort::Low;

/// The literal bytes of the `｜DSML｜` special token. The V4-Flash BPE
/// does NOT auto-merge these bytes to the single special-token id at
/// `vocab.encode()` time — it splits the chunk into 4 regular tokens
/// (`｜DS ML｜`). To emit the real special token (`vocab.dsml_id`,
/// typically 128825) we have to scan our rendered DSML markup for this
/// literal and substitute the token-id push in place. Anything around
/// the marker still goes through `vocab.encode`. See [`encode_segments`],
/// which does that scan for every [`Seg::Ours`] segment.
const DSML_MARKER: &str = "\u{ff5c}DSML\u{ff5c}";

// ---------------------------------------------------------------------------
// Segments
// ---------------------------------------------------------------------------

/// One piece of the rendered prompt, tagged with who wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Seg {
    /// Text this renderer authored (role markers, the tools block's fixed
    /// prose, DSML markup). Special-token literals inside it are emitted as
    /// their real token ids.
    Ours(String),
    /// Text supplied by the client (message content, tool schemas, tool-call
    /// arguments, tool output). BPE'd as ordinary text — a literal
    /// `<｜User｜>` or `｜DSML｜` in here must never become a control token.
    /// (The reference tokenizers do not draw this line and are forgeable.)
    Client(String),
}

impl Seg {
    pub fn text(&self) -> &str {
        match self {
            Seg::Ours(s) | Seg::Client(s) => s,
        }
    }
}

/// The literals `ds4_tokenize_rendered_chat` (`ds4.c:15874`) splits a rendered
/// prompt on, plus the Vision-Exp image placeholder. Ids are resolved against
/// the loaded vocab, falling back to the hardcoded V4-Flash ids.
fn special_literals(vocab: &BpeVocab, image_placeholder: Option<i32>) -> Vec<(&'static str, i32)> {
    let by_name = |name: &'static str, fallback: i32| {
        (name, vocab.lookup_token_id(name).unwrap_or(fallback))
    };
    let mut v = vec![
        by_name(BOS_TEXT, TOK_BOS),
        by_name(EOS_TEXT, TOK_EOS),
        by_name(USER_TEXT, TOK_USER),
        by_name(ASSISTANT_TEXT, TOK_ASSISTANT),
        by_name(THINK_BEGIN_TEXT, TOK_THINK_BEGIN),
        by_name(THINK_END_TEXT, TOK_THINK_END),
    ];
    // V4.1 (prompt_v41.rs) emits these; V4-Flash's template never does, and a
    // literal that is not in the vocab falls back to plain BPE text as before.
    for name in [crate::prompt_v41::SYSTEM_TEXT, "<\u{ff5c}latest_reminder\u{ff5c}>"] {
        if let Some(id) = vocab.lookup_token_id(name) {
            v.push((name, id));
        }
    }
    if let Some(id) = vocab.dsml_id {
        v.push((DSML_MARKER, id));
    }
    if let Some(id) = image_placeholder {
        v.push((v4flash_vision::IMAGE_PLACEHOLDER, id));
    }
    v
}

pub const BOS_TEXT: &str = "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>";
pub const EOS_TEXT: &str = "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>";
pub const USER_TEXT: &str = "<\u{ff5c}User\u{ff5c}>";
pub const ASSISTANT_TEXT: &str = "<\u{ff5c}Assistant\u{ff5c}>";
pub const THINK_BEGIN_TEXT: &str = "<think>";
pub const THINK_END_TEXT: &str = "</think>";

/// Tokenize a segment list the way the reference tokenizes a rendered prompt:
/// one contiguous BPE pass over everything, broken only where a special-token
/// literal appears — and, unlike the reference, only when that literal sits in
/// text we authored. Text accumulates ACROSS segment boundaries, so the
/// segmentation itself never shifts a token boundary.
pub(crate) fn encode_segments(vocab: &BpeVocab, segs: &[Seg], image_placeholder: Option<i32>) -> Vec<i32> {
    let specials = special_literals(vocab, image_placeholder);
    let mut out: Vec<i32> = Vec::new();
    let mut buf = String::new();
    let flush = |buf: &mut String, out: &mut Vec<i32>| {
        if !buf.is_empty() {
            out.extend(vocab.encode(buf));
            buf.clear();
        }
    };
    for seg in segs {
        match seg {
            Seg::Client(t) => buf.push_str(t),
            Seg::Ours(t) => {
                let mut rest = t.as_str();
                while !rest.is_empty() {
                    // Earliest literal wins; on a tie the longest does.
                    let hit = specials
                        .iter()
                        .filter_map(|(lit, id)| rest.find(lit).map(|at| (at, lit.len(), *id)))
                        .min_by_key(|(at, len, _)| (*at, std::cmp::Reverse(*len)));
                    let Some((at, len, id)) = hit else { break };
                    buf.push_str(&rest[..at]);
                    flush(&mut buf, &mut out);
                    out.push(id);
                    rest = &rest[at + len..];
                }
                buf.push_str(rest);
            }
        }
    }
    flush(&mut buf, &mut out);
    out
}









