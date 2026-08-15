//! Token-driven DSML scanner.
//!
//! V4-Flash encodes tool calls as **DSML text** anchored on a single
//! special token, `｜DSML｜` (id loaded from GGUF at vocab init —
//! `vocab.dsml_id`, typically 128825). The surrounding characters
//! (`<`, `>`, `tool_calls`, `invoke`, `parameter`, attribute names,
//! parameter values) are regular BPE-encoded text.
//!
//! The scanner is driven by **TOK_DSML transitions**, not by byte
//! patterns. That has two important properties:
//!
//! 1. **No false positives on user content.** A user message that
//!    literally contains the characters `｜DSML｜` produces regular
//!    BPE tokens (`28217, 10525, 7398, 28217`) — not TOK_DSML — and
//!    the scanner ignores them.
//!
//! 2. **TOK_DSML bytes are never emitted as content.** When the
//!    worker decodes TOK_DSML it produces 10 UTF-8 bytes that look
//!    like `｜DSML｜`. If those bytes went out as `delta.content`,
//!    letta would store them, the next request would re-encode them
//!    as regular BPE tokens, and the model would see literal
//!    `｜DSML｜` in its input and learn to mimic it back. We close
//!    that cycle by suppressing TOK_DSML's bytes unconditionally —
//!    inside markup (structural) or outside (stray — warn + drop).

use crate::openai::types::{ToolCall, ToolDef};

// ---------------------------------------------------------------------------
// Renderer — unchanged from the byte-scanner version. Used to build
// the system-prompt schema block and to re-render assistant turns
// that had tool_calls in history.
// ---------------------------------------------------------------------------

pub fn render_tools_prompt(tools: &[ToolDef]) -> String {
    if tools.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str(
        "## Tools\n\n\
         You have access to a set of tools to help answer the user question. \
         You can invoke tools by writing a \"<\u{ff5c}DSML\u{ff5c}tool_calls>\" block like the following:\n\n\
         <\u{ff5c}DSML\u{ff5c}tool_calls>\n\
         <\u{ff5c}DSML\u{ff5c}invoke name=\"$TOOL_NAME\">\n\
         <\u{ff5c}DSML\u{ff5c}parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</\u{ff5c}DSML\u{ff5c}parameter>\n\
         ...\n\
         </\u{ff5c}DSML\u{ff5c}invoke>\n\
         <\u{ff5c}DSML\u{ff5c}invoke name=\"$TOOL_NAME2\">\n\
         ...\n\
         </\u{ff5c}DSML\u{ff5c}invoke>\n\
         </\u{ff5c}DSML\u{ff5c}tool_calls>\n\n\
         String parameters should be specified as raw text and set `string=\"true\"`. \
         Preserve characters such as `>`, `&`, and `&&` exactly; never replace normal string characters with XML or HTML entity escapes. \
         Only if a string value itself contains the exact closing parameter tag `</\u{ff5c}DSML\u{ff5c}parameter>`, write that tag as `&lt;/\u{ff5c}DSML\u{ff5c}parameter>` inside the value. \
         For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n\
         When thinking mode is enabled, finish reasoning with </think> before any tool calls or final response.\n\n\
         Otherwise, output directly after </think> with tool calls or final response.\n\n\
         ### Available Tool Schemas\n\n",
    );
    let schemas = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".into());
    out.push_str(&schemas);
    out.push_str(
        "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls. \
         Use the exact parameter names from the schemas.",
    );
    out
}

pub fn render_tool_calls_in_history(calls: &[ToolCall]) -> String {
    if calls.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    out.push_str("\n\n<\u{ff5c}DSML\u{ff5c}tool_calls>\n");
    for tc in calls {
        out.push_str("<\u{ff5c}DSML\u{ff5c}invoke name=\"");
        push_dsml_attr(&mut out, &tc.function.name);
        out.push_str("\">\n");
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(&tc.function.arguments) {
            if let serde_json::Value::Object(map) = v {
                for (key, val) in &map {
                    let is_string = matches!(val, serde_json::Value::String(_));
                    out.push_str("<\u{ff5c}DSML\u{ff5c}parameter name=\"");
                    push_dsml_attr(&mut out, key);
                    out.push_str("\" string=\"");
                    out.push_str(if is_string { "true" } else { "false" });
                    out.push_str("\">");
                    match val {
                        serde_json::Value::String(s) => push_dsml_parameter_text(&mut out, s),
                        other => {
                            let s = serde_json::to_string(other).unwrap_or_default();
                            push_dsml_json_literal(&mut out, &s);
                        }
                    }
                    out.push_str("</\u{ff5c}DSML\u{ff5c}parameter>\n");
                }
            } else {
                out.push_str("<\u{ff5c}DSML\u{ff5c}parameter name=\"arguments\" string=\"true\">");
                push_dsml_parameter_text(&mut out, &tc.function.arguments);
                out.push_str("</\u{ff5c}DSML\u{ff5c}parameter>\n");
            }
        } else {
            out.push_str("<\u{ff5c}DSML\u{ff5c}parameter name=\"arguments\" string=\"true\">");
            push_dsml_parameter_text(&mut out, &tc.function.arguments);
            out.push_str("</\u{ff5c}DSML\u{ff5c}parameter>\n");
        }
        out.push_str("</\u{ff5c}DSML\u{ff5c}invoke>\n");
    }
    out.push_str("</\u{ff5c}DSML\u{ff5c}tool_calls>");
    out
}

fn push_dsml_attr(out: &mut String, s: &str) {
    for ch in s.chars() {
        match ch {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            other => out.push(other),
        }
    }
}

fn push_dsml_parameter_text(out: &mut String, s: &str) {
    let end = "</\u{ff5c}DSML\u{ff5c}parameter>";
    let mut i = 0;
    let bytes = s.as_bytes();
    let end_bytes = end.as_bytes();
    while i < bytes.len() {
        if bytes[i..].starts_with(end_bytes) {
            out.push_str("&lt;");
            i += 1;
        } else {
            let ch_len = utf8_char_len(bytes[i]);
            if let Ok(s) = std::str::from_utf8(&bytes[i..i + ch_len]) {
                out.push_str(s);
            }
            i += ch_len;
        }
    }
}

fn push_dsml_json_literal(out: &mut String, s: &str) {
    let end = "</\u{ff5c}DSML\u{ff5c}parameter>";
    let mut i = 0;
    let bytes = s.as_bytes();
    let end_bytes = end.as_bytes();
    while i < bytes.len() {
        if bytes[i..].starts_with(end_bytes) {
            out.push_str("\\u003c");
            i += 1;
        } else {
            let ch_len = utf8_char_len(bytes[i]);
            if let Ok(s) = std::str::from_utf8(&bytes[i..i + ch_len]) {
                out.push_str(s);
            }
            i += ch_len;
        }
    }
}

fn utf8_char_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first < 0xC0 {
        1
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Scanner — token-driven.
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum DsmlEvent {
    /// Plain assistant text outside any DSML markup. Bytes are NOT
    /// UTF-8-validated — caller maintains its own cross-chunk buffer.
    Text(Vec<u8>),
    /// One completed `<invoke>` block.
    ToolCall {
        index: u32,
        id: String,
        name: String,
        arguments: String,
    },
    /// `</tool_calls>` — outer block closed.
    ToolCallsEnd,
}

/// Sentinel value: "no token, just process the leftover bytes." Used
/// when a header-completion (`>` found) leaves bytes that the new
/// mode should also see.
const NO_TOKEN: i32 = -1;

#[derive(Debug)]
enum Frame {
    /// Inside `<｜DSML｜tool_calls>…</｜DSML｜tool_calls>`.
    ToolCalls { next_invoke_index: u32 },
    /// Inside `<｜DSML｜invoke …>…</｜DSML｜invoke>`.
    Invoke {
        index: u32,
        name: String,
        params: Vec<(String, bool, String)>,
    },
}

#[derive(Debug, PartialEq, Eq)]
enum Mode {
    /// Streaming text. Outside frames → emit as `Text(bytes)`. Inside
    /// frames → drop (between tags).
    Text,
    /// Accumulating bytes between `<｜DSML｜` and the next `>`.
    OpenHeader,
    /// Accumulating bytes between `</｜DSML｜` and the next `>`.
    CloseHeader,
    /// Inside a `<｜DSML｜parameter …>…</｜DSML｜parameter>` body.
    ParameterBody,
    /// After the outer `</tool_calls>` — anything else is ignored.
    Done,
}

pub struct DsmlScanner {
    tok_dsml: i32,
    frames: Vec<Frame>,
    mode: Mode,
    /// Trailing 0–2 bytes that might be `<` or `</` — used to
    /// disambiguate tag direction when the next token is TOK_DSML.
    /// Reset to empty by every TOK_DSML transition and on byte flush.
    tail: Vec<u8>,
    /// Mode-specific accumulator (header bytes for OpenHeader /
    /// CloseHeader; parameter value bytes for ParameterBody).
    buf: Vec<u8>,
    /// Set when mode == ParameterBody or we just transitioned into
    /// CloseHeader from ParameterBody. `(name, is_string, body)` —
    /// the body slot is filled when we leave ParameterBody so
    /// dispatch_close("parameter") can finalize it.
    current_param: Option<(String, bool, Vec<u8>)>,
    /// Sticky flag: true once any unknown-tag fallback occurred
    /// (e.g. the model emitted `<｜DSML｜command name="ls"…>` instead
    /// of `<｜DSML｜parameter name="command"…>ls`) or an orphan
    /// closing tag arrived with no matching open frame. The handler
    /// checks this at end-of-turn and reports
    /// `finish_reason: "error"` so letta knows the turn is broken
    /// rather than treating the corrupted markup as content.
    ///
    /// NOTE: since the DSML-repair port (upstream ds4 0ffaabd/596f49c/
    /// 7bcc4e8), plain truncation (EOS mid-DSML) no longer sets this
    /// when the open frames can be repaired into valid tool calls —
    /// see `finish()`.
    malformed: bool,
    /// Count of ToolCall events emitted this turn (streaming closes +
    /// repair). Used by `finish()` to decide between "repaired tool
    /// call" and "hallucinated / unrecoverable" outcomes.
    tool_calls_emitted: u32,
    /// True once a closing tag arrived with no matching open frame
    /// (closes outnumber opens). Not a truncation pattern — when set,
    /// `finish()` refuses to repair (upstream 7bcc4e8 guard b; there
    /// the unsigned open-minus-close subtraction would underflow, here
    /// the stack makes underflow impossible but the refusal semantics
    /// are kept).
    saw_orphan_close: bool,
    /// Text captured while inside a bare `<tool_calls>` block that has
    /// not opened any `<invoke>` yet. Normally inter-tag whitespace
    /// (discarded once an invoke opens); but when the model
    /// hallucinates `<tool_calls>…plain prose…</tool_calls>` (or
    /// truncates before any invoke), upstream ds4 (596f49c/20df6c7)
    /// strips the tags and returns the prose as plain content — this
    /// buffer is what lets us do the same.
    halluc_buf: Vec<u8>,
}

impl DsmlScanner {
    /// True if the scanner ever encountered an unknown DSML tag and
    /// fell back to Text mode. Sticky for the lifetime of the
    /// scanner — caller queries at end-of-turn.
    pub fn saw_malformed(&self) -> bool {
        self.malformed
    }
}

impl DsmlScanner {
    pub fn new(tok_dsml: i32) -> Self {
        Self {
            tok_dsml,
            frames: Vec::new(),
            mode: Mode::Text,
            tail: Vec::new(),
            buf: Vec::new(),
            current_param: None,
            malformed: false,
            tool_calls_emitted: 0,
            saw_orphan_close: false,
            halluc_buf: Vec::new(),
        }
    }

    pub fn push_token(&mut self, tok: i32, bytes: &[u8]) -> Vec<DsmlEvent> {
        let mut events = Vec::new();
        let mut leftover = self.step(tok, bytes, &mut events);
        while !leftover.is_empty() {
            let next = self.step(NO_TOKEN, &leftover, &mut events);
            if next.len() == leftover.len() && next == leftover {
                // No progress — bail out (defensive, shouldn't happen).
                break;
            }
            leftover = next;
        }
        events
    }

    pub fn finish(&mut self) -> Vec<DsmlEvent> {
        let mut events = Vec::new();
        // End-of-stream cleanup. There are three cases:
        //
        // 1. Clean end (Mode::Text + no open frames): flush pending tail
        //    bytes as text. The normal-path case.
        //
        // 2. Mid-stream EOS inside an open DSML frame (Mode::Text +
        //    frames non-empty, or Mode in {OpenHeader, CloseHeader,
        //    ParameterBody}): the model hit length-cap or dropped the
        //    closing tags during a long generation (its attention
        //    degrades past ~2000 tokens of tool-call output). Port of
        //    upstream ds4's `try_repair_dsml` (0ffaabd, 596f49c,
        //    20df6c7, guards from 7bcc4e8): instead of failing the
        //    turn, finalize the in-flight parameter into the open
        //    invoke, emit a ToolCall for each recoverable open invoke,
        //    and close the tool_calls block. Upstream measured 100%
        //    repair success in production (0 finish=error across 156+
        //    requests).
        //
        // 3. Repair impossible (no recoverable invoke): fall back to
        //    the pre-repair behavior — flush whatever bytes we have as
        //    plain text (so letta gets a non-empty turn instead of
        //    "Empty LLM response" + retry) and mark the turn malformed
        //    so the handler reports finish_reason=internal_error.
        //
        // Think-boundary guard (upstream 7bcc4e8 guard a): upstream's
        // text-level repair had to explicitly ignore DSML tags before
        // the last `</think>` — DSML *quoted inside reasoning* would
        // otherwise inflate the tag counts and trigger false-positive
        // repairs. Our architecture enforces that guard upstream of
        // the scanner, twice over:
        //   1. The engine worker tracks `in_think` off the dedicated
        //      TOK_THINK_BEGIN / TOK_THINK_END special tokens and tags
        //      every chunk with `reasoning: in_think`
        //      (engine_worker.rs); both scanner call sites
        //      (`accumulate()` and the SSE stream loop in
        //      openai/handler.rs) `continue` on reasoning chunks
        //      BEFORE calling `push_token` — think-internal tokens
        //      never reach the scanner at all.
        //   2. The scanner is TOK_DSML-driven, not byte-driven: DSML
        //      markup *quoted as text* (in thinking or anywhere else)
        //      BPE-encodes to regular tokens, never opens a frame, and
        //      therefore can neither trigger nor distort repair. See
        //      the module docs and `think_quoted_dsml_text_is_inert`.
        //
        // Extra-closing-tags guard (upstream 7bcc4e8 guard b): upstream
        // counted opens minus closes with size_t arithmetic and had to
        // refuse repair when closes outnumbered opens (the subtraction
        // underflowed and appended a huge suffix). Our scanner tracks
        // nesting structurally with a frame stack, so a "negative"
        // count cannot exist by construction: an orphan closing tag
        // finds no matching open frame in `dispatch_close`, marks the
        // turn malformed there, and `finish()` only ever repairs
        // frames that were genuinely opened. No subtraction anywhere.
        //
        // The recovered bytes are guaranteed not to contain literal
        // `｜DSML｜` chars (the markers themselves were consumed by
        // TOK_DSML transitions and stripped from tail).
        let in_open_frame = !self.frames.is_empty();
        let mid_tag = matches!(
            self.mode,
            Mode::OpenHeader | Mode::CloseHeader | Mode::ParameterBody
        );
        if self.mode == Mode::Text && !in_open_frame {
            // Case 1: clean end.
            if !self.tail.is_empty() {
                let drained = std::mem::take(&mut self.tail);
                events.push(DsmlEvent::Text(drained));
            }
        } else if in_open_frame || mid_tag {
            self.repair_truncated(&mut events);
        }
        self.mode = Mode::Done;
        events
    }

    /// Case 2/3 of `finish()`: EOS arrived mid-DSML. Try to repair
    /// (upstream semantics: append missing closing tags in reverse
    /// nesting order parameter → invoke → tool_calls, then verify the
    /// result parses); fall back to the malformed flush when no valid
    /// invoke can be recovered.
    fn repair_truncated(&mut self, events: &mut Vec<DsmlEvent>) {
        // Snapshot the raw buffered bytes first — the fallback path
        // flushes them as text so letta gets a non-empty turn (the
        // pre-repair behavior).
        let mut recovered_snapshot: Vec<u8> = Vec::new();
        recovered_snapshot.extend_from_slice(&self.buf);
        recovered_snapshot.extend_from_slice(&self.tail);

        // Guard b (7bcc4e8): closing tags outnumbered opening tags at
        // some point in this turn. That is corruption, not truncation
        // — refuse to fabricate a repair.
        if self.saw_orphan_close {
            self.malformed = true;
            self.frames.clear();
            self.current_param = None;
            self.buf.clear();
            self.tail.clear();
            if !recovered_snapshot.is_empty() {
                events.push(DsmlEvent::Text(recovered_snapshot));
            }
            tracing::warn!(
                "DSML stream ended mid-markup after orphan closing tags; \
                 refusing repair (closes outnumbered opens)"
            );
            return;
        }

        // Step 1: resolve the mode-specific in-flight state, i.e. the
        // implicit `</parameter>`.
        match self.mode {
            Mode::ParameterBody => {
                // Truncated mid-parameter-body: buf holds the body so
                // far, tail may hold a `<` / `</` lookahead that never
                // became a tag. Both are body bytes.
                let mut body = std::mem::take(&mut self.buf);
                body.extend(std::mem::take(&mut self.tail));
                if let Some((_, _, slot)) = self.current_param.as_mut() {
                    *slot = body;
                }
                self.finalize_current_param();
            }
            Mode::CloseHeader => {
                // Truncated inside a `</｜DSML｜…` header. If we came
                // from ParameterBody the body was already stashed into
                // current_param; the partial header text in buf is
                // markup, not content — drop it.
                self.buf.clear();
                self.tail.clear();
                self.finalize_current_param();
            }
            Mode::OpenHeader => {
                // Truncated inside a `<｜DSML｜…` header. The tag never
                // completed (no `name="…"` guaranteed, no body) — drop
                // the partial header and repair the frames below it.
                tracing::warn!(
                    header = %String::from_utf8_lossy(&self.buf),
                    "DSML stream truncated mid open-tag header; dropping partial tag"
                );
                self.buf.clear();
                self.tail.clear();
            }
            Mode::Text => {
                // Truncated between tags inside an open frame. Route
                // any held-back tail like regular inter-tag text
                // (hallucination capture or drop).
                self.flush_tail_to_current_mode(events);
            }
            Mode::Done => {}
        }

        // Step 2: did any invoke ever open? Distinguishes truncation
        // (repairable) from a hallucinated `<tool_calls>` block that
        // wraps plain prose (strip tags, return prose — upstream
        // 596f49c/20df6c7 mode 3).
        let mut had_tool_calls_frame = false;
        let mut any_invoke_opened = self.tool_calls_emitted > 0;
        for f in &self.frames {
            match f {
                Frame::Invoke { .. } => any_invoke_opened = true,
                Frame::ToolCalls { next_invoke_index } => {
                    had_tool_calls_frame = true;
                    if *next_invoke_index > 0 {
                        any_invoke_opened = true;
                    }
                }
            }
        }

        // Step 3: implicit `</invoke>` for every open invoke frame
        // (reverse nesting order), then the implicit `</tool_calls>`.
        let emitted_before = self.tool_calls_emitted;
        self.close_open_invokes(events);
        self.frames.clear();

        if self.tool_calls_emitted > 0 {
            // Repair succeeded — at least one structurally valid tool
            // call this turn (recovered now or streamed earlier).
            events.push(DsmlEvent::ToolCallsEnd);
            tracing::warn!(
                recovered = self.tool_calls_emitted - emitted_before,
                total = self.tool_calls_emitted,
                "repaired unterminated DSML tool call block \
                 (appended missing closing tags)"
            );
        } else if had_tool_calls_frame && !any_invoke_opened {
            // Hallucinated tool_calls: the block never contained an
            // invoke. Strip the tags and surface the captured inner
            // text as plain content; NOT an error (upstream 20df6c7).
            let halluc = std::mem::take(&mut self.halluc_buf);
            tracing::warn!(
                bytes = halluc.len(),
                "DSML stream ended inside a tool_calls block with no \
                 invoke; treating block contents as plain text"
            );
            if !halluc.is_empty() {
                events.push(DsmlEvent::Text(halluc));
            }
        } else {
            // Unrecoverable (e.g. open invoke with an empty name, or
            // a parameter with no enclosing invoke). Pre-repair
            // behavior: flush recovered bytes as text + flag malformed
            // so the handler reports a retryable error.
            self.malformed = true;
            if !recovered_snapshot.is_empty() {
                tracing::warn!(
                    bytes = recovered_snapshot.len(),
                    "DSML stream ended mid-markup and repair found no \
                     valid invoke; flushing partial bytes as text to \
                     avoid empty response"
                );
                events.push(DsmlEvent::Text(recovered_snapshot));
            } else {
                tracing::warn!(
                    "DSML stream ended mid-markup with no buffered bytes \
                     and no recoverable invoke; reporting malformed so \
                     handler returns retryable error"
                );
            }
        }
    }

    /// Finalize the in-flight `current_param` (if any) into the
    /// innermost open invoke frame. No-op when there is no pending
    /// parameter; the parameter is dropped when no invoke frame is
    /// open to receive it.
    fn finalize_current_param(&mut self) {
        if let Some((name, is_string, body_bytes)) = self.current_param.take() {
            let body_str = if is_string {
                dsml_param_decode_string(&String::from_utf8_lossy(&body_bytes))
            } else {
                dsml_param_decode_json_literal(&String::from_utf8_lossy(&body_bytes))
            };
            if let Some(Frame::Invoke { params, .. }) = self.frames.last_mut() {
                params.push((name, is_string, body_str));
            }
        }
    }

    /// Pop every open `Frame::Invoke` off the stack (innermost first)
    /// and emit a ToolCall for each one that has a non-empty name.
    /// Used by repair paths: an explicit `</invoke>` goes through
    /// `dispatch_close` instead.
    fn close_open_invokes(&mut self, events: &mut Vec<DsmlEvent>) {
        while matches!(self.frames.last(), Some(Frame::Invoke { .. })) {
            if let Some(Frame::Invoke {
                index,
                name,
                params,
            }) = self.frames.pop()
            {
                if name.is_empty() {
                    tracing::warn!(
                        index,
                        "dropping open DSML invoke with empty name during repair"
                    );
                    continue;
                }
                let arguments = params_to_json(&params);
                events.push(DsmlEvent::ToolCall {
                    index,
                    id: format!("call_{}", uuid::Uuid::now_v7().simple()),
                    name,
                    arguments,
                });
                self.tool_calls_emitted += 1;
            }
        }
    }

    fn step(&mut self, tok: i32, bytes: &[u8], events: &mut Vec<DsmlEvent>) -> Vec<u8> {
        if self.mode == Mode::Done {
            return Vec::new();
        }

        // TOK_DSML is always a structural marker. Its bytes never
        // reach the output. The transition depends on `tail`.
        if tok == self.tok_dsml {
            return self.on_tok_dsml(bytes, events);
        }

        // Regular bytes — behaviour per mode.
        match self.mode {
            Mode::Text => self.consume_text_bytes(bytes, events),
            Mode::OpenHeader => self.consume_header_bytes(bytes, false, events),
            Mode::CloseHeader => self.consume_header_bytes(bytes, true, events),
            Mode::ParameterBody => self.consume_param_body_bytes(bytes),
            Mode::Done => Vec::new(),
        }
    }

    /// Handle a TOK_DSML token. The token's byte content is dropped
    /// regardless of context (it's purely structural — `bytes` is
    /// accepted as a parameter only to match the step() signature
    /// and is otherwise ignored).
    fn on_tok_dsml(&mut self, _bytes_ignored: &[u8], events: &mut Vec<DsmlEvent>) -> Vec<u8> {
        match self.mode {
            Mode::Text | Mode::ParameterBody => {
                // The tail tells us whether this is `<` (open) or
                // `</` (close). Trim that suffix from tail, treat any
                // remaining tail bytes appropriately, transition.
                let is_close = self.tail.ends_with(b"</");
                let is_open = !is_close && self.tail.ends_with(b"<");
                if is_close {
                    self.tail.truncate(self.tail.len() - 2);
                    self.flush_tail_to_current_mode(events);
                    // If we were accumulating a parameter body in buf,
                    // hand it off to current_param so the upcoming
                    // CloseHeader's "parameter>" tag can find it.
                    if self.mode == Mode::ParameterBody {
                        let body = std::mem::take(&mut self.buf);
                        if let Some((_, _, slot)) = self.current_param.as_mut() {
                            *slot = body;
                        }
                    }
                    self.mode = Mode::CloseHeader;
                    self.buf.clear();
                } else if is_open {
                    self.tail.truncate(self.tail.len() - 1);
                    self.flush_tail_to_current_mode(events);
                    self.mode = Mode::OpenHeader;
                    self.buf.clear();
                } else {
                    // Stray TOK_DSML — neither `<` nor `</` preceded
                    // it. Flush tail as content (text or body), warn,
                    // and silently drop the DSML marker.
                    self.flush_tail_to_current_mode(events);
                    tracing::warn!(
                        mode = ?self.mode,
                        frames = self.frames.len(),
                        "stray TOK_DSML in stream (no preceding `<` or `</`); dropping the marker bytes"
                    );
                }
            }
            Mode::OpenHeader | Mode::CloseHeader => {
                // TOK_DSML inside a tag header is malformed. The only
                // valid headers are `tool_calls>`, `invoke ...>`,
                // `parameter ...>` — none contain DSML markers
                // internally. Log and drop.
                tracing::warn!(
                    mode = ?self.mode,
                    "TOK_DSML inside tag header; dropping"
                );
            }
            Mode::Done => {}
        }
        // TOK_DSML's bytes are NEVER content; always return empty so
        // the outer leftover-loop doesn't reprocess them as text.
        Vec::new()
    }

    /// Flush the `tail` buffer as content appropriate to the current
    /// mode (text emit, body append, or drop).
    fn flush_tail_to_current_mode(&mut self, events: &mut Vec<DsmlEvent>) {
        if self.tail.is_empty() {
            return;
        }
        let drained = std::mem::take(&mut self.tail);
        match self.mode {
            Mode::Text => {
                if self.frames.is_empty() {
                    events.push(DsmlEvent::Text(drained));
                } else if self.in_bare_tool_calls() {
                    // Inside <tool_calls> before any <invoke>: keep the
                    // bytes so a hallucinated block can be surfaced as
                    // plain text (see halluc_buf).
                    self.halluc_buf.extend(drained);
                }
                // else: inside an Invoke frame (or after invokes
                // started), between tags. Drop.
            }
            Mode::ParameterBody => {
                self.buf.extend(drained);
            }
            _ => {}
        }
    }

    /// True when the innermost open frame is a `<tool_calls>` block
    /// that has not opened any `<invoke>` yet — the only position
    /// where inter-tag text might be hallucinated prose we need to
    /// keep (rather than structural whitespace).
    fn in_bare_tool_calls(&self) -> bool {
        matches!(
            self.frames.last(),
            Some(Frame::ToolCalls {
                next_invoke_index: 0
            })
        )
    }

    fn consume_text_bytes(&mut self, bytes: &[u8], events: &mut Vec<DsmlEvent>) -> Vec<u8> {
        self.tail.extend_from_slice(bytes);
        let hold = tag_lookahead(&self.tail);
        if self.tail.len() > hold {
            let emit: Vec<u8> = self.tail.drain(..self.tail.len() - hold).collect();
            if self.frames.is_empty() {
                events.push(DsmlEvent::Text(emit));
            } else if self.in_bare_tool_calls() {
                // Possible hallucinated-block prose — keep it (see
                // halluc_buf). Discarded as soon as a real invoke
                // opens.
                self.halluc_buf.extend(emit);
            }
            // else: drop — we're between tags inside a frame.
        }
        Vec::new()
    }

    fn consume_header_bytes(
        &mut self,
        bytes: &[u8],
        is_close: bool,
        events: &mut Vec<DsmlEvent>,
    ) -> Vec<u8> {
        self.buf.extend_from_slice(bytes);
        if let Some(gt_pos) = self.buf.iter().position(|&b| b == b'>') {
            let head_str = String::from_utf8_lossy(&self.buf[..gt_pos]).into_owned();
            let leftover = self.buf[gt_pos + 1..].to_vec();
            self.buf.clear();
            if is_close {
                self.dispatch_close(&head_str, events);
            } else {
                self.dispatch_open(&head_str);
            }
            return leftover;
        }
        Vec::new()
    }

    fn consume_param_body_bytes(&mut self, bytes: &[u8]) -> Vec<u8> {
        // In ParameterBody, bytes accumulate into the param value.
        // Only the trailing `</` is held back (for the next TOK_DSML
        // check). Everything else goes to `buf` (the body).
        self.tail.extend_from_slice(bytes);
        let hold = tag_lookahead(&self.tail);
        if self.tail.len() > hold {
            let absorbed: Vec<u8> = self.tail.drain(..self.tail.len() - hold).collect();
            self.buf.extend(absorbed);
        }
        Vec::new()
    }

    /// Parse an opening-tag header (already with the leading
    /// `<｜DSML｜` stripped and the trailing `>` consumed). The
    /// header is one of: `tool_calls`, `invoke …`, `parameter …`.
    fn dispatch_open(&mut self, head: &str) {
        let trimmed = head.trim_start();
        if trimmed.starts_with("tool_calls") {
            self.halluc_buf.clear();
            self.frames.push(Frame::ToolCalls {
                next_invoke_index: 0,
            });
            self.mode = Mode::Text;
            return;
        }
        if trimmed.starts_with("invoke") {
            // A real invoke: whatever text sat between <tool_calls>
            // and here was structural whitespace, not hallucinated
            // prose. Discard it.
            self.halluc_buf.clear();
            let name = parse_attr(trimmed, "name").unwrap_or_default();
            let next_invoke_index = match self.frames.last_mut() {
                Some(Frame::ToolCalls { next_invoke_index }) => {
                    let i = *next_invoke_index;
                    *next_invoke_index += 1;
                    i
                }
                _ => 0,
            };
            self.frames.push(Frame::Invoke {
                index: next_invoke_index,
                name,
                params: Vec::new(),
            });
            self.mode = Mode::Text;
            return;
        }
        if trimmed.starts_with("parameter") {
            let name = parse_attr(trimmed, "name").unwrap_or_default();
            let is_string = parse_attr(trimmed, "string")
                .map(|s| s == "true")
                .unwrap_or(true);
            self.current_param = Some((name, is_string, Vec::new()));
            self.buf.clear();
            self.mode = Mode::ParameterBody;
            return;
        }
        tracing::warn!(head, "unknown DSML open tag; falling back to Text");
        self.malformed = true;
        self.mode = Mode::Text;
    }

    fn dispatch_close(&mut self, head: &str, events: &mut Vec<DsmlEvent>) {
        let trimmed = head.trim_start();
        if trimmed.starts_with("tool_calls") {
            // Repair (upstream 596f49c mode 2 — outer tags balanced,
            // inner tags dropped): finalize any in-flight parameter
            // and close any still-open invokes before closing the
            // block, so `…<parameter …>body</tool_calls>` still yields
            // the tool call.
            self.finalize_current_param();
            self.close_open_invokes(events);
            match self.frames.pop() {
                Some(Frame::ToolCalls { next_invoke_index })
                    if next_invoke_index == 0 && self.tool_calls_emitted == 0 =>
                {
                    // Hallucinated tool_calls: a closed block that
                    // never contained an invoke. Upstream (596f49c /
                    // 20df6c7) strips the tags and treats the contents
                    // as plain text rather than an error. Keep
                    // scanning in Text mode — the model is still
                    // producing prose.
                    let halluc = std::mem::take(&mut self.halluc_buf);
                    tracing::warn!(
                        bytes = halluc.len(),
                        "DSML tool_calls block closed with no invoke; \
                         treating block contents as plain text"
                    );
                    if !halluc.is_empty() {
                        events.push(DsmlEvent::Text(halluc));
                    }
                    self.mode = Mode::Text;
                }
                Some(_) => {
                    // Real block close.
                    self.halluc_buf.clear();
                    self.frames.clear();
                    events.push(DsmlEvent::ToolCallsEnd);
                    self.mode = Mode::Done;
                }
                None => {
                    // Orphan `</tool_calls>` — closes outnumber opens.
                    // Not a truncation pattern; record it so `finish()`
                    // refuses to fabricate a repair (7bcc4e8 guard b).
                    tracing::warn!(
                        "orphan </tool_calls> with no open block; repair disabled for this turn"
                    );
                    self.saw_orphan_close = true;
                    self.mode = Mode::Text;
                }
            }
            return;
        }
        if trimmed.starts_with("invoke") {
            // Repair (upstream mode 2): a missing `</parameter>` right
            // before `</invoke>` leaves current_param pending —
            // finalize it into this invoke first.
            self.finalize_current_param();
            if matches!(self.frames.last(), Some(Frame::Invoke { .. })) {
                if let Some(Frame::Invoke {
                    index,
                    name,
                    params,
                }) = self.frames.pop()
                {
                    let arguments = params_to_json(&params);
                    events.push(DsmlEvent::ToolCall {
                        index,
                        id: format!("call_{}", uuid::Uuid::now_v7().simple()),
                        name,
                        arguments,
                    });
                    self.tool_calls_emitted += 1;
                }
            } else {
                // Orphan `</invoke>` — no matching open frame. Don't
                // pop (it would swallow an enclosing ToolCalls frame,
                // the pre-fix behavior); record it so `finish()`
                // refuses to repair (7bcc4e8 guard b).
                tracing::warn!(
                    "orphan </invoke> with no open invoke; repair disabled for this turn"
                );
                self.saw_orphan_close = true;
            }
            self.mode = Mode::Text;
            return;
        }
        if trimmed.starts_with("parameter") {
            // Finalise the current parameter (body was stashed when
            // we left ParameterBody → CloseHeader).
            self.finalize_current_param();
            self.mode = Mode::Text;
            return;
        }
        tracing::warn!(head, "unknown DSML close tag; falling back to Text");
        self.malformed = true;
        self.mode = Mode::Text;
    }
}

impl Default for DsmlScanner {
    fn default() -> Self {
        // For tests that don't have a vocab: 128825 is the V4-Flash
        // value. Callers in production pass it explicitly via `new()`.
        Self::new(128825)
    }
}

/// Largest number of trailing bytes in `b` that are a strict prefix
/// of `</` — i.e. could be the start of a closing tag waiting for
/// TOK_DSML. Returns 0, 1 (just `<`), or 2 (`</`).
fn tag_lookahead(b: &[u8]) -> usize {
    if b.ends_with(b"</") {
        2
    } else if b.ends_with(b"<") {
        1
    } else {
        0
    }
}

fn parse_attr(header: &str, attr: &str) -> Option<String> {
    let needle = format!("{attr}=\"");
    let start = header.find(&needle)? + needle.len();
    let rest = &header[start..];
    let end_rel = rest.find('"')?;
    Some(dsml_attr_decode(&rest[..end_rel]))
}

fn dsml_attr_decode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '&' {
            let mut peek = String::new();
            for _ in 0..5 {
                match it.peek() {
                    Some(&ch) => {
                        peek.push(ch);
                        it.next();
                        if ch == ';' {
                            break;
                        }
                    }
                    None => break,
                }
            }
            match peek.as_str() {
                "amp;" => out.push('&'),
                "lt;" => out.push('<'),
                "gt;" => out.push('>'),
                "quot;" => out.push('"'),
                _ => {
                    out.push('&');
                    out.push_str(&peek);
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn dsml_param_decode_string(s: &str) -> String {
    dsml_attr_decode(s)
}

fn dsml_param_decode_json_literal(s: &str) -> String {
    s.replace("\\u003c", "<")
}

fn params_to_json(params: &[(String, bool, String)]) -> String {
    let mut map = serde_json::Map::new();
    for (k, is_string, body) in params {
        if *is_string {
            map.insert(k.clone(), serde_json::Value::String(body.clone()));
        } else {
            match serde_json::from_str::<serde_json::Value>(body.trim()) {
                Ok(v) => {
                    map.insert(k.clone(), v);
                }
                Err(_) => {
                    map.insert(k.clone(), serde_json::Value::String(body.clone()));
                }
            }
        }
    }
    serde_json::Value::Object(map).to_string()
}

/// Convenience for the non-streaming path: take a token-id stream and
/// fully-decoded byte payload-per-token, run the scanner to
/// completion, and return the final event vector.
pub fn tool_calls_from_events(events: Vec<DsmlEvent>) -> Vec<ToolCall> {
    use crate::openai::types::ToolCallFunction;
    events
        .into_iter()
        .filter_map(|ev| {
            if let DsmlEvent::ToolCall {
                id,
                name,
                arguments,
                ..
            } = ev
            {
                Some(ToolCall {
                    id,
                    kind: "function".into(),
                    function: ToolCallFunction { name, arguments },
                })
            } else {
                None
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic vocab: token IDs map to ASCII bytes by convention.
    // Token 0 = TOK_DSML (with bytes `｜DSML｜`).
    // Token N (N>0) = bytes derived from the lower 7 bits as ASCII.
    // Tests construct sequences directly with bytes.

    const TOK_DSML_TEST: i32 = 999;

    /// Feed a sequence of (tok, bytes) pairs to the scanner and
    /// collect all events.
    fn drive(seq: &[(i32, &[u8])]) -> Vec<DsmlEvent> {
        drive_sc(seq).0
    }

    /// Like `drive` but also returns the scanner so tests can inspect
    /// `saw_malformed()`.
    fn drive_sc(seq: &[(i32, &[u8])]) -> (Vec<DsmlEvent>, DsmlScanner) {
        let mut sc = DsmlScanner::new(TOK_DSML_TEST);
        let mut out = Vec::new();
        for &(t, b) in seq {
            out.extend(sc.push_token(t, b));
        }
        out.extend(sc.finish());
        (out, sc)
    }

    /// Collect all Text-event bytes.
    fn all_text(ev: &[DsmlEvent]) -> Vec<u8> {
        ev.iter()
            .filter_map(|e| {
                if let DsmlEvent::Text(b) = e {
                    Some(b.clone())
                } else {
                    None
                }
            })
            .flatten()
            .collect()
    }

    /// Collect all (name, arguments) pairs from ToolCall events.
    fn all_calls(ev: &[DsmlEvent]) -> Vec<(String, String)> {
        ev.iter()
            .filter_map(|e| {
                if let DsmlEvent::ToolCall {
                    name, arguments, ..
                } = e
                {
                    Some((name.clone(), arguments.clone()))
                } else {
                    None
                }
            })
            .collect()
    }

    #[test]
    fn plain_text_pass_through() {
        let ev = drive(&[(1, b"hello world")]);
        assert_eq!(ev.len(), 1);
        match &ev[0] {
            DsmlEvent::Text(b) => assert_eq!(b.as_slice(), b"hello world"),
            other => panic!("unexpected {:?}", other),
        }
    }

    #[test]
    fn stray_tok_dsml_dropped() {
        let ev = drive(&[(1, b"hello "), (TOK_DSML_TEST, b"\xef\xbd\x9cDSML\xef\xbd\x9c"), (1, b" world")]);
        // The stray TOK_DSML's bytes must NOT appear in the output.
        let combined: Vec<u8> = ev
            .iter()
            .filter_map(|e| if let DsmlEvent::Text(b) = e { Some(b.clone()) } else { None })
            .flatten()
            .collect();
        assert_eq!(combined, b"hello  world".to_vec());
        // No tool_call event.
        assert!(!ev.iter().any(|e| matches!(e, DsmlEvent::ToolCall { .. })));
    }

    #[test]
    fn single_tool_call() {
        // "<｜DSML｜tool_calls>" then "<｜DSML｜invoke name=\"bash\">" then
        // "<｜DSML｜parameter name=\"cmd\" string=\"true\">ls</｜DSML｜parameter>"
        // then "</｜DSML｜invoke>" then "</｜DSML｜tool_calls>"
        //
        // We model TOK_DSML's bytes as the actual `｜DSML｜` UTF-8
        // (10 bytes) — the scanner must drop them regardless.
        let dsml_bytes: &[u8] = b"\xef\xbd\x9cDSML\xef\xbd\x9c";
        let ev = drive(&[
            (1, b"Sure! "),
            (1, b"<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke name=\"bash\">\n<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"parameter name=\"cmd\" string=\"true\">ls -la</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"parameter>\n</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke>\n</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"tool_calls>"),
        ]);
        // No literal `｜DSML｜` bytes anywhere in Text events.
        for e in &ev {
            if let DsmlEvent::Text(b) = e {
                assert!(
                    !contains_subseq(b, b"\xef\xbd\x9cDSML\xef\xbd\x9c"),
                    "Text event leaked literal ｜DSML｜ bytes: {:?}",
                    String::from_utf8_lossy(b)
                );
            }
        }
        // First event is "Sure! " text.
        assert!(matches!(&ev[0], DsmlEvent::Text(t) if t.as_slice() == b"Sure! "));
        // Has a ToolCall with name="bash" and args containing "ls -la".
        let tc = ev.iter().find_map(|e| {
            if let DsmlEvent::ToolCall { name, arguments, .. } = e {
                Some((name.clone(), arguments.clone()))
            } else {
                None
            }
        });
        let (name, args) = tc.expect("expected a ToolCall event");
        assert_eq!(name, "bash");
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v["cmd"], "ls -la");
        // ToolCallsEnd at the end.
        assert!(ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
    }

    #[test]
    fn parallel_tool_calls_indices() {
        let dsml_bytes: &[u8] = b"\xef\xbd\x9cDSML\xef\xbd\x9c";
        let ev = drive(&[
            (1, b"<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke name=\"a\">\n</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke>\n<"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke name=\"b\">\n</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"invoke>\n</"),
            (TOK_DSML_TEST, dsml_bytes),
            (1, b"tool_calls>"),
        ]);
        let calls: Vec<_> = ev
            .iter()
            .filter_map(|e| {
                if let DsmlEvent::ToolCall { index, name, .. } = e {
                    Some((*index, name.clone()))
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(calls, vec![(0, "a".into()), (1, "b".into())]);
    }

    // ------------------------------------------------------------------
    // DSML repair suite — port of upstream ds4's
    // test_dsml_repair_produces_parseable_calls (0ffaabd / 596f49c /
    // 20df6c7) plus the 7bcc4e8 edge-case guards (tests 11-13). Where
    // upstream repairs a text buffer by appending closing tags and
    // re-parsing, our event scanner repairs at finish() by finalizing
    // the in-flight parameter and closing open frames — the assertions
    // check the same structural outcomes (tool name + arguments).
    // ------------------------------------------------------------------

    const DSML_B: &[u8] = b"\xef\xbd\x9cDSML\xef\xbd\x9c";

    #[test]
    fn repair_missing_tool_calls_close() {
        // Upstream TEST 1: complete invoke, missing only </tool_calls>.
        // Previously: finish=error "unterminated tool call" even though
        // a full ToolCall had already streamed. Now: emit ToolCallsEnd,
        // no malformed.
        let (ev, sc) = drive_sc(&[
            (1, b"thinking done\n\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"bash\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"command\" string=\"true\">ls -la</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter>\n</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>\n"),
            // EOS — missing </tool_calls>.
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "bash");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["command"], "ls -la");
        assert!(ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        assert!(!sc.saw_malformed(), "repairable truncation must not flag malformed");
    }

    #[test]
    fn repair_missing_invoke_and_tool_calls_close() {
        // Upstream TEST 2: missing </invoke> AND </tool_calls>. The
        // open invoke (with its completed parameter) is finalized at
        // finish().
        let (ev, sc) = drive_sc(&[
            (1, b"\n\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"edit\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"path\" string=\"true\">/tmp/test.c</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter>\n"),
            // EOS — missing </invoke> + </tool_calls>.
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "edit");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["path"], "/tmp/test.c");
        assert!(ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        assert!(!sc.saw_malformed());
    }

    #[test]
    fn repair_missing_parameter_close() {
        // Upstream TEST 3: truncated mid parameter BODY — missing
        // </parameter>, </invoke> and </tool_calls>. The in-flight
        // parameter body is finalized into the invoke.
        let (ev, sc) = drive_sc(&[
            (1, b"\n\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"bash\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"command\" string=\"true\">echo hello"),
            // EOS — nothing closed.
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "bash");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["command"], "echo hello");
        assert!(ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        assert!(!sc.saw_malformed());
        // The recovered body must NOT also leak out as text.
        assert!(!contains_subseq(&all_text(&ev), b"echo hello"));
    }

    #[test]
    fn repair_missing_parameter_close_with_explicit_invoke_close() {
        // Upstream mode 2 (596f49c): outer tags present but the inner
        // </parameter> was dropped — `…>pwd</invoke></tool_calls>`.
        // dispatch_close("invoke") finalizes the pending parameter.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"execute_command\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"command\" string=\"true\">pwd</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>\n</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>"),
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "execute_command");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["command"], "pwd");
        assert!(!sc.saw_malformed());
    }

    #[test]
    fn repair_multi_parameter_missing_tool_calls_close() {
        // Upstream TEST 4 analog: two parameters, missing only the
        // outer close.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"write_file\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"path\" string=\"true\">/tmp/out.txt</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"content\" string=\"true\">hello world</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter>\n</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>\n"),
            // EOS — missing </tool_calls>.
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].0, "write_file");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["path"], "/tmp/out.txt");
        assert_eq!(v["content"], "hello world");
        assert!(!sc.saw_malformed());
    }

    #[test]
    fn hallucinated_tool_calls_closed_block_is_plain_text() {
        // Upstream 596f49c/20df6c7 mode 3: <tool_calls> wrapping plain
        // prose with no <invoke> inside. The tags are stripped and the
        // prose surfaces as content; not an error, and NOT reported as
        // a tool_calls finish (no ToolCallsEnd).
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>I should probably call bash here.</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls> More text after."),
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(!ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        let text = all_text(&ev);
        assert!(
            contains_subseq(&text, b"I should probably call bash here."),
            "hallucinated block contents must surface as text, got {:?}",
            String::from_utf8_lossy(&text)
        );
        // Scanning continues after the stripped block.
        assert!(contains_subseq(&text, b" More text after."));
        assert!(!sc.saw_malformed());
    }

    #[test]
    fn hallucinated_tool_calls_truncated_is_plain_text() {
        // Mode 3 + truncation: <tool_calls> then prose then EOS with
        // no invoke and no closing tag.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>Let me think about this instead."),
            // EOS.
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(!ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        assert!(contains_subseq(
            &all_text(&ev),
            b"Let me think about this instead."
        ));
        assert!(!sc.saw_malformed());
    }

    #[test]
    fn truncated_bare_tool_calls_is_stripped_not_error() {
        // Model opens <｜DSML｜tool_calls> then EOS immediately. Zero
        // invokes → hallucinated-block handling (upstream repairs to
        // an empty content turn, not an error). Must not panic.
        let (ev, sc) = drive_sc(&[
            (1, b"preamble <"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>"),
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(!ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
        assert!(!sc.saw_malformed());
        assert_eq!(all_text(&ev), b"preamble ".to_vec());
    }

    #[test]
    fn extra_closing_tags_refuse_repair() {
        // Upstream 7bcc4e8 TEST 12: closing tags outnumber opening
        // tags. Upstream's size_t open-minus-close subtraction would
        // underflow and append a huge suffix; the guard refuses
        // instead. Our frame stack cannot underflow by construction —
        // assert nothing is fabricated and nothing panics.
        let (ev, _sc) = drive_sc(&[
            (1, b"done\n\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls></"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls></"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>"),
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(!ev.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
    }

    #[test]
    fn orphan_close_then_truncation_refuses_repair() {
        // 7bcc4e8 guard b at finish(): an orphan </invoke> earlier in
        // the turn means tag structure is corrupted, not truncated —
        // refuse to fabricate tool calls from the open frames.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls></"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"bash\">"),
            // EOS with an open invoke — would normally repair, but the
            // orphan close disables it.
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(sc.saw_malformed(), "orphan closes + truncation must be malformed");
    }

    #[test]
    fn unrecoverable_truncation_still_flushes_text_and_marks_malformed() {
        // Fallback path (pre-repair behavior): a parameter with no
        // enclosing invoke is not a recoverable tool call. The partial
        // body is flushed as text (letta must not see an empty turn)
        // and the turn is flagged malformed.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"cmd\" string=\"true\">curl --max-time 30"),
            // EOS.
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(sc.saw_malformed());
        assert!(contains_subseq(&all_text(&ev), b"curl --max-time 30"));
    }

    #[test]
    fn open_invoke_with_empty_name_is_not_recovered() {
        // "Emit a ToolCall for each open Frame::Invoke that has a
        // non-empty name" — an invoke without a name cannot be
        // executed; with nothing else recoverable the turn falls back
        // to malformed.
        let (ev, sc) = drive_sc(&[
            (1, b"<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>"),
            // EOS — open invoke, no name.
        ]);
        assert!(all_calls(&ev).is_empty());
        assert!(sc.saw_malformed());
    }

    #[test]
    fn think_quoted_dsml_text_is_inert() {
        // Upstream 7bcc4e8 TESTS 11+13: DSML quoted inside <think> must
        // neither trigger nor distort repair. Our architecture enforces
        // the guard upstream of the scanner (see the finish() comment):
        //
        //  1. Think-internal tokens NEVER reach the scanner: the engine
        //     worker tags chunks with `reasoning: in_think` (tracked
        //     off TOK_THINK_BEGIN/TOK_THINK_END special tokens,
        //     engine_worker.rs) and both call sites skip reasoning
        //     chunks before push_token. This test therefore simply does
        //     not feed the quoted-DSML think tokens — exactly what the
        //     callers guarantee.
        //
        //  2. Even if quoted DSML *text* reaches the scanner (model
        //     quoting markup in its answer), it arrives as regular BPE
        //     bytes without TOK_DSML transitions and cannot open a
        //     frame or affect repair accounting.
        //
        // Both properties together are the event-scanner equivalent of
        // upstream's "only scan DSML tags after the last </think>".
        let (ev, sc) = drive_sc(&[
            // Property 2: literal DSML-looking bytes, NO TOK_DSML.
            (1, b"The protocol uses <"),
            (2, DSML_B), // regular token whose bytes merely LOOK like the marker
            (1, b"tool_calls> tags, but this is only a quote.\n\n<"),
            // Property 1 aftermath / TEST 13: real DSML after the quote
            // still repairs normally.
            (TOK_DSML_TEST, DSML_B),
            (1, b"tool_calls>\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke name=\"bash\">\n<"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter name=\"command\" string=\"true\">date</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"parameter>\n</"),
            (TOK_DSML_TEST, DSML_B),
            (1, b"invoke>\n"),
            // EOS — missing </tool_calls>; repair must recover exactly
            // ONE call (the quoted tags must not distort the repair).
        ]);
        let calls = all_calls(&ev);
        assert_eq!(calls.len(), 1, "quoted DSML must not add or block tool calls");
        assert_eq!(calls[0].0, "bash");
        let v: serde_json::Value = serde_json::from_str(&calls[0].1).unwrap();
        assert_eq!(v["command"], "date");
        assert!(!sc.saw_malformed());
        // The quoted markup passes through as plain text.
        let text = all_text(&ev);
        assert!(contains_subseq(&text, b"tool_calls> tags, but this is only a quote."));
    }

    #[test]
    fn balanced_tool_calls_not_modified_by_finish() {
        // Upstream TEST 6: a fully balanced block needs no repair —
        // finish() after Mode::Done must add nothing.
        let dsml_bytes: &[u8] = DSML_B;
        let mut sc = DsmlScanner::new(TOK_DSML_TEST);
        let mut ev = Vec::new();
        for &(t, b) in &[
            (1i32, &b"<"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"tool_calls>\n<"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"invoke name=\"bash\">\n<"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"parameter name=\"command\" string=\"true\">ls</"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"parameter>\n</"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"invoke>\n</"[..]),
            (TOK_DSML_TEST, dsml_bytes),
            (1, &b"tool_calls>"[..]),
        ] {
            ev.extend(sc.push_token(t, b));
        }
        let pre_finish = ev.len();
        ev.extend(sc.finish());
        assert_eq!(ev.len(), pre_finish, "finish() added events to a balanced turn");
        assert_eq!(all_calls(&ev).len(), 1);
        assert!(!sc.saw_malformed());
    }

    fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        haystack
            .windows(needle.len())
            .any(|w| w == needle)
    }
}
