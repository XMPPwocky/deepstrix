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
         If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n\
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
    /// of `<｜DSML｜parameter name="command"…>ls`). The handler
    /// checks this at end-of-turn and reports
    /// `finish_reason: "error"` so letta knows the turn is broken
    /// rather than treating the corrupted markup as content.
    malformed: bool,
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
        // Flush any pending tail bytes if we're still in Text mode
        // outside all frames.
        if self.mode == Mode::Text && self.frames.is_empty() && !self.tail.is_empty() {
            let drained = std::mem::take(&mut self.tail);
            events.push(DsmlEvent::Text(drained));
        }
        self.mode = Mode::Done;
        events
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
                }
                // else: inside a frame (ToolCalls/Invoke), between
                // tags. Drop.
            }
            Mode::ParameterBody => {
                self.buf.extend(drained);
            }
            _ => {}
        }
    }

    fn consume_text_bytes(&mut self, bytes: &[u8], events: &mut Vec<DsmlEvent>) -> Vec<u8> {
        self.tail.extend_from_slice(bytes);
        let hold = tag_lookahead(&self.tail);
        if self.tail.len() > hold {
            let emit: Vec<u8> = self.tail.drain(..self.tail.len() - hold).collect();
            if self.frames.is_empty() {
                events.push(DsmlEvent::Text(emit));
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
            self.frames.push(Frame::ToolCalls {
                next_invoke_index: 0,
            });
            self.mode = Mode::Text;
            return;
        }
        if trimmed.starts_with("invoke") {
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
            // Pop the ToolCalls frame; emit ToolCallsEnd.
            self.frames.clear();
            events.push(DsmlEvent::ToolCallsEnd);
            self.mode = Mode::Done;
            return;
        }
        if trimmed.starts_with("invoke") {
            // Pop the Invoke frame; emit ToolCall.
            if let Some(Frame::Invoke {
                index,
                name,
                params,
            }) = self.pop_frame()
            {
                let arguments = params_to_json(&params);
                events.push(DsmlEvent::ToolCall {
                    index,
                    id: format!("call_{}", uuid::Uuid::now_v7().simple()),
                    name,
                    arguments,
                });
            }
            self.mode = Mode::Text;
            return;
        }
        if trimmed.starts_with("parameter") {
            // Finalise the current parameter (body was stashed when
            // we left ParameterBody → CloseHeader).
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
            self.mode = Mode::Text;
            return;
        }
        tracing::warn!(head, "unknown DSML close tag; falling back to Text");
        self.malformed = true;
        self.mode = Mode::Text;
    }

    fn pop_frame(&mut self) -> Option<Frame> {
        self.frames.pop()
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
        let mut sc = DsmlScanner::new(TOK_DSML_TEST);
        let mut out = Vec::new();
        for &(t, b) in seq {
            out.extend(sc.push_token(t, b));
        }
        out.extend(sc.finish());
        out
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

    fn contains_subseq(haystack: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        haystack
            .windows(needle.len())
            .any(|w| w == needle)
    }
}
