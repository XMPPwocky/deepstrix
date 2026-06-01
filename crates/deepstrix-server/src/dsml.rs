//! DSML scanner + tool-prompt renderer.
//!
//! V4-Flash emits tool calls as text-encoded DSML markup whose only
//! special token is `｜DSML｜` (id 128825). The rest is regular BPE'd
//! text. We don't need to special-case the token in the scanner — we
//! just look at decoded bytes.
//!
//! Reference (canonical engine): `external/ds4/ds4_server.c`:
//!   * tool-prompt schema block: lines 1646-1671
//!   * DSML tool-calls renderer:  lines 1857-1877
//!   * DSML streaming scanner:    lines 3233-4010
//!
//! Phase 2 scope: a **block-buffering** scanner — we accumulate the
//! entire tool block before emitting OpenAI events, rather than
//! streaming arg deltas. Letta consumes tool calls atomically anyway;
//! the streaming-arg path is an optimization that can land in a future
//! phase if needed.

use crate::openai::types::{ToolCall, ToolCallFunction, ToolDef};

// ---------------------------------------------------------------------------
// Renderer — used to construct the system-prompt schema block and to
// re-render assistant turns that contained tool calls in history.
// ---------------------------------------------------------------------------

/// Append the standard tool-prompt header + the JSON schemas for each
/// declared tool. Returns the rendered string; caller concatenates it
/// into the system prompt at first-turn. Mirrors `ds4_server.c:1646`.
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

/// Render an assistant turn's `tool_calls` back into DSML text for
/// history replay. Mirrors `ds4_server.c:1857`. The output is meant to
/// be tokenized (via BpeVocab.encode) and prefixed to the assistant's
/// reply content.
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
        // Try to parse arguments as a JSON object and emit one parameter
        // per key. Fall back to a single string-typed arguments param if
        // the JSON is malformed.
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
                // Non-object JSON: emit as a single string-typed
                // arguments param.
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
    // Mirrors ds4_server.c:1777-1788. Only escape an in-text occurrence
    // of `</｜DSML｜parameter>` (the closing tag), by replacing its leading
    // `<` with `&lt;`. Other special chars pass through.
    let end = "</\u{ff5c}DSML\u{ff5c}parameter>";
    let mut i = 0;
    let bytes = s.as_bytes();
    let end_bytes = end.as_bytes();
    while i < bytes.len() {
        if bytes[i..].starts_with(end_bytes) {
            out.push_str("&lt;");
            i += 1; // advance past '<' only; rest re-scanned
        } else {
            // Push the next char (UTF-8 boundary aware).
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
        1 // continuation byte (shouldn't happen at boundary; defensive)
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Scanner — accumulates decoded token bytes; emits events.
// ---------------------------------------------------------------------------

const TAG_TC_OPEN: &str = "<\u{ff5c}DSML\u{ff5c}tool_calls>";
const TAG_TC_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}tool_calls>";
const TAG_INVOKE_OPEN_PREFIX: &str = "<\u{ff5c}DSML\u{ff5c}invoke";
const TAG_INVOKE_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}invoke>";
const TAG_PARAM_OPEN_PREFIX: &str = "<\u{ff5c}DSML\u{ff5c}parameter";
const TAG_PARAM_CLOSE: &str = "</\u{ff5c}DSML\u{ff5c}parameter>";

#[derive(Debug, Clone)]
pub enum DsmlEvent {
    /// Plain assistant text outside any tool block.
    Text(String),
    /// A complete tool call. Emitted when `</｜DSML｜invoke>` is reached.
    /// `index` numbers calls within the current `<tool_calls>` block.
    ToolCall {
        index: u32,
        id: String,
        name: String,
        /// JSON object string (the value for OpenAI's `function.arguments`).
        arguments: String,
    },
    /// End of the enclosing `</｜DSML｜tool_calls>` block. Caller may use
    /// this to set `finish_reason="tool_calls"`.
    ToolCallsEnd,
}

#[derive(Debug)]
enum ScanMode {
    /// Outside any DSML markup. Stream bytes as text.
    Text,
    /// We're between `<｜DSML｜tool_calls>` and `</｜DSML｜tool_calls>`.
    InToolCalls { next_index: u32 },
    /// Inside an `<｜DSML｜invoke>` block.
    InInvoke {
        index: u32,
        name: String,
        params: Vec<(String, bool, String)>, // (name, is_string, body)
    },
    /// Done — any further input is ignored. Caller may break out of the
    /// decode loop.
    Done,
}

pub struct DsmlScanner {
    buf: Vec<u8>,
    mode: ScanMode,
}

impl DsmlScanner {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            mode: ScanMode::Text,
        }
    }

    /// Feed a chunk of decoded text into the scanner. Returns any
    /// events produced.
    pub fn push_text(&mut self, s: &str) -> Vec<DsmlEvent> {
        self.buf.extend_from_slice(s.as_bytes());
        let mut events = Vec::new();
        self.drain(&mut events, false);
        events
    }

    /// Mark end-of-stream. Emits any remaining text (drops unmatched
    /// partial tags). Returns any final events.
    pub fn finish(&mut self) -> Vec<DsmlEvent> {
        let mut events = Vec::new();
        self.drain(&mut events, true);
        events
    }

    fn drain(&mut self, events: &mut Vec<DsmlEvent>, eof: bool) {
        loop {
            match &mut self.mode {
                ScanMode::Done => {
                    self.buf.clear();
                    return;
                }
                ScanMode::Text => {
                    // Scan for TAG_TC_OPEN. Emit everything before it as Text.
                    let buf_str = std::str::from_utf8(&self.buf).unwrap_or("");
                    if let Some(pos) = buf_str.find(TAG_TC_OPEN) {
                        if pos > 0 {
                            let prefix: String = buf_str[..pos].into();
                            events.push(DsmlEvent::Text(prefix));
                        }
                        let consumed_bytes = pos + TAG_TC_OPEN.len();
                        self.buf.drain(..consumed_bytes);
                        self.mode = ScanMode::InToolCalls { next_index: 0 };
                        continue;
                    }
                    // No full open-tag found. If the buf ends with a
                    // prefix of TAG_TC_OPEN, hold it back.
                    let hold = trailing_prefix_len(buf_str, TAG_TC_OPEN);
                    if buf_str.len() > hold {
                        let emit_end = buf_str.len() - hold;
                        let emit: String = buf_str[..emit_end].into();
                        events.push(DsmlEvent::Text(emit));
                        self.buf.drain(..emit_end);
                    }
                    if eof && !self.buf.is_empty() {
                        // Flush any leftover bytes as text.
                        let buf_str = std::str::from_utf8(&self.buf)
                            .unwrap_or("")
                            .to_string();
                        events.push(DsmlEvent::Text(buf_str));
                        self.buf.clear();
                    }
                    return;
                }
                ScanMode::InToolCalls { next_index } => {
                    // We expect either `<｜DSML｜invoke ...>` (start a call)
                    // or `</｜DSML｜tool_calls>` (end the block). Whitespace
                    // is permitted between them.
                    let buf_str = std::str::from_utf8(&self.buf).unwrap_or("");
                    // Skip leading whitespace cheaply.
                    let trimmed_start = buf_str
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or(buf_str.len());
                    if buf_str[trimmed_start..].starts_with(TAG_TC_CLOSE) {
                        let consumed = trimmed_start + TAG_TC_CLOSE.len();
                        self.buf.drain(..consumed);
                        events.push(DsmlEvent::ToolCallsEnd);
                        self.mode = ScanMode::Done;
                        continue;
                    }
                    if buf_str[trimmed_start..].starts_with(TAG_INVOKE_OPEN_PREFIX) {
                        // We need the whole `<｜DSML｜invoke name="..."` …`>`
                        // header before we can start the call. Find the
                        // closing `>` of the header (the FIRST `>` after
                        // TAG_INVOKE_OPEN_PREFIX, but watch out for `>`
                        // inside a quoted attribute — we only have name=,
                        // so safe to use bare `>` search).
                        let header_start = trimmed_start;
                        let scan_from = header_start + TAG_INVOKE_OPEN_PREFIX.len();
                        if let Some(rel_close) = buf_str[scan_from..].find('>') {
                            let header_end = scan_from + rel_close + 1;
                            let header_slice = &buf_str[header_start..header_end];
                            let name =
                                parse_attr(header_slice, "name").unwrap_or_default();
                            self.buf.drain(..header_end);
                            let idx = *next_index;
                            *next_index += 1;
                            self.mode = ScanMode::InInvoke {
                                index: idx,
                                name,
                                params: Vec::new(),
                            };
                            continue;
                        }
                        // Incomplete header; wait for more bytes.
                        return;
                    }
                    // Neither tag matched yet. If the leading non-ws
                    // is a strict prefix of either tag, wait. Otherwise
                    // tolerate stray text by emitting it as Text and
                    // moving on (matches ds4's behavior).
                    let after_ws = &buf_str[trimmed_start..];
                    let hold_close = trailing_prefix_len(after_ws, TAG_TC_CLOSE);
                    let hold_open = trailing_prefix_len(after_ws, TAG_INVOKE_OPEN_PREFIX);
                    let hold = hold_close.max(hold_open);
                    if after_ws.len() > hold {
                        // Emit the leading whitespace + stray text as Text,
                        // hold the trailing partial.
                        let emit_end = trimmed_start + (after_ws.len() - hold);
                        let emit: String = buf_str[..emit_end].into();
                        events.push(DsmlEvent::Text(emit));
                        self.buf.drain(..emit_end);
                    }
                    return;
                }
                ScanMode::InInvoke {
                    index,
                    name,
                    params,
                } => {
                    let buf_str = std::str::from_utf8(&self.buf).unwrap_or("");
                    let trimmed_start = buf_str
                        .find(|c: char| !c.is_whitespace())
                        .unwrap_or(buf_str.len());
                    if buf_str[trimmed_start..].starts_with(TAG_INVOKE_CLOSE) {
                        // Complete the call.
                        let consumed = trimmed_start + TAG_INVOKE_CLOSE.len();
                        self.buf.drain(..consumed);
                        let arguments = params_to_json(params);
                        let id = format!("call_{}", uuid::Uuid::now_v7().simple());
                        events.push(DsmlEvent::ToolCall {
                            index: *index,
                            id,
                            name: std::mem::take(name),
                            arguments,
                        });
                        // After </｜DSML｜invoke>, we're back in the
                        // surrounding <tool_calls> block.
                        let next_idx = *index + 1;
                        self.mode = ScanMode::InToolCalls {
                            next_index: next_idx,
                        };
                        continue;
                    }
                    if buf_str[trimmed_start..].starts_with(TAG_PARAM_OPEN_PREFIX) {
                        // Parse `<｜DSML｜parameter name="..." string="...">`.
                        let header_start = trimmed_start;
                        let scan_from = header_start + TAG_PARAM_OPEN_PREFIX.len();
                        let Some(rel_close) = buf_str[scan_from..].find('>') else {
                            // header incomplete; wait
                            return;
                        };
                        let header_end = scan_from + rel_close + 1;
                        let header_slice = &buf_str[header_start..header_end];
                        let param_name =
                            parse_attr(header_slice, "name").unwrap_or_default();
                        let is_string = parse_attr(header_slice, "string")
                            .map(|s| s == "true")
                            .unwrap_or(true);
                        // We now need the body up to `</｜DSML｜parameter>`.
                        // The body is everything from `header_end` to the
                        // first occurrence of TAG_PARAM_CLOSE.
                        let after_header = &buf_str[header_end..];
                        if let Some(rel_body_end) = after_header.find(TAG_PARAM_CLOSE) {
                            let body: String = after_header[..rel_body_end].into();
                            let total =
                                header_end + rel_body_end + TAG_PARAM_CLOSE.len();
                            self.buf.drain(..total);
                            params.push((param_name, is_string, body));
                            continue;
                        }
                        // Incomplete body — wait.
                        return;
                    }
                    // Not a recognized tag yet. Hold partial prefix.
                    let after_ws = &buf_str[trimmed_start..];
                    let hold_close = trailing_prefix_len(after_ws, TAG_INVOKE_CLOSE);
                    let hold_open = trailing_prefix_len(after_ws, TAG_PARAM_OPEN_PREFIX);
                    let hold = hold_close.max(hold_open);
                    if after_ws.len() > hold {
                        // Stray text between params — discard (matches ds4 lenient parse).
                        let advance = trimmed_start + (after_ws.len() - hold);
                        self.buf.drain(..advance);
                    }
                    return;
                }
            }
        }
    }
}

impl Default for DsmlScanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Length of the longest prefix of `lit` that is a suffix of `s`.
/// Used to decide how many trailing bytes to hold back so a partial
/// tag crossing a token boundary isn't emitted as text.
fn trailing_prefix_len(s: &str, lit: &str) -> usize {
    let lit_bytes = lit.as_bytes();
    let s_bytes = s.as_bytes();
    // Largest k > 0 such that s ends with lit[..k] AND k <= lit_bytes.len()-1
    // (i.e. proper prefix).
    let max_k = lit_bytes.len().saturating_sub(1).min(s_bytes.len());
    for k in (1..=max_k).rev() {
        if s_bytes.ends_with(&lit_bytes[..k]) {
            // Must also be a valid char-boundary in s.
            let split = s_bytes.len() - k;
            if s.is_char_boundary(split) {
                return k;
            }
        }
    }
    0
}

/// Find an `attr="value"` pair in a tag header slice and return the
/// (DSML-decoded) value.
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
            // peek for &amp; &lt; &gt; &quot;
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

/// Decode `&amp;`, `&lt;`, `&gt;`, `&quot;` in a parameter body.
fn dsml_param_decode_string(s: &str) -> String {
    dsml_attr_decode(s)
}

/// Decode `<` back to `<` (the only DSML→JSON escape used by ds4's
/// renderer for JSON literals, per ds4_server.c:1790).
fn dsml_param_decode_json_literal(s: &str) -> String {
    s.replace("\\u003c", "<")
}

/// Assemble a JSON object string from the parsed parameter list.
/// Parameters with `string="true"` become JSON strings; `string="false"`
/// values are passed through as already-valid JSON literals.
fn params_to_json(params: &[(String, bool, String)]) -> String {
    let mut map = serde_json::Map::new();
    for (k, is_string, body) in params {
        if *is_string {
            map.insert(
                k.clone(),
                serde_json::Value::String(dsml_param_decode_string(body)),
            );
        } else {
            let decoded = dsml_param_decode_json_literal(body);
            match serde_json::from_str::<serde_json::Value>(decoded.trim()) {
                Ok(v) => {
                    map.insert(k.clone(), v);
                }
                Err(_) => {
                    // Fall back to raw string if the literal is malformed.
                    map.insert(k.clone(), serde_json::Value::String(decoded));
                }
            }
        }
    }
    serde_json::Value::Object(map).to_string()
}

/// Convert a list of parsed `ToolCall`s (one per `<invoke>`) into the
/// OpenAI `tool_calls` array used in `Choice::message::tool_calls`.
/// Phase 1 helper for the non-streaming response path.
pub fn tool_calls_from_events(events: Vec<DsmlEvent>) -> Vec<ToolCall> {
    let mut out = Vec::new();
    for ev in events {
        if let DsmlEvent::ToolCall {
            id,
            name,
            arguments,
            ..
        } = ev
        {
            out.push(ToolCall {
                id,
                kind: "function".into(),
                function: ToolCallFunction { name, arguments },
            });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one_chunk(s: &str) -> Vec<DsmlEvent> {
        let mut sc = DsmlScanner::new();
        let mut ev = sc.push_text(s);
        ev.extend(sc.finish());
        ev
    }

    #[test]
    fn plain_text_passes_through() {
        let ev = one_chunk("hello world");
        assert_eq!(ev.len(), 1);
        match &ev[0] {
            DsmlEvent::Text(t) => assert_eq!(t, "hello world"),
            other => panic!("unexpected: {:?}", other),
        }
    }

    #[test]
    fn single_tool_call() {
        let s = "Sure! \n\n<\u{ff5c}DSML\u{ff5c}tool_calls>\n\
                 <\u{ff5c}DSML\u{ff5c}invoke name=\"bash\">\n\
                 <\u{ff5c}DSML\u{ff5c}parameter name=\"command\" string=\"true\">ls -la</\u{ff5c}DSML\u{ff5c}parameter>\n\
                 </\u{ff5c}DSML\u{ff5c}invoke>\n\
                 </\u{ff5c}DSML\u{ff5c}tool_calls>";
        let ev = one_chunk(s);
        // Expect: Text("Sure! \n\n"), ToolCall {bash}, ToolCallsEnd
        assert!(matches!(&ev[0], DsmlEvent::Text(t) if t == "Sure! \n\n"));
        match &ev[1] {
            DsmlEvent::ToolCall { name, arguments, index, .. } => {
                assert_eq!(name, "bash");
                assert_eq!(*index, 0);
                let v: serde_json::Value = serde_json::from_str(arguments).unwrap();
                assert_eq!(v["command"], "ls -la");
            }
            other => panic!("unexpected: {:?}", other),
        }
        assert!(matches!(&ev[2], DsmlEvent::ToolCallsEnd));
    }

    #[test]
    fn partial_tag_split_across_chunks() {
        let mut sc = DsmlScanner::new();
        let ev1 = sc.push_text("hello <\u{ff5c}DSML");
        // We've now sent a prefix of the open tag mid-buffer. Should
        // emit "hello " text and hold the partial.
        assert_eq!(ev1.len(), 1);
        match &ev1[0] {
            DsmlEvent::Text(t) => assert_eq!(t, "hello "),
            other => panic!("unexpected: {:?}", other),
        }
        let ev2 = sc.push_text(
            "\u{ff5c}tool_calls>\n<\u{ff5c}DSML\u{ff5c}invoke name=\"x\">\n\
             </\u{ff5c}DSML\u{ff5c}invoke>\n</\u{ff5c}DSML\u{ff5c}tool_calls>",
        );
        // Now we expect ToolCall and ToolCallsEnd.
        assert!(ev2.iter().any(|e| matches!(e, DsmlEvent::ToolCall { name, .. } if name == "x")));
        assert!(ev2.iter().any(|e| matches!(e, DsmlEvent::ToolCallsEnd)));
    }

    #[test]
    fn json_typed_parameter() {
        let s = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n\
                 <\u{ff5c}DSML\u{ff5c}invoke name=\"f\">\n\
                 <\u{ff5c}DSML\u{ff5c}parameter name=\"n\" string=\"false\">42</\u{ff5c}DSML\u{ff5c}parameter>\n\
                 <\u{ff5c}DSML\u{ff5c}parameter name=\"flag\" string=\"false\">true</\u{ff5c}DSML\u{ff5c}parameter>\n\
                 </\u{ff5c}DSML\u{ff5c}invoke>\n\
                 </\u{ff5c}DSML\u{ff5c}tool_calls>";
        let ev = one_chunk(s);
        let tc = ev
            .iter()
            .find_map(|e| {
                if let DsmlEvent::ToolCall { arguments, .. } = e {
                    Some(arguments.clone())
                } else {
                    None
                }
            })
            .expect("tool call present");
        let v: serde_json::Value = serde_json::from_str(&tc).unwrap();
        assert_eq!(v["n"], 42);
        assert_eq!(v["flag"], true);
    }

    #[test]
    fn parallel_tool_calls() {
        let s = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n\
                 <\u{ff5c}DSML\u{ff5c}invoke name=\"a\">\n\
                 </\u{ff5c}DSML\u{ff5c}invoke>\n\
                 <\u{ff5c}DSML\u{ff5c}invoke name=\"b\">\n\
                 </\u{ff5c}DSML\u{ff5c}invoke>\n\
                 </\u{ff5c}DSML\u{ff5c}tool_calls>";
        let ev = one_chunk(s);
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

    #[test]
    fn escaped_param_close_in_body() {
        // The model is instructed to emit "&lt;/｜DSML｜parameter>" when
        // a string body contains the closing tag verbatim.
        let s = "<\u{ff5c}DSML\u{ff5c}tool_calls>\n\
                 <\u{ff5c}DSML\u{ff5c}invoke name=\"t\">\n\
                 <\u{ff5c}DSML\u{ff5c}parameter name=\"x\" string=\"true\">a &amp; b &lt; c</\u{ff5c}DSML\u{ff5c}parameter>\n\
                 </\u{ff5c}DSML\u{ff5c}invoke>\n\
                 </\u{ff5c}DSML\u{ff5c}tool_calls>";
        let ev = one_chunk(s);
        let tc = ev
            .iter()
            .find_map(|e| {
                if let DsmlEvent::ToolCall { arguments, .. } = e {
                    Some(arguments.clone())
                } else {
                    None
                }
            })
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&tc).unwrap();
        assert_eq!(v["x"], "a & b < c");
    }
}
