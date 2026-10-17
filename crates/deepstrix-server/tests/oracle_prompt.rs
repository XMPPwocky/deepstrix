//! Differential tests: the server's DeepSeek-V4.1 prompt rendering and
//! model-output parsing vs DeepSeek's official reference implementation,
//! `deepseek-recipe` (github.com/deepseek-ai/deepseek-recipe, MIT; pinned in
//! Cargo.toml), released with DeepSeek-V4.1-Flash-1017.
//!
//! * Prompt: the SAME OpenAI request JSON goes through
//!   `deepseek_recipe::openai::ChatCompletionRequest::convert` +
//!   `DeepseekV41Encoding::render_conversation` (the oracle) and through the
//!   server's own deserializer + `prompt_v41::request_segments_v41` (what
//!   `handler::chat_completions` renders). Prompts must be byte-identical,
//!   token ids identical, and an input the oracle rejects must be rejected.
//! * Output: one model output (a token stream: special tokens + text) goes
//!   through the oracle's `StreamProcessor` (text chunks, as a backend would
//!   decode them) and through the server's worker-event consumers
//!   (`engine_worker::accumulate` + `handler::nonstream_message`, and the SSE
//!   driver via `handler::sse_transcript`). Content, reasoning, tool calls
//!   (name + arguments as parsed JSON) and finish reason must agree.
//!
//! Known, named deviations are asserted explicitly in `deviation_*` tests and
//! listed in `KNOWN_DEVIATIONS`; nothing is silently skipped.
//!
//! Case counts: `ORACLE_CASES` (default 512) per property; a long run is
//! `ORACLE_CASES=20000`. Failures persist to `proptest-regressions/`.
//! Token ids need the V4.1 tokenizer (`V41_TOKENIZER_JSON`, default
//! `/weights2/dsv4.1f-1017/tokenizer.json`); without it the id leg is skipped
//! with a message. The secondary oracle (the checkpoint's `encoding.py`) runs
//! in `secondary_oracle_encoding_py_vs_recipe` when `ENCODING_PY_DIR` (default
//! `/weights2/dsv4.1f-1017/encoding`) exists.
//!
//!   CARGO_TARGET_DIR=target-v41 nix develop -c cargo test -p deepstrix-server \
//!       --features v41 --release --test oracle_prompt

use std::sync::OnceLock;

use deepseek_recipe::openai::ChatCompletionRequest as OracleRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use deepstrix_server::engine_worker::{accumulate, FinishReason, WorkerEvent};
use deepstrix_server::openai::types::ChatCompletionRequest;
use deepstrix_server::prompt::{ReasoningEffort, Seg};
use deepstrix_server::prompt_v41;
use proptest::prelude::*;
use proptest::strategy::ValueTree;
use proptest::test_runner::{Config, FileFailurePersistence, TestRunner};
use serde_json::{json, Value};
use v4flash_core::BpeVocab;

// ---------------------------------------------------------------------------
// The two sides
// ---------------------------------------------------------------------------

/// The image placeholder: the oracle's bundled tokenizer spells id 129264
/// `<｜image｜>`, the checkpoint's tokenizer.json (which the server loads)
/// `<｜deepseek_image｜>`. Same id, so prompts are compared with the oracle's
/// spelling mapped to ours (KNOWN_DEVIATIONS: IMAGE_PLACEHOLDER_SPELLING).
const IMG_OURS: &str = "<\u{ff5c}deepseek_image\u{ff5c}>";
const IMG_ORACLE: &str = "<\u{ff5c}image\u{ff5c}>";

/// What the server uses for a request that names no effort
/// (`--default-reasoning-effort`'s compiled-in default).
fn server_default_effort() -> ReasoningEffort {
    deepstrix_server::prompt::DEFAULT_EFFORT
}

fn oracle_prompt(req: &Value) -> Result<String, String> {
    let r: OracleRequest = serde_json::from_value(req.clone()).map_err(|e| format!("oracle deserialize: {e}"))?;
    let c = r.convert(ConversionOptions::default()).map_err(|e| format!("oracle convert: {e}"))?;
    let p = DeepseekV41Encoding::new().render_conversation(&c.conversation);
    Ok(p.prompt.replace(IMG_ORACLE, IMG_OURS))
}

fn ours_request(req: &Value) -> Result<ChatCompletionRequest, String> {
    serde_json::from_value(req.clone()).map_err(|e| format!("ours deserialize: {e}"))
}

fn ours_segments(req: &Value) -> Result<Vec<Seg>, String> {
    let r = ours_request(req)?;
    prompt_v41::request_segments_v41(&r, server_default_effort()).map(|(s, _)| s).map_err(|e| format!("ours render: {e:#}"))
}

fn ours_prompt(req: &Value) -> Result<String, String> {
    Ok(ours_segments(req)?.iter().map(Seg::text).collect())
}

/// First difference, with context, for a readable failure.
fn diff(a: &str, b: &str) -> String {
    let at = a.char_indices().zip(b.chars()).find(|((_, x), y)| x != y).map(|((i, _), _)| i).unwrap_or(a.len().min(b.len()));
    let lo = a[..at].char_indices().rev().nth(60).map(|(i, _)| i).unwrap_or(0);
    let tail = |s: &str| s[at..].chars().take(80).collect::<String>();
    format!("first difference at byte {at}\n  context: {:?}\n   oracle: {:?}\n     ours: {:?}", &a[lo..at], tail(a), tail(b))
}

// ---------------------------------------------------------------------------
// Token ids
// ---------------------------------------------------------------------------

struct Vocab {
    bpe: BpeVocab,
    /// Every added token of tokenizer.json, as HF matches them in text.
    added: Vec<(String, i32)>,
    image_id: Option<i32>,
}

fn vocab() -> Option<&'static Vocab> {
    static V: OnceLock<Option<Vocab>> = OnceLock::new();
    V.get_or_init(|| {
        let path = std::env::var("V41_TOKENIZER_JSON").unwrap_or_else(|_| "/weights2/dsv4.1f-1017/tokenizer.json".into());
        if !std::path::Path::new(&path).exists() {
            eprintln!("oracle_prompt: {path} not found (set V41_TOKENIZER_JSON); token-id comparison SKIPPED");
            return None;
        }
        let bpe = BpeVocab::from_tokenizer_json(&path, Some("joyai-llm".to_string())).expect("load tokenizer.json");
        let raw: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut added: Vec<(String, i32)> = raw["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| (t["content"].as_str().unwrap().to_string(), t["id"].as_i64().unwrap() as i32))
            .collect();
        added.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
        let image_id = bpe.lookup_token_id(IMG_OURS);
        Some(Vocab { bpe, added, image_id })
    })
    .as_ref()
}

/// How HF `tokenizers` encodes a prompt string (`encode(text, false)`, as
/// `deepseek_recipe_encoding::PromptEncoding::encode` does): every added-token
/// literal, wherever it occurs, is its id (leftmost-longest); the gaps are
/// BPE'd. The BPE itself is the server's (`BpeVocab::encode`) -- the oracle's
/// HF tokenizer object is not reachable without a direct `tokenizers`
/// dependency (KNOWN_DEVIATIONS note TOKEN_IDS_VIA_OUR_BPE).
fn hf_style_ids(v: &Vocab, text: &str) -> Vec<i32> {
    let mut out = Vec::new();
    let mut rest = text;
    loop {
        let hit = v
            .added
            .iter()
            .filter_map(|(lit, id)| rest.find(lit.as_str()).map(|at| (at, lit.len(), *id)))
            .min_by_key(|(at, len, _)| (*at, std::cmp::Reverse(*len)));
        match hit {
            Some((at, len, id)) => {
                if at > 0 {
                    out.extend(v.bpe.encode(&rest[..at]));
                }
                out.push(id);
                rest = &rest[at + len..];
            }
            None => {
                if !rest.is_empty() {
                    out.extend(v.bpe.encode(rest));
                }
                return out;
            }
        }
    }
}

/// Token-id leg: the server's ids for the request vs HF-style ids of the
/// oracle's prompt. They may differ ONLY where client text spells an added
/// token (the server never lets client text become a control token:
/// KNOWN_DEVIATIONS CLIENT_SPECIALS_INERT).
fn check_ids(req: &Value, oracle: &str) -> Result<(), String> {
    let Some(v) = vocab() else { return Ok(()) };
    let r = ours_request(req)?;
    let segs = prompt_v41::request_segments_v41(&r, server_default_effort()).map_err(|e| format!("{e:#}"))?.0;
    let (ours, _) = prompt_v41::render_request_v41(&v.bpe, &r, server_default_effort(), v.image_id).map_err(|e| format!("{e:#}"))?;
    let reference = hf_style_ids(v, oracle);
    if ours == reference {
        return Ok(());
    }
    let client_spells_special = segs.iter().any(|s| matches!(s, Seg::Client(t) if v.added.iter().any(|(lit, _)| t.contains(lit.as_str()))));
    if client_spells_special {
        return Ok(());
    }
    let at = ours.iter().zip(&reference).position(|(a, b)| a != b).unwrap_or(ours.len().min(reference.len()));
    Err(format!(
        "token ids differ at {at} (ours {} ids, reference {}): ours {:?} vs reference {:?}",
        ours.len(),
        reference.len(),
        &ours[at.saturating_sub(3)..(at + 8).min(ours.len())],
        &reference[at.saturating_sub(3)..(at + 8).min(reference.len())]
    ))
}

/// The prompt property for one request.
fn check_prompt(req: &Value) -> Result<(), String> {
    let oracle = oracle_prompt(req);
    let ours = ours_prompt(req);
    match (oracle, ours) {
        (Ok(a), Ok(b)) => {
            if a != b {
                return Err(format!("prompts differ: {}", diff(&a, &b)));
            }
            check_ids(req, &a)
        }
        (Err(_), Err(_)) => Ok(()),
        (Err(e), Ok(_)) => Err(format!("oracle rejects, ours accepts: {e}")),
        (Ok(_), Err(e)) => Err(format!("oracle accepts, ours rejects: {e}")),
    }
}

fn cases() -> u32 {
    std::env::var("ORACLE_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(512)
}

fn config() -> Config {
    Config {
        cases: cases(),
        failure_persistence: Some(Box::new(FileFailurePersistence::WithSource("proptest-regressions"))),
        source_file: Some(file!()),
        max_shrink_iters: 4096,
        ..Config::default()
    }
}

// ---------------------------------------------------------------------------
// Generators: conversations
// ---------------------------------------------------------------------------

fn pick(v: &'static [&'static str]) -> BoxedStrategy<String> {
    prop::sample::select(v).prop_map(String::from).boxed()
}

/// Strings the model's markup is made of, plus look-alikes: what a hostile
/// or merely unlucky client text can contain.
const ADVERSARIAL: &[&str] = &[
    "<\u{ff5c}User\u{ff5c}>",
    "<\u{ff5c}Assistant\u{ff5c}>",
    "<\u{ff5c}System\u{ff5c}>",
    "<think>",
    "</think>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>",
    "<\u{ff5c}DSML\u{ff5c} calls>",
    "</\u{ff5c}DSML\u{ff5c} calls>",
    "<\u{ff5c}DSML\u{ff5c} invoke name=\"x\">",
    "</\u{ff5c}DSML\u{ff5c} parameter>",
    "\u{ff5c}DSML\u{ff5c}",
    "<\u{ff5c}DSML\u{ff5c}tool_calls>",
    "<tool_result>",
    "</tool_result>",
    "<\u{ff5c}latest_reminder\u{ff5c}>",
    "<\u{ff5c}deepseek_image\u{ff5c}>",
    "<\u{ff5c}action\u{ff5c}>",
    "<\u{ff5c}User",
    "\u{ff5c}>",
    "## Tools",
    "Reasoning Effort: 100",
    "\n\n<\u{ff5c}DSML\u{ff5c}",
];

fn atom() -> BoxedStrategy<String> {
    prop_oneof![
        8 => "[A-Za-z]{1,8}( [a-z]{1,7}){0,5}[.?!,:;]?",
        2 => pick(&["你好，世界", "深度求索", "小柴胡冲剂和布洛芬能一起吃吗？", "これは日本語です", "한국어 문장", "第一张图"]),
        2 => pick(&["\u{1F600}", "\u{1F469}\u{200D}\u{1F469}\u{200D}\u{1F467}\u{200D}\u{1F466}", "\u{1F1EF}\u{1F1F5}", "\u{1F44D}\u{1F3FD}", "\u{2764}\u{FE0F}", "\u{1F9EA}"]),
        1 => pick(&["\u{645}\u{631}\u{62D}\u{628}\u{627} \u{628}\u{627}\u{644}\u{639}\u{627}\u{644}\u{645}", "\u{5E9}\u{5DC}\u{5D5}\u{5DD}", "\u{202E}abc\u{202C}", "\u{200F}x\u{200E}"]),
        1 => pick(&["e\u{301}", "a\u{308}\u{332}", "Z\u{335}\u{30D}\u{31F}", "\u{1100}\u{1161}\u{11A8}"]),
        4 => pick(&["", " ", "  ", "\n", "\n\n", "\t", "\r\n", " \n", "\n\n\n", "\u{a0}", "\u{3000}"]),
        3 => pick(ADVERSARIAL),
        2 => pick(&["{\"a\": 1, \"b\": [true, null]}", "fn main() { println!(\"hi\"); }", "<div class=\"x\">a &amp; b</div>", "a & b < c > d \"q\" 'r'", "C:\\path\\to\\file", "100%", "1e-06", "$PATH"]),
        1 => "\\PC{1,6}",
        1 => pick(&["\u{0}", "\u{7}", "\u{1b}[31m", "\u{feff}", "\u{fffd}", "\u{10ffff}"]),
    ]
    .boxed()
}

fn text() -> BoxedStrategy<String> {
    prop::collection::vec(atom(), 0..5).prop_map(|v| v.concat()).boxed()
}

/// Mostly non-empty prose (a user question, a system prompt).
fn prose() -> BoxedStrategy<String> {
    prop_oneof![
        6 => ("[A-Z][a-z]{2,8}( [a-z]{1,8}){1,8}[.?]", text()).prop_map(|(a, b)| format!("{a}{b}")),
        2 => text(),
    ]
    .boxed()
}

#[derive(Debug, Clone)]
enum PartG {
    Text(String),
    Image(u8),
}

#[derive(Debug, Clone)]
enum ContentG {
    Null,
    Str(String),
    Parts(Vec<PartG>),
}

impl ContentG {
    fn to_json(&self) -> Value {
        match self {
            ContentG::Null => Value::Null,
            ContentG::Str(s) => Value::String(s.clone()),
            ContentG::Parts(ps) => Value::Array(
                ps.iter()
                    .map(|p| match p {
                        PartG::Text(t) => json!({"type": "text", "text": t}),
                        PartG::Image(k) => json!({"type": "image_url", "image_url": {"url": format!("data:image/png;base64,iVBORw0KGgo{k}AAAA")}}),
                    })
                    .collect(),
            ),
        }
    }
}

fn text_parts() -> BoxedStrategy<ContentG> {
    prop::collection::vec(text().prop_map(PartG::Text), 0..4).prop_map(ContentG::Parts).boxed()
}

/// User / tool content: a string, or content parts (text and images).
fn rich_content() -> BoxedStrategy<ContentG> {
    let part = prop_oneof![4 => text().prop_map(PartG::Text), 1 => (0u8..4).prop_map(PartG::Image)];
    prop_oneof![
        6 => prose().prop_map(ContentG::Str),
        2 => text().prop_map(ContentG::Str),
        2 => prop::collection::vec(part, 0..5).prop_map(ContentG::Parts),
    ]
    .boxed()
}

/// System content: string or text-only parts.
fn system_content() -> BoxedStrategy<ContentG> {
    prop_oneof![6 => prose().prop_map(ContentG::Str), 2 => text().prop_map(ContentG::Str), 1 => text_parts()].boxed()
}

/// Assistant content (never images).
fn assistant_content(allow_null: bool) -> BoxedStrategy<ContentG> {
    let some = prop_oneof![
        5 => prose().prop_map(ContentG::Str),
        2 => text().prop_map(ContentG::Str),
        2 => (prose(), pick(&["\n\n", "\n", " ", "\n\n\n"])).prop_map(|(a, b)| ContentG::Str(a + &b)),
        1 => text_parts(),
    ];
    if allow_null {
        prop_oneof![5 => some, 1 => Just(ContentG::Null)].boxed()
    } else {
        some.boxed()
    }
}

const FLOATS: &[f64] = &[0.5, 1e-6, 2.5e-3, 1e-5, 0.0001, 3.0, -2.5, 100.0, 1e16, 1e30, 1.2345678901234567e20, -0.0, 6.02e23];

fn json_scalar() -> BoxedStrategy<Value> {
    prop_oneof![
        3 => text().prop_map(Value::String),
        2 => any::<i32>().prop_map(|n| json!(n)),
        1 => prop::sample::select(FLOATS).prop_map(|f| json!(f)),
        1 => any::<bool>().prop_map(Value::Bool),
        1 => Just(Value::Null),
        1 => prop::sample::select(BIG_INTS).prop_map(|n| json!(n)),
    ]
    .boxed()
}

const BIG_INTS: &[i64] = &[i64::MAX, i64::MIN, 0, 9007199254740993];

const KEYS: &[&str] = &["location", "unit", "query", "num_results", "path", "options", "items", "flag", "limit", "nested", "content", "command", "id"];

fn json_value() -> BoxedStrategy<Value> {
    json_scalar()
        .prop_recursive(3, 24, 4, |inner| {
            prop_oneof![
                prop::collection::vec(inner.clone(), 0..4).prop_map(Value::Array),
                prop::collection::vec((prop::sample::select(KEYS), inner), 0..4)
                    .prop_map(|kv| Value::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect())),
            ]
        })
        .boxed()
}

/// A tool call's `arguments` string, as clients send it.
fn arguments() -> BoxedStrategy<String> {
    let obj = prop::collection::vec((prop::sample::select(KEYS), json_value()), 0..5)
        .prop_map(|kv| Value::Object(kv.into_iter().map(|(k, v)| (k.to_string(), v)).collect()));
    prop_oneof![
        8 => obj.clone().prop_map(|v| v.to_string()),
        2 => obj.clone().prop_map(|v| serde_json::to_string_pretty(&v).unwrap()),
        1 => obj.prop_map(|v| Value::String(v.to_string()).to_string()),
        1 => pick(&["", "not json", "[1, 2]", "\"just a string\"", "{}", "null", "{\"a\": 1", "42"]),
    ]
    .boxed()
}

fn schema_leaf() -> BoxedStrategy<Value> {
    prop_oneof![
        3 => prop::option::of(text()).prop_map(|d| match d { Some(d) => json!({"type": "string", "description": d}), None => json!({"type": "string"}) }),
        1 => Just(json!({"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"})),
        1 => Just(json!({"type": "integer", "minimum": 0, "default": 10})),
        1 => prop::sample::select(FLOATS).prop_map(|f| json!({"type": "number", "default": f})),
        1 => Just(json!({"type": "boolean"})),
        1 => Just(json!({"type": "array", "items": {"type": "string"}})),
        1 => Just(json!({"anyOf": [{"type": "integer"}, {"type": "string"}], "default": -1, "description": "ID or URL"})),
    ]
    .boxed()
}

fn schema_object(depth: u32) -> BoxedStrategy<Value> {
    let prop_s = if depth == 0 {
        schema_leaf()
    } else {
        prop_oneof![4 => schema_leaf(), 1 => schema_object(depth - 1), 1 => schema_object(depth - 1).prop_map(|o| json!({"type": "array", "items": o}))].boxed()
    };
    (prop::sample::subsequence(KEYS, 0..5), prop::collection::vec(prop_s, 5), any::<u8>(), any::<bool>())
        .prop_map(|(keys, props, req_mask, closed)| {
            let mut properties = serde_json::Map::new();
            for (k, p) in keys.iter().zip(props) {
                properties.insert(k.to_string(), p);
            }
            let required: Vec<&str> = keys.iter().enumerate().filter(|(i, _)| req_mask & (1 << i) != 0).map(|(_, k)| *k).collect();
            let mut o = json!({"type": "object", "properties": properties});
            if !required.is_empty() {
                o["required"] = json!(required);
            }
            if closed {
                o["additionalProperties"] = json!(false);
            }
            o
        })
        .boxed()
}

const TOOL_NAMES: &[&str] = &["get_weather", "search", "open", "run_shell", "read_file", "write_file", "calc", "fetch_url", "list_dir", "todo-list"];

#[derive(Debug, Clone)]
struct ToolG {
    name: String,
    description: Option<String>,
    parameters: Option<Value>,
    strict: Option<bool>,
    /// Client key order description-after-parameters.
    reordered: bool,
}

impl ToolG {
    fn to_json(&self) -> Value {
        let mut f = serde_json::Map::new();
        f.insert("name".into(), json!(self.name));
        if !self.reordered {
            if let Some(d) = &self.description {
                f.insert("description".into(), json!(d));
            }
        }
        if let Some(p) = &self.parameters {
            f.insert("parameters".into(), p.clone());
        }
        if self.reordered {
            if let Some(d) = &self.description {
                f.insert("description".into(), json!(d));
            }
        }
        if let Some(s) = self.strict {
            f.insert("strict".into(), json!(s));
        }
        json!({"type": "function", "function": f})
    }
}

fn tools() -> BoxedStrategy<Vec<ToolG>> {
    let one = (
        prop_oneof![6 => prose().prop_map(Some), 1 => text().prop_map(Some), 1 => Just(None)],
        prop_oneof![6 => schema_object(2).prop_map(Some), 1 => Just(None)],
        prop_oneof![8 => Just(None), 1 => any::<bool>().prop_map(Some)],
        prop::bool::weighted(0.1),
    );
    (prop::sample::subsequence(TOOL_NAMES, 0..5), prop::collection::vec(one, 5))
        .prop_map(|(names, rest)| {
            names
                .into_iter()
                .zip(rest)
                .map(|(n, (description, parameters, strict, reordered))| ToolG { name: n.to_string(), description, parameters, strict, reordered })
                .collect()
        })
        .boxed()
}

#[derive(Debug, Clone)]
struct CallG {
    name: usize,
    arguments: String,
}

#[derive(Debug, Clone)]
enum TurnG {
    User(ContentG),
    System(ContentG),
    Reminder(String),
    Assistant { content: ContentG, reasoning: Option<String> },
    /// An assistant tool-call turn, its results (in `order`, a permutation
    /// of the calls), and optionally a user message after them.
    ToolTurn { content: ContentG, reasoning: Option<String>, calls: Vec<CallG>, order: Vec<usize>, results: Vec<ContentG>, then_user: Option<ContentG> },
}

fn reasoning() -> BoxedStrategy<Option<String>> {
    prop_oneof![5 => prose().prop_map(Some), 1 => text().prop_map(Some), 2 => Just(None)].boxed()
}

fn tool_turn() -> BoxedStrategy<TurnG> {
    (1usize..4)
        .prop_flat_map(|n| {
            (
                assistant_content(true),
                reasoning(),
                prop::collection::vec((0usize..16, arguments()).prop_map(|(name, arguments)| CallG { name, arguments }), n),
                Just((0..n).collect::<Vec<usize>>()).prop_shuffle(),
                prop::collection::vec(rich_content(), n),
                prop_oneof![3 => Just(None), 1 => rich_content().prop_map(Some)],
            )
        })
        .prop_map(|(content, reasoning, calls, order, results, then_user)| TurnG::ToolTurn { content, reasoning, calls, order, results, then_user })
        .boxed()
}

fn turn() -> BoxedStrategy<TurnG> {
    prop_oneof![
        3 => rich_content().prop_map(TurnG::User),
        2 => (assistant_content(false), reasoning()).prop_map(|(content, reasoning)| TurnG::Assistant { content, reasoning }),
        4 => tool_turn(),
        1 => system_content().prop_map(TurnG::System),
        1 => prose().prop_map(TurnG::Reminder),
    ]
    .boxed()
}

#[derive(Debug, Clone)]
enum ToolChoiceG {
    Absent,
    Mode(&'static str),
    Named(usize),
}

#[derive(Debug, Clone)]
struct ConvG {
    system: Option<ContentG>,
    tools: Vec<ToolG>,
    /// Send `"tools": []` when there are none.
    empty_tools_field: bool,
    tool_choice: ToolChoiceG,
    thinking: Option<&'static str>,
    effort: Option<&'static str>,
    response_format: Option<&'static str>,
    /// Put the word "JSON" in the first user turn (json_object needs it).
    json_hint: bool,
    lead_user: Option<ContentG>,
    turns: Vec<TurnG>,
    /// Restart call ids at call_0 in every tool turn (clients do).
    per_turn_ids: bool,
}

fn conversation() -> BoxedStrategy<ConvG> {
    (
        (
            prop_oneof![3 => system_content().prop_map(Some), 1 => Just(None)],
            prop_oneof![1 => Just(Vec::new()), 3 => tools()],
            any::<bool>(),
            prop_oneof![
                6 => Just(ToolChoiceG::Absent),
                1 => Just(ToolChoiceG::Mode("auto")),
                1 => Just(ToolChoiceG::Mode("none")),
                1 => Just(ToolChoiceG::Mode("required")),
                1 => (0usize..6).prop_map(ToolChoiceG::Named),
            ],
            prop_oneof![5 => Just(None), 2 => Just(Some("enabled")), 2 => Just(Some("disabled")), 1 => Just(Some("adaptive"))],
            prop_oneof![
                5 => Just(None),
                1 => Just(Some("none")),
                1 => Just(Some("low")),
                1 => Just(Some("medium")),
                1 => Just(Some("high")),
                1 => Just(Some("max")),
            ],
        ),
        (
            prop_oneof![8 => Just(None), 1 => Just(Some("text")), 2 => Just(Some("json_object")), 1 => Just(Some("json_schema"))],
            prop::bool::weighted(0.8),
            prop_oneof![9 => rich_content().prop_map(Some), 1 => Just(None)],
            prop::collection::vec(turn(), 0..7),
            any::<bool>(),
        ),
    )
        .prop_map(|((system, tools, empty_tools_field, tool_choice, thinking, effort), (response_format, json_hint, lead_user, turns, per_turn_ids))| ConvG {
            system,
            tools,
            empty_tools_field,
            tool_choice,
            thinking,
            effort,
            response_format,
            json_hint,
            lead_user,
            turns,
            per_turn_ids,
        })
        .boxed()
}

impl ConvG {
    fn to_request(&self) -> Value {
        let mut msgs: Vec<Value> = Vec::new();
        if let Some(s) = &self.system {
            msgs.push(json!({"role": "system", "content": s.to_json()}));
        }
        if let Some(u) = &self.lead_user {
            let mut c = u.to_json();
            if self.json_hint && self.response_format == Some("json_object") {
                c = match c {
                    Value::String(s) => Value::String(format!("Answer in JSON. {s}")),
                    Value::Array(mut a) => {
                        a.insert(0, json!({"type": "text", "text": "Answer in JSON."}));
                        Value::Array(a)
                    }
                    other => other,
                };
            }
            msgs.push(json!({"role": "user", "content": c}));
        }
        let call_name = |k: usize| -> String {
            if self.tools.is_empty() {
                TOOL_NAMES[k % TOOL_NAMES.len()].to_string()
            } else {
                self.tools[k % self.tools.len()].name.clone()
            }
        };
        for (ti, t) in self.turns.iter().enumerate() {
            match t {
                TurnG::User(c) => msgs.push(json!({"role": "user", "content": c.to_json()})),
                TurnG::System(c) => msgs.push(json!({"role": "system", "content": c.to_json()})),
                TurnG::Reminder(s) => msgs.push(json!({"role": "latest_reminder", "content": s})),
                TurnG::Assistant { content, reasoning } => {
                    let mut m = json!({"role": "assistant", "content": content.to_json()});
                    if let Some(r) = reasoning {
                        m["reasoning_content"] = json!(r);
                    }
                    msgs.push(m);
                }
                TurnG::ToolTurn { content, reasoning, calls, order, results, then_user } => {
                    let id = |j: usize| if self.per_turn_ids { format!("call_{j}") } else { format!("call_{ti}_{j}") };
                    let tcs: Vec<Value> = calls
                        .iter()
                        .enumerate()
                        .map(|(j, c)| json!({"id": id(j), "type": "function", "function": {"name": call_name(c.name), "arguments": c.arguments}}))
                        .collect();
                    let mut m = json!({"role": "assistant", "content": content.to_json(), "tool_calls": tcs});
                    if let Some(r) = reasoning {
                        m["reasoning_content"] = json!(r);
                    }
                    msgs.push(m);
                    for &j in order {
                        msgs.push(json!({"role": "tool", "tool_call_id": id(j), "content": results[j].to_json()}));
                    }
                    if let Some(u) = then_user {
                        msgs.push(json!({"role": "user", "content": u.to_json()}));
                    }
                }
            }
        }
        if msgs.is_empty() {
            msgs.push(json!({"role": "user", "content": "hi"}));
        }
        let mut req = json!({"model": "deepseek-v4.1-flash", "messages": msgs});
        if !self.tools.is_empty() {
            req["tools"] = Value::Array(self.tools.iter().map(ToolG::to_json).collect());
        } else if self.empty_tools_field {
            req["tools"] = json!([]);
        }
        match &self.tool_choice {
            ToolChoiceG::Absent => {}
            ToolChoiceG::Mode(m) => req["tool_choice"] = json!(m),
            ToolChoiceG::Named(k) => {
                let name = self.tools.get(*k).map(|t| t.name.clone()).unwrap_or_else(|| "no_such_tool".into());
                req["tool_choice"] = json!({"type": "function", "function": {"name": name}});
            }
        }
        if let Some(t) = self.thinking {
            req["thinking"] = json!({"type": t});
        }
        if let Some(e) = self.effort {
            req["reasoning_effort"] = json!(e);
        }
        match self.response_format {
            None => {}
            Some("json_schema") => req["response_format"] = json!({"type": "json_schema", "json_schema": {"name": "x", "schema": {"type": "object"}}}),
            Some(t) => req["response_format"] = json!({"type": t}),
        }
        req
    }
}

fn run_prop<S: Strategy>(name: &str, strategy: S, f: impl Fn(&S::Value) -> Result<(), String>)
where
    S::Value: std::fmt::Debug,
{
    // Deterministic by default (CI reproduces a failure from the seed alone);
    // ORACLE_RANDOM=1 explores fresh cases. Failures persist either way.
    let mut runner = if std::env::var("ORACLE_RANDOM").is_ok() {
        TestRunner::new(config())
    } else {
        TestRunner::new_with_rng(config(), proptest::test_runner::TestRng::deterministic_rng(proptest::test_runner::RngAlgorithm::ChaCha))
    };
    let result = runner.run(&strategy, |v| f(&v).map_err(|e| TestCaseError::fail(e)));
    if let Err(e) = result {
        panic!("{name}: {e}");
    }
}

// ---------------------------------------------------------------------------
// Prompt properties
// ---------------------------------------------------------------------------

/// The main property: any conversation the generators produce renders to
/// the same bytes and token ids as DeepSeek's reference, or both reject it.
#[test]
fn prompt_matches_oracle() {
    run_prop("prompt sweep", conversation(), |conv| {
        let req = conv.to_request();
        check_prompt(&req).map_err(|e| format!("{e}\nrequest: {}", serde_json::to_string(&req).unwrap()))
    });
}

// ---------------------------------------------------------------------------
// Official vectors (encoding/tests of the 1017 checkpoint)
// ---------------------------------------------------------------------------

fn vector(i: u32) -> (Value, String) {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dsv41_1017_encoding");
    let input: Value = serde_json::from_str(&std::fs::read_to_string(format!("{dir}/test_input_{i}.json")).unwrap()).unwrap();
    let output = std::fs::read_to_string(format!("{dir}/test_output_{i}.txt")).unwrap();
    (input, output)
}

/// The vector as the OpenAI request a client would send. Mechanical only:
/// a bare message list becomes a case; `tools` move to the request; the
/// thinking mode becomes `thinking`; tool calls get the ids the vectors leave
/// out (the tool results then answer them, as the vectors' order implies);
/// image paths become data URLs (the oracle accepts only http(s)/base64 URLs;
/// the URL never reaches the prompt).
fn vector_request(input: &Value) -> Value {
    let case = if input.is_array() { json!({"messages": input}) } else { input.clone() };
    let mut msgs = case["messages"].as_array().unwrap().clone();
    let mut pending: Vec<String> = Vec::new();
    let mut n = 0;
    for m in msgs.iter_mut() {
        if let Some(tcs) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) {
            if tcs.is_empty() {
                m.as_object_mut().unwrap().remove("tool_calls");
            } else {
                pending.clear();
                for tc in tcs.iter_mut() {
                    if tc.get("id").is_none() {
                        tc["id"] = json!(format!("call_{n}"));
                        n += 1;
                    }
                    pending.push(tc["id"].as_str().unwrap().to_string());
                }
                pending.reverse();
            }
        }
        if m["role"] == "tool" {
            let id = pending.pop().unwrap_or_default();
            m["tool_call_id"] = json!(id);
        }
        if let Some(parts) = m.get_mut("content").and_then(|c| c.as_array_mut()) {
            for p in parts {
                if p["type"] == "image_url" {
                    p["image_url"]["url"] = json!("data:image/jpeg;base64,/9j/4AAQ");
                }
            }
        }
    }
    let mut req = json!({"model": "deepseek-v4.1-flash", "messages": msgs});
    if let Some(t) = case.get("tools") {
        req["tools"] = t.clone();
    }
    let thinking = case.get("thinking_mode").and_then(|t| t.as_str()) == Some("thinking");
    req["thinking"] = json!({"type": if thinking { "enabled" } else { "disabled" }});
    if let Some(e) = case.get("reasoning_effort").filter(|e| !e.is_null()) {
        req["reasoning_effort"] = e.clone();
    }
    req
}

/// The reference encoder renders no generation prompt after a final
/// assistant message; the oracle always ends with one (it renders the
/// request for the NEXT assistant turn). Strip it to compare with the golden.
fn strip_generation_prompt(prompt: &str) -> &str {
    prompt
        .strip_suffix("<\u{ff5c}Assistant\u{ff5c}><think>")
        .or_else(|| prompt.strip_suffix("<\u{ff5c}Assistant\u{ff5c}></think>"))
        .unwrap_or(prompt)
}

/// Vectors 1, 2 and 5 are expressible as OpenAI requests: the oracle and
/// the server must both reproduce the golden output (modulo the trailing
/// generation prompt when the case ends on an assistant message, and the
/// placeholder spelling).
#[test]
fn official_vectors_expressible_as_requests() {
    for i in [1, 2, 5] {
        let (input, golden) = vector(i);
        let req = vector_request(&input);
        let ends_on_assistant = req["messages"].as_array().unwrap().last().unwrap()["role"] == "assistant";
        let oracle = oracle_prompt(&req).unwrap_or_else(|e| panic!("vector {i}: oracle: {e}"));
        let oracle_cmp = if ends_on_assistant { strip_generation_prompt(&oracle) } else { oracle.as_str() };
        assert_eq!(oracle_cmp, golden, "vector {i}: deepseek-recipe vs encoding.py golden: {}", diff(&golden, oracle_cmp));
        let ours = ours_prompt(&req).unwrap_or_else(|e| panic!("vector {i}: ours: {e}"));
        assert_eq!(ours, oracle, "vector {i}: ours vs deepseek-recipe: {}", diff(&oracle, &ours));
        check_ids(&req, &oracle).unwrap_or_else(|e| panic!("vector {i}: {e}"));
    }
}

/// Vectors 3 and 4 use reference-encoder features no OpenAI request can
/// express: tools attached to a MID-conversation system message (3; a
/// request's `tools` always attach to the first system message) and the
/// internal `task` field (4, `<｜action｜>`). Pinned here so the reason is
/// explicit, not a silent skip: the oracle renders the request form, and
/// ours must match the oracle; neither can match the golden.
#[test]
fn official_vectors_not_expressible_as_requests() {
    for i in [3, 4] {
        let (input, golden) = vector(i);
        let req = vector_request(&input);
        let oracle = oracle_prompt(&req).unwrap_or_else(|e| panic!("vector {i}: oracle: {e}"));
        assert_ne!(strip_generation_prompt(&oracle), golden, "vector {i} became expressible; move it to the test above");
        let ours = ours_prompt(&req).unwrap_or_else(|e| panic!("vector {i}: ours: {e}"));
        assert_eq!(ours, oracle, "vector {i}: ours vs deepseek-recipe: {}", diff(&oracle, &ours));
    }
}

// ---------------------------------------------------------------------------
// Secondary oracle: the checkpoint's encoding.py vs deepseek-recipe
// ---------------------------------------------------------------------------

/// Map an OpenAI request to `encoding.encode_messages` arguments the way the
/// checkpoint's README / `load_cases` do: tools on messages[0], thinking
/// mode, effort name. `drop_thinking` is set by the caller.
fn encoding_py_case(req: &Value, drop_thinking: bool) -> Option<Value> {
    let o: OracleRequest = serde_json::from_value(req.clone()).ok()?;
    let conv = o.convert(ConversionOptions::default()).ok()?.conversation;
    let mut msgs = req["messages"].as_array()?.clone();
    if conv.tool_choice != deepseek_recipe_core::tools::ToolChoice::Auto || req.get("response_format").is_some() {
        return None; // request-level features encoding.py has no input for
    }
    if let Some(t) = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())) {
        msgs[0]["tools"] = t.clone();
    }
    let effort = conv.reasoning_effort.map(|e| match e {
        deepseek_recipe_core::conversation::ReasoningEffort::Low => "low",
        deepseek_recipe_core::conversation::ReasoningEffort::Max => "max",
        _ => "high",
    });
    Some(json!({
        "messages": msgs,
        "thinking_mode": if conv.thinking_mode { "thinking" } else { "chat" },
        "reasoning_effort": effort,
        "drop_thinking": drop_thinking,
    }))
}

/// Disagreement classes between the two references, found by this test (see
/// the report in the commit message). Each is a predicate on the request; a
/// difference no class explains fails the test.
fn classify_oracle_disagreement(req: &Value, recipe: &str, py: &str) -> Option<&'static str> {
    let msgs = req["messages"].as_array().unwrap();
    let last_role = msgs.last().map(|m| m["role"].as_str().unwrap_or("")).unwrap_or("");
    // 1. Generation prompt after a final assistant / first-system / reminder message.
    if !matches!(last_role, "user" | "tool") && strip_generation_prompt(recipe) == py {
        return Some("GEN_PROMPT_ALWAYS");
    }
    let text_of = |c: &Value| -> String {
        match c {
            Value::String(s) => s.clone(),
            Value::Array(a) => a.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n\n"),
            _ => String::new(),
        }
    };
    if msgs.iter().any(|m| m["role"] == "system" && text_of(&m["content"]).is_empty()) {
        return Some("EMPTY_SYSTEM_DROPPED");
    }
    let has_tools = req.get("tools").and_then(|t| t.as_array()).is_some_and(|a| !a.is_empty());
    if has_tools && msgs.first().is_some_and(|m| m["role"] != "system") {
        return Some("TOOLS_WITHOUT_SYSTEM");
    }
    if has_tools {
        let normalized = req["tools"].as_array().unwrap().iter().all(|t| {
            let f = t["function"].as_object().unwrap();
            let keys: Vec<&str> = f.keys().map(|k| k.as_str()).collect();
            keys == ["name", "description", "parameters"]
        });
        if !normalized {
            return Some("TOOL_SCHEMA_NORMALIZED");
        }
    }
    let floats_big = recipe.contains("e1") || recipe.contains("e2") || py.contains("e+");
    if floats_big {
        return Some("FLOAT_POSITIVE_EXPONENT");
    }
    let double_encoded = msgs.iter().filter_map(|m| m["tool_calls"].as_array()).flatten().any(|tc| {
        let a = tc["function"]["arguments"].as_str().unwrap_or("");
        serde_json::from_str::<Value>(a).is_ok_and(|v| v.is_string())
    });
    if double_encoded {
        return Some("DOUBLE_ENCODED_ARGUMENTS");
    }
    if req["messages"].as_array().unwrap().iter().any(|m| m["role"] == "latest_reminder") {
        return Some("LATEST_REMINDER_TRANSITION");
    }
    // encoding.py emits `<｜Assistant｜>` only as the transition after a user
    // (or mid-conversation system) message; the recipe renders it on every
    // assistant message (first message, after another assistant, after the
    // leading system message).
    let header_only_after_user = msgs.iter().enumerate().any(|(i, m)| {
        m["role"] == "assistant" && (i == 0 || matches!(msgs[i - 1]["role"].as_str(), Some("assistant")) || (i == 1 && msgs[0]["role"] == "system"))
    });
    if header_only_after_user {
        return Some("ASSISTANT_HEADER_ONLY_AFTER_USER");
    }
    // A user message with `content: []` merges into the previous user turn
    // as nothing in encoding.py, as "" joined by "\n\n" in the recipe.
    if msgs.iter().any(|m| m["role"] == "user" && m["content"].as_array().is_some_and(|a| a.is_empty())) {
        return Some("EMPTY_PARTS_USER_JOIN");
    }
    None
}

#[test]
fn secondary_oracle_encoding_py_vs_recipe() {
    let dir = std::env::var("ENCODING_PY_DIR").unwrap_or_else(|_| "/weights2/dsv4.1f-1017/encoding".into());
    if !std::path::Path::new(&dir).join("encoding.py").exists() {
        eprintln!("secondary oracle: {dir}/encoding.py not found (set ENCODING_PY_DIR); SKIPPED");
        return;
    }
    let driver = concat!(env!("CARGO_MANIFEST_DIR"), "/../../scripts/v41_oracle/run_encoding_py.py");
    let n: usize = std::env::var("ENCODING_PY_CASES").ok().and_then(|s| s.parse().ok()).unwrap_or(2000);
    let mut runner = TestRunner::new_with_rng(Config::default(), proptest::test_runner::TestRng::deterministic_rng(proptest::test_runner::RngAlgorithm::ChaCha));
    let strategy = conversation();
    let mut reqs = Vec::new();
    let mut lines = String::new();
    for _ in 0..n {
        let req = strategy.new_tree(&mut runner).unwrap().current().to_request();
        let Ok(recipe) = oracle_prompt(&req) else { continue };
        for drop in [false, true] {
            let Some(case) = encoding_py_case(&req, drop) else { continue };
            lines.push_str(&case.to_string());
            lines.push('\n');
            reqs.push((req.clone(), recipe.clone(), drop));
        }
    }
    let mut child = std::process::Command::new("python3")
        .args(["-I", driver, &dir])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .expect("python3");
    {
        use std::io::Write;
        let mut stdin = child.stdin.take().unwrap();
        let input = lines.clone();
        std::thread::spawn(move || stdin.write_all(input.as_bytes()).unwrap());
    }
    let out = child.wait_with_output().unwrap();
    let outs: Vec<Value> = String::from_utf8(out.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(outs.len(), reqs.len(), "driver output count");
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut unexplained = Vec::new();
    for ((req, recipe, drop), out) in reqs.iter().zip(outs) {
        let key = |k: &str| format!("drop_thinking={drop}: {k}");
        let Some(py) = out["ok"].as_str() else {
            let why = out["err"].as_str().unwrap_or("").split(':').next().unwrap_or("").to_string();
            let what = if why == "ValueError" && out["err"].as_str().unwrap_or("").contains("image") { "image placeholder text" } else { "other" };
            *counts.entry(key(&format!("encoding.py rejects ({why}: {what}), recipe accepts"))).or_default() += 1;
            if what == "other" && !*drop {
                eprintln!("encoding.py rejects: {} -- {}", out["err"], serde_json::to_string(req).unwrap().chars().take(300).collect::<String>());
            }
            continue;
        };
        let py = py.to_string();
        if *recipe == py {
            *counts.entry(key("identical")).or_default() += 1;
            continue;
        }
        if *drop {
            // drop_thinking=True is encoding.py's DEFAULT; the recipe never drops.
            *counts.entry(key("differs (DROP_THINKING and/or the classes below)")).or_default() += 1;
            continue;
        }
        match classify_oracle_disagreement(req, recipe, &py) {
            Some(c) => *counts.entry(key(c)).or_default() += 1,
            None => unexplained.push(format!("{}\n  recipe vs encoding.py: {}", serde_json::to_string(req).unwrap(), diff(recipe, &py))),
        }
    }
    eprintln!("secondary oracle (encoding.py vs deepseek-recipe), {} generated cases:", reqs.len());
    for (k, v) in &counts {
        eprintln!("  {v:6}  {k}");
    }
    for u in unexplained.iter().take(5) {
        eprintln!("UNEXPLAINED: {u}");
    }
    assert!(unexplained.is_empty(), "{} unexplained encoding.py vs deepseek-recipe differences", unexplained.len());
}

// ---------------------------------------------------------------------------
// Output parsing
// ---------------------------------------------------------------------------

/// One piece of a model output: text (any tokenization), or a special token.
#[derive(Debug, Clone, PartialEq)]
enum Piece {
    T(String),
    ThinkEnd,
    Dsml,
    Eos,
}

const DSML_ID: i32 = 128825;
const DSML_TEXT: &str = "\u{ff5c}DSML\u{ff5c}";

#[derive(Debug, Clone)]
enum ParamG {
    Str(String),
    Json(Value),
}

#[derive(Debug, Clone)]
struct OCallG {
    name: usize,
    params: Vec<(usize, ParamG)>,
}

#[derive(Debug, Clone)]
struct OutputG {
    thinking: bool,
    reasoning: String,
    /// Model closes `</think>` (thinking mode).
    close_think: bool,
    content: String,
    calls: Vec<OCallG>,
    /// Ends with EOS (else the token cap cut it).
    eos: bool,
    stop: Vec<String>,
    /// Random byte split points for the server's tokens, char split points
    /// for the oracle's text chunks.
    seed: u64,
}

/// Output text that both parsers read as plain text: no markup spelling
/// (the server's scanner is token-driven and ignores look-alike TEXT, the
/// oracle's is text-driven -- KNOWN_DEVIATIONS TOKEN_DRIVEN_SCANNER).
fn out_text() -> BoxedStrategy<String> {
    text()
        .prop_map(|s| {
            let mut s = s;
            for bad in ["</think>", "<think>", DSML_TEXT, "\u{ff5c}", "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>"] {
                s = s.replace(bad, "");
            }
            s
        })
        .boxed()
}

fn param_value() -> BoxedStrategy<ParamG> {
    prop_oneof![3 => out_text().prop_map(ParamG::Str), 2 => json_value().prop_filter("no markup", |v| !v.to_string().contains('\u{ff5c}') && !v.is_string()).prop_map(ParamG::Json)].boxed()
}

fn output() -> BoxedStrategy<OutputG> {
    let call = (0usize..TOOL_NAMES.len(), prop::collection::vec((0usize..KEYS.len(), param_value()), 0..4)).prop_map(|(name, params)| {
        // DSML parameter names are unique within a call.
        let mut seen = std::collections::HashSet::new();
        let params = params.into_iter().filter(|(k, _)| seen.insert(*k)).collect();
        OCallG { name, params }
    });
    (
        any::<bool>(),
        out_text(),
        prop::bool::weighted(0.9),
        (out_text(), prop_oneof![3 => Just(""), 1 => Just("\n"), 1 => Just("\n\n"), 1 => Just("\n\n\n")]).prop_map(|(a, b)| format!("{b}{a}")),
        prop_oneof![2 => Just(Vec::new()), 2 => prop::collection::vec(call, 1..4)],
        prop::bool::weighted(0.9),
        prop_oneof![4 => Just(Vec::new()), 1 => prop::collection::vec("[a-z]{1,3}|\n|\\.", 1..3)],
        any::<u64>(),
    )
        .prop_map(|(thinking, reasoning, close_think, content, calls, eos, stop, seed)| OutputG { thinking, reasoning, close_think, content, calls, eos, stop, seed })
        .boxed()
}

impl OutputG {
    fn pieces(&self) -> Vec<Piece> {
        let mut p = Vec::new();
        if self.thinking {
            p.push(Piece::T(self.reasoning.clone()));
            if self.close_think {
                p.push(Piece::ThinkEnd);
            }
        }
        p.push(Piece::T(self.content.clone()));
        if !self.calls.is_empty() {
            p.push(Piece::T("\n\n<".into()));
            p.push(Piece::Dsml);
            p.push(Piece::T(" calls>\n".into()));
            for (i, c) in self.calls.iter().enumerate() {
                if i > 0 {
                    p.push(Piece::T("\n".into()));
                }
                p.push(Piece::T("<".into()));
                p.push(Piece::Dsml);
                p.push(Piece::T(format!(" invoke name=\"{}\">\n", TOOL_NAMES[c.name])));
                for (j, (k, v)) in c.params.iter().enumerate() {
                    if j > 0 {
                        p.push(Piece::T("\n".into()));
                    }
                    let (is_str, body) = match v {
                        ParamG::Str(s) => ("true", s.clone()),
                        ParamG::Json(v) => ("false", deepstrix_server::dsml::to_json_hf(v)),
                    };
                    p.push(Piece::T("<".into()));
                    p.push(Piece::Dsml);
                    p.push(Piece::T(format!(" parameter name=\"{}\" string=\"{is_str}\">{body}</", KEYS[*k])));
                    p.push(Piece::Dsml);
                    p.push(Piece::T(" parameter>".into()));
                }
                p.push(Piece::T("\n</".into()));
                p.push(Piece::Dsml);
                p.push(Piece::T(" invoke>".into()));
            }
            p.push(Piece::T("\n</".into()));
            p.push(Piece::Dsml);
            p.push(Piece::T(" calls>".into()));
        }
        if self.eos {
            p.push(Piece::Eos);
        }
        p
    }

    /// The request whose response this output is (the oracle's parsing
    /// options come from converting it).
    fn request(&self) -> Value {
        let tools: Vec<Value> = TOOL_NAMES
            .iter()
            .map(|n| json!({"type": "function", "function": {"name": n, "description": "", "parameters": {"type": "object", "properties": {}}}}))
            .collect();
        let mut r = json!({"model": "m", "messages": [{"role": "user", "content": "q"}], "tools": tools,
                           "thinking": {"type": if self.thinking { "enabled" } else { "disabled" }}});
        if !self.stop.is_empty() {
            r["stop"] = json!(self.stop);
        }
        r
    }
}

/// A parsed assistant turn, normalized for comparison: empty content /
/// reasoning == absent (KNOWN_DEVIATIONS EMPTY_FIELD_NULL), tool-call
/// arguments as parsed JSON (raw string when they do not parse).
#[derive(Debug, Clone, PartialEq)]
struct Parsed {
    content: String,
    reasoning: String,
    tool_calls: Vec<(String, Value)>,
    finish: String,
}

fn args_value(s: &str) -> Value {
    serde_json::from_str(s).unwrap_or_else(|_| Value::String(format!("<unparsed> {s}")))
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| tokio::runtime::Builder::new_current_thread().enable_all().build().unwrap())
}

fn oracle_parse(o: &OutputG, chunked: bool) -> Result<Parsed, String> {
    let req: OracleRequest = serde_json::from_value(o.request()).map_err(|e| e.to_string())?;
    let conv = req.convert(ConversionOptions::default()).map_err(|e| e.to_string())?;
    let generator = OracleRequest::chunk_generator(&conv, "id".into(), "m".into());
    let processor = StreamProcessor::new(generator, conv.parsing_options);
    // The backend decodes special tokens to their text and stops at EOS.
    let mut text = String::new();
    for p in o.pieces() {
        match p {
            Piece::T(t) => text.push_str(&t),
            Piece::ThinkEnd => text.push_str("</think>"),
            Piece::Dsml => text.push_str(DSML_TEXT),
            Piece::Eos => break,
        }
    }
    let mut chunks: Vec<InferenceChunk> = Vec::new();
    if chunked {
        let mut rng = Rng(o.seed | 1);
        let mut rest = text.as_str();
        while !rest.is_empty() {
            let mut cut = 1 + rng.below(rest.len().min(12));
            while !rest.is_char_boundary(cut) {
                cut += 1;
            }
            chunks.push(InferenceChunk::Text { content: rest[..cut].to_string(), content_tokens: 1 });
            rest = &rest[cut..];
        }
    } else if !text.is_empty() {
        chunks.push(InferenceChunk::Text { content: text.clone(), content_tokens: 1 });
    }
    chunks.push(InferenceChunk::Finish { finish_reason: if o.eos { InferenceFinishReason::Stop } else { InferenceFinishReason::Length } });
    let stream = processor.process(tokio_stream_iter(chunks));
    let mut resp = deepseek_recipe::openai::chat_completion::response::ChatCompletionResponse::new("id".into(), "m".into(), 0, 0, 0);
    runtime().block_on(async {
        use futures_util::StreamExt;
        let mut stream = std::pin::pin!(stream);
        while let Some(c) = stream.next().await {
            resp.append(c.map_err(|e| e.to_string())?);
        }
        Ok::<(), String>(())
    })?;
    let ch = resp.choices.into_iter().next().ok_or("no choice")?;
    Ok(Parsed {
        content: ch.message.content.unwrap_or_default(),
        reasoning: ch.message.reasoning_content.unwrap_or_default(),
        tool_calls: ch.message.tool_calls.unwrap_or_default().into_iter().map(|t| (t.function.name, args_value(&t.function.arguments))).collect(),
        finish: serde_json::to_value(ch.finish_reason).unwrap().as_str().unwrap_or("").to_string(),
    })
}

fn tokio_stream_iter(chunks: Vec<InferenceChunk>) -> impl futures_util::Stream<Item = InferenceChunk> + Send {
    futures_util::stream::iter(chunks)
}

/// The worker's event stream for an output: what `engine_worker` /
/// `multistream` send (TOK_THINK_* suppressed, `reasoning` = in_think, EOS
/// ends the turn with Stop, a missing EOS = the token cap). Text pieces are
/// split at random BYTE positions (tokens can end mid-character).
fn worker_events(o: &OutputG, rng: &mut Rng) -> Vec<WorkerEvent> {
    let mut evs = Vec::new();
    let mut in_think = o.thinking;
    let mut n = 0u32;
    let mut finish = FinishReason::Length;
    for p in o.pieces() {
        match p {
            Piece::T(t) => {
                let b = t.as_bytes();
                let mut i = 0;
                while i < b.len() {
                    let len = 1 + rng.below((b.len() - i).min(6));
                    evs.push(WorkerEvent::Chunk { token_id: 1000 + rng.below(1000) as i32, bytes: b[i..i + len].to_vec(), reasoning: in_think });
                    n += 1;
                    i += len;
                }
            }
            Piece::ThinkEnd => {
                in_think = false;
                n += 1;
            }
            Piece::Dsml => {
                evs.push(WorkerEvent::Chunk { token_id: DSML_ID, bytes: DSML_TEXT.as_bytes().to_vec(), reasoning: in_think });
                n += 1;
            }
            Piece::Eos => {
                finish = FinishReason::Stop;
                n += 1;
                break;
            }
        }
    }
    evs.push(WorkerEvent::Done { prompt_tokens: 1, completion_tokens: n, finish });
    evs
}

fn ours_parse_nonstream(o: &OutputG) -> Parsed {
    let mut rng = Rng(o.seed.rotate_left(17) | 1);
    let evs = worker_events(o, &mut rng);
    let result = runtime().block_on(async {
        let (tx, rx) = tokio::sync::mpsc::channel(evs.len() + 1);
        for e in evs {
            tx.send(e).await.unwrap();
        }
        drop(tx);
        accumulate(rx, Some(DSML_ID), o.stop.clone()).await.unwrap()
    });
    let (finish, msg) = deepstrix_server::openai::handler::nonstream_message(result);
    Parsed {
        content: msg.content.unwrap_or_default(),
        reasoning: msg.reasoning_content.unwrap_or_default(),
        tool_calls: msg.tool_calls.into_iter().map(|t| (t.function.name, args_value(&t.function.arguments))).collect(),
        finish: finish.to_string(),
    }
}

fn ours_parse_stream(o: &OutputG) -> Parsed {
    let mut rng = Rng(o.seed.rotate_left(29) | 1);
    let evs = worker_events(o, &mut rng);
    let chunks = runtime().block_on(deepstrix_server::openai::handler::sse_transcript(evs, Some(DSML_ID), o.stop.clone()));
    let mut p = Parsed { content: String::new(), reasoning: String::new(), tool_calls: Vec::new(), finish: String::new() };
    let mut args: Vec<String> = Vec::new();
    for c in chunks {
        let ch = &c["choices"][0];
        let d = &ch["delta"];
        if let Some(s) = d["content"].as_str() {
            p.content.push_str(s);
        }
        if let Some(s) = d["reasoning_content"].as_str() {
            p.reasoning.push_str(s);
        }
        if let Some(tcs) = d["tool_calls"].as_array() {
            for tc in tcs {
                let i = tc["index"].as_u64().unwrap() as usize;
                if let Some(name) = tc["function"]["name"].as_str() {
                    p.tool_calls.push((name.to_string(), Value::Null));
                    args.push(String::new());
                }
                if let Some(a) = tc["function"]["arguments"].as_str() {
                    args[i].push_str(a);
                }
            }
        }
        if let Some(f) = ch["finish_reason"].as_str() {
            p.finish = f.to_string();
        }
    }
    for (tc, a) in p.tool_calls.iter_mut().zip(args) {
        tc.1 = args_value(&a);
    }
    p
}

fn check_output(o: &OutputG) -> Result<(), String> {
    let whole = oracle_parse(o, false)?;
    let chunked = oracle_parse(o, true)?;
    if whole != chunked {
        return Err(format!("oracle is not chunking-invariant: whole {whole:?} vs chunked {chunked:?}"));
    }
    let ns = ours_parse_nonstream(o);
    if ns != whole {
        return Err(format!("non-streaming: oracle {whole:?}\n   ours {ns:?}\n pieces {:?}", o.pieces()));
    }
    let st = ours_parse_stream(o);
    if st != whole {
        return Err(format!("streaming: oracle {whole:?}\n   ours {st:?}\n pieces {:?}", o.pieces()));
    }
    Ok(())
}

/// Well-formed model outputs (reasoning, content, DSML tool calls with
/// string and JSON parameters, stop strings, EOS or the token cap outside a
/// call) parse to the same message on both sides, streamed or not.
#[test]
fn output_parse_matches_oracle() {
    run_prop("output parse", output().prop_filter("cut outside a call", |o| o.eos || o.calls.is_empty()), check_output);
}

// ---------------------------------------------------------------------------
// Named deviations: each pins the server's current, deliberate behavior
// where it is NOT identical to the oracle. Changing one is a decision.
// ---------------------------------------------------------------------------

/// Every known, deliberate difference from deepseek-recipe (id, summary).
pub const KNOWN_DEVIATIONS: &[(&str, &str)] = &[
    ("IMAGE_PLACEHOLDER_SPELLING", "prompt text spells id 129264 <｜deepseek_image｜> (checkpoint tokenizer.json), the oracle <｜image｜> (its bundled tokenizer); same token id"),
    ("CLIENT_SPECIALS_INERT", "client text spelling a special token (<｜User｜>, </think>, ｜DSML｜ ...) is BPE'd as text; the oracle's HF tokenization makes it the control token. Prompt bytes identical, ids differ"),
    ("TOKEN_IDS_VIA_OUR_BPE", "the id leg tokenizes the oracle prompt HF-style with the server's BPE; the oracle's own HF tokenizer object needs a direct `tokenizers` dev-dependency"),
    ("TOKEN_DRIVEN_SCANNER", "output parsing keys on the DSML/think special TOKENS; markup spelled as plain text by the model is content to us, markup to the text-driven oracle"),
    ("EMPTY_FIELD_NULL", "an empty content/reasoning_content is null in our response, \"\" in the oracle's"),
];

#[test]
fn deviation_client_special_text_is_inert() {
    let Some(v) = vocab() else { return };
    let req = json!({"model": "m", "messages": [{"role": "user", "content": "a <｜User｜> b </think>"}], "thinking": {"type": "disabled"}});
    let oracle = oracle_prompt(&req).unwrap();
    assert_eq!(ours_prompt(&req).unwrap(), oracle, "bytes identical");
    let r = ours_request(&req).unwrap();
    let (ours, _) = prompt_v41::render_request_v41(&v.bpe, &r, server_default_effort(), v.image_id).unwrap();
    let reference = hf_style_ids(v, &oracle);
    let user_id = v.bpe.lookup_token_id("<\u{ff5c}User\u{ff5c}>").unwrap();
    assert_eq!(reference.iter().filter(|&&t| t == user_id).count(), 2, "HF makes the client's <｜User｜> a control token");
    assert_eq!(ours.iter().filter(|&&t| t == user_id).count(), 1, "ours keeps it text");
}
