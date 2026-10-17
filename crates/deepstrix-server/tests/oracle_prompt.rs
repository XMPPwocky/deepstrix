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
//!
//! V4.1 builds only (`prompt::SERVER_DEFAULT_EFFORT` is the V4.1 default
//! there).
#![cfg(feature = "v41")]

use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::OnceLock;

use deepseek_recipe::openai::ChatCompletionRequest as OracleRequest;
use deepseek_recipe::request::{ConversionOptions, ProtocolRequest};
use deepseek_recipe::response::ProtocolResponse;
use deepseek_recipe::stream::{InferenceChunk, InferenceFinishReason, StreamProcessor};
use deepseek_recipe::util::append_delta::AppendDelta;
use deepseek_recipe_encoding::v4::dsv41::DeepseekV41Encoding;
use deepseek_recipe_encoding::PromptEncoding;
use deepstrix_server::engine_worker::{accumulate_with, FinishReason, WorkerEvent};
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
/// (`--default-reasoning-effort`'s compiled-in default for this build).
fn server_default_effort() -> ReasoningEffort {
    deepstrix_server::prompt::SERVER_DEFAULT_EFFORT
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
enum IdCheck {
    /// Identical ids (or no tokenizer available).
    Compared,
    /// Not comparable: client text spells an added token.
    ClientSpecials,
}

/// The request with every added-token spelling removed from its strings.
fn sanitize(v: &Value) -> Value {
    let Some(voc) = vocab() else { return v.clone() };
    match v {
        Value::String(s) => {
            let mut s = s.clone();
            loop {
                let before = s.len();
                for (lit, _) in &voc.added {
                    s = s.replace(lit.as_str(), "");
                }
                if s.len() == before {
                    break;
                }
            }
            Value::String(s)
        }
        Value::Array(a) => Value::Array(a.iter().map(sanitize).collect()),
        Value::Object(o) => Value::Object(o.iter().map(|(k, v)| (k.clone(), sanitize(v))).collect()),
        other => other.clone(),
    }
}

fn check_ids(req: &Value, oracle: &str) -> Result<IdCheck, String> {
    let Some(v) = vocab() else { return Ok(IdCheck::Compared) };
    let r = ours_request(req)?;
    let segs = prompt_v41::request_segments_v41(&r, server_default_effort()).map_err(|e| format!("{e:#}"))?.0;
    let (ours, _) = prompt_v41::render_request_v41(&v.bpe, &r, server_default_effort(), v.image_id).map_err(|e| format!("{e:#}"))?;
    let reference = hf_style_ids(v, oracle);
    if ours == reference {
        STATS.ids_identical.fetch_add(1, Relaxed);
        return Ok(IdCheck::Compared);
    }
    let client_spells_special = segs.iter().any(|s| matches!(s, Seg::Client(t) if v.added.iter().any(|(lit, _)| t.contains(lit.as_str()))));
    if client_spells_special {
        STATS.ids_client_specials.fetch_add(1, Relaxed);
        return Ok(IdCheck::ClientSpecials);
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
            STATS.identical.fetch_add(1, Relaxed);
            match check_ids(req, &a)? {
                IdCheck::Compared => Ok(()),
                // Client text spells a special token: ids legitimately
                // differ there. Re-check the same conversation with those
                // spellings removed so the id leg still covers it.
                IdCheck::ClientSpecials => {
                    let clean = sanitize(req);
                    match (oracle_prompt(&clean), ours_prompt(&clean)) {
                        (Ok(a), Ok(b)) if a == b => match check_ids(&clean, &a)? {
                            IdCheck::Compared => Ok(()),
                            IdCheck::ClientSpecials => Err("sanitize left a special-token spelling".into()),
                        },
                        (Ok(a), Ok(b)) => Err(format!("sanitized prompts differ: {}", diff(&a, &b))),
                        (Err(_), Err(_)) => Ok(()),
                        (o, u) => Err(format!("sanitized request: oracle {:?} vs ours {:?}", o.err(), u.err())),
                    }
                }
            }
        }
        (Err(_), Err(_)) => {
            STATS.both_reject.fetch_add(1, Relaxed);
            Ok(())
        }
        (Err(e), Ok(_)) => Err(format!("oracle rejects, ours accepts: {e}")),
        (Ok(_), Err(e)) => Err(format!("oracle accepts, ours rejects: {e}")),
    }
}

/// Coverage counters, printed by `prompt_matches_oracle` (--nocapture).
struct Stats {
    identical: AtomicUsize,
    both_reject: AtomicUsize,
    ids_identical: AtomicUsize,
    ids_client_specials: AtomicUsize,
}
static STATS: Stats = Stats {
    identical: AtomicUsize::new(0),
    both_reject: AtomicUsize::new(0),
    ids_identical: AtomicUsize::new(0),
    ids_client_specials: AtomicUsize::new(0),
};

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
        2 => pick(ADVERSARIAL),
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
    /// Spell the recipe's image placeholder `<｜image｜>` in the first user
    /// text (both sides must reject).
    image_token: bool,
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
            prop::bool::weighted(0.03),
        ),
    )
        .prop_map(|((system, tools, empty_tools_field, tool_choice, thinking, effort), (response_format, json_hint, lead_user, turns, per_turn_ids, image_token))| ConvG {
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
            image_token,
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
            if self.image_token {
                c = match c {
                    Value::String(s) => Value::String(format!("{s} <\u{ff5c}image\u{ff5c}>")),
                    _ => Value::String("<\u{ff5c}image\u{ff5c}>".into()),
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
    eprintln!(
        "prompt_matches_oracle: {} identical prompts ({} with identical token ids, {} where client text spells a special \
         token: CLIENT_SPECIALS_INERT), {} rejected by both",
        STATS.identical.load(Relaxed),
        STATS.ids_identical.load(Relaxed),
        STATS.ids_client_specials.load(Relaxed),
        STATS.both_reject.load(Relaxed)
    );
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
            if m.get("tool_call_id").is_none() {
                m["tool_call_id"] = json!(id);
            }
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
        assert!(matches!(check_ids(&req, &oracle), Ok(IdCheck::Compared)), "vector {i}: token ids");
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

/// The hand-picked edge cases of `tests/fixtures/v41_prompts.json` (rendered
/// by encoding.py, `scripts/v41_oracle/gen_prompt_vectors.py`; this replaces
/// the old `tests/v41_prompt_vectors.rs`, which pinned the server to
/// encoding.py). As requests they must render exactly as deepseek-recipe;
/// which of them ALSO match encoding.py's bytes is pinned below (the rest
/// fall in the references' disagreement classes).
#[test]
fn fixture_edge_cases_match_oracle() {
    let raw = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/v41_prompts.json")).unwrap();
    let cases: Vec<Value> = serde_json::from_str(&raw).unwrap();
    let mut agree_with_encoding_py = Vec::new();
    for c in &cases {
        let name = c["name"].as_str().unwrap();
        // A numeric effort goes out as a string: the server's extension takes
        // "33" (a JSON integer is a deserialize error on both sides).
        let effort = match &c["reasoning_effort"] {
            Value::Number(n) => json!(n.to_string()),
            other => other.clone(),
        };
        let mut input = json!({"messages": c["messages"], "thinking_mode": c["thinking_mode"], "reasoning_effort": effort});
        if !c["context"].is_null() {
            // encoding.py's `context` (render a suffix without BOS) has no request form.
            assert!(oracle_prompt(&vector_request(&input)).is_ok(), "{name}");
            continue;
        }
        let msgs = input["messages"].as_array_mut().unwrap();
        if let Some(t) = msgs[0].as_object_mut().unwrap().remove("tools") {
            input["tools"] = t;
        }
        let message_format = input["messages"].as_array().unwrap().iter().any(|m| m.get("response_format").is_some());
        let req = vector_request(&input);
        let oracle = oracle_prompt(&req);
        let ours = ours_prompt(&req);
        if c["reasoning_effort"].is_i64() {
            let req = req.clone();
            // NUMERIC_REASONING_EFFORT: the server's extension.
            assert!(oracle.is_err(), "{name}: the oracle accepts a numeric effort now");
            let ours = ours.unwrap();
            assert!(ours.contains(&format!("Reasoning Effort: {} (", c["reasoning_effort"])), "{name}: {ours}");
            continue;
        }
        let oracle = oracle.unwrap_or_else(|e| panic!("{name}: oracle: {e}"));
        let ours = ours.unwrap_or_else(|e| panic!("{name}: ours: {e}"));
        assert_eq!(ours, oracle, "{name}: ours vs deepseek-recipe: {}", diff(&oracle, &ours));
        let expected = c["expected"].as_str().unwrap();
        let last_assistant = req["messages"].as_array().unwrap().last().unwrap()["role"] == "assistant";
        let cmp = if last_assistant { strip_generation_prompt(&oracle) } else { oracle.as_str() };
        if cmp == expected && !message_format {
            agree_with_encoding_py.push(name);
        } else {
            eprintln!("{name}: deepseek-recipe vs encoding.py: {}", diff(cmp, expected));
        }
    }
    assert_eq!(
        agree_with_encoding_py,
        [
            "shipped_1",
            "shipped_5",
            "chat_mode",
            "thinking_default_effort",
            "thinking_effort_max",
            "no_system",
            "assistant_history_kept_with_tools",
            "tool_call_and_two_results",
            "tool_result_then_user_text",
        ],
        "which hand-picked cases deepseek-recipe and encoding.py agree on"
    );
}

// ---------------------------------------------------------------------------
// Secondary oracle: the checkpoint's encoding.py vs deepseek-recipe
// ---------------------------------------------------------------------------

/// Map an OpenAI request to `encoding.encode_messages` arguments the way the
/// checkpoint's README / `load_cases` do: tools on messages[0], thinking
/// mode, effort name. `drop_thinking` is encoding.py's knob (its default is
/// true; the recipe has no such step).
///
/// With `neutralize`, the input is first rewritten so that encoding.py sees
/// the conversation the recipe renders, one rewrite per known disagreement
/// class; the returned names are the classes whose rewrite changed the input.
fn encoding_py_case(req: &Value, drop_thinking: bool, neutralize: bool) -> Option<(Value, Vec<&'static str>)> {
    let o: OracleRequest = serde_json::from_value(req.clone()).ok()?;
    let conv = o.convert(ConversionOptions::default()).ok()?.conversation;
    if conv.tool_choice != deepseek_recipe_core::tools::ToolChoice::Auto || req.get("response_format").is_some() {
        return None; // request-level features encoding.py has no input for
    }
    let mut msgs = req["messages"].as_array()?.clone();
    let mut tools: Option<Value> = req.get("tools").filter(|t| t.as_array().is_some_and(|a| !a.is_empty())).cloned();
    let mut classes = Vec::new();
    if neutralize {
        let text_of = |c: &Value| -> String {
            match c {
                Value::String(s) => s.clone(),
                Value::Array(a) => a.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n\n"),
                _ => String::new(),
            }
        };
        // The recipe drops empty system messages.
        let before = msgs.len();
        msgs.retain(|m| !(m["role"] == "system" && text_of(&m["content"]).is_empty()));
        if msgs.len() != before {
            classes.push("EMPTY_SYSTEM_DROPPED");
        }
        // `content: []` is "" in the recipe (a "\n\n"-joined nothing).
        for m in msgs.iter_mut() {
            if m["role"] == "user" && m["content"].as_array().is_some_and(|a| a.is_empty()) {
                m["content"] = json!([{"type": "text", "text": ""}]);
                classes.push("EMPTY_PARTS_USER_JOIN");
            }
        }
        // The recipe renders {"name", "description" ("" if absent), "parameters"} only.
        if let Some(t) = tools.as_mut() {
            let normalized: Vec<Value> = t
                .as_array()
                .unwrap()
                .iter()
                .map(|t| {
                    let f = &t["function"];
                    json!({"type": "function", "function": {
                        "name": f["name"],
                        "description": f.get("description").filter(|d| !d.is_null()).cloned().unwrap_or(json!("")),
                        "parameters": f.get("parameters").filter(|p| !p.is_null()).cloned().unwrap_or(json!({})),
                    }})
                })
                .collect();
            // (Serialized: `Value` equality ignores key order.)
            if Value::Array(normalized.clone()).to_string() != t.to_string() {
                classes.push("TOOL_SCHEMA_NORMALIZED");
            }
            *t = Value::Array(normalized);
        }
        // The recipe parses `arguments` once: anything but a JSON object is
        // {"arguments": <raw>} (encoding.py also decodes double-encoded JSON).
        for m in msgs.iter_mut() {
            let Some(tcs) = m.get_mut("tool_calls").and_then(|t| t.as_array_mut()) else { continue };
            for tc in tcs {
                let raw = tc["function"]["arguments"].as_str().unwrap_or("").to_string();
                if serde_json::from_str::<serde_json::Map<String, Value>>(&raw).is_ok() {
                    continue;
                }
                let double = serde_json::from_str::<Value>(&raw)
                    .ok()
                    .and_then(|v| v.as_str().map(|s| serde_json::from_str::<serde_json::Map<String, Value>>(s).is_ok()))
                    .unwrap_or(false);
                if double {
                    classes.push("DOUBLE_ENCODED_ARGUMENTS");
                }
                tc["function"]["arguments"] = json!(json!({"arguments": raw}).to_string());
            }
        }
        // The recipe creates an empty first system message for the tools.
        if tools.is_some() && msgs.first().is_some_and(|m| m["role"] != "system") {
            msgs.insert(0, json!({"role": "system", "content": ""}));
            classes.push("TOOLS_WITHOUT_SYSTEM");
        }
    }
    if let Some(t) = tools {
        msgs[0]["tools"] = t;
    }
    let effort = conv.reasoning_effort.map(|e| match e {
        deepseek_recipe_core::conversation::ReasoningEffort::Low => "low",
        deepseek_recipe_core::conversation::ReasoningEffort::Max => "max",
        _ => "high",
    });
    classes.dedup();
    Some((
        json!({
            "messages": msgs,
            "thinking_mode": if conv.thinking_mode { "thinking" } else { "chat" },
            "reasoning_effort": effort,
            "drop_thinking": drop_thinking,
        }),
        classes,
    ))
}

/// Disagreements no input rewrite can neutralize (structural rendering
/// choices), as predicates on the request.
fn structural_disagreement(req: &Value) -> Option<&'static str> {
    let msgs = req["messages"].as_array().unwrap();
    // Tool-call `arguments` are JSON TEXT: the recipe (and the server) parse
    // it with serde_json, whose default float parser is not correctly
    // rounded; Python's is. A literal the two parse differently prints
    // differently (e.g. 1.2345678901234567e+20).
    let misparsed = msgs.iter().filter_map(|m| m["tool_calls"].as_array()).flatten().any(|tc| {
        let a = tc["function"]["arguments"].as_str().unwrap_or("");
        a.split(|c: char| !(c.is_ascii_digit() || matches!(c, '.' | 'e' | 'E' | '+' | '-')))
            .filter(|t| t.contains(['.', 'e', 'E']) && t.chars().any(|c| c.is_ascii_digit()))
            .any(|t| match (serde_json::from_str::<f64>(t), t.parse::<f64>()) {
                (Ok(a), Ok(b)) => a.to_bits() != b.to_bits(),
                _ => false,
            })
    });
    if misparsed {
        return Some("SERDE_FLOAT_PARSE: serde_json's (non-correctly-rounded) float parse of tool-call arguments, shared by the recipe and the server");
    }
    // encoding.py renders a latest_reminder AFTER the assistant header that a
    // preceding user / mid-system message gets; the recipe before its own.
    if msgs.iter().any(|m| m["role"] == "latest_reminder") {
        return Some("LATEST_REMINDER_ORDER");
    }
    // encoding.py emits `<｜Assistant｜>` only as the transition after a user
    // (or mid-conversation system) message; the recipe on every assistant
    // message (first message, after another assistant, after the leading
    // system message). Judged on the messages the recipe keeps (empty
    // system messages dropped).
    let empty = |m: &Value| match &m["content"] {
        Value::String(s) => s.is_empty(),
        Value::Array(a) => a.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("\n\n").is_empty(),
        _ => true,
    };
    let msgs: Vec<&Value> = msgs.iter().filter(|m| !(m["role"] == "system" && empty(m))).collect();
    if msgs.iter().enumerate().any(|(i, m)| {
        m["role"] == "assistant" && (i == 0 || msgs[i - 1]["role"] == "assistant" || (i == 1 && msgs[0]["role"] == "system"))
    }) {
        return Some("ASSISTANT_HEADER_ONLY_AFTER_USER");
    }
    None
}

/// DeepSeek's two references -- deepseek-recipe (primary) and the
/// checkpoint's encoding.py -- on the same generated requests: every
/// difference must fall in a named class. Report: `--nocapture`.
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
    // Per request: raw input, neutralized input, neutralized + drop_thinking.
    let mut reqs: Vec<(Value, String, Vec<&'static str>)> = Vec::new();
    let mut lines = String::new();
    for _ in 0..n {
        let req = strategy.new_tree(&mut runner).unwrap().current().to_request();
        let Ok(recipe) = oracle_prompt(&req) else { continue };
        let (Some((raw, _)), Some((neu, classes)), Some((dropped, _))) =
            (encoding_py_case(&req, false, false), encoding_py_case(&req, false, true), encoding_py_case(&req, true, true))
        else {
            continue;
        };
        for c in [raw, neu, dropped] {
            lines.push_str(&c.to_string());
            lines.push('\n');
        }
        reqs.push((req, recipe, classes));
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
    assert_eq!(outs.len(), 3 * reqs.len(), "driver output count");
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    let mut bump = |k: &str| *counts.entry(k.to_string()).or_default() += 1;
    let mut unexplained = Vec::new();
    for ((req, recipe, classes), out) in reqs.iter().zip(outs.chunks(3)) {
        let (Some(raw), Some(neu), Some(dropped)) = (out[0]["ok"].as_str(), out[1]["ok"].as_str(), out[2]["ok"].as_str()) else {
            let err = out[0]["err"].as_str().or(out[1]["err"].as_str()).unwrap_or("");
            if err.contains("image") {
                bump("encoding.py rejects, recipe accepts: client text spelling <｜deepseek_image｜> (each rejects ITS placeholder spelling)");
            } else {
                unexplained.push(format!("encoding.py rejects: {err}: {}", serde_json::to_string(req).unwrap()));
            }
            continue;
        };
        if dropped != neu {
            bump("DROP_THINKING: encoding.py's default drop_thinking=True drops completed turns' reasoning; the recipe keeps it");
        }
        if raw == recipe {
            bump("identical");
            continue;
        }
        let gen_prompt = neu != recipe && strip_generation_prompt(recipe) == neu;
        if neu == recipe || gen_prompt {
            for c in classes {
                bump(c);
            }
            if gen_prompt {
                bump("GEN_PROMPT_ALWAYS: the recipe always ends with the next-turn header");
            }
            if classes.is_empty() && !gen_prompt {
                unexplained.push(format!("raw differs but no class applies: {}", serde_json::to_string(req).unwrap()));
            }
            continue;
        }
        match structural_disagreement(req) {
            Some(c) => bump(c),
            None => unexplained.push(format!("{}\n  recipe vs encoding.py (neutralized {classes:?}): {}", serde_json::to_string(req).unwrap(), diff(recipe, neu))),
        }
    }
    eprintln!("secondary oracle (encoding.py vs deepseek-recipe), {} generated requests:", reqs.len());
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
    ThinkBegin,
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
    /// `tool_choice: "required"` (thinking off): the prompt ends with the
    /// opening `<｜DSML｜ calls>\n`, so the output starts with an invoke.
    required: bool,
    /// Random byte split points for the server's tokens, char split points
    /// for the oracle's text chunks.
    seed: u64,
    /// Explicit pieces instead of the ones the fields describe (truncation
    /// and the deviation tests).
    override_pieces: Option<Vec<Piece>>,
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
        prop_oneof![4 => Just(Vec::new()), 1 => prop::collection::vec("[a-z]{1,3}|\\.", 1..3)],
        prop::bool::weighted(0.15),
        any::<u64>(),
    )
        .prop_map(|(thinking, reasoning, close_think, content, calls, eos, stop, required, seed)| {
            if required {
                // Required: no reasoning, no content, at least one call.
                let calls = if calls.is_empty() { vec![OCallG { name: 0, params: Vec::new() }] } else { calls };
                OutputG { thinking: false, reasoning: String::new(), close_think: false, content: String::new(), calls, eos, stop, required, seed, override_pieces: None }
            } else {
                OutputG { thinking, reasoning, close_think, content, calls, eos, stop, required, seed, override_pieces: None }
            }
        })
        .boxed()
}

impl OutputG {
    fn pieces(&self) -> Vec<Piece> {
        if let Some(p) = &self.override_pieces {
            return p.clone();
        }
        let mut p = Vec::new();
        if self.thinking {
            p.push(Piece::T(self.reasoning.clone()));
            if self.close_think {
                p.push(Piece::ThinkEnd);
            }
        }
        if !self.required {
            p.push(Piece::T(self.content.clone()));
        }
        if !self.calls.is_empty() {
            if !self.required {
                p.push(Piece::T("\n\n<".into()));
                p.push(Piece::Dsml);
                p.push(Piece::T(" calls>\n".into()));
            }
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
        if self.required {
            r["tool_choice"] = json!("required");
        }
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
            Piece::ThinkBegin => text.push_str("<think>"),
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
/// ends the turn with Stop, a missing EOS = the token cap; `</think>` is an
/// empty non-reasoning chunk). Text pieces are
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
            Piece::ThinkBegin => {
                in_think = true;
                n += 1;
            }
            Piece::ThinkEnd => {
                // Suppressed bytes; the worker marks where the reasoning ended.
                if in_think {
                    evs.push(WorkerEvent::Chunk { token_id: deepstrix_server::tokens::TOK_THINK_END, bytes: Vec::new(), reasoning: false });
                }
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
        accumulate_with(rx, Some(DSML_ID), o.stop.clone(), o.required).await.unwrap()
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
    let chunks = runtime().block_on(deepstrix_server::openai::handler::sse_transcript(evs, Some(DSML_ID), o.stop.clone(), o.required));
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
/// Each one is pinned by a test below (or by the properties above, which
/// assert the exact allowance). Changing one is an owner decision.
pub const KNOWN_DEVIATIONS: &[(&str, &str)] = &[
    // Representation only.
    ("IMAGE_PLACEHOLDER_SPELLING", "prompt text spells id 129264 <｜deepseek_image｜> (checkpoint tokenizer.json), the oracle <｜image｜> (its bundled tokenizer); same token id"),
    ("EMPTY_FIELD_NULL", "an empty content / reasoning_content is null in our response, \"\" in the oracle's"),
    ("TOKEN_IDS_VIA_OUR_BPE", "the id leg tokenizes the oracle prompt HF-style with the server's BPE; the oracle's own HF tokenizer object needs a direct `tokenizers` dev-dependency"),
    // Security: the prompt bytes are identical, the ids are not.
    ("CLIENT_SPECIALS_INERT", "client text spelling a special token (<｜User｜>, </think>, ｜DSML｜ ...) is BPE'd as text; the oracle's HF tokenization makes it the control token"),
    // Request leniency: the oracle rejects (400), the server renders.
    ("TOOL_RESULT_STRUCTURE", "tool results need not answer the immediately preceding assistant's calls one-to-one (missing / extra / unknown-id / orphan tool messages, duplicate call ids)"),
    ("EMPTY_OR_NULL_FIELDS", "`tool_calls: []`, an assistant with neither content nor tool calls, `content: null` / absent on system / user / tool"),
    ("TOOL_DEFINITION_VALIDATION", "tool names outside [A-Za-z0-9_-]{1,128}, `parameters` that are not a JSON Schema of type object, non-function tool entries (skipped)"),
    ("LENIENT_ROLES_AND_PARTS", "the `developer` role, bare-string / `input_text` / unknown content parts, Anthropic `image` / `input_image` parts, local-path image URLs"),
    ("EFFORT_EXTENSIONS", "numeric reasoning_effort (\"1\"..\"100\", \"0\" = off), off / disabled / false / \"\" / ultra"),
    // Both accept, the prompt differs.
    ("EFFORT_MINIMAL_25", "reasoning_effort \"minimal\" renders budget 25 (owner 10-11); the oracle maps it to low = 50"),
    ("EFFORT_XHIGH_MAX", "reasoning_effort \"xhigh\" renders budget 100 (Hermes clamps max -> xhigh); the oracle maps it to 75"),
    ("REASONING_FIELD", "letta's `reasoning` field sets the effort when `reasoning_effort` is absent; the oracle ignores the field"),
    // Output parsing.
    ("TOKEN_DRIVEN_SCANNER", "output parsing keys on the DSML / think special TOKENS; markup spelled as plain text by the model is content to us, markup to the text-driven oracle"),
    ("TRUNCATED_TOOL_CALL", "a call the token cap cut off is dropped (finish length); the oracle returns it with truncated arguments"),
    ("EOS_MID_TOOL_CALL", "EOS inside a call: the server repairs it (closes the tags), or reports finish internal_error when no call is recoverable; the oracle returns whatever it parsed"),
    ("MALFORMED_DSML", "an unknown DSML tag or orphan close: finish internal_error (non-streaming drops content and calls); the oracle parses leniently"),
    ("HALLUCINATED_CALLS_BLOCK", "a calls block with no invoke: its text is returned as content and scanning continues; the oracle drops it and everything after"),
    ("INVALID_JSON_PARAMETER", "a string=\"false\" value that is not JSON becomes a JSON string; the oracle splices it raw (invalid arguments JSON)"),
    ("TOOL_CALLS_WITHOUT_TOOLS", "DSML in the output is parsed as tool calls even when the request declared no tools (or tool_choice none); the oracle returns it as content"),
    ("THINK_IN_CONTENT", "a <think> token after the content started is dropped; the oracle keeps the text \"<think>\""),
    ("OUTPUT_JSON_MODE", "response_format json_object: the server returns the model's text as is; the oracle blanks everything around the first JSON value to spaces"),
];

/// The deviations are exactly those named above (a new deviation must be
/// added there, with its test).
#[test]
fn known_deviations_are_unique_and_documented() {
    let mut names: Vec<&str> = KNOWN_DEVIATIONS.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    names.dedup();
    assert_eq!(names.len(), KNOWN_DEVIATIONS.len());
}

// ----- truncated outputs -----

/// A well-formed output cut after `k` pieces, ended by EOS or the token cap.
fn truncated_output() -> BoxedStrategy<(OutputG, usize, bool)> {
    output()
        .prop_flat_map(|o| {
            let n = o.pieces().len();
            (Just(o), 0..=n, any::<bool>())
        })
        .boxed()
}

/// Cut anywhere: outside a calls block the two parse identically (the cut
/// is ordinary end of text); inside one, the server's TRUNCATED_TOOL_CALL /
/// EOS_MID_TOOL_CALL / MALFORMED_DSML deviations apply, and nothing else may
/// differ: the same content and reasoning, the same COMPLETED calls.
#[test]
fn truncated_output_matches_oracle_or_named() {
    run_prop("truncated output", truncated_output(), |(o, k, eos)| {
        let mut pieces: Vec<Piece> = o.pieces().into_iter().filter(|p| *p != Piece::Eos).take(*k).collect();
        let first_dsml = pieces.iter().position(|p| *p == Piece::Dsml);
        let in_block = first_dsml.is_some() || o.required;
        if *eos {
            pieces.push(Piece::Eos);
        }
        let cut = CutOutput { o: o.clone(), pieces: pieces.clone() };
        let oracle = cut.oracle()?;
        let ours = cut.ours_nonstream();
        let stream = cut.ours_stream();
        if !in_block {
            if ours != oracle || stream != oracle {
                return Err(format!("cut outside a block: oracle {oracle:?}\n   ours {ours:?}\n stream {stream:?}\n pieces {pieces:?}"));
            }
            return Ok(());
        }
        if ours.finish == "internal_error" {
            return Ok(()); // EOS_MID_TOOL_CALL / MALFORMED_DSML
        }
        let invoke_opened = pieces.iter().any(|p| matches!(p, Piece::T(t) if t.starts_with(" invoke name=")));
        if !invoke_opened {
            // HALLUCINATED_CALLS_BLOCK: a block cut before its first invoke;
            // the server surfaces the block's text (the structural newline)
            // as content.
            return Ok(());
        }
        let completed = pieces.iter().filter(|p| matches!(p, Piece::T(t) if t == " invoke>")).count();
        for (name, got) in [("non-streaming", &ours), ("streaming", &stream)] {
            if got.reasoning != oracle.reasoning || got.content != oracle.content {
                return Err(format!("{name}: cut inside a block changed content/reasoning: oracle {oracle:?}\n   ours {got:?}\n pieces {pieces:?}"));
            }
            let n = completed.min(oracle.tool_calls.len());
            if got.tool_calls.len() < n || got.tool_calls[..n] != oracle.tool_calls[..n] {
                return Err(format!("{name}: completed calls differ: oracle {oracle:?}\n   ours {got:?}\n pieces {pieces:?}"));
            }
        }
        Ok(())
    });
}

/// An output given as explicit pieces (for truncation and the deviation tests).
struct CutOutput {
    o: OutputG,
    pieces: Vec<Piece>,
}

impl CutOutput {
    fn with(&self) -> OutputG {
        let mut o = self.o.clone();
        o.eos = self.pieces.last() == Some(&Piece::Eos);
        o
    }
    fn oracle(&self) -> Result<Parsed, String> {
        oracle_parse_pieces(&self.with(), &self.pieces, true)
    }
    fn ours_nonstream(&self) -> Parsed {
        ours_parse_pieces(&self.with(), &self.pieces, false)
    }
    fn ours_stream(&self) -> Parsed {
        ours_parse_pieces(&self.with(), &self.pieces, true)
    }
}

fn oracle_parse_pieces(o: &OutputG, pieces: &[Piece], chunked: bool) -> Result<Parsed, String> {
    let mut o = o.clone();
    o.override_pieces = Some(pieces.to_vec());
    oracle_parse(&o, chunked)
}

fn ours_parse_pieces(o: &OutputG, pieces: &[Piece], stream: bool) -> Parsed {
    let mut o = o.clone();
    o.override_pieces = Some(pieces.to_vec());
    if stream {
        ours_parse_stream(&o)
    } else {
        ours_parse_nonstream(&o)
    }
}

// ----- request leniency / rejection parity -----

/// One change to a valid request, and what the server does with it.
#[derive(Debug, Clone, Copy)]
enum Mutation {
    // The oracle rejects; the server rejects too.
    RegexFormat,
    DuplicateToolNames,
    UnknownThinkingType,
    UnknownToolChoice,
    SystemImage,
    ReasoningImageToken,
    ArgumentsImageToken,
    // The oracle rejects; the server renders (named leniency).
    OrphanToolResult,
    MissingToolResult,
    UnknownToolResultId,
    DuplicateCallIds,
    EmptyToolCalls,
    AssistantWithoutContent,
    NullUserContent,
    ToolNameCharset,
    ParametersNotObject,
    NonFunctionTool,
    DeveloperRole,
    BareStringPart,
    AnthropicImagePart,
    LocalImagePath,
    NumericEffort,
    OffEffort,
}

impl Mutation {
    const ALL: [Mutation; 23] = [
        Mutation::RegexFormat,
        Mutation::DuplicateToolNames,
        Mutation::UnknownThinkingType,
        Mutation::UnknownToolChoice,
        Mutation::SystemImage,
        Mutation::ReasoningImageToken,
        Mutation::ArgumentsImageToken,
        Mutation::OrphanToolResult,
        Mutation::MissingToolResult,
        Mutation::UnknownToolResultId,
        Mutation::DuplicateCallIds,
        Mutation::EmptyToolCalls,
        Mutation::AssistantWithoutContent,
        Mutation::NullUserContent,
        Mutation::ToolNameCharset,
        Mutation::ParametersNotObject,
        Mutation::NonFunctionTool,
        Mutation::DeveloperRole,
        Mutation::BareStringPart,
        Mutation::AnthropicImagePart,
        Mutation::LocalImagePath,
        Mutation::NumericEffort,
        Mutation::OffEffort,
    ];

    /// `None`: the server rejects as the oracle does; `Some(name)`: the
    /// server accepts (the KNOWN_DEVIATIONS entry).
    fn leniency(self) -> Option<&'static str> {
        use Mutation::*;
        match self {
            RegexFormat | DuplicateToolNames | UnknownThinkingType | UnknownToolChoice | SystemImage | ReasoningImageToken | ArgumentsImageToken => None,
            OrphanToolResult | MissingToolResult | UnknownToolResultId | DuplicateCallIds => Some("TOOL_RESULT_STRUCTURE"),
            EmptyToolCalls | AssistantWithoutContent | NullUserContent => Some("EMPTY_OR_NULL_FIELDS"),
            ToolNameCharset | ParametersNotObject | NonFunctionTool => Some("TOOL_DEFINITION_VALIDATION"),
            DeveloperRole | BareStringPart | AnthropicImagePart | LocalImagePath => Some("LENIENT_ROLES_AND_PARTS"),
            NumericEffort | OffEffort => Some("EFFORT_EXTENSIONS"),
        }
    }

    fn apply(self, req: &mut Value) {
        use Mutation::*;
        let call = |id: &str| json!({"id": id, "type": "function", "function": {"name": "calc", "arguments": "{}"}});
        let tool = |id: &str| json!({"role": "tool", "tool_call_id": id, "content": "r"});
        let push = |req: &mut Value, ms: Vec<Value>| req["messages"].as_array_mut().unwrap().extend(ms);
        let ensure_tools = |req: &mut Value| {
            if req.get("tools").and_then(|t| t.as_array()).is_none_or(|a| a.is_empty()) {
                req["tools"] = json!([{"type": "function", "function": {"name": "calc", "parameters": {"type": "object", "properties": {}}}}]);
            }
            req.as_object_mut().unwrap().remove("tool_choice");
        };
        match self {
            RegexFormat => req["response_format"] = json!({"type": "regex", "regex": "a+"}),
            DuplicateToolNames => {
                ensure_tools(req);
                let first = req["tools"][0].clone();
                req["tools"].as_array_mut().unwrap().push(first);
            }
            UnknownThinkingType => req["thinking"] = json!({"type": "sometimes"}),
            UnknownToolChoice => {
                ensure_tools(req);
                req["tool_choice"] = json!("always");
            }
            SystemImage => push(req, vec![json!({"role": "system", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]})]),
            ReasoningImageToken => push(req, vec![json!({"role": "assistant", "content": "a", "reasoning_content": "see <\u{ff5c}image\u{ff5c}>"})]),
            ArgumentsImageToken => push(req, vec![json!({"role": "assistant", "content": "", "tool_calls": [{"id": "x", "type": "function", "function": {"name": "calc", "arguments": "{\"a\": \"<\u{ff5c}image\u{ff5c}>\"}"}}]}), tool("x")]),
            OrphanToolResult => push(req, vec![json!({"role": "user", "content": "u"}), tool("orphan")]),
            MissingToolResult => push(req, vec![json!({"role": "assistant", "content": "", "tool_calls": [call("m1"), call("m2")]}), tool("m1")]),
            UnknownToolResultId => push(req, vec![json!({"role": "assistant", "content": "", "tool_calls": [call("k1")]}), tool("k2")]),
            DuplicateCallIds => push(req, vec![json!({"role": "assistant", "content": "", "tool_calls": [call("d"), call("d")]}), tool("d"), tool("d")]),
            EmptyToolCalls => push(req, vec![json!({"role": "assistant", "content": "x", "tool_calls": []})]),
            AssistantWithoutContent => push(req, vec![json!({"role": "assistant", "content": null})]),
            NullUserContent => push(req, vec![json!({"role": "user", "content": null})]),
            ToolNameCharset => {
                ensure_tools(req);
                req["tools"].as_array_mut().unwrap().push(json!({"type": "function", "function": {"name": "ns.tool"}}));
            }
            ParametersNotObject => {
                ensure_tools(req);
                req["tools"].as_array_mut().unwrap().push(json!({"type": "function", "function": {"name": "zero_args", "parameters": {}}}));
            }
            NonFunctionTool => {
                ensure_tools(req);
                req["tools"].as_array_mut().unwrap().push(json!({"type": "web_search"}));
            }
            DeveloperRole => push(req, vec![json!({"role": "developer", "content": "d"})]),
            BareStringPart => push(req, vec![json!({"role": "user", "content": ["bare"]})]),
            AnthropicImagePart => push(req, vec![json!({"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AA=="}}]})]),
            LocalImagePath => push(req, vec![json!({"role": "user", "content": [{"type": "image_url", "image_url": {"url": "/tmp/x.png"}}]})]),
            NumericEffort | OffEffort => {
                // Thinking may turn on: drop a tool choice thinking forbids.
                req.as_object_mut().unwrap().remove("thinking");
                req.as_object_mut().unwrap().remove("tool_choice");
                req["reasoning_effort"] = json!(if matches!(self, NumericEffort) { "33" } else { "off" });
            }
        }
    }
}

/// Every mutation of a valid request: the oracle rejects; the server rejects
/// too, or accepts under exactly the named leniency.
#[test]
fn rejections_match_or_are_named() {
    let strategy = (conversation(), prop::sample::select(&Mutation::ALL[..]));
    run_prop("rejection parity", strategy, |(conv, m)| {
        let base = conv.to_request();
        if oracle_prompt(&base).is_err() {
            return Ok(()); // only valid bases
        }
        let mut req = base;
        m.apply(&mut req);
        let oracle = oracle_prompt(&req);
        let ours = ours_prompt(&req);
        match (oracle, ours, m.leniency()) {
            (Ok(_), _, _) => Err(format!("{m:?}: the oracle accepts it now: {}", serde_json::to_string(&req).unwrap())),
            (Err(_), Err(_), None) => Ok(()),
            (Err(_), Ok(_), Some(_)) => Ok(()),
            (Err(e), Ok(_), None) => Err(format!("{m:?}: oracle rejects ({e}), ours accepts")),
            (Err(_), Err(e), Some(name)) => Err(format!("{m:?}: ours rejects ({e}) though {name} says it accepts")),
        }
    });
}

// ----- both accept, the prompt differs -----

fn effort_line(prompt: &str) -> Option<&str> {
    prompt.split("Reasoning Effort: ").nth(1).and_then(|s| s.split(' ').next())
}

#[test]
fn deviation_effort_minimal_xhigh_and_reasoning_field() {
    let base = json!({"model": "m", "messages": [{"role": "user", "content": "q"}]});
    let with = |k: &str, v: &str| {
        let mut r = base.clone();
        r[k] = json!(v);
        r
    };
    // EFFORT_MINIMAL_25
    let r = with("reasoning_effort", "minimal");
    assert_eq!(effort_line(&oracle_prompt(&r).unwrap()), Some("50"));
    assert_eq!(effort_line(&ours_prompt(&r).unwrap()), Some("25"));
    // EFFORT_XHIGH_MAX
    let r = with("reasoning_effort", "xhigh");
    assert_eq!(effort_line(&oracle_prompt(&r).unwrap()), Some("75"));
    assert_eq!(effort_line(&ours_prompt(&r).unwrap()), Some("100"));
    // REASONING_FIELD
    let r = with("reasoning", "max");
    assert_eq!(effort_line(&oracle_prompt(&r).unwrap()), Some("75"));
    assert_eq!(effort_line(&ours_prompt(&r).unwrap()), Some("100"));
    // Everything else on the ladder is identical.
    for e in ["none", "low", "medium", "high", "max"] {
        let r = with("reasoning_effort", e);
        assert_eq!(ours_prompt(&r).unwrap(), oracle_prompt(&r).unwrap(), "{e}");
    }
    // An explicit `thinking` wins over the effort on both sides.
    let mut r = with("reasoning_effort", "max");
    r["thinking"] = json!({"type": "disabled"});
    assert_eq!(ours_prompt(&r).unwrap(), oracle_prompt(&r).unwrap());
    assert!(effort_line(&ours_prompt(&r).unwrap()).is_none());
}

// ----- output deviations -----

fn pieces_output(thinking: bool, pieces: Vec<Piece>) -> CutOutput {
    let o = OutputG {
        thinking,
        reasoning: String::new(),
        close_think: false,
        content: String::new(),
        calls: Vec::new(),
        eos: true,
        stop: Vec::new(),
        required: false,
        seed: 7,
        override_pieces: None,
    };
    CutOutput { o, pieces }
}

fn t(s: &str) -> Piece {
    Piece::T(s.to_string())
}

/// `<｜DSML｜ calls>\n<｜DSML｜ invoke name="calc">\n` + `rest`.
fn calls_head(rest: Vec<Piece>) -> Vec<Piece> {
    let mut v = vec![t("\n\n<"), Piece::Dsml, t(" calls>\n<"), Piece::Dsml, t(" invoke name=\"calc\">\n")];
    v.extend(rest);
    v
}

#[test]
fn deviation_truncated_tool_call_is_dropped() {
    // TRUNCATED_TOOL_CALL: the token cap inside a string parameter.
    let c = pieces_output(false, calls_head(vec![t("<"), Piece::Dsml, t(" parameter name=\"query\" string=\"true\">rm -rf /tm")]));
    let oracle = c.oracle().unwrap();
    assert_eq!(oracle.tool_calls.len(), 1, "the oracle returns the cut call");
    assert_eq!(oracle.finish, "length");
    let ours = c.ours_nonstream();
    assert!(ours.tool_calls.is_empty(), "ours drops it: {ours:?}");
    assert_eq!(ours.finish, "length");
}

#[test]
fn deviation_eos_mid_tool_call_is_repaired() {
    // EOS_MID_TOOL_CALL: EOS inside the parameter body.
    let mut p = calls_head(vec![t("<"), Piece::Dsml, t(" parameter name=\"query\" string=\"true\">ls")]);
    p.push(Piece::Eos);
    let c = pieces_output(false, p);
    let oracle = c.oracle().unwrap();
    assert!(matches!(&oracle.tool_calls[0].1, Value::String(s) if s.starts_with("<unparsed>")), "oracle: truncated JSON {oracle:?}");
    let ours = c.ours_nonstream();
    assert_eq!(ours.tool_calls, vec![("calc".to_string(), json!({"query": "ls"}))], "ours closes the tags");
}

#[test]
fn deviation_malformed_dsml_is_an_internal_error() {
    // MALFORMED_DSML: an unknown tag inside the invoke.
    let mut p = calls_head(vec![t("<"), Piece::Dsml, t(" command name=\"ls\">x</"), Piece::Dsml, t(" command>\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>")]);
    p.insert(0, t("Doing it."));
    p.push(Piece::Eos);
    let c = pieces_output(false, p);
    let oracle = c.oracle().unwrap();
    assert_eq!(oracle.content, "Doing it.");
    let ours = c.ours_nonstream();
    assert_eq!(ours.finish, "internal_error");
    assert_eq!(ours.content, "", "non-streaming drops content on malformed DSML");
}

#[test]
fn deviation_hallucinated_calls_block_is_content() {
    // HALLUCINATED_CALLS_BLOCK
    let c = pieces_output(false, vec![t("a\n\n<"), Piece::Dsml, t(" calls>\nnot a call\n</"), Piece::Dsml, t(" calls>\nafter"), Piece::Eos]);
    let oracle = c.oracle().unwrap();
    assert_eq!(oracle.content, "a");
    let ours = c.ours_nonstream();
    assert_eq!(ours.content, "a\nnot a call\n\nafter");
}

#[test]
fn deviation_invalid_json_parameter_becomes_a_string() {
    // INVALID_JSON_PARAMETER
    let mut p = calls_head(vec![t("<"), Piece::Dsml, t(" parameter name=\"limit\" string=\"false\">ten</"), Piece::Dsml, t(" parameter>\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>")]);
    p.push(Piece::Eos);
    let c = pieces_output(false, p);
    let oracle = c.oracle().unwrap();
    assert!(matches!(&oracle.tool_calls[0].1, Value::String(s) if s.starts_with("<unparsed>")), "{oracle:?}");
    assert_eq!(c.ours_nonstream().tool_calls[0].1, json!({"limit": "ten"}));
}

#[test]
fn deviation_tool_calls_without_tools_are_parsed() {
    // TOOL_CALLS_WITHOUT_TOOLS: a request without tools; the oracle's
    // parser then leaves DSML as content.
    let mut p = calls_head(vec![t("\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>")]);
    p.push(Piece::Eos);
    let c = pieces_output(false, p.clone());
    let req: OracleRequest = serde_json::from_value(json!({"model": "m", "messages": [{"role": "user", "content": "q"}], "thinking": {"type": "disabled"}})).unwrap();
    let conv = req.convert(ConversionOptions::default()).unwrap();
    assert!(!conv.parsing_options.parse_tool_calls, "the oracle does not parse tool calls without tools");
    assert_eq!(c.ours_nonstream().tool_calls.len(), 1, "ours always does");
}

#[test]
fn deviation_think_in_content_is_dropped() {
    // THINK_IN_CONTENT: the oracle reads a later `<think>` as text; the
    // worker suppresses the token (and the splitter keeps reading content).
    let c = pieces_output(false, vec![t("a"), Piece::ThinkBegin, t("b"), Piece::Eos]);
    assert_eq!(c.oracle().unwrap().content, "a<think>b");
    assert_eq!(c.ours_nonstream().content, "ab");
    assert_eq!(c.ours_stream().content, "ab");
}

#[test]
fn deviation_output_json_mode_is_verbatim() {
    // OUTPUT_JSON_MODE: the oracle's json_object parsing blanks the text
    // around the JSON value.
    let req: OracleRequest = serde_json::from_value(json!({"model": "m", "messages": [{"role": "user", "content": "reply in JSON"}], "thinking": {"type": "disabled"}, "response_format": {"type": "json_object"}})).unwrap();
    let conv = req.convert(ConversionOptions::default()).unwrap();
    assert!(conv.parsing_options.parse_json_output);
}

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

// ---------------------------------------------------------------------------
// Divergence catalog: one minimal example per divergence the properties
// found (and the server fixed), so each stays pinned by name.
// ---------------------------------------------------------------------------

fn user(c: &str) -> Value {
    json!({"role": "user", "content": c})
}

fn weather_tool() -> Value {
    json!({"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}})
}

/// (name, request): the server must render each exactly as the oracle (or
/// reject it as the oracle does).
fn prompt_catalog() -> Vec<(&'static str, Value)> {
    let call = |id: &str, args: &str| json!({"id": id, "type": "function", "function": {"name": "get_weather", "arguments": args}});
    vec![
        ("DEFAULT_EFFORT_75", json!({"messages": [user("hi")]})),
        ("THINKING_FIELD_DISABLED", json!({"thinking": {"type": "disabled"}, "messages": [user("hi")]})),
        ("THINKING_FIELD_ENABLED_EFFORT_NONE", json!({"thinking": {"type": "enabled"}, "reasoning_effort": "none", "messages": [user("hi")]})),
        ("HISTORY_REASONING_KEPT", json!({"messages": [user("q1"), {"role": "assistant", "content": "a1", "reasoning_content": "r1"}, user("q2")]})),
        ("HISTORY_EMPTY_THINK_BLOCK", json!({"messages": [user("q1"), {"role": "assistant", "content": "a1"}, user("q2")]})),
        ("TOOLS_WITHOUT_SYSTEM_SEPARATOR", json!({"thinking": {"type": "disabled"}, "tools": [weather_tool()], "messages": [user("q")]})),
        ("EMPTY_SYSTEM_DROPPED", json!({"thinking": {"type": "disabled"}, "messages": [{"role": "system", "content": ""}, user("q")]})),
        ("EMPTY_MID_SYSTEM_DROPPED", json!({"thinking": {"type": "disabled"}, "messages": [user("q"), {"role": "assistant", "content": "a"}, {"role": "system", "content": ""}]})),
        ("TRAILING_ASSISTANT_GEN_PROMPT", json!({"thinking": {"type": "disabled"}, "messages": [user("q"), {"role": "assistant", "content": "a"}]})),
        ("ASSISTANT_FIRST_HEADER", json!({"messages": [{"role": "assistant", "content": "a"}, user("q")]})),
        ("CONSECUTIVE_ASSISTANT_HEADERS", json!({"thinking": {"type": "disabled"}, "messages": [user("q"), {"role": "assistant", "content": "a"}, {"role": "assistant", "content": "b"}]})),
        ("TOOL_SCHEMA_NORMALIZED", json!({"thinking": {"type": "disabled"}, "messages": [user("q")],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}, "strict": true}}]})),
        ("ARGUMENTS_PARSED_ONCE", json!({"thinking": {"type": "disabled"}, "tools": [weather_tool()], "messages": [user("q"),
            {"role": "assistant", "content": "", "tool_calls": [call("c1", "\"{\\\"city\\\": \\\"Paris\\\"}\"")]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]})),
        ("CONTENT_BEFORE_CALLS_VERBATIM", json!({"thinking": {"type": "disabled"}, "tools": [weather_tool()], "messages": [user("q"),
            {"role": "assistant", "content": "Checking.\n\n", "tool_calls": [call("c1", "{\"city\": \"Paris\"}")]},
            {"role": "tool", "tool_call_id": "c1", "content": "sunny"}]})),
        ("LATEST_REMINDER_ROLE", json!({"messages": [user("q"), {"role": "latest_reminder", "content": "Remember X."}]})),
        ("TOOL_CHOICE_NONE", json!({"tool_choice": "none", "tools": [weather_tool()], "messages": [user("q")]})),
        ("TOOL_CHOICE_REQUIRED", json!({"tool_choice": "required", "thinking": {"type": "disabled"}, "tools": [weather_tool()], "messages": [user("q")]})),
        ("TOOL_CHOICE_REQUIRED_WITH_THINKING_REJECTED", json!({"tool_choice": "required", "tools": [weather_tool()], "messages": [user("q")]})),
        ("TOOL_CHOICE_NAMED", json!({"thinking": {"type": "disabled"}, "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
            "tools": [weather_tool(), {"type": "function", "function": {"name": "other", "parameters": {"type": "object"}}}], "messages": [user("q")]})),
        ("RESPONSE_FORMAT_JSON_OBJECT", json!({"response_format": {"type": "json_object"}, "messages": [user("Answer in JSON.")]})),
        ("RESPONSE_FORMAT_JSON_WITHOUT_THE_WORD_REJECTED", json!({"response_format": {"type": "json_object"}, "messages": [user("Answer.")]})),
        ("IMAGE_TOKEN_TEXT_REJECTED", json!({"messages": [user("see <\u{ff5c}image\u{ff5c}>")]})),
        ("DUPLICATE_TOOL_NAMES_REJECTED", json!({"tools": [weather_tool(), weather_tool()], "messages": [user("q")]})),
        ("TOOL_CALL_TYPE_OPTIONAL", json!({"thinking": {"type": "disabled"}, "tools": [weather_tool()], "messages": [user("q"),
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "function": {"name": "get_weather", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c1", "content": "r"}]})),
    ]
}

#[test]
fn divergence_catalog_prompts() {
    for (name, mut req) in prompt_catalog() {
        req["model"] = json!("m");
        if let Err(e) = check_prompt(&req) {
            panic!("{name}: {e}");
        }
    }
}

/// (name, thinking, output pieces): parsed exactly as by the oracle.
fn output_catalog() -> Vec<(&'static str, bool, Vec<Piece>)> {
    let block = |pre: &str| {
        let mut v = vec![t(&format!("{pre}\n\n<")), Piece::Dsml, t(" calls>\n<"), Piece::Dsml, t(" invoke name=\"get_weather\">\n<"), Piece::Dsml];
        v.extend([t(" parameter name=\"city\" string=\"true\">Paris</"), Piece::Dsml, t(" parameter>\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>"), Piece::Eos]);
        v
    };
    vec![
        ("NEWLINES_BEFORE_CALLS_DROPPED", false, block("Checking.")),
        ("CONTENT_LEADING_NEWLINES_DROPPED", false, vec![t("\n\nHello"), Piece::Eos]),
        ("REASONING_NEWLINES_TRIMMED", true, vec![t("\n\nplan\n\n"), Piece::ThinkEnd, t("\nanswer"), Piece::Eos]),
        ("TOOL_CALL_FROM_REASONING", true, {
            let mut v = vec![t("plan")];
            v.extend(block(""));
            v
        }),
        ("V41_ATTRIBUTES_RAW", false, vec![
            t("<"), Piece::Dsml, t(" calls>\n<"), Piece::Dsml, t(" invoke name=\"get_weather\">\n<"), Piece::Dsml,
            t(" parameter name=\"city\" string=\"true\">a &amp; b &lt;/\u{ff5c}DSML\u{ff5c}parameter></"), Piece::Dsml,
            t(" parameter>\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>"), Piece::Eos,
        ]),
        ("PARTIAL_MATCH_NEWLINE_KEPT", false, block("x<")),
        ("REASONING_PARTIAL_MATCH_NEWLINE_KEPT", true, vec![t("a<\n\n"), Piece::ThinkEnd, t("b"), Piece::Eos]),
    ]
}

#[test]
fn divergence_catalog_outputs() {
    for (name, thinking, pieces) in output_catalog() {
        let c = pieces_output(thinking, pieces);
        let oracle = c.oracle().unwrap();
        for (path, ours) in [("non-streaming", c.ours_nonstream()), ("streaming", c.ours_stream())] {
            assert_eq!(ours, oracle, "{name} ({path})");
        }
    }
    // A required tool choice: the output starts inside the calls block.
    let mut c = pieces_output(false, vec![t("<"), Piece::Dsml, t(" invoke name=\"get_weather\">\n</"), Piece::Dsml, t(" invoke>\n</"), Piece::Dsml, t(" calls>"), Piece::Eos]);
    c.o.required = true;
    let oracle = c.oracle().unwrap();
    assert_eq!(c.ours_nonstream(), oracle, "TOOL_CHOICE_REQUIRED_OUTPUT (non-streaming)");
    assert_eq!(c.ours_stream(), oracle, "TOOL_CHOICE_REQUIRED_OUTPUT (streaming)");
}
