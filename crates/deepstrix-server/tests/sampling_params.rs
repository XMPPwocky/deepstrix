//! Server-level plumbing: a request's `top_p` must reach `GenerateReq`.
//!
//! Before this change `top_p` was accepted by the OpenAI schema and then
//! silently dropped — `GenerateReq` carried only tokens/max_new/temperature/
//! min_p_rel/seed, so the sampler ran untruncated no matter what the client
//! asked for. These tests exercise the exact function `chat_completions`
//! uses to build the worker request (no model load, no GPU).

use deepstrix_server::openai::handler::{build_generate_req, resolve_top_p, DEFAULT_TOP_P};
use deepstrix_server::openai::types::ChatCompletionRequest;

fn parse(body: &str) -> ChatCompletionRequest {
    serde_json::from_str(body).expect("request body parses")
}

fn req_with(extra: &str) -> ChatCompletionRequest {
    parse(&format!(
        r#"{{"model":"deepseek-v4-flash","messages":[{{"role":"user","content":"hi"}}]{extra}}}"#
    ))
}

#[test]
fn request_top_p_reaches_generate_req() {
    let req = req_with(r#","top_p":0.5"#);
    assert_eq!(req.top_p, Some(0.5), "schema still parses top_p");
    let gen = build_generate_req(&req, vec![1, 2, 3], Vec::new(), Vec::new(), DEFAULT_TOP_P);
    assert_eq!(gen.top_p, 0.5, "top_p must survive into GenerateReq");
    // The other sampling fields are unchanged by this plumbing.
    assert_eq!(gen.temperature, 1.0);
    assert_eq!(gen.min_p_rel, 0.0);
    assert_eq!(gen.tokens, vec![1, 2, 3]);
}

#[test]
fn omitted_max_tokens_is_the_64k_default_and_flagged_as_such() {
    // No max_tokens: the server default (64K since 2026-09-23; DeepSeek's card
    // recommends >= 256K), flagged so admission may shrink it to fit the
    // context instead of rejecting a long prompt.
    let gen = build_generate_req(&req_with(""), Vec::new(), Vec::new(), Vec::new(), DEFAULT_TOP_P);
    assert_eq!(gen.max_new, 65536);
    assert!(gen.max_new_defaulted);

    // An explicit cap is the client's: kept as sent, never silently shrunk.
    let gen = build_generate_req(&req_with(r#","max_tokens":1000"#), Vec::new(), Vec::new(), Vec::new(), DEFAULT_TOP_P);
    assert_eq!(gen.max_new, 1000);
    assert!(!gen.max_new_defaulted);
}

#[test]
fn omitted_top_p_takes_the_server_default() {
    let req = req_with("");
    assert_eq!(req.top_p, None);
    let gen = build_generate_req(&req, Vec::new(), Vec::new(), Vec::new(), DEFAULT_TOP_P);
    assert_eq!(gen.top_p, 0.95, "model-card agent recipe is the default");
    assert_eq!(DEFAULT_TOP_P, 0.95);

    // ... and the operator can put it back to the pre-change behaviour
    // with `--default-top-p 1.0` (which is the no-op fast path).
    let gen = build_generate_req(&req, Vec::new(), Vec::new(), Vec::new(), 1.0);
    assert_eq!(gen.top_p, 1.0);
}

#[test]
fn top_p_is_clamped_into_the_unit_interval() {
    // Above 1 → 1.0 (no truncation) rather than a 400: same permissive
    // style as `temperature`, which just falls through to argmax at <= 0.
    assert_eq!(resolve_top_p(Some(2.0), DEFAULT_TOP_P), 1.0);
    // At or below 0 → the smallest cutoff the sampler accepts, which is
    // effectively greedy.
    assert!(resolve_top_p(Some(0.0), DEFAULT_TOP_P) > 0.0);
    assert!(resolve_top_p(Some(-1.0), DEFAULT_TOP_P) > 0.0);
    assert!(resolve_top_p(Some(0.0), DEFAULT_TOP_P) < 1e-5);
    // NaN → server default.
    assert_eq!(resolve_top_p(Some(f32::NAN), 0.9), 0.9);
    // Exactly 1.0 stays 1.0 so the bit-identical legacy chain is selected.
    assert_eq!(resolve_top_p(Some(1.0), DEFAULT_TOP_P), 1.0);
    // In-range values pass through untouched.
    assert_eq!(resolve_top_p(Some(0.7), DEFAULT_TOP_P), 0.7);
}

#[test]
fn default_reasoning_effort_stays_low_unless_the_operator_changes_it() {
    use deepstrix_server::prompt::{ReasoningEffort, DEFAULT_EFFORT};
    // The compiled-in default must not move: raising it changes the
    // rendered prompt and invalidates every cached KV prefix.
    assert_eq!(DEFAULT_EFFORT, ReasoningEffort::Low);
    assert_eq!(
        ReasoningEffort::from_request_fields(None, None),
        Ok(ReasoningEffort::Low)
    );
    // `--default-reasoning-effort high` only affects requests that omit
    // the field; an explicit request value still wins.
    assert_eq!(
        ReasoningEffort::from_request_fields_with_default(None, None, ReasoningEffort::High),
        Ok(ReasoningEffort::High)
    );
    assert_eq!(
        ReasoningEffort::from_request_fields_with_default(
            None,
            Some("low"),
            ReasoningEffort::Max
        ),
        Ok(ReasoningEffort::Low)
    );
}
