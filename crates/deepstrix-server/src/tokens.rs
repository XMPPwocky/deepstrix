//! V4-Flash special-token IDs. Mirrored from `deepstrix-chat`
//! (`crates/deepstrix-cli/src/bin/chat.rs:65-71`). Same model, same vocab,
//! same numeric IDs — kept here so the server crate doesn't depend on
//! deepstrix-cli.
//!
//! The `dsml_id` for `｜DSML｜` is loaded dynamically from the GGUF
//! vocab in Phase 2 (it's a true name-lookup rather than a hardcoded
//! position, matching `external/ds4/ds4.c:14952`). The rest are stable
//! IDs from the V4-Flash tokenizer.

pub const TOK_BOS: i32 = 0;
pub const TOK_EOS: i32 = 1;
pub const TOK_USER: i32 = 128803;
pub const TOK_ASSISTANT: i32 = 128804;
pub const TOK_THINK_BEGIN: i32 = 128821;
pub const TOK_THINK_END: i32 = 128822;

/// Returns true if `tok` is a role-boundary marker that ends the current
/// assistant turn (whether by completion or by the model hallucinating a
/// new role). Used by the decode loop's stop condition.
pub fn is_turn_end(tok: i32) -> bool {
    tok == TOK_EOS || tok == TOK_USER || tok == TOK_ASSISTANT || tok == TOK_BOS
}

/// Structural tokens that are emitted but never displayed (think markers).
pub fn is_think_marker(tok: i32) -> bool {
    tok == TOK_THINK_BEGIN || tok == TOK_THINK_END
}
