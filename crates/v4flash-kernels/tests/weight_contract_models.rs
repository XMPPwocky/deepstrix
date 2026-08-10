//! Weight-contract validation against the real model files.
//!
//! - The production (antirez 0731) file must validate CLEAN — it is the mix
//!   every wired kernel was built for.
//! - The unsloth UD-IQ2_XXS file is expected to fail until its kernels land
//!   (plan Phase 2); the test asserts the violations are exactly the known
//!   pending-kernel roles and nothing else (a violation outside that set
//!   means the file differs from what we planned for, or a regression).
//!   When Phase 2 completes and the pending set empties, this test asserts
//!   the file validates clean.
//!
//! Run: cargo test --release --test weight_contract_models -- --ignored --nocapture

use v4flash_core::MappedGguf;
use v4flash_kernels::weight_contract::{role_of, validate_model};

const ANTIREZ_PATH: &str =
    "/persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf";
const UNSLOTH_PATH: &str =
    "/persist/lumi/models/ds4f-unsloth/UD-IQ2_XXS/DeepSeek-V4-Flash-0731-UD-IQ2_XXS-00001-of-00003.gguf";

/// Roles whose new dtypes still lack wired kernels (shrinks as plan
/// Phase 2 lands each family; delete entries as the contract's allowed
/// lists grow).
const PENDING_KERNEL_ROLES: &[&str] = &[
    "blk.N.ffn_down_exps.weight",     // IQ3_XXS ×41, MXFP4 ×2
    "blk.N.ffn_gate_exps.weight",     // IQ2_S (blk.26)
    "blk.N.ffn_up_exps.weight",       // IQ2_S (blk.26)
    "blk.N.ffn_gate_shexp.weight",    // Q5_K/Q6_K
    "blk.N.ffn_up_shexp.weight",      // Q5_K/Q6_K
    "blk.N.ffn_down_shexp.weight",    // Q6_K
    "blk.N.attn_q_a.weight",          // Q5_K/Q6_K
    "output.weight",                  // Q4_K
    "token_embd.weight",              // Q4_K
];

#[test]
#[ignore]
fn antirez_file_validates_clean() {
    let m = MappedGguf::open(ANTIREZ_PATH).expect("open antirez GGUF");
    validate_model(m.gguf()).expect("production mix must satisfy the contract");
}

#[test]
#[ignore]
fn unsloth_violations_are_exactly_the_pending_kernel_set() {
    let m = MappedGguf::open(UNSLOTH_PATH).expect("open unsloth GGUF (3 shards)");
    assert_eq!(m.gguf().n_tensors, 1328, "merged shard tensor count");
    match validate_model(m.gguf()) {
        Ok(()) => {
            assert!(
                PENDING_KERNEL_ROLES.is_empty(),
                "unsloth validates clean but PENDING_KERNEL_ROLES is non-empty — \
                 kernels landed? delete the stale entries"
            );
        }
        Err(e) => {
            let msg = format!("{e}");
            eprintln!("--- unsloth missing-kernel list ---\n{msg}\n---");
            let mut unexpected = Vec::new();
            for line in msg.lines().filter(|l| l.trim_start().contains(": dtype ")) {
                let name = line.trim_start().split(':').next().unwrap_or("");
                let role = role_of(name);
                if !PENDING_KERNEL_ROLES.contains(&role.as_str()) {
                    unexpected.push(line.trim_start().to_string());
                }
            }
            assert!(
                unexpected.is_empty(),
                "violations outside the known pending-kernel set:\n{}",
                unexpected.join("\n")
            );
            // Dims violations are never expected.
            assert!(
                !msg.contains("!= expected"),
                "dims violation present:\n{msg}"
            );
        }
    }
}
