//! Laguna-S-2.1 GENERATE-VS-ORACLE correctness harness.
//!
//! Greedily generates N tokens from a FIXED prompt (token ids hardcoded to match
//! the llama.cpp oracle's tokenization exactly, so no tokenizer divergence can
//! contaminate the comparison) via the production [`LagunaHetModel`] decode path,
//! and prints, per step, the argmax token id + detokenized text + the top-5
//! logits. Run it once per attention config (forced via env) and diff the
//! generated token sequence + logits against the oracle.
//!
//! The prompt exercises the split-KV decode path (>16 tokens) with a
//! deterministic factual continuation so greedy is stable.
//!
//! Usage:
//!   nix develop --command cargo run --release -p deepstrix-cli \
//!       --bin laguna-oracle-gen -- [gguf]
//!
//! Attention config is selected by the same env vars the model reads:
//!   LAGUNA_DECODE_ATTN=naive|flash|splitkv     (default: head-grouped split-KV)
//!   LAGUNA_DECODE_ATTN_NAIVE=1                  (force naive decode)
//!   LAGUNA_ATTN_NAIVE / LAGUNA_ATTN_FLASH / LAGUNA_ATTN_WMMA (prefill; unused
//!       here since this harness prefills token-by-token via the decode path)
//!
//! Env:
//!   LAGUNA_GEN_N        tokens to generate (default 40)
//!   LAGUNA_GEN_PROMPT   optional: override the hardcoded ids with a text prompt
//!                       tokenized by our BPE (NOT oracle-aligned — for probing)

use std::io::Write;

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna_het::LagunaHetModel;

const GGUF_DEFAULT: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";
const DEFAULT_HOT_EXPERTS: &str = "artifacts/laguna_hot_experts_k8.txt";

// Oracle-aligned prompt tokens (llama-tokenize on the Q4_K_M GGUF). Leading 2 is
// 〈|EOS|〉 doubling as BOS. Text:
// "The capital of France is Paris. The capital of Germany is Berlin. The capital
//  of Japan is Tokyo. The capital of Italy is"
const PROMPT_TOKENS: &[i32] = &[
    2, 785, 9626, 377, 15360, 395, 22345, 83, 524, 9626, 377, 14220, 395, 30778, 83, 524, 9626,
    377, 16958, 395, 38798, 83, 524, 9626, 377, 22532, 395,
];

// Laguna control-token ids (mirror laguna_chat / the glm_thinking_v8 template).
const TOK_EOS: i32 = 2;
const TOK_THINK_BEGIN: i32 = 18;
const TOK_THINK_END: i32 = 19;
const TOK_ASSISTANT: i32 = 23;
const TOK_EOT: i32 = 24;

const DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. \
You are here to be helpful to users through natural language conversations.";

/// Build the first-turn templated prompt (glm_thinking_v8) around `user_msg`,
/// mirroring laguna_chat::TurnBuilder. `think` opens with <think> vs </think>.
fn build_templated(vocab: &BpeVocab, user_msg: &str, system: &str, think: bool) -> Vec<i32> {
    let mut out = Vec::new();
    out.push(TOK_EOS);
    let mut text = String::new();
    if !system.is_empty() {
        text.push_str("<system>");
        text.push_str(system);
        text.push_str("</system>\n");
    }
    text.push_str("<user>");
    text.push_str(user_msg);
    text.push_str("</user>\n");
    out.extend(vocab.encode_laguna_opts(&text, false));
    out.push(TOK_ASSISTANT);
    out.push(if think { TOK_THINK_BEGIN } else { TOK_THINK_END });
    out
}

fn build_byte_decoder() -> std::collections::HashMap<char, u8> {
    let mut bs: Vec<u32> = (b'!' as u32..=b'~' as u32)
        .chain(0xA1..=0xAC)
        .chain(0xAE..=0xFF)
        .collect();
    let mut cs = bs.clone();
    let mut n = 0u32;
    for b in 0u32..256 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (char::from_u32(c).unwrap(), b as u8))
        .collect()
}

fn detok(token_bytes: &[u8], dec: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let s = std::str::from_utf8(token_bytes).unwrap_or("");
    let mut out = Vec::with_capacity(token_bytes.len());
    for ch in s.chars() {
        if let Some(&b) = dec.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
        }
    }
    out
}

fn pick(arch_prefix: &str) -> eyre::Result<Device> {
    Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with(arch_prefix))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no {arch_prefix} device found"))
}

fn argmax(logits: &[f32]) -> usize {
    logits
        .iter()
        .enumerate()
        .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
            if v > bv {
                (i, v)
            } else {
                (bi, bv)
            }
        })
        .0
}

fn top5(logits: &[f32]) -> Vec<(usize, f32)> {
    let mut ord: Vec<usize> = (0..logits.len()).collect();
    ord.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal));
    ord.iter().take(5).map(|&i| (i, logits[i])).collect()
}

fn main() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();
    let gguf_path = std::env::args().nth(1).unwrap_or_else(|| GGUF_DEFAULT.to_string());
    let n_gen: usize = std::env::var("LAGUNA_GEN_N").ok().and_then(|s| s.parse().ok()).unwrap_or(40);

    if std::env::var_os("LAGUNA_HOT_EXPERTS_DGPU").is_none()
        && std::path::Path::new(DEFAULT_HOT_EXPERTS).exists()
    {
        std::env::set_var("LAGUNA_HOT_EXPERTS_DGPU", DEFAULT_HOT_EXPERTS);
    }

    let g = Gguf::open(&gguf_path)?;
    let vocab = BpeVocab::from_gguf(&g)?;
    let byte_decoder = build_byte_decoder();

    // Prompt tokens. Modes:
    //   LAGUNA_GEN_TEMPLATE=1 : chat template (glm_thinking_v8) around
    //       LAGUNA_GEN_USERMSG (in-distribution; matches the oracle conv mode).
    //   LAGUNA_GEN_PROMPT=<text> : raw BPE of the text (+BOS).
    //   default : the hardcoded oracle-aligned capitals tokens (raw completion).
    let prompt: Vec<i32> = if std::env::var("LAGUNA_GEN_TEMPLATE").as_deref() == Ok("1") {
        let user = std::env::var("LAGUNA_GEN_USERMSG")
            .unwrap_or_else(|_| "What is the capital of Italy?".to_string());
        let system = std::env::var("LAGUNA_GEN_SYSTEM").unwrap_or_else(|_| DEFAULT_SYSTEM.to_string());
        let think = std::env::var("LAGUNA_GEN_THINK").as_deref() == Ok("1");
        build_templated(&vocab, &user, &system, think)
    } else {
        match std::env::var("LAGUNA_GEN_PROMPT") {
            Ok(text) => vocab.encode_laguna_opts(&text, true),
            Err(_) => PROMPT_TOKENS.to_vec(),
        }
    };
    let _ = (TOK_THINK_END, TOK_EOT);

    // Report the attention config in effect (read the same env the model reads).
    let dec_variant = std::env::var("LAGUNA_DECODE_ATTN").unwrap_or_else(|_| "(default hg)".into());
    let dec_naive = std::env::var("LAGUNA_DECODE_ATTN_NAIVE").unwrap_or_default();
    let min_kv = std::env::var("LAGUNA_DECODE_FLASH_MIN_KV").unwrap_or_default();
    eprintln!(
        "=== CONFIG: LAGUNA_DECODE_ATTN={dec_variant} LAGUNA_DECODE_ATTN_NAIVE={dec_naive:?} \
         LAGUNA_DECODE_FLASH_MIN_KV={min_kv:?}  prompt_len={}  n_gen={n_gen}",
        prompt.len()
    );

    let dgpu = pick("gfx1201")?;
    let igpu = pick("gfx1151")?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;

    let t0 = std::time::Instant::now();
    let mut model = LagunaHetModel::load(&gguf_path, dgpu, &dgpu_arch, igpu, &igpu_arch, 8192)?;
    eprintln!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    model.reset();
    let mut pos: usize = 0;

    // TEACHER-FORCED mode: feed an exact token-id sequence (one id per line, e.g.
    // from llama-tokenize) and print ONLY the last-position logits (sum, argmax,
    // top-5). No generation — a clean forward-pass checksum vs the oracle
    // (llama-eval-callback result_output) that is immune to greedy divergence and
    // chat-template mismatch. Used to validate SWA windowing at ctx > 512.
    if let Ok(idfile) = std::env::var("LAGUNA_GEN_IDS_FILE") {
        let ids: Vec<i32> = std::fs::read_to_string(&idfile)?
            .lines()
            .filter_map(|l| l.trim().parse::<i32>().ok())
            .collect();
        eprintln!("teacher-forced: {} ids from {idfile}", ids.len());
        let last = ids.len() - 1;
        let mut sum = 0f64;
        let mut logits = Vec::new();
        for (i, &tok) in ids.iter().enumerate() {
            if i == last {
                logits = model.forward_logits_full(tok as usize, i)?;
            } else {
                model.forward_no_logits(tok as usize, i)?;
            }
        }
        for &l in &logits {
            sum += l as f64;
        }
        let am = argmax(&logits);
        let t5 = top5(&logits);
        let swa_off = std::env::var("LAGUNA_SWA_OFF").as_deref() == Ok("1");
        println!(
            "TEACHER-FORCED  swa_off={swa_off}  n_ids={}  result_output_sum={sum:.3}  argmax={am} (logit {:.4})",
            ids.len(),
            logits[am]
        );
        let t5s: String = t5.iter().map(|(i, v)| format!("{i}:{v:.3}")).collect::<Vec<_>>().join("  ");
        println!("  top5: {t5s}");
        return Ok(());
    }

    // --- prefill (token-by-token via the decode path) ---
    let last = prompt.len() - 1;
    let mut next: i32 = 0;
    for (i, &tok) in prompt.iter().enumerate() {
        let p = pos + i; // absolute KV position (was incorrectly fixed at `pos`)
        if i == last {
            let logits = model.forward_logits_full(tok as usize, p)?;
            next = argmax(&logits) as i32;
        } else {
            model.forward_no_logits(tok as usize, p)?;
        }
    }
    pos += prompt.len();

    // --- greedy decode, printing per-step diagnostics ---
    println!("STEP  POS   TOKEN_ID  TEXT                 TOP5(id:logit)");
    let mut gen_ids: Vec<i32> = Vec::new();
    let mut gen_text: Vec<u8> = Vec::new();
    for step in 0..n_gen {
        let tid = next;
        gen_ids.push(tid);
        let text = vocab.token_text(tid).map(|b| detok(b, &byte_decoder)).unwrap_or_default();
        gen_text.extend_from_slice(&text);
        let text_disp = String::from_utf8_lossy(&text).replace('\n', "\\n");

        let logits = model.forward_logits_full(tid as usize, pos)?;
        pos += 1;
        let t5 = top5(&logits);
        let t5s: String = t5
            .iter()
            .map(|(i, v)| format!("{i}:{v:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!("{step:>4} {:>5} {tid:>9}  {text_disp:<20}  {t5s}", pos - 1);
        next = argmax(&logits) as i32;
    }

    println!("\n=== GENERATED TOKEN IDS ({}): {:?}", gen_ids.len(), gen_ids);
    print!("=== GENERATED TEXT: ");
    std::io::stdout().write_all(&gen_text).ok();
    println!();
    Ok(())
}
