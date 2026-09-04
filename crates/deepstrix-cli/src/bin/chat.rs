//! Interactive multi-turn chat REPL for V4-Flash.
//!
//! Uses the heterogeneous (dGPU + iGPU) engine that the decode-perf work
//! has been tuned against. Wraps each user turn with the V4-Flash chat
//! template and streams the greedy-decoded assistant response token-by-
//! token. Conversation history lives in the KV cache (`HetModelState`),
//! so prefill on turn N only covers the new user message + role markers,
//! not the entire transcript.
//!
//! Usage:
//!   HIP_VISIBLE_DEVICES=0,1 deepstrix-chat <gguf>
//!
//! Env overrides:
//!   CHAT_KV_MAX   total KV slots to allocate up front (default 8192)
//!   CHAT_MAX_NEW  cap on tokens per assistant turn  (default 1024)
//!   CHAT_SYSTEM   system prompt prepended after BOS on the first turn
//!   CHAT_NOCOLOR  set to disable grey-on-think ANSI coloring
//!
//!   CHAT_SAMPLER  "argmax" | "multinomial" (default "multinomial" —
//!                 matches V4-Flash's recommended sampler).
//!   CHAT_TEMP     temperature for multinomial mode (default 1.0,
//!                 matches DeepSeek's published recipe). Set to 0.0 to
//!                 force argmax.
//!   CHAT_MIN_P    min-p threshold relative to most-likely token
//!                 (default 0.0 = off, matches the recipe). Try 0.05
//!                 for a long-tail prune.
//!   CHAT_SEED     u64 seed for the host PRNG that feeds the device
//!                 sampler. 0 = deterministic baseline. Defaults to
//!                 a fixed value (0xD5C0DE) so chat sessions are
//!                 reproducible across runs unless explicitly changed.
//!
//! REPL commands:
//!   /think     enable chain-of-thought for subsequent turns (0731 "low"
//!              effort: thinking on, no preamble)
//!   /nothink   disable (default)
//!   /thinkhigh enable 0731 "high" reasoning preamble (the pre-0731 ds4
//!              DS4_THINK_MAX text; ≥384K-ctx guidance, first turn only)
//!   /thinkmax  enable 0731 "max" reasoning preamble (≥384K-ctx
//!              guidance, first turn only)
//!   /load <path>  substitute file contents as one user turn
//!   /quit     exit
//!   (EOF)     exit
//!
//! Template applied per user turn (matches ds4_server.c render_chat_prompt_text
//! at the token-ID level — encoder doesn't recognize special-token text):
//!   (first turn only) <BOS> [effort preamble if /thinkhigh|/thinkmax] [CHAT_SYSTEM content]
//!   <｜User｜>{user_msg}<｜Assistant｜>{<think> if thinking else </think>}
//!   …model output… <EOS>
//! The model emits `</think>` itself at end-of-reasoning, then its answer.
//! Some V4-Flash training distributions also emit a trailing `</think>`
//! near end-of-turn; the token printer below suppresses display of all
//! `<think>` / `</think>` tokens regardless of position (matches ds4_cli.c
//! token_printer_process behavior) and only renders content tokens.

use std::io::{BufRead, Write};

use color_eyre::eyre::{self, eyre};
use v4flash_core::{tokenizer::BpeVocab, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchDgpuShared, BatchIgpuScratch, BatchIgpuShared, DgpuScratch, ExecMode,
    HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch, SampleMode, B_MAX,
};
use v4flash_kernels::sampler::SamplerRng;
use v4flash_kernels::RopeParams;

// V4-Flash chat-template special token IDs (from GGUF tokenizer.ggml.tokens).
const TOK_BOS: i32 = 0;              // <｜begin▁of▁sentence｜>
const TOK_EOS: i32 = 1;              // <｜end▁of▁sentence｜>
const TOK_USER: i32 = 128803;        // <｜User｜>
const TOK_ASSISTANT: i32 = 128804;   // <｜Assistant｜>
const TOK_THINK_BEGIN: i32 = 128821; // <think>
const TOK_THINK_END: i32 = 128822;   // </think>

/// V4-Flash 0731 "high" reasoning-effort preamble — byte-for-byte
/// `REASONING_EFFORT_PROMPTS["high"]` from the HF repo's
/// `encoding/encoding_dsv4.py`. This is the pre-0731 ds4.c
/// DS4_REASONING_EFFORT_MAX_PREFIX text, now the "high" level.
/// Used at /thinkhigh. Only safe at ≥384K ctx — preamble is large.
const REASONING_HIGH_PREFIX: &str = "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n\
You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n\
Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

/// V4-Flash 0731 "max" reasoning-effort preamble — byte-for-byte
/// `REASONING_EFFORT_PROMPTS["max"]` from `encoding_dsv4.py`.
/// Used at /thinkmax. Only safe at ≥384K ctx — preamble is large.
const REASONING_MAX_PREFIX: &str = "Reasoning Effort: Beyond maximum — exhaustive, relentless, and uncompromising.\n\
You MUST reason with the utmost depth and rigor, leaving absolutely nothing to chance: exhaustively decompose the problem into its most fundamental components, trace every causal chain to its root, and resolve the underlying cause rather than any surface symptom.\n\
Do not stop reasoning until you have independently verified the solution from multiple angles and are certain that no assumption remains unchecked and no error remains undiscovered.\n\n";

/// 0731 effort levels: On = "low" (thinking, no preamble), High/Max add
/// the corresponding preamble on the first turn.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ThinkMode { Off, On, High, Max }

impl ThinkMode {
    fn is_on(self) -> bool { !matches!(self, ThinkMode::Off) }
    fn prefill_tok(self) -> i32 {
        // ds4_server.c:1972: open the generating assistant turn with
        // <think> if think-on (any flavor), else </think>.
        if self.is_on() { TOK_THINK_BEGIN } else { TOK_THINK_END }
    }
}

/// Per-turn token-list builder. Matches the incremental path of
/// `render_chat_prompt_text` in ds4_server.c — emits BOS + system on the
/// first turn, then just [User|content|Assistant|think-open] each turn.
/// Assumes the assistant's previous EOS is already in KV from the last
/// decode loop (we forward EOS into KV on hit_eos — see decode loop).
struct ChatTurnBuilder {
    is_first_turn: bool,
    system_prompt: Option<String>,
}

impl ChatTurnBuilder {
    fn build(&mut self, vocab: &BpeVocab, user_msg: &str, think: ThinkMode) -> Vec<i32> {
        let mut out = Vec::new();
        if self.is_first_turn {
            out.push(TOK_BOS);
            // 0731: the selected effort's preamble goes at the very
            // beginning of the conversation — after BOS, before the
            // system message. First turn only.
            match think {
                ThinkMode::High => out.extend(vocab.encode(REASONING_HIGH_PREFIX)),
                ThinkMode::Max => out.extend(vocab.encode(REASONING_MAX_PREFIX)),
                ThinkMode::Off | ThinkMode::On => {}
            }
            if let Some(sys) = &self.system_prompt {
                if !sys.is_empty() {
                    out.extend(vocab.encode(sys));
                }
            }
            self.is_first_turn = false;
        }
        out.push(TOK_USER);
        out.extend(vocab.encode(user_msg));
        out.push(TOK_ASSISTANT);
        out.push(think.prefill_tok());
        out
    }
}

/// Streaming token printer with V4-Flash special-token handling. Suppresses
/// `<think>` / `</think>` from display (they're structural markers, not
/// content), greys content emitted between an unmatched `<think>` and its
/// `</think>` if `format_thinking` is on, and signals turn-end when the
/// model emits any of the role-boundary tokens. Token-ID level — no need
/// for ds4_cli.c's byte-state-machine because each special token is one
/// token in our vocab.
struct ChatTokenPrinter {
    format_thinking: bool,
    use_color: bool,
    in_think: bool,
    color_open: bool,
    last_was_newline: bool,
}

impl ChatTokenPrinter {
    fn new(format_thinking: bool, use_color: bool, start_in_think: bool) -> Self {
        Self {
            format_thinking, use_color,
            in_think: start_in_think,
            color_open: false,
            last_was_newline: false,
        }
    }

    /// Returns `true` when caller should treat this token as end-of-turn
    /// (EOS, or a hallucinated role marker mid-decode).
    fn push_token<W: Write>(
        &mut self, w: &mut W, token: i32, text: Option<&[u8]>,
    ) -> std::io::Result<bool> {
        // Hard turn-end markers.
        if token == TOK_EOS {
            self.reset_color(w)?;
            return Ok(true);
        }
        // Hallucinated next-turn markers — treat as turn-end (model is
        // pretending to start a new turn). Don't display.
        if token == TOK_USER || token == TOK_ASSISTANT || token == TOK_BOS {
            self.reset_color(w)?;
            return Ok(true);
        }
        // Structural — suppress display, toggle think-mode rendering state.
        if token == TOK_THINK_BEGIN {
            self.in_think = true;
            self.set_grey(w)?;
            return Ok(false);
        }
        if token == TOK_THINK_END {
            self.in_think = false;
            self.reset_color(w)?;
            // Insert a separating newline once thinking ends.
            if !self.last_was_newline {
                w.write_all(b"\n")?;
                self.last_was_newline = true;
                w.flush()?;
            }
            return Ok(false);
        }
        // Content token: render it.
        if let Some(bytes) = text {
            if !bytes.is_empty() {
                if self.in_think { self.set_grey(w)?; }
                w.write_all(bytes)?;
                self.last_was_newline = bytes.last() == Some(&b'\n');
                w.flush()?;
            }
        }
        Ok(false)
    }

    fn set_grey<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        if self.use_color && self.format_thinking && !self.color_open {
            w.write_all(b"\x1b[90m")?;
            self.color_open = true;
        }
        Ok(())
    }

    fn reset_color<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        if self.color_open {
            w.write_all(b"\x1b[0m")?;
            self.color_open = false;
        }
        Ok(())
    }

    fn finish<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        self.reset_color(w)?;
        if !self.last_was_newline {
            w.write_all(b"\n")?;
            self.last_was_newline = true;
        }
        w.flush()?;
        Ok(())
    }
}

// V4-Flash RoPE constants (from GGUF metadata + ds4.c).
const ROPE_FREQ_BASE_DENSE: f32 = 10000.0;
const ROPE_FREQ_BASE_COMP: f32 = 160000.0;
const ROPE_SCALE_FACTOR: f32 = 16.0;
const ROPE_ORIG_CTX: u64 = 65536;
const ROPE_BETA_FAST: f32 = 32.0;
const ROPE_BETA_SLOW: f32 = 1.0;

fn pick_dgpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1201") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1201 (9070 XT) device found"))
}

fn pick_igpu() -> eyre::Result<Device> {
    for d in Device::all()? {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(d);
        }
    }
    Err(eyre!("no gfx1151 (Strix iGPU) device found"))
}

fn rope_for_layer(layer: i32) -> RopeParams {
    let ratio = COMPRESS_RATIOS[layer as usize];
    let compressed = ratio != 0;
    let freq_base = if compressed { ROPE_FREQ_BASE_COMP } else { ROPE_FREQ_BASE_DENSE };
    let freq_scale = if compressed { 1.0 / ROPE_SCALE_FACTOR } else { 1.0 };
    let ext_factor = if compressed && ROPE_SCALE_FACTOR > 1.0 { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    let n_ctx_orig = if compressed { ROPE_ORIG_CTX } else { 0 };
    let floats = [
        freq_base, freq_scale, ext_factor, attn_factor,
        ROPE_BETA_FAST, ROPE_BETA_SLOW,
    ];
    RopeParams::from_dump_blob(&floats, n_ctx_orig).expect("valid rope params")
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let s: u32 = (sign as u32) << 31;
    let f32_bits: u32 = match exp {
        0 if mant == 0 => s,
        0 => {
            let mantissa = mant as f32 / 1024.0;
            let v = mantissa * (1.0 / (1u64 << 14) as f32);
            return if sign == 1 { -v } else { v };
        }
        0x1f => s | 0x7f800000 | ((mant as u32) << 13),
        _ => s | ((exp as u32 + 112) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

fn embed_lookup(
    token_embd_bytes: &[u8],
    dtype: v4flash_core::gguf::GgufType,
    token_id: i32,
    out: &mut [f32],
) {
    v4flash_kernels::embed::embed_lookup(token_embd_bytes, dtype, token_id, out)
        .expect("embed_lookup: dtype validated at load");
}

fn build_gpt2_byte_decoder() -> std::collections::HashMap<char, u8> {
    let printable: Vec<u8> = (b'!'..=b'~')
        .chain(0xA1u8..=0xACu8)
        .chain(0xAEu8..=0xFFu8)
        .collect();
    let mut bs: Vec<u8> = printable.clone();
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0u8..=255 {
        if !printable.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut m = std::collections::HashMap::with_capacity(256);
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        if let Some(ch) = char::from_u32(c) {
            m.insert(ch, b);
        }
    }
    m
}

fn gpt2_decode_token(token_bytes: &[u8], dec: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let s = std::str::from_utf8(token_bytes).unwrap_or("");
    let mut out = Vec::with_capacity(token_bytes.len());
    for ch in s.chars() {
        if let Some(&b) = dec.get(&ch) {
            out.push(b);
        } else {
            let mut buf = [0u8; 4];
            let s = ch.encode_utf8(&mut buf);
            out.extend_from_slice(s.as_bytes());
        }
    }
    out
}

fn main() -> eyre::Result<()> {
    install_panic_handler()?;
    let mut args = std::env::args().skip(1);
    let gguf_path = args
        .next()
        .ok_or_else(|| eyre!("usage: deepstrix-chat <gguf>"))?;

    let n_kv_max: u32 = std::env::var("CHAT_KV_MAX")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8192);
    let max_new: usize = std::env::var("CHAT_MAX_NEW")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1024);

    let sampler_kind = std::env::var("CHAT_SAMPLER").unwrap_or_else(|_| "multinomial".to_string());
    let temperature: f32 = std::env::var("CHAT_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    let min_p_rel: f32 = std::env::var("CHAT_MIN_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let seed: u64 = std::env::var("CHAT_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0x00D5C0DE_u64);

    let sample_mode = match sampler_kind.as_str() {
        "argmax" => SampleMode::Argmax,
        "multinomial" => {
            if temperature <= 0.0 {
                SampleMode::Argmax
            } else {
                SampleMode::Multinomial { temperature, min_p_rel }
            }
        }
        other => return Err(eyre!("CHAT_SAMPLER={other} (want \"argmax\" or \"multinomial\")")),
    };
    let mut rng = SamplerRng::new(seed);
    eprintln!(
        "sampler: {} (T={}, min_p_rel={}, seed=0x{:x})",
        sampler_kind, temperature, min_p_rel, seed
    );

    eprintln!("loading model from {gguf_path}…");
    let gguf = MappedGguf::open(&gguf_path)?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;

    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!("dGPU={} (id={})  iGPU={} (id={})", dgpu_arch, dgpu.id, igpu_arch, igpu.id);

    let token_embd_t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("missing token_embd.weight"))?;
    if !v4flash_kernels::weight_contract::TOKEN_EMBD_ALLOWED.contains(&token_embd_t.dtype) {
        return Err(eyre!(
            "token_embd dtype {:?} unsupported (allowed: {:?})",
            token_embd_t.dtype,
            v4flash_kernels::weight_contract::TOKEN_EMBD_ALLOWED
        ));
    }
    let token_embd_dtype = token_embd_t.dtype;
    let token_embd_bytes = gguf.read_tensor(token_embd_t)?;
    let token_embd_bytes: &[u8] = &token_embd_bytes;

    let rope = |layer: i32| -> eyre::Result<RopeParams> { Ok(rope_for_layer(layer)) };

    eprintln!("loading het weights (dGPU ~9 GiB + iGPU ~52 GiB)…");
    let t0 = std::time::Instant::now();
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope)?;
    eprintln!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let engine =
        HeterogeneousEngine::new(dgpu, &dgpu_arch, igpu, &igpu_arch, ExecMode::HetParallel)?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
    // Reused across turns. See [[m50-prefill-state]] for the architecture.
    // Two-lane pipelined prefill: each lane holds at most ceil(B_MAX/2)
    // rows of a chunk (forward_prompt_batch_v2_pipelined), so size the
    // per-lane scratch at that instead of the full chunk.
    let lane_rows = B_MAX.div_ceil(2);
    let mut bd_a = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_a = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    let mut bd_b = BatchDgpuScratch::alloc_rows(dgpu, lane_rows)?;
    let mut bi_b = BatchIgpuScratch::alloc_rows(igpu, lane_rows)?;
    // Shared set: one instance for both lanes, same row capacity.
    let mut sd = BatchDgpuShared::alloc_rows(dgpu, lane_rows)?;
    let mut si = BatchIgpuShared::alloc_rows(igpu, lane_rows)?;
    eprintln!("KV cache: {n_kv_max} slots");

    let byte_decoder = build_gpt2_byte_decoder();
    let mut residual = vec![0f32; HC_DIM as usize];

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut line = String::new();
    let mut pos: u32 = 0;
    let mut think_mode = ThinkMode::Off;

    let use_color = std::env::var_os("CHAT_NOCOLOR").is_none();
    let mut builder = ChatTurnBuilder {
        is_first_turn: true,
        system_prompt: std::env::var("CHAT_SYSTEM").ok(),
    };

    eprintln!("ready. /think, /nothink, /thinkhigh, /thinkmax, /load <file>, /quit.");
    loop {
        print!("\n\x1b[1mUser:\x1b[0m ");
        std::io::stdout().flush().ok();
        line.clear();
        let n = stdin.read_line(&mut line)?;
        if n == 0 {
            eprintln!("\n<EOF>");
            break;
        }
        let user_msg = line.trim_end_matches('\n').trim_end_matches('\r');
        if user_msg.trim().is_empty() {
            continue;
        }
        // Slash-command dispatch. /load is a multi-line escape: substitutes
        // the file's full contents (newlines preserved) as the user turn.
        let trimmed = user_msg.trim();
        let mut owned_msg = String::new();
        let effective_msg: &str = match trimmed {
            "/quit" => break,
            "/think" => {
                think_mode = ThinkMode::On;
                eprintln!("[think mode: on]");
                continue;
            }
            "/nothink" => {
                think_mode = ThinkMode::Off;
                eprintln!("[think mode: off]");
                continue;
            }
            "/thinkhigh" => {
                if builder.is_first_turn {
                    think_mode = ThinkMode::High;
                    eprintln!("[think mode: HIGH (reasoning preamble will be injected on first turn — \
                               needs ≥384K ctx per ds4 convention; CHAT_KV_MAX={n_kv_max})]");
                } else {
                    // Per ds4: the effort prefix is part of the system block
                    // which we already passed. Downgrading to on prevents a
                    // silent no-op.
                    think_mode = ThinkMode::On;
                    eprintln!("[think mode: on (HIGH prefix only takes effect on first turn)]");
                }
                continue;
            }
            "/thinkmax" => {
                if builder.is_first_turn {
                    think_mode = ThinkMode::Max;
                    eprintln!("[think mode: MAX (reasoning preamble will be injected on first turn — \
                               needs ≥384K ctx per ds4 convention; CHAT_KV_MAX={n_kv_max})]");
                } else {
                    // Per ds4: the effort prefix is part of the system block
                    // which we already passed. Downgrading to on prevents a
                    // silent no-op.
                    think_mode = ThinkMode::On;
                    eprintln!("[think mode: on (MAX prefix only takes effect on first turn)]");
                }
                continue;
            }
            cmd if cmd.starts_with("/load ") => {
                let path = cmd[6..].trim();
                match std::fs::read_to_string(path) {
                    Ok(s) => {
                        eprintln!("[loaded {} bytes from {path}]", s.len());
                        owned_msg = s;
                        owned_msg.as_str()
                    }
                    Err(e) => {
                        eprintln!("[/load {path}: {e}]");
                        continue;
                    }
                }
            }
            _ => user_msg,
        };
        let user_msg = effective_msg;

        // Build this turn's prefix via the shared template builder. First
        // turn includes BOS + (optional) HIGH/MAX effort preamble + system;
        // every turn adds [User|content|Assistant|(<think>|</think>)].
        let turn_tokens = builder.build(&vocab, user_msg, think_mode);

        if pos as usize + turn_tokens.len() + 1 > n_kv_max as usize {
            eprintln!(
                "\n[context exhausted: pos={pos}, +{} prefill, cap={n_kv_max}. \
                 raise CHAT_KV_MAX to continue.]",
                turn_tokens.len()
            );
            break;
        }

        // Build per-token input HC vectors for the batched prefill.
        let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(turn_tokens.len());
        for &tok in &turn_tokens {
            let mut v = vec![0f32; HC_DIM as usize];
            embed_lookup(token_embd_bytes, token_embd_dtype, tok, &mut v);
            input_hcs.push(v);
        }

        let t0 = std::time::Instant::now();
        let last_logits = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b,
            &mut sd, &mut si,
            &mut dgpu_scratch,
            &mut state, &weights,
            &input_hcs, &turn_tokens, pos,
            true, // last_only
            None, // stats
            None, // cancel
            None, // on_chunk_done
            None, // image_spans
        )?;
        pos += turn_tokens.len() as u32;
        let prefill_secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "[prefill: {} tok in {:.2}s = {:.1} tok/s]",
            turn_tokens.len(),
            prefill_secs,
            turn_tokens.len() as f64 / prefill_secs.max(1e-6)
        );

        if last_logits.len() != N_VOCAB as usize {
            return Err(eyre!(
                "prefill logits len {} != N_VOCAB {}",
                last_logits.len(), N_VOCAB
            ));
        }
        // dgpu_scratch.logits still holds the last-token logits from the
        // prefill head — sample on-device. (last_logits is the host copy
        // produced by forward_prefill_pipelined for symmetry; we ignore
        // it here so prefill and decode share one sampling path.)
        let _ = last_logits;
        let mut next = engine.sample_next(&mut dgpu_scratch, sample_mode, rng.next_f32())?;

        print!("\x1b[1mAssistant:\x1b[0m ");
        std::io::stdout().flush().ok();

        // The injected last prefill token was THINK_BEGIN/END (per builder),
        // so the printer starts in the right rendering state.
        let mut printer = ChatTokenPrinter::new(
            /*format_thinking=*/ true,
            /*use_color=*/ use_color,
            /*start_in_think=*/ think_mode.is_on(),
        );
        let mut stdout = std::io::stdout().lock();

        let t0 = std::time::Instant::now();
        let mut n_decoded = 0usize;
        let mut hit_eos = false;
        for _ in 0..max_new {
            // Render this token (returns true if turn-end marker).
            let text_owned: Option<Vec<u8>> = vocab
                .token_text(next)
                .map(|b| gpt2_decode_token(b, &byte_decoder));
            let end_of_turn = printer
                .push_token(&mut stdout, next, text_owned.as_deref())
                .ok()
                .unwrap_or(false);

            if end_of_turn {
                // Forward EOS into KV so next turn picks up after a clean
                // boundary. (Also true for hallucinated role markers — we
                // treat them as if the model meant to end the turn.)
                embed_lookup(token_embd_bytes, token_embd_dtype, TOK_EOS, &mut residual);
                engine.forward_token(
                    &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                    &residual, pos, TOK_EOS,
                )?;
                pos += 1;
                hit_eos = true;
                break;
            }

            embed_lookup(token_embd_bytes, token_embd_dtype, next, &mut residual);
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                &residual, pos, next,
            )?;
            pos += 1;
            n_decoded += 1;
            if pos >= n_kv_max {
                eprintln!("\n[KV cache full at pos={pos}, stopping turn]");
                break;
            }
            next = engine.sample_next(&mut dgpu_scratch, sample_mode, rng.next_f32())?;
        }
        let decode_secs = t0.elapsed().as_secs_f64();
        printer.finish(&mut stdout).ok();
        drop(stdout);
        eprintln!(
            "[decode: {} tok in {:.2}s = {:.1} tok/s{}]",
            n_decoded,
            decode_secs,
            n_decoded as f64 / decode_secs.max(1e-6),
            if hit_eos { ", EOS" } else { ", max_new" }
        );
    }
    // Drain both devices to idle before the buffers/streams drop. Without
    // this, a per-buffer hipFree's implicit SyncAllStreams can orphan an
    // in-flight cross-device wait and busy-spin forever at teardown.
    engine.shutdown()?;
    Ok(())
}
