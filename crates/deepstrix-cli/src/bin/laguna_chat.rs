//! Interactive multi-turn chat REPL for Laguna-S-2.1 (poolside).
//!
//! Loads the Q4_K_M GGUF split across the two GPUs via [`LagunaHetModel`]
//! (attention + non-expert weights on the dGPU, the routed experts resident on
//! the iGPU) with the server het default (K=8 hot experts on the dGPU), applies
//! the `laguna_glm_thinking_v8` chat template, prefills the templated prompt,
//! then greedily decodes the assistant reply token-by-token, streaming to
//! stdout. Conversation history lives in the KV cache, so prefill on turn N only
//! covers the new user message + role markers, not the whole transcript.
//!
//! Usage (run from the repo root so the hot-expert artifact resolves):
//!   nix develop --command cargo run --release -p deepstrix-cli \
//!       --bin laguna-chat -- [gguf]
//! `gguf` defaults to the int4 Q4_K_M model path.
//!
//! Env overrides:
//!   LAGUNA_CHAT_KV_MAX   total KV slots allocated up front (default 8192)
//!   LAGUNA_CHAT_MAX_NEW  cap on tokens per assistant turn  (default 1024)
//!   LAGUNA_CHAT_SYSTEM   system prompt (default: the poolside template default)
//!   LAGUNA_CHAT_NOCOLOR  set to disable grey-on-think ANSI coloring
//!   LAGUNA_CHAT_TEMP     sampling temperature (default 0.7; 0 = greedy argmax —
//!                        greedy on this thinking model degenerates into
//!                        repetition, so a small temperature is the default)
//!   LAGUNA_CHAT_TOP_P    nucleus top-p (default 0.95; 1.0 = off)
//!   LAGUNA_CHAT_TOP_K    top-k cutoff (default 0 = off)
//!   LAGUNA_CHAT_SEED     u64 RNG seed (default 0xD5C0DE, reproducible)
//!   LAGUNA_HOT_EXPERTS_DGPU  hot-expert residency file (defaults to the K=8
//!                            artifact — the server het default)
//!
//! The template (`laguna_glm_thinking_v8`) renders role markers as a MIX of
//! control tokens and literal text: `<assistant>` (23), `</assistant>` (24 =
//! eot / the stop), `<think>` (18) and `</think>` (19) are single CONTROL /
//! USER_DEFINED vocab tokens injected by id, while `<system>`/`</system>` and
//! `<user>`/`</user>` are plain markup that gets BPE-encoded like any other
//! text. The leading `〈|EOS|〉` (2) doubles as BOS. Generation stops on token
//! 24 (`</assistant>`) or 2 (EOS).
//!
//! REPL commands:
//!   /think    enable chain-of-thought for subsequent turns
//!   /nothink  disable (default)
//!   /reset    clear the KV cache / start a fresh session
//!   /quit     exit
//!   (EOF)     exit

use std::io::{BufRead, Write};

use color_eyre::eyre::{self, eyre};
use v4flash_core::gguf::Gguf;
use v4flash_core::tokenizer::BpeVocab;
use v4flash_hip::Device;
use v4flash_kernels::laguna_het::LagunaHetModel;

const GGUF_DEFAULT: &str = "/persist/lumi/models/laguna-s-2.1-int4/laguna-s-2.1-Q4_K_M.gguf";

// Laguna control-token ids (from GGUF tokenizer.ggml.token_type). These are the
// role/thinking markers the template injects by id (not BPE).
const TOK_EOS: i32 = 2; // 〈|EOS|〉 — also the leading BOS
const TOK_THINK_BEGIN: i32 = 18; // <think>
const TOK_THINK_END: i32 = 19; // </think>
const TOK_ASSISTANT: i32 = 23; // <assistant>
const TOK_EOT: i32 = 24; // </assistant> — primary generation stop

/// Template default system message (`laguna_glm_thinking_v8`). Injected unless
/// `LAGUNA_CHAT_SYSTEM` overrides it (empty string opts out of the block).
const DEFAULT_SYSTEM: &str = "You are a helpful, conversationally-fluent assistant made by Poolside. \
You are here to be helpful to users through natural language conversations.";

/// Default K=8 hot-expert residency artifact (the server het default).
const DEFAULT_HOT_EXPERTS: &str = "artifacts/laguna_hot_experts_k8.txt";

/// GPT-2 byte<->unicode table for detokenizing token bytes back to raw bytes.
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

fn pick_dgpu() -> eyre::Result<Device> {
    Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1201"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1201 (9070 XT dGPU) device found"))
}

fn pick_igpu() -> eyre::Result<Device> {
    Device::all()?
        .into_iter()
        .find(|d| {
            d.properties()
                .map(|p| p.gcn_arch_name.starts_with("gfx1151"))
                .unwrap_or(false)
        })
        .ok_or_else(|| eyre!("no gfx1151 (Strix iGPU) device found"))
}

/// Per-turn token builder for `laguna_glm_thinking_v8`. Interleaves BPE'd markup
/// text with the injected control-token ids, mirroring how llama.cpp tokenizes
/// the rendered template (special tokens split out, surrounding markup BPE'd).
struct TurnBuilder {
    first_turn: bool,
    system: Option<String>,
}

impl TurnBuilder {
    fn build(&mut self, vocab: &BpeVocab, user_msg: &str, think: bool) -> Vec<i32> {
        let mut out = Vec::new();
        // The middle text span between control tokens — BPE'd as one unit so
        // merges match llama.cpp's per-span tokenization.
        let mut text = String::new();
        if self.first_turn {
            out.push(TOK_EOS); // leading 〈|EOS|〉 (BOS)
            if let Some(sys) = &self.system {
                if !sys.is_empty() {
                    text.push_str("<system>");
                    text.push_str(sys);
                    text.push_str("</system>\n");
                }
            }
            self.first_turn = false;
        } else {
            // Closes the previous `</assistant>` (24, already in KV) before the
            // next `<user>` block, matching the template's trailing newline.
            text.push('\n');
        }
        text.push_str("<user>");
        text.push_str(user_msg);
        text.push_str("</user>\n");
        // BPE the markup span; no implicit BOS (we control the leading token).
        out.extend(vocab.encode_laguna_opts(&text, false));
        out.push(TOK_ASSISTANT);
        // Open the generating turn with <think> (thinking) or </think> (direct).
        out.push(if think { TOK_THINK_BEGIN } else { TOK_THINK_END });
        out
    }
}

/// Streaming printer: suppresses the structural `<think>`/`</think>` markers,
/// greys the thinking span, and detokenizes content bytes to stdout.
struct Printer {
    use_color: bool,
    in_think: bool,
    color_open: bool,
    last_was_newline: bool,
}

impl Printer {
    fn new(use_color: bool, start_in_think: bool) -> Self {
        Self { use_color, in_think: start_in_think, color_open: false, last_was_newline: false }
    }

    fn emit<W: Write>(&mut self, w: &mut W, token: i32, text: Option<&[u8]>) -> std::io::Result<()> {
        if token == TOK_THINK_BEGIN {
            self.in_think = true;
            self.set_grey(w)?;
            return Ok(());
        }
        if token == TOK_THINK_END {
            self.in_think = false;
            self.reset_color(w)?;
            if !self.last_was_newline {
                w.write_all(b"\n")?;
                self.last_was_newline = true;
                w.flush()?;
            }
            return Ok(());
        }
        if let Some(bytes) = text {
            if !bytes.is_empty() {
                if self.in_think {
                    self.set_grey(w)?;
                }
                w.write_all(bytes)?;
                self.last_was_newline = bytes.last() == Some(&b'\n');
                w.flush()?;
            }
        }
        Ok(())
    }

    fn set_grey<W: Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        if self.use_color && !self.color_open {
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
        }
        w.flush()
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn env_f32(key: &str, default: f32) -> f32 {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

/// Host-side sampler: temperature + top-k + nucleus (top-p), seeded. `temp <= 0`
/// falls back to greedy argmax. Operates on the raw logit vector returned by
/// `forward_logits_full`.
struct Sampler {
    temp: f32,
    top_p: f32,
    top_k: usize,
    state: u64, // SplitMix64 PRNG state
}

impl Sampler {
    fn new(temp: f32, top_p: f32, top_k: usize, seed: u64) -> Self {
        Self { temp, top_p, top_k, state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn sample(&mut self, logits: &[f32]) -> usize {
        if self.temp <= 0.0 {
            return logits
                .iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                    if v > bv { (i, v) } else { (bi, bv) }
                })
                .0;
        }
        // Rank by logit descending. top-k caps the candidate set; a modest cap
        // also keeps the sort cheap without changing behavior when top_k=0.
        let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
        idx.sort_unstable_by(|&a, &b| {
            logits[b as usize].partial_cmp(&logits[a as usize]).unwrap_or(std::cmp::Ordering::Equal)
        });
        let k = if self.top_k == 0 { idx.len() } else { self.top_k.min(idx.len()) };
        let cand = &idx[..k];

        // Temperature-scaled softmax over the candidate set.
        let max_logit = logits[cand[0] as usize];
        let mut probs: Vec<f32> = cand
            .iter()
            .map(|&i| ((logits[i as usize] - max_logit) / self.temp).exp())
            .collect();
        let sum: f32 = probs.iter().sum();
        for p in &mut probs {
            *p /= sum;
        }

        // Nucleus (top-p): keep the smallest prefix whose cumulative prob >= p.
        let mut cutoff = probs.len();
        if self.top_p < 1.0 {
            let mut cum = 0.0f32;
            for (j, &p) in probs.iter().enumerate() {
                cum += p;
                if cum >= self.top_p {
                    cutoff = j + 1;
                    break;
                }
            }
        }
        let renorm: f32 = probs[..cutoff].iter().sum();
        let r = self.next_f32() * renorm;
        let mut acc = 0.0f32;
        for j in 0..cutoff {
            acc += probs[j];
            if r <= acc {
                return cand[j] as usize;
            }
        }
        cand[cutoff - 1] as usize
    }
}

fn main() -> eyre::Result<()> {
    let _ = v4flash_hip::install_panic_handler();
    let gguf_path = std::env::args().nth(1).unwrap_or_else(|| GGUF_DEFAULT.to_string());

    // Default to the K=8 hot-expert residency (server het default) unless the
    // caller already set it. LagunaHetModel reads this env at load.
    if std::env::var_os("LAGUNA_HOT_EXPERTS_DGPU").is_none()
        && std::path::Path::new(DEFAULT_HOT_EXPERTS).exists()
    {
        std::env::set_var("LAGUNA_HOT_EXPERTS_DGPU", DEFAULT_HOT_EXPERTS);
    }

    let n_kv_max = env_u32("LAGUNA_CHAT_KV_MAX", 8192) as usize;
    let max_new = env_u32("LAGUNA_CHAT_MAX_NEW", 1024) as usize;
    let use_color = std::env::var_os("LAGUNA_CHAT_NOCOLOR").is_none();
    let system = match std::env::var("LAGUNA_CHAT_SYSTEM") {
        Ok(s) => Some(s), // may be "" to opt out of the system block
        Err(_) => Some(DEFAULT_SYSTEM.to_string()),
    };
    let temp = env_f32("LAGUNA_CHAT_TEMP", 0.7);
    let top_p = env_f32("LAGUNA_CHAT_TOP_P", 0.95);
    let top_k = env_u32("LAGUNA_CHAT_TOP_K", 0) as usize;
    let seed = std::env::var("LAGUNA_CHAT_SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(0x00D5C0DE_u64);
    let mut sampler = Sampler::new(temp, top_p, top_k, seed);
    eprintln!("sampler: temp={temp} top_p={top_p} top_k={top_k} seed=0x{seed:x}");

    if !std::path::Path::new(&gguf_path).exists() {
        return Err(eyre!("model not found: {gguf_path}"));
    }
    eprintln!("loading Laguna-S-2.1 from {gguf_path}…");
    let g = Gguf::open(&gguf_path)?;
    let vocab = BpeVocab::from_gguf(&g)?;

    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    let dgpu_arch = dgpu.properties()?.gcn_arch_name;
    let igpu_arch = igpu.properties()?.gcn_arch_name;
    eprintln!("dGPU={dgpu_arch} (id={})  iGPU={igpu_arch} (id={})", dgpu.id, igpu.id);
    if let Ok(hot) = std::env::var("LAGUNA_HOT_EXPERTS_DGPU") {
        eprintln!("hot experts: {hot}");
    }

    let t0 = std::time::Instant::now();
    let mut model =
        LagunaHetModel::load(&gguf_path, dgpu, &dgpu_arch, igpu, &igpu_arch, n_kv_max)?;
    eprintln!("model loaded in {:.1}s  (KV: {n_kv_max} slots)", t0.elapsed().as_secs_f32());

    let byte_decoder = build_byte_decoder();
    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut line = String::new();

    let mut think_mode = false;
    let mut pos: usize = 0;
    let mut builder = TurnBuilder { first_turn: true, system: system.clone() };
    model.reset();

    eprintln!("ready. /think, /nothink, /reset, /quit.");
    loop {
        print!("\n\x1b[1mUser:\x1b[0m ");
        std::io::stdout().flush().ok();
        line.clear();
        if stdin.read_line(&mut line)? == 0 {
            eprintln!("\n<EOF>");
            break;
        }
        let user_msg = line.trim_end_matches(['\n', '\r']);
        let trimmed = user_msg.trim();
        if trimmed.is_empty() {
            continue;
        }
        match trimmed {
            "/quit" | "/exit" => break,
            "/think" => {
                think_mode = true;
                eprintln!("[think mode: on]");
                continue;
            }
            "/nothink" => {
                think_mode = false;
                eprintln!("[think mode: off]");
                continue;
            }
            "/reset" => {
                model.reset();
                pos = 0;
                builder = TurnBuilder { first_turn: true, system: system.clone() };
                eprintln!("[session reset]");
                continue;
            }
            _ => {}
        }

        let turn = builder.build(&vocab, user_msg, think_mode);
        if std::env::var_os("LAGUNA_CHAT_DEBUG").is_some() {
            eprintln!("[debug] turn tokens ({}): {:?}", turn.len(), turn);
            let dbg: String = turn
                .iter()
                .map(|&t| {
                    let s = vocab
                        .token_text(t)
                        .map(|b| String::from_utf8_lossy(&detok(b, &byte_decoder)).into_owned())
                        .unwrap_or_default();
                    format!("{t}:{s:?}")
                })
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("[debug] {dbg}");
        }
        if pos + turn.len() + max_new > n_kv_max {
            eprintln!(
                "\n[context would exceed KV cap ({n_kv_max}); raise LAGUNA_CHAT_KV_MAX or /reset]"
            );
            break;
        }

        // --- prefill this turn's tokens incrementally at the running position
        //     (token-by-token — the oracle-validated forward path). ---
        let t_pref = std::time::Instant::now();
        let last = turn.len() - 1;
        let mut next: i32 = 0;
        for (i, &tok) in turn.iter().enumerate() {
            let p = pos + i;
            if i == last {
                let logits = model.forward_logits_full(tok as usize, p)?;
                next = sampler.sample(&logits) as i32;
            } else {
                model.forward_no_logits(tok as usize, p)?;
            }
        }
        pos += turn.len();
        let pref_s = t_pref.elapsed().as_secs_f64();
        eprintln!(
            "[prefill {} tok in {:.2}s = {:.1} tok/s]",
            turn.len(),
            pref_s,
            turn.len() as f64 / pref_s.max(1e-9)
        );

        // --- greedy decode loop, streaming ---
        print!("\x1b[1mAssistant:\x1b[0m ");
        std::io::stdout().flush().ok();
        let mut printer = Printer::new(use_color, think_mode);
        let mut stdout = std::io::stdout().lock();

        let t_dec = std::time::Instant::now();
        let mut n_decoded = 0usize;
        let mut stop_reason = "max_new";
        for _ in 0..max_new {
            if next == TOK_EOT || next == TOK_EOS || next == TOK_ASSISTANT {
                // Keep the stop marker in KV so the next turn continues cleanly.
                model.forward_no_logits(next as usize, pos)?;
                pos += 1;
                stop_reason = if next == TOK_ASSISTANT { "role-marker" } else { "eot" };
                break;
            }
            let text: Option<Vec<u8>> =
                vocab.token_text(next).map(|b| detok(b, &byte_decoder));
            printer.emit(&mut stdout, next, text.as_deref()).ok();

            let logits = model.forward_logits_full(next as usize, pos)?;
            pos += 1;
            n_decoded += 1;
            next = sampler.sample(&logits) as i32;
            if pos >= n_kv_max {
                stop_reason = "kv_full";
                break;
            }
        }
        let dec_s = t_dec.elapsed().as_secs_f64();
        printer.finish(&mut stdout).ok();
        drop(stdout);
        eprintln!(
            "[decode {} tok in {:.2}s = {:.1} tok/s, {stop_reason}]",
            n_decoded,
            dec_s,
            n_decoded as f64 / dec_s.max(1e-9)
        );
    }
    Ok(())
}
