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
//!
//! REPL commands:
//!   /think    enable chain-of-thought for subsequent turns
//!   /nothink  disable (default)
//!   /quit     exit
//!   (EOF)     exit
//!
//! Template applied per user turn (matches V4-Flash GGUF chat_template):
//!   (first turn only) <BOS>
//!   <｜User｜>{user_msg}<｜Assistant｜>{<think> if thinking else </think>}
//!   …model output… <EOS>
//! Picking `</think>` skips reasoning; `<think>` opens a reasoning block
//! that the model itself closes with `</think>` before answering.

use std::io::{BufRead, Write};

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, tokenizer::BpeVocab, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::config::{COMPRESS_RATIOS, HC_DIM, N_EMBD, N_HC, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch,
};
use v4flash_kernels::RopeParams;

// V4-Flash chat-template special token IDs (from GGUF tokenizer.ggml.tokens).
const TOK_BOS: i32 = 0;              // <｜begin▁of▁sentence｜>
const TOK_EOS: i32 = 1;              // <｜end▁of▁sentence｜>
const TOK_USER: i32 = 128803;        // <｜User｜>
const TOK_ASSISTANT: i32 = 128804;   // <｜Assistant｜>
const TOK_THINK_BEGIN: i32 = 128821; // <think>
const TOK_THINK_END: i32 = 128822;   // </think>

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

fn embed_lookup(token_embd_bytes: &[u8], token_id: i32, out: &mut [f32]) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    assert_eq!(out.len(), n_embd * n_hc);
    let row_off = (token_id as usize) * n_embd * 2;
    for i in 0..n_embd {
        let b0 = token_embd_bytes[row_off + i * 2];
        let b1 = token_embd_bytes[row_off + i * 2 + 1];
        let bits = u16::from_le_bytes([b0, b1]);
        out[i] = f16_to_f32(bits);
    }
    for h in 1..n_hc {
        let (head, tail) = out.split_at_mut(h * n_embd);
        let src = &head[0..n_embd];
        let dst = &mut tail[0..n_embd];
        dst.copy_from_slice(src);
    }
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

fn argmax(v: &[f32]) -> usize {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &x) in v.iter().enumerate() {
        if x > bv {
            bv = x;
            best = i;
        }
    }
    best
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
    if token_embd_t.dtype != GgufType::F16 {
        return Err(eyre!("token_embd dtype {:?} != F16", token_embd_t.dtype));
    }
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
    // Two-lane pipelined prefill scratches (~208 MB total). Reused
    // across turns. See [[m50-prefill-state]] for the architecture.
    let mut bd_a = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_a = BatchIgpuScratch::alloc(igpu)?;
    let mut bd_b = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_b = BatchIgpuScratch::alloc(igpu)?;
    eprintln!("KV cache: {n_kv_max} slots");

    let byte_decoder = build_gpt2_byte_decoder();
    let mut residual = vec![0f32; HC_DIM as usize];
    let mut logits_host = vec![0f32; N_VOCAB as usize];

    let stdin = std::io::stdin();
    let mut stdin = stdin.lock();
    let mut line = String::new();
    let mut pos: u32 = 0;
    let mut is_first_turn = true;
    let mut think_mode = false;

    eprintln!("ready. type a message and hit enter. /think, /nothink, /quit.");
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
        match user_msg.trim() {
            "/quit" => break,
            "/think" => {
                think_mode = true;
                eprintln!("[think mode ON]");
                continue;
            }
            "/nothink" => {
                think_mode = false;
                eprintln!("[think mode OFF]");
                continue;
            }
            _ => {}
        }

        // Build this turn's prefix.
        //   first turn:  [BOS, USER, ...encode(user), ASSISTANT, THINK_END]
        //   later turns: [USER, ...encode(user), ASSISTANT, THINK_END]
        let mut turn_tokens: Vec<i32> = Vec::new();
        if is_first_turn {
            turn_tokens.push(TOK_BOS);
            is_first_turn = false;
        }
        turn_tokens.push(TOK_USER);
        turn_tokens.extend(vocab.encode(user_msg));
        turn_tokens.push(TOK_ASSISTANT);
        turn_tokens.push(if think_mode { TOK_THINK_BEGIN } else { TOK_THINK_END });

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
            embed_lookup(token_embd_bytes, tok, &mut v);
            input_hcs.push(v);
        }

        let t0 = std::time::Instant::now();
        let last_logits = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b,
            &mut dgpu_scratch,
            &mut state, &weights,
            &input_hcs, &turn_tokens, pos,
            true, // last_only
            None, // stats
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
        logits_host.copy_from_slice(&last_logits);
        let mut next = argmax(&logits_host) as i32;

        print!("\x1b[1mAssistant:\x1b[0m ");
        std::io::stdout().flush().ok();

        let t0 = std::time::Instant::now();
        let mut n_decoded = 0usize;
        let mut hit_eos = false;
        for _ in 0..max_new {
            if next == TOK_EOS {
                // Forward EOS into KV so the next turn sees a clean boundary.
                embed_lookup(token_embd_bytes, next, &mut residual);
                engine.forward_token(
                    &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                    &residual, pos, next,
                )?;
                pos += 1;
                hit_eos = true;
                break;
            }
            if let Some(bytes) = vocab.token_text(next) {
                let raw = gpt2_decode_token(bytes, &byte_decoder);
                std::io::stdout().write_all(&raw).ok();
                std::io::stdout().flush().ok();
            } else {
                print!("<?{next}>");
            }
            embed_lookup(token_embd_bytes, next, &mut residual);
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
            dgpu_scratch.logits.copy_to_host(&mut logits_host)?;
            next = argmax(&logits_host) as i32;
        }
        let decode_secs = t0.elapsed().as_secs_f64();
        println!();
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
