//! Raw-completion CLI for V4-Flash. Takes a prompt, prefills, greedy-samples
//! N tokens, prints them. No chat template (yet); just `tokenize → forward →
//! argmax → forward → ...` until N tokens or EOS.
//!
//! Usage:
//!   HIP_VISIBLE_DEVICES=1 deepstrix-complete <gguf> <prompt> [n_predict]
//!
//! Example:
//!   HIP_VISIBLE_DEVICES=1 deepstrix-complete \
//!     /persist/lumi/models/DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf \
//!     "DeepSeek-V4 Flash is" 30

use std::io::Write;

use color_eyre::eyre::{self, eyre};
use v4flash_core::{gguf::GgufType, tokenizer::BpeVocab, MappedGguf};
use v4flash_hip::{install_panic_handler, Device};
use v4flash_kernels::forward::{
    Engine, ModelState, ModelWeights, Scratch, COMPRESS_RATIOS, HC_DIM, N_EMBD, N_HC, N_VOCAB,
};
use v4flash_kernels::RopeParams;

// V4-Flash chat-template special token IDs (from GGUF tokenizer.ggml.tokens).
const TOK_BOS: i32 = 0;            // <｜begin▁of▁sentence｜>
const TOK_EOS: i32 = 1;            // <｜end▁of▁sentence｜>
const TOK_USER: i32 = 128803;      // <｜User｜>
const TOK_ASSISTANT: i32 = 128804; // <｜Assistant｜>
const TOK_THINK_END: i32 = 128822; // </think>

// V4-Flash RoPE constants (from GGUF metadata + ds4.c).
const ROPE_FREQ_BASE_DENSE: f32 = 10000.0;
const ROPE_FREQ_BASE_COMP: f32 = 160000.0;
const ROPE_SCALE_FACTOR: f32 = 16.0;
const ROPE_ORIG_CTX: u64 = 65536;
const ROPE_BETA_FAST: f32 = 32.0;
const ROPE_BETA_SLOW: f32 = 1.0;

fn pick_device() -> eyre::Result<Device> {
    let devices = Device::all()?;
    for d in &devices {
        if d.properties()?.gcn_arch_name.starts_with("gfx1151") {
            return Ok(*d);
        }
    }
    devices.first().copied().ok_or_else(|| eyre!("no HIP devices"))
}

/// Per-layer RoPE config for V4-Flash. Mirrors ds4's
/// `layer_rope_freq_base` / `layer_rope_freq_scale` (ds4.c:4803).
/// Dense layers (ratio==0): freq_base=10000, no YaRN.
/// Compressed layers (ratio>0): freq_base=160000, YaRN with scale=1/16,
/// n_ctx_orig=65536. attn_factor cancels the YaRN magnitude scaling that the
/// kernel applies internally (DeepSeek V4 RoPE = interpolation only).
fn rope_for_layer(layer: i32) -> RopeParams {
    let ratio = COMPRESS_RATIOS[layer as usize];
    let compressed = ratio != 0;
    let freq_base = if compressed {
        ROPE_FREQ_BASE_COMP
    } else {
        ROPE_FREQ_BASE_DENSE
    };
    let freq_scale = if compressed {
        1.0 / ROPE_SCALE_FACTOR
    } else {
        1.0
    };
    let ext_factor = if compressed && ROPE_SCALE_FACTOR > 1.0 { 1.0 } else { 0.0 };
    let mut attn_factor = 1.0f32;
    if ext_factor != 0.0 && freq_scale > 0.0 {
        attn_factor /= 1.0 + 0.1 * (1.0 / freq_scale).ln();
    }
    let n_ctx_orig = if compressed { ROPE_ORIG_CTX } else { 0 };
    let floats = [
        freq_base,
        freq_scale,
        ext_factor,
        attn_factor,
        ROPE_BETA_FAST,
        ROPE_BETA_SLOW,
    ];
    RopeParams::from_dump_blob(&floats, n_ctx_orig).expect("valid rope params")
}

/// Read token_embd row for `token_id` (F16, 4096 elements) and replicate
/// `n_hc=4` times into `out` (16384 f32 elements). Mirrors ds4's
/// `hc_from_plain_embedding` (ds4.c:4393).
fn embed_lookup(token_embd_bytes: &[u8], token_id: i32, out: &mut [f32]) {
    let n_embd = N_EMBD as usize;
    let n_hc = N_HC as usize;
    assert_eq!(out.len(), n_embd * n_hc);
    let row_off = (token_id as usize) * n_embd * 2; // F16 = 2 bytes
    // Decode the F16 row into the first n_embd slot.
    for i in 0..n_embd {
        let b0 = token_embd_bytes[row_off + i * 2];
        let b1 = token_embd_bytes[row_off + i * 2 + 1];
        let bits = u16::from_le_bytes([b0, b1]);
        out[i] = f16_to_f32(bits);
    }
    // Replicate into hc=1..n_hc.
    for h in 1..n_hc {
        let (head, tail) = out.split_at_mut(h * n_embd);
        let src = &head[0..n_embd];
        let dst = &mut tail[0..n_embd];
        dst.copy_from_slice(src);
    }
}

/// IEEE 754 binary16 → binary32. Handles subnormals + inf/nan.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits >> 15) & 0x1;
    let exp = (bits >> 10) & 0x1f;
    let mant = bits & 0x3ff;
    let s: u32 = (sign as u32) << 31;
    let f32_bits: u32 = match exp {
        0 if mant == 0 => s,
        0 => {
            // Subnormal: 2^-14 * (mant / 1024)
            let mantissa = mant as f32 / 1024.0;
            let v = mantissa * (1.0 / (1u64 << 14) as f32);
            let val = if sign == 1 { -v } else { v };
            return val;
        }
        0x1f => s | 0x7f800000 | ((mant as u32) << 13),
        _ => s | ((exp as u32 + 112) << 23) | ((mant as u32) << 13),
    };
    f32::from_bits(f32_bits)
}

/// GPT-2 BPE byte-decoder table. Inverse of the byte→unicode mapping that
/// GPT-2-style tokenizers apply so that arbitrary bytes can be stored as
/// printable unicode in the vocab. Returns 256-byte LUT indexed by char.
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

/// Decode `token_bytes` (UTF-8 with GPT-2 byte-encoded chars) into raw bytes.
fn gpt2_decode_token(token_bytes: &[u8], dec: &std::collections::HashMap<char, u8>) -> Vec<u8> {
    let s = std::str::from_utf8(token_bytes).unwrap_or("");
    let mut out = Vec::with_capacity(token_bytes.len());
    for ch in s.chars() {
        if let Some(&b) = dec.get(&ch) {
            out.push(b);
        } else {
            // Unknown char (e.g., true unicode that wasn't byte-encoded);
            // emit its utf-8 representation directly.
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
    let raw_args: Vec<String> = std::env::args().skip(1).collect();
    let chat_mode = !raw_args.iter().any(|a| a == "--raw");
    let mut args = raw_args.into_iter().filter(|a| a != "--raw");
    let gguf_path = args
        .next()
        .ok_or_else(|| eyre!("usage: deepstrix-complete [--raw] <gguf> <prompt> [n_predict]"))?;
    let prompt = args
        .next()
        .ok_or_else(|| eyre!("missing prompt"))?;
    let n_predict: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    eprintln!("loading model from {gguf_path}…");
    let gguf = MappedGguf::open(&gguf_path)?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;

    let device = pick_device()?;
    device.set_current()?;
    let arch = device.properties()?.gcn_arch_name;
    eprintln!("using device {} ({arch})", device.id);

    // Tokenize the prompt. In chat mode wrap with V4-Flash template:
    //   <BOS><｜User｜>{user}<｜Assistant｜></think>
    // In --raw mode just encode the literal prompt.
    let prompt_tokens: Vec<i32> = if chat_mode {
        let mut v = vec![TOK_BOS, TOK_USER];
        v.extend(vocab.encode(&prompt));
        v.push(TOK_ASSISTANT);
        v.push(TOK_THINK_END);
        v
    } else {
        vocab.encode(&prompt)
    };
    eprintln!(
        "prompt tokens ({}, mode={}): {:?}",
        prompt_tokens.len(),
        if chat_mode { "chat" } else { "raw" },
        prompt_tokens
    );
    if prompt_tokens.is_empty() {
        return Err(eyre!("empty prompt after tokenization"));
    }

    // Read the token_embd mmap.
    let token_embd_t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("missing token_embd.weight"))?;
    if token_embd_t.dtype != GgufType::F16 {
        return Err(eyre!("token_embd dtype {:?} != F16", token_embd_t.dtype));
    }
    let token_embd_bytes = gguf
        .tensor_bytes(token_embd_t)
        .ok_or_else(|| eyre!("token_embd bytes missing"))?;
    eprintln!("token_embd: {} bytes", token_embd_bytes.len());

    let rope = |layer: i32| -> eyre::Result<RopeParams> { Ok(rope_for_layer(layer)) };

    eprintln!("loading weights to iGPU (~10 GiB)…");
    let t0 = std::time::Instant::now();
    let weights = ModelWeights::load_all(&gguf, device.id, &rope)?;
    eprintln!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let engine = Engine::for_arch(device, &arch)?;
    let n_kv_max = (prompt_tokens.len() + n_predict) as u32 + 8;
    let mut scratch = Scratch::alloc(device.id)?;
    let mut state = ModelState::alloc(device.id, n_kv_max)?;

    let byte_decoder = build_gpt2_byte_decoder();

    let mut residual = vec![0f32; HC_DIM as usize];
    let mut logits_host = vec![0f32; N_VOCAB as usize];

    // Prefill: forward each prompt token; only last token's logits feed sampling.
    eprintln!("prefill ({} tokens)…", prompt_tokens.len());
    if !chat_mode {
        print!("{prompt}");
        std::io::stdout().flush().ok();
    }
    let t0 = std::time::Instant::now();
    for (pos, &tok) in prompt_tokens.iter().enumerate() {
        embed_lookup(token_embd_bytes, tok, &mut residual);
        engine.forward_token(
            &mut scratch,
            &mut state,
            &weights,
            &gguf,
            &residual,
            pos as u32,
            tok,
        )?;
    }
    let prefill_secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "prefill done in {:.2}s ({:.1} tok/s)",
        prefill_secs,
        prompt_tokens.len() as f64 / prefill_secs
    );

    // Sample first generated token from prefill's last logits.
    scratch.logits.copy_to_host(&mut logits_host)?;
    let mut next = argmax(&logits_host) as i32;

    // Decode loop.
    let t0 = std::time::Instant::now();
    let mut generated = Vec::with_capacity(n_predict);
    for i in 0..n_predict {
        generated.push(next);
        if let Some(bytes) = vocab.token_text(next) {
            let raw = gpt2_decode_token(bytes, &byte_decoder);
            std::io::stdout().write_all(&raw).ok();
            std::io::stdout().flush().ok();
        } else {
            print!("<?{next}>");
        }
        if next == TOK_EOS {
            eprintln!("\n<EOS>");
            break;
        }

        let pos = (prompt_tokens.len() + i) as u32;
        embed_lookup(token_embd_bytes, next, &mut residual);
        engine.forward_token(
            &mut scratch,
            &mut state,
            &weights,
            &gguf,
            &residual,
            pos,
            next,
        )?;
        scratch.logits.copy_to_host(&mut logits_host)?;
        next = argmax(&logits_host) as i32;
    }
    let decode_secs = t0.elapsed().as_secs_f64();
    println!();
    eprintln!(
        "decode: {} tokens in {:.2}s ({:.1} tok/s)",
        generated.len(),
        decode_secs,
        generated.len() as f64 / decode_secs.max(1e-6)
    );
    eprintln!("generated_tokens = {:?}", generated);
    Ok(())
}
