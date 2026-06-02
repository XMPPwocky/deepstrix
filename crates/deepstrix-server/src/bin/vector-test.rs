//! deepstrix-vector-test — compares our local engine's greedy
//! continuation against captured official DeepSeek V4 Flash API
//! continuations (the `official.vec` fixture in
//! `external/ds4/tests/test-vectors/`).
//!
//! Methodology mirrors `external/ds4/tests/ds4_test.c::test_logprob_vector_case`:
//!   1. Render the prompt with the SAME chat template the official
//!      capture used (system="", no tools, thinking disabled).
//!   2. Prefill.
//!   3. For each official continuation step:
//!        - sample_next(Argmax) → our_token
//!        - compare our_token's decoded bytes against the official
//!          selected bytes (hex equality)
//!        - advance: either feed OFFICIAL token (teacher-forced; tests
//!          each step from the same reference distribution) or feed
//!          OUR token (free-running; one wrong step poisons the rest).
//!   4. Per-case + aggregate match counts.
//!
//! Why: the official continuations were captured at temp=0 with
//! thinking disabled, so the reference distribution is deterministic.
//! If our local engine's argmax diverges from the reference, the gap
//! is implementation-side (precision, numerical safety, attention
//! kernel correctness) — independent of sampling or temperature.
//!
//! Usage:
//!   deepstrix-vector-test --gguf MODEL.gguf --vec external/ds4/tests/test-vectors/official.vec
//!   deepstrix-vector-test --gguf … --vec … --case long_code_audit --advance teacher
//!   deepstrix-vector-test --gguf … --vec … --advance free
//!
//! Notes:
//!   - Prompt file paths in the .vec are resolved relative to its
//!     parent directory (so the shipped file format JustWorks).
//!   - Per-case ctx in the .vec is treated as a hint; we allocate the
//!     max ctx across cases at startup and reuse the same state via
//!     reset_in_place between cases.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use clap::Parser;
use color_eyre::eyre::{self, eyre};

use deepstrix_server::embed::{build_gpt2_byte_decoder, embed_lookup, gpt2_decode_token};
use deepstrix_server::openai::types::{ChatMessage, Role};
use deepstrix_server::prompt::render_prompt;
use deepstrix_server::rope_for_layer;
use deepstrix_server::tokens::{TOK_BOS, TOK_EOS};

use v4flash_core::tokenizer::BpeVocab;
use v4flash_core::MappedGguf;
use v4flash_hip::Device;
use v4flash_kernels::config::{HC_DIM, N_VOCAB};
use v4flash_kernels::het::{
    BatchDgpuScratch, BatchIgpuScratch, DgpuScratch, ExecMode, HetModelState, HetModelWeights,
    HeterogeneousEngine, IgpuScratch, SampleMode,
};
use v4flash_kernels::RopeParams;

#[derive(Parser, Debug)]
#[command(about = "Compare local engine greedy continuation against captured DeepSeek API vectors.")]
struct Args {
    /// Path to the GGUF model file.
    #[arg(long)]
    gguf: PathBuf,

    /// Path to the official.vec fixture (see external/ds4/tests/test-vectors/).
    #[arg(long)]
    vec: PathBuf,

    /// Run only the case with this id (matches the `case <id>` line).
    /// Omit to run every case.
    #[arg(long)]
    case: Option<String>,

    /// KV cache slots to allocate. Must be ≥ the largest per-case ctx
    /// in the .vec file. Defaults to that maximum.
    #[arg(long)]
    ctx: Option<u32>,

    /// `teacher` (default): at each step, advance by feeding the
    /// OFFICIAL token, so every step's argmax is judged from the same
    /// reference history.
    /// `free`: advance by feeding OUR argmax — a single mismatch
    /// diverges the trajectory and all subsequent steps trivially miss.
    #[arg(long, default_value = "teacher")]
    advance: AdvanceMode,

    /// Print the per-step decoded bytes for every step (verbose).
    #[arg(long, short)]
    verbose: bool,

    /// Diagnostic: do the final prompt token's forward pass via
    /// `engine.forward_token` instead of folding it into the batched
    /// prefill. The two paths SHOULD produce the same logits at step 0
    /// — but `forward_token` is where `DEEPSTRIX_SUBSTITUTE_RESIDUAL`'s
    /// per-layer hooks fire, so we need this mode to bisect divergence
    /// by substituting ds4-CPU's per-layer residuals into our pass.
    /// Only step 0 of the chosen case is evaluated in this mode.
    #[arg(long)]
    substitute_eval: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum AdvanceMode {
    Teacher,
    Free,
}

#[derive(Debug)]
struct VectorStep {
    /// Decoded UTF-8/raw bytes of the official greedy-selected token
    /// at this step. The .vec stores these as hex; we keep the raw
    /// bytes for comparison and for token-id lookup against the
    /// vocab's `token_text`.
    selected: Vec<u8>,
}

#[derive(Debug)]
struct VectorCase {
    id: String,
    ctx: u32,
    prompt_path: PathBuf,
    steps: Vec<VectorStep>,
}

fn parse_vec_file(path: &Path) -> eyre::Result<Vec<VectorCase>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| eyre!("read {}: {e}", path.display()))?;
    let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
    // The .vec file uses paths like "tests/test-vectors/prompts/foo.txt"
    // which are relative to the ds4 source root, NOT the .vec's parent.
    // We resolve them by joining against the .vec's grandparent's
    // grandparent (i.e. external/ds4/). Fall back to base_dir for any
    // path that already resolves there.
    let ds4_root = base_dir
        .ancestors()
        .find(|p| p.join("ds4.c").exists())
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| base_dir.to_path_buf());

    let mut cases: Vec<VectorCase> = Vec::new();
    let mut cur: Option<VectorCase> = None;
    let mut cur_step_target: Option<usize> = None;

    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        match it.next() {
            Some("case") => {
                if let Some(c) = cur.take() {
                    cases.push(c);
                }
                let id = it.next().ok_or_else(|| eyre!("case line missing id"))?.to_string();
                let ctx: u32 = it
                    .next()
                    .ok_or_else(|| eyre!("case line missing ctx"))?
                    .parse()
                    .map_err(|e| eyre!("case ctx parse: {e}"))?;
                let nsteps: usize = it
                    .next()
                    .ok_or_else(|| eyre!("case line missing nsteps"))?
                    .parse()
                    .map_err(|e| eyre!("case nsteps parse: {e}"))?;
                let prompt_rel = it
                    .next()
                    .ok_or_else(|| eyre!("case line missing prompt path"))?;
                // Resolve prompt path: try the .vec's grandparent (ds4
                // root) first, fall back to vec's own directory.
                let primary = ds4_root.join(prompt_rel);
                let fallback = base_dir.join(prompt_rel);
                let prompt_path = if primary.exists() {
                    primary
                } else {
                    fallback
                };
                cur = Some(VectorCase {
                    id,
                    ctx,
                    prompt_path,
                    steps: Vec::with_capacity(nsteps),
                });
                cur_step_target = None;
            }
            Some("step") => {
                let c = cur
                    .as_mut()
                    .ok_or_else(|| eyre!("step line outside a case"))?;
                let idx: usize = it
                    .next()
                    .ok_or_else(|| eyre!("step missing index"))?
                    .parse()?;
                let hex = it.next().ok_or_else(|| eyre!("step missing token hex"))?;
                // ntop is third field but we don't need top-K (the
                // shipped .vec always has ntop=1, redundant with the
                // selected token).
                let bytes = hex_decode(hex)?;
                if idx != c.steps.len() {
                    return Err(eyre!("step index {idx} out of order (expected {})", c.steps.len()));
                }
                c.steps.push(VectorStep { selected: bytes });
                cur_step_target = Some(idx);
            }
            Some("top") => {
                // Skipped — we don't compare top-K logprobs.
                let _ = cur_step_target;
            }
            Some("end") => {
                // Finalize current case at the next case/EOF; nothing
                // to do here.
            }
            Some(other) => return Err(eyre!("unexpected .vec directive: {other}")),
            None => continue,
        }
    }
    if let Some(c) = cur.take() {
        cases.push(c);
    }
    Ok(cases)
}

fn hex_decode(s: &str) -> eyre::Result<Vec<u8>> {
    if s.len() % 2 != 0 {
        return Err(eyre!("odd-length hex string: {s:?}"));
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    for chunk in bytes.chunks(2) {
        let hi = hex_nibble(chunk[0])?;
        let lo = hex_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn hex_nibble(b: u8) -> eyre::Result<u8> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(eyre!("non-hex byte 0x{b:02x}")),
    }
}

fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8] = b"0123456789abcdef";
    let mut s = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0x0f) as usize] as char);
    }
    s
}

/// Build a reverse map from decoded-token-bytes to token-id. Used to
/// resolve official `selected` bytes back to a vocab id so we can
/// teacher-force-advance.
///
/// V4-Flash's BPE encodes raw bytes with GPT-2's byte-decoder shuffle
/// (each byte 0..256 mapped to a printable codepoint at vocab-build
/// time). To compare against the official API's raw byte stream we
/// undo that shuffle via `gpt2_decode_token`.
fn build_reverse_byte_map(
    vocab: &BpeVocab,
    byte_decoder: &HashMap<char, u8>,
) -> HashMap<Vec<u8>, i32> {
    let mut map: HashMap<Vec<u8>, i32> = HashMap::with_capacity(vocab.vocab_size());
    for id in 0..vocab.vocab_size() as i32 {
        let Some(raw) = vocab.token_text(id) else {
            continue;
        };
        let decoded = gpt2_decode_token(raw, byte_decoder);
        if !decoded.is_empty() {
            // Multiple ids can decode to the same bytes for special
            // tokens (rare). We keep the first; collisions are not
            // expected for the byte-range used by official.vec.
            map.entry(decoded).or_insert(id);
        }
    }
    map
}

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

fn main() -> eyre::Result<()> {
    color_eyre::install()?;
    let args = Args::parse();

    // Parse vectors first so we can size the KV cache appropriately
    // and short-circuit obvious problems before paying the
    // 50-second-load price.
    let mut cases = parse_vec_file(&args.vec)?;
    if let Some(filter) = &args.case {
        cases.retain(|c| &c.id == filter);
        if cases.is_empty() {
            return Err(eyre!("no case matches --case {filter:?}"));
        }
    }
    let max_case_ctx = cases.iter().map(|c| c.ctx).max().unwrap_or(8192);
    let n_kv_max = args.ctx.unwrap_or(max_case_ctx);
    if n_kv_max < max_case_ctx {
        return Err(eyre!(
            "--ctx {n_kv_max} < largest per-case ctx {max_case_ctx}; raise --ctx"
        ));
    }
    eprintln!(
        "vector-test: {} cases, n_kv_max={n_kv_max}, advance={:?}",
        cases.len(),
        args.advance
    );
    for c in &cases {
        eprintln!("  case {} (ctx={}, steps={}, prompt={})",
            c.id, c.ctx, c.steps.len(), c.prompt_path.display());
        if !c.prompt_path.exists() {
            return Err(eyre!("prompt file not found: {}", c.prompt_path.display()));
        }
    }

    // ---- Engine init (mirrors deepstrix-cli/src/bin/chat.rs:388-426) ----
    eprintln!("loading model: {}", args.gguf.display());
    let gguf = MappedGguf::open(&args.gguf)?;
    let vocab = BpeVocab::from_gguf(gguf.gguf())?;

    let dgpu = pick_dgpu()?;
    let igpu = pick_igpu()?;
    eprintln!(
        "dGPU={} iGPU={}",
        dgpu.properties()?.gcn_arch_name,
        igpu.properties()?.gcn_arch_name
    );

    let token_embd_t = gguf
        .gguf()
        .tensor("token_embd.weight")
        .ok_or_else(|| eyre!("missing token_embd.weight"))?;
    let token_embd_bytes = gguf.read_tensor(token_embd_t)?.to_vec();

    let rope = |layer: i32| -> eyre::Result<RopeParams> { Ok(rope_for_layer(layer)) };

    let t0 = std::time::Instant::now();
    let weights = HetModelWeights::load_all(&gguf, dgpu, igpu, &rope)?;
    eprintln!("weights loaded in {:.1}s", t0.elapsed().as_secs_f64());

    let engine = HeterogeneousEngine::new(
        dgpu,
        &dgpu.properties()?.gcn_arch_name,
        igpu,
        &igpu.properties()?.gcn_arch_name,
        ExecMode::HetParallel,
    )?;
    let mut dgpu_scratch = DgpuScratch::alloc(dgpu)?;
    let mut igpu_scratch = IgpuScratch::alloc(igpu)?;
    let mut state = HetModelState::alloc(dgpu, igpu, n_kv_max)?;
    let mut bd_a = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_a = BatchIgpuScratch::alloc(igpu)?;
    let mut bd_b = BatchDgpuScratch::alloc(dgpu)?;
    let mut bi_b = BatchIgpuScratch::alloc(igpu)?;
    eprintln!("KV cache: {n_kv_max} slots");

    let byte_decoder = build_gpt2_byte_decoder();
    let mut residual = vec![0f32; HC_DIM as usize];

    eprintln!("building reverse byte→id map (one-time)…");
    let t0 = std::time::Instant::now();
    let reverse_map = build_reverse_byte_map(&vocab, &byte_decoder);
    eprintln!(
        "reverse map: {} entries in {:.2}s",
        reverse_map.len(),
        t0.elapsed().as_secs_f64()
    );

    // ---- Per-case loop ----
    let mut total_steps: usize = 0;
    let mut total_matches: usize = 0;
    let mut per_case_report: Vec<(String, usize, usize)> = Vec::new();

    for case in &cases {
        eprintln!("\n=== case {} ===", case.id);
        // Fresh state per case (the official continuations were each
        // captured from a stateless prompt — no cross-case priming).
        state.reset_in_place(dgpu, igpu)?;

        let prompt_text = std::fs::read_to_string(&case.prompt_path)
            .map_err(|e| eyre!("read prompt {}: {e}", case.prompt_path.display()))?;

        // Build the chat-rendered prompt. Match the official capture
        // settings: empty system, no tools, thinking disabled.
        let messages = vec![ChatMessage {
            role: Role::User,
            content: Some(prompt_text),
            tool_calls: Vec::new(),
            tool_call_id: None,
            name: None,
        }];
        let tokens = render_prompt(&vocab, &messages, None, /*think_mode=*/ false)?;
        eprintln!("rendered prompt: {} tokens", tokens.len());

        if (tokens.len() as u32) + case.steps.len() as u32 + 1 > n_kv_max {
            eprintln!("  SKIP: prompt+steps {} > n_kv_max {}", tokens.len() + case.steps.len() + 1, n_kv_max);
            continue;
        }

        // Prefill — match chat.rs's pipelined path. In substitute_eval
        // mode we hold back the LAST token from prefill so we can run
        // its forward pass via engine.forward_token (where the
        // per-layer substitution hook fires).
        let prefill_len = if args.substitute_eval {
            tokens.len() - 1
        } else {
            tokens.len()
        };
        let prefill_tokens = &tokens[..prefill_len];
        let mut input_hcs: Vec<Vec<f32>> = Vec::with_capacity(prefill_tokens.len());
        for &tok in prefill_tokens {
            let mut v = vec![0f32; HC_DIM as usize];
            embed_lookup(&token_embd_bytes, tok, &mut v);
            input_hcs.push(v);
        }
        let t0 = std::time::Instant::now();
        let last_logits = engine.forward_prefill_pipelined(
            &mut bd_a, &mut bi_a, &mut bd_b, &mut bi_b,
            &mut dgpu_scratch,
            &mut state, &weights,
            &input_hcs, prefill_tokens, /*pos0=*/ 0,
            /*last_only=*/ true,
            /*stats=*/ None,
        )?;
        let mut pos = prefill_tokens.len() as u32;
        // In substitute_eval mode, now run forward_token for the
        // held-back final prompt token. This is where the per-layer
        // residual substitution hook fires (engine.rs).
        if args.substitute_eval {
            let last_tok = tokens[tokens.len() - 1];
            let mut residual_in = vec![0f32; HC_DIM as usize];
            embed_lookup(&token_embd_bytes, last_tok, &mut residual_in);
            eprintln!(
                "substitute_eval: forward_token(id={last_tok}) at pos={pos}…"
            );
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                &residual_in, pos, last_tok,
            )?;
            pos += 1;
        }
        let prefill_secs = t0.elapsed().as_secs_f64();
        eprintln!(
            "prefill: {} tok in {:.2}s ({:.1} tok/s)",
            tokens.len(), prefill_secs, tokens.len() as f64 / prefill_secs.max(1e-6)
        );
        if last_logits.len() != N_VOCAB as usize {
            return Err(eyre!(
                "prefill logits len {} != N_VOCAB {}",
                last_logits.len(), N_VOCAB
            ));
        }

        // Per-step argmax compare. In substitute_eval mode we only
        // run step 0 (the one where divergence was observed) — the
        // substituted residual makes steps 1+ meaningless since the
        // injected state isn't from a coherent trajectory.
        let mut case_matches: usize = 0;
        let n_steps = if args.substitute_eval { 1.min(case.steps.len()) } else { case.steps.len() };
        for (i, step) in case.steps.iter().take(n_steps).enumerate() {
            // u01=0 is unused for Argmax mode.
            let our_tok = engine.sample_next(&mut dgpu_scratch, SampleMode::Argmax, 0.0)?;

            let our_bytes: Vec<u8> = vocab
                .token_text(our_tok)
                .map(|b| gpt2_decode_token(b, &byte_decoder))
                .unwrap_or_default();

            let matched = our_bytes == step.selected;
            if matched {
                case_matches += 1;
            }
            if args.verbose || !matched {
                eprintln!(
                    "  step {i:>2}  our_id={our_tok:>6} our={:?} ({})  official={:?} ({})  {}",
                    String::from_utf8_lossy(&our_bytes),
                    hex_encode(&our_bytes),
                    String::from_utf8_lossy(&step.selected),
                    hex_encode(&step.selected),
                    if matched { "MATCH" } else { "MISS" }
                );
                // Dump our top-K logits + logprobs at this step. The
                // logits buffer at dgpu_scratch.logits is what the
                // sampler just argmaxed; copy to host, partial-sort,
                // compute log-softmax via the standard max-subtract
                // trick, print top-K. Lets us compare distribution
                // shape against ds4-CPU's --dump-logprobs output to
                // distinguish "near-tie ordering flip" from "totally
                // different distribution".
                const TOPK: usize = 20;
                let mut logits_host = vec![0.0f32; N_VOCAB as usize];
                dgpu_scratch.logits.copy_to_host(&mut logits_host)?;
                let mut idx_logit: Vec<(usize, f32)> = logits_host
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| (i, v))
                    .collect();
                idx_logit.select_nth_unstable_by(TOPK, |a, b| b.1.partial_cmp(&a.1).unwrap());
                let mut top = idx_logit[..TOPK].to_vec();
                top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                let max_l = top[0].1;
                // log-Z computed over FULL vocab so logprobs sum to ~1.0
                // (would be misleading from just top-K).
                let z: f32 = logits_host
                    .iter()
                    .map(|&l| (l - max_l).exp())
                    .sum();
                let log_z = max_l + z.ln();
                eprintln!("    our top-{TOPK} (logit, logprob, token-bytes-hex, text):");
                for (id, logit) in &top {
                    let logprob = logit - log_z;
                    let bytes: Vec<u8> = vocab
                        .token_text(*id as i32)
                        .map(|b| gpt2_decode_token(b, &byte_decoder))
                        .unwrap_or_default();
                    eprintln!(
                        "      id={id:>6} logit={logit:>10.4} logprob={logprob:>10.4} bytes={} text={:?}",
                        hex_encode(&bytes),
                        String::from_utf8_lossy(&bytes)
                    );
                }
            }

            // Skip the advance after the last step we care about. In
            // substitute_eval mode n_steps=1, so the advance after
            // step 0 would call forward_token a SECOND time with the
            // teacher-forced "next" token — that second call also
            // fires the dump hook and overwrites the layer-NN files
            // with the next token's intermediate residuals, not the
            // failing forward pass's. Skipping the trailing advance
            // keeps the dumps faithful to the diverging step's
            // computation.
            if i + 1 >= n_steps {
                break;
            }
            // Advance.
            let advance_tok = match args.advance {
                AdvanceMode::Teacher => {
                    // Look up the official token's id from the
                    // reverse map. If not found (rare; shouldn't
                    // happen for tokens that round-trip the API), we
                    // fall back to OUR token to keep going.
                    match reverse_map.get(&step.selected) {
                        Some(&id) => id,
                        None => {
                            eprintln!(
                                "    teacher-force lookup miss for {:?}; advancing with our token",
                                hex_encode(&step.selected)
                            );
                            our_tok
                        }
                    }
                }
                AdvanceMode::Free => our_tok,
            };

            embed_lookup(&token_embd_bytes, advance_tok, &mut residual);
            engine.forward_token(
                &mut dgpu_scratch, &mut igpu_scratch, &mut state, &weights,
                &residual, pos, advance_tok,
            )?;
            pos += 1;
            if pos >= n_kv_max {
                eprintln!("  ctx full at pos={pos}, stopping case early");
                break;
            }
        }

        // Push EOS to keep KV state sane (mirrors chat.rs end-of-turn).
        let _ = TOK_EOS;
        let _ = TOK_BOS;

        eprintln!(
            "  {}: {}/{} steps matched ({:.0}%)",
            case.id,
            case_matches,
            case.steps.len(),
            100.0 * case_matches as f64 / case.steps.len().max(1) as f64
        );
        per_case_report.push((case.id.clone(), case_matches, case.steps.len()));
        total_matches += case_matches;
        total_steps += case.steps.len();
    }

    // ---- Summary ----
    println!();
    println!("== summary ==");
    for (id, m, n) in &per_case_report {
        println!("  {id:<32} {m}/{n}  ({:.0}%)", 100.0 * *m as f64 / (*n).max(1) as f64);
    }
    println!(
        "  {:<32} {}/{}  ({:.1}%)",
        "TOTAL",
        total_matches,
        total_steps,
        100.0 * total_matches as f64 / total_steps.max(1) as f64
    );

    Ok(())
}
