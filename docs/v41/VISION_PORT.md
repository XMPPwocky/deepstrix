# V4.1 multimodal (vision) port — scope + status (2026-09-13)

The goal requires multimodal. V4.1 ships a standard ViT + MLP aligner; the checkpoint has 266
`vision.*` / `aligner.*` tensors. There is already a `v4flash-vision` crate (Tower: ViT + aligner
+ 2-D RoPE kernels) built for V4-Flash Vision-Exp, so this is an ADAPTER, not a rewrite.

## V4.1 ViT config (inference/config.json)
- vision_dim 1024, vision_inter_dim 2816, vision_n_heads 16, vision_n_layers 32, patch 14,
  rope_theta 10000, downsample_ratio 3, max_n_token 1024, min_pixels 295936 (544²),
  max_wh_ratio null, image_token_id 129264, text `dim` 5120.
- Tensors (all BF16, shards 1–2): `vision.patch_embed.proj.{weight,bias}` (`nn.Linear(588, 1024)`
  over the `(c,y,x)`-flattened patch — same math as V4-Flash's conv), `vision.blocks.N.{norm1,
  attn.wqkv(+bias) [3072,1024], attn.wo(+bias), norm2, mlp.w1 [5632,1024] (fused gate‖up, no
  bias), mlp.w2 [1024,2816]}`, `vision.norm`, `aligner.w1 [5120,9216] (+bias)`, `aligner.w2
  [5120,5120] (+bias)`, `image_start` / `image_newline` / `image_end` [5120]. 925.6 MiB.
- model.py: `encode_image = aligner(vision(patches, h, w))`; `merge_image_embeddings` splices
  aligner rows into the IMAGE slots in reading order and the three learned embeddings into the
  delimiters; routing uses `bias_vl` for image rows; Engram masks image tokens out.

## Status: BUILT + VALIDATED (tower), WIRED (server), NOT YET SERVED (needs a restart)

### 1. Loader — `crates/v4flash-vision/src/hf_v41.rs`
`Tower::load_v41(dir)` / `Tower::load_v41_from(&SafetensorsDir, &config)` read the 266 tensors
through `v4flash_core::SafetensorsDir` (the same pread reader `V41HfWeights` uses) into the
existing `MmprojHost`, casting at read time: weights bf16 → f16 bits (lossless for every bf16
value inside f16's normal range — 8 vs 11 significant bits — the loader counts underflows and
refuses overflows; 0 overflow in this checkpoint), biases / norms / sentinels bf16 → f32. No
model files are written. Fused `wqkv` is split into the q/k/v rows the tower re-fuses; fused
`mlp.w1` maps straight to gate‖up (`chunk(2)` = gate first), which settles the orientation
question `reference.rs` had left open. `check_config` cross-checks every architecture number
and the image-processor limits against `inference/config.json`.

Generalisation of the crate (V4-Flash path unchanged, all old vectors still pass):
`VisionCfg { kind, min_pixels, max_wh_ratio, max_n_token, text_dim }` with `V4_FLASH` / `V41`
profiles; `text_dim` is a runtime property of `MmprojHost` / `Tower` (workspace, aligner GEMMs,
`place_rows`); `ImageLayout.compress_pad` is a field (0 for V4.1); `img_pad` sentinel is
optional. The crate no longer depends on `v4flash-kernels` (it never used it).

### 2. Aligner
Unchanged kernels: 3×3 `vit_unfold` (channel-major, zero pad = `F.pad` + `F.unfold(r, stride r)`)
→ `mm.1` 9216→5120 (+b) → GELU(erf) → `mm.2` 5120→5120 (+b). The only V4.1 difference is the
width, threaded through as `text_dim`.

### 3. Validation — `crates/v4flash-vision/tests/canonical_v41.rs`
Reference: `scripts/gen_v41_vision_vectors.py` loads the bf16 tensors into DeepSeek's UNMODIFIED
`inference/vision.py::{ViT, Aligner}` (f32, CPU, 8 threads, 3.2 GB RSS, 32 s for all cases) and
dumps patches (bf16-rounded as `image_processor.load_image` makes them), hidden, aligner rows and
the merged span; plus the 54-size `plan_image_grid` / `image_token_types` table
(`tests/data/layout_cases_v41.json`, bit-exact in `tests/layout_v41_vectors.rs`).
Dumps: `~/.cache/deepstrix/v41/vision_canon/`.

GPU run 2026-09-13 on the **dGPU (gfx1201, scalar GEMM path — the WMMA path is gfx115x-only)**,
the canonical patches replayed verbatim through `Tower::encode_rows`, one tower load (926 MiB,
3.0 s), 222 MiB workspace:

| case | patches → rows (span) | rms_err/rms | 1-cos | argmax aligner | argmax span | encode |
|---|---|---|---|---|---|---|
| synth 4×6 (LCG) | 24 → 4 (8) | 5.8e-4 | 1.7e-7 | 4/4 | 8/8 | 14 ms |
| png 640×480 | 1610 → 192 (206) | 2.0e-3 | 2.1e-6 | 192/192 | 206/206 | 186 ms |
| corn.jpeg 450×308 (upscaled ×1.46) | 1551 → 176 (189) | 2.2e-3 | 2.3e-6 | 176/176 | 189/189 | 169 ms |
| carrots.jpeg 1024×701 | 3774 → 425 (444) | 2.5e-3 | 3.2e-6 | 425/425 | 444/444 | 697 ms |

All sentinel rows bit-exact; encode deterministic. This is the f16 floor (V4-Flash validation:
1-cos 7.0e-6, argmax 209/209). Preprocessing (`preprocess_v41_matches_python`): PNG bit-exact
after bf16 rounding (0/946680 differ); JPEG differs by ≤ 4/255 on ~8–10 % of samples because
the `image` crate's JPEG decoder (zune-jpeg) is not Pillow's libjpeg (IDCT / chroma upsampling).

End to end from OUR decoder + preprocessing (what the server feeds), same test, vs the canonical
aligner rows — two input roundings, because the reference casts its patches to bf16 and the
server (like V4-Flash) keeps f32 → f16:

| case | f32 input (server path) | bf16 input (as the reference) |
|---|---|---|
| png 640×480 | rms 2.1e-2, 1-cos 2.2e-4, argmax 191/192 | **rms 2.04e-3, argmax 192/192 = the replay** |
| corn.jpeg | rms 3.0e-2, 1-cos 4.4e-4, argmax 170/176 | rms 2.8e-2, 1-cos 3.9e-4, argmax 171/176 |
| carrots.jpeg | rms 4.3e-2, 1-cos 9.0e-4, argmax 410/425 | rms 4.2e-2, 1-cos 8.9e-4, argmax 408/425 |

Reading: the PNG column shows our chain is exact and that the reference's own bf16 input
rounding is ~10× the tower's f16 error (our extra precision is the deliberate V4-Flash choice).
The JPEG rows are the decoder mismatch: ~3–4e-2 relative on the aligner rows, 2–4 % of rows
flipping argmax — a mild re-encode-sized perturbation, not a decode bug.
**WONTFIX (user, 2026-09-13):** "as long as we're decoding jpegs correctly, if it's just numerics
idgaf." PNG bit-exactness already isolates our math, and the end-to-end carrots caption below is
from the JPEG path. Do not add a libjpeg-exact decoder dependency for this.

### 4. Server wiring (`deepstrix-server`, `--features v41`)
- `--mmproj <HF snapshot dir>` enables vision in the V4.1 build (`vision_v41::load_tower`);
  the tower stays on the iGPU as before. `bias_vl`: `vision_v41::ensure_bias_vl` writes the
  engine's `bias_vl.bin` sidecar (`~/.cache/deepstrix/models/<snapshot-hash>/bias_vl.bin`,
  40 × 384 f32) from the checkpoint's own `layers.N.ffn.gate.bias_vl` (presented as
  `blk.N.exp_probs_b_vl.bias`) if it is missing, then attaches it — `het/weights.rs` untouched.
- Prompt: `prompt_v41.rs` renders image parts as `<｜deepseek_image｜>` (id 129264 in
  tokenizer.json), joined by "\n\n" (the `encoding/README.md` example is a unit test); tool-result
  images stay inline in the `<tool_result>` body. `render_prompt_v41` takes the placeholder id.
- Span: `vision_prompt::vision_cfg()` picks `VisionCfg::V41` under the feature; the placeholder
  expands to the flat `START (IMAGE×w NEWLINE)×h END` span of synthetic ids (`VOCAB_SIZE + type`,
  same convention as V4-Flash — the engine's `id >= N_VOCAB` image predicate IS the reference's
  `token_types >= 0`). The reference puts `image_token_id` in every span slot of `input_ids`;
  those ids are only ever consumed by the embedding (overwritten by the merge), the image mask
  and Engram, all of which the synthetic ids reproduce.
- Merge: `prefill_suffix` already broadcasts the tower's rows into the 4 HC copies at image
  slots; `N_EMBD` = 5120 under the feature, and `load_tower` refuses a tower of another width.
- Attention: V4.1 has NO `get_image_visible` — image tokens attend causally. The engine is given
  NO `image_spans` under the feature (no raw-window widening, no chunk/lane constraints);
  `check_vision_ctx_fits` is skipped. Routing still selects `bias_vl` per image row by id.
- Engram masking (`v4flash_core::engram_hash`): ids outside the vocab compress to `DEAD`;
  `hash_ids` reproduces `NgramHashState.forward`'s cumulative `blocked` (look-back stops at a
  dead slot → pad); `EngramCtx` stages all-zero rows for dead positions, which is exactly the
  reference's zeroed gate because `wkv` has no bias (value = 0). Unit test with a hand-run table.

### 5. END TO END — WORKS (2026-09-13 09:31, server restarted with the projector)

`/tmp/run_v41_server.sh` now passes `--mmproj "$MODEL"` (env `MMPROJ=` disables it). Startup:
bias_vl sidecar auto-written from the checkpoint (40 layers), tower read from the HF safetensors
(266 tensors, 925.6 MiB, 497 f16 underflows of ~485M values, 0 overflow) and uploaded to the
iGPU in 2.5 s (gfx1151 → **WMMA** GEMM path), `text_dim=5120`.

Text first, so a failure would be attributable: "The capital of France is" → **"The capital of
France is \*\*Paris\*\*."**; "What is 17\*23?" → **"391"**. Both `finish_reason: stop`.
(With the default effort the whole budget goes to the think block and `content` is empty —
these were run with `reasoning_effort: "none"`.)

Image: `inference/examples/images/carrots.jpeg` (1024×701) as a `data:image/jpeg;base64` URL
with "Describe this image in one or two sentences." →

> This image shows a small pile of four fresh, bright orange carrots with green stems arranged
> on a plain white background. One long carrot rests diagonally across the top of the other three.

Correct, including the spatial relation (the photo is exactly that). **Control**: the same text
with no image part → "A black dog sits behind a white picket fence…" — pure confabulation, so
the image request is genuinely reading pixels.

Geometry matched the layout code exactly: `encoded image vit="51x74" llm="17x25"
block_tokens=444` — 3774 patches → 425 aligner rows → 17·(25+1)+2 = **444 span tokens**, the
number `layout_for_grid_cfg(51, 74, .., V41)` predicts. Prompt 457 tokens (444 span + 13 text).

Timing (cold, expert tier paging from SSD): image encode **613 ms** on the iGPU; prefill of the
456-token prompt **58.6 s total**, of which the 128-token CED bounded replay is 30.9 s; decode
38 tokens in the remaining ~32 s. For scale, the 12-token text-only prefill on the same server
took 57.3 s (30.6 s of replay) — at these lengths prefill is dominated by per-request fixed cost
(cold expert paging + the replay), not by the image span, so the 444-token span is nearly free
relative to the floor. This says nothing about the 24.6 tok/s figure measured on a 1874-token
prompt; it is the same fixed cost amortised over fewer tokens.

**CED interaction — checked, no regression.** The replay segment landed at `seg_pos0=328`, i.e.
*inside* the image span (tokens 12–455), and `forward_prefill.rs` passes `image_spans: None` to
it. For V4.1 that is CORRECT, not a gap: `image_spans` feeds only (a) the raw-window widening
and (b) chunk/lane cut planning, both of which exist for V4-Flash's bidirectional image window —
which V4.1 does not have (no `get_image_visible` in `model.py`; image tokens attend causally).
The one image-specific thing the replay must not lose is routing, and it does not: the replay
passes the real `seg_tokens` (synthetic ids ≥ `N_VOCAB`) down to `forward_prompt_batch_v2_
pipelined_range`, which slices them per lane, so `image_runs` still selects `bias_vl` for image
rows on the replayed decoder layers. The lane split can cut an image run in two, which is also
fine causally. The only artifact is a now-stale comment at that call site — "V4.1 vision is not
wired yet" — which should read "V4.1 image tokens are causal; spans are deliberately not passed"
(that file belongs to another agent, so it is reported rather than edited).

To reproduce / restart:

```
CARGO_TARGET_DIR=target-v41 nix develop --profile ~/.cache/deepstrix/devshell --command \
  cargo build --release -p deepstrix-server --features v41
pkill -x deepstrix-serve && nohup /tmp/run_v41_server.sh > logs/v41-vision.log 2>&1 &
# (the script now adds --mmproj "$MODEL"; MMPROJ= disables it)
# then
curl -s localhost:18141/v1/chat/completions -H 'content-type: application/json' -d '{
  "model": "deepseek-v4.1-flash", "max_tokens": 200,
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 ~/.cache/deepstrix/models/dsv4.1f/inference/examples/images/carrots.jpeg)"'"}}]}]}'
```
Send `reasoning_effort: "none"` unless you want the answer inside the think block. Expected
answer: carrots.

Open items: (a) which prompt renderer the handler uses under `v41` is unchanged by this work
(`prompt.rs` still, which also renders the placeholder); (b) `het/weights.rs` could read
`exp_probs_b_vl.bias` from the source directly and drop the sidecar; (c) the stale
`image_spans: None` comment in the CED replay (see §5); (d) the tower's WMMA GEMM is RDNA3-only —
the dGPU runs the scalar path (505–700 ms @ 3774 patches), the iGPU the WMMA path (613 ms), which
is where production keeps it.

Effort was moderate as planned; the V4.1 tower differs from V4-Flash's only in tensor
names/fusion, `text_dim`, the image-processor limits and the span layout — plus the text-side
rules above (causal attention over images, Engram masking), which are engine/server matters.
