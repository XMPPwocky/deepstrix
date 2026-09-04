# Vision-Exp (`vision-exp` branch) — status

DeepSeek-V4-Flash-**Vision-Exp** image support: the `mmproj-F16.gguf` ViT +
aligner tower on the iGPU, the text-side image geometry in the engine, and
the OpenAI-compatible image parts in the server.

---

## Verification pass — 2026-09-04

Environment: NixOS, ROCm 7.2.3, dGPU gfx1201 (untouched — production server
owns it), **iGPU gfx1151 / "AMD Radeon 8060S Graphics" = HIP device 1** for
every GPU test here. mmproj =
`/persist/lumi/models/dsv4f-exp-q2-k-xl/mmproj-F16.gguf` (427 tensors,
`clip` / `deepseek4v`, 466,376,704 params).

### Build

`cargo build --release --workspace --all-targets` and
`cargo test --release --workspace --no-run` both **exit 0**, zero errors.
Warnings are all pre-existing `v4flash-kernels` `unused_*` ones; the vision
crates add none. All six `v4flash-vision` test binaries link.

### Tests

| suite | result |
|---|---|
| `v4flash-vision` CPU (lib + layout_vectors + pillow_vectors + preprocess + unit) | **23 passed**, 0 failed (13 GPU tests `#[ignore]`d) |
| `deepstrix-server` | **53 passed**, 0 failed, 7 ignored |
| `v4flash-kernels --test image_visibility` (text-side window arithmetic) | **7 passed** |
| `v4flash-kernels --test bias_vl_sidecar` | **3 passed** (incl. the real 44,032 B file for both Vision-Exp models) |
| `v4flash-vision --test kernel_oracles` (iGPU) | **3 passed** — gemm / attention / rmsnorm+rope vs CPU |
| `v4flash-vision --test tower_load` (iGPU) | **2 passed** (after the fix below) |
| `v4flash-vision --test tower_encode` (iGPU) | **8 passed** — GPU-vs-CPU-twin `rms_err/rms` 3.1e-3 … 4.4e-3 |
| `v4flash-kernels --test attention_swa_visible_window` (iGPU) | **1 passed** — non-trailing / wide windows, `max_abs` ≤ 2.4e-7 |
| `v4flash-vision --test canonical_vs_python` (iGPU, **new**) | **3 passed** — see below |

**One real defect found and fixed** (uncommitted, in the worktree):
`tests/tower_load.rs` asserted `device_bytes() == 932_339_712 + 827_392`
and failed with `left: 933208064, right: 933167104`. The constant forgot
that the device copy of `v.patch_embd.weight` is zero-padded on K from
`PATCH_ELEMS` (588) to `PATCH_K_PAD` (608) — exactly
`1024 * (608-588) * 2 = 40_960` extra bytes. **The tower is correct; the
expectation was stale.** The assertion now derives the pad term.

### Tower vs the CANONICAL PyTorch reference

New harness (both halves are throwaway-free and re-runnable):

* Python: `scripts/gen_canonical_vision_vectors.py` (self-contained GGUF
  reader + driver) loads `mmproj-F16.gguf` straight into the reference
  `vision.py` `ViT`/`Aligner` modules (f32 CPU), runs
  `image_processor.load_image` and `build_image_block`, and dumps raw f32
  patches / hidden / aligner / block. Needs `vision.py` +
  `image_processor.py` on `--ref-dir`; torch/numpy/pillow come from
  `nix-shell -p python3Packages.{torch,numpy,pillow}`.
* Rust: `crates/v4flash-vision/tests/canonical_vs_python.rs` replays the
  **same patch tensor** through `Tower::encode_rows` on the iGPU and diffs.
  Env: `CANON_DIR`, `CANON_PNG`, `DEEPSTRIX_MMPROJ`, `DEEPSTRIX_VISION_DEVICE=1`.

Cases: a 640×480 PNG (gradients + circle/rect/triangle/diagonals, made with
Pillow 12.3.0) and the 4×6 synthetic grid.

| case | rows | max_abs | rms_err/rms | 1−cos | **argmax agreement** | top-5 overlap |
|---|---|---|---|---|---|---|
| 640×480 aligner (35×46 → 12×16) | 192 | 5.21e-3 | **3.75e-3** | 7.0e-6 | **192/192** | 99.7% |
| 640×480 assembled block | 209 | 5.21e-3 | 3.74e-3 | 7.0e-6 | **209/209** | 99.7% |
| synth 4×6 aligner | 4 | 9.64e-4 | 2.79e-3 | 2.6e-6 | **4/4** | 100% |
| synth 4×6 block | 13 | 9.64e-4 | 2.66e-3 | 2.5e-6 | **13/13** | 100% |

Metrics were computed twice — in the Rust test and independently in numpy —
and agree. The residual is f16-activation noise, and it is **an order of
magnitude smaller than the bf16 path the reference itself runs**
(`vision.py` bf16 vs f32: `rms_err/rms` 4.1e-2 … 6.6e-2).

Two further cross-checks:

* **Layout is exact, not approximate.** `layout_for` reproduces Python's
  `build_image_block` `types` and `perm` vectors *element-for-element*
  (asserted, both cases), including the compress pad, the odd-row pad, the
  column-major 2-row interleave and the trailing `pad_last`.
* **Preprocessing is bit-exact.** Our decode → PIL-semantics resize →
  `ImageOps.pad` → normalise → patchify chain, rounded to bf16, equals
  `image_processor.load_image`'s tensor on **0 of 946,680 elements
  differing** for the real 640×480 PNG. Un-rounded, 914,462 elements differ
  by at most **1.930e-3 = half a bf16 ulp for |x|<1** — i.e. the *only*
  difference is that we deliberately keep f32 patches where the reference
  casts to bf16 for its bf16 model. That is strictly more precise input.

### Memory + wall time (iGPU, `mem_info_gtt_used` deltas)

| | |
|---|---|
| tower weights resident | `device_bytes()` = **890.0 MiB** (932,339,712 B f16 + 40,960 B patch-embed K pad + 827,392 B f32 norms/biases) |
| GTT delta, weights loaded | **+936.1 MiB** (steady state, host copy dropped) |
| GTT delta during 640×480 encode | **+106.0 MiB** (`workspace_bytes()` = 94.3 MiB) |
| GTT peak, tower + 640×480 encode | **+1042 MiB** ≈ 1.02 GiB |
| GTT after `drop(tower)` | back to baseline (0.0 MiB above start) |
| **640×480 encode wall** | **189.8 ms cold / 187.1 ms warm** (bench best 191.4 ms); deterministic run-to-run (byte-identical outputs asserted) |

Scaling and where the time goes (`encode_bench`, profiled):

| image | patches | best ms | attention share | workspace |
|---|---|---|---|---|
| 448×448 | 1024 (32×32) | 129.9 | 20.0% | 59.9 MiB |
| 640×480 | 1610 (35×46) | 191.4 | 31.7% | 94.3 MiB |
| 1920×1080 | 3108 (42×74) | 453.2 | 55.6% | 181.1 MiB |

`vit_attention` is the lever at large `n` (O(n²), VALU-issue bound); see the
roofline block at the top of `tower.rs`.

---

## What landed

* **`crates/v4flash-vision`** (new) — `preprocess` (decode / PIL resize /
  pad / normalise / patchify), `layout` (`grid_tokens`, `solve_resize_ratio`,
  `safe_resize`, `build_image_block`), `rope` (2-D ViT RoPE), `mmproj`
  (typed 427-tensor loader), `kernels` + `kernels/vit.hip` (f16 WMMA GEMM,
  bidirectional attention, rmsnorm, rope-split, swiglu, unfold),
  `tower` (`Tower::load` / `encode` / `encode_rows` / `place_rows`), and
  `reference` (CPU f32 / f16 / bf16 twin used as the correctness bar).
* **Engine** — `v4flash_kernels::het::image_spans`: per-row image
  visibility, the widened raw-attention window
  (`IMAGE_RAW_WINDOW_MAX = SWA_WINDOW + 384`), and chunk/lane cut planning
  so an `[IMAGE_START..IMAGE_END]` span is always prefilled inside one
  KV-visible unit. `attention_swa_batched` now takes non-trailing windows.
  `bias_vl` routing bias loaded from a per-model sidecar.
* **Server** — `ChatMessage` keeps an ordered `parts` list alongside the
  text view; `vision_prompt` expands `<｜deepseek_image｜>` placeholders
  into synthetic-id blocks (`VOCAB_SIZE + type`), mixes each image's blake3
  `content_hash` into the snapshot key / LCP byte stream so two prompts with
  identical layouts but different pixels cannot alias, and holds a bounded
  memo of tower outputs.

## API

```
deepstrix-server --mmproj <mmproj-F16.gguf> [--allow-image-dir <dir> ...]
  env: DEEPSTRIX_MMPROJ, DEEPSTRIX_ALLOW_IMAGE_DIRS (':'-separated)
```

Without `--mmproj`, requests carrying images get **HTTP 400**; startup also
fails fast if the text GGUF lacks `<｜deepseek_image｜>` or the `bias_vl`
sidecar is missing, rather than dying on the first image request.

`POST /v1/chat/completions` takes OpenAI content parts:

```jsonc
{"role": "user", "content": [
  {"type": "text", "text": "what is in this image?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR..."}},
  {"type": "image_url", "image_url": {"url": "/abs/path/pic.jpg"}}   // needs --allow-image-dir
]}
```

**Image sources are `data:<type>;base64,...` and absolute local paths only.
The server never fetches http(s) URLs** (`ImageSource::Unsupported` → 400).
Local paths must sit under an `--allow-image-dir`; with no such flag, only
`data:` URLs are accepted. `MAX_IMAGE_BYTES` = 16 MiB, and axum's body
limit is derived from it (`MAX_REQUEST_BODY_BYTES`) so a large `data:` URL
no longer dies with an opaque 413.

Library surface: `Tower::load(&Path, Device)` → `encode(&PreprocessedImage,
&ImageLayout)` (block-ordered rows, sentinels placed) or `encode_rows`
(aligner rows only) + `place_rows`; `preprocess(&[u8])`,
`layout_for(&PreprocessedImage, start_pos)`.

## NEEDS USER SIGN-OFF — the `image` crate

`image 0.25.10` (`default-features = false`, features `["png", "jpeg"]`) is
the **only** new dependency, used solely for `decode → RGB8`; resize, pad,
normalise and patchify are implemented in-crate to PIL semantics.

Per the stop-after-`Cargo.toml` rule, the honest surface is **14 crates into
`Cargo.lock`, not two**: `image`; `png`, `fdeflate`, `flate2`,
`miniz_oxide`, `zlib-rs`, `crc32fast`, `simd-adler32`; `zune-jpeg`,
`zune-core`; `bytemuck`, `byteorder-lite`; `moxcms`, `pxfm`.
**`moxcms`/`pxfm` (ICC colour math) are non-optional deps of `image`
0.25.10, so `default-features = false` cannot drop them.**

Narrower alternative if that surface is unwanted: depend on `png` and
`zune-jpeg` directly and sniff the format in `preprocess::decode_rgb` —
drops 5 of the 14 crates, costs a hand-rolled `DynamicImage → RGB8`.

## Verified

* Build + full workspace test link, clean.
* Preprocessing **bit-exact** vs Pillow 12.3.0 + PyTorch on a real PNG.
* Block layout (`types`, `perm`) **exact** vs `build_image_block`.
* Tower forward vs **canonical PyTorch**: `rms_err/rms` ≤ 3.8e-3,
  **argmax agreement 100%** on every row, on a real image and a synthetic one.
* Tower forward vs the Rust CPU f16 twin and f32 oracle (`tower_encode`),
  at 4×6, 23×34 and 32×32 grids.
* Per-kernel oracles (gemm / attention / rmsnorm / rope) vs CPU.
* Encode is deterministic (byte-identical across runs).
* Text-side image visibility + raw-window arithmetic vs a Python table
  derived from `model.py` (`image_visibility`, 7 cases).
* `bias_vl` sidecar format / path derivation / corruption rejection.
* Memory: tower + a 640×480 encode fit in ~1.02 GiB of iGPU GTT and are
  fully released on drop.

## What remains

1. **End-to-end image prefill vs llama.cpp / ds4 — NOT DONE.** This needs
   the full text model resident, i.e. **the production server down**, and
   was out of scope for this pass. Everything above stops at the tower
   output + the host-side geometry; nothing has yet checked that a real
   prompt containing an image produces the right *logits*. This is the
   single biggest open item.
2. **`bias_vl` sidecar — fetched and sanity-checked, not behaviourally
   validated.** `bias_vl.bin` (44,032 B = 43×256 f32) is present for BOTH
   Vision-Exp models under `~/.cache/deepstrix/models/<gguf stem>/`, built
   by `scripts/fetch_bias_vl.py` from the HF safetensors, and
   `real_sidecar_loads_and_is_plausible` now passes on both: 11,008 values,
   range [−0.8231, 2.8869], layer-0 mean 1.6123, layer-42 mean 2.2592.
   That is format + range + identical content across the two model dirs —
   it has still **never been checked against a reference forward pass**,
   i.e. nothing proves the routing it produces for IMAGE tokens matches
   `Gate.forward`.
3. **Bidirectional-window oracle on REAL KV.** `attention_swa_visible_window`
   passes on the iGPU (text causal ≤128 keys `max_abs` 1.19e-7; a
   non-trailing image-span row with a 493-key forward-looking window
   2.38e-7; the 512-key cap 1.79e-7), and `image_visibility` pins the index
   arithmetic against a Python table from `model.py` — but both use
   **synthetic buffers**. No test yet runs the widened window over a real
   model's KV cache and compares against a reference attention output.
   Same blocker as (1).
4. `vision_max_n_token` budget (384) is only exercised by layout unit
   tests; no end-to-end check that a huge image degrades gracefully.
5. Multi-image prompts are implemented (`expand_images` loops) but only
   unit-tested — no GPU/e2e coverage.
