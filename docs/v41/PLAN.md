# DeepSeek-V4.1-Flash on two Strix Halo boxes + one 9070 XT — plan (rev 6, end of 2026-09-12)

**Where things stand (for the morning):**
- **Weights: DOWNLOADED**, 476 GB, all 48 shards, at `~/.cache/deepstrix/models/dsv4.1f` (symlink
  into `/persist/hf_cache`). Abandoned partial copy at `/persist/lumi/models/dsv4.1f-full` (65 GB,
  root-owned) — ignore for now.
- **CPU oracle: VALIDATED** (`scripts/v41_oracle/`): 40 layers → " Paris", deterministic. A
  ~120-token reference dump was generating overnight → `scratchpad/oracle_t200`.
- **V4-Flash shipped today, all bit-exact, UNCOMMITTED:** prefill 716→815 tok/s @32K (compressor
  gather + tiled matvec), model load 86→46 s (hot-expert 17x read amplification fixed).
  Server is up on that build with `DEEPSTRIX_EXPERT_TRACE` armed.
- **Decision-relevant measurements:** all-native V4.1 ≈ 32 tok/s at zero quality loss (LRU
  residency + workqueue bypass + direct-to-device miss path); dGPU is the prefill ceiling; expert
  E2M1 entropy 3.9/4 bits. Next unvalidated assumption: caching result under AGENTIC traffic.
- **Second box** arrives ~2026-09-14. Put its weights on a plaintext partition.
- **Engine port plan: `docs/v41/ENGINE_PORT.md`** (rev 1, under review): feature-gated consts, `WeightSrc` seam, M0–M8.
- **No GGUF conversion (user call, 2026-09-12 afternoon): the engine loads the HF safetensors
  directly.** `v4flash-core` gained `SafetensorsDir` (sharded pread reader) and `V41HfWeights`
  (HF tensors presented under llama.cpp `deepseek4` names with the converter's transforms —
  MXFP4 repack, fp8(32×32 e8m0)→Q8_0, casts — applied at read time; per-expert reads for the
  het loader). `scripts/v41_convert/to_gguf.py` is kept only as the byte oracle
  (`tests/hf_v41_fixture.rs`): **layers 0, 1, 2 byte-identical, 85 tensors / 1.1 GiB**, incl.
  Engram, compressor + indexer roles. Engram tables and DSpark are exposed raw, not presented yet.
  Transform throughput (64 experts, cold): MXFP4 repack is disk-bound (2.2 GB/s at 8 threads,
  10.9 warm); fp8→Q8_0 is CPU-bound at 1.1 GB/s / 8 threads (~7 s per model) — acceptable, and
  native fp8 dGPU kernels would delete it.

Status: **DRAFT (rev 6)**. Earlier revision notes follow.

---

## 0. TL;DR

- 552B backbone + 196B Engram. 40 layers = 20-layer causal encoder + 20-layer decoder (CED).
  384 routed + 1 shared experts, top-6. Pure CSA2 sparse attention with cross-layer KV/index
  reuse. FP4 main KV → **890 B/token global KV; 1M context = 890 MB**. Engram = hash-table
  memory (12 KB gathered/token) that lives on NVMe. DSpark = 5-position speculative drafter.
- Native backbone is ~286 GiB (experts MXFP4 269 GiB + FP8 rest). The boxes are **96 + 128 GB**
  (this box is 96: lsmem 96G, MemTotal 93.4 GiB) → ~211 GiB for routed experts. **It does not
  fit at native precision.** Phase 1 = **uniform IQ3_XXS experts (194 GiB + DSpark 5) — fits
  with ~12 GiB spare, no SSD tier, and ~2 ms *less* decode than a native-heavy mix.** The spare
  promotes the hottest ~15% of experts (~2,600) to native MXFP4 ≈ half of all decode picks.
  SSD-cold streaming and a 3rd box are optional later levers, not prerequisites. (User: the
  V4-Flash IQ3_XXS quant is fine in practice → 3-bit is an acceptable prior for this family.)
- Topology: **expert-parallel**, this box = hub (keeps the dGPU on its PCIe4 x4 link), new box =
  remote expert executor. One RTT per layer per decode step.
- Prefill: **layer-major over 16-64K super-chunks** (residual 40 KB/token stays in RAM). Kills the
  per-chunk expert re-read (60× less weight traffic), makes SSD-cold experts free on long
  prompts, and enables streaming experts to the dGPU for its spare FLOPs.
- Goals (tok/s): pp 2000/1800/1200 @32K/100K/1M base, 3000/2700/1600 with dGPU expert streaming.
  tg **32/31/29 all-native at ZERO quality loss** (44/43/40 w/ DSpark), 37/36/33 at IQ3_S. See §7.

---

## 1. Model facts (from config.json / report §2)

| | V4.1-Flash | notes |
|---|---|---|
| layers | 40 (+3 DSpark) | enc 0-19, dec 20-39; layers 0-1 SWA-only |
| hidden / mHC | 5120, hc_mult 4 | residual = 4×5120 f16 = 40 KB/token |
| experts | 384 routed + 1 shared, top-6 | `sqrtsoftplus`, `noaux_tc`, `swiglu_limit` 10, scale 1.5 |
| expert size | 3×5120×2304 = 35.4M | **17.9 MiB (18.8 MB) at MXFP4** (17 B / 32 w) |
| attention | MLA: 64 heads, head_dim 512, rope 64, q_lora 1280, kv heads 1 | W_O grouped low-rank (o_lora 1024, 8 groups) |
| CSA2 modes | Full: 2,8,14,20 · Reindex: 24,28,32,36 · rest Reuse | enc ratio 2, dec ratio 1 |
| indexer | 32 heads × 128, top-512; FP4 QAT | hierarchical: layer-20 Full builds 2048 blocks×8 = 16K pool for Reindex layers |
| SWA | window 128, every layer, FP8 KV | decoder SWA via **bounded replay** of last 128 tokens |
| main KV | E2M1 + E4M3 scale / 16 ch, quantized after RoPE | ≈ 890 B/token global |
| Engram | layers 1, 14; 8 heads × orders {2,3,4}; ~16M-row FP8 tables | 189 GiB on disk; 48 random 256-B reads/token |
| DSpark | 3 blocks, SWA-128, 128 experts top-3, block 5, Markov rank 256 | ~6.7 GiB experts at FP4 |
| vision | 32-layer ViT, hidden 1024, patch 14, 3×3 unshuffle | different tower from V4-Flash |
| rope | yarn ×16 from 64K → 1M | `rms_norm_eps` 1e-20 (compute norms in f32) |

Param accounting: routed experts 40×384×35.4M ≈ 544B ≈ **269 GiB** native; non-expert ≈ 17 GiB
(attn+hc+gate ~6.7 GB FP8 — wq_b/wo_a/wo_b are 117 MB/layer, see ARCH_SPEC §6 —, shared 1.4,
embd+head 2.5, DSpark ~7, ViT 0.6, Engram projections).

CED prefill: prompt runs **only the 20 encoder layers**; decoder global KV for its Full layer is
projected from H_20 (cheap matvec); decoder runs over the last 128 tokens only.

---

## 2. Hardware + memory budget

| device | physical | usable for routed experts | notes |
|---|---|---|---|
| this box (hub, 96 GB) | 93.4 GiB visible | ~85 GiB | minus OS/process ~4.7, super-chunk residual 1.3-2.6, misc |
| new box (expert server, 128 GB) | ~125 GiB | ~120 GiB | daemon only |
| 9070 XT | 15 GiB | ~4 GiB hot | attn+hc+gate 6.0 + shared 1.4 + head 0.7 + KV@1M 0.9 + scratch 2.5 (ARCH_SPEC §6) |
| **total** | | **~209 GiB** | need 269 + 6.7 (DSpark) = **276 GiB native → deficit ~67 GiB** |

dGPU link: **OCuLink**, PCIe 4.0 x4 (00:03.1 → Navi 48 internal switch), **~7 GB/s measured** (PHASE0).
Box↔box link: TBD — USB4 host-to-host first; **measure RTT + iperf3 on day one**.
NVMe: PCIe 4 x4, ~7 GB/s seq per box. This box: 1.1 TB free after deleting the 731 GB oracle dump.

---

## 3. Placement — phase 1 is uniform IQ3_XXS; tiers and SSD streaming are later levers

**Phase 1 (target):** every routed expert at IQ3_XXS (12.9 MiB each → 194 GiB) + DSpark experts
(4.8 GiB) = ~199 GiB of ~211 available. Decode expert bytes 240 × 13.5 MB = 3.2 GB/token (vs
4.1 GB native-heavy) → ~2 ms faster. **Requires our own imatrix** (no llama.cpp support): the
layer-streaming oracle (§8.2) emits per-channel activation statistics per expert input, which
is exactly an imatrix. First experiment when weights land: requant a few layers MXFP4→IQ3_XXS
with that imatrix, compare per-layer output error vs native; also scan per-layer sensitivity
(V4-Flash needed blk.26 at higher precision — expect analogues).

**Phase 1b (quality upgrade in the spare):** promote experts chosen by **measured expected
error per byte with a per-evaluation error floor** (see below), not by pick count, to IQ3_S or
native MXFP4 in the ~12 GiB spare plus the dGPU. Placement (which device) still follows pick
frequency — that is a bandwidth question; precision is an error question. Same het-split loop,
one more format per slot.

**Quantizer dependency:** no llama.cpp support → no llama-quantize. IQ3_XXS/IQ3_S with imatrix
are the codebook-search half of ggml; plan is a small FFI shim onto `ggml-quants.c`
(self-contained, MIT) rather than a reimplementation.

**MEASURED 2026-09-11 (layer 1, 142M expert weights from the landing shard): the E2M1 codes
are near-uniform — entropy 3.90 bits/weight of 4.** Magnitude shares 0/.5/1/1.5/2/3/4/6 =
11.7/14.1/20.4/9.9/17.1/12.5/10.2/4.2%; 14% of weights at |w|∈{4,6} vs ~4% for a Gaussian at
that block scale. Lossless coding = 4.15 bpw (2% below native). All 4096 magnitude 4-tuples
occur; the top-256 cover 23%, top-512 37% → IQ3_XXS/IQ3_S hit the exact tuple <¼ / ~⅓ of the
time. **Any 3-bit format discards ~0.85 bits of genuinely used information per weight.** This
is a materially worse prior than V4-Flash's bf16→IQ3_XXS (bell-shaped source). Consequences:
- the phase-1 quality experiment is now load-bearing, not a formality;
- lossless options on 96+128 are only *native + ~25% SSD-streamed* (≈9 cold reads/token ≈ +24
  ms → ~20 tok/s decode; prefill unaffected under layer-major) or a **3rd 128 GB box** (329 GiB
  vs 276 needed, full speed) — the histogram argues for the 3rd box if output quality is the goal;
- block-of-32 max is always code 4 (40%) or 6 (60%); −0 code used 5.8% (sign kept on underflow).
Script: scratchpad `e2m1_hist.py` (reads tensors from a partially-downloaded shard).

**Correction to the paragraph above:** the entropy finding kills *lossless* and the *fit of
IQ3's fixed codebooks*; it does not make the lossy trade-off worse than for a bf16 model. At a
given rate the distortion relative to weight variance is what it is (~2% at 3.4 bpw), and the
QAT'd model absorbed its own 4-bit noise for free. So a 3-bit requant here is the standard
"3-bit a 70B" experiment (small, real loss on hard evals; ~nothing on PPL), provided the
quantizer is matched to the source — which is what the recipe below is for.

### 3.2 Requant recipe (ranked by lever size at fixed bytes)
1. **Hessian-aware error-feedback rounding (LDLQ/GPTQ)** into the *existing* IQ3_S/IQ3_XXS
   formats: quantise 4 columns at a time into the codebook, push the residual into the columns
   not yet done. Worth ~1 bit at this rate in every ablation; zero new inference kernels.
   (Current plan = diagonal imatrix + nearest, i.e. weighted RTN — this replaces it.)
2. **Random signed Hadamard rotation on the input side** (W' = WH offline, x' = Hᵀx online;
   5120 = H₂₀⊗H₂₅₆, 2304 = H₃₆⊗H₆₄): rotated weights are Gaussian i.i.d. by CLT → the
   Gaussian-designed IQ codebooks are back on their design point, block scales become nearly
   redundant, the Hessian becomes incoherent (what LDLQ wants). Runtime: one 5120-pt FWHT per
   token per layer (gate/up share it) + one 2304-pt per token-expert before down, fused into
   the gather; no un-rotations. **Per-expert opt-in**: native (E2M1) experts must stay
   unrotated (rotation destroys exact representability), so rotated experts get x' and native
   ones get x — two activation views per token per layer, +10 KB/token, trivial.
3. **Trellis-coded quantisation** (QTIP-style / ik_llama IQ3_KT) only if 1+2 into IQ3_S still
   shows visible damage: near the Gaussian R-D bound; new dequant kernels (a few int ops per
   weight, comparable to IQ3's LUT+sign path).

**One pipeline:** Hessians are E[xxᵀ] per expert (5120² shared by w1/w3, 2304² for w2) = 48 GB
per layer fp32 → must be layer-major: run calibration tokens through layer l with native
weights, accumulate the 384 Hessians, LDLQ-quantise the layer's experts, discard, advance.
This *is* the layer-streaming oracle + imatrix collector + quantiser as one program, same loop
nest as the prefill design; running layer l+1's calibration on layer l's *quantised* outputs
gives cross-layer error feedback for free. A few GPU-hours for all 15,360 experts. Cold experts
with few samples: standard damping; if their per-evaluation error still exceeds the floor they
go native (the rare-but-important guard).

**Expectation:** rotated + Hessian-aware IQ3_S ≈ the standard 3.4-bpw band (a couple of % on
hard evals), roughly half the noise of plain imatrix IQ3_S on this grid source. Still lossy;
the 3rd box remains the only zero-loss full-speed answer. Measure on the user's transcripts
(§3b) before committing bytes.

**Why not a bigger uniform quant.** Per-expert format × (15,360 + 384 DSpark) vs ~211 GiB:
IQ3_XXS 3.06 bpw = 199 GiB (fits, ~12 spare) · "IQ3_XS" ≈3.3 = ~214 (over ~3; it is a model-level
IQ3_S/IQ3_XXS mix, not a tensor type) · IQ3_S 3.44 = 223 (over ~12) · MXFP4 4.25 = 276 (over
~65; IQ4_XS is the same 4.25 so ≥ that means keep native). Nothing standard sits in 3.5-4.2.
Since any "XS" is already a two-format mix, choose the format **per expert by measured expected
error, not by tensor role and not by pick frequency alone** (user, 2026-09-11: rarity is the wrong
importance metric — some rare experts matter a lot). Expert e's contribution to output damage ≈
P(e fires) × E[g² ‖(W_q − W)x‖² over e's real inputs] × layer sensitivity. Measure all three:
(a) per-expert per-format error on real inputs — free during the oracle's imatrix pass (~120
TFLOP/layer on the calibration set, ~1 h CPU for 40 layers); (b) Σ gate weight per expert added
to the stats sidecar next to counts; (c) per-layer sensitivity by injecting one layer's quant
error into the oracle forward and measuring final-logit KL (40 runs). Allocate greedily by error
reduction per byte, **subject to a minimax floor: no expert exceeds a per-evaluation error cap
however rarely it fires** — that is the guard for the rare-but-critical case. The V4-Flash blk.26
upcast was a sensitivity finding, not a frequency one; expect analogues.

Illustrative sizing of what the ~12 GiB spare can buy (final split comes from the measurement):
- **IQ3_S on ~7,700 experts (50%)** — +1.6 MiB each; no new kernels (iq3_s kwide exists and is
  faster than iq3_xxs); ~3.6 GB/token decode bytes if those are the frequent ones; or
- **native MXFP4 on ~2,600 (17%, lossless there)** — +5.0 MiB each; needs the MXFP4 expert
  kernel; ~3.9 GB/token if those are the frequent ones.
Decide by the per-layer oracle error: if IQ3_S closes most of the gap to native, take it. Note
phase 1 + IQ3_S promotion needs **no MXFP4 GPU kernel at all** (oracle runs native on CPU).

**Later / optional:** SSD-cold tier to push the native share higher. The math below is kept for
that decision.

### 3.1 SSD-cold streaming math (deferred)

Measured on V4-Flash production routing (Vision-Exp IQ3_XXS `expert_stats.json`, 1.8M decode /
9M prefill tokens, experts ranked globally by pick count):

| coldest share of experts | share of decode picks | cold reads / V4.1 token (240 picks) | decode tax @2.7 ms/read |
|---|---|---|---|
| 5% | 0.15% | 0.4 | ~1 ms |
| 10% | 0.59% | 1.4 | ~4 ms |
| 15% | 1.32% | 3.2 | ~9 ms |
| 20% | 2.34% | 5.5 | ~15 ms |

(prefill picks are flatter: coldest 15% = 2.9%. V4.1 has 384 experts at the same top-6, so the
tail should be at least this thin; re-measure on V4.1 traffic and re-place — same M62 loop.)

Cold reads sit on the decode critical path (routing at layer l known only after attention l).
DSpark verification touches more distinct experts per step, so big cold sets hurt more with it.
**Short prefills (agentic appends of 0.5-1K tokens) touch ~80-90% of the cold set per layer
regardless of loop order** → every turn pays ~cold_set_bytes / SSD_BW once. That, not long
prefill, is what bounds the cold set: 5% ≈ 1 s/turn on one SSD, 0.5 s split over two.

Candidate tierings if pushing the native share beyond phase 1b (each IQ3_XXS expert saves
5.0 MiB vs native; each streamed expert saves 17.9 MiB):

| tiering (native / IQ3_XXS / SSD) | decode tax | native tier covers (decode picks) |
|---|---|---|
| 30% / 65% / 5% | ~1 ms | ~70% |
| 40% / 50% / 10% | ~4 ms | ~78% |
| 0% / 100% / 0% (plain IQ3_XXS requant) | 0 | 0% |
| 100% / 0% / 0% needs a **3rd 128 GB box** | 0 | 100% |

Residual quality caveat: MXFP4 → IQ3_XXS quantizes a 16-level E2M1 grid with a codebook fit to
continuous weights; could land better or worse than bf16 → IQ3_XXS. Measured by the phase-1
experiment above, not assumed.

Engram: never resident. mmap the 189 GiB tables; 48 random 256-B reads per token (≈47 IOPS/token,
~100 µs from NVMe, hidden under layer 0 at decode; batched one super-chunk ahead at prefill).

---

## 3a. Decision table — quality loss × speed × cost (a-priori; to be replaced by measurement)

Goals are **quality, performance and cost together**; the question is *how much quality is lost*
per configuration. Estimates relative to native; anchors: best-in-class 3.4-3.5 bpw on 70B-class
models = 1-3% on hard evals; 3.0 bpw = 2-5%; V4-Flash IQ3_XXS fine in daily use, IQ2_XXS not
(23% argmax disagreement). Uncertainty on every lossy row ≈ 2× either way.

| configuration | est. loss (hard tasks, relative) | est. decode | extra cost |
|---|---|---|---|
| native + SSD tier, **LRU residency** (§3.3, measured) | **0** | **~33 tok/s** | none |
| native + SSD tier, static placement | 0 | ~13 tok/s | none |
| ~40% native + rotated-LDLQ IQ3_S rest + 5% streamed | ~0.5-1.5% | ~35 | none |
| uniform rotated-LDLQ IQ3_S | ~2-3% | ~37 | none |
| uniform rotated-LDLQ IQ3_XXS | ~3-5% | ~40 | none |
| plain imatrix IQ3_XXS, unrotated (V4-Flash recipe) | ~5-8% | ~40 | none |
| 3rd 128 GB box, all native | 0 | ~37+ | one box (**devalued by §3.3**) |

**When the real numbers arrive:** within ~1-2 days of the weights landing, the CPU layer-streaming
oracle gives per-expert per-format error tables + end-to-end teacher-forced KLD / decision-token
agreement on the user's transcripts for each whole-model configuration → the rows above become
measured, and the allocation curve (loss vs bytes spent on native experts) is filled in *before
any GPU port exists*. Free-running task evals and DSpark acceptance confirm once the port runs.
**Therefore the quantiser pipeline (§3.2) is on the critical path for deciding, not just
building — start it the moment the download completes, ahead of kernel work.**

### 3.3 Cold-expert residency: dynamic caching beats static placement — **MEASURED**

Full report: **`docs/v41/COLD_EXPERT_CACHING.md`** (4,672-token decode trace, two conversations,
new `DEEPSTRIX_EXPERT_TRACE` hook). Origin: user asked whether the Single-Pass-mHC trick (trade
staleness for a broken dependency) applies to expert routing.

**The retrain is unnecessary.** Pre-gated MoE moves layer l+1's gate into layer l and needs
retraining. We do not need to *change* routing, only to *guess* it: a wrong guess is a **cache
miss, not a wrong answer**. Zero quality cost, zero retrain.

**But the dominant lever turned out to be caching, not prediction.** Measured, at equal memory:

| resident | segment | static (frequency placement) | LRU | gain |
|---|---|---|---|---|
| 75% | EN technical, steady | 21.3 | 1.45 | 14.7× |
| 75% | ZH history, steady | 63.3 | 1.71 | 37.0× |
| 85% | EN technical, steady | 8.9 | 0.83 | 10.8× |
| 85% | ZH history, steady | 40.0 | 0.84 | 47.6× |
| 85% | 300 tok after a topic switch | 39.2 | 3.08 | 12.7× |

(SSD reads per token.) A static hot set from aggregate stats fits the *average* of traffic and
therefore no individual conversation: 1,165 of 11,008 expert slots were touched only by the
Chinese conversation. **"Globally cold but locally hot" confirmed and large.** Separately, 50% of
misses were also picked at token t-1 and 84% within 32 tokens, so prefetch composes on top.

**Consequence for V4.1:** at 209/276 GiB = 75.7% resident, ≈1.4 SSD reads/token instead of ~20,
i.e. ~3.8 ms instead of ~54. **The all-native, zero-quality-loss configuration goes from ~13
tok/s to ~33**, within ~10% of uniform IQ3_S (~37) — which devalues the third box and makes
"lose no quality at all" the leading option. See the revised §3a.

**Implementation may be nearly free:** dynamic residency over mmap'd expert weights is what the
OS page cache already does. Open questions are 4 KiB granularity vs 18.8 MB experts (needs
`MADV_WILLNEED` over the expert extent), and whether GTT-mapped iGPU access can be served from
page cache. Fallback is an explicit LRU over expert slots, a few hundred lines.

**Caveats:** measured on V4-Flash (256 experts/layer), not V4.1 (384); two conversations;
**agentic traffic with interleaved tool calls is untested and matters most here.**

## 3b. Evaluation protocol — what we actually optimise is output quality

The objective is **model capability on this workload**. The proxy chain is: per-layer output
damage × layer sensitivity → final KLD vs native → capability. Each link can lie: per-layer
error misses cross-layer correlation and decision-critical low-variance directions; mean KLD is
dominated by easy bulk tokens while capability lives in the tail; any teacher-forced metric is
blind to compounding in free-running generation. **Proxies allocate; real metrics validate.**

Validation ladder (closest to the goal first):
1. **Free-running task outcomes** on a small battery of the user's real agentic tasks, quant vs
   native: pass/fail, tool-call validity, length/loop pathologies. Ground truth; run for final
   decisions.
2. **Decision-token agreement, teacher-forced on the user's own transcripts**: top-1 flip rate
   and KLD restricted to tokens where native is not trivially peaked (and within code/tool-call
   spans), plus tail stats (p99 / max KLD). Catches rare-expert damage that means average away.
3. **Mean KLD vs native** — cheap regression guard only, never the target.
4. Per-layer / per-expert error tables (§3) — allocation signal only.

Native reference logits are cheap on this hardware because of the layer-major design: prefill
with *every* expert streamed from SSD reads 276 GiB once per super-chunk (~20 s on two NVMes vs
~30 s compute) → the native model runs teacher-forced over transcripts at near full prefill
speed (it just cannot decode usefully). Before the box arrives, the CPU layer-streaming oracle
gives the same at minutes per prompt.

**Live fidelity meter:** DSpark's drafter is trained against the native backbone, so its
acceptance rate on our quant, in production, is a continuous free measurement of how far our
sampled outputs sit from native (cf. MTP-K1 acceptance capped at 0.72 on IQ2_XXS V4-Flash).
Track it; a drop below the paper's rate is a quality regression signal before any eval runs.

The eval set is the user's workload (agentic coding, tool calls, long contexts), not a general
benchmark — "rare but important" is workload-specific by construction.

## 4. Topology: expert-parallel, hub + remote expert executor

- **Hub (this box):** dGPU holds attention/indexer/shared experts/hot experts for all 40 layers;
  iGPU holds its share of routed experts; KV cache, sampling, control, Engram gather, snapshot
  cache. = today's deepstrix with het-split gaining tiers.
- **Expert server (new box):** daemon that holds the other share of routed experts (+ its SSD-cold
  set). RPC: `(layer, token ids, FP8 activations, expert list) → weighted f16 sums`. No attention,
  no KV, no dGPU.
- Per decode step: 40 RTTs. Budget: RTT ≤ 100 µs → ~4 ms/token. Per prefill layer per 1024
  tokens: ~5 MB out (FP8, each token once) + ~10 MB back (f16) → ≥10 Gbps needed to stay off the
  critical path at ~2500 tok/s.
- Both boxes' expert legs run in parallel per layer; expected max of a 6-pick split over two
  boxes is ~3.7 vs mean 3 → ~20% imbalance built into the decode estimate.
- **Fallback if the link is bad:** layer-pipeline (encoder on A, decoder on B): 1 crossing/token,
  tiny bandwidth, but box B idles during CED prefill and its attention runs on its iGPU.

---

## 5. Prefill design

### 5.1 Layer-major over super-chunks
Loop nest becomes `for super-chunk S: for layer l: for chunk c in S: {attn on dGPU ∥ experts}`.
Only the **residual (40 KB/token)** crosses layer boundaries; all other activations are per-chunk
within a layer as today. KV is per-position and already exists; chunk c at layer l needs only
layer-l KV for earlier positions (written by earlier chunks of the same layer) and the source
layer's KV for Reuse/Reindex layers (complete). SWA needs only the trailing 128 tokens of the
previous chunk. Two-lane chunk pipelining carries over unchanged.

**Note (2026-09-12):** on V4-Flash this restructure would buy ~nothing, because its mechanism is
amortizing *iGPU* expert weight reads and the iGPU has 38% slack there. The V4.1 case is different
and stands: the point there is amortizing **SSD** reads, which V4-Flash has none of. Do not port
it to V4-Flash expecting a prefill win.

| S | residual buffer | SSD time / compute time (15% cold, 1 SSD, ~1000 tok/s) |
|---|---|---|
| 1K (today) | 40 MB | 3.1 |
| 4K | 160 MB | 0.77 |
| 16K | 655 MB | 0.19 |
| 64K | 2.6 GB | 0.05 |

Below 1.0 the cold prefetcher hides it fully (next layer's whole cold set, no routing prediction
needed since coverage ≈ 100% at S ≥ 16K). **No SSD swap of the residual**: it would be fine on
bandwidth (1.6 MB/token) but writes 0.8 MB/token → a 1M prefill burns 800 GB of endurance.

Give-ups: no partial progress if a prefill is interrupted mid-super-chunk. Prefix snapshots are
unaffected (KV up to position t exists after the pass).

### 5.2 Expert-major GEMMs
At B=1024, each expert sees ~16 tokens and the iGPU re-reads all ~144 GB of encoder expert
weights per 1024 tokens → **~1500 tok/s weight-BW ceiling per iGPU**, which V4.1 would hit
(MXFP4 dequant is cheap). Inside a super-chunk, gather each expert's ~1000 tokens (S=64K) into
one GEMM and read its weights once → ceiling lifted ~60×, compute becomes the limit. This is the
q2k_down by-expert inversion generalised to the whole MoE.

### 5.3 Stream experts to the dGPU for its spare FLOPs — **PREMISE UNVERIFIED (2026-09-12)**

> **Measured on V4-Flash at 32K (`docs/v41/PREFILL_BALANCE_2026-09-12.md`): the dGPU is the
> prefill CEILING, not the iGPU — 85.9% busy vs 62.3%, doing 38% more device work.** The June
> "iGPU is the prefill ceiling" note is stale. So the "dGPU has ~50% spare after attention"
> premise below is exactly the assumption that just failed on the model we can measure. V4.1 has
> a structural reason to differ (CED runs only the 20 encoder layers in prefill, halving the dGPU
> attention leg), but **treat this section as unproven until that is measured.** Also note
> V4-Flash's own prefill is dominated by `dgpu.kv_append_compressor_serial` at 28% of the binding
> device — the compressor lever flagged 2026-09-08 and never taken.


18.8 MB/expert = 2.7 ms over PCIe4 x4 ≈ the dGPU GEMM time for ~1000 token-evals → matched at
S=64K. Link budget: activations ~1 GB/s at 2500 tok/s, leaving ~6 GB/s ≈ every encoder expert
once per 64K super-chunk (~24 s vs ~26 s compute). dGPU spare after attention ≈ 50% ≈ one
extra iGPU of expert throughput (**+~50% MoE leg**). The dGPU already holds every token's
post-attention state and runs the router → its share costs no extra activation traffic; resident
hot experts are free. Mechanism: ring of ~8 staging slots, hipMemcpyAsync one layer ahead.
**Decode: never** (2.7 ms transfer serves one token) — the "no expert streaming" rule stands there.

---

## 6. Decode model (per token, ~32K ctx)

| leg | bytes / work | device | ms |
|---|---|---|---|
| routed experts | 240 picks × 13.5 MB (IQ3_XXS) ≈ 3.2 GB, split 2 boxes (+20% imbalance) | 2 iGPUs ∥ | ~8 |
| attention + hc + gate + head | ~6.7 GB FP8 (ARCH_SPEC §6) | dGPU, serial per layer | ~10.5 |
| shared experts + dGPU hot experts | 1.4 GB + hot | dGPU, overlaps the iGPU expert leg | hidden |
| network | 40 RTT | USB4 | ~4 |
| glue / launches | graph-captured | | ~3 |
| cold reads | none in phase 1 | | 0 |
| **total** | | | **~26-28 → ~37 tok/s** |

Context scaling: 4 full-range indexers (3 enc @N/2 + layer 20 @N) ≈ 20 GFLOP FP4 at 1M on the
dGPU < 2 ms; KV reads are 40 × 512 × 288 B ≈ 6 MB. **tg is nearly flat to 1M.**

DSpark: we are weight-BW bound, so verifying K drafts multiplies expert bytes by the distinct
experts touched (K=5 → ~5.7×; K=2 → ~3×; K=1 → ~2×). Sweet spot is 1-2 drafts at acceptance
~0.75 → ~50-55 tok/s. The scheduler's throughput-curve input is exactly this trade-off. Native
experts on most picks should lift acceptance vs the MTP-K1 cap we measured on IQ2XXS.

---

## 7a. Headline goals — rev 4 (2026-09-13, after M1 measurements)

**What is now measured (not modelled):**
- V4.1 MXFP4 gate/up decode leg at V4.1 shapes on the iGPU: 6 experts × 2 × 6.27 MB = 75 MB in
  0.351 ms = **214 GB/s effective** (`MXFP4_PAIR_BENCH=1`) — the rev-3 "~230 GB/s" expert term holds
  (down kernel: same family, half the bytes).
- Layer 0 of V4.1 runs at parity through the existing het engine (M1), so the per-layer kernel
  chain IS V4-Flash's chain with V4.1 widths — which makes V4-Flash's measured decode the right
  anchor: **34 ms/token = 0.79 ms/layer, dGPU-bound, ~2× above its bandwidth floor** (the per-layer
  launch/dispatch floor, see `project_decode_at_floor`). Rev 3 priced the attention leg at pure
  bandwidth (6.7 GB / 640 GB/s = 10.5 ms); the measured chain says ~22 ms for 40 layers.
- Not measured yet: the box-to-box RTT (the day-one number; plan budget 100 µs, typical
  `thunderbolt-net` 100–300 µs).

**Per-layer decode model (32K), V4-Flash-anchored:** attention chain on the dGPU
0.50 × 1.1 (5120-wide) = **0.55 ms**; MoE phase = max(local iGPU picks, RTT + remote picks + xfer,
dGPU shared+hot) with 6 picks × 18.8 MB split ~3/3 → local 56 MB / 214 GB/s = 0.26 ms, remote
0.26 + RTT + 0.02; head ≈ 1.4 ms/token (129280 × 5120 Q8).

| configuration | per-layer ms | ms/token | **tg 32K** | notes |
|---|---|---|---|---|
| two boxes, RTT 100 µs | 0.55 + 0.38 | 37 + 1.4 | **~26** | remote leg hides behind local only if issued right after the router |
| two boxes, RTT 200 µs | 0.55 + 0.48 | 41 + 1.4 | **~23** | |
| two boxes, RTT 300 µs | 0.55 + 0.58 | 45 + 1.4 | **~21** | serialised RTT is a third of the token |
| single box + SSD tier | 0.55 + 0.53 | 43 + 1.4 + misses | ~22 at the §3.3 miss rate; **~10–15 at realistic 24% residency** | 96 GB holds ~70 GB of 289 GB experts |
| + DSpark (accept ~0.75, 1–2 drafts) | | | **+30–40% → 30–36** | unmeasured |
| rev 3 said | | 31 | 32 | attention leg at bandwidth, RTT 100 µs |

tg stays nearly flat in context (top-512 + window attention; the four full-range indexers add
~2–3 ms at 1M): 26 / 25 / 23 for the RTT-100 case.

**Prefill (pp):** V4-Flash measures 815 tok/s @32K with the dGPU as the ceiling (85.9% busy).
V4.1 attention is ~1.25× the bytes and FLOPs at 5120 wide over 40 layers → ~700 tok/s for a
plain 40-layer pass; **CED halves it**: only the 20 encoder layers run over the whole prompt, the
decoder layers produce KV/keys for all positions from H_20 (cheap) and run attention+MoE over
the tail → ~**1300–1400 tok/s @32K**, ~1200 @100K, ~900 @1M (indexer growth). The second box
adds little to prefill (dGPU-bound) unless encoder layers are pipelined across boxes (+~30%
from box 2's iGPU); dGPU expert streaming is dead (§5.3). Rev 3 said 2000 / 1800 / 1200.

**What the second box actually buys:** residency. Expert-parallel halves the expert leg
(−10 ms/token) but the RTT gives 4–12 ms back; the real win is not running a 24%-resident
SSD tier, which on realistic traffic would sit at 10–15 tok/s. What recovers rev 3's 32: the
per-layer launch floor (fusion, not graphs — measured on V4-Flash), overlapping the remote
request with the local expert compute, and DSpark. **Decision input from day one: the RTT.**

### 7a.1 Numbers under the §7b schedule (rev 4b, 2026-09-13)

Inputs: attention chain 0.55 ms/layer (V4-Flash's measured launch-bound chain × 1.1), expert
pick 18.8 MB at the measured 214 GB/s = 0.088 ms, picks land where experts live (routing is
flat, so shares ≈ residency: box 1 ~75 GB ≈ 38% of picks, box 2 ~115 GB ≈ 47%, dGPU hot ~5%,
SSD ~10% behind the LRU), remote branch = its picks + RTT + 20 µs, head 1.4 ms, LRU misses at the
§3.3 placeholder (1.9 ms/token — unmeasured at 66% residency, could be 5–10).

| | RTT 100 µs | RTT 200 µs | RTT 300 µs |
|---|---|---|---|
| single stream, 32K / 100K / 1M | **26 / 25 / 23** | 23 / 22 / 20 | 21 / 20 / 19 |
| + DSpark K=2, acceptance 0.75 | **37 / 36 / 33** | 35 | 33 |
| + DSpark K=2, acceptance 0.60 | 31 | 30 | 28 |
| two concurrent requests, aggregate | ~42 | ~38 | ~35 |
| pp 32K / 100K / 1M (dGPU-bound, CED) | **1300–1400 / 1200 / 900**; encoder layers pipelined over both boxes: ~1700 / 1500 / 1100 | | |

DSpark arithmetic (per verify step, K drafts, RTT 100 µs): K=1: 1.18 ms/layer → 52 ms for
1.75 expected tokens = 34 tok/s; **K=2: 1.46 ms/layer → 63 ms for 2.31 tokens = 37**; K=3:
1.71 ms/layer → 74 ms for 2.73 tokens = 37. Speculation amortises the serial per-layer chain
over a batch; it does not overlap steps (each step waits for the verify result). True cross-step
overlap only comes from a second independent stream (~1.6× aggregate, latency unchanged).

Upside not in the table: the launch floor (V4-Flash's chain is ~2× its bandwidth time; fusing
it to 1.5× is worth ~+4 tok/s single-stream); a third box (residency → fewer misses, small).
Unmeasured inputs, in order of leverage: RTT, DSpark acceptance, LRU miss rate at 66%.

## 7c. Three independent decode designs (2026-09-13) — `docs/v41/decode_designs/`
A (bytes/launches): persistent per-layer dGPU kernels + doorbells + fp8-native projections →
single 32 (28–35), DSpark ~43. C (graph restructure): nothing beats the current placement; exact
mHC bridge split (+1–5 ms); DSpark drafter must live on the dGPU; single 27–29, DSpark ~37.
B (latency hiding): routing prediction measured DEAD; **layer-wavefront verify** (two sub-batches
trailing by one layer, exact) hides the RTT and lights the tracks → 47 at p=0.75 without misses;
**SSD misses are the un-hideable term**: at 66% residency ~3.5 misses/position → 25–29 whatever
the schedule; 75% + box-2 plaintext cold tier → 36–38. Synthesis and order in §7d.

## 7d. Synthesis of A/B/C and the order of work (2026-09-13)

The three designs are complementary, and together they say where 50 tok/s at zero loss would come
from (matches the backward analysis: the single-stream bandwidth roofline of this placement is ~48):

| lever | owner | single-stream effect | with speculation | status |
|---|---|---|---|---|
| fused per-layer chain (persistent kernels, doorbells, fp8-native projections) | A | 0.55 → 0.34 ms/layer (+6 tok/s) | enables 3-stage wavefront (47 → 55) | Exp-0 gates the persistent kernels |
| layer-wavefront verify (exact) | B | — | hides RTT, lights all tracks: 33 → 47 at p=0.75 | needs B=k verify on the decode graph |
| residency: disjoint LRUs, directory, cold tier on box-2 plaintext NVMe | B | misses are the un-hideable term | 66% → 25–29; 75% + 3 ms cold → 36–42 | trace-measured on V4-Flash; V4.1 rate unknown |
| mHC bridge split (exact) | C | +1–5 ms/token | fills a slice of the dGPU idle | 2–3 days, bankable on V4-Flash |
| drafter on the dGPU, adaptive K, early-exit, Engram prefetch at step start | C/B | — | D 9 → 3.5–5 ms/step; +0–8% | after acceptance is measured |
| routing prediction | B | id→id predictor dead (0.15); **hidden-state predictor MEASURED alive: layer (l+k)'s router on the mean-copy residual after layer l recovers 70/85 % of true top-6 within top-6/12 at k=1, 56/69 at k=3, 47/59 at k=5** (`scripts/v41_oracle/route_probe.py`) | — | **prefetch still dead on economics**: misses are the low-confidence tail — at 66 % residency, k=2–3, prefetching predicted top-6 non-residents catches 42–47 % of misses but issues 15–17 reads/token (280–310 MB, ~10 % precision); rank-limited (top-3) catches 24 % for 3.9 reads/token (73 MB ≈ 18 ms of NVMe) to save ~2 ms. **Correction (user): the NVMe is idle most of the time** — 3.6 misses × 4.8 ms ≈ 17 ms of a ~40 ms token, ~43 % busy (≈ 22 % per box with two NVMes). Prefetch issued at strictly lower priority than real misses and capped to the idle budget (~23 ms ≈ 90 MB/token per NVMe) never hurts the critical path: top-2/top-3 predictions at k=3–5 fit the budget and hide 0.6–0.9 misses/token → **2.5–4 ms/token (6–10 %) for free**; prefetch into host RAM (page cache) rather than device slots so a wrong guess costs nothing but RAM, and promote on real use. Beyond that, use the predictor for eviction protection / promotion (zero I/O) and pre-staging. |
| learned lookahead router (user idea) | — | **MEASURED (`route_probe_train.py`, k=2, 704 train / 302 test positions of one transcript):** probe initialised from gate_{l+k} and ridge-regularised toward it — plain CE: recall@6 0.624 → 0.668 overall but **0.541 → 0.164 on the rarer half** (learns the head, forgets the tail); inverse-frequency CE + 10× prior: 0.624 → 0.609 overall, 0.541 → 0.531 rare (no change). With this little data the shortcut is already what the data supports. Next: train on the model's own generations at scale (the GPU engine will emit routing traces by the thousand) with the 4-copy input so the probe can learn the collapse (the k=0 shortcut loses 30 % there). | — | needs own-generation traces |
| box-2 layer placement / TP over OCuLink / stale experts | C | **dead** by arithmetic or exactness | — | closed |

Expected, all levers, p=0.75, RTT ≤ 300 µs: **~55 tok/s with no misses; ~40 at 75% residency with
the plaintext cold tier; ~32 at 66%.** So "50 at zero loss" = fused chain + wavefront + acceptance
≥ 0.75 + residency ≥ 75% (≈ 30 GB more resident than two boxes hold, i.e. a third box, or a
smarter-than-LRU policy, or the model's own DSpark experts quantised to free dGPU/host bytes).

**Measured 2026-09-13, V4.1's OWN routing** (`scripts/v41_sched/v41_lru_from_dump.py` on the
1006-token agentic transcript's per-layer top-6 ids, warm LRU over global slots): misses/token at
50 / 60 / 66 / 75 / 85 % resident = **12.9 / 5.9 / 3.6 / 2.5 / 2.5**; the transcript touches
11 494 of 15 360 experts in 1006 tokens and the top-5 % of experts carry 44 % of picks. The 66 %
figure matches design B's V4-Flash-trace estimate (3.4–3.7); the ≥ 75 % plateau at 2.5 is the
compulsory first-touch rate of a 1006-token trace (a longer trace would lower it) — so with two
boxes at ~66 % residency expect **~3.5 misses/token ≈ 17 ms/token at 4.8 ms, ~11 ms at the
plaintext 3 ms**, i.e. the miss term, not the RTT, is the largest single item in the decode budget.

**Order of work.** (0) Measurements, cheapest first: RTT ping-pong the day box 2 arrives; DSpark
acceptance on ~200 tokens of real agentic traffic through the CPU oracle (`forward_spec`; ~1 day
to wire the drafter into the oracle + ~5 h CPU); LRU miss rate under V4.1's 384-expert routing once
the engine runs end to end (M6). (1) Parity milestones M2–M6 continue — every design needs the
model running. (2) fp8-native projections (quality-required, 1 wk). (3) B=k verify on the decode
graph (2–3 wk, shared prerequisite). (4) Wavefront scheduler, validated single-box (1 wk).
(5) Residency: directory + disjoint LRUs + cold tier on box 2 (1–2 wk, is M8 anyway). (6) mHC bridge
split (3 d). (7) Fusion pass to ~12 launches (1–2 wk), then Exp-0 and, if it passes, persistent
kernels (3–4 wk). (8) Drafter placement, adaptive K, send-before-route.

## 7e. Questioning the "48 tok/s single-stream roofline" (2026-09-13, user challenge)

The 48 assumed: (1) attention weights stream only from the dGPU, (2) experts stream only during
the expert phase, (3) the two phases never overlap for one stream, (4) no cross-token weight
reuse, (5) exactly one position in flight. Per-token bytes are already bandwidth-balanced across
the three memory systems (dGPU ~6.5 GB ≈ 10 ms at 640 GB/s; iGPUs 4.5 GB ≈ 10.5 ms at 2×214), so
placement cannot lower the bytes side; what it can attack is the serialisation. The true ceiling if
every memory system streamed all the time is 12 GB / 1068 GB/s ≈ 11 ms → **~89 tok/s**.

Levers that attack the assumptions, all exact:
- **(3) CED-boundary speculative pipeline.** encoder(t+1) depends only on token t+1's identity and
  encoder-KV ≤ t; decoder(t) depends only on encoder(t). Run encoder(t+1) for the drafter's top
  1–2 candidates (B=2, weights read once) concurrently with decoder(t); on a miss re-run the encoder
  for the sampled token. Token time → max(dGPU chain, expert phase) instead of their sum: today
  max(22, 15) + 15% × 11 ≈ 24 ms → **~42 tok/s** single stream with only top-candidate
  speculation; with the fused chain (13.6 ms) → max(13.6, 15) + miss ≈ 17 ms → **~58**. This is
  design B's wavefront with a 20-layer skew and no batching; the batched 3+3 wavefront is the
  general form and B's 47/55 already include the effect. (Design C's "CED gives decode nothing"
  is true of a non-speculative chain only.)
- **(4) dGPU MALL prefetch.** The 64 MB infinity cache can be warmed with ~40% of the next
  layer's attention weights during the expert wait (the dGPU is idle then); those bytes then read
  at cache speed. **Measured (`tests/bench_mall_retention.rs`): a Q8_0 matvec re-reading a
  16–64 MB weight runs at 1.1–1.6 TB/s (3.2× the cold ~500 GB/s); at 96 MB+ the ratio is 1.1×,
  i.e. the MALL holds ~64 MB.** So ~64 of a layer's ~137 MB can be warmed during the expert wait:
  attention-phase byte time 0.25 → ~0.17 ms/layer (−3 ms/token at the bandwidth floor; visible only
  once the launch floor is fused away). Cold matvec efficiency is 85–90% of 640 GB/s.
- **(1) head-split attention inside box 1.** Shard q_b/wo_a/wo_b heads 75/25 between the dGPU and
  iGPU1 over 10 µs doorbells (not the 59 µs event path): attention phase 0.24 → ~0.19 at
  bandwidth, iGPU1 streams during the attention phase too. Pays only after fusion.
- Placement changes that do NOT help: any layer/head split across the USB4 link (RTT ≥ 100 µs
  per sync); experts on the dGPU (memory); attention on the iGPUs (3.3× slower per layer);
  the CPU (shares the iGPU's LPDDR5X bandwidth).
So at zero loss: one position in flight caps at ~48 (or ~60 with MALL prefetch + fusion); above
that needs ≥ 2 positions in flight — the cheapest being the CED-boundary candidate pipeline —
and the ceiling becomes the aggregate-bandwidth 89.

## 7f. DSpark acceptance — MEASURED (2026-09-13)

Teacher-forced through DeepSeek's unmodified drafter (`scripts/v41_oracle/dspark_accept.py`),
greedy drafts vs the main model's greedy token, drafter window seeded with the prefix:

| text | steps | pos-1 | pos-2 | pos-3 | E[tokens/step] K=1 / 2 / 3 | main in drafter top-5 |
|---|---|---|---|---|---|---|
| synthetic agentic transcript (mine) | 450 | 0.25 | 0.10 | 0.08 | 1.25 / 1.34 / 1.38 | 0.44 |
| **model's own continuation** (`oracle_generate.py`, 96 tokens of a tool-call turn) | 89 | **0.44** | 0.26 | 0.14 | **1.44 / 1.67 / 1.78** | 0.66 |

Confidence-head AUC rises from ~0.5 at position 1 to ~0.84 at positions 4–5 (it knows when the
tail is hopeless, not whether the first draft is right). Caveats: 89 steps (±5 % on pos-1); one
code-heavy tool-call turn (paths, grep patterns) — prose may accept higher; greedy match, not
rejection sampling; harness had two bugs fixed (target-layer INPUT, f32 expert cache) and passes
the structured-token sanity check (DSML markup predicted exactly).

**Consequences for §7a.1 / §7d (p = 0.44, not 0.75):** batched verify K=2 yields 1.67 tokens per
~63 ms pass ≈ **26 tok/s — no gain over the single stream**; design B's wavefront interpolates
to ~28–30 (its table: 34 @ 0.60, 47 @ 0.75). The **CED-boundary candidate pipeline** needs only
position-1: ~44 % of steps overlap with one candidate, ~55 % with two (top-5 containment 0.66)
→ 26 → **~30–31 tok/s**, and it is the cheapest of the speculative designs. So at zero loss on
this hardware: **~30 single-stream with the pipeline, ~35 with fusion on top; 50 needs a better
drafter** (fine-tune DSpark on the user's traffic — its experts are the free-approximation zone).

## 7b. Two-box scheduling principle (user, 2026-09-13): maximise busy time on every track, keep the pipeline solid, use all the hardware

Rules for the M8 decode schedule, in the order they matter:

1. **Three expert branches per layer, concurrent, balanced by latency.** After the router
   (dGPU): box-1 iGPU picks ∥ box-2 picks over the link ∥ dGPU shared + hot picks. The branch
   lengths must match so nobody waits: the remote branch carries RTT, so it gets *fewer*
   per-token picks than the local one — i.e. the hottest experts live on box 1 and the dGPU, the
   tail lives on box 2 (it has the memory) and behind it the SSD. Placement is chosen for
   per-layer balance, not for residency alone.
2. **The link is never idle-then-bursty.** The activation for box 2 leaves the moment the router
   finishes (before local experts start); box 2's reply lands while the dGPU is still doing
   shared/hot work; persistent connection, pinned 40 KB buffers, no per-layer setup. Prefill
   sends the next super-chunk's activations while the current one computes.
3. **Misses are branches too.** A cold expert's pread (measured 4.8 ms per 18.8 MB at 64
   threads, straight into the device slot) is issued at router time and overlaps the other
   branches. (Correction from design C: a verify batch's routing at layer l is NOT known early —
   it needs the verify pass to reach layer l; only *predicted* routing, design B's axis, can
   prefetch ahead of the router.)
4. **One token in flight cannot light every track** — the per-layer chain (attention → router →
   experts → post) is serial, so with a single stream the dGPU idles during the expert phase
   (~35% of a layer) and both iGPUs idle during attention (~50%). The only fillers are more
   tokens in flight: DSpark verification batches (2–3 tokens/layer) and, for the server, a
   second concurrent request. Speculation is therefore also the pipelining mechanism, not only
   an acceptance win — design the verify pass so token t+1's attention overlaps token t's
   expert branches.
5. **The metric is the trace.** Per-device + link busy fraction from `analyze_pftrace_gaps.py`
   is the M8 acceptance number; a schedule change that raises tok/s but lowers busy fraction is
   leaving something on the table.

## 7. Headline goals (rev 3 — recomputed after the §3.3 caching measurement)

Decode budget, all-native + LRU residency, 32K (ms/token):
experts 240 picks x 18.8 MB / 2 boxes x 1.2 imbalance / ~230 GB/s = **11.8** · attn+mHC+gate+head
6.7 GB FP8 / 640 GB/s = **10.5** · network 40 RTT = **4.0** · glue **3.0** · SSD 1.4 misses x 2.7 ms,
~half prefetched = **1.9** → **31.2 ms → 32 tok/s**.

| configuration | quality loss | tg 32K/100K/1M | tg + DSpark | pp 32K/100K/1M |
|---|---|---|---|---|
| **all-native, LRU residency** | **0** | **32 / 31 / 29** | 44 / 43 / 40 | 2000 / 1800 / 1200 |
| uniform rotated-LDLQ IQ3_S | ~2-3% | 37 / 36 / 33 | 51 / 50 / 46 | same |
| uniform rotated-LDLQ IQ3_XXS | ~3-5% | 39 / 38 / 35 | 53 / 52 / 48 | same |
| all-native on a 3rd box | 0 | 39 / 38 / 35 | 53 / 52 / 48 | slightly higher |
| all-native, static placement (superseded) | 0 | ~13 | — | same |
| V4-Flash today, for reference | — | 28 / 27.5 @4K/96K | — | 735 / 683 |

pp is unchanged across quantization choices: under super-chunk layer-major + expert-major GEMMs
each expert's weights are read once per super-chunk (weight traffic stops being the limit), and
MXFP4 dequant is cheaper than the IQ3 codebook path. Cold experts cost one 32 GiB pass per
64K super-chunk vs ~32 s of compute → hidden. Add ~50% to pp if dGPU expert streaming (§5.3) works.
tg is nearly flat in context: attention per token is constant (top-512 + 128 window); only the
4 full-range indexers grow, ~20 GFLOP FP4 on the dGPU at 1M (<2-3 ms).

**What changed:** the price of zero quality loss fell from 24 tok/s to ~5. The 3rd box now buys
+22% decode rather than being the only route to a working system. **DSpark (~+12 tok/s) is worth
more than the entire quantization spread (~7)** — and is the least certain number here (acceptance
unmeasured; our V4-Flash MTP history was poor, but that cap was quant infidelity, which all-native
removes).

**Sensitivity — the §3.3 result is from 2 monologue conversations; agentic traffic with
interleaved tool calls is untested and is what the user actually runs:**

| LRU miss rate vs measured | SSD reads/token | tg all-native |
|---|---|---|
| as measured | 1.4 | 32 |
| 3x worse | 4.2 | 27 |
| 10x worse | 14 | 20 |

Even 10x worse lands at the pre-measurement static estimate, so the downside is bounded.

## 8. Workstreams (vague, roughly ordered)

**Lessons from the 2026-09-12 V4-Flash compressor work that carry to V4.1 kernels**
(`docs/v41/PREFILL_BALANCE_2026-09-12.md`, memory `compressor-gather-2026-09-12`):
- **The V4.1 compressor is MUCH simpler than V4's** — `compress_ratios` are 0/2/1, not 4/128,
  and CSA2 drops the overlapping window and the absolute-position embedding (ARCH_SPEC §1.3). So
  the ring-buffer + shuffle machinery that cost 28% of V4-Flash's binding device mostly does not
  exist here: ratio 1 is a plain projection, ratio 2 pools 2 rows. **Do not port V4's compressor
  state machine.**
- **Do NOT write the projection as `grid.z = batch`.** V4-Flash's `matvec_pair_batched` re-read
  the whole weight matrix per batch row (8.6 GB/layer/chunk) and survived only because the weight
  fit in Infinity Cache. V4.1's compressor projection is [B,5120]×[5120,512] and would hit exactly
  the same trap. Tile the batch (TILE_B=8 was worth +9.8% e2e, bit-exact).
- **Do not reach for WMMA on a bandwidth-bound stage.** WMMA on RDNA3/4 takes only 16-bit inputs,
  so "matrix cores" means "cast activations to f16" — which broke the 5e-2 oracle bar here because
  the compressor gate feeds a softmax. Fixing the access pattern won more, at zero precision cost.
- **Prefill is dGPU-bound (85.9% vs iGPU 62.3%)**, which is why §5.3 is marked unverified.

**Before the box arrives (no HW dependency):**
0. **Dynamic-residency prototype — PHASE A DONE 2026-09-12** (`tests/bench_expert_miss_cost.rs`,
   addendum in COLD_EXPERT_CACHING.md): a cold 18.8 MB miss costs **5.1 ms with a 16-thread reader,
   12-18 ms with a naive pread** — NOT the 2.7 ms assumed. **The miss path MUST be parallel/async;
   that is now a requirement, not an optimisation.** The ~2.6 ms H2D copy is removable by mapping
   page-cache pages into GTT (phase B). All-native lands ~30 tok/s, not 32. Conclusion survives.
   **PHASE B (still open):** §3.3's 11-48x LRU result is a
   trace simulation that has never run in an engine, and the whole all-native plan rests on it.
   Cap resident experts below what fits, page the rest from mmap, measure real decode. Answers:
   (a) can the OS page cache serve GTT-mapped expert reads at all? (b) does 4 KiB granularity vs
   18.8 MB experts need MADV_WILLNEED? (c) does the simulated miss rate survive a real allocator?
   **Needs no V4.1 weights and no second box.** If it fails, all-native dies and the 3rd box returns.
0a. **Fast model load via direct-to-device pread (spun out of phase B, 2026-09-12).** The current
   path does THREE passes per tensor: `Vec::with_capacity` + `resize(n, 0)` (**zeroes every byte,
   then the read immediately overwrites them**), single-threaded `pread`, then a staged
   pageable->device copy. Measured end-to-end: **~690 MB/s** (61 GiB resident in 95 s). The phase-B
   technique — allocate the DeviceBuffer first, then multi-threaded `pread` straight into it —
   measured **3.90 GB/s**, i.e. ~5.6x, taking a 95 s load under 20 s.
   **This matters far more for V4.1 than for V4-Flash: 475 GiB native at 690 MB/s is ~11.5 min per
   load; at 3.9 GB/s it is ~2 min.** It also deletes a host staging buffer on a box where host RAM
   is the binding constraint. Caveats: tensors with a conversion role (ToF16, Q8_0 repack) must
   still go through host memory (a minority of bytes — experts dominate and are passthrough);
   parallelising ACROSS the 1328 tensors is simpler than chunking within one; and CPU-addressability
   of device allocations was probed on ONE buffer — verify it holds at scale under memory pressure.

0b. **Leave `DEEPSTRIX_EXPERT_TRACE` on during real agentic use** — §3.3 was measured on two long
   monologues; tool-call interleaving is untested and is the workload that matters.
0c. ~~Quantiser + calibration pipeline (§3.2)~~ **DEPRIORITISED** — if all-native holds we may not
   need a quantiser at all. Do not start it until 0 reports back.
1. Download weights (475 GiB; 1.1 TB free here) — in progress, ~11 MB/s, ETA ~2026-09-12 morning.
1b. ~~Cold-expert prefetch feasibility~~ **DONE 2026-09-11** (§3.3, docs/v41/COLD_EXPERT_CACHING.md):
   LRU residency beats static placement 11-48×; zero-loss config is viable at ~33 tok/s.
   **Follow-up still open: re-run the trace on real agentic traffic (tool calls interleaved),
   and test whether the OS page cache can serve GTT-mapped expert reads.** Read transformers 5.6 + vLLM main model code
   for exact Engram hashing/gating, DSpark heads, CSA2 mode bookkeeping, sqrtsoftplus router.
2. **Layer-streaming CPU oracle — STARTED 2026-09-12, `scripts/v41_oracle/`.** Runs DeepSeek's
   UNMODIFIED `inference/model.py` one Block at a time (CPU kernel shim + LazyMoE/LazyEngram; ~1 GiB
   RAM, ~4 s/layer at T=6). **VALIDATED: 40 layers → " Paris" top-1 (+2.5 logits), 0 NaN, bit-deterministic across runs;
   `compare.py` scores dumps with the standing oracles' Δ/scale metric.** Design +
   gotchas in memory `v41-oracle`. This replaces ds4 as the oracle.
3. ~~Converter safetensors → deepstrix blobs~~ **REPLACED 2026-09-12 by HF-direct loading**
   (`V41HfWeights`, see top): no second on-disk copy, both boxes just rsync the HF snapshot.
   Native experts come out as MXFP4 in ggml blocks at load; fp8 projections as Q8_0 (later:
   native fp8 WMMA on the dGPU, gfx1201 has it). The requant experiment stays a separate
   producer: **layer-major calibrate+quantise pipeline (§3.2)**, writing a llama.cpp-named GGUF
   that the old `MappedGguf` path can read unchanged: Hessian accumulation → LDLQ into
   IQ3_S/IQ3_XXS (ggml-quants FFI as the baseline encoder to beat), optional Hadamard rotation
   per expert; per-expert format from the measured error (+ floor). The same pass emits the
   per-expert per-format error table and Σ-gate-weight stats.
4. Kernels (single-box, subset-of-layers testable): [MXFP4 kwide matvec/GEMM (LUT + e8m0 shift)
   only if native promotion wins in §3]; FP4 main-KV attention (extend the E2M1 indexer-key code); CSA2
   compressor (no overlap / no abs-pos); indexer K from main KV; hierarchical candidate pool;
   grouped low-rank W_O; single-pass mHC (A_{l-1}); sqrtsoftplus + clamped SwiGLU; Engram gather
   + gate; new ViT (later).

**When the box arrives:**
5. NixOS from the same flake, ROCm 7.2.3. Measure USB4 host-to-host RTT/throughput —
   **plan: `docs/v41/SECOND_BOX.md`** (two `nixosConfigurations` sharing modules, `thunderbolt-net`
   recipe, day-one measurements incl. `scripts/netbench/rtt_pingpong.py`; the RTT decides
   serialised vs overlapped remote-expert scheduling). Box 2 arrives 2026-09-13.
6. Expert-server daemon + hub client; het-split gains "remote" and "SSD-cold" tiers.
7. CED prefill scheduling + decoder bounded replay (keep an *exact* all-40-layer mode for oracle
   parity — the reference is exact, production bounded replay is approximate, ARCH_SPEC §4); super-chunk layer-major loop; expert-major
   GEMM path; cold prefetcher; dGPU expert streaming ring.
8. Decode graph for 40 layers + RTT hops; DSpark drafter + verify at K=1-2; scheduler.
9. Snapshot cache / system-prefix cache adapted (KV is tiny now — snapshots become cheap).
10. Vision tower.

---

## 9. Risks / open questions
- MXFP4→IQ3_XXS quality (unmeasured; **prior is now poor**: codes are 3.9 bits/weight of
  information, see §3). If bad: native + SSD-cold tier at ~20 tok/s, or a 3rd box.
- USB4 host-to-host RTT on Linux (thunderbolt-net). If >200 µs, decode loses ~8 ms → fallback §4.
- Hub box is only 96 GB; residual buffer competes with experts (S=32K → 1.3 GiB is the compromise).
- Oracle: no independent CPU reference beyond our own layer-streamer; cross-check vLLM vs
  transformers where they differ.
- Engram random-read IOPS at 1M-token prefill (~47K IOPS at 1000 tok/s — fine for a good NVMe,
  but page-cache thrash on a full 96 GB box is untested).
- DSpark acceptance on our quant mix unknown until measured.
- V4-Flash production must come down for any two-box V4.1 run; kernel/oracle work stays on the
  new box in isolation until then.

### 7f.1 Rejection-sampling acceptance (2026-09-13, `dspark_accept.py` on the regenerated tool-call turn, 89 steps)

The deployment metric (sampling at T=1): acceptance of draft 1 = Σ_x min(p_main(x), q_draft(x)),
computed from the main model's full logits (head recomputed from the dumped head input) and the
drafter's logits incl. the markov bias.

| metric | value |
|---|---|
| greedy agreement, draft 1 | 0.438 |
| **rejection-sampling acceptance, draft 1** | **0.376** |
| … at positions where main entropy < 1 nat (75/89) | 0.428 |
| … at positions where main entropy ≥ 1 nat (14/89) | 0.100 |
| mean main entropy | 0.57 nats |
| expected accepted tokens/step, K=1/2/3 (greedy chain) | 1.44 / 1.67 / 1.78 |

Rejection sampling does NOT rescue the number here: the main model is peaked on this text
(84% of positions below 1 nat), so the overlap collapses to the drafter's own probability on
the main's top token, and the drafter puts only ~0.4 there (0.1 where the main is uncertain).
Harness re-audited against the reference (target hidden = attention input of layers 37/38/39,
seeding/stepping = the reference's own loop, DSpark router 128/top-3 through the same
`get_moe_config`, markov bias in place, no YaRN on SWA-only DSpark blocks): no discrepancy
found. Drafts 2–5 are one parallel pass with noise-token inputs (no autoregression), so the
steep fall-off is by design. Pending: the same measurement on plain prose (running) to separate
"weak drafter" from "hard text". Consequence for §7c–§7e stands: batched K=2 ≈ no gain,
wavefront ~28–30, CED-boundary pipeline ~30–31; 50 tok/s needs a better drafter.
