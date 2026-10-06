# deepstrix docs — start here

Index written 2026-10-04 with the docs-vs-code audit. Most files in `docs/` are
dated journals; this page says which ones describe the system as it is.

## What the system is (2026-10-04)

- **Model:** DeepSeek-V4.1-Flash (40 layers, 384 routed experts, top-6, CSA2
  compressed attention with the sparse indexer, Engram, mHC, the DSpark
  drafter). Spec: `v41/ARCH_SPEC.md`. Vocabulary: `GLOSSARY.md` (canonical).
- **Two boxes.** Box 1 = the hub (`deepstrix-server`, 9070 XT dGPU + Strix Halo
  iGPU). Box 2 = `deepstrix-expertd`, serving routed experts over USB4 /
  `thunderbolt0`. Routed experts are native MXFP4, paged through LRU pools on
  both boxes from NVMe; box 1 owns each layer's hottest experts (T2 partition +
  hot set), box 2 the rest.
- **Serving:** the multistream arena (8 slots, prefill/decode bursts, one or two
  lanes per step), arena DSpark (speculative decoding while at most
  `V41_MS_DSPARK_STREAMS` streams are live; two since 2026-10-06), arena stage
  graphs keyed by (stage, rows), snapshot / prefix restore, vision
  (`v41/VISION_PORT.md`), OpenAI-compatible API, and `/v1/embeddings` through the
  embed phase (Qwen3-Embedding-4B on the dGPU, `--embed-gguf`).
- **Dropped:** V4-Flash / GGUF (2026-09-24). Its code is still in the tree until
  the removal branch merges; its docs are history.
- **Laguna** support lives beside it (`laguna/`), CLI only.

## Living references — keep these in sync with the code

| doc | what |
|---|---|
| `v41/ARCH_SPEC.md` | model constants, tensor names, engine presentation |
| `GLOSSARY.md` | canonical names (owner sign-off 2026-10-04); `scripts/ontology_census.py` ratchets it |
| `v41/TUNING.md` | knobs and host settings; §0 = production vs code default |
| `v41/KNOWN_BUGS.md` | correctness ledger; its status index lists what is open |
| `v41/REMOTE_EXPERTS.md` | box-2 daemon and wire protocol (cited from the code) |
| `crates/v4flash-kernels/src/knobs.rs` (header) | how a knob is resolved: default < env < legacy file < `V41_KNOBS_FILE`; live vs static |
| `v41/KNOB_AUDIT_2026-10-04.md` | dated inventory of env reads still outside `knobs.rs` |

**Production configuration is not in the repo.** The hub is launched by
`~/run_v41_server.sh` plus env files under `~/scratch-ms/`, and live knobs come
from `/dev/shm/deepstrix-knobs.txt` (box 2: `~/expertd-knobs.txt`). Where a doc
says "default", it should mean the code default; `v41/TUNING.md` §0 has the
production values as of 2026-10-04.

## Designs and plans in flight

| doc | state (2026-10-04) |
|---|---|
| `v41/MS_DSPARK_STREAMS_DESIGN.md` | built; two speculating streams live since 2026-10-06, default 2 (b5def35) |
| `v41/GRAPH_KEYS_DESIGN.md` | built; `V41_MS_GRAPH_KEYS=stage_b` live since 2026-10-05, default since b5def35 |
| `v41/EMBED_PHASE_DESIGN.md` | built; embed phase deployed 2026-10-05 |
| `v41/DSPARK_ARENA_PLAN.md` | built: the live DSpark path |
| `v41/MULTISTREAM_DECODE_PLAN.md` | M1 built; later milestones open or replaced |
| `v41/KV_PREFIX_STORE_DESIGN.md` | approved design; M1 code on branch `worktree-kv-prefix-store` |
| `v41/EVTRACE_REBUILD_PLAN.md` | P0, P1, P3 built |
| `ONTOLOGY_REVIEW_2026-10-04.md` | glossary adopted; renames pending |
| `v41/HOT_SPLIT_SIM.md` | simulator + its recommendations (`scripts/split_sim`) |

## History

Everything else is a dated record. Docs overtaken by later work carry a
`Status (docs audit 2026-10-04)` banner under the title saying what changed and
where the truth now lives. Rough groups:

- **V4.1 engine bring-up (09-11..09-18):** `v41/PLAN.md`, `v41/ENGINE_PORT.md`,
  `v41/STATE_2026-09-13.md`, `v41/REVIEW_2026-09-13.md`, `v41/M0_AUDIT.md`,
  `v41/M7_EXPERT_TIER.md`, `v41/INDEXER_PORT_PLAN.md`,
  `v41/INDEXER_DENSE_VS_SPARSE.md`, `v41/ROADMAP_2026-09-18.md`, `v41/SECOND_BOX.md`
  (+ `v41/second_box_flake.patch`, superseded by the lumi flake).
- **Decode, paging and placement investigations:** `v41/DECODE_*`,
  `v41/decode_designs/`, `v41/MULTISTREAM_M1A_*`, `v41/COLD_EXPERT_CACHING.md`,
  `v41/EXPERT_ALLOCATION_REWORK.md`, `v41/ALLOCATION_STRATEGY.md`,
  `v41/GLOBAL_POOL_FEASIBILITY.md`, `v41/BOX2_MISS_SUBSTITUTION.md`,
  `v41/LINK_IDLE_LATENCY.md`, `v41/PREFETCH_STUDY_2026-09-18.md`,
  `v41/ROUTER_LOOKAHEAD_AND_PREFETCH.md`, `v41/STATIC_PLACEMENT_DOES_NOT_GENERALIZE.md`,
  `v41/WHY_THE_BIG_POOL_REGRESSED.md`, `v41/SLOT_RESPLIT_MEASURED.md`,
  `v41/MIXED_PRECISION_BY_RESIDENCY.md`, `v41/CLAIM_CAP_ZERO_RETRACTED.md`,
  `v41/MATCHING_THE_SINGLE_BOX_DECODE.md`, `v41/EXPERT_FORMAT_IS_THE_REMAINING_LEVER.md`,
  `v41/ROUTED_MOE_IS_NOT_A_KERNEL.md`.
- **DSpark before the arena (legacy prefill-shaped verify, 09-13..09-16):**
  `v41/DSPARK_DESIGN.md`, `v41/DSPARK_BUILD_PLAN.md`, `v41/DSPARK_STATE_2026-09-15.md`,
  `v41/DSPARK_VERIFY_*`, `v41/DSPARK_WHAT_IS_LEFT.md`, `v41/DSPARK_DRAFTER_GAP.md`,
  `v41/DSPARK_ACCEPTANCE_INVESTIGATION.md`, `v41/DSPARK_SINGLE_STREAM_PERF.md`,
  `v41/VERIFY_*`, `v41/WHY_THE_VERIFY_SPLIT_CHANGES_FAILED.md`.
- **Audits and reviews:** `ARCHITECTURE_REVIEW_2026-09-24.md`,
  `v41/AUDIT_2026-09-22.md`, `v41/PROFILING_AUDIT_2026-09-21.md`,
  `v41/KERNEL_PERF_REVIEW.md`, `v41/OVERNIGHT_2026-09-22.md`,
  `v41/PREFILL_100K_PROFILE.md`, `v41/PREFILL_BALANCE_2026-09-12.md`,
  `TOOL_PROMPT_FIDELITY.md`.
- **V4-Flash / GGUF era (05..09-10):** `DESIGN.md` (the original design),
  `PHASE0.md`, `PHASE1_*`, `PHASE2_KERNEL_VALIDATION.md`, `M40_*`, `M5x_*`,
  `FP8_KV_IMPL_2026-09.md`, `E2M1_INDEXER_KEYS_2026-09.md`, `IQ2_*`, `IQ3_S_*`,
  `IGPU_VRAM_POOL_PLAN.md`, `VRAM_FREE_PLAN_2026-09.md`,
  `EJPIR_DS4HIP_PREFILL_ANALYSIS.md`, `UNSLOTH_UD_IQ2XXS.md`, `VISION_STATUS.md`.
- **Laguna:** `laguna/` (`laguna/PLAN.md` §0 has the outcome).
- `CREDITS.md`.

## Conventions

- Dates in docs are UTC (`YYYY-MM-DD`, `HH:MM UTC`).
- "Default" = the code default. A launch-script value is "production sets X".
- When a measurement or figure is retracted, put a banner on every doc that
  stated it, not only on the new doc that retracts it.
- A plan's `Status:` line says built / partly built (which parts) / abandoned /
  superseded by X. Update it when the code lands, not later.
- Link Claude memory notes sparingly: they are outside the repo and a reader
  without them cannot follow the reference.
