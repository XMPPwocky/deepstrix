# IQ3_S gate/up kernel plan (unsloth UD-IQ3_XXS blk.26)

Status: PLAN, 2026-09-02. Read-only survey; nothing implemented.

## 0. Premise check (from the real shard headers)

Range-fetched the four UD-IQ3_XXS shard headers from HF
(`unsloth/DeepSeek-V4-Flash-0731-GGUF/UD-IQ3_XXS/`, 97.05 GiB total, routed
experts 90.17 GiB). Actual mix — **not** "iq2_s everywhere":

| role | types |
|---|---|
| `ffn_gate/up_exps` | IQ2_XS ×25, IQ3_XXS ×17, **IQ3_S ×1 (blk.26)** |
| `ffn_down_exps` | IQ3_XXS ×41, MXFP4 ×2 (blk.26, 42) |
| `output.weight`, `token_embd.weight` | **Q6_K** (contract today: Q8_0/Q4_K; F16/Q4_K/Q5_K) |
| everything else | already inside the contract |

So four contract violations block loading: blk.26 gate/up IQ3_S (this plan),
plus Q6_K head + Q6_K embedding (small, §3.9). IQ3_S never appears at
`down` in this mix — no down kernel needed.

## 1. Format spec (verbatim, llama.cpp master `ggml/src/ggml-common.h`)

```c
// 3.4375 bpw
#define IQ3S_N_SCALE QK_K/64
typedef struct {
    ggml_half d;
    uint8_t qs[QK_K/4];
    uint8_t qh[QK_K/32];
    uint8_t signs[QK_K/8];
    uint8_t scales[IQ3S_N_SCALE];
} block_iq3_s;
static_assert(sizeof(block_iq3_s) == sizeof(ggml_half) + 13*(QK_K/32) + IQ3S_N_SCALE, "wrong iq3_s block size/padding");
```

Byte offsets in the 110-byte block: `d`@0 (f16), `qs`@2 (64), `qh`@66 (8),
`signs`@74 (32), `scales`@106 (4). `2+64+8+32+4 = 110` ✓ matches
`GgufType::IQ3_S => (256,110)` (gguf.rs:574) and ds4 `quants.c:59`.
Grid: `GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)` — 4 magnitudes per u32,
odd values 0x01..0x0f. Sign mask: `kmask_iq2xs = {1,2,4,8,16,32,64,128}`.

Dequant loop, `ggml-quants.c:2607` (verbatim):

```c
void dequantize_row_iq3_s(const block_iq3_s * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;
    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}
```

Integer-domain dot (`ggml-cpu/quants.c:1094`, `ggml_vec_dot_iq3_s_q8_K_generic`):
`s = Σ_blk f16(d)·y.d · Σ_ib32 ls(ib32)·sumi(ib32)` with
`ls = 2*nibble + 1`, nibble = `scales[ib32/2]` low (even ib32) / high (odd).
**No fractional prefactor** (iq2_s: 0.125, iq3_xxs: 0.25). Per ib32: 8 qs
bytes, 1 qh byte (bit `2l` → grid1 of subgroup l, bit `2l+1` → grid2),
4 raw sign bytes, half a scale byte.

Diff vs the two templates:

| | IQ2_S (82 B) | IQ3_XXS (98 B) | **IQ3_S (110 B)** |
|---|---|---|---|
| grid | 1024 × u64 (8 w/entry), 8 KiB | 256 × u32 (4 w), 1 KiB | 512 × u32 (4 w), 2 KiB |
| index | 8 b + 2 qh bits | 8 b | 8 b + 1 qh bit |
| signs | raw bytes `qs[32..64)` | 7-bit ksigns in aux32 | raw bytes `signs[32]` |
| scale | nibble / 16 w | nibble / 32 w (aux32>>28) | nibble / 64 w |
| prefactor | 0.125 | 0.25 | 1 |
| bpw | 2.5625 | 3.0625 | 3.4375 |

## 2. Existing kernel structure (what gets twinned)

All three families (`iq2_s_pair_matvec.hip` 389 lines,
`iq2_xs_pair_matvec.hip`, `iq3_xxs_pair_matvec.hip` 657 lines) share one
skeleton per variant; only the per-block helper and the LDS tables differ:

- **Decode `_fused_swiglu_batch`** / **`_batch_hetsplit`** (iq3_xxs_pair:189/257):
  grid `(n_rows/8, n_used)`, 256 threads, warp = row, lane pair splits a
  super-block into ib32 halves (`dot_block_half_*`), `s_xq[16*292]` staged
  once, tables staged per WG (`IQ*_STAGE_TABLES`; divergent `__constant__`
  reads were 3.7× slower). hetsplit adds the M63 remap decode
  (`e = mode==0 ? -dense-1 : dense`).
- **Prefill `_chunked`** (:351): same dot, serial member loop, re-dequants
  per member — the 3× trap fixed by ae9bd61; kept only as `PAIR_VARIANT=chunked`.
- **Prefill `_kwide`** (:463): warp = row, lanes split an sb PAIR
  (`sbh=lane>>4`, `l15=lane&15`, `sub_block=l15>>1`, `half=l15&1`); each lane
  dequants its 16 gate + 16 up weights ONCE into 8 dwords with the scale
  folded, members' q8 cooperatively staged (`s_q8v[32][2][16] uint4` = 16 KiB,
  member-major), one `ds_load_b128` → 8 dot4 per member, 2×32 f32
  accumulators, fused reduce+swiglu. Wrapper asserts `n_blocks%2==0`,
  `chunk<=32`, `n_rows%8==0`.

Rust wrappers (`src/iq2_s.rs`, `src/iq3_xxs_pair.rs`): `include_bytes!(env!("KERNEL_<STEM>_<ARCH>"))`,
`Module::load_data`, `get_function`, `launch_kernel!` with grid
`(n_rows/8, n_used|n_work_items)`. `build.rs` compiles every `kernels/*.hip`
for gfx1201+gfx1151 and derives the env name from the file stem — no list to
edit. Engine fields `e.iq2s`/`e.iq3pair` at `het/engine.rs:110-113`, built
at :197-199.

## 3. Touch points (file:line)

1. `crates/v4flash-core/src/gguf.rs:461,501,537,574,610` — IQ3_S already
   complete (id 21, (256,110), name). `tests/block_shape_vs_ds4.rs:74` pins it. **No change.**
2. `crates/v4flash-kernels/src/weight_contract.rs:76-78` — add `IQ3_S` to the
   gate/up `Quant` list; update comments :57-60, :72-75; add a
   `bytes_per_expert(IQ3_S, 4096, 2048) == 2048*16*110 = 3_604_480` assert in
   the tests mod (:313-332). `bytes_per_expert` (:134) is generic.
3. `crates/v4flash-kernels/src/het/dispatch.rs` — new arms: `:44-56`
   (`moe_gate_up_batch`), `:83-101` (hetsplit), `:270-292` (`moe_gate_up_chunked`:
   `IQ3_S if use_kwide => e.iq3s.launch_fused_swiglu_kwide`, else `_chunked`).
   `moe_down_*` (:121-168) untouched.
4. `crates/v4flash-kernels/src/het/engine.rs:110-113` field
   `pub iq3s: crate::iq3_s::Iq3SPairMatvec`; `:197-199` constructor.
5. `crates/v4flash-kernels/src/lib.rs:36-38` — `pub mod iq3_s; pub mod iq3_s_tables;`.
6. New: `kernels/iq3_s_pair_matvec.hip`, `kernels/iq3_s_tables.inc`
   (`__constant__ uint32_t iq3s_grid[512]`), `src/iq3_s.rs`, `src/iq3_s_tables.rs`.
7. `het/weights.rs` — **no change**: strides come from dtype (:438-443, :510,
   :576) and are cross-checked against buffer size (:445-458); hot-expert
   packing (:566-598), M63 dedup/remap (:617-740, :896-918) are dtype-agnostic.
   The hard error the user hits today is `validate_model` at :848.
8. `het/forward_prefill.rs` — **no change**: both prefill sites already go
   through the dispatcher (:2394 dGPU hot-expert path, :2692 iGPU cold path);
   `CHUNK_SIZE=32` (:2505) ≤ `KW_MAX_CHUNK`; the `hybrid` guard (:2567)
   already rejects non-IQ2_XXS layers. `het/forward_layer.rs:191,1375,1451,1464`
   (decode) likewise dispatch-routed. `batch_scratch.rs`/`state.rs` have no
   per-type sizing (grep clean). The staged/tile8/hybrid zoo is IQ2_XXS-only
   by construction (`moe_gate_up_chunked` returns `Ok(false)` only for IQ2_XXS).
9. **Q6_K extras** (not iq3_s, but needed to load): `weight_contract.rs:64`
   add Q6_K to `output.weight` — decode already falls to `dispatch::dense_matvec`
   → `e.q6d.matvec` (`forward_head.rs:107`, `dispatch.rs:195`); verify the
   prefill last-token head uses `dense_gemm` (Q6_K supported, `dense_gemm.rs:83`).
   `weight_contract.rs:108` add Q6_K to `TOKEN_EMBD_ALLOWED`; `src/embed.rs:79-121`
   add a `Q6_K` arm with `dequant_q6k_superblock` (port of `dequantize_row_q6_K`,
   210 B = ql[128] qh[64] scales[16] d; ~30 lines); `deepstrix-server/src/embed.rs:12`
   and `engine_worker.rs:337` pass the dtype through unchanged.
10. Tests: new `tests/iq3_s_pair_oracle.rs`, `tests/iq3_s_cpu_ref.rs` +
    `tests/ref/iq3_s_gen.c`; `tests/weight_contract_models.rs:18-20` add the
    UD-IQ3_XXS path when on disk; `tests/bench_iq2_xs_isolated.rs:60-133`
    add `BENCH_FMT=iq3s`.
11. ds4 reference: `external/ds4/ds4.c` has **no iq3_s** (only the type-traits
    row at :1202). A Track-R style patch 0012 mirrors 0010: struct+assert (:159-185),
    grid table (:329), `ds4_dequant_row_iq3_s`/`ds4_vec_dot_iq3_s_q8_K` (:2351-2496),
    `ds4_ref_*` cases (:2533-2547), `expert_pair_dot_iq3_s` + `case 21` (:2596-2601),
    routed-type predicate (:2865-2869), `routed_expert_block_bytes` (:2875-2880),
    `DS4_TENSOR_IQ3_S = 21` (:1225).
12. Docs: `docs/UNSLOTH_UD_IQ2XXS.md` new "UD-IQ3_XXS" section after :143;
    memory note update.

## 4. Kernel design

**File `iq3_s_pair_matvec.hip`** = the `iq2_s_pair_matvec.hip` skeleton
(decode batch, hetsplit, chunked) + the `iq3_xxs_pair` kwide, with:

- `IQ3S_STAGE_TABLES(s_sign_pair256, s_grid3s)`: iq2_s's sign-pair builder
  verbatim (raw-byte domain, 256 × u64 computed from `tid`, 2 KiB) +
  `s_grid3s[512]` u32 from `iq3s_grid` (2 KiB, 2 staging iterations of 256
  threads). LDS decode: 2+2+4.7 = 8.7 KiB/WG (iq2_s 14.7).
- `dot_block_half_iq3s(w, y, ib32_lo, ...)` from `dot_block_half_iq2s`
  (iq2_s:88-134): per ib32 `qs=w+2+8*ib32`, `qh_b=w[66+ib32]`,
  `sg=w+74+4*ib32`, `ls = 1+2*((w[106+(ib32>>1)] >> (4*(ib32&1))) & 0xf)`;
  per l: `g1 = qs[2l] | ((qh_b<<(8-2l))&256)`, `g2 = qs[2l+1] | ((qh_b<<(7-2l))&256)`,
  `sp = s_sign_pair256[sg[l]]`, two `sudot4(cond_neg_bytes_fast(grid, sp_lo/hi), q8)`.
  `cond_neg_bytes_fast` (xor+carry trick) is safe: magnitudes ≤ 0x0f, no
  byte carry. Return `d*yd*bsum` — drop the `0.125f`. int32 headroom:
  32·15·127·31·8 ≈ 15M per block.
- **kwide** from `iq3_xxs_pair:463-657`, only the per-lane unpack changes:
  lane owns ib32 `sub_block`, subgroups `l=2*half, 2*half+1` (16 weights = 4
  grid u32). Loads per matrix: `q3 = load_u32_2aligned_par(blk + 2 + 8*sb + 4*half)`
  (offset ≡ 2 mod 4 — **must** stay the 2-aligned u16-pair load, as iq3_xxs
  does for its 98-byte blocks); `hi = (blk[66+sb] >> (4*half)) & 0xf` (bit i → grid
  byte i of `q3`); `sg = load_u16(blk + 74 + 4*sb + 2*half)` → two
  `s_sign_pair256` lookups; `ls` nibble from `blk[106+(sb>>1)]`. ls covers 64
  weights ⊇ the lane's 16, so folding `gds = f16(d)*ls` per lane is exact
  (same argument as iq3_xxs's per-ib32 fold). Keep `IQ3S_KW_MAX_CHUNK=32`,
  `UNROLL 4`, no `__launch_bounds__` games. LDS 2+2+16+0.25+0.125 = 20.4 KiB
  (iq3_xxs 18.4, iq2_xs 21.4) → 3 WGs/CU by LDS = 24 waves, above the
  VGPR-limited 12; occupancy limiter stays VGPR, as for the siblings.
- Sharing: keep the tree's one-self-contained-`.hip`-per-format convention;
  don't refactor four shipped kernels into a header without a bench window.

**Wrapper `src/iq3_s.rs`**: copy `iq3_xxs_pair.rs` (4 launches, same asserts),
env `KERNEL_IQ3_S_PAIR_MATVEC_GFX{1201,1151}` (derived from the stem — the
stem must be exactly `iq3_s_pair_matvec`). `src/iq3_s_tables.rs`:
`IQ3S_GRID: [u32;512]`, `BLOCK_IQ3_S_BYTES = 110`, `cpu_dot_iq3_s_q8_k`.

## 5. Oracle strategy

How the siblings get truth: a scalar Rust CPU dot in `src/*_tables.rs`
mirroring `ggml_vec_dot_*_q8_K_generic` on LCG-random blocks
(`mxfp4_iq2s_oracle.rs`, `iq3_xxs_pair_oracle.rs`); rel tolerance 1e-2
(measured 1e-4); the CPU dot itself pinned to upstream by a C harness
(`tests/ref/iq2_xs_gen.c` → constants in `iq2_xs_cpu_ref.rs`, rel 1e-6).
ds4 is the reference only for full-forward dumps.

1. **`cpu_dot_iq3_s_q8_k`** (~45 lines): transcribe the generic vec_dot above
   (the 0x2e66-forced-d + LCG layout of `iq2_xs_gen.c`; `~/llama.cpp` is
   checked out for the `-I ggml/src` include). `tests/iq3_s_cpu_ref.rs`
   asserts 3 seeds at rel<1e-6. Optional `dequant_row_iq3_s` in v4flash-core
   only if `gguf_dequant_dense` needs it.
2. **`tests/iq3_s_pair_oracle.rs`** (copy of `iq3_xxs_pair_oracle.rs`,
   110-byte blocks): decode batch at B=1 with `sel=[9,240,33,9,61,128]`
   (duplicate expert), hetsplit identity (mode0+mode1 == full), chunked and
   kwide at B=40, chunk=16, `n_rows=2048`, `nb=16`. **Vary `d` per super-block
   for both weights (`F16_SCALES[rng&3]`) and xq (`gen_xq` does)** — the
   tile8 oracle passed a wrong kernel when `d` was uniform.
3. **Realism (range request)**, tensors live in shard 3/4
   (`...-00003-of-00004.gguf`, header_end 40296, alignment 32 → data_start
   40320): `blk.26.ffn_gate_exps.weight` offset 12624798208 → abs
   **12624838528**; `ffn_up_exps` abs **13558611840**; each 922,746,880 B;
   per-expert stride 3,604,480. Steps: (a) `curl -sL -r 0-8388607 -o hdr3.bin
   https://huggingface.co/unsloth/DeepSeek-V4-Flash-0731-GGUF/resolve/main/UD-IQ3_XXS/DeepSeek-V4-Flash-0731-UD-IQ3_XXS-00003-of-00004.gguf`;
   (b) parse with `Gguf::parse_reader` (gguf.rs:81) or the 60-line python used
   for this survey (v3: magic, version, n_tensors u64, n_kv u64, kv, tensor infos
   {name,n_dims,dims,type,offset}; data_start = align_up(header_end, 32));
   (c) per expert e∈{9,33,61,128,240}: `curl -sL -r $((abs+e*3604480))-$((abs+(e+1)*3604480-1))`
   for gate and up → 10 × 3.44 MiB = 34 MiB under `reference/iq3s_blk26/`;
   (d) the oracle reads `DEEPSTRIX_IQ3S_BLOB_DIR` and runs the same four checks on
   real blocks (also sanity-checks `d` finite and |w| ≤ 15·31·d), skipping to
   synthetic-only when unset. Real blocks catch layout misreads; synthetic
   covers all 512 grid rows.

## 6. Performance expectations (gfx1151)

- Bytes/weight 0.4297 (110/256) vs 0.3828 (iq3_xxs) vs 0.2891 (iq2_xs):
  gate+up 7.21 MB/expert vs 6.42/4.85. One layer of 43 on the iGPU-bound
  prefill path: +12% bytes on 1/43 of the MoE → **≤0.3% e2e** at per-byte
  parity, ≤1% even at 2× per-byte. The mix's real prefill cost is the 17
  IQ3_XXS gate/up layers (+32% bytes vs IQ2_XS) — do not attribute that to iq3_s.
- LDS: 512×4 B grid = 2 KiB vs iq2_s 8 KiB / iq3_xxs 1 KiB; sign pairs 2 KiB
  vs 1 KiB. kwide 20.4 KiB → occupancy unchanged (VGPR-bound at 12 waves).
  Data-dependent LDS gathers on a 512-entry table behave like iq3_xxs's 256.
- Target: `iq3_s_kwide` µs/call ≤ 1.12× `iq3_xxs_pair_kwide` (byte ratio) in
  `rocprofv3` kernel trace and `bench_iq2_xs_isolated BENCH_FMT=iq3s`.
- **Write all four variants at once.** Decode batch/hetsplit/chunked are a
  mechanical copy of iq2_s (~1 h); kwide is a copy of iq3_xxs kwide with a
  20-line unpack change (~2 h). Shipping chunked-only prefill would re-enter
  the ae9bd61 trap (3× on that layer ≈ +5% wall) for no saving.

## 7. Step order and effort

| # | step | est |
|---|---|---|
| 1 | `iq3_s_tables.{inc,rs}` (grid from ggml-common.h:1052), `cpu_dot_iq3_s_q8_k`, `tests/ref/iq3_s_gen.c` + `iq3_s_cpu_ref.rs` pin | 1.5 h |
| 2 | `iq3_s_pair_matvec.hip`: decode batch, hetsplit, chunked, kwide | 3 h |
| 3 | `src/iq3_s.rs`, `lib.rs`, `engine.rs`, `dispatch.rs` arms, `weight_contract.rs` (+test) | 1 h |
| 4 | `iq3_s_pair_oracle.rs` synthetic, all 4 variants | 1.5 h |
| 5 | range-fetch blk.26 experts, realism path in the oracle | 1 h |
| 6 | Q6_K head contract + embed row dequant (+ unit test vs Q6_K dense dequant) | 1.5 h |
| 7 | isolated bench `BENCH_FMT=iq3s`; VGPR check via `hipcc --save-temps`/`-Rpass-analysis=kernel-resource-usage` (≤128 for 12 waves) | 1 h |
| 8 | ds4 patch 0012 (deferred until a dump is wanted) | 2 h |
| 9 | docs + memory | 0.5 h |

Total ≈ 11–13 h; steps 1–7 need no model on disk.

## 8. Risks

- **Contract before dispatch**: adding IQ3_S to the allow-list without the
  arms turns the load-time error into a runtime `Err` at `dispatch.rs:56/101/292`
  (an error, not garbage, because strides are dtype-derived). Land 2+3 together.
- **Dedup / hot experts**: dtype-generic (`weights.rs:576` packs with the
  dtype's bpe); the dGPU hot-expert prefill (`forward_prefill.rs:2394`) runs
  the same kwide kernel on gfx1201, so the `.hip` must build for both
  targets (siblings prove `sudot4` does).
- **Alignment**: 110 ≡ 2 mod 4 — a plain `uint32_t*` deref of `qs` faults or
  splits; use the 2-aligned loads. Row stride 1760 and expert stride
  3,604,480 are 32-aligned.
- **Occupancy cliff**: any accumulator/unroll change that pushes VGPRs past
  128 drops 12→9 waves (+29–48% measured). Keep the sibling's register shape.
- **build.rs**: nothing to register, but a wrong stem silently changes the
  env name and fails at `include_bytes!(env!())` — good failure mode.
- **Oracle traps**: uniform `d` hides scale bugs; decode kernels read ONE
  token's xq, chunked/kwide index per token (see sibling oracle headers).
- **bench-LANES=1 masking**: `bench_prefill_chunked` defaults to
  `PIPELINE_LANES=1`; always bench with `=2`. Also, a single-layer kernel can
  hide entirely behind dGPU work at LANES=2 — judge it by rocprofv3 per-kernel
  time and the isolated bench, not e2e.
- **Memory**: 97.05 GiB total vs UD-Q2_K_XL's 90.2 GiB; iGPU +~6.5 GiB before
  dedup — outside today's budget (see host-RAM audit). Kernels validate
  without the model.
- **ds4 dump**: the CPU reference has no iq3_s; regenerating a UD-IQ3_XXS dump
  needs patch 0012 and mmap paging above MemTotal (feasible but slow: only
  6/256 experts per layer are touched per token).

## 9. E2E validation once UD-IQ3_XXS fits

1. `cargo test -p v4flash-kernels --test iq3_s_pair_oracle --test iq3_s_cpu_ref -- --ignored --nocapture` green (synthetic + `DEEPSTRIX_IQ3S_BLOB_DIR`).
2. `weight_contract_models` with the UD-IQ3_XXS path: `validate_model` clean, 1328 tensors.
3. Patch 0012 + dump → `forward_per_layer_vs_ds4` at blk.26 with `DEEPSTRIX_GGUF`/`DEEPSTRIX_DUMP_DIR`, zero tolerance changes.
4. `deepstrix-vector-test --gguf …UD-IQ3_XXS…` ≥ the 16/17 the other UD mixes score.
5. Perf, back-to-back same thermal window: `bench_prefill_chunked` 4K/24K at `PIPELINE_LANES=2` and decode @4K vs UD-Q2_K_XL; rocprofv3 kernel trace: `iq3_s_pair_matvec_fused_swiglu_kwide` ≤ 1.12× `iq3_xxs_pair_…_kwide` µs/call.
6. Server smoke via `run_deepstrix.sh --gguf … --bg` (K=8, dedup on) so blk.26 hot experts exercise the gfx1201 build; wipe the snapshot dir (fingerprint changes).
