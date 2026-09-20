# V4.1 known bugs and open correctness questions

Living list. Add here the moment something is found, even when it is not being
fixed right now, and delete only when it is fixed AND has a regression test.
Ranked by risk of SILENT WRONGNESS (produces wrong numbers rather than an error).

Status key: **OPEN** / *MITIGATED* / ~~FIXED~~

---

## Open

### 25. FIXED 2026-09-20 — CED replay emptied the decoder rings on a CONTINUATION, so a short suffix decoded with a 1-row window

Both CED drivers (`prefill_job_finish` and `forward_prefill_pipelined`) set
`n_raw = raw_off = 0` on layers 20..39 before replaying the suffix's last
`SWA_WINDOW` rows. That is the design for a fresh prompt (ENGINE_PORT M7 step 4:
"empty rings + W=128 give exactly the paper's truncation"), but a restored
snapshot's decoder rings hold the previous turn's replay rows at positions
`[pos0 - k, pos0)` — the window the reference decoder carries incrementally. Every
agent turn is a continuation with a short suffix (typically the 1-token turn
marker or a few hundred tool-result tokens), so the decoder saw a window of
`len(suffix)` rows instead of 128 for its first ~128 generated tokens. The
multistream harness had already tripped over it ("pf1 reference with
`last_only=true` wiped decoder windows, n_raw=1 at layers 20-39") and it was
mis-filed as a harness mistake. Fix: empty the rings only when `pos0 == 0`; on a
continuation the replay appends and evicts like any other batch.

### 23. FIXED 2026-09-20 — multistream prefill hashed DOUBLE-compressed ids for its Engram rows (production, 20:10-20:48 UTC)

`multistream::start_prefill` built `compressed = token_map[tok]` for the whole
sequence and then called `hasher.hash_sequence(&compressed)`, which compresses
its input again (`engram_hash.rs:171`). `token_map` is not idempotent (128,613 of
129,280 entries map to a different id on the second application), so nearly every
suffix row of every multistream prefill gathered Engram rows for the wrong n-grams
at layers 1 and 14. Decode rows (`hash_ids(&s.compressed, ..)`) and the legacy
path (`EngramCtx::rows_for_chunk`) were correct; snapshots saved from those
prefills carry the wrong contribution baked into their KV. Fix `da7f8f2`: hash the
compressed ids once with `hash_ids`; then `1249a98` moved the whole gather into
per-chunk lazy inputs (the same code shape as `rows_for_chunk`). Lesson: two
functions named `hash_*` took different input spaces (raw vs compressed ids);
neither the type nor the name said so.

### 24. FIXED 2026-09-20 — multistream prefill blocked the scheduler for a bulk Engram gather and held the whole prompt's inputs in host RAM

`start_prefill` called `gather_position` per (token, layer) — 24 spawned OS
threads per call, ~1.5 ms/token (a 51,877-token suffix took 78 s, every live stream
frozen, and looked like a hang) — and allocated `HC_DIM` + 2 x `ENGRAM_IN` f32 per
token up front (~15 GB for a 135K prompt on a 96 GB box running a 78 GB pool). The
batched gather (`da7f8f2`) cut it to ~0.26 ms/token (35 s for 135K, still one
blocking call); `1249a98` makes `PrefillJob` take lazy per-chunk inputs
(`next_chunk_range` / `set_chunk_inputs`), so each 512-row chunk costs ~60 MB and
~150 ms of gather inside the prefill burst.

### 22. FIXED 2026-09-20 — a short prompt after a long one reused the LONG one's indexer top-k selection (PRODUCTION prefill bug)

`bd.indexer_saved_store` (the S2 "shared selection" gate, per prefill lane) was
set whenever an index-source layer's indexer FIRED (rows with > INDEXER_TOP_K comp
rows, i.e. prompts beyond ~1K tokens) and never cleared. The reuse layers (21-23,
25-27, 29-31, 33-35, 37-39) take `s2_reuse` when `saved_store == index_source_of
(layer)`, so every later batched forward whose own layer 20 did NOT fire — any
short prompt, chunk, or arena step after a long prompt — gathered attention rows
with the PREVIOUS request's `indexer_sel_saved` (row indices into a store that is
now shorter), then attended garbage on 15 of 40 layers. Found in the multi-stream
harness: a 260-token prompt prefilled on a cold process was 0.686 nats from the same
prompt prefilled after a 1500-token one, deterministic, independent of chunking
(`MS_DIAG=job:*` and `MS_DIAG=kv`); the 1500-token prompt itself was bit-identical
(its layer 20 re-fires), and a 33-token prompt run only after the long one was
"identical" both times because both were wrong the same way. Also the reason the
S=8/16 harness runs with a 1500-token stream failed G5a (alone vs batched differed
on 21/24 rows). Fix: at an index-source layer that does not fire, set
`indexer_saved_store = -1`, so the reuse layers see this call's own decision.
Production impact: every short user turn following a > ~1K-token one in the same
server process, since 40d1201 (2026-09-14, S2). The decode path was not affected
(its `last_idx_gather_src` is reset per token).

### 20. FIXED 2026-09-20 — the batched layer driver paged its local experts ONLY when a remote was attached

**Root cause (found with `V41_GROUP_AUDIT_VERBOSE` + brace counting):** since
`f1fbe3f` (2026-09-16, "submit to box 2 BEFORE box-1 paging") the `if
remote_split_on {` block in the union branch of the pager stage closed at the very
end of the branch, so the box-2 submit, `pg.ensure(...)` (the LOCAL paging), the
exclusion and the routing audit all sat inside it. With a remote attached
(production) everything ran; without one (the harness, any layer whose
`owned_count == 0`) nothing was ever paged: every pick kept the pool's stale remap
(0 = "owned elsewhere"), the group builder skipped it, and the routed MoE was
silently DROPPED. The history dependence was the stale per-layer `remap_dev`: a step
whose picks decode had just paged inherited a valid map and computed correctly
(the 0.013-nat case); after other tokens the map belonged to other picks.
**Fix:** close the remote block right after the submit; the paging always runs (the
later exclusion/audit are self-gated on the remote). After the fix, on Paris:
arena == decode at **0.002 nats** in every history (warm, three prior tokens, fresh
pager); alone == batched == the contiguous reference **bit-identically**; 4 streams x
6 steps G5a 0/24 differ. The two "open" paragraphs below are kept as the record of
what was ruled out before the audit found it.

#### (pre-fix record) the batched layer driver's LOCAL MoE output depends on prior pager/decode history

Found by the multi-stream harness (`tests/multistream_step.rs`, run with the server
DOWN). On the 6-token Paris prompt, one K=1 arena step (`forward_step_arena`,
`RowLayout::Arena`) after the prefill matches today's decode at **0.013 nats** and
predicts " Paris" — but only when exactly ONE decode token ran on the engine before
it. After three decode tokens (or a second stream's prefill) the SAME step on the
SAME KV copy is 1.2-2.2 nats off and predicts token 8760. Bisected by per-stage dumps
of the step in both histories: layer 0's residual, mHC collapse, Q, attention output,
expert selection and expert weights are BIT-IDENTICAL, and the routed-MoE output
(`ffn_moe_recv` before the shared add, new `pf_ffn_routed` dump) differs — the bad
one has ~30% of the magnitude and correlation 0.47 with the good one (the good one
matches decode's own routed output to 1.2%). Same experts, same inputs, wrong sum.

Ruled out, each by a harness arm: pager pool state (fresh `ExpertPager` before the
step: still wrong, DIFFERENT value 2.169 vs 1.196 — so the output IS a function of
pool state, but a cold pool is wrong too), pager pool size (20/40/60 GB identical),
`V41_T2_CATCHALL`/`T2_PARTITION`/`REMOTE_SPLIT` off (identical), unified prefill
pool off, `SpeculativeAppend` scope, decode HIP graphs off (`V41_PAGER_NOGRAPH`),
a SECOND `HeterogeneousEngine` instance for the decode tokens (identical), full
device syncs after decode, and full device syncs after `pg.ensure` before the MoE
dispatch (`V41_PAGER_SYNC_AFTER_ENSURE=1`, identical). The miss path is synchronous
(`repack_in_place` drains its stream; `copy_from_host` blocks). Everything is
deterministic run to run (bit-identical dumps).

The contiguous one-row continuation (`forward_prefill_pipelined` at `pos0 > 0`,
`last_only=false`, the DSpark verify's call) is off by **1.14 nats** on the same
prompt even with one prior decode token — so it has this or a related problem in
every history; the 0.0066-nat verify measurement of 2026-09-16 does not reproduce
in the harness. `V41_VERIFY_DECODE_MOE=1` does not fix either path.

Repro (server down, ~90 s each; see the harness header for the env):
`MS_STREAMS=1 MS_PROMPT_IDS=0,671,6102,294,8760,344 MS_SKIP_PF1=1 MS_STEPS=1` (good,
0.013) vs `MS_STEPS=3` (bad, 1.196) vs `MS_STEPS=3 MS_FRESH_PAGER=1` (bad, 2.169).
Dumps: `DEEPSTRIX_DUMP_SUBTENSOR_LAYERS=0,1 DEEPSTRIX_DUMP_SUBTENSOR_DIR=...`
(pf_ tags carry the row's real position; `pf_ffn_routed_p<pos>` is routed-only).
Static pass (2026-09-20, after the window): with no box 2 attached nothing in the
pick loop can leave a pick out of `ids` (every exclusion is gated on
`remote_split_on` / `owns_remote`), `ensure` resets `remap` to identity slots and
writes `-(slot)-1` for hits AND misses, the builder's bound is the pool size under
the unified pool, and every expert-offset in the MXFP4 kernels is 64-bit. So the
mechanism is not visible from the code; the bad routed output has ~1/6 of the
magnitude, i.e. most picks are DROPPED or land on empty groups, not computed wrong.
Probe for the next window (compiled, off by default): `V41_GROUP_AUDIT=1
V41_GROUP_AUDIT_VERBOSE=1` prints per layer the groups the builder filled
(`slot:count`), enqueued vs expected, and for row 0's six picks `id / remap /
pager slot_of / FNV of the slot's first 64 KB of gate bytes`. Run it in the good
(`MS_STEPS=1`) and bad (`MS_STEPS=3`, `MS_FRESH_PAGER=1`) histories with
`DEEPSTRIX_DUMP_SUBTENSOR_LAYERS=0`; the first line that differs is the bug.

**Why it matters:** the arena is the multi-stream decode step; G5b cannot pass
until this is closed. It very likely also bites PRODUCTION prefill after decode
(a second turn's prefill runs this same driver after decode tokens).

### 21. FIXED 2026-09-20 — same root cause as #20: the harness's prompt prefill dropped every routed expert

After the #20 fix, on Paris: **KL(bf16 oracle || decode) = 0.0033 nats (was 1.24)**,
max |logit Δ| 1.2 (was 16); the arena step 0.0019; against the Q8-weights oracle
0.027 / 0.029 (the format floor, ROADMAP item 6). Decode's own math was never wrong:
the prompt prefill in the harness ran the batched driver with no remote, paged
nothing, and built layer >= 1 K/V without any routed MoE; decode then attended that
KV. Production prefill has a remote and was not affected on layers box 2 owns
experts in. The layer-1 "seed" in the record below is exactly the first layer whose
K/V depends on layer 0's MoE.

#### (pre-fix record) decode is 0.9-1.2 nats from the CPU oracle on the Paris prompt

`scripts/v41_oracle` (DeepSeek's unmodified model.py, layer-streamed on CPU) on
`[0,671,6102,294,8760,344]`: top-1 " Paris" 23.77, then " a" 21.28. The engine's
DECODE path after a 5-token batched prefill: top-1 " Paris" too, but **KL(oracle||
dec) = 1.24 nats, max |logit diff| 16.0**, top-5 `[Paris, 'Ġ', 16, ' a', 680]`.
Decode-only (2-token prefill `[0,671]`, then 6102/294/8760/344 teacher-forced
through decode): **0.88 nats**, max |d| 14.0, and the per-position argmax matches
the oracle at 3 of 4 positions (pos 4: engine 734 vs oracle 344 " is"). On a random
33-token prompt: 1.8 nats, argmax differs. This was never measured as a KL before;
the 2026-09-12 `compare_t200_q8_floor` record already showed a DIFFERENT top-1 on a
200-token real prompt and per-position deep-layer relative errors of 0.1-0.9, filed
as the "Q8 floor". Q8_0 dense projections + f16 KV + f32-vs-bf16 residual do not
plausibly cost a nat; something is structurally off vs the reference. Not caused by
the multi-stream work (the arena reproduces decode to 0.013 nats when it works).
Repro: oracle `--prompt-ids 0,671,6102,294,8760,344 --dump-argmax`, harness
`MS_PROMPT_IDS=0,671,6102 MS_FORCE_CONT=6102,294,8760,344 MS_SAVE_LOGITS=DIR`, then
compare `s0_t3_dec.bin` with `logits_last.pt` (scratch `cmp_oracle2.py`).
MEASURED 2026-09-20: the load-time Q8_0 requant of the fp8 dense projections
(ENGINE_PORT.md R19; ROADMAP item 6) is NOT the cause — the Q8-weights oracle
(`V41_ORACLE_Q8=1`) is only **0.037 nats** from the bf16 oracle on Paris (same top-1,
max |Δ| 4.9), while decode is **1.19 nats** from the Q8 oracle (decode-only 0.87).
So ~1 nat is engine error. Per-layer bisection vs both oracles (scratch
`cmp_floor.py`, port-doc metric max|Δ|/max|ref| at position 5): the residual
ENTERING layer 1 is at the floor (1.3e-2 vs 4.9e-3) and entering layer 2 it is
0.19 (batched) / 0.25 (decode) against a 0.035 floor — 6x over — then it compounds.
**Layer 1's output is the seed, in BOTH paths.** Layer 1 differs from layer 0 only
by Engram. Verified static (2026-09-20): `engram_gate_add.hip` matches the
reference `Engram.forward` line for line; the gather (`gather_uncached` and the
row cache) round identically and are unchanged since 2026-09-13 apart from the
cache; the hash-parameter dumps date from 2026-09-12; coalesced expert reads are
off on box 1. Unverified: the Engram `wkv` projection's activation quantisation
(engine Q8 int8 vs the reference's fp8 `act_quant` of the bf16 rows) and layer 1's
loaded weights. The 2026-09-13 M6 "layer 1 passes at 2x floor" ran with ALL experts
resident (no pager) and before 4616474 tightened the vacuous gates, so it may never
have tested this. Next window: `v41_layer0_parity` with `V41_LAYER=1 V41_ISOLATED=1`
on the Paris fixture, resident and with `V41_PAGER=1`; if isolated layer 1 fails,
dump `engram_kv` (key/value after wkv) and compare against the reference's.

### `deepstrix-expert-bench --check-layer` MISMATCHES at B=4 (decode branch) — **OPEN**, found 2026-09-19
`--check-layer 0 --check-n 3` against a 3-expert daemon: B=1 and B=64 are
BIT-IDENTICAL to the local shard, **B=4 differs in 16,769/20,480 f32 values (19 in
f16)** — identically on the 2026-09-18 box-2 build and on the 2026-09-19 build
with the fixed-cost changes, so it is not from those. `REMOTE_EXPERTS.md` records
B=4 as bit-identical at the time it was written, so this is a regression since.
B=4 takes the daemon's DECODE branch (`b <= decode_max_b`), B=64 the batched one;
the local reference in the bench uses the same executor. Suspects: the
batch-size-dependent decode-vs-verify disagreement (`VERIFY_DISAGREES_WITH_DECODE.md`)
or the B>1 `moe_gate_up_batch_hetsplit`/`moe_down_batched_hetsplit` row handling.
Production decode is B=1 (identical), so this bites DSpark verify (B=2..6) only.

2026-09-21 addendum: with the bench's local reference forced onto the SAME path as
the remote (`--batched` on both, bench patched that day), a `--paged` daemon and
`--catchall` ids, 30 checks at L5/L7, B=1/4/32, hits-first on and off, 40- and
200-slot pools were ALL bit-identical, including B=4 while faulting. One B=4 run
that day showed 2 of 4 rows off at the f32 LSB (f16 identical) and could not be
reproduced in 6 repeats of its exact configuration; the bench binary of that run
predates the same-path patch, so it most likely compared the batched remote against
the local DECODE path — the ~1e-7 batched-vs-decode difference above. The original
entry (decode branch on both sides, static 3-expert daemon) is still open.

## Start here

### Why DSpark output was degenerate, in one paragraph -- **CLOSED 2026-09-16**
Accept mode emits `corrected = row_argmax(0)` -- the VERIFY's row-0 token
(`engine_worker.rs`). Row 0 is the NON-speculative position, i.e. the token
plain decode would emit. The in-tree cross-check
(`V41_VERIFY_PROBE=k,k V41_VERIFY_BATCHED=1` -> `dspark.xcheck`) measured that
agreement at **~64%**, so ~36% of every generated token differed from what
decode would produce -- exactly the observed degeneracy.

**That was #0b, and #0b is fixed** (`b6ce2b8`, `056e262`, and the widening that
followed): the MoE group builder silently dropped every sparse-resident routed
expert because its group-id bound was `N_EXPERT` while the sparse verify
residency emits ABSOLUTE pool slots. Agreement is now 71/71 at B=1 and 68/71 at
B=6, KLD 1.707 -> 0.0066. **Everything below this line in "Start here" is the
pre-fix investigation and is kept only as a record of what was ruled out** --
the attention-kernel hypothesis it builds toward was REFUTED by measurement
(the kernels are arithmetically identical and attend the same slots).

**This is independent of accept rate.** E reached 3.500 (oracle base 4.382) with
output still degenerate. E buys SPEED; row-0 agreement buys CORRECTNESS. Do not
conflate them -- most of 2026-09-16 was spent doing so.

    to make DSpark USABLE:  drive dspark.xcheck `agree` -> ~100%
    to make DSpark FAST:    drive `e_tokens_per_step` up

Measured attempts on the agreement axis: aligning every matvec/projection in the
attention chain to decode's kernel moved it 69 -> 71/111 and cut KLD 2.026 ->
1.631, i.e. ~20% of the divergence, and the projection half of that REGRESSED E
(see the methodology warning below, and 16ec20a). Running the verify on decode
primitives (`V41_VERIFY_DECODE_PATH=1`) did not help either: still degenerate,
E 1.073.

So the residual is NOT simply kernel-family mismatch.

**Ruled out: the speculative KV append/rollback.** With `V41_PROBE_FPRINT=1`,
111 of 111 probes report "probe rollback restored all components" and zero
report a failure. The probe is transparent, decode is not corrupted by the
verify running before it, and the row-0 disagreement is genuinely the verify
computing different logits.

**Everything ruled out so far for the row-0 disagreement:** batching (B=1 alone
diverges 2.026 nats; B=1->6 adds only 0.2), prompt length (same ~61% agreement
at 33 and 647 tokens), the mHC carry, the indexer, CED, box 2's batched
down/reduce (6%), the mHC mix (now bit-identical), every matvec/projection in
the attention chain (~20% total, and the projection half regresses E), the
KV append/rollback (fingerprint-clean), and running the verify on decode
primitives (`V41_VERIFY_DECODE_PATH=1`, still degenerate at E 1.073).

What remains, per the per-layer bisection (`938ebba`): the attention kernels
themselves. `heads` diverges 2-6e-03 while the mHC collapse feeding it is
bit-identical, and that divergence GROWS with position (2.2e-03 at pos 42 ->
5.8e-03 at pos 45), i.e. it scales with the KV window. Decode uses
B=1-specialised score/smwsum; prefill uses batched per-row windows. Compare
those two directly -- what each attends over, and in what order it reduces.

### Reproducing #0b in one command

    env V41_T2_CATCHALL=2 V41_PAGER_WINDOWS=21 \
        V41_VERIFY_PROBE=1,1 V41_VERIFY_BATCHED=1 \
        bash ~/run_v41_server.sh
    # then one long request (>= ~600 prompt tokens), and read:
    #   dspark.xcheck: ... agree=N total=M ... mean_kld_nats=...
    # `agree/total` is the number that matters (accept mode emits row_argmax(0)).
    # Current: ~69/111 (62%), KLD ~1.9 nats.

Per-layer bisection, both paths, same position:

    env ... DEEPSTRIX_DUMP_SUBTENSOR_LAYERS=0,1 DEEPSTRIX_DUMP_SUBTENSOR_DIR=/tmp/d ...
    # writes layer_<NN>_{pf,dec}_{pre_residual,attn_cur,attn_out,heads}_p<POS>.bin
    # diff pf vs dec at the same layer and position.

**Compressed-row selection is RULED OUT.** It was the obvious next hypothesis --
prefill clamps `n_comp_per` to `min(actual, INDEXER_TOP_K)` while decode takes
sparse `INDEXER_TOP_K.min(n_index_comp)` only when its indexer gate fires and
dense `n_comp_full` otherwise, so the counts could agree while the SELECTED ROWS
differ. But at a 33-token prompt `n_comp` is ~10-20, far below
`INDEXER_TOP_K=512`, so BOTH paths are dense over the SAME rows -- and agreement
is 50/81 (61.7%), statistically identical to the 647-token case (68/111, 61.3%).
Same rows, same disagreement.

**THE TIGHTEST STATEMENT OF THE REMAINING BUG.** At a 33-token prompt, B=1:

    residual entering layer 0   bit-identical (0.000e+00)
    mHC collapse (attn_cur)     bit-identical (0.000e+00)
    KV attended                 same rows, both dense
    -> heads                    DIVERGES (2-6e-03)
    -> row-0 argmax             differs 38% of the time

Identical inputs, identical KV, different attention output. So the attention
kernels themselves compute different results: decode's
`launch_score_b1_htiled_wmma` + its smwsum twin versus prefill's
`launch_score_batched_htiled_wmma*` + `launch_softmax_wsum_batched_*`. Note
prefill defaults to the `_f16s` (f16 scores) variants while decode uses f32;
`DEEPSTRIX_F32_SCORES=1` switches prefill to f32 and cuts KLD 1.904 -> 1.711 but
makes agreement WORSE (69 -> 66/111), so precision alone does not explain it.

Compare those two kernel families directly -- tiling, the order keys are
reduced in, and the softmax normalisation -- rather than swapping variants and
re-measuring.

**First thing to check there, because it is arithmetic and not numerics:** the
two kernels express the attended window DIFFERENTLY, and whether they resolve to
the same slots at row 0 has not been verified.

    decode   attention_mixed_score_b1_htiled_wmma
             scalar `n_raw`, `raw_off` (passed as 0, buffer pre-sliced), `n_comp`
    prefill  attention_mixed_score_batched_htiled_wmma
             per-row `n_raw_per[]`, `n_raw_offset_per[]`, `n_comp_per[]`, built as
             causal_end = n_raw_before + i + 1
             n_per      = min(causal_end, SWA_WINDOW)
             offset     = causal_end - SWA_WINDOW

The verify APPENDS its row before attending, so `n_raw_before` differs from the
`ls.n_raw` decode passes, and the two can disagree by a slot depending on where
each sits relative to the append and the `n_raw`/`raw_off` update
(`forward_layer.rs`: `if ls.n_raw < SWA_WINDOW { ls.n_raw += 1 } else
{ ls.raw_off += 1 }`). A one-slot window shift is exactly the kind of difference
that survives every precision fix and produces a differently-shaped
distribution.

CHEAP TEST: dump `n_raw_per[0]`, `n_raw_offset_per[0]`, `n_comp_per[0]` from the
verify and the corresponding scalars from decode, at the same position, and
diff. No kernel work required, and it either finds the bug or removes the whole
window-convention question from the search.

## Methodology warning

### Verify-vs-decode KLD and ACCEPT RATE can move in OPPOSITE directions
Measured 2026-09-16. Aligning the Q/KV/output projections to decode's kernels
reduced verify-vs-decode KLD 2.026 -> 1.631 nats while DROPPING accept rate
1.788 -> 1.372. Reverted (16ec20a). One of those changes also fed
`q8.matvec_batched` a stale `xq_n_embd` -- the f16x arm it replaced consumes
`x16_n_embd`, and the Q8 quantisation is only performed on the non-f16x arms.

**Score verify changes on E and on output coherence, not on KLD alone.** KLD is
a proxy; it is not the objective, and this session shows it can point the wrong
way.

**Worse: KLD and ARGMAX AGREEMENT are themselves decoupled here.** Three changes
improved one and degraded the other:

    Q/KV/output projections -> decode kernels   KLD 2.026->1.631  E 1.788->1.372
    attention scores f16 -> f32 (decode-matching)
                                                KLD 1.904->1.711  agree 69->66/111

So the verify is not a noisier decode -- reducing average distributional
distance does not move top-1. It is computing a differently SHAPED
distribution. Any fix has to be validated on `agree` (and on the generated
text), because that is what accept mode actually emits.

For reference, what the kept fixes are worth on E (back-to-back, same prompt):

    pre-session kernels                     E 1.788
    + mHC pre-scaled, fp32 gate/compressor/indexer
                                            E 3.500   (oracle base 4.382)

Output is still degenerate at E 3.500, so ACCEPTANCE and COHERENCE are separable
problems here -- raising E does not by itself fix #0b.

## Fixed 2026-09-18

### FIXED — box 2's per-layer region was a HARD BOUND (`3eede97`, `eb002bb`)
Box 2 carved its 6160 slots into 40 equal 154-slot regions and confined the
prefill victim search to the layer's own region. A chunk whose union for one
layer exceeded 154 found no legal victim and failed the request outright
("layer 19 has no evictable slot") while ~6000 slots sat evictable elsewhere.
Now one global LRU for every request shape; `V41_B2_GLOBAL_POOL=0` restores the
old search. Measured: prefill 9.3K tok 27.1 -> 21.6 s, 13.3K tok 31.5 -> 25.4 s.

### FIXED — one box-2 error bricked the server permanently (`11a6c82`)
The writer thread exits on the broken pipe, `tx_req` closes, and every later
submit fails forever: the process served 500s until restarted by hand. The client
now redials at the next request BOUNDARY (never mid-request — the old socket's
in-flight tickets cannot be reconciled and a fresh HELLO may report different
ownership). Verified by killing box 2 and watching box 1 recover unaided.

### FIXED — attention scratch was sized off a DEAD predicate (`3409948`)
`attn_max_scored_keys` branched on `indexer_gathers(ratio)` = `!cfg!(v41) &&
ratio == 4`, dead under V4.1, so every compressed layer was charged its whole
store. S2 makes every layer attend only its index source's top-512, so the real
bound is `raw_window + INDEXER_TOP_K`, with no `n_kv_max` term. Scratch 2080 ->
192 MiB and CONSTANT in context; dGPU freed 1.9 GiB; `--ctx 307200` now boots
with identical throughput and bit-identical output.

### FIXED — the `developer` role 422'd every request (`c6888df`)
OpenAI renamed the system role for o1-era models; prime-agent sends `developer`
and we accepted only `system`, so deserialization failed on messages[0] and the
client's retries gave up. `#[serde(alias = "developer")]` on the existing variant.

## Open, found 2026-09-18

### 17. OPEN — ARCH_SPEC 1.5 candidate pool is NOT wired (fidelity)
`CANDIDATE_SOURCE_LAYER/TOPK_BLOCKS/BLOCK_SIZE` sat in config.rs referenced by
nothing. The reference masks index sources 24/28/32/36 to the 2048x8 = 16384
candidate positions layer 20 publishes (`index_score.masked_fill(~candidates,
-inf)`); we select from the whole store, so those layers can pick positions the
model never considers. Kernels + oracle landed (`candidate_blocks.hip`,
`candidate_blocks_oracle.rs`, verified against the CPU transcription of
`select_candidate_blocks` at 16385/65536/307200 and a ragged batch) but are NOT
wired into forward_layer/forward_prefill. NO-OP BELOW 16384 compressed positions,
so short context was always conformant; long context is not.

### 18. OPEN — relaunching box 1 races the driver's VRAM reclaim
Killing the server frees its VRAM asynchronously: `mem_info_vram_used` drops
below 1 GiB before a large `hipMalloc` can succeed, so a relaunch inside that
window dies mid-weight-load with `hipErrorOutOfMemory`. Observed 3x on
2026-09-18; a retry always succeeded. Waiting on the counter is necessary but not
sufficient — `~/scripts/start_v41.sh` retries up to 5 times.

## Silent wrongness

### 0c. FIXED (2026-09-17, `fedbea2`) — decode DOUBLE-COUNTED every pick box 1 held

Under `V41_T2_CATCHALL=1` decode submitted the whole 6-pick `sel_host` to box 2
(`submit_unmasked`, "the hub already decided the partition in `owns_remote`").
Box 2's `run_path` computes EVERY non-`NO_PICK` entry it is handed, and
`ffn_combine` adds that partial to the local iGPU one — so each pick box 1 kept
(resident) was computed on BOTH boxes and added twice. `verify_routing_exactly_once`
validates the hub's CLAIM (remap vs `owns_remote`), not what box 2 computes, so it
never fired. The prefill/verify path already blanked non-remote picks
(`sel_for_remote`); decode did not.

This is what the mode-2 comment in `forward_layer.rs` described as "mode 1 after a
37-tok request -> DEGENERATE ... f32 addition is not associative" (it was not
associativity: it was residency-dependent double-adding), and it is a large part
of #0's run-to-run nondeterminism and of "a 1396-slot box-1 LRU measured SLOWER".

Fix: `sel_remote`/`ew_remote` blank the local picks (`V41_REMOTE_NOMASK=1`
reproduces). With the mask, mode 1 is still history-dependent (which box computes
an expert changes the f32 grouping); `V41_T2_PARTITION=1` (fixed hash home per
expert) is bit-stable across repeats — measured sha-equal on two turns, twice.

### 0b. ~~FIXED~~ (2026-09-16) — verify vs decode ~1.9 nats: the MoE group builder dropped every sparse-resident expert

**See the FIXED entry under "Sparse verify residency" below for the root cause,
the fix, and the measured before/after.** In one line: `moe_group_builder.hip:118`
bounds group ids by `n_expert`, that bound is a BUFFER LIMIT, and the sparse
verify residency emits absolute pool slots above it. 43/71 -> 71/71, KLD 1.707 ->
0.0066.

**The section below is the pre-fix investigation, kept as a record of what was
ruled out. Its conclusion ("a different `hc_mixes` kernel", then "the attention
kernels themselves") is WRONG** -- the attention kernels were diffed by hand and
are arithmetically identical, and were shown to attend the same slots at three
overlapping positions. The per-layer bisection was reading amplification of the
MoE drop, not a seed in attention.


**LOCALISED 2026-09-16.** Per-layer residual diff (both paths now dump; see
`92257cc`, `3e4a17c`), same position, B=1:

    entering layer 0   relRMSE 0.000e+00   BIT-IDENTICAL
    entering layer 1   relRMSE 6.504e-03   <- FIRST DIVERGENCE
    entering layer 2   8.753e-02
    entering layer 5   2.773e-01
    entering layer 39  5.602e-01           -> the ~1.9 nats at the head

Both paths enter layer 0 identically and diverge leaving it. Layer 0 is the
SEED; everything after is amplification.

**Character of the seed:** position-INDEPENDENT (~6e-3 at positions 42,43,44,46),
so not attention/KV, which would scale with context. Spread over 20380/20480
channels and all four HC copies (copy 2 is 10x cleaner than the rest), so not a
few wrong experts or a slot bug -- it is the same math from a DIFFERENT KERNEL.

**ROOT CAUSE (2026-09-16): the verify is built on the PREFILL path, and that
path uses a different kernel family from decode at EVERY stage.**

`docs/v41/DECODE_M8_PLAN.md:102` already concluded this architecturally --
"verify forward MUST be built from DECODE primitives ... forward_verify_batch(B
<= 6) = the decode chain with a row dimension" -- but the verify was built on
`forward_prefill_pipelined` anyway. The batched path was optimised with WMMA
GEMMs throughout while decode kept `matvec`, so the verify cannot reproduce
decode, which is the one property DSpark requires.

Sites found so far (decode kernel -> prefill kernel):

    mHC pre_attn/pre_ffn  matvec_pre_scaled      -> gemm_batched_wmma  FIXED 21cb203,1fa5c35
    MoE gate              matvec                 -> gemm_batched_wmma  FIXED 5b51bcd
    compressor kv/score   matvec                 -> gemm_batched_wmma  FIXED e63fd08
    indexer index_k/q     matvec                 -> gemm_batched_wmma  FIXED e63fd08
    Q projection (q_b)    q8.matvec              -> q8_wmma.gemm_f16x  OPEN
    KV projection         q8.matvec              -> q8_wmma.gemm_f16x  OPEN
    output projection     q8_grouped.matvec_grouped + q8.matvec
                                                 -> q8_wmma.gemm_f16x  OPEN

Patching these one at a time is chasing symptoms. The durable fix is the one
DECODE_M8_PLAN specified: a verify built from decode primitives with a row
dimension, so parity holds by construction rather than by matching kernels
pairwise forever.

**Measured bracket inside layer 0** (both paths now dump; `938ebba`):

    entering layer 0   0.000e+00   identical
    mHC collapse       0.000e+00   identical  (after 1fa5c35)
    heads              2-6e-03     <- seed: QKV chain / attention
    attn_out           8e-03..1.2e-02         (output projection doubles it)
    layer 0 out        6.5e-03
    layer 39           5.6e-01     -> 1.72 nats at the head

**Mechanism (mHC instance, now fixed):** the two paths computed `hc_mixes` with
different kernels.

    decode   forward_layer.rs:609  matvec_narrow_ksplit_pre_scaled
                                   (K split into 20 chunks + reduce, RMS folded IN)
             forward_layer.rs:615  matvec_pre_scaled  (non-split, RMS folded IN)
    prefill  forward_prefill.rs    matvec_narrow_batched
                                   (single-pass warp reduction, RMS applied SEPARATELY)

Different reduction order AND a different point of applying the RMS scalar. Then
`hc_split_sinkhorn` runs **20 doubly-stochastic iterations** on those coefficients
(`ARCH_SPEC:45`) before they scale the residual -- an iterative amplifier that
turns f32-level mix differences into 6e-3 on the residual.

**Fix:** give prefill a BATCHED form of decode's exact kernel
(`matvec_narrow_ksplit_pre_scaled` / `matvec_pre_scaled`, i.e. pre-scaled and
K-split the same way), so the verify reproduces decode bit-for-bit at b<=64.
Swapping to `matvec_narrow_batched` (`21cb203`) fixed the f16 SPEC violation and
was worth +83% and E 2.185->3.077, but it is still not decode's kernel.

Earlier framing (superseded by the localisation above):
QUANTIFIED 2026-09-16 with the in-tree cross-check
(`V41_VERIFY_PROBE=6,6 V41_VERIFY_BATCHED=1` -> `dspark.xcheck`), 648-token
prompt, deterministic split, box 2 fixed:

    verify vs decode, f32 matvecs : agree 62/111 (55.9%)  cos 0.778  KLD 2.21 nats
    verify vs decode, all-WMMA    : agree 66/111 (59.5%)  cos 0.781  KLD 2.06 nats

    for scale: decode vs CPU ORACLE = 7.52e-04 nats

So the verify is ~3000x further from decode than decode is from the fp32
reference, and disagrees outright on ~42% of positions. **This is structural,
not numeric** -- the f16->f32 matvec fixes moved accept rate a lot (E 2.185 ->
3.077) but barely moved this (2.06 -> 2.21), so precision was a second-order
term on top of a path computing something different.

DSpark cannot emit correct tokens at any speed until this closes. Decode is the
right reference to iterate against (it is near-exact vs the oracle), so no CPU
oracle is needed in the loop -- just drive `dspark.xcheck` KLD toward ~0.

Candidates not yet separated: the speculative KV append's window addressing
(`SpeculativeAppend`, the raw_off slide), the compressor rollback, CED mode
(the verify runs `last_only=false` => `CedMode::Exact` while decode does not),
and the batched prefill attention at B=6 generally.

Earlier framing (still true, now explained):
Now cleanly reproducible, and no longer maskable as noise (#0 is fixed, runs are
bit-identical). Same box 2, same 648-token prompt, same config, temp 0:

    decode only : "# Maintaining a Lighthouse Through a Winter Storm: A Keeper's
                   Night..."                                        COHERENT
    DSpark      : "# Maintaining Lamp, Lens and the signal apparatus, and the the
                   the the"                                         DEGENERATE

The verify decides which tokens are emitted, so the verify is producing them.
Short prompts stay coherent; long ones degenerate, so suspect state that only a
multi-chunk / large-B prefill establishes (KV window addressing across the
speculative append, the compressor rollback, or the ring seeding's interaction
with either). E is ~2.2 either way, so this is not simply low acceptance.


### 0. FIXED (2026-09-16, `7f89090`) — was box 2's batched group-id bound
**ROOT CAUSE + FIX.** Box 2's remap encodes ABSOLUTE pool slots (since
2026-09-14) but the batched MoE passed `N_EXPERT` (384) as the group-id bound;
`moe_group_builder.hip:118` dropped every `g >= 384` while the reducer still
counted those picks as ours and summed their ZEROED partial rows. Every routed
expert above slot 383 contributed exactly 0.0 -- essentially all of layers 2..39
with the shipped 6160-slot assignment. B=1 decode was spared (no group builder);
every prefill chunk and every DSpark verify batch was hit.

Discrete because each pick is binary; bimodal because the pool settles into a few
slot layouts; survived box-1 restarts because the ShardPool LRU is never reset.

VERIFIED after the fix, two DSpark runs in separate processes, 648-token prompt:

    relRMSE 0.000e+00   KLD 0.000e+00   BIT-IDENTICAL   (was relRMSE 0.42)

The determinism gate is trustworthy again. **Every two-box measurement taken
before `7f89090` ran against a model silently missing most of its experts** --
including everything in this session and any earlier A/B that used the batched
path. Treat those numbers as void.

Original report follows.

This was the highest-priority bug in this file and it invalidated a methodology
the project depends on. `scripts/v41_determinism_gate.sh` scores expert-cache
changes by requiring `T2_CATCHALL=2` + temp 0 + same prompt to reproduce. It does
not. Three runs of one fixed config (catchall=2, WINDOWS=21, 648-token prompt,
temp 0), first-token logits:

    a vs b : relRMSE 0.170   KLD 0.00196
    a vs c : relRMSE 0.408   KLD 0.567
    b vs c : relRMSE 0.428   KLD 0.545

argmax was stable (5) in all three, and a is close to b while c is far -- BIMODAL,
not drift, which points at a discrete state difference rather than accumulating
float error.

**LOCALISED TO BOX 2 (2026-09-16, measured).** Box 1 run SOLO with no remote at
all (`V41_REMOTE_SPLIT=0`, `V41_PAGER_STRIDE=384` so the full 384-expert union
fits a window) is **BIT-IDENTICAL across two separate server processes**:

    box 1 solo        relRMSE 0.000e+00   KLD 0.000e+00   BIT-IDENTICAL
    box 1 + box 2     relRMSE 0.42        KLD 1.3-1.5     argmax 35 or 5

So box 1's whole pipeline -- prefill, pager, two-lane driver, KV, mHC, MoE -- is
deterministic. **100% of the nondeterminism is box 2.** That also explains the
6.02-vs-14.21 tok/s swing on an identical config, the apparent geometry
sensitivity (#1, retracted), and the long-prompt DSpark degeneracy.

Ruled out inside box 1: float atomics (none in the V4.1 path); HashMap iteration
order (every map is lookup-only, LRU order comes from a VecDeque); the two-lane
pager/iGPU race (`V41_PAGER_SYNC_IGPU` guard changed nothing: 0.416 vs 0.406).

**Next:** box 2's `ShardPool` assigns experts to slots by LRU history, which
persists across box-1 restarts. If its partial-sum reduction iterates or groups
by SLOT, f32 summation order changes with residency history -- changing results
without changing which experts are computed, and bimodally if the pool settles
into one of a few states. Test cheaply by restarting `deepstrix-expertd` between
two box-1 runs.

**Prior hypothesis (superseded by the solo measurement above):** box 2. Its LRU persists across box-1
restarts, and with `V41_B2_POOL_FLOOR=0` its global victim search assigns experts
to different SLOTS depending on history. If its MoE groups/reduces by slot order,
the f32 summation order changes with residency -- residency changing VALUES, on
box 2, even under the deterministic split. Test: restart `deepstrix-expertd`
fresh before each run and see whether the spread collapses.

**Consequence for everything else in this file and in memory:** any logit- or
sha-based comparison taken without a same-config control is uninterpretable if
its effect is below ~0.43 relRMSE. That includes the geometry results in #1 and
the mHC kernel comparison. Re-measure with a noise floor, or interleave arms
inside ONE process.


### 1. RETRACTED — the geometry 'corruption' was NOISE. See #0.
> **RETRACTED 2026-09-16 (same day).** The 14% relRMSE below is INSIDE the
> run-to-run noise floor, which I had never measured. Three runs of an IDENTICAL
> config differ by up to relRMSE 0.43 / KLD 0.57 (see #0). The geometry deltas
> (0.136, 0.145) are smaller than that. There is no evidence `V41_PAGER_WINDOWS`
> corrupts anything. I called this bug resolved, then confirmed, then retracted
> it in one session -- each time by changing one variable and attributing the
> difference without establishing what "no change" looks like. **Measure the
> noise floor before attributing any delta.**

**SUPERSEDED — the measurement below is real but is noise-dominated (2026-09-16).** sha comparison cannot
distinguish 1e-7 from corruption -- greedy decoding amplifies any difference into
a different token and then totally different text. First-token logits
(`V41_DUMP_FIRST_LOGITS`) have no trajectory amplification. Deterministic split
(`T2_CATCHALL=2`), 648-token prompt, ONLY the window geometry varying:

    WINDOWS=21 (reference)   argmax 35    logit rms 4.08
    WINDOWS=8    max|d|=3.27  RMSE=0.589  relRMSE=14.5%  KLD=0.057 nats  argmax 35
    WINDOWS=4    max|d|=2.82  RMSE=0.556  relRMSE=13.6%  KLD=0.140 nats  argmax  5  DIFF

**relRMSE ~14%.** f32 re-association is ~1e-6. This is six orders of magnitude
too large to be numerics: the pager geometry CORRUPTS THE MODEL, and WINDOWS=4
changes the argmax of the FIRST generated token.

Consequences:
- Every throughput number taken at a non-default WINDOWS is measured on a
  different (wrong) model. The 14.21 tok/s decode result at WINDOWS=4 is void.
- WHICH geometry is correct is still unknown -- 21 is only the reference here,
  not a verified truth. Needs `scripts/v41_oracle`.
- This is the highest-priority bug in this file. It is upstream of all perf work.

**Ruled out as the cause:** residency-driven split / f32 re-association (this test
holds the split deterministic); prefill lane racing (single-lane prefill diverges
too -- 1-lane W21 sha a03322848dd6 vs 1-lane W4 d2ee0b0f07e3).

**PARTIAL RESOLUTION, THEN REOPENED (2026-09-16).** The "resolved" claim below
held only for a SHORT prompt and is WRONG in general. With a ~648-token prompt,
still under the deterministic split, geometry changes the output again:

    LONG prompt, catchall=2, two-lane prefill
      WINDOWS=21 -> sha 328d5151f9de
      WINDOWS=4  -> sha 2a27d105ff60      DIFFERENT

So f32 re-association from a residency-driven split does NOT explain it. A short
prompt is a single chunk; a long one is multi-chunk and TWO-LANE, which is the
discriminator.

**Leading hypothesis (from the architecture review, not yet confirmed):** the
prefill steady state is `post_A(L), pre_A(L+1), post_B(L), pre_B(L+1)`
(`forward_prefill.rs:731-760`), so `ensure_layer_union(L+1)` for lane A runs on
the host BEFORE lane B's layer-L MoE has been waited on. When `window_of(L)` ==
`window_of(L+1)` the union path clears `slot_key` for the whole window and
reassigns from `next_free = 0` (`expert_pager.rs:988-996`), overwriting bytes
lane B's queued kernels will read AND the single shared `remap_dev`. Both
survive only on an accidental null-stream drain. The failure is MONOTONE IN
WINDOW COUNT exactly as observed (more windows -> fewer L/L+1 collisions;
WINDOWS=1 -> all 40 layers in one window -> garbage).

**Decisive test:** single-lane prefill (`V41_PREFILL_SINGLE_LANE_MAX` large) at
two geometries. If the shas then agree, the lane race is the cause.

The short-prompt observation below is still true and still useful, but it is NOT
a resolution.

Under the DETERMINISTIC split (`V41_T2_CATCHALL=2`) SHORT-prompt output
is geometry-INDEPENDENT. Same prompt, temp 0, 648-token prompt:

    catchall=2, WINDOWS=21 -> sha 0a9a37457b53   167 ms/tok   6.00 tok/s
    catchall=2, WINDOWS=4  -> sha 0a9a37457b53    70 ms/tok  14.21 tok/s   IDENTICAL

So `V41_PAGER_WINDOWS` does NOT mis-address weights. The geometry-dependence
seen under `catchall=1` is the documented mode-1 behaviour: the split is decided
by `pg.is_resident`, so geometry changes WHICH BOX computes which experts,
changing the f32 partial-sum grouping (`forward_layer.rs:2296-2320`). That is
exactly what mode 2 exists to remove.

Consequences:
- **E is not a correctness probe** under mode 1. E moved with geometry because
  the SPLIT moved, not because the main model computes differently. An earlier
  note in this file claimed otherwise; it was wrong.
- **WINDOWS is safe to tune under mode 2**, and there it is worth 2.4x on
  decode at byte-identical output. Tune it there, never under mode 1.
- Any A/B that varies residency MUST run under `V41_T2_CATCHALL=2`, per
  `scripts/v41_determinism_gate.sh`.

Original report follows.

Residency must never change numerics; a miss is a page-in, not a different
answer. LRU size is correctly neutral (25 -> 1510 slots: byte-identical output).
But the DENSE packed-window geometry is not. Same prompt, temp 0, 120 tokens:

    WINDOWS=21  sha (A)  E 2.380      <- production default
    WINDOWS=8   sha (B)  E 2.000
    WINDOWS=4   sha (C)  E 1.595
    WINDOWS=1   GARBAGE  E 1.000      <- see #3

E falls monotonically with `dense_windows`. **E is a CORRECTNESS PROBE here,
not a tuning knob**: the MTP drafter has its own resident weights and never
touches the expert pager (`mtp.rs` has zero pager references), so its proposals
are invariant under geometry. E is the agreement between that fixed drafter and
the VERIFY, and the verify IS the main model -- so E moving means the main
model's logits moved, i.e. the pager fed it wrong weights. Use E as a cheap
correctness signal; the CPU oracle is only needed to say WHICH geometry is right.

**Affects PLAIN DECODE too**, so this is not DSpark-specific and no path is
safe: same prompt, temp 0, no DSpark --

    WINDOWS=21  sha 93317dccd2e6   124 ms/tok   8.07 tok/s
    WINDOWS=4   sha 5c9e906bedcc    92 ms/tok  10.91 tok/s

so the faster geometry is computing a different model. Any throughput number
taken at a non-default WINDOWS is untrustworthy until this is fixed.

Ruled out so far: #2 (`remote_split_on=true` makes `set_remote_exclusion`
rebuild the remap afterwards); union OVERFLOW (errors loudly at
`expert_pager.rs:1068`, never truncates); window RECLAIM (clears `slot_key` and
`slot_of` together, `:990`); LRU/dense REGION overlap (`ensure` skips to
`lru_lo = dense_slots()` and only evicts above it, `:1382`). Root cause NOT yet
found. #2 was the leading suspect and
is ruled out for the default config (`remote_split_on=true` makes
`set_remote_exclusion` rebuild the remap afterwards). Needs `scripts/v41_oracle`
to say which geometry is even correct.
- **Window arithmetic (REFUTED 2026-09-16, `V41_WINDOW_DBG=1`).** Both paths
  were instrumented at the point where they compute the attended window and run
  back-to-back in one process. They agree exactly:
- **ROOT-CAUSED 2026-09-16: the SPARSE VERIFY RESIDENCY path.** One env flag
  moves every metric together, which is the confirmation the KLD warning below
  demands. 71 verify/decode pairs, B=1 probe, otherwise identical config:
- **FIXED 2026-09-16 (056e262). Root cause: `moe_group_builder.hip:118`.**

      g = (dense >= 0) ? e : (-dense - 1);
      if ((unsigned int)g >= n_expert) return;        // n_expert = 384

  `group_count` / `expert_members` are sized N_EXPERT and indexed
  `g * max_per_expert + pos`, so that bound is a BUFFER LIMIT, not a guard. The
  dense/window view produces `g` in [0,384). The sparse verify-residency view
  produces `g` = ABSOLUTE pool slot, and `ExpertPager::ensure` allocates from
  `dense_slots() = (pinned_windows*window_stride + N_EXPERT).min(n_slots)`,
  which is >= 384 for any pool over 384 slots. Every sparse-resident routed
  expert was therefore dropped SILENTLY -- no error, plausible output.

  Fix: `ExpertPager::sparse_group_ids_in_range()` is now part of the
  `sparse_resid_layer` predicate (it cannot be a later check -- once `ensure`
  has run it has written absolute slots into the shared remap, and the window
  view needs the identity map, so there is no safe post-hoc fallback).

  MEASURED, default config, 71 verify/decode pairs:

      before   agree 43/71 (0.6056)  cos 0.759226  kld 1.70697
      after    agree 71/71 (1.0000)  cos 0.998369  kld 0.00658

  Same class as the box-2 `ensure_group_bound` fix (7f89090).

  FOLLOW-UP DONE -- **the sparse view is back ON, widened, not gated off.**
  The gating fix above bought correctness by disabling the view, which cost
  ~2.2x on the probe clock. The group-id space is now sized to the space the
  remap actually encodes:

    * `ExpertPager::sparse_group_bound()` -> the pool (`n_slots`).
    * `BatchIgpuScratch::ensure_group_bound` / `BatchIgpuShared::
      ensure_group_capacity` grow `group_count`, `expert_members` and the three
      work-item arrays to that bound. Grow-only, and callers drain the stream
      first (growing FREES buffers a queued kernel may still be reading).
    * `moe_group_bound` is read next to `sparse_resid_layer`, because by the
      time the builder runs the pager is borrowed by `routed_src`.
    * `max_per_expert` drops from `B_MAX` to the actual `b` in the wide case --
      `expert_members` is a dense `[bound x max_per_expert]` matrix, so a
      4454-slot bound at `B_MAX` would be gigabytes while at b<=6 it is 107 KB
      and fits the existing 384 x B_MAX allocation with no reallocation at all.
      A group holds at most one entry per token (a token's top-k picks are
      distinct experts, and the pager maps distinct experts to distinct slots),
      so `b` is exact, not a heuristic.
    * `MAX_MOE_GROUP_IDS` (65536) is the real ceiling: work items pack
      `(group_id << 16) | member_start`. `sparse_group_ids_in_range()` now
      guards THAT, and falls back to the dense window above it.
      Both builders reject an over-wide bound rather than alias ids.

  MEASURED, default config, widened sparse view vs the known-correct window
  path, back-to-back, same prompt:

      B=1 probe   sparse WIDE   agree 71/71  cos 0.998369  kld 0.00658
      B=6 probe   sparse WIDE   agree 68/71  cos 0.998397  kld 0.00731   691.5 ms
      B=6 probe   window (=0)   agree 68/71  cos 0.998397  kld 0.00731  1500.1 ms

  Bit-identical to the correct path at both batch sizes, and 2.17x faster. The
  server logs `moe group space: WIDE <n> ids x <m> members` once when the wide
  space is live, so "is the sparse view actually on" is not a guess.

  The residual 3/71 at B=6 is PRE-EXISTING and identical in both arms (cos
  0.9984, kld 0.0073 -- argmax ties at near-identical logits), so it is batch
  numerics, not this bug.

  NOT taken: packing the verify's experts into a <N_EXPERT-wide region with its
  own LRU. Widening is strictly less machinery and has no residency policy to
  keep in sync.

      V41_SPARSE_VERIFY_RESIDENCY=0   agree 71/71 (1.0000)  cos 0.998369  kld 0.00658
      V41_SPARSE_VERIFY_RESIDENCY=1   agree 43/71 (0.6056)  cos 0.759226  kld 1.70697

  The =1 arm's 0.6056 reproduces the long-reported "~64% argmax agreement" that
  #0b was tracked by all session, and 1.707 nats is the "2.2 nats" that was
  being hunted separately. They are the same bug.

  Bisected to layer 0, verify vs decode, maxd/scale (bar 5e-2), pos 42/43/44:

      layer input                  0.000e+00   identical
      attention out                6.4e-03     clean
      MoE input                    7.5e-03     clean
      router expert ids            --          IDENTICAL
      router gate weights          3.1e-03     clean
      shared expert out            2.5e-03     clean
      layer OUTPUT                 8.3e-01     WRONG

  Same input, same six experts, same gates, clean shared half => the ROUTED
  expert compute reads the wrong weights under the sparse view.

  NOT the two-box path: all-local (V41_REMOTE_SPLIT=0) reproduces it identically
  (cos 0.583 vs 0.569). NOT batching: this is B=1. NOT windows: bit-identical
  with and without V41_PAGER_WINDOWS=21. NOT attention: the SWA kernels were
  diffed by hand and are arithmetically identical, and the attended slots were
  measured equal at three overlapping positions.

  Still OPEN: the exact defect inside the sparse path. `ensure` DOES write the
  correct negative slot encoding (`-(slot)-1`) that `moe_group_builder.hip:116`
  decodes, so the sign convention is not it. Next suspects: whether the pool
  slot's CONTENTS match the id `ensure` recorded (page-in races with
  `mark_remote_after_ensure`), and whether `pg.routed`'s bytes-per-expert stride
  matches the window view's.

  MITIGATION AVAILABLE NOW: `V41_SPARSE_VERIFY_RESIDENCY=0` gives a verify that
  reproduces decode exactly. It costs the sparse-residency perf win, so it is a
  correctness/throughput trade until the defect above is fixed.

      verify  (prefill row i)  n_raw_before + i + 1 slots, offset 0
      decode                   n_raw = pos + 1 slots,      raw_off  0

  `n_raw_before + i` IS the absolute position, so both resolve to `[0, pos+1)`.
  Measured at pos 33: verify 34 slots, decode 34 slots, both offset 0.
  Layer 0 is `ratio==0` (pure SWA, NO compressed rows), so for the layer #0b is
  bracketed to, those two scalars describe the window COMPLETELY -- there is no
  compressed-row component left to disagree about. The two kernels are reading
  the same KV slots. The divergence is in the kernels or in numerics, not in
  addressing.

### 2. OPEN — `ensure_layer_dense` never writes `self.remap`
`expert_pager.rs:532`. `ensure_layer_union` publishes via `write_window_remap`
("the remap IS the slot table now") but the dense twin does not, yet
`pg.remap_dev` is handed to the dispatch unconditionally
(`forward_prefill.rs`). Reachable with a STALE (previous layer's) remap when no
exclusion builder runs afterwards: the >=90% union->dense arm with
`remote_split_on == false`, and `V41_PAGER_UNION=0`. On the first call
`remap_dev` is raw `hipMalloc` memory. Neither path runs
`verify_routing_exactly_once`. Not active in the default config.

### 3. OPEN — `V41_PAGER_WINDOWS=1` is degenerate and silently accepted
`dense_windows=1` => `pinned_windows=0` => `window_base(w)=0` for EVERY layer,
so all 40 layers share one 384-slot window and overwrite each other. Emits CJK
garbage, E=1.000. Should be rejected at startup. Do NOT use as an "unpacked
reference" -- it is not one.

### 4. OPEN — `remap_dev` carries three incompatible slot spaces, untyped
`ensure` -> ABSOLUTE pool slot (whole-pool view); `ensure_layer_union`/`_dense`
-> WINDOW-RELATIVE (window view); box 2's `LayerShard` -> ABSOLUTE shard slot.
The allocator, the exclusion builder and the WEIGHTS VIEW must be chosen as one
decision; two of three right is still silently wrong. No bound check that
`-r-1 < view.n_slots` (only `set_remote_exclusion` checks its own case).
Fix: `ExpertPlan { view, remap }` from `plan_decode()` / `plan_prefill()`, the
five primitives made private. See memory `project-v41-remap-slot-spaces`.

### 5. OPEN — remote wire mask is applied against the static HELLO bitmap
`remote_experts.rs:3319`. `submit_inner` masks picks by what box 2 STATICALLY
advertises, not by the hub's `owns_eff`, so a pick the hub reassigned to box 2
can be replaced by `NO_PICK` and computed by nobody. Its own comment says
`verify_routing_exactly_once` cannot see it (that validates the HUB's view).
`submit_inner` can also return `Ok(None)` after the local exclusion remap is
already published.

### 6. OPEN — `V41_REMOTE_SPLIT=2` is documented as an arithmetically identical
control, and is not. With `dry=true`, `ids` still skips box-2-owned experts but
`owns_eff` only carries `extra_remote`, so under `mark_remote_after_ensure`
those experts keep `ensure`'s default `-(e)-1` = "ours at pool slot e",
pointing at whatever occupies slot `e`. The exactly-once audit PASSES.

### 7. OPEN — the DRAFTER changes the accepted output at temperature 0
With a correct verify, which tokens are ACCEPTED must not depend on what the
drafter proposed -- the verify decides, the drafter only decides how many are
won per step. Measured 2026-09-16, same prompt/temp 0/120 tokens, changing only
`V41_DSPARK_DENSE_RING`:

    ring off  sha 0a0b617de9cf  E 2.380
    ring on   sha e63bf762b1bd  E 2.553

Output DIFFERS. Consistent with the known "verify is exact only at B<=2" result
(f32 non-associativity between a B=6 batched verify and a B=1 decode), so this
is probably a fidelity limit rather than a logic error -- but it has not been
confirmed against `scripts/v41_oracle`, and until it is, any A/B that changes
the drafter is also changing the answer.

### 8. OPEN — `V41_RESET_ZERO` parks an open correctness question behind a flag
`state.rs`. A 3.9 GB memset that exists to test whether "kernels never read past
`n_comp`" is false. Resolve it or make it a bounds check in the kernels.

### 9. MITIGATED — compressor lend/return pairs are not exception-safe
`engine.rs:840`/`:1000` (28 `?` between) and `forward_prefill.rs:727`/`:783`.
Any `?` in between leaks the store, so every LATER request fails with
"L{src}: missing compressor state" naming the WRONG layer.
`HetModelState::restore_compressor_lending` now repairs ownership at forward
entry -- but NOT state: if the forward failed after a compressor boundary
fired, `n_comp`/`state_kv` are advanced relative to rolled-back KV and the next
request inherits that silently. Real fix is a `CompressorLoan` RAII guard;
`with_kv_source` is the model.

### 15. OPEN — the expert pool is statically partitioned BY PHASE

UPDATE 2026-09-18: still true (384 dense slots reserved for prefill, 4070 for
decode), but the measured stakes are smaller than the table below suggests. On a
pick trace of live traffic the cache is near its ceiling at this capacity: LRU
2.247% miss vs a frequency ORACLE 1.965%, the hash partition exactly matches an
ideal shared pool, and only ~18 GB of host RAM headroom exists on the two boxes
combined. The dominant term is per-miss COST (~13 ms for an 18.8 MB expert at
1.8-2.4 GB/s through dm-crypt), not slot allocation.

`dense_windows` reserves N windows for PREFILL's encoder layers; decode cannot
use them even when idle. At the default (pool 78 GB, WINDOWS=21) that is 2944
of 4454 slots withheld from decode, leaving it 1510. Measured 2026-09-16,
plain decode, per-token:

    1510 LRU slots   219 ms/tok   4.57 tok/s   box2 srv 158.7 ms
    3174 LRU slots   236 ms/tok   4.23 tok/s   box2 srv  84.2 ms
    3686 LRU slots   185 ms/tok   5.40 tok/s   box2 srv  83.9 ms

Giving box 1 more slots HALVES box 2's server time, because the hub's
residency catch-all then keeps the picks locally instead of shipping them --
the max(t_box1, t_box2 + rtt) rebalance. The partition should be dynamic
(phase-aware pool), not a startup constant.

For contrast, box 2 does NOT have this problem: its `--experts` spec and
placement file are only the INITIAL LOAD, after which `enable_paging()` sets
`owned = true` on every layer and runs a shard-wide global LRU with no
per-layer floor. Box 1 is the statically partitioned, non-evicting side (see
#16), not box 2.

NOTE: the windows are NOT id-indexed. `ensure_layer_union` assigns slots
"densely by arrival order, not by expert id", so a window holds the DEMANDED
union. The `slot == id` dense twin refuses on a packed window and is
unreachable at STRIDE<384. Do not repeat the claim that windows pin "each
layer's first 128 ids" -- that was wrong.

### 16. OPEN (CATCH-ALL MODE ONLY) — box 1's decode "LRU" never evicts

SCOPE CORRECTION 2026-09-18: the `budget = pg.lru_free_slots()` gate is inside the
`else if catchall` branch (`forward_layer.rs:2493`). Production runs
`V41_T2_PARTITION=1`, which takes the partition branch instead, where residency is
decided by `partition_box2()` and the pager pages normally. Measured on live
traffic in partition mode: box 1 decode_hit 0.969-0.986 with thousands of misses
and real disk reads per request, and a pick-trace refetch ratio of 7.2 — i.e. it
evicts and re-admits constantly. The freeze below is real for catch-all, not for
the mode we ship.

`forward_layer.rs:2331` takes `budget = pg.lru_free_slots()`, and
`lru_free_slots` (`expert_pager.rs:1538`) counts slots that are **empty**, not
evictable. So under the T2 catch-all box 1 admits a new expert only while the
decode LRU still has VIRGIN slots; once it is full `budget == 0` on every
subsequent call and box 1 never admits another expert for the life of the
process. Whatever arrived first is frozen in.

This makes the in-tree note "a 1396-slot LRU filled that way was SLOWER than a
25-slot one (igpu.routed_moe +89%)" a symptom, not a tuning result: a bigger
frozen cache just locks in a worse set. Contrast box 2, which pages dynamically
across its whole pool with no per-layer floor (V41_B2_POOL_FLOOR=0) and reaches
a 96.9% hit rate.

Note the admission GATE itself is correct and wanted: `victim_cache()` (default
ON, `expert_pager.rs:219`) only admits experts box 2 reported missing
(`box2_missed`), so box 1 is a victim cache for box 2 rather than a duplicate of
its hot set. The bug is the freeze, not the gate.

### 19. ~~FIXED~~ (2026-09-18) — `--ctx 307200` SILENTLY TRUNCATED the indexer to 131,200 positions

**The worst bug found in this engine to date**, because it degraded answers
without a single error, warning or failing test, for a day of live agent use.

`scored_keys_are_gathered()` reads `V41_INDEX_K` **at runtime**. With the
indexer on it reports every compressed layer as gathered, so
`attn_max_scored_keys()` — the sole input to the server's `--ctx` admission
check — collapsed to `SWA_WINDOW + INDEXER_TOP_K` = **640 at any context**. The
check therefore admitted `--ctx 307200` while every comp-indexed scratch buffer
was still sized off the compile-time `ATTN_MIXED_MAX_KEYS` = 131,200.

`n_index_comp` was then `.min(ATTN_MIXED_MAX_KEYS)`-ed at three sites
(`forward_layer.rs` x2, `forward_prefill.rs` x1), each carrying a comment
asserting production could not reach it. So past 131,200 tokens the indexer
scored only the FIRST 131,200 compressed positions, and **every later token was
invisible to all 38 compressed layers** — reachable only through the 128-token
raw sliding window. Layers 0-1 are sliding-window-only. The model saw the head
of the conversation, a hole, and the last 128 tokens.

Presented to the user as "it's like it isn't listening to me, sometimes — like
it sees my messages out of order."

**Why no test caught it.** Every cap test routes through the env-dependent
`attn_max_scored_keys`, and `V41_INDEX_K` is unset under `cargo test` — the one
regime where the old bound is correct. `decode_cap_admits_a_real_context`
asserted `ctx == 131_072` and passed, in a process configured unlike production.

**Introduced by `3409948`** (2026-09-18), which changed the gather predicate and
raised the default `--ctx` to 307200 in the same commit. The ceiling that had
been refusing `--ctx > 131_072` was load-bearing and its removal was not noticed
because it was never the stated subject of the change.

**Fixed:** `ATTN_MIXED_MAX_KEYS` raised to
`V41_MAX_CTX + IMAGE_RAW_WINDOW_MAX` (+~550 MB dGPU at `lane_rows` 512) — NOT
`+ SWA_WINDOW`, which was the first attempt: `engine_worker` passes the wider
vision raw window whenever `--mmproj` is set, which production does, so the cap
came out 384 keys short and the server refused to start. The first version of
`cap_covers_the_shipped_ctx` made the same substitution and passed anyway; it
now takes `IMAGE_RAW_WINDOW_MAX`. Same class of error as the bug itself — a
bound computed in a configuration unlike production. Also new:
`indexer_max_scored_keys()` with no
gathered shortcut and no env dependence, which the admission check now takes the
max against; all three truncating `.min()`s are hard errors; two regression
tests, one verified to fail at the old cap.

**Follow-up (not done):** the indexer buffers are still strided by the constant
rather than by `n_kv_max`, so a small `--ctx` pays the 307K footprint.
`alloc_rows_ctx` already carries `n_kv_max`; tightening means threading the
stride to ~10 launch sites that currently pass `ATTN_MIXED_MAX_KEYS`, and a
missed site is silent corruption. Do it with the stride as a struct field, once.

**How to apply:** a runtime flag must never be an input to a bound that sizes a
buffer allocated once. And a `.min()` on a quantity an admission check is
supposed to guarantee is not defensive — it converts a loud failure into a
silent one, and deletes the evidence.

## Structural / ergonomic

### 10. OPEN — two sources of truth for the current HIP device
`DeviceGuard` is correct and `pub(crate)` (invisible to the crate that needs
it); the engine keeps a `current_device: AtomicI32` mirroring a THREAD-LOCAL
HIP property. 91 `set_current*` calls in `het/`, zero guards. One site is
hand-patched with a comment explaining the cache goes stale.

### 11. OPEN — env-var sprawl: 292 `std::env::var` reads in production `src/`
113 distinct names in `het/` alone. 21 per layer in the prefill hot path
(~840/step) -- measured at ~42 us/layer, i.e. NOT a bottleneck, but several
select different NUMERICS from a free integer rather than a validated enum.
Should be one `EngineConfig` parsed once, with a `validate()` rejecting the
contradictory combinations, printed in the startup banner and in run
fingerprints.

### 12. OPEN — `KvMark.slid: bool` changes the meaning of `per_layer` and
disables the wrap check in `rollback_kv`. Two mark kinds in one struct; should
be an enum so `rollback_kv` cannot be handed a mark whose provenance it must
trust.

### 13. OPEN — doc-comment drift attaches `///` blocks to the WRONG item
Verified at `expert_pager.rs:154`, `:257`, `:1188`, `forward_prefill.rs:61`,
`:81`, `state.rs:425`. In a codebase where comments ARE the invariant
documentation, a comment on the wrong item is worse than none.

### 14. OPEN — `CompKvStore::fp8_enabled()`/`e2m1_enabled()` read env at every
allocation, so one process can hold layers that disagree on storage format.

---

## Fixed 2026-09-16 (kept briefly for cross-reference)

- ~~sparse verify allocator paired with `set_remote_exclusion`~~ -> picks
  computed by nobody; `86808af`, tests in `v41_sparse_verify_remap_pairing.rs`
- ~~sparse verify used `routed_window` with ABSOLUTE slots~~ -> silently read
  another expert's weights; worth E 1.804 -> 2.261; `dea8525`
- ~~chat "extend" path never called `normalize_raw_windows`~~ -> after any turn
  over 128 tokens the suffix prefill overwrote live KV; `76ce146`
- ~~`V41_B2_POOL_FLOOR` defaulted to 0.90~~ against its own doc-comment's
  measurement; `148044b`
