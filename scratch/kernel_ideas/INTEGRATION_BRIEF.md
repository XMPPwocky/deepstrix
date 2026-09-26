# Integration brief — kernel-ideas sweep 2026-09-26

You are integrating the CONFIRMED kernel wins of ONE family from `scratch/kernel_ideas/LEDGER.md`
into the production tree of this worktree (`/home/claude-code/deepstrix/.claude/worktrees/kernel-ideas-2026-09-26`,
branch `worktree-kernel-ideas-2026-09-26`, base 361d4f9 = the deployed hub). The owner's decision:
**everything confirmed goes in at once, behind per-item knobs that default ON, and ONE fidelity gate
runs on the combined build** (bisect by knob only if it fails). So: wire faithfully, keep each item
independently switchable, do not run fidelity gates yourself.

## Sources of truth, in order
1. `scratch/kernel_ideas/LEDGER.md` — your family's section: for each idea the candidate files, the
   engineer's integration notes, and the REVIEWER's `merge notes` / `issues` (gates such as "b=2 must
   stay on the old kernel", "add tB3", "decode path only"). Those caveats are REQUIREMENTS.
2. `scratch/kernel_ideas/<family>/NOTES.md` (often has an "integration recipe") and the candidate
   `.hip` files under `scratch/kernel_ideas/<family>/`. The reviewer's rebuilt copies under
   `<family>/review/` are the same code; the engineer's files are canonical.
3. `scratch/kernel_ideas/INVENTORY.md` — where each production kernel is launched (wrapper file:line,
   call sites in `crates/v4flash-kernels/src/het/forward_prefill.rs` = FP, `remote_experts.rs` = RE,
   `dispatch.rs`), grid/args, and the knob conventions.

## Conventions (match the surrounding code exactly)
- Kernel sources live in `crates/v4flash-kernels/kernels/*.hip`; `build.rs` compiles every `.hip`
  for gfx1201 + gfx1151 with `-O3 --genco -DDEEPSTRIX_V41=1 -DMHC_N_EMBD=5120 -DMHC_HC_DIM=20480
  -DROUTER_MAX_EXPERTS=512` and exposes `KERNEL_<STEM>_<ARCH>` env vars; Rust wrappers load symbols
  by name (see how the sibling kernel of your target is wired: module load, `launch`, arg packing).
  Put a new kernel next to the production one it replaces (same file, or a new file with the same
  naming style); keep the candidate's symbol name unless it collides.
- Knobs: `static X: LazyLock<bool> = LazyLock::new(|| std::env::var("V41_FOO").as_deref() != Ok("0"))`
  with a doc comment `/// \`V41_FOO\` (default ON; \`0\` = the previous kernel): …` — exactly like
  `V41_MHC_FAST` / `V41_ATTN_DEC_SCORE` in FP:814-862. One knob per ledger item (or per tightly
  coupled pair). Default ON. `=0` must restore the exact previous path.
- Shape gates from the reviewer go in the wrapper (e.g. `if b >= 4 && *GATHER_B128`), not in the kernel.
- Keep the production launch geometry the candidate was measured with (the ledger repro/notes state it).
- Comments: say WHAT the kernel does and the measured number + date in one line
  (`// 2026-09-26 sweep: 1.97x at 235K b=4, bit-exact (E_indexer/score_qreg_hw)`), plus any
  non-obvious mechanism (e.g. the quantize `blocks + 1` grid: the reviewer measured the dispatch
  itself at 14 us on a power-of-two grid — write that down or someone will "fix" it).
- Do NOT touch anything outside your family's kernels/wrappers/call sites. Do not reformat files.
  Do not change defaults of existing knobs. Do not delete the old kernels (the `=0` path needs them).

## Build and test (GPUs are free: the production hub is DOWN; there is no other GPU user but you)
- Build: `bash scratch/kernel_ideas/_infra/build.sh -p v4flash-kernels` (kernels crate; fast) and at
  the end `bash scratch/kernel_ideas/_infra/build.sh` (server + expertd). Uses the sweep's own
  `CARGO_TARGET_DIR=/home/claude-code/deepstrix/target-v41-ki`; never the production `target-v41`.
- Tests: run the existing oracle / bit-exact tests that cover your kernels (INVENTORY.md §4 lists them
  per family; `crates/v4flash-kernels/tests/`). Use
  `CARGO_TARGET_DIR=/home/claude-code/deepstrix/target-v41-ki nix develop -c cargo test --release
  --features v41 -p v4flash-kernels --test <name> -- --nocapture` (add `--ignored` if the test is
  ignored). Tests that load the 86 GB model are allowed if the test already exists and you need it
  (one load = ~1 min); do not write new model-loading tests. If a test for the OLD path exists,
  run it with your knob `=0` too. Bit-exact items must stay bit-exact; if one is not, STOP and report,
  do not "fix" the numerics.
- Quick sanity for a wired kernel without a test: a small `#[test]`/bench harness is fine, but the
  sweep harnesses in `scratch/kernel_ideas/<family>/` already have correctness checks against the
  production kernel — you may point them at the in-tree `.hip` you just edited (rebuild the hsaco
  with `_infra/kcc.sh`) to confirm the in-tree copy still matches the candidate bit-for-bit.
- GPU runs in the scratch harnesses go through `scratch/kernel_ideas/_infra/gpu_run.sh` (see
  `_infra/README.md`); cargo tests may run directly.

## Deliverable
- Commit on this branch when your family builds and its tests pass:
  `git add <files>` then `git -c user.name=Mimir commit -m "perf(v41): <family> -- <items> (sweep 2026-09-26)\n\n<one line per item: what, measured x, bit-exact?, knob>\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"`.
  Do not push. Do not touch other branches or the stash.
- Return (as your final message, plain text): items integrated with their knob names; items skipped
  and why; tests run with results; anything the combined gate must watch for (non-bit-exact items,
  shape gates); any deviation from the ledger's recipe.
