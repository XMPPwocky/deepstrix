# M51 — iGPU MoE FFN prefill optimization journal

Goal: cold-cache prefill 230 → 300+ tok/s at T=4K-32K (floor; real target = HW max,
realistic dp4a ceiling estimated ~445). Plan: `~/.claude/plans/hello-i-d-like-to-virtual-lark.md`.
Method: profile-first (ablations + ISA + PMC; ATT attempt bounded), then
S0 staged micro-fixes → S1 k-widened iq2 → S2 k-widened q2k. Every experiment
gets a journal entry here, committed BEFORE the run.

Baseline HEAD: `0959f65` (230 tok/s cold T=4K-32K per project_current_state).

---

## 2026-06-09 — Phase 0a: occupancy baseline (static)

`nix develop -c hipcc --offload-arch=gfx1151 -O3 -c -Rpass-analysis=kernel-resource-usage`:

| kernel | VGPR | SGPR | LDS B/WG | scratch | spills | waves/SIMD |
|---|---|---|---|---|---|---|
| iq2 `_chunked_staged` (prod) | 97 | 70 | 18944 | 0 | 0 | **12** |
| iq2 `_tile8_row32` (opt-in) | 192 | 56 | 3072 | 424 B/lane | **309 VGPR!** | 8 |
| q2k `_by_expert` (prod) | 70 | — | 0 | 0 | 0 | **16** (full) |

Findings:
- staged occupancy (12) is LDS-bound at 18.9 KiB/WG → S0b's −8.5 KiB LDS may
  raise occupancy directly.
- tile8 spills 309 VGPRs to scratch — a second reason (beyond the 4×-fewer-WGs
  grid) for its cold-cache regression.
- q2k by_expert is at FULL occupancy with zero LDS — its 2.6×-over-roofline is
  pure per-member re-unpack VALU overhead, not occupancy. Supports the S2 design.

Gate for all new variants: waves/SIMD ≥ 12 (iq2) / 16 (q2k), zero spills.

## Next: Phase 0c ISA audit of staged member loop (v_mul_lo? VOPD? dead LDS round-trip?)
