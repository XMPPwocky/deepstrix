#!/usr/bin/env bash
# Apply deepstrix-local patches to the ds4 submodule.
# Idempotent: re-running is a no-op if patches already applied.
#
# STATE NOTE (2026-08): the pinned submodule commit (d1b9565 "ds4:
# dump_residual driver + activation-callback hooks") already BAKES IN the
# content of patches 0001-0009 — with one drift: 0006's comp_allowed_mask
# hunk was later reworked (heap alloc replacing the 1024-entry stack cap)
# without recommitting, so on a fresh checkout 0002-0009 fail BOTH the
# reverse-check (stack hunks interleave) and the forward apply (content
# present). On such a tree, run with DS4_SKIP_BAKED=1 to warn-and-continue
# past them so later patches (0010+, generated against the baked state +
# the 0006 heap-fix hunk) still apply. Do NOT use DS4_SKIP_BAKED on a
# pristine upstream ds4 — a real apply failure would be masked.
#
# Usage: external/apply-patches.sh
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -d ds4 ]]; then
    echo "external/apply-patches.sh: ds4 submodule missing — run 'git submodule update --init' first" >&2
    exit 1
fi

for patch in ds4-patches/*.patch; do
    [[ -e "$patch" ]] || continue
    rel_patch="$(realpath --relative-to=ds4 "$patch")"
    if (cd ds4 && git apply --reverse --check "$rel_patch") 2>/dev/null; then
        echo "  [skip] $(basename "$patch") already applied"
        continue
    fi
    if ! (cd ds4 && git apply --check "$rel_patch") 2>/dev/null; then
        if [[ "${DS4_SKIP_BAKED:-}" == "1" ]]; then
            echo "  [WARN] $(basename "$patch") applies in neither direction — assuming baked into the submodule commit, SKIPPING" >&2
            continue
        fi
        echo "  [FAIL] $(basename "$patch") applies in neither direction." >&2
        echo "         If this is the pinned d1b9565 submodule (patches 0001-0009 baked in)," >&2
        echo "         re-run with DS4_SKIP_BAKED=1. See the state note in this script." >&2
        exit 1
    fi
    echo "  [apply] $(basename "$patch")"
    (cd ds4 && git apply "$rel_patch")
done

echo "external/apply-patches.sh: done"
