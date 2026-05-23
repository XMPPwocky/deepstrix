#!/usr/bin/env bash
# Apply deepstrix-local patches to the ds4 submodule.
# Idempotent: re-running is a no-op if patches already applied.
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
    echo "  [apply] $(basename "$patch")"
    (cd ds4 && git apply "$rel_patch")
done

echo "external/apply-patches.sh: done"
