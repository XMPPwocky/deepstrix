#!/usr/bin/env python3
"""Knob-off identity at the ISA level: every kernel symbol that exists in BASE's build of a
kernels/*.hip file must compile to the SAME instructions in the working tree's build.

For a change that only ADDS kernels / template instantiations (e.g. `V41_ATTN_O_FP8`'s `_o8`
twins), this proves the default-off path runs bit-identical code without a GPU: same instruction
stream per pre-existing symbol (addresses / branch offsets normalised away).

Compares the device assembly (`hipcc -S --cuda-device-only`, the same -O3 / model flags as build.rs),
per kernel, local labels normalised. Usage (inside `nix develop`, or with HIPCC and the ROCm env set):
  python3 scripts/isa_diff_kernels.py BASE_REV file.hip [file.hip ...] [--arch gfx1201 --arch gfx1151]
The base tree's kernels/ directory (all .hip and .inc files at BASE_REV) is materialised in a temp
dir so its #includes resolve to the base versions.
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KDIR = "crates/v4flash-kernels/kernels"
MODEL_FLAGS = ["-DDEEPSTRIX_V41=1", "-DMHC_N_EMBD=5120", "-DMHC_HC_DIM=20480", "-DROUTER_MAX_EXPERTS=512"]


def sh(cmd, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if r.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{r.stderr[-4000:]}")
    return r.stdout


def materialise_base(rev, dst):
    names = sh(["git", "-C", ROOT, "ls-tree", "--name-only", f"{rev}:{KDIR}"]).split()
    for n in names:
        if n.endswith((".hip", ".inc", ".h")):
            data = subprocess.run(["git", "-C", ROOT, "show", f"{rev}:{KDIR}/{n}"], capture_output=True).stdout
            open(os.path.join(dst, n), "wb").write(data)


def kernels_of(asm_path):
    """{symbol: [normalised instruction lines]} from the device assembly (-S): the lines between a
    kernel's `sym:` label and its `.Lfunc_end`, with local labels and comments normalised away."""
    out, cur = {}, None
    for line in open(asm_path).read().splitlines():
        m = re.match(r"^([A-Za-z_][\w$.]*):\s*(;.*)?$", line)
        if m and not m.group(1).startswith("."):
            cur = m.group(1)
            out[cur] = []
            continue
        if cur is None:
            continue
        if line.startswith(".Lfunc_end"):
            cur = None
            continue
        ins = re.sub(r";.*$", "", line).strip()
        if not ins or ins.startswith("."):          # directives, local labels
            continue
        ins = re.sub(r"\.LBB\d+_\d+", ".LBB", ins)
        ins = re.sub(r"\.Ltmp\d+", ".Ltmp", ins)
        out[cur].append(ins)
    # the kernel descriptors (register / LDS / scratch counts): `.amdhsa_kernel sym` ... `.end_amdhsa_kernel`
    desc, cur = {}, None
    for line in open(asm_path).read().splitlines():
        t = line.strip()
        if t.startswith(".amdhsa_kernel "):
            cur = t.split()[1]
            desc[cur] = []
        elif t.startswith(".end_amdhsa_kernel"):
            cur = None
        elif cur is not None and t:
            desc[cur].append(t)
    for k, d in desc.items():
        out.setdefault(k, []).extend(["DESC " + x for x in d])
    # per-TU metadata pseudo-symbols, not kernels
    return {k: v for k, v in out.items() if v and not k.startswith(("__hip_cuid", "amdhsa."))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--arch", action="append")
    a = ap.parse_args()
    archs = a.arch or ["gfx1201", "gfx1151"]
    hipcc = os.environ.get("HIPCC", "hipcc")
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        bdir = os.path.join(tmp, "base")
        os.makedirs(bdir)
        materialise_base(a.base, bdir)
        for f in a.files:
            name = os.path.basename(f)
            for arch in archs:
                objs = {}
                for tag, src in (("base", os.path.join(bdir, name)), ("head", os.path.join(ROOT, KDIR, name))):
                    o = os.path.join(tmp, f"{tag}_{name}_{arch}.s")
                    sh([hipcc, "-O3", f"--offload-arch={arch}", "--cuda-device-only", "-S", *MODEL_FLAGS, src, "-o", o])
                    objs[tag] = kernels_of(o)
                base, head = objs["base"], objs["head"]
                same = [k for k in base if head.get(k) == base[k]]
                diff = [k for k in base if k in head and head[k] != base[k]]
                gone = [k for k in base if k not in head]
                new = [k for k in head if k not in base]
                print(f"{name} @ {arch}: {len(same)}/{len(base)} pre-existing symbols identical"
                      f"{'' if not new else f', {len(new)} new'}")
                for k in diff:
                    print(f"  DIFFERENT: {k} ({len(base[k])} -> {len(head[k])} instructions)")
                for k in gone:
                    print(f"  MISSING:   {k}")
                bad += len(diff) + len(gone)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
