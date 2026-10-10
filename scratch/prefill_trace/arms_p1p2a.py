#!/usr/bin/env python3
"""P1/P2a live arms + statistical Gate B (docs/v41/P1_P2A_GPU_RUNBOOK.md sec 2) against the LIVE hub. No restart.

The hub must run the gate binary with V41_PREFILL_LOGITS_DUMP=<DUMP>. One fixed long prompt (a single user message, no system
block) is sent once per arm with max_tokens 1. Identical prompts would restore the previous arm's whole-prompt snapshot and skip
prefill, so after each run this moves THIS prompt's snapshot dirs (the full save + its 32K checkpoints, matched by tokens.bin
prefix) out of the snapshot root: the next restore of that hash fails, is evicted ("restore failed; evicting") and the request
prefills cold. Arms are knob-file edits (/dev/shm/deepstrix-knobs.txt, re-read every second; applied at the next window / job).

  arms_p1p2a.py [--arms a0,a0n,lo,split,split_lo,a0e] [--chars 260000]     (redirect stdout to a file)
"""
import argparse, json, math, os, re, shutil, struct, subprocess, sys, time, urllib.request

URL = "http://127.0.0.1:18080"
KNOBS = "/dev/shm/deepstrix-knobs.txt"
DUMP = "/home/claude-code/logs/prefill_logits_20261010.bin"
SNAP = "/home/claude-code/.cache/deepstrix/snapshots-v41"
TRASH = "/home/claude-code/.cache/deepstrix/snapshots-v41-gateb-moved"
WT = "/home/claude-code/deepstrix/.claude/worktrees/igpu-wmma-moe"
OUT = "/home/claude-code/gateb_20261010"
NV = 129280
REC = NV * 4
B2 = "10.43.0.95"
OURS = ("V41_LM_LAYER_OUTER", "V41_PREFILL_SPLIT")
ARMS = {
    "a0": {}, "a0n": {}, "lo": {"V41_LM_LAYER_OUTER": "1"},
    "split": {"V41_PREFILL_SPLIT": "1"}, "split_lo": {"V41_PREFILL_SPLIT": "1", "V41_LM_LAYER_OUTER": "1"}, "a0e": {},
}


def log(*a):
    print(time.strftime("%H:%M:%S", time.gmtime()), *a, flush=True)


def build_prompt(chars):
    """Deterministic: repo docs at a FIXED commit, sorted, concatenated until `chars`."""
    commit = "b8323821"
    names = subprocess.run(["git", "-C", WT, "ls-tree", "--name-only", "-r", commit, "docs/"], capture_output=True, text=True,
                           check=True).stdout.split()
    parts, n = [], 0
    for p in sorted(x for x in names if x.endswith(".md")):
        t = subprocess.run(["git", "-C", WT, "show", f"{commit}:{p}"], capture_output=True, text=True, check=True).stdout
        parts.append(f"\n\n===== {p} =====\n\n{t}")
        n += len(parts[-1])
        if n >= chars:
            break
    body = "".join(parts)[:chars]
    return ("Below is a large set of engineering notes. Read all of it, then answer with one word: which hardware component do "
            "the notes say is the bottleneck most often?\n" + body + "\n\nOne word answer:")


def set_knobs(want):
    with open(KNOBS) as f:
        lines = [l for l in f.read().splitlines() if not l.split("=", 1)[0].strip() in OURS]
    lines += [f"{k}={v}" for k, v in want.items()]
    tmp = KNOBS + ".gateb.tmp"
    with open(tmp, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, KNOBS)
    time.sleep(3)  # re-read every second
    eff = open(KNOBS + ".effective").read() if os.path.exists(KNOBS + ".effective") else ""
    for k in OURS:
        m = re.search(rf"^{k}\s*=\s*(\S+)", eff, re.M)
        log(f"  effective {k} = {m.group(1) if m else '?'} (want {want.get(k, 'default')})")


def b2_miss_par(v):
    cmd = f"sed -i 's/^miss_par=.*/miss_par={v}/' ~/expertd-knobs.txt && grep '^miss_par' ~/expertd-knobs.txt"
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", B2, cmd], capture_output=True, text=True)
    log(f"  box 2 {r.stdout.strip()} rc={r.returncode} {r.stderr.strip()[:200]}")


def chat(prompt):
    models = json.loads(urllib.request.urlopen(URL + "/v1/models", timeout=30).read())
    mid = [m["id"] for m in models["data"] if "embedding" not in m["id"]][0]
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps({
        "model": mid, "max_tokens": 1, "temperature": 0, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json"})
    t = time.time()
    r = json.loads(urllib.request.urlopen(req, timeout=3600).read())
    return time.time() - t, r.get("usage", {}), (r["choices"][0]["message"].get("content") or "")[:40]


def tokens_of(d):
    with open(os.path.join(d, "tokens.bin"), "rb") as f:
        return f.read()


def move_my_snapshots(t0, n_prompt, arm):
    """Move the full-prompt save (newest dir since t0 with token_count in [n-2, n]) and every dir since t0 whose tokens.bin is a
    prefix of it. Returns the moved names."""
    new = []
    for name in os.listdir(SNAP):
        d = os.path.join(SNAP, name)
        try:
            meta = json.load(open(os.path.join(d, "meta.json")))
        except Exception:
            continue
        if meta.get("created_at_unix", 0) >= t0 - 1:
            new.append((name, d, meta["token_count"]))
    full = [x for x in new if n_prompt - 3 <= x[2] <= n_prompt]
    if len(full) != 1:
        log(f"  WARNING: {len(full)} full-prompt snapshot candidates since t0 (new={[(x[0][:12], x[2]) for x in new]}); NOT moving")
        return []
    ft = tokens_of(full[0][1])
    os.makedirs(TRASH, exist_ok=True)
    moved = []
    for name, d, n in new:
        if ft.startswith(tokens_of(d)):
            dst = os.path.join(TRASH, f"{name}.{arm}")
            assert not os.path.exists(dst), dst
            os.rename(d, dst)
            moved.append((name[:12], n))
    log(f"  moved my snapshots: {moved}")
    return moved


def kld(p_logits, q_logits):
    def lsm(x):
        m = max(x)
        s = math.log(sum(math.exp(v - m) for v in x)) + m
        return [v - s for v in x]
    lp, lq = lsm(p_logits), lsm(q_logits)
    return sum(math.exp(a) * (a - b) for a, b in zip(lp, lq))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="a0,a0n,lo,split,split_lo,a0e")
    ap.add_argument("--chars", type=int, default=260000)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    prompt = build_prompt(a.chars)
    log(f"prompt {len(prompt)} chars")
    recs = {}
    sp = os.path.join(OUT, "summary.json")
    summary = json.load(open(sp))["runs"] if os.path.exists(sp) else []
    for arm in a.arms.split(","):
        if arm == "split":
            b2_miss_par(1)  # owner: revert miss_par before the P1 arm
        log(f"== arm {arm}: {ARMS[arm]}")
        set_knobs(ARMS[arm])
        sz0 = os.path.getsize(DUMP) if os.path.exists(DUMP) else 0
        t0 = time.time()
        dt, usage, text = chat(prompt)
        sz1 = os.path.getsize(DUMP)
        n_new = (sz1 - sz0) // REC
        log(f"  {arm}: {dt:.1f} s wall, usage {usage}, reply {text!r}, dump +{n_new} record(s)")
        rec = None
        if n_new == 1 and (sz1 - sz0) % REC == 0:
            with open(DUMP, "rb") as f:
                f.seek(sz0)
                rec = list(struct.unpack(f"<{NV}f", f.read(REC)))
            with open(os.path.join(OUT, f"{arm}.f32"), "wb") as f:
                f.write(struct.pack(f"<{NV}f", *rec))
            recs[arm] = rec
        else:
            log("  AMBIGUOUS dump (other traffic prefilled meanwhile): this arm has no logits")
        moved = move_my_snapshots(t0, usage.get("prompt_tokens", 0), arm)
        summary.append({"arm": arm, "knobs": ARMS[arm], "t0": t0, "wall_s": dt, "usage": usage, "dump_records": n_new,
                        "moved": moved})
        json.dump({"runs": summary}, open(sp, "w"), indent=1)
        time.sleep(5)
    set_knobs({})
    log("knobs restored (our keys removed); box 2 miss_par left at 1")
    # Gate B: KLD vs the arm-0 null
    res = {}
    for fn in sorted(os.listdir(OUT)):
        if fn.endswith(".f32") and fn[:-4] not in recs:
            with open(os.path.join(OUT, fn), "rb") as f:
                recs[fn[:-4]] = list(struct.unpack(f"<{NV}f", f.read(REC)))
    ref = recs.get("a0")
    if ref:
        for arm, r in recs.items():
            if arm == "a0":
                continue
            am = max(range(NV), key=lambda i: r[i])
            res[arm] = {"kld_vs_a0": kld(ref, r), "argmax": am, "argmax_a0": max(range(NV), key=lambda i: ref[i])}
            log(f"  KLD(a0 || {arm}) = {res[arm]['kld_vs_a0']:.3e}  argmax {am} (a0 {res[arm]['argmax_a0']})")
    json.dump({"runs": summary, "gate_b": res}, open(sp, "w"), indent=1)
    log(f"summary -> {OUT}/summary.json")


if __name__ == "__main__":
    main()
