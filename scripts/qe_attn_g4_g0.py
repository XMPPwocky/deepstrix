#!/usr/bin/env python3
"""G0a/G0b for qe_attn_g4 (docs/v41/EMBED_PHASE_DESIGN.md §19.11). Offline: reads the
.log (-Rpass-analysis remarks), .notes (llvm-readelf --notes) and .s (llvm-objdump -d)
that scripts/qe_attn_g4_g0.sh wrote into OUT. Exit 1 on any violation.

usage: qe_attn_g4_g0.py OUT
"""
import collections
import os
import re
import sys

OUT = sys.argv[1]
FAILS = []


def check(cond, msg):
    print(("  PASS " if cond else "  FAIL ") + msg)
    if not cond:
        FAILS.append(msg)


def remarks(path):
    """function -> {remark key: value} from hipcc -Rpass-analysis=kernel-resource-usage."""
    res, cur = {}, None
    for line in open(path, errors="replace"):
        m = re.search(r"remark: Function Name: (\S+)", line)
        if m:
            cur = m.group(1)
            res[cur] = {}
            continue
        m = re.search(r"remark:\s+(.+?): (\S+) \[-Rpass", line)
        if m and cur:
            res[cur][re.sub(r"\s*\[.*?\]", "", m.group(1)).strip()] = m.group(2)
    return res


def notes(path):
    """kernel name -> {metadata key: value} (kernel-level keys only)."""
    recs, cur = {}, None
    for line in open(path, errors="replace"):
        if re.match(r"^  - \.", line):
            cur = {}
        m = re.match(r"^  (?:- |  )\.(\w+):\s+(.*?)\s*$", line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2)
            if m.group(1) == "name":
                recs[m.group(2)] = cur
    return recs


def disasm(path, kern):
    """[(addr, op, args)] of kernel `kern`, and the function base address."""
    ins, base, cur = [], None, None
    targets = {}
    for line in open(path, errors="replace"):
        m = re.match(r"^([0-9a-f]+) <(\S+)>:", line)
        if m:
            cur = m.group(2)
            if cur == kern:
                base = int(m.group(1), 16)
            continue
        if cur != kern:
            continue
        m = re.match(r"^\s+(\S+)\s*(.*?)\s*//\s*([0-9A-Fa-f]+):", line)
        if not m:
            continue
        a = int(m.group(3), 16)
        ins.append((a, m.group(1), m.group(2)))
        t = re.search(r"<\S+\+0x([0-9a-f]+)>", line)
        if t and (m.group(1).startswith("s_cbranch") or m.group(1) == "s_branch"):
            targets[a] = base + int(t.group(1), 16)
    return ins, targets


TRANS = ("v_exp_", "v_log_", "v_rcp_", "v_rsq_", "v_sqrt_", "v_sin_", "v_cos_")


def is_valu(op):
    return op.startswith("v_") and not op.startswith("v_wmma") and not op.startswith(TRANS)


def cfg(ins, targets):
    """Basic blocks [(first, last)] over instruction indices, and successor lists."""
    idx = {a: i for i, (a, _, _) in enumerate(ins)}
    leaders = {0}
    for a, t in targets.items():
        if t in idx:
            leaders.add(idx[t])
        leaders.add(idx[a] + 1)
    for i, (_, op, _) in enumerate(ins):
        if op in ("s_endpgm", "s_setpc_b64"):
            leaders.add(i + 1)
    leaders = sorted(x for x in leaders if x < len(ins))
    blocks = [(s, (leaders[j + 1] - 1) if j + 1 < len(leaders) else len(ins) - 1) for j, s in enumerate(leaders)]
    start = {s: b for b, (s, _) in enumerate(blocks)}
    succ = []
    for b, (s, e) in enumerate(blocks):
        a, op, _ = ins[e]
        out = []
        if a in targets and targets[a] in idx:
            out.append(start[idx[targets[a]]])
        if op not in ("s_branch", "s_endpgm", "s_setpc_b64") and b + 1 < len(blocks):
            out.append(b + 1)
        succ.append(out)
    return blocks, succ


def dominators(n, succ):
    pred = [[] for _ in range(n)]
    for b, out in enumerate(succ):
        for s in out:
            pred[s].append(b)
    full = set(range(n))
    dom = [full.copy() for _ in range(n)]
    dom[0] = {0}
    changed = True
    while changed:
        changed = False
        for b in range(1, n):
            ps = [dom[p] for p in pred[b]]
            new = (set.intersection(*ps) if ps else set()) | {b}
            if new != dom[b]:
                dom[b], changed = new, True
    return dom, pred


def ledger(sfile, kern, cpt, nw16):
    print(f"G0b {kern} ({os.path.basename(sfile)})")
    ins, targets = disasm(sfile, kern)
    check(len(ins) > 0, f"{kern}: found in the disassembly")
    if not ins:
        return None
    allblocks, succ = cfg(ins, targets)
    dom, pred = dominators(len(allblocks), succ)
    # Natural loops (back edge u -> h, h dominates u); the tile loop = the one with the
    # most WMMAs. The compiler places some loop blocks out of line (after the latch),
    # so the loop is a block SET, not an address range.
    loops = collections.defaultdict(set)
    latches = collections.defaultdict(set)
    for u, out in enumerate(succ):
        for h in out:
            if h in dom[u]:
                body, stack = {h, u}, [u]
                while stack:
                    x = stack.pop()
                    for p in pred[x]:
                        if p not in body:
                            body.add(p)
                            stack.append(p)
                loops[h] |= body
                latches[h].add(u)

    def nwmma(bs):
        return sum(1 for b in bs for k in range(allblocks[b][0], allblocks[b][1] + 1) if ins[k][1].startswith("v_wmma"))

    best = max(loops, key=lambda h: nwmma(loops[h]), default=None)
    check(best is not None and nwmma(loops[best]) > 0, f"{kern}: a natural loop with WMMAs (the tile loop)")
    if best is None:
        return None
    header, body = best, loops[best]
    blocks = [allblocks[b] for b in sorted(body)]

    def kind(s, e):
        ops = [ins[k] for k in range(s, e + 1)]
        text = " ".join(op + " " + args for _, op, args in ops)
        n_mul = sum(1 for _, op, _ in ops if op.startswith("v_mul_f32")) + 2 * sum(1 for _, op, _ in ops if op == "v_dual_mul_f32") \
            + 2 * sum(1 for _, op, _ in ops if op.startswith("v_pk_mul_f32"))
        n_exp = sum(1 for _, op, _ in ops if op.startswith("v_exp_f32"))
        if "0xffff0000" in text:
            return "edge"        # edge-tile zero-fill of the staged K/V (<= 2 tiles per WG)
        if "0xff800000" in text and any(op.startswith(("v_cndmask", "v_dual_cndmask")) for _, op, _ in ops):
            return "mask"        # diagonal / segment-start tile
        if n_exp >= 1 and n_mul >= 32:
            return "rescale"     # exact rescale (only when a column max rose)
        return "steady"

    bkind = {b: kind(*allblocks[b]) for b in body}
    kinds = [(allblocks[b][0], allblocks[b][1], bkind[b]) for b in sorted(body)]
    # The steady path (interior tile: stores, t+2 loads, compute, no mask / rescale /
    # edge fill) = the longest header -> latch path through non-side blocks of the loop
    # DAG (back edges into the header removed), in EXECUTION order.
    memo = {}

    def longest(b):
        if b in memo:
            return memo[b]
        w = allblocks[b][1] - allblocks[b][0] + 1
        best_tail = None
        for s in succ[b]:
            if s == header or s not in body or bkind[s] != "steady":
                continue
            t = longest(s)
            if t is not None and (best_tail is None or t[0] > best_tail[0]):
                best_tail = t
        if best_tail is None:
            memo[b] = (w, [b]) if b in latches[header] else None
        else:
            memo[b] = (w + best_tail[0], [b] + best_tail[1])
        return memo[b]

    path = longest(header)
    check(path is not None, f"{kern}: a steady header -> latch path")
    if path is None:
        return None
    steady = [k for b in path[1] for k in range(allblocks[b][0], allblocks[b][1] + 1)]
    side = collections.defaultdict(list)
    for b in body:
        if bkind[b] != "steady":
            side[bkind[b]].extend(range(allblocks[b][0], allblocks[b][1] + 1))
    loop = [k for b in sorted(body) for k in range(allblocks[b][0], allblocks[b][1] + 1)]
    lo, hi = min(loop), max(loop)

    def cnt(sel, pred):
        return sum(1 for k in sel if pred(ins[k][1], ins[k][2]))

    c = {
        "wmma": cnt(steady, lambda o, a: o.startswith("v_wmma_f32_16x16x16_f16")),
        "ds_load_b128": cnt(steady, lambda o, a: o == "ds_load_b128"),
        "ds_store_b128": cnt(steady, lambda o, a: o == "ds_store_b128"),
        "global_load_b128": cnt(steady, lambda o, a: o == "global_load_b128"),
        "v_exp_f32": cnt(steady, lambda o, a: o.startswith("v_exp_f32")),
        "valu": cnt(steady, lambda o, a: is_valu(o)),
        "salu": cnt(steady, lambda o, a: o.startswith("s_") and not o.startswith(("s_wait", "s_delay", "s_nop"))),
        "wait_delay": cnt(steady, lambda o, a: o.startswith(("s_wait", "s_delay", "s_nop"))),
    }
    exp_all = c["v_exp_f32"] + cnt(side["rescale"], lambda o, a: o.startswith("v_exp_f32"))
    print(f"  loop {ins[lo][0]:x}..{ins[hi][0]:x}: {hi - lo + 1} instrs, blocks " +
          ", ".join(f"{kd}:{e - s + 1}" for s, e, kd in kinds))
    print("  steady path per wave per 32-key tile: " + ", ".join(f"{k} {v}" for k, v in c.items()))
    vh = collections.Counter(ins[k][1] for k in steady if is_valu(ins[k][1]))
    print("  steady VALU: " + ", ".join(f"{o} x{n}" for o, n in vh.most_common()))
    for kd in ("mask", "rescale", "edge"):
        if side[kd]:
            print(f"  side block {kd}: {len(side[kd])} instrs, VALU {cnt(side[kd], lambda o, a: is_valu(o))}, "
                  f"exp {cnt(side[kd], lambda o, a: o.startswith('v_exp_f32'))}")
    check(c["wmma"] == 32, f"{kern}: 32 v_wmma on the steady path (got {c['wmma']})")
    check(c["ds_load_b128"] == 32, f"{kern}: 32 ds_load_b128 (got {c['ds_load_b128']})")
    check(c["ds_store_b128"] == 2 * cpt, f"{kern}: {2 * cpt} ds_store_b128 (got {c['ds_store_b128']})")
    check(c["global_load_b128"] == 2 * cpt, f"{kern}: {2 * cpt} global_load_b128 (got {c['global_load_b128']})")
    check(c["v_exp_f32"] == 16, f"{kern}: 16 v_exp_f32 on the steady path (got {c['v_exp_f32']})")
    check(exp_all == 17, f"{kern}: 17 v_exp_f32 with the rescale block (got {exp_all})")
    bad = cnt(loop, lambda o, a: o.startswith(("global_inv", "ds_load_u16", "ds_load_b32", "ds_load_2addr", "global_load_u16",
                                               "v_rcp_iflag", "buffer_", "scratch_")) or o.startswith("ds_load_u8"))
    check(bad == 0, f"{kern}: 0 global_inv / narrow ds or global loads / integer divides / scratch in the loop (got {bad})")
    sig_all = [k for k in loop if ins[k][1] == "s_barrier_signal"]
    wt_all = [k for k in loop if ins[k][1] == "s_barrier_wait"]
    check(len(sig_all) == 1 and len(wt_all) == 1,
          f"{kern}: exactly 1 s_barrier_signal + 1 s_barrier_wait in the loop (got {len(sig_all)} + {len(wt_all)})")
    # Order checks along the steady path in execution order (positions in `steady`).
    P = list(range(len(steady)))
    op_at = [ins[k][1] for k in steady]
    sig = [p for p in P if op_at[p] == "s_barrier_signal"]
    wm = [p for p in P if op_at[p].startswith("v_wmma")]
    check(len(sig) == 1, f"{kern}: the steady path signals once (got {len(sig)})")
    if sig and wm:
        if nw16:
            first_ds = min(p for p in P if op_at[p] == "ds_load_b128")
            check(sig[0] < first_ds, f"{kern}: the signal precedes the first QK ds_load (one tile of slack)")
            gl = [p for p in P if op_at[p] == "global_load_b128"]
            between = [p for p in range(max(gl) + 1, sig[0]) if op_at[p] == "s_wait_loadcnt"] if gl else []
            check(bool(gl) and max(gl) < sig[0] and not between,
                  f"{kern}: no s_wait_loadcnt between the t+2 loads and the signal (loads stay in flight)")
        else:
            check(sig[0] > max(wm), f"{kern}: the signal follows the last WMMA (2-stage joined pattern)")
    # PV pipelining: the last 16 WMMAs; a WMMA is pipelined unless an s_wait_dscnt 0 sits
    # between it and the previous WMMA.
    pv = wm[-16:]
    piped = 0
    for j, p in enumerate(pv):
        prev = pv[j - 1] if j else wm[-17]
        waits = [ins[steady[x]][2] for x in range(prev + 1, p) if op_at[x] == "s_wait_dscnt"]
        if not waits or int(waits[-1], 0) >= 1:
            piped += 1
    check(piped >= 13, f"{kern}: s_wait_dscnt >= 1 (or none) before >= 13 of the 16 PV WMMAs (got {piped})")
    check(c["valu"] <= 85, f"{kern}: steady non-trans VALU <= 85 (got {c['valu']})")
    wmma_clk = 16 * c["wmma"]
    slots = c["valu"] + c["v_exp_f32"]
    print(f"  predicted per wave per tile: {wmma_clk} WMMA clk vs {c['valu']} VALU + {c['v_exp_f32']} exp issue slots "
          f"-> issue ceiling {wmma_clk / (wmma_clk + slots) * 100:.1f} % (exp at 1/4 rate: "
          f"{wmma_clk / (wmma_clk + c['valu'] + 4 * c['v_exp_f32']) * 100:.1f} %); SALU {c['salu']} (scalar port), "
          f"waits/delays {c['wait_delay']}")
    return c


def module(stem, kern, lds, wgp, cpt, nw16):
    print(f"G0a {stem}: {kern}")
    rm = remarks(os.path.join(OUT, stem + ".log")).get(kern, {})
    nt = notes(os.path.join(OUT, stem + ".notes")).get(kern, {})
    check(bool(rm) and bool(nt), f"{kern}: remarks and metadata present")
    if not (rm and nt):
        return None
    vg = int(nt.get("vgpr_count", "999"))
    print(f"  VGPRs {vg}, SGPRs {nt.get('sgpr_count')}, scratch {nt.get('private_segment_fixed_size')}, "
          f"LDS {nt.get('group_segment_fixed_size')}, occupancy {rm.get('Occupancy')}, "
          f"wgp_mode {nt.get('workgroup_processor_mode')}, spills v{nt.get('vgpr_spill_count')}/s{nt.get('sgpr_spill_count')}")
    check(vg <= 192, f"{kern}: VGPRs {vg} <= 192 (8 waves/SIMD)")
    check(nt.get("private_segment_fixed_size") == "0" and rm.get("ScratchSize") == "0", f"{kern}: scratch 0")
    check(nt.get("vgpr_spill_count") == "0" and nt.get("sgpr_spill_count") == "0", f"{kern}: 0 spills")
    check(nt.get("group_segment_fixed_size") == str(lds), f"{kern}: LDS {nt.get('group_segment_fixed_size')} == {lds}")
    check(rm.get("Occupancy") == "8", f"{kern}: occupancy {rm.get('Occupancy')} == 8 waves/SIMD")
    check(nt.get("workgroup_processor_mode") == str(wgp), f"{kern}: workgroup_processor_mode {nt.get('workgroup_processor_mode')} == {wgp}")
    check(nt.get("uses_dynamic_stack") == "false", f"{kern}: no dynamic stack")
    led = ledger(os.path.join(OUT, stem + ".s"), kern, cpt, nw16)
    return {"vgpr": vg, "sgpr": nt.get("sgpr_count"), "lds": nt.get("group_segment_fixed_size"), "ledger": led}


r16 = module("qwen3_attn_g4", "qe_attn_g4_n16", 49152, 0, 1, True)
r8 = module("qwen3_attn_g4n8", "qe_attn_g4_n8", 32768, 0, 2, False)
rw = module("qwen3_attn_g4wgp", "qe_attn_g4_n16", 49152, 1, 1, True)
print("G0a qwen3_embed producers")
rm = remarks(os.path.join(OUT, "qwen3_embed.log"))
nt = notes(os.path.join(OUT, "qwen3_embed.notes"))
for k in ("qe_qk_norm_rope", "qe_v_frag"):
    r, n = rm.get(k, {}), nt.get(k, {})
    print(f"  {k}: VGPRs {n.get('vgpr_count')}, scratch {n.get('private_segment_fixed_size')}, LDS {n.get('group_segment_fixed_size')}, occupancy {r.get('Occupancy')}")
    check(n.get("private_segment_fixed_size") == "0" and n.get("vgpr_spill_count") == "0", f"{k}: scratch 0, 0 spills")
print()
if FAILS:
    print(f"G0: {len(FAILS)} FAILED:")
    for f in FAILS:
        print("  - " + f)
    sys.exit(1)
print("G0: all checks passed")
