# What does the drafter's confidence head buy as the K policy?
# Replay over the reference records (gen2 = code/tool calls, prose), per block start:
#   realized acceptance at T=1 = the recorded per-depth rejection-sampling acceptance `rs`
#   (each conditional on the drafted prefix), so E[tokens | K] = 1 + sum_k prod_{j<=k} rs_j.
# Policies: fixed K; CONF = choose K maximizing predicted tokens/time with
# P(accept through k) = prod sigmoid(conf_j) (the head's own calibration, no fitting);
# ORACLE = choose K with the realized rs (upper bound for any per-block policy).
import json, math
t = {1: 60, 2: 81, 3: 101, 4: 107, 5: 125, 6: 139, 7: 156, 8: 172}
def cost(r): return t[r] if r in t else 172 + 17 * (r - 8)
sig = lambda x: 1 / (1 + math.exp(-x))
B = '/home/claude-code/.cache/deepstrix/v41/agentic'
recs = {n: json.load(open(f'{B}/{n}/dspark_accept_base.json'))['records'] for n in ('gen2', 'prose')}

def exp_tokens(acc, K):            # acc[k] = P(depth k accepted | earlier accepted)
    s, run = 1.0, 1.0
    for k in range(K):
        run *= acc[k]; s += run
    return s

def run(name, S, D, policy):
    """Throughput of S lockstep streams all on transcript `name`, tok/s."""
    tokens = time = 0.0
    for r in recs[name]:
        rs = r['rs']; pc = [sig(c) for c in r['conf']]
        if policy == 'conf':
            K = max(range(6), key=lambda K: S * exp_tokens(pc, K) / (cost(S * (1 + K)) + (D if K else 0)))
        elif policy == 'oracle':
            K = max(range(6), key=lambda K: S * exp_tokens(rs, K) / (cost(S * (1 + K)) + (D if K else 0)))
        else:
            K = policy
        tokens += S * exp_tokens(rs, K)
        time += cost(S * (1 + K)) + (D if K else 0)
    return tokens / time * 1000

for D in (0, 20):
    print(f"\n=== drafter {D} ms (0 = hidden) ===")
    print("streams | content | plain | best fixed K      | CONF head | oracle")
    for S in (1, 2, 4):
        res = {}
        for name in ('gen2', 'prose'):
            plain = run(name, S, D, 0)
            fixed = max((run(name, S, D, K), K) for K in range(1, 6))
            res[name] = (plain, fixed, run(name, S, D, 'conf'), run(name, S, D, 'oracle'))
            p, (fv, fk), c, o = res[name]
            print(f"   {S}    | {name:5s}   | {p:5.1f} | {fv:5.1f} (K={fk}, {fv/p:.2f}x) | {c:5.1f} ({c/p:.2f}x) | {o:5.1f} ({o/p:.2f}x)")
        # 55% code-like, 45% reasoning (production snapshots): per-token time blend.
        bl = lambda i: 1 / (0.55 / res['gen2'][i] + 0.45 / res['prose'][i])
        blf = 1 / (0.55 / res['gen2'][1][0] + 0.45 / res['prose'][1][0])
        print(f"   {S}    | blend   | {bl(0):5.1f} | {blf:5.1f} ({blf/bl(0):.2f}x)      | {bl(2):5.1f} ({bl(2)/bl(0):.2f}x) | {bl(3):5.1f} ({bl(3)/bl(0):.2f}x)")

# How well does the head rank blocks? realized E at K=5 by predicted-E tercile.
for name in ('gen2', 'prose'):
    rows = sorted(((exp_tokens([sig(c) for c in r['conf']], 5), exp_tokens(r['rs'], 5)) for r in recs[name]))
    n = len(rows); thirds = [rows[:n//3], rows[n//3:2*n//3], rows[2*n//3:]]
    print(f"{name}: realized E(K=5) by predicted tercile (low/mid/high):",
          [round(sum(x[1] for x in th) / len(th), 2) for th in thirds],
          " predicted:", [round(sum(x[0] for x in th) / len(th), 2) for th in thirds])
