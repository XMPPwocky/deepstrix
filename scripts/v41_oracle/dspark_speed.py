# Expected DSpark decode speed on the 09-30..10-01 step ladder (p50 ms by rows).
t = {1: 60, 2: 81, 3: 101, 4: 107, 5: 125, 6: 139, 7: 156, 8: 172}
def cost(r): return t[r] if r in t else 172 + 17 * (r - 8)   # >8 rows extrapolated
# Reference drafter, prefix acceptance per depth (first k drafts all accepted).
curves = {
  ("code", "T=1 RS"): [0.851, 0.694, 0.607, 0.527, 0.453],
  ("prose", "T=1 RS"): [0.538, 0.202, 0.058, 0.018, 0.007],
  ("code", "T=0"): [0.843, 0.730, 0.674, 0.596, 0.539],
  ("prose", "T=0"): [0.562, 0.258, 0.101, 0.045, 0.022],
}
E = lambda p, K: 1 + sum(p[:K])
def best(p, S, D):
    plain = S / cost(S) * 1000
    opts = [(S * E(p, K) / (cost(S * (1 + K)) + (D if K else 0)) * 1000, K) for K in range(0, 6)]
    v, K = max(opts)
    return v, K, plain
print("per-depth E(K) =", {k: [round(E(p, K), 2) for K in range(1, 6)] for k, p in curves.items()})
for D in (0, 20):
    print(f"\n=== drafter {D} ms per step (0 = hidden behind the other lane) ===")
    print("streams | regime        | plain tok/s | best K | DSpark tok/s | x")
    for S in (1, 2, 4):
        for key in [("code", "T=1 RS"), ("prose", "T=1 RS"), ("code", "T=0"), ("prose", "T=0")]:
            v, K, plain = best(curves[key], S, D)
            print(f"   {S}    | {key[0]:5s} {key[1]:7s} |   {plain:5.1f}     |   {K}    |   {v:5.1f}      | {v/plain:.2f}")
        vc, _, plain = best(curves[("code", "T=1 RS")], S, D)
        vp, _, _ = best(curves[("prose", "T=1 RS")], S, D)
        blend = 1 / (0.55 / vc + 0.45 / vp)
        print(f"   {S}    | BLEND T=1 55/45 |   {plain:5.1f}     |  adapt |   {blend:5.1f}      | {blend/plain:.2f}")
