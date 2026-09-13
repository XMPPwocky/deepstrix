"""What the missing indexer costs in attention work, per decoded token.

Pure arithmetic over `inference/config.json` — no weights, no GPU. Prints, for a
set of context lengths, how many compressed rows every layer scores under the
reference's top-`index_topk` selection versus deepstrix's dense scoring, plus the
KV bytes and MACs that follow from it.
"""
import json
import os
import sys

MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f"))
cfg = json.load(open(os.path.join(MODEL, "inference", "config.json")))

N_LAYERS = cfg["n_layers"]
RATIOS = cfg["compress_ratios"][:N_LAYERS]
TOPK = cfg["index_topk"]
WIN = cfg["window_size"]
HEADS, HDIM = cfg["n_heads"], cfg["head_dim"]
IDX_SRC = cfg["index_source_layers"]
IDX_HEADS, IDX_HDIM = cfg["index_n_heads"], cfg["index_head_dim"]
# compressed KV row: E2M1 values + one E4M3 scale per 16
ROW_BYTES = HDIM // 2 + HDIM // 16
WIN_BYTES = HDIM + 4 * (HDIM // 32)          # FP8 values + f32 scales per 32


def rows(C):
    """per-layer compressed store size at context C"""
    return [0 if r == 0 else C // r for r in RATIOS]


def report(C):
    st = rows(C)
    sparse = [min(TOPK, n) for n in st]
    dense = st
    n_ratio2 = sum(1 for r in RATIOS if r == 2)
    n_ratio1 = sum(1 for r in RATIOS if r == 1)
    # attention (score + AV) MACs over the compressed part, per decoded token
    mac_s = sum(2 * n * HEADS * HDIM for n in sparse)
    mac_d = sum(2 * n * HEADS * HDIM for n in dense)
    # the indexer that the sparse path pays for instead (FP4, score only)
    mac_i = sum(st[l] * IDX_HEADS * IDX_HDIM for l in IDX_SRC if l < N_LAYERS)
    by_s = sum(n * ROW_BYTES for n in sparse) + N_LAYERS * min(C, WIN) * WIN_BYTES
    by_d = sum(n * ROW_BYTES for n in dense) + N_LAYERS * min(C, WIN) * WIN_BYTES
    print(f"C={C:>9,}  store rows/layer: ratio2 {C//2 if C>=2 else 0:>9,} ({n_ratio2} layers)  "
          f"ratio1 {C:>9,} ({n_ratio1} layers)")
    print(f"            rows READ per token   sparse {sum(sparse):>12,}   dense {sum(dense):>12,}   "
          f"dense/sparse {sum(dense)/max(sum(sparse),1):>7.1f}x")
    print(f"            compressed-KV bytes   sparse {by_s/1e6:>9.2f} MB  dense {by_d/1e6:>9.2f} MB  "
          f"ratio {by_d/by_s:>7.1f}x")
    print(f"            attn MACs/token       sparse {mac_s/1e9:>9.3f} G   dense {mac_d/1e9:>9.3f} G   "
          f"+ indexer {mac_i/1e9:.3f} G  =>  dense/(sparse+idx) {mac_d/(mac_s+mac_i):>6.1f}x")


if __name__ == "__main__":
    print(f"index_topk={TOPK}, window={WIN}, head_dim={HDIM}, heads={HEADS}; "
          f"ratios: {RATIOS.count(0)}x0, {RATIOS.count(2)}x2, {RATIOS.count(1)}x1")
    print(f"compressed row = {ROW_BYTES} B (E2M1+E4M3/16), window row = {WIN_BYTES} B\n")
    for C in [int(x) for x in (sys.argv[1:] or [512, 1024, 2048, 4096, 16384, 32768, 100000, 1000000])]:
        report(C)
