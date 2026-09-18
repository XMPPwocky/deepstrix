"""Compare an indexer_ab run's row dir against a stock oracle.py dump."""
import os, sys, torch
a, b = sys.argv[1], sys.argv[2]   # indexer_ab row dir, stock oracle dump
worst = 0.0
for L in range(40):
    pa = os.path.join(a, f"layer_{L:02d}_residual.pt"); pb = os.path.join(b, f"layer_{L:02d}_residual.pt")
    if not (os.path.exists(pa) and os.path.exists(pb)): continue
    x, y = torch.load(pa).float(), torch.load(pb).float()
    d = (x - y).abs().max().item(); s = y.abs().max().item()
    eq = (x == y).float().mean().item()
    worst = max(worst, d / max(s, 1e-9))
    if L % 8 == 0 or L >= 38 or d > 0:
        print(f"L{L:02d} max|d| {d:.3e}  max|ref| {s:.3e}  rel {d/max(s,1e-9):.2e}  bit-equal {eq*100:.2f}%")
print(f"worst relative residual drift over layers: {worst:.2e}")
# final logits
pa = os.path.join(a, "head_input.pt")
if os.path.exists(os.path.join(b, "logits_last.pt")):
    lb = torch.load(os.path.join(b, "logits_last.pt")).float()
    print("stock logits_last top-5:", lb[0].topk(5).indices.tolist())
    am = torch.load(os.path.join(a, "main_argmax.pt"))
    print("ab row argmax (last position):", int(am[-1]))
