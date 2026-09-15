"""Reference values for the engine's DSpark drafter, for gate 2 of DSPARK_DESIGN.md.

Emits, for one position, the tensors the Rust `MtpState` must reproduce:

    main_hidden  cat(mean-over-hc(residual after layers 36/37/38))   [15360]

`proj` and `main_x` are NOT computed here: `main_proj` is F8_E4M3 with separate
32x32 F8_E8M0 block scales, and `Checkpoint.get` returns the raw on-disk dtype,
so the obvious `main_hidden @ ck.get(...).float().t()` silently drops the scales
and lands ~5% off in direction (cos 0.946) while keeping a plausible magnitude.
Run `mtp_entry_ref.py` (numpy, no torch) afterwards to write proj.bin/main_x.bin.

`main_hidden` is the ATTENTION INPUT of layers 37/38/39, which is the residual
AFTER layers 36/37/38 — the dump's `layer_{l}_residual.pt` is post-layer, so the
indices here are 36/37/38 even though the config's dspark_target_layer_ids is
[37, 38, 39]. Getting that off by one would be invisible: the drafter would still
emit fluent tokens, just from the wrong residual.

  nix-shell -p python3Packages.{torch,numpy,safetensors,pillow,transformers} --run \
    "python3 dump_mtp_ref.py ~/.cache/deepstrix/v41/agentic/main --pos 200"
"""
import argparse, os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

SRC_LAYERS = [36, 37, 38]  # post-layer residuals = attention input of 37/38/39


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--pos", type=int, default=200)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(a.dump, "mtp_ref")
    os.makedirs(out, exist_ok=True)

    parts = []
    for l in SRC_LAYERS:
        r = torch.load(os.path.join(a.dump, f"layer_{l:02d}_residual.pt"))[0]  # [T, hc, dim]
        parts.append(r.mean(dim=1)[a.pos].float())                             # [dim]
    main_hidden = torch.cat(parts)                                             # [3*dim]

    for name, t in [("main_hidden", main_hidden)]:
        torch.save(t, os.path.join(out, f"{name}.pt"))
        with open(os.path.join(out, f"{name}.bin"), "wb") as f:
            f.write(t.contiguous().numpy().astype("float32").tobytes())
        print(f"{name:12} {tuple(t.shape)}  mean {t.mean():+.6f}  std {t.std():.6f}  "
              f"absmax {t.abs().max():.6f}")
    with open(os.path.join(out, "meta.txt"), "w") as f:
        f.write(f"pos={a.pos}\nsrc_layers={SRC_LAYERS}\ndump={a.dump}\n")
    print(f"\nwrote {out} (pos {a.pos}, src layers {SRC_LAYERS})")
    print("now run: nix-shell -p python3Packages.numpy --run "
          f"'python3 {os.path.join(HERE, 'mtp_entry_ref.py')} --dump {out}'")


if __name__ == "__main__":
    main()
