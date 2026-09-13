"""Convert an oracle dump dir (embed_hc.pt, layer_NN_residual.pt, logits_last.pt)
into the engine's activation-dump layout read by
`crates/v4flash-kernels/src/oracle.rs` (manifest.json + one .bin per tensor).

  tag "embed_hc"  layer -1  token t   shape [hc_mult, dim]  f32  (input to layer 0)
  tag "residual"  layer L   token t   shape [hc_mult, dim]  f32  (output of layer L)
  tag "logits"    layer -1  token T-1 shape [vocab]         f32

Usage: nix-shell -p python3Packages.torch --run \
  "python3 export_bins.py ~/.cache/deepstrix/v41/oracle_full ~/.cache/deepstrix/v41/oracle_full_bins"
"""
import glob
import json
import os
import re
import sys

import torch


def main(src: str, dst: str) -> None:
    os.makedirs(dst, exist_ok=True)
    tensors = []

    def put(tag: str, layer: int, token: int, t: torch.Tensor, rel: str) -> None:
        t = t.detach().float().contiguous()
        path = os.path.join(dst, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(t.numpy().tobytes())
        tensors.append(dict(tag=tag, layer=layer, token=token, dtype="f32",
                            shape=list(t.shape), bytes=t.numel() * 4, path=rel, is_weight=False))

    emb = torch.load(os.path.join(src, "embed_hc.pt"))[0]  # [T, hc, dim]
    T = emb.shape[0]
    for t in range(T):
        put("embed_hc", -1, t, emb[t], f"embed/T{t:04d}/hc.bin")
    layers = sorted(glob.glob(os.path.join(src, "layer_*_residual.pt")))
    for f in layers:
        L = int(re.search(r"layer_(\d+)_residual", f).group(1))
        h = torch.load(f)[0]
        for t in range(T):
            put("residual", L, t, h[t], f"L{L:02d}/T{t:04d}/residual.bin")
    for f in sorted(glob.glob(os.path.join(src, "layer_*_topk_ids.pt"))):
        L = int(re.search(r"layer_(\d+)_topk_ids", f).group(1))
        ids = torch.load(f).reshape(-1, torch.load(f).shape[-1])  # [T, k] int32
        for t in range(ids.shape[0]):
            rel = f"L{L:02d}/T{t:04d}/topk_ids.bin"
            path = os.path.join(dst, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(ids[t].to(torch.int32).contiguous().numpy().tobytes())
            tensors.append(dict(tag="topk_ids", layer=L, token=t, dtype="i32",
                                shape=[int(ids.shape[1])], bytes=int(ids.shape[1]) * 4, path=rel, is_weight=False))
    for f in sorted(glob.glob(os.path.join(src, "layer_*_stage_*.pt"))):
        m = re.search(r"layer_(\d+)_stage_(.+)\.pt", f)
        L, name = int(m.group(1)), m.group(2)
        t = torch.load(f)
        t = t.reshape(-1, *t.shape[2:]) if t.dim() >= 3 else t.reshape(-1, t.shape[-1])  # [T, ...]
        for tt in range(t.shape[0]):
            put(f"stage_{name}", L, tt, t[tt], f"L{L:02d}/T{tt:04d}/stage_{name}.bin")
    lp = os.path.join(src, "logits_last.pt")
    vocab = 0
    if os.path.exists(lp):
        lg = torch.load(lp)
        lg = lg.reshape(-1, lg.shape[-1])
        vocab = lg.shape[-1]
        for i in range(lg.shape[0]):
            put("logits", -1, T - lg.shape[0] + i, lg[i], f"logits/T{T - lg.shape[0] + i:04d}/logits.bin")
    man = dict(meta=dict(source=os.path.abspath(src), hc_mult=int(emb.shape[1]), dim=int(emb.shape[2]),
                         n_layers=len(layers)),
               tensors=tensors, n_tensors=len(tensors), n_logit_rows=1 if vocab else 0,
               vocab_size=vocab, prompt_len=T)
    tj = os.path.join(src, "tokens.json")
    if os.path.exists(tj):
        import shutil
        shutil.copy(tj, os.path.join(dst, "tokens.json"))
    with open(os.path.join(dst, "manifest.json"), "w") as f:
        json.dump(man, f)
    print(f"{dst}: {len(tensors)} tensors, T={T}, {len(layers)} layers, vocab={vocab}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
