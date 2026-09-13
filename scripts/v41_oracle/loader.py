"""mmap-backed safetensors reader for the V4.1 checkpoint.

Reads tensors as the raw dtype stored on disk (packed fp4 as uint8, e8m0
scales, e4m3, bf16, f32) straight out of the page cache — nothing is copied
until a caller dequantises it. That is what lets the oracle run one expert at
a time beside a server that has taken all but ~3 GiB of RAM.
"""
import json
import mmap
import os
import struct

import torch

_DT = {
    "I8": torch.int8, "U8": torch.uint8, "I32": torch.int32, "I64": torch.int64,
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
    "F8_E4M3": torch.float8_e4m3fn, "F8_E8M0": torch.float8_e8m0fnu,
}


class Checkpoint:
    def __init__(self, root: str):
        self.root = root
        self.index = json.load(open(os.path.join(root, "model.safetensors.index.json")))["weight_map"]
        self.config = json.load(open(os.path.join(root, "config.json")))
        self._shards: dict[str, tuple[mmap.mmap, dict, int]] = {}

    def _shard(self, fname: str):
        if fname not in self._shards:
            path = os.path.join(self.root, fname)
            f = open(path, "rb")
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            n = struct.unpack("<Q", mm[:8])[0]
            hdr = json.loads(mm[8 : 8 + n])
            hdr.pop("__metadata__", None)
            self._shards[fname] = (mm, hdr, 8 + n)
        return self._shards[fname]

    def has(self, name: str) -> bool:
        return name in self.index

    def get(self, name: str) -> torch.Tensor:
        """Zero-copy view of a tensor in its on-disk dtype."""
        fname = self.index[name]
        mm, hdr, base = self._shard(fname)
        meta = hdr[name]
        a, b = meta["data_offsets"]
        buf = memoryview(mm)[base + a : base + b]
        dt = _DT[meta["dtype"]]
        t = torch.frombuffer(buf, dtype=dt)
        return t.view(meta["shape"])

    def dtype_of(self, name: str) -> str:
        fname = self.index[name]
        _, hdr, _ = self._shard(fname)
        return hdr[name]["dtype"]

    def shape_of(self, name: str):
        fname = self.index[name]
        _, hdr, _ = self._shard(fname)
        return hdr[name]["shape"]

    def names(self, prefix: str):
        return [k for k in self.index if k.startswith(prefix)]
