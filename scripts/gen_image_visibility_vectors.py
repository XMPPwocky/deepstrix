#!/usr/bin/env python3
"""Reference vectors for the Vision-Exp text-side attention window.

Pure-Python port (no torch) of DeepSeek `model.py`:
  get_image_visible(input_ids, vocab_size, max_image_tokens)
  get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens)
evaluated on the WHOLE prompt (start_pos == 0, as the reference requires
for image prompts). For every row it records the absolute key range
[win_start, win_end] (inclusive) of the RAW window; -1 entries of the
reference matrix are dropped.

Output: crates/v4flash-kernels/tests/data/image_visibility_cases.json
consumed by crates/v4flash-kernels/tests/image_visibility.rs, which
replays the same prompt through `het::image_spans` chunk by chunk and
checks (left, right) and the cache-slot window against this table.
"""
import json
import os
import random

VOCAB = 129280
IMAGE_START, IMAGE_PAD, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(5)
W = 128
MAX_IMG = 384


def get_image_visible(ids, vocab_size, max_image_tokens):
    n = len(ids)
    is_start = [t == vocab_size + IMAGE_START for t in ids]
    is_end = [t == vocab_size + IMAGE_END for t in ids]
    cs_start, cs_end = [], []
    a = b = 0
    for i in range(n):
        a += is_start[i]
        b += is_end[i]
        cs_start.append(a)
        cs_end.append(b)
    valid = [(cs_start[i] > cs_end[i]) or is_end[i] for i in range(n)]
    # starts = cummax(where(is_start, idx, 0))
    starts, m = [], 0
    for i in range(n):
        m = max(m, i if is_start[i] else 0)
        starts.append(m)
    left = [(i - starts[i]) * int(valid[i]) for i in range(n)]
    # ends = reverse-cummin(where(is_end, idx, seqlen))
    ends, m = [0] * n, n
    for i in range(n - 1, -1, -1):
        m = min(m, i if is_end[i] else n)
        ends[i] = m
    right = [(ends[i] - i) * int(valid[i]) for i in range(n)]
    left = [min(l, max_image_tokens - 1) for l in left]
    right = [min(r, max_image_tokens) for r in right]
    return left, right


def get_window_topk_idxs_visible(window_size, seqlen, left, right, max_image_tokens):
    width = min(seqlen, window_size + max_image_tokens)
    rows = []
    for idx in range(seqlen):
        left_add = max(left[idx] - (window_size - 1), 0)
        start = max(idx - (window_size - 1) - left_add, 0)
        keys = [start + j for j in range(width)]
        keys = [k if k <= idx + right[idx] else -1 for k in keys]
        rows.append(keys)
    return rows


def image_block(n_img, start_pos):
    """A minimal block shaped like build_image_block: compress pads,
    START, n_img IMAGE/NEWLINE tokens, END. (Exact interleaving is
    irrelevant for visibility; only START/END positions matter.)"""
    pads = [VOCAB + IMAGE_PAD] * (3 - start_pos % 4)
    body = [VOCAB + (IMAGE if (i % 9) else IMAGE_NEW_LINE) for i in range(n_img)]
    return pads + [VOCAB + IMAGE_START] + body + [VOCAB + IMAGE_END]


def build_prompt(rng, pieces):
    """pieces: list of ('text', n) | ('img', n_img). Returns ids and spans."""
    ids, spans = [], []
    for kind, n in pieces:
        if kind == 'text':
            ids += [rng.randrange(1, 129000) for _ in range(n)]
        else:
            blk = image_block(n, len(ids))
            s = len(ids) + blk.index(VOCAB + IMAGE_START)
            ids += blk
            e = len(ids) - 1
            assert ids[e] == VOCAB + IMAGE_END
            spans.append([s, e - s + 1])
    return ids, spans


def main():
    rng = random.Random(20260903)
    cases = []
    specs = [
        ("single_small_start0", [('img', 20), ('text', 200)], [64, 150]),
        ("text_then_image", [('text', 300), ('img', 150), ('text', 100)], [300, 128]),
        ("two_images", [('text', 50), ('img', 366), ('text', 10), ('img', 198), ('text', 500)], [60, 300, 900]),
        ("image_near_chunk_end", [('text', 900), ('img', 100), ('text', 30)], [700, 1024]),
        ("max_image_384_plus", [('text', 5), ('img', 389), ('text', 700)], [512, 600]),
        ("tiny_image_len2", [('text', 130), ('img', 0), ('text', 130)], [130, 200]),
        ("adjacent_images", [('img', 40), ('img', 60), ('img', 30), ('text', 200)], [256]),
        ("long_text_gaps", [('text', 1500), ('img', 250), ('text', 1500), ('img', 80), ('text', 40)], [1024, 2048, 3072]),
    ]
    for name, pieces, cut_hint in specs:
        ids, spans = build_prompt(rng, pieces)
        n = len(ids)
        left, right = get_image_visible(ids, VOCAB, MAX_IMG)
        mat = get_window_topk_idxs_visible(W, n, left, right, MAX_IMG)
        rows = []
        for i in range(n):
            keys = [k for k in mat[i] if k >= 0]
            assert keys == list(range(keys[0], keys[-1] + 1))
            rows.append([left[i], right[i], keys[0], keys[-1]])
        # Chunk cuts: hints that are legal (not strictly inside a span) are
        # kept; the Rust test also exercises its own planner.
        def legal(p):
            return all(not (s < p <= s + l - 1) for s, l in spans)
        cuts = sorted({c for c in cut_hint if 0 < c < n and legal(c)})
        bad_cuts = sorted({c for c in cut_hint if 0 < c < n and not legal(c)})
        cases.append({
            "name": name,
            "ids": ids,
            "spans": spans,
            "rows": rows,
            "legal_cuts": cuts,
            "illegal_cuts": bad_cuts,
        })
    out = os.path.join(os.path.dirname(__file__), "..", "crates", "v4flash-kernels", "tests", "data",
                       "image_visibility_cases.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"vocab": VOCAB, "window": W, "max_image_tokens": MAX_IMG, "cases": cases}, f)
    print(f"wrote {out}: {len(cases)} cases, {sum(len(c['rows']) for c in cases)} rows")


if __name__ == "__main__":
    main()
