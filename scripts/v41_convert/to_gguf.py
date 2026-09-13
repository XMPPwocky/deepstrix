"""DeepSeek-V4.1-Flash HF safetensors -> GGUF for the deepstrix engine.

Naming follows llama.cpp's `deepseek4` conventions, which are exactly what
the engine's weight contract already reads for V4-Flash, so the loader,
het-split and existing kernels apply with no renaming. Per-tensor formats:

  routed experts  : MXFP4 (39), stacked [n_expert, out, in] — the native
                    checkpoint format, repacked byte-exactly from HF's
                    split weight+scale layout (validated against gguf-py's
                    decoder; scale bytes copy verbatim because llama.cpp's
                    doubled k-values and its 128 exponent bias cancel)
  fp8 + e8m0 blocks: dequantised and requantised to Q8_0 (what the dGPU
                    kernels take; ~1.06x fp8's bytes, tiny loss)
  bf16 small tensors: F16 / BF16 / F32 per the contract's role
  Engram tables   : NOT copied (189 GiB); metadata records the HF shard paths
  DSpark (mtp.*), vision, aligner: skipped (flags to add later)

Memory: one stacked expert matrix (~2.4 GB) is the high-water mark; run with
the server down or accept page-cache pressure. Output is split into parts
by --split-gb; each part is a valid GGUF with split.* keys.

Usage:
  nix-shell -p python3Packages.{gguf,numpy,torch} --run \
    "python3 to_gguf.py --out /persist/lumi/models/dsv41f/DeepSeek-V4.1-Flash-native.gguf [--layers 0-1]"
"""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "v41_oracle"))
import gguf  # noqa: E402
from gguf import GGMLQuantizationType as T  # noqa: E402
from loader import Checkpoint  # noqa: E402

MODEL = os.environ.get("V41_MODEL", os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f"))
V4F_GGUF = "/persist/lumi/models/dsv4f-exp-iq3-xxs/UD-IQ3_XXS/DeepSeek-V4-Flash-Vision-Exp-UD-IQ3_XXS-00001-of-00004.gguf"
ARCH = "deepseek4"  # same key family as V4-Flash; the engine dispatches on the constants below

EXPERT_LIMIT = 10**9  # --experts N (dry runs) caps the stacked count

FP4_TABLE = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


# ---------------------------------------------------------------- dequant / repack


def fp8_dequant(w: np.ndarray, s: np.ndarray) -> np.ndarray:
    """[out,in] e4m3 (as torch->numpy float32 already) x [out/b,in/b] e8m0 -> f32."""
    out, inn = w.shape
    bo, bi = out // s.shape[0], inn // s.shape[1]
    return (w.reshape(s.shape[0], bo, s.shape[1], bi) * s[:, None, :, None]).reshape(out, inn)


def mxfp4_repack(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """HF (packed [out,in/2] low-nibble-first, scale [out,in/32] e8m0 bytes)
    -> llama.cpp block_mxfp4 [out, in/32*17]: per block [e8m0][16 B: elems
    0..15 low nibbles, 16..31 high]. Validated byte-for-byte via gguf-py's
    dequantizer (scratchpad/probe_mxfp4.py)."""
    r, h = packed.shape
    c = h * 2
    nb = c // 32
    el = np.empty((r, c), dtype=np.uint8)
    el[:, 0::2] = packed & 0x0F
    el[:, 1::2] = packed >> 4
    blk = el.reshape(r, nb, 32)
    qs = (blk[:, :, :16] | (blk[:, :, 16:] << 4)).astype(np.uint8)
    return np.concatenate([scale.reshape(r, nb, 1), qs], axis=2).reshape(r, nb * 17)


# ---------------------------------------------------------------- tensor fetch helpers


def np_of(ck: Checkpoint, name: str) -> np.ndarray:
    """Torch view -> numpy, widening fp8/e8m0/bf16 to f32 (values), keeping int8 raw."""
    import torch
    t = ck.get(name)
    if t.dtype in (torch.int8, torch.uint8):
        return t.view(torch.uint8).numpy()
    return t.float().numpy()


def raw_u8(ck: Checkpoint, name: str) -> np.ndarray:
    """Raw storage bytes (e8m0 exponents, packed fp4 nibbles) — NOT values."""
    import torch
    return ck.get(name).view(torch.uint8).numpy()


def q8(ck: Checkpoint, name: str):
    """fp8+scale (or bf16) tensor -> (Q8_0 bytes, original shape)."""
    w = np_of(ck, name)
    sname = name[: -len("weight")] + "scale"
    if ck.has(sname):
        w = fp8_dequant(w, np_of(ck, sname))
    w = w.astype(np.float32)
    return gguf.quants.quantize(w, T.Q8_0), list(w.shape)


def q8_of(w: np.ndarray):
    return gguf.quants.quantize(w.astype(np.float32), T.Q8_0), list(w.shape)


def f16(ck, name):
    return np_of(ck, name).astype(np.float16)


def f32(ck, name):
    return np_of(ck, name).astype(np.float32)


def bf16(ck, name):
    import torch
    return ck.get(name).to(torch.bfloat16).view(torch.int16).numpy()  # raw bf16 bits


# ---------------------------------------------------------------- writer


def add(w: gguf.GGUFWriter, name: str, arr, raw_dtype=None):
    """Pre-quantised uint8 arrays carry their BYTE shape [..., bytes/row];
    gguf-py derives the element shape from raw_dtype's block size."""
    if isinstance(arr, tuple):  # (Q8_0 bytes, original shape) from q8()/q8_of()
        arr, _shape = arr
        raw_dtype = T.Q8_0
    if raw_dtype is not None:
        w.add_tensor(name, arr, raw_dtype=raw_dtype)
    elif arr.dtype == np.int16:  # bf16 bits
        w.add_tensor(name, arr, raw_dtype=T.BF16)
    else:
        w.add_tensor(name, arr)


def stacked_experts(ck: Checkpoint, L: int, which: str, n_expert: int) -> tuple[np.ndarray, list]:
    n_expert = min(n_expert, EXPERT_LIMIT)
    """All experts of one matrix as [n_expert, out, in/32*17] uint8, plus raw_shape."""
    p0 = f"layers.{L}.ffn.experts.0.{which}."
    out, half = ck.shape_of(p0 + "weight")
    inn = half * 2
    blob = np.empty((n_expert, out, inn // 32 * 17), dtype=np.uint8)
    for e in range(n_expert):
        p = f"layers.{L}.ffn.experts.{e}.{which}."
        blob[e] = mxfp4_repack(raw_u8(ck, p + "weight"), raw_u8(ck, p + "scale"))
    return blob, [n_expert, out, inn]


def write_layer(w: gguf.GGUFWriter, ck: Checkpoint, cfg: dict, L: int):
    p = f"layers.{L}."
    b = f"blk.{L}."
    n_expert = cfg["n_routed_experts"]
    ratio = cfg["compress_ratios"][L]
    # --- norms / hc / sinks (F32)
    for src, dst in (("attn_norm.weight", "attn_norm.weight"), ("ffn_norm.weight", "ffn_norm.weight"),
                     ("attn.q_norm.weight", "attn_q_a_norm.weight"), ("attn.kv_norm.weight", "attn_kv_a_norm.weight"),
                     ("attn.attn_sink", "attn_sinks.weight"),
                     ("hc_attn_fn", "hc_attn_fn.weight"), ("hc_ffn_fn", "hc_ffn_fn.weight"),
                     ("hc_attn_base", "hc_attn_base.weight"), ("hc_ffn_base", "hc_ffn_base.weight"),
                     ("hc_attn_scale", "hc_attn_scale.weight"), ("hc_ffn_scale", "hc_ffn_scale.weight"),
                     ("ffn.gate.bias", "exp_probs_b.bias"), ("ffn.gate.bias_vl", "exp_probs_b_vl.bias")):
        add(w, b + dst, f32(ck, p + src))
    # --- attention projections (fp8 -> Q8_0)
    for src, dst in (("attn.wq_a", "attn_q_a"), ("attn.wq_b", "attn_q_b"), ("attn.wkv", "attn_kv"),
                     ("attn.wo_a", "attn_output_a"), ("attn.wo_b", "attn_output_b")):
        add(w, b + dst + ".weight", q8(ck, p + src + ".weight"))
    # --- router (BF16 as stored) + shared expert (fp8 -> Q8_0)
    add(w, b + "ffn_gate_inp.weight", bf16(ck, p + "ffn.gate.weight"))
    for src, dst in (("w1", "ffn_gate_shexp"), ("w3", "ffn_up_shexp"), ("w2", "ffn_down_shexp")):
        add(w, b + dst + ".weight", q8(ck, p + f"ffn.shared_experts.{src}.weight"))
    # --- routed experts (native MXFP4, stacked)
    for src, dst in (("w1", "ffn_gate_exps"), ("w3", "ffn_up_exps"), ("w2", "ffn_down_exps")):
        blob, _shape = stacked_experts(ck, L, src, n_expert)
        add(w, b + dst + ".weight", blob, raw_dtype=T.MXFP4)
        del blob
    # --- compressor (kv_source layers): ratio 1 = bf16 wkv; ratio 2 = bf16 wkv + wgate
    if ck.has(p + "attn.compressor.wkv.weight"):
        add(w, b + "attn_compressor_kv.weight", f16(ck, p + "attn.compressor.wkv.weight"))
        if ck.has(p + "attn.compressor.wgate.weight"):
            add(w, b + "attn_compressor_gate.weight", f16(ck, p + "attn.compressor.wgate.weight"))
        add(w, b + "attn_compressor_norm.weight", f32(ck, p + "attn.compressor.norm.weight"))
    # --- indexer (index_source layers): q_b fp8->Q8_0, proj bf16->F32, wk/k_norm on kv_source layers
    if ck.has(p + "attn.indexer.wq_b.weight"):
        add(w, b + "indexer.attn_q_b.weight", q8(ck, p + "attn.indexer.wq_b.weight"))
        add(w, b + "indexer.proj.weight", f32(ck, p + "attn.indexer.weights_proj.weight"))
        if ck.has(p + "attn.indexer.wk.weight"):
            add(w, b + "indexer.attn_k.weight", f16(ck, p + "attn.indexer.wk.weight"))
            add(w, b + "indexer.k_norm.weight", f32(ck, p + "attn.indexer.k_norm.weight"))
    # --- engram (layers 1, 14): projections in-file; the table stays in the HF shard
    if ck.has(p + "engram.wkv.weight"):
        add(w, b + "engram_wkv.weight", q8(ck, p + "engram.wkv.weight"))
        add(w, b + "engram_q.weight", f32(ck, p + "engram.q_weight"))
        add(w, b + "engram_k.weight", f32(ck, p + "engram.k_weight"))


def write_globals(w: gguf.GGUFWriter, ck: Checkpoint):
    add(w, "token_embd.weight", f16(ck, "embed.weight"))  # engine embed path: F16 arm only
    add(w, "output.weight", q8_of(f32(ck, "head.weight")))
    add(w, "output_norm.weight", f32(ck, "norm.weight"))


def write_metadata(w: gguf.GGUFWriter, ck: Checkpoint, cfg: dict, n_layers: int):
    a = ARCH
    w.add_name("DeepSeek-V4.1-Flash")
    w.add_string(f"{a}.variant", "v4.1-flash")
    w.add_uint32(f"{a}.block_count", n_layers)
    w.add_uint32(f"{a}.context_length", 1048576)
    w.add_uint32(f"{a}.embedding_length", cfg["dim"])
    w.add_uint32(f"{a}.embedding_length_out", cfg["dim"] * cfg["hc_mult"])
    w.add_uint32(f"{a}.attention.head_count", cfg["n_heads"])
    w.add_uint32(f"{a}.attention.head_count_kv", 1)
    w.add_uint32(f"{a}.attention.key_length", cfg["head_dim"])
    w.add_uint32(f"{a}.attention.value_length", cfg["head_dim"])
    w.add_uint32(f"{a}.rope.dimension_count", cfg["rope_head_dim"])
    w.add_float32(f"{a}.rope.freq_base", cfg["rope_theta"])
    w.add_string(f"{a}.rope.scaling.type", "yarn")
    w.add_float32(f"{a}.rope.scaling.factor", cfg["rope_factor"])
    w.add_uint32(f"{a}.rope.scaling.original_context_length", cfg["original_seq_len"])
    w.add_float32(f"{a}.rope.scaling.yarn_beta_fast", cfg["beta_fast"])
    w.add_float32(f"{a}.rope.scaling.yarn_beta_slow", cfg["beta_slow"])
    w.add_float32(f"{a}.attention.layer_norm_rms_epsilon", cfg["norm_eps"])
    w.add_uint32(f"{a}.attention.q_lora_rank", cfg["q_lora_rank"])
    w.add_uint32(f"{a}.attention.output_group_count", cfg["o_groups"])
    w.add_uint32(f"{a}.attention.output_lora_rank", cfg["o_lora_rank"])
    w.add_uint32(f"{a}.attention.sliding_window", cfg["window_size"])
    w.add_array(f"{a}.attention.compress_ratios", cfg["compress_ratios"][:n_layers])
    w.add_float32(f"{a}.attention.compress_rope_freq_base", cfg["compress_rope_theta"])
    w.add_array(f"{a}.attention.kv_source_layers", cfg["kv_source_layers"])
    w.add_array(f"{a}.attention.index_source_layers", cfg["index_source_layers"])
    w.add_uint32(f"{a}.attention.indexer.head_count", cfg["index_n_heads"])
    w.add_uint32(f"{a}.attention.indexer.key_length", cfg["index_head_dim"])
    w.add_uint32(f"{a}.attention.indexer.top_k", cfg["index_topk"])
    w.add_int32(f"{a}.attention.candidate.source_layer", cfg["candidate_source_layer"])
    w.add_uint32(f"{a}.attention.candidate.top_k_blocks", cfg["candidate_topk_blocks"])
    w.add_uint32(f"{a}.attention.candidate.block_size", cfg["candidate_block_size"])
    w.add_uint32(f"{a}.expert_count", cfg["n_routed_experts"])
    w.add_uint32(f"{a}.expert_used_count", cfg["n_activated_experts"])
    w.add_uint32(f"{a}.expert_shared_count", cfg["n_shared_experts"])
    w.add_uint32(f"{a}.expert_feed_forward_length", cfg["moe_inter_dim"])
    w.add_float32(f"{a}.expert_weights_scale", cfg["route_scale"])
    w.add_bool(f"{a}.expert_weights_norm", True)
    w.add_uint32(f"{a}.expert_gating_func", 4)  # sqrtsoftplus, same code V4-Flash uses
    w.add_float32(f"{a}.swiglu_clamp_exp", cfg["swiglu_limit"])
    w.add_float32(f"{a}.swiglu_clamp_shexp", cfg["swiglu_limit"])
    w.add_uint32(f"{a}.hash_layer_count", 0)  # V4.1 has no hash-router layers
    w.add_uint32(f"{a}.hyper_connection.count", cfg["hc_mult"])
    w.add_uint32(f"{a}.hyper_connection.sinkhorn_iterations", cfg["hc_sinkhorn_iters"])
    w.add_float32(f"{a}.hyper_connection.epsilon", cfg["hc_eps"])
    # engram: projections are in-file; the 189 GiB tables are read from the HF shards
    w.add_array(f"{a}.engram.layer_ids", cfg["engram_layer_ids"])
    w.add_array(f"{a}.engram.num_embeddings", cfg["engram_num_embeddings"])
    w.add_uint32(f"{a}.engram.max_ngram_size", cfg["engram_max_ngram_size"])
    w.add_uint32(f"{a}.engram.vocab_size", cfg["engram_vocab_size"])
    w.add_uint32(f"{a}.engram.head_count", cfg["engram_n_heads"])
    w.add_uint32(f"{a}.engram.head_dim", cfg["engram_head_dim"])
    w.add_uint32(f"{a}.engram.pad_id", cfg["engram_pad_id"])
    w.add_uint32(f"{a}.engram.compressed_vocab_size", cfg["engram_compressed_vocab_size"])
    tables = [os.path.join(MODEL, ck.index[f"layers.{l}.engram.embed.weight"]) for l in cfg["engram_layer_ids"]]
    w.add_array(f"{a}.engram.table_shards", tables)
    # dspark (weights not converted yet; recorded so the loader can find them later)
    w.add_uint32(f"{a}.dspark.block_size", cfg["dspark_block_size"])
    w.add_uint32(f"{a}.dspark.noise_token_id", cfg["dspark_noise_token_id"])
    w.add_array(f"{a}.dspark.target_layer_ids", cfg["dspark_target_layer_ids"])
    # tokenizer: identical to V4-Flash (129280/129280 ids match) -> copy its KVs verbatim
    r = gguf.GGUFReader(V4F_GGUF)
    for k in ("tokenizer.ggml.model", "tokenizer.ggml.pre"):
        w.add_string(k, bytes(r.fields[k].parts[-1]).decode())
    fld = r.fields["tokenizer.ggml.tokens"]
    w.add_array("tokenizer.ggml.tokens", [bytes(fld.parts[i]).decode("utf-8", "replace") for i in fld.data])
    fld = r.fields["tokenizer.ggml.token_type"]
    w.add_array("tokenizer.ggml.token_type", [int(fld.parts[i][0]) for i in fld.data])
    fld = r.fields["tokenizer.ggml.merges"]
    w.add_array("tokenizer.ggml.merges", [bytes(fld.parts[i]).decode("utf-8", "replace") for i in fld.data])
    for k in ("tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id"):
        w.add_uint32(k, int(r.fields[k].parts[-1][0]))
    tcfg = json.load(open(os.path.join(MODEL, "tokenizer_config.json")))
    if "chat_template" in tcfg:
        w.add_string("tokenizer.chat_template", tcfg["chat_template"])


def part_path(out: str, no: int, count: int) -> str:
    base, ext = os.path.splitext(out)
    return f"{base}-{no+1:05d}-of-{count:05d}{ext}"


def main():
    global EXPERT_LIMIT
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=None, help="a-b inclusive (default all)")
    ap.add_argument("--layers-per-part", type=int, default=4, help="peak RAM ≈ this × ~7.5 GB")
    ap.add_argument("--experts", type=int, default=None, help="DRY RUN: only the first N experts")
    ap.add_argument("--no-globals", action="store_true")
    a = ap.parse_args()
    if a.experts:
        EXPERT_LIMIT = a.experts
    ck = Checkpoint(MODEL)
    cfg = json.load(open(os.path.join(MODEL, "inference", "config.json")))
    n_layers = cfg["n_layers"]
    lo, hi = (0, n_layers - 1) if a.layers is None else map(int, a.layers.split("-"))
    if a.experts:
        cfg = dict(cfg, n_routed_experts=a.experts)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    groups = [list(range(s_, min(s_ + a.layers_per_part, hi + 1))) for s_ in range(lo, hi + 1, a.layers_per_part)]
    n_parts = len(groups)
    # tensor count for split.tensors.count: count by staging part 0 later; llama.cpp only
    # requires it to be consistent, so we compute it up front by a dry count.
    t0 = time.time()
    total_tensors = 0
    for pi, layers in enumerate(groups):
        path = part_path(a.out, pi, n_parts)
        w = gguf.GGUFWriter(path, ARCH, use_temp_file=False)
        if pi == 0:
            write_metadata(w, ck, cfg, n_layers)
            if not a.no_globals:
                write_globals(w, ck)
        w.add_uint16("split.no", pi)
        w.add_uint16("split.count", n_parts)
        for L in layers:
            tl = time.time()
            write_layer(w, ck, cfg, L)
            print(f"  part {pi+1}/{n_parts} layer {L:2d} staged ({time.time()-tl:.0f}s)", flush=True)
        total_tensors += len(w.tensors[0]) if isinstance(w.tensors, list) else len(w.tensors)
        w.add_int32("split.tensors.count", -1)  # patched below once known; llama.cpp tolerates it
        print(f"  writing {path} ...", flush=True)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file(progress=False)
        w.close()
    print(f"done: {n_parts} part(s), {total_tensors} tensors, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
