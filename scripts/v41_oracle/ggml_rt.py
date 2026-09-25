"""Round-trip weights through ggml's real quantizers (ctypes into libggml-base).

    q = quantize(TYPE, w, imatrix)   # w: [nrows, n_per_row] f32, rows = output dim
    w2 = dequantize(TYPE, q, nrows, n_per_row)

ggml quantizes each row in blocks along n_per_row (the input dim), as llama.cpp
stores expert tensors. The IQ2 grids are built once per process by
ggml_quantize_init. IQ2_XXS / IQ2_XS refuse to quantize without an importance
matrix (ggml aborts), so "no imatrix" means a uniform one: every column weighted
1, which leaves ggml's own per-weight sqrt(sigma2 + x^2) term in charge.
Library: $GGML_LIB, else the llama-cpp build in box 2's nix store.
"""
import ctypes
import os

import numpy as np

F32, Q8_0, IQ2_XXS, IQ2_XS, IQ2_S = 0, 8, 16, 17, 22
NAMES = {"q8_0": Q8_0, "iq2_xxs": IQ2_XXS, "iq2_xs": IQ2_XS, "iq2_s": IQ2_S}

_DEFAULT = "/nix/store/gk5m87r61y55scqksiqnky1m4v8h9s8n-llama-cpp-9190/lib/libggml-base.so"


class _Traits(ctypes.Structure):  # ggml.h struct ggml_type_traits (ggml 0.12)
    _fields_ = [
        ("type_name", ctypes.c_char_p),
        ("blck_size", ctypes.c_int64),
        ("blck_size_interleave", ctypes.c_int64),
        ("type_size", ctypes.c_size_t),
        ("is_quantized", ctypes.c_bool),
        ("to_float", ctypes.c_void_p),
        ("from_float_ref", ctypes.c_void_p),
    ]


_TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
_lib = None
_inited = set()


def lib():
    global _lib
    if _lib is None:
        L = ctypes.CDLL(os.environ.get("GGML_LIB", _DEFAULT))
        L.ggml_get_type_traits.restype = ctypes.POINTER(_Traits)
        L.ggml_get_type_traits.argtypes = [ctypes.c_int]
        L.ggml_quantize_init.argtypes = [ctypes.c_int]
        L.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        L.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
        L.ggml_row_size.restype = ctypes.c_size_t
        L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]
        _lib = L
    return _lib


def type_name(t: int) -> str:
    return lib().ggml_get_type_traits(t).contents.type_name.decode()


def quantize(t: int, w: np.ndarray, imatrix: np.ndarray | None = None) -> bytes:
    L = lib()
    if t not in _inited:
        L.ggml_quantize_init(t)
        _inited.add(t)
    w = np.ascontiguousarray(w, dtype=np.float32)
    nrows, n_per_row = w.shape
    if imatrix is None and L.ggml_quantize_requires_imatrix(t):
        imatrix = np.ones(n_per_row, dtype=np.float32)
    if imatrix is not None:
        imatrix = np.ascontiguousarray(imatrix, dtype=np.float32)
        assert imatrix.shape == (n_per_row,), (imatrix.shape, n_per_row)
    out = ctypes.create_string_buffer(L.ggml_row_size(t, n_per_row) * nrows)
    n = L.ggml_quantize_chunk(t, w.ctypes.data, out, 0, nrows, n_per_row,
                              None if imatrix is None else imatrix.ctypes.data)
    assert n == len(out), (n, len(out))
    return out.raw


def dequantize(t: int, q: bytes, nrows: int, n_per_row: int) -> np.ndarray:
    tr = lib().ggml_get_type_traits(t).contents
    y = np.empty((nrows, n_per_row), dtype=np.float32)
    buf = ctypes.create_string_buffer(q, len(q))
    _TO_FLOAT(tr.to_float)(buf, y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), nrows * n_per_row)
    return y
