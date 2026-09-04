#!/usr/bin/env python3
"""Golden vectors for `deepstrix-server`'s `render_prompt`, generated from the
model's OWN `tokenizer.chat_template` (jinja2) — the ground truth our
hand-rolled renderer has to reproduce byte for byte.

Outputs two files:

  crates/deepstrix-server/tests/data/tool_prompt_cases.json
      one entry per conversation: the request as the server sees it
      (messages / tools / reasoning effort) plus `canonical_text`, the exact
      string the jinja template renders for it.

  crates/deepstrix-server/tests/data/tool_prompt_vocab.json
      a TRIMMED BPE vocab (sparse id -> token map + the merge list, original
      ids preserved, merge order preserved) that is provably sufficient to
      tokenize every `canonical_text` above. The trim criterion is purely
      lexical and implementation-independent: a merge `A B` can only ever fire
      if `AB` occurs as a contiguous substring of the GPT-2 byte-encoded text,
      and a token can only be emitted if its bytes occur there too. Keeping
      every token/merge that passes that test therefore cannot change any
      tokenization of these texts, while dropping the other ~125 k entries
      keeps the checked-in file at ~100 KB instead of ~5 MB.

Usage (jinja2 is not in the dev shell; pull it from nixpkgs):

    nix-shell -p python3Packages.jinja2 --run \
      'python3 scripts/gen_tool_prompt_vectors.py \
         /persist/lumi/models/dsv4f-exp-q2-k-xl/UD-Q2_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf'

Only the GGUF *metadata header* is read (chat template + tokenizer arrays);
no tensor data is touched, so this never loads the model.
"""

import hashlib
import json
import os
import struct
import sys

# --------------------------------------------------------------------------
# Minimal GGUF metadata reader (header only)
# --------------------------------------------------------------------------

_SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
_FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}


def _rd(f, n):
    b = f.read(n)
    if len(b) < n:
        raise EOFError("short read")
    return b


def _rstr(f):
    (n,) = struct.unpack("<Q", _rd(f, 8))
    return _rd(f, n).decode("utf-8")


def _rval(f, t):
    if t == 8:
        return _rstr(f)
    if t == 9:
        (et,) = struct.unpack("<I", _rd(f, 4))
        (n,) = struct.unpack("<Q", _rd(f, 8))
        return [_rval(f, et) for _ in range(n)]
    return struct.unpack("<" + _FMT[t], _rd(f, _SZ[t]))[0]


def read_gguf_metadata(path):
    with open(path, "rb") as f:
        assert _rd(f, 4) == b"GGUF", "not a GGUF file"
        struct.unpack("<I", _rd(f, 4))
        _n_tensors, n_kv = struct.unpack("<QQ", _rd(f, 16))
        kv = {}
        for _ in range(n_kv):
            k = _rstr(f)
            (t,) = struct.unpack("<I", _rd(f, 4))
            kv[k] = _rval(f, t)
    return kv


# --------------------------------------------------------------------------
# GPT-2 byte encoding (mirrors v4flash_core::tokenizer::byte_encode)
# --------------------------------------------------------------------------


def _gpt2_table():
    tbl = []
    n = 0
    for b in range(256):
        if (33 <= b <= 126) or (161 <= b <= 172) or b >= 174:
            tbl.append(chr(b))
        else:
            tbl.append(chr(256 + n))
            n += 1
    return tbl


_BYTE_TO_CP = _gpt2_table()


def byte_encode(s: str) -> str:
    return "".join(_BYTE_TO_CP[b] for b in s.encode("utf-8"))


# --------------------------------------------------------------------------
# The conversations
# --------------------------------------------------------------------------

SYSTEM = "You are deepstrix, a coding agent.\n\nBe concise and prefer small diffs."

# Three tools: nested + optional params, an enum, and (on `edit_file`) extra
# top-level fields in a non-canonical key order — the client's JSON must
# survive verbatim (D10).
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the project sandbox.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute."},
                    "timeout_ms": {
                        "type": "integer",
                        "description": "Optional timeout in milliseconds.",
                        "default": 120000,
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_repo",
            "description": "Search the repository for a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "mode": {
                        "type": "string",
                        "enum": ["content", "files", "count"],
                        "description": "What the search returns.",
                    },
                    "opts": {
                        "type": "object",
                        "description": "Optional switches.",
                        "properties": {
                            "case_insensitive": {"type": "boolean"},
                            "glob": {"type": "string"},
                        },
                        "required": [],
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            # Deliberately NOT name-first, and carrying fields our ToolDef
            # struct has no field for.
            "description": "Replace an exact string in a file.",
            "name": "edit_file",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
            "x-deepstrix-hint": "prefer the smallest edit that works",
        },
    },
]

# Scientific-notation bounds/defaults. `json.dumps` writes these as Python
# `repr` does (`1e-06`, `1e+30`, `1.5e-05`), which is NOT what serde_json's
# default float writer produces — see `dsml::fmt_f64_python`.
#
# Every literal here is one serde_json's float PARSER reproduces exactly.
# Some (`1e-30`, `2.5e-30`) it rounds one ULP below CPython, which no amount
# of output formatting can undo; see the KNOWN RESIDUAL note on
# `fmt_f64_python`. Keep this list to exactly-parsed values so the case pins
# the formatting, not the parser.
TOOL_EXP_FLOATS = [
    {
        "type": "function",
        "function": {
            "name": "tune",
            "description": "Set a solver tolerance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tol": {
                        "type": "number",
                        "default": 1e-06,
                        "minimum": 1e-20,
                        "maximum": 1e30,
                    },
                    "step": {"type": "number", "default": 1.5e-05},
                    "cap": {"type": "number", "default": 1e16, "minimum": 1e15},
                    "plain": {"type": "number", "default": 0.0001},
                },
                "required": ["tol"],
            },
        },
    }
]

# A tool with no `description` at all — must not render `"description": null`.
TOOL_NO_DESC = [
    {
        "type": "function",
        "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
    }
]

U1 = "Find where the retry limit is set and bump it to 5."
U2 = "Great — make the change."

ASSIST_SEARCH = {
    "role": "assistant",
    "content": "I'll search the repo first.",
    "tool_calls": [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "search_repo",
                "arguments": json.dumps(
                    {
                        "pattern": "RETRY_LIMIT",
                        "mode": "content",
                        "opts": {"case_insensitive": True, "glob": "*.rs"},
                    }
                ),
            },
        }
    ],
}

ASSIST_PARALLEL = {
    "role": "assistant",
    "content": "Checking both call sites.",
    "tool_calls": [
        {
            "id": "call_a",
            "type": "function",
            "function": {
                "name": "bash",
                "arguments": json.dumps({"command": "grep -n RETRY_LIMIT src/net.rs", "timeout_ms": 5000}),
            },
        },
        {
            "id": "call_b",
            "type": "function",
            "function": {
                "name": "search_repo",
                "arguments": json.dumps({"pattern": "retry", "mode": "files"}),
            },
        },
    ],
}

TOOL_RESULT_A = {
    "role": "tool",
    "tool_call_id": "call_a",
    "content": "src/net.rs:42: const RETRY_LIMIT: u32 = 3;",
}
TOOL_RESULT_B = {
    "role": "tool",
    "tool_call_id": "call_b",
    "content": "src/net.rs\nsrc/retry.rs",
}

CASES = [
    # ---- no tools -------------------------------------------------------
    dict(name="no_tools_no_thinking", effort="off", tools=None,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="no_tools_thinking", effort="low", tools=None,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="no_tools_effort_high", effort="high", tools=None,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="no_tools_effort_max", effort="max", tools=None,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="no_tools_multiturn", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {"role": "assistant", "content": "The limit lives in src/net.rs."},
             {"role": "user", "content": U2},
         ]),
    dict(name="no_tools_two_system_messages", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "system", "content": "Never touch external/."},
             {"role": "user", "content": U1},
         ]),
    dict(name="no_system_message", effort="low", tools=None,
         messages=[{"role": "user", "content": U1}]),

    # ---- tools ----------------------------------------------------------
    dict(name="tools_first_user_turn", effort="low", tools=TOOLS,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="tools_no_thinking", effort="off", tools=TOOLS,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="tools_effort_high", effort="high", tools=TOOLS,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    dict(name="tools_effort_max", effort="max", tools=TOOLS,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),

    # The audit's 5-message agentic conversation (D1/D2/D4 all fire here).
    dict(name="tools_agentic_second_user_turn", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             ASSIST_SEARCH,
             TOOL_RESULT_A,
             {"role": "user", "content": U2},
         ]),
    dict(name="tools_agentic_no_thinking", effort="off", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             ASSIST_SEARCH,
             TOOL_RESULT_A,
             {"role": "user", "content": U2},
         ]),
    # Two parallel tool results, run NOT followed by a user turn.
    dict(name="tools_parallel_results", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             ASSIST_PARALLEL,
             TOOL_RESULT_A,
             TOOL_RESULT_B,
         ]),
    # Two parallel results FOLLOWED by a user turn (the `\n\n` merge, twice).
    dict(name="tools_parallel_results_then_user", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             ASSIST_PARALLEL,
             TOOL_RESULT_A,
             TOOL_RESULT_B,
             {"role": "user", "content": U2},
         ]),
    # Assistant history turn carrying reasoning_content (replayed inside
    # <think>…</think> when tools are present).
    dict(name="tools_assistant_reasoning_content", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             dict(ASSIST_SEARCH, reasoning_content="The constant is probably in the net module."),
             TOOL_RESULT_A,
             {"role": "user", "content": U2},
         ]),
    # Consecutive user turns (the in_user latch merges them with "\n\n").
    dict(name="two_user_turns_in_a_row", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {"role": "user", "content": "Actually, bump it to 7."},
         ]),
    # System message present but empty: the "\n\n" separator still shows up.
    dict(name="empty_system_with_tools", effort="low", tools=TOOLS,
         messages=[{"role": "system", "content": ""}, {"role": "user", "content": U1}]),
    dict(name="no_system_with_tools", effort="low", tools=TOOLS,
         messages=[{"role": "user", "content": U1}]),
    dict(name="tool_without_description", effort="low", tools=TOOL_NO_DESC,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    # Conversation ending on an assistant turn: add_generation_prompt still
    # opens a fresh assistant turn.
    dict(name="history_ends_on_assistant", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {"role": "assistant", "content": "Done."},
         ]),
    # D4 in isolation: a tool-result run followed by a user turn, with NO
    # `tools` array declared (the only no-tools divergence the audit found).
    dict(name="no_tools_tool_result_then_user", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             ASSIST_SEARCH,
             TOOL_RESULT_A,
             TOOL_RESULT_B,
             {"role": "user", "content": U2},
         ]),
    # Assistant turn whose predecessor is a system message: the template
    # emits no `<｜Assistant｜>` prefix at all.
    dict(name="assistant_after_system_no_prefix", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "assistant", "content": "Continuing from last time."},
             {"role": "user", "content": U1},
         ]),
    # No tools, history ends on an assistant turn: that turn is past
    # last_user_idx so it KEEPS its reasoning, and the generation prompt
    # still opens a new assistant turn.
    dict(name="no_tools_history_ends_on_assistant", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {"role": "assistant", "content": "It is in src/net.rs.",
              "reasoning_content": "Grep said net.rs."},
         ]),
    # Multimodal user turn (`dsv4_media` join rule): the image placeholder is
    # a real token id, and the "\n\n" separators fold into the neighbouring
    # text spans.
    dict(name="user_turn_with_image_parts", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": [
                 {"type": "text", "text": "What is in this screenshot?"},
                 {"type": "image_url",
                  "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}},
                 {"type": "text", "text": "Answer briefly."},
             ]},
         ]),
    # Text-only array-form `content`: `dsv4_media` joins EVERY part with
    # "\n\n", image or not. (The server's deserializer used to concatenate
    # text parts with no separator at all.)
    dict(name="user_turn_with_text_parts_only", effort="low", tools=None,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": [
                 {"type": "text", "text": "first part"},
                 {"type": "text", "text": "second part"},
             ]},
         ]),
    # Exponent-form floats in a tool schema: `json.dumps` float repr.
    dict(name="tool_schema_exponent_floats", effort="low", tools=TOOL_EXP_FLOATS,
         messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": U1}]),
    # Exponent-form floats inside a tool-call argument, both as a bare
    # `string="false"` value and nested inside a container.
    dict(name="tool_call_exponent_float_arguments", effort="low", tools=TOOL_EXP_FLOATS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {
                 "role": "assistant",
                 "content": "Tightening the tolerance.",
                 "tool_calls": [
                     {
                         "id": "c1",
                         "type": "function",
                         "function": {
                             "name": "tune",
                             "arguments": json.dumps(
                                 {"tol": 1e-07, "step": 1.5e-05,
                                  "bounds": [1e-20, 1e30, 0.0001]}
                             ),
                         },
                     }
                 ],
             },
         ]),
    # Tool-call arguments whose values are an object / an array / numbers and
    # booleans: `string="false"` + HF `tojson` spacing.
    dict(name="tool_call_container_arguments", effort="low", tools=TOOLS,
         messages=[
             {"role": "system", "content": SYSTEM},
             {"role": "user", "content": U1},
             {
                 "role": "assistant",
                 "content": "Running two greps.",
                 "tool_calls": [
                     {
                         "id": "c1",
                         "type": "function",
                         "function": {
                             "name": "search_repo",
                             "arguments": json.dumps(
                                 {
                                     "pattern": "retry",
                                     "opts": {"case_insensitive": False, "glob": "*.rs"},
                                     "limits": [1, 2, 3],
                                     "max": 12,
                                     "deep": True,
                                     "cutoff": None,
                                 }
                             ),
                         },
                     }
                 ],
             },
             TOOL_RESULT_A,
         ]),
]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

# Our ReasoningEffort -> the template's (thinking, reasoning_effort) pair.
EFFORT = {
    "off": (False, None),
    "low": (True, None),
    "high": (True, "high"),
    "max": (True, "max"),
}


def build_env():
    import jinja2
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
        # transformers' `tojson` binding (tokenization_utils_base.py).
        return json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys
        )

    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.filters["tojson"] = tojson
    env.filters["from_json"] = json.loads
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env


def main():
    gguf_path = sys.argv[1] if len(sys.argv) > 1 else (
        "/persist/lumi/models/dsv4f-exp-q2-k-xl/UD-Q2_K_XL/"
        "DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf"
    )
    kv = read_gguf_metadata(gguf_path)
    template_src = kv["tokenizer.chat_template"]
    tokens = kv["tokenizer.ggml.tokens"]
    merges = kv["tokenizer.ggml.merges"]
    bos_text = tokens[kv["tokenizer.ggml.bos_token_id"]]

    env = build_env()
    template = env.from_string(template_src)

    out_cases = []
    for c in CASES:
        thinking, effort = EFFORT[c["effort"]]
        text = template.render(
            messages=c["messages"],
            tools=c["tools"],
            bos_token=bos_text,
            add_generation_prompt=True,
            thinking=thinking,
            reasoning_effort=effort,
        )
        out_cases.append(
            {
                "name": c["name"],
                "effort": c["effort"],
                "messages": c["messages"],
                "tools": c["tools"],
                "canonical_text": text,
            }
        )

    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data = os.path.join(here, "crates", "deepstrix-server", "tests", "data")
    os.makedirs(data, exist_ok=True)

    cases_doc = {
        "_comment": "GENERATED by scripts/gen_tool_prompt_vectors.py — do not edit by hand.",
        "model_gguf": os.path.basename(gguf_path),
        "chat_template_sha256": hashlib.sha256(template_src.encode()).hexdigest(),
        "cases": out_cases,
    }
    with open(os.path.join(data, "tool_prompt_cases.json"), "w") as f:
        json.dump(cases_doc, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # ---- trimmed vocab ---------------------------------------------------
    universe = "\n".join(byte_encode(c["canonical_text"]) for c in out_cases)
    keep_ids = {}
    for i, t in enumerate(tokens):
        if not t:
            continue
        if len(t.encode()) == 1 or t in universe:
            keep_ids[i] = t
    # Structural ids the renderer pushes directly must always be present.
    for name in (
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
        "<｜deepseek_image｜>",
    ):
        keep_ids[tokens.index(name)] = name
    kept_merges = [m for m in merges if m.replace(" ", "", 1) in universe]

    vocab_doc = {
        "_comment": (
            "GENERATED by scripts/gen_tool_prompt_vectors.py — a TRIMMED BPE vocab, "
            "sufficient by construction to tokenize every canonical_text in "
            "tool_prompt_cases.json exactly as the full vocab would. Token ids are the "
            "model's real ids; merges keep their relative order (only the ordering "
            "matters to the merge loop)."
        ),
        "model_gguf": os.path.basename(gguf_path),
        "vocab_size": len(tokens),
        "bos_token_id": kv.get("tokenizer.ggml.bos_token_id"),
        "eos_token_id": kv.get("tokenizer.ggml.eos_token_id"),
        "pre": kv.get("tokenizer.ggml.pre"),
        "tokens": {str(i): t for i, t in sorted(keep_ids.items())},
        "merges": kept_merges,
    }
    with open(os.path.join(data, "tool_prompt_vocab.json"), "w") as f:
        json.dump(vocab_doc, f, ensure_ascii=False)
        f.write("\n")

    sys.stderr.write(
        f"{len(out_cases)} cases; vocab trimmed to {len(keep_ids)}/{len(tokens)} tokens, "
        f"{len(kept_merges)}/{len(merges)} merges\n"
    )


if __name__ == "__main__":
    main()
