# Tool-prompt fidelity audit — `render_prompt` vs the GGUF chat template

**Date:** 2026-09-04 · **Repo HEAD:** `21acf48` · **ds4 submodule:** `b8b2351`
**Model audited:** `/persist/lumi/models/dsv4f-exp-q2-k-xl/UD-Q2_K_XL/DeepSeek-V4-Flash-Vision-Exp-UD-Q2_K_XL-00001-of-00003.gguf`
(`tokenizer.chat_template`, 14 679 bytes, re-extracted from the live GGUF; byte-identical to the
saved `vis_chat_template.jinja`).

## Verdict

**DIVERGES — materially, in the tool paths specifically.**

With `tools` absent our render is byte-identical to the canonical template for the same
conversation except for one tool-result/user-run merge rule. With `tools` present, the block the
model reads to learn *what a tool call looks like* is a different data format from the one the
model was trained on, and every historical assistant turn is missing one special token.

Same 5-message agentic conversation, 3 tools, thinking on:

| | bytes | tokens |
|---|---|---|
| canonical (GGUF jinja) | 2 659 | **681** |
| ours (`render_prompt`) | 4 140 | **933** |

Token LCP between the two id streams: **13** (the divergence starts inside the system block).

Three of the findings diverge from **both** the GGUF template *and* `external/ds4`'s C — i.e. they
are not "we faithfully mirror ds4 and ds4 is the odd one out"; they are ours alone.

## Method

* Canonical: `tokenizer.chat_template` extracted from the live GGUF, rendered with jinja2 3.1.6
  (`nix-shell -p python3Packages.jinja2`), `tojson` bound to HF-transformers semantics
  (`json.dumps(..., ensure_ascii=False)`), `add_generation_prompt=True`.
* Ours: `crates/deepstrix-server/src/prompt.rs::render_prompt` + `dsml.rs::render_tools_prompt` /
  `render_tool_calls_in_history` / `build_tool_result_text` re-implemented line-by-line in Python
  (the repo was read-only for this audit, so no Rust test was added — the port follows the Rust
  statement-for-statement, including the `serde_json` `preserve_order` feature that is enabled in
  the workspace `Cargo.toml:23`).
* Tokenizer: `crates/v4flash-core/src/tokenizer.rs` (joyai pre-tokenizer + GPT-2 byte encoding +
  BPE) ported to Python against `tokenizer.ggml.tokens` / `.merges` from the same GGUF.
  Validated by exact round-trip decode and by reproducing the documented
  `｜DSML｜ → 28217 10525 7398 28217` split and `dsml_id = 128825`.
* ds4 reference read directly: `external/ds4/ds4_server.c` `render_chat_prompt_text` (1901),
  `append_tools_prompt_text` (1646), `parse_tools_value` (1211), `append_raw_json_line` (1074).

## Vision-Exp vs 0731 template

Diffed in full. The **only** difference is the `dsv4_media` macro (multimodal `content` arrays) and
its two call sites. **Every tool path — `tools_header`, `tools_footer`, the schema loop, the
`tool_calls` emitter, the `tool` role, `tp.has` — is byte-identical between the two.** No
tool-related change came in with Vision-Exp.

## Divergence table

Severity: **B** = would-change-behaviour, **C** = cosmetic / edge-case only.

| # | What | Ours | Canonical | ds4 | Sev |
|---|---|---|---|---|---|
| D1 | Tool-schema serialization format | pretty-printed JSON **array** of `{"type","function":{…}}`, 2-space indent | **one compact function object per line**, unwrapped, no array, no indent | same as canonical | **B** |
| D2 | History assistant turn opener (tools present, thinking on) | `<｜Assistant｜></think>` | `<｜Assistant｜><think>…</think>` | same as canonical | **B** |
| D3 | "thinking mode" sentence in the tools header | paraphrase | `If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.` | same as canonical | **B** |
| D4 | Tool-result run merging | new `<｜User｜>` per run boundary, results concatenated with no separator | one `<｜User｜>`, results and following user text joined by `\n\n` | same as ours | **B** |
| D5 | Tools-block footer | `\n\n` + sentence + ` Use the exact parameter names from the schemas.` (no trailing NL) | `\n` + sentence + `\n` | same as ours | C |
| D6 | String-parameter paragraph | ds4's 3-sentence anti-entity-escape text (+69 tok) | 2 sentences | same as ours | C |
| D7 | Token boundary at system↔tools seam | `.` + `\n\n` (2 encode calls) | `.\n\n` (1 token) | n/a (ds4 tokenizes once) | C |
| D8 | `"help answer the user question"` | missing `'s` | `the user's question` | same as ours | C |
| D9 | `reasoning_content` on assistant history | field does not exist on `ChatMessage`; silently dropped | replayed inside `<think>…</think>` | `m->reasoning` replayed | C→B if a client sends it |
| D10 | Extra tool-def fields (`strict`, …) | dropped; `"description": null` emitted when absent; key order forced to name/description/parameters | client's raw JSON preserved verbatim | raw bytes preserved | C |
| D11 | `developer` / `function` roles | **HTTP 400** (`Role` enum has only system/user/assistant/tool) | `developer` handled | `function` handled | C |
| D12 | `tool_choice` | not parsed at all | n/a | `"none"` drops the whole tools block | C |
| D13 | Empty-but-present system message + tools | no `\n\n` separator | still emits `\n\n` | same as canonical | C |
| D14 | `</tool_result>` inside tool output | defanged to `&lt;` | inserted raw | same as ours | C (safety-positive) |

Effort preamble placement was checked and is **correct**: gated on thinking, emitted immediately
after BOS and before the system text, exactly as the template's `{{- bos_token -}}` →
`reasoning_effort_high/max` → `{{- ns.system_prompt -}}` ordering. `high`/`max` strings are
byte-identical to `reasoning_effort_high` / `reasoning_effort_max` in the template. No divergence.

## Worst divergence (D1) — exact bytes

`dsml.rs:60`:

```rust
let schemas = serde_json::to_string_pretty(tools).unwrap_or_else(|_| "[]".into());
```

Template (`tools_header` … `### Available Tool Schemas\n\n`, then):

```jinja
{%- for tool in tools -%}
  {%- if tool['type'] == 'function' -%}
    {%- set ts.schemas = ts.schemas + (tool['function'] | tojson) + '\n' -%}
```

ds4 (`parse_tools_value` 1211 → `openai_function_schema_from_tool` 1080 →
`append_raw_json_line` 1074): unwraps `{"type":"function","function":{…}}` to the inner object and
emits the client's **raw JSON bytes, one per line**.

Canonical bytes for our 3-tool set (1 050 B / **303 tokens**):

```
{"name": "bash", "description": "Run a shell command in the project sandbox.", "parameters": {"type": "object", "properties": {"command": {"type": "string", …}}, "required": ["command"]}}
{"name": "search_repo", …}
{"name": "edit_file", …}
```

Our bytes (2 255 B / **491 tokens**, +62 %):

```
[
  {
    "type": "function",
    "function": {
      "name": "bash",
      "description": "Run a shell command in the project sandbox.",
      "parameters": {
        "type": "object",
        "properties": {
          "command": {
            "type": "string",
…
```

Four independent deltas in one: (a) a wrapping array with `[` / `]` / `,` lines, (b) the
`"type":"function"` envelope the reference strips, (c) newline+indent between every key, (d) a
different token stream for identical semantics. This is the block immediately preceding
`You MUST strictly follow the above defined tool name and parameter schemas` — the model's
in-context definition of the tool surface. The pretty-printed form is off-distribution.

*Caveat on JSON spacing:* the exact separator (`", "` vs `","`) is client-dependent under ds4
(raw passthrough) and `', '` under HF `tojson`. What is unambiguous and agreed by both references
is: **one function object per line, unwrapped, not indented, not inside an array.**

## D2 — the "stops calling tools after the first turn" candidate

Template, assistant branch:

```jinja
{%- set keep_reasoning = tp.has or (loop.index0 > last_user_idx.value) -%}
…
{{- '<｜Assistant｜>' -}}
{%- if keep_reasoning and thinking -%}
  {{- thinking_start_token -}}{# + reasoning_content #}{{- thinking_end_token -}}
{%- else -%}
  {{- thinking_end_token -}}
{%- endif -%}
```

`tp.has` is true whenever `tools` is present, so with tools **every** historical assistant turn
opens `<｜Assistant｜><think>…</think>`. ds4 has the same rule
(`if (tool_context || i > last_user_idx)`, `ds4_server.c:1953`).

`prompt.rs:319-320` unconditionally emits `TOK_ASSISTANT, TOK_THINK_END`. So each prior assistant
turn is missing token `128821` (`<think>`), and the shape the model sees for a turn that *did*
produce a tool call is not the shape it is currently being asked to produce (the open turn does
get `<think>`). Cost: 1 token per historical assistant turn; effect: the in-context exemplars of
"assistant turn that calls a tool" are structurally different from the turn being generated.

Observed (with tools, thinking on):

```
canon: …<｜Assistant｜><think></think>I'll search the repo first.\n\n<｜DSML｜tool_calls>…
ours : …<｜Assistant｜></think>I'll search the repo first.\n\n<｜DSML｜tool_calls>…
```

## D4 — tool-result runs

Parallel tool calls (two `role:"tool"` messages), and a `tool` message followed by a `user`
message:

```
canon: <｜User｜><tool_result>AAA</tool_result>\n\n<tool_result>BBB</tool_result><｜Assistant｜><think>
ours : <｜User｜><tool_result>AAA</tool_result><tool_result>BBB</tool_result><｜Assistant｜><think>

canon: <｜User｜><tool_result>…</tool_result>\n\nNow make the change.<｜Assistant｜><think>
ours : <｜User｜><tool_result>…</tool_result><｜User｜>Now make the change.<｜Assistant｜><think>
```

The template's `state.in_user` latch is set by `user` **and** `tool` and only cleared by
`assistant` / `developer` / `latest_reminder`; ours (`prompt.rs:283`) clears it on every `user`
message and never emits the `\n\n`. In the no-tools render this is the **only** difference between
ours and canonical, so it is the cleanest single fix.

## Recommended fixes

| # | File:line | Fix |
|---|---|---|
| D1 | `crates/deepstrix-server/src/dsml.rs:60` | Replace `to_string_pretty(tools)` with a per-tool loop emitting `serde_json::to_string(&t.function)` + `'\n'` for each `t.kind == "function"`. To also fix D10, keep the raw `serde_json::Value` of each tool (add `#[serde(flatten)] extra` or change `ToolDef.function` to `serde_json::Value`) so client key order and extra keys survive. |
| D2 | `crates/deepstrix-server/src/prompt.rs:319-320` | Emit `TOK_THINK_BEGIN` then any `reasoning_content` then `TOK_THINK_END` when `effort.thinking_enabled() && (tools_present \|\| msg_index > last_user_like_index)`; else keep `TOK_THINK_END` alone. Compute `last_user_like_index` (user/tool) in a pre-pass, mirroring `ds4_server.c:1912-1915`. |
| D3 | `crates/deepstrix-server/src/dsml.rs:56` | Restore the canonical sentence: `If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.` (reverts the wording change in `52f69d1`). |
| D4 | `crates/deepstrix-server/src/prompt.rs:265-292` | Rename `user_tool_block_open` to an `in_user` latch set by **both** `Role::User` and `Role::Tool`, cleared only by `Role::Assistant`; when already set, emit `vocab.encode("\n\n")` instead of `TOK_USER`. |
| D5 | `dsml.rs:62-65` | Footer → `"\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n"` (drop the extra sentence, single leading NL, add trailing NL). |
| D6 | `dsml.rs:52-55` | Reduce to `String parameters should be specified as is and set \`string="true"\`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set \`string="false"\`.` |
| D7 | `prompt.rs:242-250` | Build one `String` (`system + "\n\n" + tools_block`) and pass it through a single `encode_text`, so the seam tokenizes as the template does. |
| D8 | `dsml.rs:41` | `the user question` → `the user's question`. |
| D9 | `crates/deepstrix-server/src/openai/types.rs:36` | Add `#[serde(default)] pub reasoning_content: Option<String>` to `ChatMessage`; emit it inside the D2 `<think>…</think>`. |
| D11 | `types.rs:13` | `#[serde(alias = "developer")] System` (llama.cpp remaps developer→system) and `#[serde(alias = "function")] Tool`. |
| D12 | `types.rs` / `handler.rs:92` | Parse `tool_choice`; when `"none"`, pass `None` for tools (matches `ds4_server.c:2130`). |
| D13 | `prompt.rs:216-224` | Track "a system message was present" separately from "system text is non-empty", and emit the `\n\n` separator on presence. |

Fix order by expected effect on tool-calling propensity: **D1 → D2 → D3 → D4**, then the rest.

## What was checked and found clean

* Effort preamble text and position (both `high` and `max`, byte-identical, correct slot).
* `<｜DSML｜>` emitted as the real special token id `128825`, not the 4-token BPE split, in both the
  schema block and re-rendered history tool calls (`encode_with_special_marker`, `prompt.rs:136`).
* Tool-call markup shape in assistant history: `\n\n<｜DSML｜tool_calls>\n` / `<｜DSML｜invoke
  name="…">\n` / `<｜DSML｜parameter name="…" string="true|false">…</｜DSML｜parameter>\n` /
  `</｜DSML｜invoke>\n` / `</｜DSML｜tool_calls>` — byte-identical to the template's emitter,
  including `string="true"` for JSON strings and compact `tojson` for everything else, and
  including the argument key order (`preserve_order` is on).
* `<tool_result>` / `</tool_result>` wrapper names and placement inside a `<｜User｜>` turn.
* Tools attached to the system turn (not their own turn), after the system text, separated by
  `\n\n`; multiple system messages joined by `\n\n`.
* BOS / EOS / `<｜User｜>` / `<｜Assistant｜>` / `<think>` / `</think>` ids (0, 1, 128803, 128804,
  128821, 128822) all match the GGUF vocab.
* Vision-Exp vs 0731 template: no tool-path difference whatsoever.
* Trailing open turn: `<｜Assistant｜><think>` (thinking) / `<｜Assistant｜></think>` (off) matches
  `add_generation_prompt`.
