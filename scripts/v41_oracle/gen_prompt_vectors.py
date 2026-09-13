"""Golden prompt vectors for the V4.1 chat encoding, straight from DeepSeek's
reference encoder (encoding/encoding.py): the checkpoint's five shipped cases
plus edge cases (chat vs thinking, numeric/named effort, tools, mid-conversation
system, latest_reminder, response_format, multiple tool results, drop_thinking
with and without tools). Output: crates/deepstrix-server/tests/fixtures/v41_prompts.json
"""
import json, os, sys, copy
MODEL = os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f")
sys.path.insert(0, os.path.join(MODEL, "encoding"))
from encoding import encode_messages, load_cases  # noqa: E402

cases = []
for i in range(1, 6):
    for c in load_cases(os.path.join(MODEL, "encoding", "tests", f"test_input_{i}.json")):
        cases.append(dict(name=f"shipped_{i}", messages=c["messages"], context=c["context"],
                          thinking_mode=c["thinking_mode"] or "thinking", reasoning_effort=c["reasoning_effort"]))
tool = {"type": "function", "function": {"name": "read_file", "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "lines": {"type": "integer"}}, "required": ["path"]}}}
base = [{"role": "system", "content": "You are terse."}, {"role": "user", "content": "Hi there"}]
edge = {
    "chat_mode": (base, "chat", None, None),
    "thinking_default_effort": (base, "thinking", None, None),
    "thinking_effort_int_33": (base, "thinking", None, 33),
    "thinking_effort_max": (base, "thinking", None, "max"),
    "no_system": ([{"role": "user", "content": "Only a user turn"}], "thinking", None, None),
    "assistant_history_dropped": ([{"role": "user", "content": "q1"}, {"role": "assistant", "reasoning_content": "r1", "content": "a1"},
                                   {"role": "user", "content": "q2"}], "thinking", None, None),
    "assistant_history_kept_with_tools": ([{"role": "system", "content": "sys", "tools": [tool]}, {"role": "user", "content": "q1"},
                                           {"role": "assistant", "reasoning_content": "r1", "content": "a1"}, {"role": "user", "content": "q2"}], "thinking", None, None),
    "tool_call_and_two_results": ([{"role": "system", "content": "sys", "tools": [tool]}, {"role": "user", "content": "read two files"},
                                   {"role": "assistant", "reasoning_content": "plan", "content": "Reading.", "tool_calls": [
                                       {"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "a.rs", "lines": 3})}},
                                       {"id": "c2", "type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "b.rs"})}}]},
                                   {"role": "tool", "tool_call_id": "c2", "content": "fn b() {}"},
                                   {"role": "tool", "tool_call_id": "c1", "content": "fn a() {}"}], "thinking", None, None),
    "tool_result_then_user_text": ([{"role": "system", "content": "sys", "tools": [tool]}, {"role": "user", "content": "go"},
                                    {"role": "assistant", "content": "ok", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"x\"}"}}]},
                                    {"role": "tool", "tool_call_id": "c1", "content": "X"},
                                    {"role": "user", "content": "and now summarize"}], "thinking", None, None),
    "mid_conversation_system": ([{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                                 {"role": "system", "content": "New rule: be brief."}], "thinking", None, None),
    "latest_reminder": ([{"role": "user", "content": "q1"}, {"role": "latest_reminder", "content": "Remember X."}], "thinking", None, None),
    "response_format": ([{"role": "system", "content": "sys", "response_format": {"type": "object", "properties": {"ok": {"type": "boolean"}}}},
                         {"role": "user", "content": "yes?"}], "chat", None, None),
    "unicode_and_ws": ([{"role": "user", "content": "  中文 🚀\n\nsecond paragraph\t tabbed "}], "thinking", None, 7),
    "with_context": ([{"role": "user", "content": "new turn"}], "thinking", [{"role": "user", "content": "old turn"}, {"role": "assistant", "content": "old answer"}], None),
}
for name, (msgs, mode, ctx, eff) in edge.items():
    cases.append(dict(name=name, messages=msgs, context=ctx, thinking_mode=mode, reasoning_effort=eff))
out = []
for c in cases:
    try:
        prompt = encode_messages(copy.deepcopy(c["messages"]), c["thinking_mode"], context=copy.deepcopy(c["context"]) if c["context"] else None,
                                 reasoning_effort=c["reasoning_effort"])
    except Exception as ex:
        prompt = None; c["error"] = f"{type(ex).__name__}: {ex}"
    out.append(dict(c, expected=prompt))
    print(f"{c['name']:36} {'ERR ' + c['error'] if prompt is None else str(len(prompt)) + ' chars'}")
json.dump(out, open("crates/deepstrix-server/tests/fixtures/v41_prompts.json", "w"), ensure_ascii=False, indent=1)
print("wrote", len(out), "cases")
