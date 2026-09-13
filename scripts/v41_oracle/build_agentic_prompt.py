"""Render a realistic multi-turn agentic transcript (tool calls + results) in
DeepSeek-V4.1's own prompt format via the checkpoint's encoding.py, and save
its token ids for teacher-forced measurements (DSpark acceptance, expert traces).
"""
import json, os, sys
MODEL = os.path.expanduser("~/.cache/deepstrix/models/dsv4.1f")
sys.path.insert(0, os.path.join(MODEL, "encoding"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encoding import encode_messages  # noqa: E402
from oracle import _TokShim  # noqa: E402

tools = [{"type": "function", "function": {"name": "read_file", "description": "Read a file from the repository",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}},
         {"type": "function", "function": {"name": "run_shell", "description": "Run a shell command and return stdout",
          "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]}}}]
messages = [
    {"role": "system", "content": "You are a coding agent working in a Rust repository. Use tools to inspect the code before editing. Be concise."},
    {"role": "user", "content": "The test `weight_src_arms` fails with 'range [0,6266880) exceeds tensor length'. Find the cause and fix it."},
    {"role": "assistant", "reasoning_content": "The error text comes from the safetensors reader's bounds check. I should look at how read_expert_into computes the offset for the HF arm.",
     "content": "I'll look at the reader's range check first.",
     "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"cmd": "grep -n 'exceeds tensor length' -r crates/v4flash-core/src"})}}]},
    {"role": "tool", "content": "crates/v4flash-core/src/safetensors.rs:181:                \"{}: range [{byte_off},{end}) exceeds tensor length {}\","},
    {"role": "assistant", "reasoning_content": "So the bound is in read_range_into. The caller passes an expert offset; if the descriptor's len is the packed size rather than the repacked size, the check fails for the last expert.",
     "content": "The check is in `read_range_into`. Let me see the caller in the HF view.",
     "tool_calls": [{"type": "function", "function": {"name": "read_file", "arguments": json.dumps({"path": "crates/v4flash-core/src/hf_v41.rs"})}}]},
    {"role": "tool", "content": "pub fn read_expert_into(&self, vt: &VTensor, e: usize, dst: &mut [u8]) -> eyre::Result<()> {\n    let Kind::Experts { layer, which } = &vt.kind else { return Err(eyre!(\"{}: not a stacked expert tensor\", vt.name)); };\n    if e >= self.n_expert { return Err(eyre!(\"{}: expert {e} >= {}\", vt.name, self.n_expert)); }\n    if dst.len() != self.expert_bytes(vt) { return Err(eyre!(\"dst len mismatch\")); }\n    self.read_expert_raw(*layer, which, e, dst)\n}"},
    {"role": "assistant", "reasoning_content": "read_expert_into itself is fine. The failing range must come from the GGUF arm, where abs_offset + e * dst.len() is used, and the test builds dst with the HF byte count while the fixture GGUF was written with a different expert count. So the fixture and the view disagree on n_expert.",
     "content": "The reader is consistent; the mismatch is between the fixture's expert count and the view's. The test should pass the same `--experts` value used to build the fixture. I'll check the fixture's dims.",
     "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"cmd": "python3 -c \"import gguf; r=gguf.GGUFReader('fixtures/v41-dry-00001-of-00001.gguf'); t=[x for x in r.tensors if x.name=='blk.0.ffn_down_exps.weight'][0]; print(t.shape)\""})}}]},
    {"role": "tool", "content": "[2304 5120 8]"},
    {"role": "assistant", "content": "The fixture holds 8 experts while the view was opened with the full 384, so the per-expert byte count differs. Fix: open the HF view with `Some(experts)` where `experts` is read from the fixture's third dimension instead of an environment default.\n\n```rust\nlet experts = t.dims[2] as usize;\nlet hf = V41HfWeights::open(&dir, Some(experts))?;\n```\n\nI'll apply that and rerun the test."},
    {"role": "user", "content": "Do it, then also make the message say which side is wrong."},
    {"role": "assistant", "reasoning_content": "Two edits: derive experts from the fixture dims, and improve the error text in read_range_into to include the tensor name and both lengths.",
     "content": "Applying both changes.",
     "tool_calls": [{"type": "function", "function": {"name": "run_shell", "arguments": json.dumps({"cmd": "cargo test -p v4flash-core --release --test weight_src_arms -- --nocapture"})}}]},
    {"role": "tool", "content": "test gguf_arm_is_transparent_and_hf_arm_agrees ... ok\nOK: 25 tensors through both arms"},
    {"role": "assistant", "content": "Done. The test derives the expert count from the fixture now and passes for all 25 tensors. The bounds error also names the tensor and both lengths, so the next mismatch will say which side is short."},
]
prompt = None
for mode in ("thinking", "chat", "non-thinking"):
    try:
        prompt = encode_messages(messages, mode, context=tools, drop_thinking=False)
        print("thinking_mode:", mode)
        break
    except Exception as ex:
        print("mode", mode, "failed:", str(ex)[:120])
if prompt is None:
    prompt = encode_messages(messages, "thinking")
tok = _TokShim(os.path.join(MODEL, "tokenizer.json"))
ids = [0] + tok.backend_tokenizer.encode(prompt, add_special_tokens=False).ids if hasattr(tok.backend_tokenizer, "encode") else None
out = os.path.expanduser("~/.cache/deepstrix/v41/agentic")
json.dump(ids, open(os.path.join(out, "tokens.json"), "w"))
open(os.path.join(out, "prompt.txt"), "w").write(prompt)
print("tokens:", len(ids), "chars:", len(prompt)); print(prompt[:300].replace("\n", "\\n"))
