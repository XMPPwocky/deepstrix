import json, sys, urllib.request, time
port = 18155
def ask(prompt, max_tokens=24, legacy=False, label=""):
    body = {
        "model": "deepseek-v4.1-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    out = []
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                except Exception:
                    continue
                for ch in d.get("choices", []):
                    t = ch.get("delta", {}).get("content")
                    if t:
                        out.append(t)
    except Exception as e:
        body = getattr(e, "read", lambda: b"")()
        print(f"[{label}] HTTP FAILURE after {time.time()-t0:.1f}s: {e}\n{body[:2000]!r}")
        return None
    txt = "".join(out)
    print(f"[{label}] {time.time()-t0:.1f}s -> {txt!r}")
    return txt

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "smoke":
        ask("What is the capital of France? Answer with one word.", 16, label="france")
        ask("What is 17 x 23? Answer with the number only.", 24, label="mul")
    else:
        p = open(sys.argv[2]).read()
        ask(p, 16, label=sys.argv[3])
