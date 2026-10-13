"""isa_count.py <asm.s> <kernel> <start-regex> <end-regex> -- count instructions by opcode between the
first line matching start-regex and the next line matching end-regex inside <kernel>."""
import re
import sys
from collections import Counter

path, kern, a, b = sys.argv[1:5]
lines = open(path).read().splitlines()
inside = False
body = []
for ln in lines:
    if ln.startswith(kern + ":"):
        inside = True
        continue
    if inside:
        if ln.startswith(".Lfunc_end"):
            break
        body.append(ln)
i0 = next(i for i, l in enumerate(body) if re.search(a, l))
i1 = next(i for i in range(i0 + 1, len(body)) if re.search(b, body[i]))
seg = [l.strip() for l in body[i0:i1 + 1] if l.strip() and not l.strip().startswith((";", "."))]
ops = Counter(l.split()[0] for l in seg)
print(f"lines {i0}..{i1}: {len(seg)} instructions, VALU {sum(v for k, v in ops.items() if k.startswith('v_'))}")
for k, v in ops.most_common(40):
    print(f"  {v:5d} {k}")
