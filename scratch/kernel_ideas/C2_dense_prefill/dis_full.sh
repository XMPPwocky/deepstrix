#!/usr/bin/env bash
# dis_full.sh <hsaco> <kernel> <out.s> : full disassembly of one kernel (isa.sh --dis caps at 400 lines)
set -u
cd "$(dirname "$0")"
bash ../_infra/in_env.sh llvm-objdump -d --mcpu=gfx1201 "$1" > "$1.full.s" 2>/dev/null
python3 - "$1.full.s" "$2" "$3" <<'EOF'
import sys, re
lines = open(sys.argv[1]).read().split('\n')
out = []; on = False
for l in lines:
    m = re.match(r'^[0-9a-f]+ <(.*)>:$', l)
    if m:
        on = (m.group(1) == sys.argv[2])
        continue
    if on and l.strip(): out.append(l)
open(sys.argv[3], 'w').write('\n'.join(out) + '\n')
print(sys.argv[3], len(out), 'lines')
EOF
