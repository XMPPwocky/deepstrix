import re,sys
SP="/tmp/claude-1000/-home-claude-code-deepstrix/e6bbeb19-ceab-4950-b666-df1b6c344139/scratchpad"
ansi=re.compile(r'\x1b\[[0-9;]*m')
L=[ansi.sub('',l) for l in open(SP+"/"+sys.argv[1]) if "dspark.step" in l]
use=L[len(L)//2:]; n=len(use)
keys=["fwd_ms","argmax_ms","roll_ms","draft_ms","step_ms"]
agg={k:0.0 for k in keys}
for l in use:
    for k in keys:
        m=re.search(k+r'="([\d.]+)"',l)
        if m: agg[k]+=float(m.group(1))
print(f"{sys.argv[1]:14s} steps={n}  " + "  ".join(f"{k[:-3]}={agg[k]/n:7.1f}" for k in keys))
