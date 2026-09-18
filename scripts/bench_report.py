#!/usr/bin/env python3
"""Score a bench.sh run and REFUSE to report if the controls disagree."""
import sys, re, statistics, pathlib
SC = pathlib.Path(__file__).parent
name = sys.argv[1]
tol = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08

def series(log):
    if not log.exists(): return None
    txt = re.sub(r'\x1b\[[0-9;]*m', '', log.read_text(errors='ignore'))
    ms = [float(x) for x in re.findall(r'ms_per_tok="([0-9.]+)"', txt)]
    e  = [float(x) for x in re.findall(r'e_tokens_per_step="([0-9.]+)"', txt)]
    return ms, e

arms = []
for p in sorted(SC.glob(f"{name}_*.log")):
    arm = p.stem[len(name)+1:]
    s = series(p)
    if s and s[0]: arms.append((arm, s[0], s[1]))

if not arms:
    print(f"no arms found for {name}"); sys.exit(1)

# the warm value: drop the first request (cold), take the min of the rest
def warm(ms): return min(ms[1:]) if len(ms) > 1 else ms[0]

print(f"{'arm':<26}{'E':>8}{'warm ms/tok':>13}{'all':>28}")
for arm, ms, e in arms:
    ev = e[-1] if e else float('nan')
    print(f"{arm:<26}{ev:>8.3f}{warm(ms):>13.2f}   {' '.join(f'{m:.1f}' for m in ms)}")

base = [a for a in arms if not a[0].endswith('_CTRL')]
ctrl = [a for a in arms if a[0].endswith('_CTRL')]
if not ctrl:
    print("\nNO CONTROL ARM — result is not usable."); sys.exit(2)
a0, c0 = warm(base[0][1]), warm(ctrl[0][1])
drift = abs(a0 - c0) / max(a0, c0)
print(f"\ncontrol check: {base[0][0]} {a0:.2f} vs {ctrl[0][0]} {c0:.2f}  ->  drift {drift*100:.1f}%")
if drift > tol:
    print(f"*** UNUSABLE: controls disagree by more than {tol*100:.0f}%. Any difference between")
    print("*** arms is inside the drift. Do not report a winner from this run.")
    sys.exit(3)
print(f"controls agree within {tol*100:.0f}% — arms are comparable.")
for arm, ms, e in base[1:]:
    d = (warm(ms) - a0) / a0
    print(f"  {arm:<24} {d*100:+.1f}% vs {base[0][0]}")
