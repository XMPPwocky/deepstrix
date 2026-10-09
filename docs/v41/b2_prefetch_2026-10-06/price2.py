#!/usr/bin/env python3
"""Rev-2 pricing of predicted-miss hints on the MEASURED speculative-read path.
Inputs: b2tail price.log 'paged: removed if X us earlier' per cell (ms/step), regime shares
(whatif.py SH), lane periods (late.log moe medians x2 for two lanes), spec_reads.log walls.
A caught hint saves min(lead, read) when the reader set is idle, lead x (2.26/6.7) when busy
(the chunked read progresses at ~1/3 speed until the demand ensure promotes it)."""
import bisect
REM = {  # X us earlier -> paged lateness removed, ms/step
 'lone_r3': {100: 0.35, 300: 1.04, 500: 1.73, 1000: 3.46, 2000: 6.7},
 'lone_r4': {100: 0.34, 300: 1.01, 500: 1.68, 1000: 3.34, 2000: 6.14},
 'lone_r5': {100: 0.42, 300: 1.27, 500: 2.11, 1000: 4.19, 2000: 7.33},
 'lone_r6': {100: 0.45, 300: 1.36, 500: 2.26, 1000: 4.42, 2000: 7.49},
 'plain_M3': {100: 0.41, 300: 1.24, 500: 2.07, 1000: 4.14, 2000: 8.04},
 'two_r7': {100: 0.61, 300: 1.8, 500: 2.98, 1000: 5.74, 2000: 9.19},
 'plain_M8': {100: 0.76, 300: 2.24, 500: 3.63, 1000: 6.6, 2000: 10.05},
}
PAGED = {'lone_r3': 8.72, 'lone_r4': 8.35, 'lone_r5': 10.84, 'lone_r6': 10.32, 'plain_M3': 11.54, 'two_r7': 13.36, 'plain_M8': 14.96}
# regime shares of step wall (whatif.py SH; M4-M7 folded onto M3/M8 by rows)
SH = {'lone_r3': 0.483 * 331 / 2082, 'lone_r4': 0.483 * 1019 / 2082, 'lone_r5': 0.483 * 225 / 2082, 'lone_r6': 0.483 * 507 / 2082,
      'two_r7': 0.123, 'plain_M3': 0.141 + 0.063 + 0.5 * 0.078, 'plain_M8': 0.013 + 0.043 + 0.038 + 0.5 * 0.078}
# lane period (ms): two lanes = 2 x MoE median (late.log); single lane (<4 rows) ~1.4
PERIOD = {'lone_r3': 1.4, 'lone_r4': 1.66, 'lone_r5': 1.94, 'lone_r6': 2.1, 'plain_M3': 1.4, 'two_r7': 2.5, 'plain_M8': 3.2}
HANDLING_DEQ = 0.10     # link ~0.1; dequeue->hints_end is 0.26 US (round-2 finding 3: evtrace stamps are ns)
HANDLING_ARR = 0.10     # applied at frame arrival: same at the median; the gain is the b2q tail (~0.1-0.2 ms/step)
NORM = 0.71             # paged-only ceiling: the sim's +9.0 hides ALL late replies; paged share ~0.87 -> 7.9 ms; 11.1 -> 7.9
READ = 2.26             # isolated speculative / certain read, ms
BUSY_WALL = 6.7         # busy speculative wall, ms
STEP = {'lone_r3': 73.9, 'lone_r4': 81.2, 'lone_r5': 92.9, 'lone_r6': 100.8, 'plain_M3': 75.5, 'two_r7': 119.7, 'plain_M8': 144.6}

def removed(cell, x_ms):
    t = REM[cell]; xs = sorted(t); x = x_ms * 1000
    if x <= 0: return 0.0
    if x >= xs[-1]: return min(PAGED[cell], t[xs[-1]] + (t[xs[-1]] - t[xs[-2]]) / (xs[-1] - xs[-2]) * (x - xs[-1]))
    i = bisect.bisect_left(xs, x)
    if i == 0: return t[xs[0]] * x / xs[0]
    a, b = xs[i - 1], xs[i]
    return t[a] + (t[b] - t[a]) * (x - a) / (b - a)

def price(label, recall, pb, handling, k2=False, prec=0.97, not_started=0.0):
    tot = 0.0; totpct = 0.0; row = []
    for c in REM:
        lead = PERIOD[c] - handling + (PERIOD[c] if k2 else 0.0)
        gain_idle = min(lead, READ)
        gain_busy = min(lead * READ / BUSY_WALL, READ)
        gain = ((1 - pb) * gain_idle + pb * gain_busy) * (1 - not_started)
        x = gain * recall * prec
        r = removed(c, x)
        row.append(f"{c} {r:.1f}")
        tot += SH[c] * r; totpct += SH[c] * r / STEP[c]
    w = sum(SH.values())
    print(f"{label:58s} weighted {NORM * tot / w:4.1f} ms/step ({100 * NORM * totpct / w:.1f}%) raw {tot / w:4.1f} | " + ' '.join(row))

print("per-cell ms/step removed; weighted by regime share\n")
for R, rec in (('R=1', 0.71), ('R=2', 0.86), ('R=3', 0.91)):
    print(f"-- {R} (rank-1 recall {rec})")
    price(f"(b) hub-only, busy share 0.23 (time-weighted bound), 10% never started", rec, 0.23, HANDLING_DEQ, not_started=0.10)
    price(f"(b) hub-only, busy share 0.60 (reviewer: per-read share)", rec, 0.60, HANDLING_DEQ, not_started=0.10)
    price(f"(a) daemon hint class, applied at dequeue", rec, 0.0, HANDLING_DEQ)
    price(f"(a) daemon hint class, applied at frame arrival", rec, 0.0, HANDLING_ARR)
    price(f"(a) + k2 (recall x0.9 on the remainder approx)", rec * 0.95, 0.0, HANDLING_ARR, k2=True)
print("\nceiling: every paged reply never late =", f"{NORM * sum(SH[c] * PAGED[c] for c in REM) / sum(SH.values()):.1f} ms/step weighted (normalised; raw {sum(SH[c] * PAGED[c] for c in REM) / sum(SH.values()):.1f}); per-cell values are raw, quote +-15%")
