# The USB4 link costs ~575 us/layer because it goes to sleep between layers
### measured 2026-09-14, prompted by a user observation in a perfetto trace

Decode submits to box 2 once per layer and then leaves the link idle for ~5.5 ms
while box 1 runs the next layer's dGPU chain (246 ms/token / 40 layers = 6.15 ms,
of which box 2's own service is ~0.4 ms). That idle is not free.

## Measurement

`deepstrix-expert-bench --gap-us N` spins N us between a reply and the next
request, reproducing the duty cycle. B=1, picks 3, pool 3 (so D is constant and
no disk is involved), 250 iters, p50 us:

    gap us      rtt    srv    link        <- link = rtt - srv
         0      434    386      47
       200      520    409     120
       500     1255    408     843        <- cliff between 200 and 500
      1000     1552    407    1155
      2000     1370    407     966
      5000      994    407     594        <- the real decode regime

**`srv` is flat at ~407 us across the entire sweep.** Box 2's iGPU does not
downclock; the penalty is entirely in the link. This rules out GPU DPM as a
lever — `power_dpm_force_performance_level` is `auto` on box 2 and changing it
would measure nothing.

It is also not TCP slow-start after idle (`tcp_slow_start_after_idle=1` on both,
but that fires on an RTO ~200 ms, not 500 us), and not receive-side polling
(`--busy-poll 500` bought 10%).

Confirmed with the daemon removed entirely — bare ICMP:

    ping -i 0.005 (warm)   avg 0.492 ms
    ping -i 1     (idle)   avg 0.788 ms

## Cause 1: Thunderbolt runtime power management — FIXED, worth ~2% today

`/sys/class/net/thunderbolt0/device/power/control` was `auto` on BOTH boxes.
Setting it to `on` (runtime suspend disabled):

    gap us    rtt auto -> on          link auto -> on
       500      1255 ->  592  -53%     843 ->  200  -76%
      1000      1552 ->  980  -37%    1155 ->  538  -53%
      2000      1370 ->  981  -28%     966 ->  568  -41%
      5000       994 ->  978   -2%     594 ->  574   -3%

Large below 2 ms of idle, **~nothing at 5 ms**, which is where decode actually
sits. So it is worth ~2% now — but it becomes worth much more the moment the
hub's per-layer work shrinks, which is what every other decode lever does.
Non-persistent; re-apply after reboot.

## Cause 2: CPU C-state exit latency — 350 us, UNFIXED

Box 2's idle states:

    POLL  lat 0      C1  lat 1      C2  lat 18      C3  lat 350

governor `menu`. A 5 ms idle reliably reaches C3, whose **350 us exit latency**
accounts for most of the 574 us that survives the runtime-PM fix. `/dev/cpu_dma_latency`
(the PM QoS interface a daemon would normally hold open at 0) is `crw------- root
root`, so capping it needs either a udev rule or disabling C3:

    sudo sh -c 'for c in /sys/devices/system/cpu/cpu*/cpuidle/state3/disable; do echo 1 > $c; done'

on BOTH boxes — box 1's CPU is blocked waiting on box 2 and idles too. Expected
recovery ~350 us/layer = ~14 ms/token, ~5.7% of a 246 ms token. Costs idle power:
the CPU parks in C2.

## Why this is worth chasing at all

575 us/layer x 40 layers = **~23 ms/token, ~9% of decode**, spent on nothing but
waking a link and a CPU up. It is also pure latency — it does not shrink when the
expert cache improves, so it becomes a *larger* share of every future decode win.

And note the warm baseline: even with the link hot, a bare ICMP round trip is
0.492 ms and the bench's warm rtt is 434 us for a 5.9 KB request. That is high
for a direct USB4 link and sits on the critical path 40 times per token — its own
lever, independent of idle.

## RESULT: both applied — decode 4.07 -> 4.94 tok/s (+21%)

`thunderbolt0/device/power/control=on` on both boxes AND C3 disabled on every CPU
of both boxes:

    gap 5000 (the real duty cycle)      rtt        link
      baseline  auto + C3                994        594
      + TB runtime PM on                 978        574
      + C3 disabled                      643        231     -35% rtt, -61% link

    raw ICMP idle:  0.788 ms -> 0.203 ms  (3.9x; now better than the old WARM 0.492)

End to end, 512-token generations, same prompt, n=3 each:

    baseline   4.06, 4.07, 4.09 tok/s   (mean 4.07)
    both on    4.40, 5.44, 4.99         (mean 4.94)     +21%

**The e2e gain is ~44 ms/token, 3x what the link measurement alone predicted
(14 ms).** The excess is almost certainly `sel_sync`: the host blocks in a HIP
stream sync once per layer, 40x per token, 22-23 ms/token total. A CPU that drops
into C3 while blocked pays the same 350 us exit latency on every wake. The link
and the host sync are two consumers of one fix.

Caveats, recorded honestly:
  * The baseline was measured earlier in the session on a slightly older server
    build (the verify probe was added since, but is off by default). Not a
    same-binary A/B. The effect is 3x the +/-8% e2e noise floor and has an
    independent bench measurement behind it, but it is not a paired A/B --
    C3 cannot be re-enabled without root, so one could not be run.
  * Variance rose: 4.40-5.44 (24%) against the baseline's 1%.

## Making it persistent

Both are non-persistent and reset on reboot. On both boxes:

    echo on | sudo tee /sys/class/net/thunderbolt0/device/power/control
    sudo sh -c 'for c in /sys/devices/system/cpu/cpu*/cpuidle/state3/disable; do echo 1 > $c; done'

The tidier form for C-states is a PM QoS request — a process holding
`/dev/cpu_dma_latency` open with a 0 written to it — which bounds exit latency
without disabling a state globally. It needs a udev rule here (`crw------- root
root`), and would ideally be held by `deepstrix-expertd` and the server for their
lifetime rather than set machine-wide.

## What remains

Link at gap 5000 is still 231 us against 44 us warm, so ~187 us/layer = ~7.5
ms/token of idle penalty survives both fixes. C2's exit latency is only 18 us, so
the residual is elsewhere — probably remaining Thunderbolt/PCIe link states.

And the warm baseline is untouched: 44 us of link plus ~380 us of `srv` for a
5.9 KB round trip, 40 times per token. The idle penalty is now the smaller half.

## 2026-09-19: the link is Gen2 x2 (20 Gb/s) because of the CABLE; CLx is ruled out

`tbdump -r 0 -a 2 -vv` on box 1 (route 0 = own host router, adapter 2 = the port
facing box 2; box 2 is the XDomain at route 2):

    LANE_ADP_CS_0  Supported Link Speeds  0xc   Gen2 + Gen3 (port can do 40 Gb/s)
    LANE_ADP_CS_1  Target Link Speed      0xc   "attempt Gen 3"
    LANE_ADP_CS_1  Current Link Speed     0x8   Gen 2, width x2  (= sysfs 10.0 Gb/s x2)
    PORT_CS_18     Cable Gen 3 Support    0     the cable's e-marker declares no Gen3

The port wants Gen3 and the cable refuses it. Retimers are Parade PS8830 (box 1
board) and PS8833 (box 2 board), seen in MIRRORED order from the two ends, i.e.
one per board and none in the cable: it is a passive cable. Fix: a 40 Gb/s
cable (passive <= 0.8 m marked "40", or active). Verify afterwards: CG3 = 1,
Current Link Speed = 0x4, sysfs `tx_speed` = 20.0 Gb/s. This moves prefill's
bandwidth term (785 MB/s), not decode latency (26 KB/call).

Also from the dump: LANE_ADP_CS_0 CL0s/CL1/CL2 Support = 0 and PORT_CS_18 Cable
CLx Support = 0, nothing enabled. **USB4 low-power link states are NOT the ~187
us/layer residual** named in "What remains" above; `thunderbolt.clx=Y` is moot.
The residual is on the CPU/PCIe side. 99 logical-layer errors in 6 days: the
link is not marginal.

Box 2 has no tailscale; reach it with `ssh -J mimir@lumi-brain mimir@10.99.0.2`.

## 2026-09-19 evening: the residual is PER-FRAME wakeups on a multi-frame response

Host state: governor performance, C2 off, POLL-only idle on the link CCX of each
box, expertd pinned to box 2's NHI CCX, TB runtime PM on. Production plain decode
still paid ~211 us/call of link. A 3-expert test daemon on box 2 (port 7432) and
`deepstrix-expert-bench --batches 1 --picks 3 --pool 3` from box 1, concurrent
with production:

    resp   gap us  busy-poll   rtt   srv   link p50   one-way p50
    f16     0       500        376   339     37          -
    f16  1500       500        424   360     62          29 us
    f16  1500         0        428   359     64          -
    f16  3000       500        442   359     71          -
    f32  1500       500        551   356    193          95 us   <- production's shape
    f32     0       500        384   337     46          22 us
    f32  1500  500 +quickack   542   351    193          -
    B=4 f16 1500    500        797   452    335         166 us
    B=4 f32 1500    500        819   452    362         179 us

* Scheduler queueing is NOT it: hub rexp-reader runq_wait 0.19 ms over 3,430
  wakeups in 8 s; box 2 daemon 0.3 ms. C-states/frequency are not it (all set).
* The idle penalty is SIZE-dependent: a 20.6 KB f32 response (5-6 thunderbolt-net
  4 KB frames) after a 1.5 ms gap costs 193 us; the same bytes with the reader
  still busy-polling (gap 0) cost 46. A 10 KB f16 response after the same gap
  costs 62. The penalty scales with FRAME COUNT (B=4: 335-362), i.e. each frame of
  a packet that lands on a sleeping reader pays an interrupt->NAPI->wake cycle,
  while a spinning reader drains all frames in one poll.
* Production's reader spins its 500 us SO_BUSY_POLL window right after the
  PREVIOUS frame, ~1.5 ms before the next response, so every production response
  lands on a sleeping reader. That is the ~150 us/call = ~6 ms/token.

Fixes, in order of cost: (1) reader spins through the RTT — SO_BUSY_POLL >= the
per-layer period (~2.5 ms) on the hub, plus `--busy-poll` on expertd for the
request direction; needs `net.core.busy_read` raised (script step 4s) and both
processes restarted; bench prediction 193 -> ~46 us/call. (2) f16 responses
(10 KB, 3 frames): 62 us/call, but changes decode numerics. (3) kernel: call
`tb_ring_throttling(rx_ring, ~20 us)` in thunderbolt-net so a burst of frames
raises one interrupt; the API exists, the net driver never uses it.

### The fix, measured: spin through the round trip — but ONLY for single-segment responses

`net.core.busy_read` raised to 5000 on both boxes (`link_latency_step.sh 4s on`), then
the client window swept (f32 responses, gap 1500 us unless noted):

    shape            resp     daemon bp   client bp    link p50
    B=1 (decode)     20.6 KB     500         500        179-193   <- production today
    B=1              20.6 KB     500        3000         54       <- the fix: -130 us/call
    B=1              20.6 KB    5000     2000..5000     50-54       (gap-independent: 3000 us gap = 53)
    B=4              82 KB       500     500 / 2000    367 / 358
    B=4              82 KB      5000     500/2000/3000/4000/5000   230/153/1139/2102/3172
    B=64 (prefill)   1.3 MB      500     500 / 2000   1945 / 3490
    B=64             1.3 MB     5000     500/2000/3000   6628/8123/9209

Two hard rules fall out:
  1. A response LARGER THAN THE 65,520 B MTU (>= 2 TCP segments) is HELD by the
     busy-poll loop until the window expires: B=4 link ~= window - 1900 us. And a
     spinning reader on the DAEMON side wrecks its own big sends (1.3 MB: 1.9 -> 6.6 ms).
     So the daemon stays at `--busy-poll 500`, and the hub's window must be small
     (<= 500) whenever responses exceed one segment: prefill, verify, any B >= 4.
  2. Decode's 20.6 KB response is one segment and gains the full 130 us/call at any
     window >= 2000; use ~3000 (the per-layer period is ~2.2 ms and the reader's spin
     starts at the previous handoff, so it must cover a whole period).
  => hub: SO_BUSY_POLL = 3000 while decoding, 500 otherwise (setsockopt per phase;
     needs `net.core.busy_read` >= 3000 since the hub is unprivileged). Predicted
     -5..-5.5 ms/token (~6-7%). `--busy-poll 0` on the bench/daemon does NOT mean
     "off": the socket then inherits the sysctl default.

### SHIPPED 2026-09-19 19:27Z (build with `HetEngine::remote_set_phase_busy_poll`)

First 3,700 production tokens after the restart, cold pool (box-1 misses p50 1):

    warm tokens (0 box-1 misses)   before (2c+3)   after
      exposed remote wait            30.3 ms       21.1    <- now BELOW box 2's srv
      box-2 srv                      22.0          21.6
      sel_sync                       22.9          22.2
      total                          68.8          58.5    (-15%; ~17 tok/s warm)
    all tokens, total p50            78.7          66.3    (-16%)
    per-request true link (dspark.request link=b1:)  254-266 -> 91 us/call

The response now lands while box 1 is still doing its post-submit work (shared
expert, ensure, local MoE), so `wait` finds it already there and the remote leg
is HIDDEN. Consequence for reading `het.token.summary`: `remote_rtt_us` is the
EXPOSED wait (`now - t_wait`), and `remote_link_us = exposed - srv` (saturating),
so both now read below srv / ~0. That is the accounting, not a regression; the
per-request `link=b1:` figure is the true per-call link. The pole on a warm token
is box 1 itself: sel_sync ~22 ms + its own misses + ~10 ms glue.

## 2026-09-21: multi-row replies — the hold is NOT the client's busy-poll window

Measured against a 3-expert-per-layer test daemon on box 2 (`--experts L0-L39:0-2
--listen :7432`; the production daemon serves one connection, so a bench against it
queues forever), `--picks 3 --pool 3 --batched --gap-us 500`, link = rtt - srv, us p50:

    rows   reply KB (f32)   busy 3000   busy 500   busy 100   busy 20   busy 20 + quickack
      1        20.6            62          -          -         -           -
      2        41.0           213        336        226       217         228
      4        82.0          1992       1204       1075      1272         759
      8       163.9          1897       1087       1262      1258         651
     16       327.8          1712       1262       1138      1200          -
     32       655.4          1122       1064        970       936          -
     64      1310.8          2209       2117       2097      2081          -
    f16 replies (10.3 KB/row): 4 rows = one segment, link 225; 8 rows = two, 1941.

Any reply over ONE segment (65,520-B MTU) costs ~1.1-1.3 ms of link whatever the
window, ~2 ms at the decode window, ~0.65 ms with TCP_QUICKACK re-armed per frame.
Ruled out on the daemon box during a 120-request run: `TcpExtTCPAutoCorking` did
not move (99,350 -> 99,350), `TcpExtDelayedACKs` +2. So it is neither autocorking
nor a receiver-side delayed ACK alone; the ~1 ms scale and the window-independence
point at the SENDER's segmentation path on this 64 KB-MTU device (TSO/GSO deferral
of the partial last skb, qdisc fq_codel quantum 65546). Root-level test, not yet run:
`ethtool -K thunderbolt0 tso off gso off` on box 2 (and/or `net.ipv4.tcp_min_tso_segs`,
`tcp_tso_win_divisor`), then the same bench. Until that is settled, every multi-row
decode design (MULTISTREAM_DECODE_PLAN.md M1a) has to assume ~0.65-2 ms per call
above one segment, i.e. f32 at >= 4 rows and f16 at >= 8.
