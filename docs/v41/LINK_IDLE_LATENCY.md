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
