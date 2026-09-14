# The decode miss path, and why 30 tok/s needs a smaller expert format
### measured 2026-09-14, box 2 = Crucial T500 / 128 GB Strix Halo

Two results. First, box 2's miss path was 1.6x more expensive than its own drive
and is now fixed (`c758a07`). Second — and this is the one that matters — even a
*perfect* miss path cannot reach 30 tok/s, because of how many bytes a token
reads. The binding constraint is the expert format, and it points DOWN from
MXFP4, not up.

## 1. The miss path, fixed

Measured with `deepstrix-expert-bench --catchall` against the live daemon (no
box-1 weight load), arms back-to-back on ONE binary via `V41_B2_GPU_REPACK`:

    ms_per_miss  read  [pread repack_cpu]  h2d  repack_gpu
        10.57    9.46   --        --       1.11     --      before
         7.51    6.28  6.08      0.00      1.23    0.04     + GPU repack
         6.60    6.17  5.97      0.00      0.43    0.41     + zero-copy staging   <- shipped
         6.99    6.49  6.23      0.00      0.50    0.48     + O_DIRECT (rejected)

* The HF->ggml MXFP4 permute ran on box 2's **CPU** every fault. On the iGPU it
  costs 0.04 ms. Box 1's pager moved this at M7; box 2 never followed.
* The H2D was copying system RAM to system RAM — box 2 is an APU, the "device"
  pool IS host memory. Staging is now `hipHostMalloc` NON_COHERENT, pread into
  directly and consumed in place by the kernel. The upload is gone.
* **O_DIRECT rejected a second time** (box 1 -16% decode, box 2 +6% here). Its
  bounce-buffer copy costs more than the page cache it skips.

### The drive is not the problem and never was

O_DIRECT random reads at the expert size (20.2 MB) on box 2's own NVMe:

    QD      1     2     3     4     6     8    12    16
    GB/s 4.47  4.59  4.52  4.49  4.03  4.09  3.99  3.76
    ms   4.52  4.40  4.46  4.50  5.01  4.93  5.06  5.36

**Flat.** One expert-sized read already saturates the drive, so queueing misses
buys no bandwidth — which also means batching them under speculation buys none.
4.47 GB/s is the number to plan against.

## 2. End to end

Box 1 server + box 2, 512-token generation, `temperature=0`, counters bracketed
exactly around the run:

    512 tokens in 126.1 s          = 4.06 tok/s = 246 ms/token
    picks/token                    = 260.5
    misses/token                   = 20.20   (7.8% of picks)
    ms_per_miss                    = 7.81
    => miss cost                   = 158 ms/token  = 64% of the token
    => everything else             =  88 ms/token

Back-computing the control at the old 10.57 ms/miss: 302 ms/token = 3.31 tok/s.
So the change is worth **+23% decode**. Output validated: "The capital of France
is Paris. The river Seine runs through it." — the GPU permute is byte-correct, a
wrong one would garble immediately.

The 88 ms non-miss residual reconciles with the separately measured 71 ms
zero-miss floor (that floor was taken warm and favourable).

## 3. Why this cannot reach 30 tok/s — the real wall

A token touches **20.2 missing experts x 18.8 MB = 380 MB off disk**.

    380 MB at 4.47 GB/s (the drive's measured ceiling)  =  85 ms/token
    budget for 30 tok/s                                 =  33 ms/token

So **even at infinite CPU, zero repack, zero copy, and a perfectly scheduled
drive, the miss path alone is 2.5x over the entire 30 tok/s budget.** No further
work on this path can close it. Speculation does not rescue it either: at B=5 the
distinct-expert count grows 3.2x while tokens grow 4.13x (see
`DSPARK_VERIFY_BATCH_MODEL.md`), so misses/token fall only 23%, to ~15.6 — still
66 ms/token of pure disk, still 2x the budget.

The only way out is to stop going to disk, i.e. make the expert set **fit**.

## 4. The format arithmetic — and a correction

An expert is 3 x 5120 x 2304 = 35.4M weights; the model has 15,360 of them. The
actual expert budget today is **175 GB** (box 1's 52 GB pool + box 2's 123.3 GB).

     format  bits/wt  MB/expert  full set GB  fits 175 GB?
      MXFP4     4.25      18.80        288.8      no   <- today, 61% resident
    IQ3_XXS     3.06      13.54        207.9      no
      IQ2_S     2.50      11.06        169.9      YES
    IQ2_XXS     2.06       9.11        140.0      YES

**IQ2_S holds the entire expert set in the budget we already have, with 5 GB to
spare.** Every miss disappears; decode becomes the 88 ms floor (11.4 tok/s), and
DSpark at B=5 on top of that projects into the 40s.

### The correction

`DECODE_CAPACITY_WALL.md` concluded "the path to 30 tok/s decode is a **5-bit**
expert format, not a better LRU". That was wrong in both directions: it assumed
Q8_K (8.5 bits) when the checkpoint is already MXFP4 at 4.25, so Q5_K would have
made the experts *larger*. The user caught the premise ("aren't experts fp4
lol"); the conclusion drawn at the time was that requantisation was dead. It is
not dead — it is the answer, pointing the other way. The target is **~2.5
bits/weight**, a 1.7x shrink from MXFP4, and the repo already has iq2_s and
iq3_xxs kernels from the V4-Flash era to port.

## 5. What is and is not a lever now

* **Expert format at ~2.5 bits — the only thing that reaches the goal.** Ranked
  first by a wide margin; everything else is inside the 88 ms.
* Remaining miss-path residue: read is 6.17-7.3 ms against the drive's 4.52. The
  gap is the page-cache copy. Worth ~2 ms x 20 misses = 40 ms/token TODAY, but
  worth zero once the set is resident. Do not start here.
* Cache policy: measured dead (LFU 11%, SLRU 14% of the OPT gap).
* Queueing/prefetching misses: measured dead — the drive does not scale with
  queue depth.

Reproduce: `scratchpad/qd.py` (drive scaling), `deepstrix-expert-bench
--catchall` (miss cost, no box-1 load), and the bracketed 512-token run above.
