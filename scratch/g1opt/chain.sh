#!/usr/bin/env bash
# chain.sh: lane-layer chain wall (bench_dgpu_chain_map BENCH_SECTION=chain, warm, beside the hub):
# production chain vs G1 with the new default kernels, and vs the round-1 G1 kernels (startup override
# V41_DGPU_G1_VARIANT, one process per b). dGPU only, production kernarg env (run.sh).
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-2
R=${ROUNDS:-120}
C="reuse#AA,reuse_g1,route:reuse,route:reuse_g1"
"$W/scratch/g1opt/run.sh" bench_dgpu_chain_map /tmp/g1o/chain_new.log BENCH_SECTION=chain BENCH_ROUNDS=$R BENCH_B=2,4,8 BENCH_CLASSES=$C -- bench_dgpu_chain_map > /dev/null
for pair in "2:2,64,32" "4:4,8,0" "8:8,40,20"; do
  b=${pair%%:*}; v=${pair#*:}
  "$W/scratch/g1opt/run.sh" bench_dgpu_chain_map /tmp/g1o/chain_old_b$b.log BENCH_SECTION=chain BENCH_ROUNDS=$R BENCH_B=$b BENCH_CLASSES=$C V41_DGPU_G1_VARIANT=$v -- bench_dgpu_chain_map > /dev/null
done
echo done
