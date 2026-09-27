#!/usr/bin/env bash
# collect.sh DIR : wait for every ticket in DIR/tickets.txt not yet collected; write
# DIR/results/<section>_<tag>.txt; mark collected tickets in DIR/tickets.done
set -u
cd "$1"
touch tickets.done
while read -r t tag sec; do
  [ -z "$t" ] && continue
  grep -q "^$t\$" tickets.done && continue
  bash "$(dirname "$0")/../_infra/gpu_wait.sh" "$t" > "results/${sec}_${tag}.txt" 2>&1
  echo "$t" >> tickets.done
  echo "collected $sec $tag rc=$?"
done < tickets.txt
