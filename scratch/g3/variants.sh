#!/usr/bin/env bash
# usage: variants.sh <src.hip> name:"flags" ... -- compiles each variant to scratch/g3/v_<name>.hsaco,
# prints resource usage, and prints the G3_VARIANTS list.
cd /home/claude-code/deepstrix/.claude/worktrees/wf_c1d97dd2-693-6 || exit 1
src=$1; shift
list=""
for spec in "$@"; do
  name=${spec%%:*}
  flags=${spec#*:}
  out=scratch/g3/v_$name.hsaco
  # shellcheck disable=SC2086
  r=$(scratch/g3/kc.sh "$src" "$out" $flags | grep -E "VGPRs|LDS|error" | sed 's/ \[-Rpass.*//' | tr '\n' ' ')
  echo "$name ($flags): $r"
  list="$list,$PWD/$out"
done
echo "G3_VARIANTS=${list#,}"
