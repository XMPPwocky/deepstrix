#!/usr/bin/env bash
# usage: variants.sh name:"flags" ... -- compiles kernels/attn_dec_g3o.hip per variant to
# scratch/g3o/v_<name>.hsaco, prints resource usage per kernel, and the G3O_VARIANTS list.
W=/home/claude-code/deepstrix/.claude/worktrees/wf_be8e2331-368-4
cd "$W" || exit 1
src=crates/v4flash-kernels/kernels/attn_dec_g3o.hip
list=""
for spec in "$@"; do
  name=${spec%%:*}
  flags=${spec#*:}
  out=scratch/g3o/v_$name.hsaco
  # shellcheck disable=SC2086
  r=$(scratch/g3o/kc.sh "$src" "$out" $flags | grep -E "Function Name|VGPRs:|Spill|LDS|error" | grep -v "SGPRs Spill" \
      | sed 's/ \[-Rpass.*//; s/Function Name: //; s/VGPRs: /v=/; s/VGPRs Spill: /spill=/; s/LDS Size \[bytes\/block\]: /lds=/' | tr '\n' ' ')
  echo "$name ($flags): $r"
  list="$list,$name"
done
echo "G3O_VARIANTS=${list#,}"
