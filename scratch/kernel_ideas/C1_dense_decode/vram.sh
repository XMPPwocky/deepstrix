#!/usr/bin/env bash
# Print dGPU VRAM total/used (MB) and busy%.
C=/sys/class/drm/card1/device
tot=$(cat $C/mem_info_vram_total)
used=$(cat $C/mem_info_vram_used)
echo "vram_total_MB=$((tot/1048576)) vram_used_MB=$((used/1048576)) free_MB=$(((tot-used)/1048576)) busy=$(cat $C/gpu_busy_percent)"
