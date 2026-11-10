#!/usr/bin/env bash
# Regression check: the CUDA renderer must reproduce the CPU renderer (same algorithm, same per-pixel random streams).
# History: an unspecified-evaluation-order bug (two rng.next_f() calls as function arguments) made gcc and nvcc draw
# the numbers in different orders, so the images differed by pure Monte Carlo noise (RMSE 0.15 at 128 spp) from the
# first diffuse bounce on. Correct output is RMSE ~1e-4 or below.
set -euo pipefail
R=${PT_BUILD:-$HOME/pt-build/release}/pt_render
fail=0
for cfg in "cornell 64 8" "cornell 4 3" "noisy 16 8"; do
  set -- $cfg
  line=$($R --scene "$1" --spp "$2" --depth "$3" --subdiv 4 --device both --out /tmp/gpu_check.png | grep RMSE)
  rmse=$(echo "$line" | sed -E 's/.*RMSE ([0-9.]+).*/\1/')
  echo "$1 spp=$2 depth=$3: RMSE $rmse"
  awk -v r="$rmse" 'BEGIN{exit !(r < 0.005)}' || { echo "FAIL: GPU image differs from CPU"; fail=1; }
done
exit $fail
