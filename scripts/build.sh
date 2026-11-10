#!/usr/bin/env bash
# Build in WSL. CUDA targets are built when nvcc is found (they compile for sm_120 but are NOT run by this script).
#   bash scripts/build.sh                 -> ~/pt-build/release
#   bash scripts/build.sh address,undefined   or   thread   -> sanitizer build of the tests
set -euo pipefail
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:/usr/local/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SAN="${1:-}"
NAME="${SAN:-release}"; NAME="${NAME//,/_}"
DIR="$HOME/pt-build/$NAME"
BT=Release; [ -n "$SAN" ] && BT=Debug
cmake -S "$HERE" -B "$DIR" -G Ninja -DCMAKE_BUILD_TYPE=$BT -DSANITIZE="$SAN" \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.8/bin/nvcc 2>&1 | grep -E "CUDA|Error|error" || true
cmake --build "$DIR" 2>&1 | grep -E "error|warning: |FAILED|spill|Used [0-9]+ registers" | head -40 || true
ls "$DIR"/pt_tests "$DIR"/pt_render "$DIR"/pt_bench
