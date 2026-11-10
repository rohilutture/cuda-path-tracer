#!/usr/bin/env bash
# CPU benchmarks + a reference render. The GPU benchmark is part of pt_bench but is skipped with --no-gpu
# (used while the GPU is busy). Run on a quiet machine, never alongside the sanitizer build.
#   bash scripts/run_bench.sh [--with-gpu]
export PATH=/usr/bin:/bin:/usr/local/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$HOME/pt-build/release"
mkdir -p "$HERE/results"
GPU="--no-gpu"; [ "${1:-}" = "--with-gpu" ] && GPU=""
{
  echo "# environment"; uname -sr; lscpu | grep -E "Model name|^CPU\(s\)|Thread"; gcc --version | head -1
  echo; echo "# pt_bench"
  "$BIN/pt_bench" --max-subdiv 7 --rays 4000000 $GPU --csv "$HERE/results/bench.csv"
} 2>&1 | tee "$HERE/results/bench.txt"
echo; echo "# reference render (CPU)"
"$BIN/pt_render" --scene cornell --w 640 --h 640 --spp 512 --depth 8 --device cpu --out "$HERE/results/cornell_cpu.png" | tee "$HERE/results/render.txt"
