#!/usr/bin/env bash
# Tests under ASan+UBSan (and TSan for the threaded renderer). Do NOT wrap in stdbuf: it uses LD_PRELOAD and
# breaks the ASan runtime ("ASan runtime does not come first").
export PATH=/usr/local/cuda-12.8/bin:/usr/bin:/bin:/usr/local/bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rc=0
for san in "address,undefined" thread; do
  bash "$HERE/scripts/build.sh" "$san" 2>&1 | grep -E "error|FAILED"
  name="${san//,/_}"
  echo "=== $san"
  # TSan needs ASLR off under WSL2 kernels ("unexpected memory mapping")
  timeout 1800 setarch "$(uname -m)" -R "$HOME/pt-build/$name/pt_tests" 2>&1 | tail -26
  [ "${PIPESTATUS[0]}" -ne 0 ] && rc=1
done
exit $rc
