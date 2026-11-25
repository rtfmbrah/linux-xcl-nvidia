#!/usr/bin/env bash
# Runs the unit tests of XCL modules that do not depend on Windows (today: gpu/spirv) as a
# native Linux test binary, in seconds instead of a Windows cross build + Wine run.
#   scripts/linux/test-native.sh [test filter]
# With spirv-val on PATH (or XCL_SPIRV_VAL=/path/to/spirv-val) emitted modules are validated.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${TMPDIR:-/tmp}/xcl-native-tests
mkdir -p "$OUT"
printf '#[path = "%s/crates/xcl/src/gpu/spirv/mod.rs"]\n#[allow(dead_code)]\nmod spirv;\n' "$ROOT" > "$OUT/lib.rs"
rustc --edition 2021 --test -O "$OUT/lib.rs" -o "$OUT/xcl-native-tests"
if [ -z "${XCL_SPIRV_VAL:-}" ] && command -v spirv-val >/dev/null; then export XCL_SPIRV_VAL=$(command -v spirv-val); fi
"$OUT/xcl-native-tests" "$@"
