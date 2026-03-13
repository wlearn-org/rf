#!/bin/bash
set -euo pipefail

# Build RF C11 core as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_rf_get_last_error",
  "_wl_rf_fit",
  "_wl_rf_predict",
  "_wl_rf_predict_proba",
  "_wl_rf_save",
  "_wl_rf_load",
  "_wl_rf_free",
  "_wl_rf_free_buffer",
  "_wl_rf_get_n_trees",
  "_wl_rf_get_n_features",
  "_wl_rf_get_n_classes",
  "_wl_rf_get_task",
  "_wl_rf_get_oob_score",
  "_wl_rf_get_feature_importance",
  "_wl_rf_get_criterion",
  "_wl_rf_get_sample_rate",
  "_wl_rf_get_heterogeneous",
  "_wl_rf_get_oob_weighting",
  "_wl_rf_get_alpha_trim",
  "_wl_rf_get_leaf_model",
  "_wl_rf_predict_quantile",
  "_wl_rf_predict_interval",
  "_wl_rf_permutation_importance",
  "_wl_rf_proximity",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/rf.c" \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  -I "${PROJECT_DIR}/csrc" \
  -o "${OUTPUT_DIR}/rf.js" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createRF \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: none (C11 from scratch)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/rf.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
