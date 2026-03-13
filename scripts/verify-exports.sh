#!/bin/bash
set -euo pipefail

# Verify that the built WASM module exports all expected symbols.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/rf.js"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_rf_get_last_error
  wl_rf_fit
  wl_rf_predict
  wl_rf_predict_proba
  wl_rf_save
  wl_rf_load
  wl_rf_free
  wl_rf_free_buffer
  wl_rf_get_n_trees
  wl_rf_get_n_features
  wl_rf_get_n_classes
  wl_rf_get_task
  wl_rf_get_oob_score
  wl_rf_get_feature_importance
  wl_rf_get_criterion
  wl_rf_get_sample_rate
  wl_rf_get_heterogeneous
  wl_rf_get_oob_weighting
  wl_rf_get_alpha_trim
  wl_rf_get_leaf_model
  wl_rf_predict_quantile
  wl_rf_predict_interval
  wl_rf_permutation_importance
  wl_rf_proximity
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."
