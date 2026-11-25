#!/usr/bin/env bash
# Batch inference and Haar-state verification on a test shard directory.
# Usage: bash scripts/run_infer.sh <CHECKPOINT> <TEST_DIR> [OUT_DIR] [K]
set -euo pipefail

CKPT="${1:?Usage: bash scripts/run_infer.sh <CHECKPOINT> <TEST_DIR> [OUT_DIR] [K]}"
TEST_DIR="${2:?Usage: bash scripts/run_infer.sh <CHECKPOINT> <TEST_DIR> [OUT_DIR] [K]}"
OUT_DIR="${3:-outputs/infer}"
K="${4:-8}"

export HYDRA_FULL_ERROR=1

python infer.py \
  checkpoint="\"${CKPT}\"" \
  data_path="${TEST_DIR}" \
  output_dir="${OUT_DIR}" \
  num_sample_per_seq="${K}" \
  temperature=0.2 \
  validation_size=9000 \
  inference=true \
  verify_circuit=true \
  certify_zx=false
