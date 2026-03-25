#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_SCRIPT="${ENV_SCRIPT:-$ROOT_DIR/.local/gdal-kakadu/env.sh}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$ROOT_DIR/datasets/manifests/spacenet-pan-adapt-smoke}"
PAIRS_ROOT="${PAIRS_ROOT:-$ROOT_DIR/datasets/pairs/spacenet-pan-adapt-smoke}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/runs/pan-adapt-smoke}"
MAX_CHIPS_PER_ASSET="${MAX_CHIPS_PER_ASSET:-8}"
TRAIN_LIMIT="${TRAIN_LIMIT:-512}"
EVAL_LIMIT="${EVAL_LIMIT:-512}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-16}"
TEACHER_MODEL="${TEACHER_MODEL:-dinov3_vitb16}"
STUDENT_ARCH="${STUDENT_ARCH:-residual_cnn}"
STUDENT_BASE_CHANNELS="${STUDENT_BASE_CHANNELS:-64}"
STRUCTURE_WEIGHT="${STRUCTURE_WEIGHT:-0.5}"
VIEW_CONSISTENCY_WEIGHT="${VIEW_CONSISTENCY_WEIGHT:-0.25}"
POSITIVE_PAIR_WEIGHT="${POSITIVE_PAIR_WEIGHT:-0.5}"
HARD_NEGATIVE_WEIGHT="${HARD_NEGATIVE_WEIGHT:-0.25}"
POSITIVE_EXACT_WEIGHT="${POSITIVE_EXACT_WEIGHT:-2.0}"
POSITIVE_WEAK_WEIGHT="${POSITIVE_WEAK_WEIGHT:-1.0}"
HARD_NEGATIVE_MAX_SIMILARITY="${HARD_NEGATIVE_MAX_SIMILARITY:-0.2}"
HARD_NEGATIVE_GAP_SCALE="${HARD_NEGATIVE_GAP_SCALE:-0.5}"
HARD_NEGATIVE_MIN_SIMILARITY="${HARD_NEGATIVE_MIN_SIMILARITY:--0.25}"
ADVERSARIAL_NEGATIVE_TOP_FRACTION="${ADVERSARIAL_NEGATIVE_TOP_FRACTION:-0.25}"
ADVERSARIAL_NEGATIVE_MAX_PAIRS="${ADVERSARIAL_NEGATIVE_MAX_PAIRS:-512}"
ADVERSARIAL_NEGATIVE_MIN_TEACHER_SIMILARITY="${ADVERSARIAL_NEGATIVE_MIN_TEACHER_SIMILARITY:-0.0}"
AUGMENTATION_MIN_CROP_SCALE="${AUGMENTATION_MIN_CROP_SCALE:-0.7}"
AUGMENTATION_NOISE_STD="${AUGMENTATION_NOISE_STD:-0.02}"
AUGMENTATION_GAMMA_JITTER="${AUGMENTATION_GAMMA_JITTER:-0.15}"
AUGMENTATION_BLUR_PROBABILITY="${AUGMENTATION_BLUR_PROBABILITY:-0.2}"

if [[ ! -f "$ENV_SCRIPT" ]]; then
  echo "missing GDAL environment script: $ENV_SCRIPT" >&2
  echo "build the runtime first with ./scripts/build_gdal_kakadu.sh" >&2
  exit 1
fi

set +u
source "$ENV_SCRIPT"
set -u

uv run geogrok-make-manifests \
  --output-root "$MANIFEST_ROOT"

uv run geogrok-make-pairs \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --scenes-path "$MANIFEST_ROOT/scenes.parquet" \
  --output-root "$PAIRS_ROOT" \
  --modality PAN \
  --city Jacksonville \
  --city Omaha \
  --city UCSD \
  --max-chips-per-asset "$MAX_CHIPS_PER_ASSET" \
  --positive-overlap-fraction 0.5 \
  --weak-overlap-fraction 0.2 \
  --hard-negative-radius-m 800 \
  --max-positives-per-query 2 \
  --max-hard-negatives-per-query 2

uv run geogrok-benchmark-pan-adapt \
  --chips-path "$MANIFEST_ROOT/chips.parquet" \
  --pairs-path "$PAIRS_ROOT/pairs.parquet" \
  --run-root "$RUN_ROOT" \
  --teacher-model "$TEACHER_MODEL" \
  --train-split train \
  --query-split val \
  --query-split test \
  --gallery-split val \
  --gallery-split test \
  --modality PAN \
  --train-limit "$TRAIN_LIMIT" \
  --eval-limit "$EVAL_LIMIT" \
  --teacher-batch-size "$TEACHER_BATCH_SIZE" \
  --student-image-size 128 \
  --student-arch "$STUDENT_ARCH" \
  --student-base-channels "$STUDENT_BASE_CHANNELS" \
  --epochs 24 \
  --steps-per-epoch 48 \
  --pairs-per-batch 32 \
  --eval-batch-size 64 \
  --learning-rate 0.001 \
  --temperature 0.1 \
  --weight-decay 1e-4 \
  --contrastive-weight 1.0 \
  --alignment-weight 1.0 \
  --structure-weight "$STRUCTURE_WEIGHT" \
  --view-consistency-weight "$VIEW_CONSISTENCY_WEIGHT" \
  --positive-pair-weight "$POSITIVE_PAIR_WEIGHT" \
  --hard-negative-weight "$HARD_NEGATIVE_WEIGHT" \
  --positive-exact-weight "$POSITIVE_EXACT_WEIGHT" \
  --positive-weak-weight "$POSITIVE_WEAK_WEIGHT" \
  --hard-negative-max-similarity "$HARD_NEGATIVE_MAX_SIMILARITY" \
  --hard-negative-gap-scale "$HARD_NEGATIVE_GAP_SCALE" \
  --hard-negative-min-similarity "$HARD_NEGATIVE_MIN_SIMILARITY" \
  --adversarial-negative-top-fraction "$ADVERSARIAL_NEGATIVE_TOP_FRACTION" \
  --adversarial-negative-max-pairs "$ADVERSARIAL_NEGATIVE_MAX_PAIRS" \
  --adversarial-negative-min-teacher-similarity "$ADVERSARIAL_NEGATIVE_MIN_TEACHER_SIMILARITY" \
  --augmentation-min-crop-scale "$AUGMENTATION_MIN_CROP_SCALE" \
  --augmentation-noise-std "$AUGMENTATION_NOISE_STD" \
  --augmentation-gamma-jitter "$AUGMENTATION_GAMMA_JITTER" \
  --augmentation-blur-probability "$AUGMENTATION_BLUR_PROBABILITY" \
  --device auto \
  --amp

uv run python - <<'PY' "$RUN_ROOT"
from __future__ import annotations

import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1]).resolve()
summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
print(summary)
PY

echo "PAN adaptation benchmark smoke test passed."
