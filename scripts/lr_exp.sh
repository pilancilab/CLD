#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script lives in ROOT/scripts/lr_exp.sh)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# WANDB setup
export WANDB_ENTITY="${WANDB_ENTITY:-lucigen}"
export WANDB_PROJECT="${WANDB_PROJECT:-CLD}"
# Add the lr-exp tag; we also append the config name later per run
BASE_WANDB_TAGS="lr-exp"

# External data locations (override via env if needed)
CV_DIR="${CV_DIR:-${ROOT_DIR}/data/cv-corpus-22.0-2025-06-20}"

mkdir -p "${ROOT_DIR}/data/lr_exp"

if [[ ! -d "${CV_DIR}" ]]; then
  echo "[WARN] Common Voice directory not found at: ${CV_DIR}" 1>&2
  echo "       Set CV_DIR=/absolute/path/to/cv-corpus-XX in your environment before running."
fi

shopt -s nullglob
CONFIG_DIR="${ROOT_DIR}/configs/lr_exp"
CONFIGS=("${CONFIG_DIR}"/*.json)

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found in ${CONFIG_DIR}" 1>&2
  exit 1
fi

for CFG_PATH in "${CONFIGS[@]}"; do
  CFG_NAME="$(basename "${CFG_PATH}" .json)"
  echo "\n================ Running LR experiment: ${CFG_NAME} ================\n"

  # Derive languages from config (expects exactly two top-level keys in .languages)
  LANGS_STR="$(python3 - "$CFG_PATH" <<'PY'
import json, sys
with open(sys.argv[1]) as f:
    cfg = json.load(f)
langs = list(cfg.get("languages", {}).keys())
if len(langs) < 2:
    print("")
else:
    print(f"{langs[0]} {langs[1]}")
PY
)"
  if [[ -z "${LANGS_STR}" ]]; then
    echo "[ERROR] Could not infer two languages from ${CFG_PATH}" 1>&2
    exit 1
  fi
  read -r LANG1 LANG2 <<< "${LANGS_STR}"

  # Per-config output directories
  RUN_DIR="${ROOT_DIR}/data/lr_exp/${CFG_NAME}"
  DATA_DIR="${RUN_DIR}/dataset"
  WHISPER_DIR="${RUN_DIR}/whisper"
  NN_DIR="${RUN_DIR}/nn"
  CVX_DIR="${RUN_DIR}/cvx"
  LOG_FILE="${RUN_DIR}/run.log"
  mkdir -p "${RUN_DIR}" "${DATA_DIR}" "${WHISPER_DIR}" "${NN_DIR}" "${CVX_DIR}"

  # Tag runs with lr-exp and the config name
  export WANDB_TAGS="${BASE_WANDB_TAGS},${CFG_NAME}"

  echo "[1/6] Ingesting data → ${DATA_DIR}"
  (
    set -x
    python3 "${ROOT_DIR}/data_ingestion.py" \
      --config "${CFG_PATH}" \
      --out "${DATA_DIR}" \
      --common-voice-dir "${CV_DIR}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[2/6] Training Whisper → ${WHISPER_DIR}"
  RUN_NAME="whisper-small-${CFG_NAME}-$(date +%Y%m%d-%H%M%S)"
  (
    set -x
    python3 "${ROOT_DIR}/whisper_training.py" \
      --data_dir "${DATA_DIR}" \
      --per_device_train_batch_size 16 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 2 \
      --model_id "openai/whisper-small" \
      --output_dir "${WHISPER_DIR}" \
      --learning_rate 1e-5 \
      --num_train_epochs 3 \
      --wandb_project "${WANDB_PROJECT}" \
      --wandb_entity "${WANDB_ENTITY}" \
      --eval_strategy epoch \
      --save_strategy epoch \
      --fp16 \
      --run_name "${RUN_NAME}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[3/6] Training NN CLD head → ${NN_DIR}"
  (
    set -x
    python3 "${ROOT_DIR}/train_nn_cld.py" \
      --output_dir "${NN_DIR}" \
      --data_dir "${DATA_DIR}" \
      --lang1 "${LANG1}" \
      --lang2 "${LANG2}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[4/6] Training CVX model → ${CVX_DIR}"
  (
    set -x
    python3 "${ROOT_DIR}/cronos_trainer.py" \
      --model_name whisper-small \
      --data_dir "${DATA_DIR}" \
      --output_dir "${CVX_DIR}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[5/6] Benchmarking (nn, cvx, vanilla)"

  # Try to locate NN and CVX model files if present
  NN_MODEL_FILE=""
  if compgen -G "${NN_DIR}/*.safetensors" > /dev/null; then
    NN_MODEL_FILE="$(ls -1 "${NN_DIR}"/*.safetensors | head -n1)"
  fi

  CVX_MODEL_FILE=""
  if compgen -G "${CVX_DIR}/*.pkl" > /dev/null; then
    CVX_MODEL_FILE="$(ls -1 "${CVX_DIR}"/*.pkl | head -n1)"
  fi

  # NN benchmark (if model file found)
  if [[ -n "${NN_MODEL_FILE}" && -f "${NN_MODEL_FILE}" ]]; then
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATA_DIR}" \
        --whisper_path "${WHISPER_DIR}" \
        --cld_path "${NN_MODEL_FILE}" \
        --cld_type nn \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] NN model (.safetensors) not found in ${NN_DIR}; skipping NN benchmark" | tee -a "${LOG_FILE}"
  fi

  # CVX benchmark (if model file found)
  if [[ -n "${CVX_MODEL_FILE}" && -f "${CVX_MODEL_FILE}" ]]; then
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATA_DIR}" \
        --whisper_path "${WHISPER_DIR}" \
        --cld_path "${CVX_MODEL_FILE}" \
        --cld_type cvx \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] CVX model (.pkl) not found in ${CVX_DIR}; skipping CVX benchmark" | tee -a "${LOG_FILE}"
  fi

  # Vanilla benchmark (Whisper's built-in detection)
  (
    set -x
    python3 "${ROOT_DIR}/benchmark_cld.py" \
      --dataset_path "${DATA_DIR}" \
      --whisper_path openai/whisper-small \
      --cld_path unused \
      --cld_type vanilla \
      --lang1 "${LANG1}" \
      --lang2 "${LANG2}" \
      --batch_size 8
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[6/6] Completed run: ${CFG_NAME} → outputs at ${RUN_DIR}"
done

echo "\nAll LR experiments complete. Outputs under: ${ROOT_DIR}/data/lr_exp"

