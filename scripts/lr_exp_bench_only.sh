#!/usr/bin/env bash
set -euo pipefail

# Resolve repo root (this script lives in ROOT/scripts/lr_exp_bench_only.sh)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# WANDB setup
export WANDB_ENTITY="${WANDB_ENTITY:-lucigen}"
export WANDB_PROJECT="${WANDB_PROJECT:-CLD}"
BASE_WANDB_TAGS="lr-exp-bench-only"

# Configs to iterate
CONFIG_DIR="${ROOT_DIR}/configs/lr_exp"
shopt -s nullglob
CONFIGS=("${CONFIG_DIR}"/*.json)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "No configs found in ${CONFIG_DIR}" 1>&2
  exit 1
fi

# Fixed dataset source: default to 10000_config's dataset; override via FIXED_DATA_CFG
FIXED_DATA_CFG="${FIXED_DATA_CFG:-10000_config}"
FIXED_DATA_DIR="${ROOT_DIR}/data/lr_exp/${FIXED_DATA_CFG}/dataset"

if [[ ! -d "${FIXED_DATA_DIR}" ]]; then
  echo "[ERROR] Fixed dataset not found at: ${FIXED_DATA_DIR}" 1>&2
  echo "        Generate it first (e.g., via scripts/lr_exp.sh for ${FIXED_DATA_CFG})." 1>&2
  exit 1
fi

for CFG_PATH in "${CONFIGS[@]}"; do
  CFG_NAME="$(basename "${CFG_PATH}" .json)"
  echo "\n================ Benchmark-only: ${CFG_NAME} (dataset: ${FIXED_DATA_CFG}) ================\n"

  # Derive languages from config (expects at least two top-level keys in .languages)
  LANGS_STR="$(python3 - "${CFG_PATH}" <<'PY'
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

  # Per-config model directories (models from this config), but dataset fixed to FIXED_DATA_DIR
  RUN_DIR="${ROOT_DIR}/data/lr_exp/${CFG_NAME}"
  DATA_DIR="${FIXED_DATA_DIR}"
  WHISPER_DIR="${RUN_DIR}/whisper"
  NN_DIR="${RUN_DIR}/nn"
  CVX_DIR="${RUN_DIR}/cvx"
  LOG_FILE="${RUN_DIR}/bench_only.log"
  mkdir -p "${RUN_DIR}"

  # Resolve latest checkpoint produced by the fine-tuning step for inference (if dir exists)
  WHISPER_CKPT_PATH=""
  if [[ -d "${WHISPER_DIR}" ]]; then
    WHISPER_CKPT_PATH="$(find "${WHISPER_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | grep -o '[0-9]\+$' | sort -nr | head -n 1 | xargs -I {} find "${WHISPER_DIR}" -maxdepth 1 -type d -name "checkpoint-{}" | head -n 1)"
    if [[ -z "${WHISPER_CKPT_PATH}" ]]; then
      echo "[WARN] No checkpoint-* directory found in ${WHISPER_DIR}; using ${WHISPER_DIR} as model path" | tee -a "${LOG_FILE}"
      WHISPER_CKPT_PATH="${WHISPER_DIR}"
    else
      echo "[INFO] Using Whisper checkpoint for inference: ${WHISPER_CKPT_PATH}" | tee -a "${LOG_FILE}"
    fi
  else
    echo "[WARN] Whisper dir ${WHISPER_DIR} does not exist; vanilla benchmark may fail" | tee -a "${LOG_FILE}"
  fi

  # Try to locate NN and CVX model files if present
  NN_MODEL_FILE=""
  if compgen -G "${NN_DIR}/*.safetensors" > /dev/null; then
    NN_MODEL_FILE="$(ls -1 "${NN_DIR}"/*.safetensors | head -n1)"
  fi

  CVX_MODEL_FILE=""
  if compgen -G "${CVX_DIR}/*.pkl" > /dev/null; then
    CVX_MODEL_FILE="$(ls -1 "${CVX_DIR}"/*.pkl | head -n1)"
  fi

  echo "[1/3] NN benchmark (if available)" | tee -a "${LOG_FILE}"
  if [[ -n "${NN_MODEL_FILE}" && -f "${NN_MODEL_FILE}" ]]; then
    export WANDB_TAGS="${BASE_WANDB_TAGS},${CFG_NAME},benchmark,nn,${LANG1}-${LANG2},fixed-${FIXED_DATA_CFG}"
    export WANDB_NAME="bench-nn-${LANG1}-${LANG2}-${CFG_NAME}-fixed-${FIXED_DATA_CFG}-$(date +%Y%m%d-%H%M%S)"
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATA_DIR}" \
        --whisper_path "${WHISPER_CKPT_PATH}" \
        --cld_path "${NN_MODEL_FILE}" \
        --cld_type nn \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] NN model (.safetensors) not found in ${NN_DIR}; skipping NN benchmark" | tee -a "${LOG_FILE}"
  fi

  echo "[2/3] CVX benchmark (if available)" | tee -a "${LOG_FILE}"
  if [[ -n "${CVX_MODEL_FILE}" && -f "${CVX_MODEL_FILE}" ]]; then
    export WANDB_TAGS="${BASE_WANDB_TAGS},${CFG_NAME},benchmark,cvxnn,${LANG1}-${LANG2},fixed-${FIXED_DATA_CFG}"
    export WANDB_NAME="bench-cvxnn-${LANG1}-${LANG2}-${CFG_NAME}-fixed-${FIXED_DATA_CFG}-$(date +%Y%m%d-%H%M%S)"
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATA_DIR}" \
        --whisper_path "${WHISPER_CKPT_PATH}" \
        --cld_path "${CVX_MODEL_FILE}" \
        --cld_type cvx \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] CVX model (.pkl) not found in ${CVX_DIR}; skipping CVX benchmark" | tee -a "${LOG_FILE}"
  fi

  echo "[3/3] Whisper vanilla benchmark" | tee -a "${LOG_FILE}"
  export WANDB_TAGS="${BASE_WANDB_TAGS},${CFG_NAME},benchmark,whisper-vanilla,${LANG1}-${LANG2},fixed-${FIXED_DATA_CFG}"
  export WANDB_NAME="bench-whisper-vanilla-${LANG1}-${LANG2}-${CFG_NAME}-fixed-${FIXED_DATA_CFG}-$(date +%Y%m%d-%H%M%S)"
  (
    set -x
    python3 "${ROOT_DIR}/benchmark_cld.py" \
      --dataset_path "${DATA_DIR}" \
      --whisper_path "${WHISPER_CKPT_PATH}" \
      --cld_path unused \
      --cld_type vanilla \
      --lang1 "${LANG1}" \
      --lang2 "${LANG2}" \
      --batch_size 8
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[DONE] Completed benchmarks for ${CFG_NAME} using dataset ${FIXED_DATA_CFG} → outputs at ${RUN_DIR}" | tee -a "${LOG_FILE}"
done

echo "\n================ Benchmark-only: default openai/whisper-small (dataset: ${FIXED_DATA_CFG}) ================\n"

# Extra: run vanilla benchmark with default OpenAI whisper-small on the fixed dataset
export WANDB_TAGS="${BASE_WANDB_TAGS},benchmark,whisper-vanilla,default-openai-whisper-small,fixed-${FIXED_DATA_CFG}"
export WANDB_NAME="bench-whisper-vanilla-default-openai-whisper-small-fixed-${FIXED_DATA_CFG}-$(date +%Y%m%d-%H%M%S)"
(
  set -x
  python3 "${ROOT_DIR}/benchmark_cld.py" \
    --dataset_path "${FIXED_DATA_DIR}" \
    --whisper_path "openai/whisper-small" \
    --cld_path unused \
    --cld_type vanilla \
    --batch_size 8
) 2>&1 | tee -a "${ROOT_DIR}/data/lr_exp/bench_default_openai_whisper_small.log"

echo "\nAll benchmark-only runs complete. Fixed dataset: ${FIXED_DATA_DIR}"


