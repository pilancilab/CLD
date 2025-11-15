#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/dataset" 1>&2
  exit 1
fi

DATASET_PATH="$1"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ============================
# Models to iterate over
# ============================
MODEL_NAMES=(
  "openai/whisper-small"
  "openai/whisper-medium"
  "openai/whisper-large-v3"
)

# ============================
# WANDB
# ============================
export WANDB_ENTITY="${WANDB_ENTITY:-lucigen}"
export WANDB_PROJECT="${WANDB_PROJECT:-CLD}"
export WANDB_TAGS="multi-model"

# ============================
# Languages (hardcode or infer)
# ============================
LANG1="en"
LANG2="zh"

# ============================
# Output root
# ============================
OUT_ROOT="${ROOT_DIR}/data/model_exp"
mkdir -p "${OUT_ROOT}"

echo "[INFO] Dataset path: ${DATASET_PATH}"
echo "[INFO] Output root: ${OUT_ROOT}"

# ============================================================
# LOOP OVER MODEL NAMES
# ============================================================
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  SAFE_NAME="$(basename "${MODEL_NAME}" | tr '/' '_')"  # safe for dirs
  RUN_DIR="${OUT_ROOT}/${SAFE_NAME}"
  NN_DIR="${RUN_DIR}/nn"
  CVX_DIR="${RUN_DIR}/cvx"
  LOG_FILE="${RUN_DIR}/run.log"

  mkdir -p "${RUN_DIR}" "${NN_DIR}" "${CVX_DIR}"

  echo ""
  echo "================ MODEL: ${MODEL_NAME} ================"
  echo ""

  # --------------------------------------------------------
  # 1. Train NN Head
  # --------------------------------------------------------
  echo "[1/4] Training NN head → ${NN_DIR}"
  export WANDB_TAGS="${BASE_WANDB_TAGS},nn,${MODEL_NAME}" 
  export WANDB_NAME="nn-train-${LANG1}-${LANG2}-$(date +%Y%m%d-%H%M%S)"    
  (
    set -x
    python3 "${ROOT_DIR}/train_nn_cld.py" \
      --output_dir "${NN_DIR}" \
      --model "${MODEL_NAME}" \
      --data_dir "${DATASET_PATH}" \
      --lang1 "${LANG1}" \
      --lang2 "${LANG2}" \
  ) 2>&1 | tee -a "${LOG_FILE}"

  NN_MODEL_FILE=""
  if compgen -G "${NN_DIR}/*.safetensors" > /dev/null; then
    NN_MODEL_FILE="$(ls -1 "${NN_DIR}"/*.safetensors | head -n 1)"
  fi

  # --------------------------------------------------------
  # 2. Train CVX/Chronos head
  # --------------------------------------------------------
  echo "[2/4] Training CVX (Chronos) model → ${CVX_DIR}"
  export WANDB_TAGS="${BASE_WANDB_TAGS},cvxnn,${MODEL_NAME}" 
  export WANDB_NAME="cvxnn-train-${LANG1}-${LANG2}-$(date +%Y%m%d-%H%M%S)"    
  (
    set -x
    python3 "${ROOT_DIR}/cronos_trainer.py" \
      --model_name "${MODEL_NAME}" \
      --data_dir "${DATASET_PATH}" \
      --output_dir "${CVX_DIR}"
  ) 2>&1 | tee -a "${LOG_FILE}"

  CVX_MODEL_FILE=""
  if compgen -G "${CVX_DIR}/*.pkl" > /dev/null; then
    CVX_MODEL_FILE="$(ls -1 "${CVX_DIR}"/*.pkl | head -n 1)"
  fi

  # --------------------------------------------------------
  # 3. Benchmark
  # --------------------------------------------------------
  echo "[3/4] Benchmarking NN + CVX + Vanilla"
  
  # ===== NN Benchmark =====
  if [[ -n "${NN_MODEL_FILE}" ]]; then
    echo "[INFO] Benchmarking NN → ${NN_MODEL_FILE}"
    export WANDB_TAGS="${BASE_WANDB_TAGS},nn,${MODEL_NAME}" 
    export WANDB_NAME="nn-benchmark-${LANG1}-${LANG2}-$(date +%Y%m%d-%H%M%S)"    
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATASET_PATH}" \
        --model_name "${MODEL_NAME}" \
        --cld_path "${NN_MODEL_FILE}" \
        --cld_type nn \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] No NN model found for ${MODEL_NAME}"
  fi

  # ===== CVX Benchmark =====
  if [[ -n "${CVX_MODEL_FILE}" ]]; then
    echo "[INFO] Benchmarking CVX → ${CVX_MODEL_FILE}"
    export WANDB_TAGS="${BASE_WANDB_TAGS},cvxnn,${MODEL_NAME}" 
    export WANDB_NAME="cvxnn-benchmark-${LANG1}-${LANG2}-$(date +%Y%m%d-%H%M%S)"    
    (
      set -x
      python3 "${ROOT_DIR}/benchmark_cld.py" \
        --dataset_path "${DATASET_PATH}" \
        --model_name "${MODEL_NAME}" \
        --cld_path "${CVX_MODEL_FILE}" \
        --cld_type cvx \
        --lang1 "${LANG1}" \
        --lang2 "${LANG2}" \
        --batch_size 8
    ) 2>&1 | tee -a "${LOG_FILE}"
  else
    echo "[WARN] No CVX model found for ${MODEL_NAME}"
  fi

  # ===== Vanilla Benchmark =====
  echo "[INFO] Benchmarking Whisper (vanilla)"
  export WANDB_TAGS="${BASE_WANDB_TAGS},${MODEL_NAME}" 
    export WANDB_NAME="vanilla-benchmark-${LANG1}-${LANG2}-$(date +%Y%m%d-%H%M%S)"    
    (
    set -x
    python3 "${ROOT_DIR}/benchmark_cld.py" \
      --dataset_path "${DATASET_PATH}" \
      --model_name "${MODEL_NAME}" \
      --cld_path unused \
      --cld_type vanilla \
      --lang1 "${LANG1}" \
      --lang2 "${LANG2}" \
      --batch_size 8
  ) 2>&1 | tee -a "${LOG_FILE}"

  echo "[4/4] Completed model: ${MODEL_NAME}"
done

echo ""
echo "All model sweeps complete. Outputs at: ${OUT_ROOT}"
