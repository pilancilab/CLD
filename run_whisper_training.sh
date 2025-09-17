#!/bin/bash

# Example script to run Whisper training
# Make sure you have prepared your data using data_ingestion.py first

# Set up environment
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export WANDB_PROJECT="whisper-multilingual"

# Training parameters
DATA_DIR="data/en_hi"  # Change to your data directory
MODEL_NAME="openai/whisper-small"  # or whisper-base, whisper-medium, etc.
OUTPUT_DIR="./models/whisper-multilingual-en-hi"
LANGUAGES="en hi"
LEARNING_RATE=1e-5
BATCH_SIZE=16
MAX_STEPS=5000
EVAL_STEPS=1000
SAVE_STEPS=1000

# Run training
python whisper_training.py \
    --data_dir "$DATA_DIR" \
    --model_name "$MODEL_NAME" \
    --output_dir "$OUTPUT_DIR" \
    --languages $LANGUAGES \
    --learning_rate "$LEARNING_RATE" \
    --batch_size "$BATCH_SIZE" \
    --max_steps "$MAX_STEPS" \
    --eval_steps "$EVAL_STEPS" \
    --save_steps "$SAVE_STEPS" \
    --use_wandb \
    --wandb_project "$WANDB_PROJECT" \
    --fp32 \
    --wandb_run_name "whisper-small-en-hi-$(date +%Y%m%d-%H%M%S)"

echo "Training completed! Check the output directory: $OUTPUT_DIR"
echo "View training logs at: https://wandb.ai"
