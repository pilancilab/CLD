#!/bin/bash

# Example script to run Whisper training
# Make sure you have prepared your data using data_ingestion.py first

# Set up environment
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0, adjust as needed
export WANDB_PROJECT="whisper-multilingual"

# Training parameters
DATA_DIR="data/en_hi"  # Change to your data directory
TRAIN_BATCH_SIZE=16
EVAL_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=2
MODEL_ID="openai/whisper-small"  # or whisper-base, whisper-medium, etc.
OUTPUT_DIR="./models/whisper-multilingual-en-hi"
LEARNING_RATE=1e-5
BATCH_SIZE=16
EPOCHS=5
EVAL_STEPS=1000
SAVE_STEPS=1000

# Run training
python whisper_training.py \
    --data_dir "$DATA_DIR" \
    --per_device_train_batch_size "$TRAIN_BATCH_SIZE"  \
    --per_device_eval_batch_size "$EVAL_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --model_id "$MODEL_ID" \
    --output_dir "$OUTPUT_DIR" \
    --learning_rate "$LEARNING_RATE" \
    --num_train_epochs "$EPOCHS" \
    --wandb_project "$WANDB_PROJECT" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --fp16 \
    --run_name "whisper-small-en-hi-$(date +%Y%m%d-%H%M%S)"

echo "Training completed! Check the output directory: $OUTPUT_DIR"
echo "View training logs at: https://wandb.ai"
