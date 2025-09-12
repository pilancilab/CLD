#!/usr/bin/env bash

set -e  

# Initialize best‐score tracking
best_acc=0
best_dir=""

# Define the grid
LRS=(1e-4 3e-5 1e-4 3e-5 5e-5)
EPOCHS=(5 5 8 8 10)

for lr in "${LRS[@]}"; do
  for epochs in ${EPOCHS}; do
    
    outdir="detection_head_lr${lr}_ep${epochs}"
    mkdir -p "$outdir"

    echo "==============================================="
    echo "▶ Running with lr=$lr , epochs=$epochs → $outdir"
    echo "==============================================="

    # Run the training. 
    # We pass --save_total_limit 5 so that Trainer only keeps 5 checkpoints on disk.
    # We tee the output to a log file so we can parse validation accuracy afterwards.
    python /home/ubuntu/arizonafiles/voice_clone/detect_trainer2.py \
      --learning_rate "$lr" \
      --num_train_epochs "$epochs" \
      --output_dir "$outdir" \
      2>&1 | tee "$outdir/train.log"

    # After training completes, extract the final validation accuracy from the log.
    # We assume detect_trainer.py prints a line like: "eval_accuracy = 0.82"
    acc_line=$(grep -E "eval_accuracy\s*=\s*[0-9]+\.[0-9]+" "$outdir/train.log" | tail -1)
    if [[ -z "$acc_line" ]]; then
      echo " Warning: Could not find 'eval_accuracy' in $outdir/train.log"
      continue
    fi

    # Parse out the numeric value
    acc=$(echo "$acc_line" | awk -F'=' '{print $2}' | tr -d '[:space:]')
    echo "→ Validation accuracy for (lr=$lr, epochs=$epochs) : $acc"

    # Compare to the best so far
    # Use bc for floating‐point comparison
    is_better=$(echo "$acc > $best_acc" | bc -l)
    if [[ "$is_better" -eq 1 ]]; then
      best_acc="$acc"
      best_dir="$outdir"
    fi
  done
done

echo ""
echo "==============================================="
echo "Grid search complete."
echo "Best checkpoint directory: $best_dir"
echo "Best validation accuracy: $best_acc"
echo "==============================================="
