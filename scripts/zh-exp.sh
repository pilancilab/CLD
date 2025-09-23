
# Scripts for Diverse experiment

export WANDB_ENTITY="lucigen"
export WANDB_PROJECT="CLD"
export WANDB_TAGS="en-zh-exp"

python data_ingestion.py --config configs/en_zh_config.json --out data/en_zh --musan-dir musan/noise/free-sound --common-voice-dir data/cv-corpus-22.0-2025-06-20/

# finetune whisper
python whisper_training.py \
    --data_dir "data/en_zh" \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --model_id "openai/whisper-small" \
    --output_dir "./models/whisper-multilingual-en-zh" \
    --learning_rate 1e-5 \
    --num_train_epochs 10 \
    --wandb_project "$WANDB_PROJECT" \
    --eval_strategy epoch \
    --save_strategy epoch \
    --fp16 \
    --run_name "whisper-small-en-zh-$(date +%Y%m%d-%H%M%S)"

FINETUNE_WHISPER_PATH=$(find "./models/whisper-multilingual-en-zh" -maxdepth 1 -type d -name 'checkpoint-*' | grep -o '[0-9]\+$' | sort -nr | head -n 1 | xargs -I {} find "./models/whisper-multilingual-en-zh" -maxdepth 1 -type d -name 'checkpoint-{}')
# train nn

WANDB_NAME="nn-cld-training-$(date +%Y%m%d-%H%M%S)" python train_nn_cld.py --output_dir="models/en_zh_nn" --data_dir="data/en_zh" --lang1="en" --lang2="zh"

# train cvx

WANDB_NAME="cvxnn-cld-training-$(date +%Y%m%d-%H%M%S)" python cronos_trainer.py --model_name whisper-small --data_dir data/en_zh --output_dir models/en_zh_cvx

# benchmark


WANDB_NAME="whisper-finetuned-nn-head-eval-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py --dataset_path data/en_zh --whisper_path "$FINETUNE_WHISPER_PATH" --cld_path models/en_zh_nn/model.safetensors --cld_type nn --lang1 en --lang2 zh --batch_size 8

WANDB_NAME="whisper-finetuned-cvxnn-head-eval-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py --dataset_path data/en_zh --whisper_path "$FINETUNE_WHISPER_PATH" --cld_path models/en_zh_cvx/whisper-small_trained_cvx_mlp.pkl --cld_type cvx --lang1 en --lang2 zh --batch_size 8

WANDB_NAME="whisper-finetune-eval-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py --dataset_path data/en_zh --whisper_path "$FINETUNE_WHISPER_PATH" --cld_path blah --cld_type vanilla --lang1 en --lang2 zh --batch_size 8

WANDB_NAME="whisper-eval-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py --dataset_path data/en_zh --whisper_path openai/whisper-small --cld_path blah --cld_type vanilla --lang1 en --lang2 zh --batch_size 8