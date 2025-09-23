python data_ingestion.py --config configs/random_config.json --out data/rand --musan-dir musan/noise/free-sound --common-voice-dir data/cv-corpus-22.0-2025-06-20/

WANDB_NAME="whisper-eval-random-$(date +%Y%m%d-%H%M%S)" python benchmark_cld.py --dataset_path data/rand --whisper_path openai/whisper-small --cld_path blah --cld_type vanilla --lang1 blah --lang2 blah --batch_size 8