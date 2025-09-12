#!/usr/bin/env python3
"""
train binary classification head en - zh 
PROBLEM: ratio of training data is unbalanced, add more english if necessary
python detect_trainer2.py --output_dir="detecthead0"
This script now accepts hyperparameters from the command line:
    --learning_rate
    --num_train_epochs
    --save_total_limit
    --output_dir
    --per_device_train_batch_size
    --per_device_eval_batch_size

=== Dataset Size Summary ===
Singlish train size        : 2178
Singlish test size         : 121
Singlish2 train size       : 308226
Singlish2 test size        : 34248
Mandarin train size        : 80064
Mandarin test size         : 10626
English Aug (CommonVoice) train: 36000
English Aug (CommonVoice) test : 4001

"""

import os
import sys
import argparse
import torchaudio
import torch
import numpy as np
import torch.nn as nn

from datasets import load_from_disk, Audio, DatasetDict, concatenate_datasets
from transformers import (
    WhisperProcessor,
    WhisperModel,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a tiny Whisper-based binary language detector head (en vs zh)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save the model and checkpoints.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate for the optimizer.",
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=10,
        help="Total number of training epochs.",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=2,
        help="Maximum number of checkpoints to keep. Older ones will be deleted.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=32,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of steps to accumulate gradients before updating.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use mixed-precision (FP16).",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        choices=["wandb", "none"],
        help="Where to report logs (wandb or none).",
    )
    return parser.parse_args()


def resample_audio_consistent(batch, target_sr=16000):
    """Resample and ensure consistent audio format"""
    audio = batch["audio"]
    arr = audio["array"]
    orig_sr = audio["sampling_rate"]

    if orig_sr != target_sr:
        arr = torchaudio.functional.resample(
            torch.tensor(arr, dtype=torch.float32),
            orig_sr,
            target_sr
        ).numpy()

    batch["audio"] = {
        "array": np.array(arr, dtype=np.float32),
        "sampling_rate": target_sr,
        "path": None
    }
    return batch



def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    TARGET_SR = 16000
    # Defining the consistent schema
    audio_feature = Audio(sampling_rate=16000, mono=True, decode=True)


    # 1) Load processor (tokenizer + feature_extractor)
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    # 2) Load cached datasets from disk Singlish (English) part
    singlish2_train = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/nsc_processed_dataset2/train") # 308226
    singlish2_test = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/nsc_processed_dataset2/test") # 34248
    singlish_train_ds = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/singlish_train_cached") # 2178
    singlish_test_ds  = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/singlish_test_cached") # 121

    #    Mandarin part (we only take as many samples as English+Singlish total)
    mandarin_train_all = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/mandarin_train_cached") # 80064
    mandarin_test_all  = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/mandarin_test_cached") # 10626
    
    en_full = load_from_disk("/home/ubuntu/arizonafiles/voice_clone/datasets/en_processed_disk")
    split = en_full.train_test_split(test_size=0.1, seed=1024)

    en_train_all = split["train"] # 36000
    en_test_all = split["test"] # 4001
        

    # Calculate total available samples for each language

    total_singlish2_train = len(singlish2_train)
    total_singlish2_test = len(singlish2_test)
    total_singlish_train = len(singlish_train_ds) #+ len(singlish2_train)
    total_singlish_test = len(singlish_test_ds) #+ len(singlish2_test)
    total_mandarin_train = len(mandarin_train_all)
    total_mandarin_test = len(mandarin_test_all)
    total_en_train = len(en_train_all)
    total_en_test = len(en_test_all)
    total_sing_en_train = total_singlish_train + total_en_train + total_singlish2_train
    total_sing_en_test = total_singlish_test + total_en_test + total_singlish2_test

    print("\n=== Dataset Size Summary ===")
    print(f"Singlish train size        : {len(singlish_train_ds)}")
    print(f"Singlish test size         : {len(singlish_test_ds)}")
    print(f"Singlish2 train size       : {len(singlish2_train)}")
    print(f"Singlish2 test size        : {len(singlish2_test)}")
    print(f"Mandarin train size        : {total_mandarin_train}")
    print(f"Mandarin test size         : {total_mandarin_test}")
    print(f"English (CommonVoice) train: {total_en_train}")
    print(f"English (CommonVoice) test : {total_en_test}")
    

    # Find the minimum between Singlish and Mandarin for each split
    # This ensures we can have a balanced 50/50 dataset
    n_train_per_lang = min(total_sing_en_train, total_mandarin_train)
    n_test_per_lang = min(total_sing_en_test, total_mandarin_test)

    #print(f"Available Singlish train: {total_singlish_train}, Mandarin train: {total_mandarin_train}")
    #print(f"Available Singlish test: {total_singlish_test}, Mandarin test: {total_mandarin_test}")
    print(f"Using {n_train_per_lang} samples for each individual language for train") # 38178
    print(f"Using {n_test_per_lang} samples for each individual language for test")
    # exit()

    # Now select the appropriate number of samples from each language
    # For Mandarin (always select from the beginning)
    mandarin_train_ds = mandarin_train_all.select(range(n_train_per_lang))
    mandarin_test_ds = mandarin_test_all.select(range(n_test_per_lang))

    # For Singlish, we need to handle the case where we might need to trim since it comes from min two sources
    if total_sing_en_train > n_train_per_lang:
        # Need to trim English samples
        if len(singlish_train_ds) >= n_train_per_lang:
            # Take only from singlish_train_ds
            singlish_train_final = singlish_train_ds.select(range(n_train_per_lang))
            #singlish2_train_final = singlish2_train.select(range(0))  # Empty dataset
        else:
            # Take all of singlish_train_ds and some from singlish2_train/cv_en
            singlish_train_final = singlish_train_ds
            remaining_needed = n_train_per_lang - len(singlish_train_ds)
            if remaining_needed > len(en_train_all):
                en_train_final = en_train_all
                remaining_needed_last = n_train_per_lang - len(en_train_all) - len(singlish_train_ds)
                singlish2_train_final = singlish2_train.select(range(remaining_needed_last))
            else: 
                #singlish2_train_final = singlish2_train.select(range(remaining_needed))
                en_train_final = en_train_all.select(range(remaining_needed))
                singlish2_train_final = singlish2_train.select(range(0))
    else:
        # Use all available Singlish samples (this case shouldn't happen with the min logic)
        singlish_train_final = singlish_train_ds
        #singlish2_train_final = singlish2_train
        en_train_final = en_train_all

    # Same logic for test set
    if total_sing_en_test > n_test_per_lang:
        if len(singlish_test_ds) >= n_test_per_lang:
            singlish_test_final = singlish_test_ds.select(range(n_test_per_lang))
            #singlish2_test_final = singlish2_test.select(range(0))
        else:
            singlish_test_final = singlish_test_ds
            remaining_needed = n_test_per_lang - len(singlish_test_ds)
            if remaining_needed > len(en_test_all):
                en_test_final = en_test_all
                remaining_needed_last = n_test_per_lang - len(en_test_all) - len(singlish_test_ds)
                singlish2_test_final = singlish2_test.select(range(remaining_needed_last))
            else: 
                #singlish2_test_final = singlish2_test.select(range(remaining_needed))
                en_test_final = en_test_all.select(range(remaining_needed))
                singlish2_test_final = singlish2_test.select(range(0))
    else:
        singlish_test_final = singlish_test_ds
        en_test_final = en_test_all
        #singlish2_test_final = singlish2_test

    # Verify the balance
    print(f"\nFinal dataset sizes:")
    #print(f"Singlish train: {len(singlish_train_final) + len(singlish2_train_final)}")
    print(f"Singlish and EN train: {len(singlish_train_final) + len(en_train_final) + len(singlish2_train_final)}")
    print(f"Mandarin train: {len(mandarin_train_ds)}")
    #print(f"Singlish test: {len(singlish_test_final) + len(singlish2_test_final)}")
    print(f"Singlish and EN test: {len(singlish_test_final) + len(en_test_final) + len(singlish2_test_final)}")
    print(f"Mandarin test: {len(mandarin_test_ds)}")
    
    # Apply resampling + casting, this part is slow
    singlish2_train_final = singlish2_train_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    singlish2_train_final = singlish2_train_final.cast_column("audio", audio_feature)

    singlish2_test_final = singlish2_test_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    singlish2_test_final = singlish2_test_final.cast_column("audio", audio_feature)

    en_train_final = en_train_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    en_train_final = en_train_final.cast_column("audio", audio_feature)

    en_test_final = en_test_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    en_test_final = en_test_final.cast_column("audio", audio_feature)


    singlish_train_final = singlish_train_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    singlish_train_final = singlish_train_final.cast_column("audio", audio_feature)

    singlish_test_final = singlish_test_final.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    singlish_test_final = singlish_test_final.cast_column("audio", audio_feature)

    mandarin_train_ds = mandarin_train_ds.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    mandarin_train_ds = mandarin_train_ds.cast_column("audio", audio_feature)

    mandarin_test_ds = mandarin_test_ds.map(resample_audio_consistent, fn_kwargs={"target_sr": 16000}, desc="Resampling")
    mandarin_test_ds = mandarin_test_ds.cast_column("audio", audio_feature)



    # 3) Add language labels: 0 = English/Singlish, 1 = Mandarin
    def tag_batch(example, lang_id):
        example["labels"] = lang_id
        return example

    singlish2_train_labeled = singlish2_train_final.map(lambda x: tag_batch(x, 0))
    singlish2_test_labeled  = singlish2_test_final.map(lambda x: tag_batch(x, 0))
    en_train_labeled = en_train_final.map(lambda x: tag_batch(x, 0))
    en_test_labeled  = en_test_final.map(lambda x: tag_batch(x, 0))
    singlish_train_labeled = singlish_train_final.map(lambda x: tag_batch(x, 0))
    singlish_test_labeled  = singlish_test_final.map(lambda x: tag_batch(x, 0))
    mandarin_train_labeled = mandarin_train_ds.map(lambda x: tag_batch(x, 1))
    mandarin_test_labeled  = mandarin_test_ds.map(lambda x: tag_batch(x, 1))

    # 4) Concatenate train and test splits
    train_data = concatenate_datasets([singlish_train_labeled, mandarin_train_labeled, en_train_labeled, singlish2_train_labeled])
    test_data  = concatenate_datasets([singlish_test_labeled,  mandarin_test_labeled,  en_test_labeled, singlish2_test_labeled])
    print(f"Final train size: {len(train_data)}")
    print(f"Final test size: {len(test_data)}")

    raw_datasets = DatasetDict({"train": train_data, "test": test_data}) # make a dataset dict
    raw_datasets["train"] = raw_datasets["train"].shuffle(seed=1024)

    print(f"Combined bilingual dataset: train (shuffled)={len(raw_datasets['train'])}, test={len(raw_datasets['test'])}")

    # 5) Preprocess: convert each audio→mel features, store in "input_features"
    def preprocess(batch):
        audio_array = batch["audio"]["array"]
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array)
        feats = processor.feature_extractor(
            audio_array,
            sampling_rate=TARGET_SR,
            return_tensors="pt"
        ).input_features[0]
        batch["input_features"] = feats
        return batch

    print("Preprocessing audio features...")
    raw_datasets = raw_datasets.map(
        preprocess,
        remove_columns=["audio", "text"],
        batched=False,
        desc="Extracting features",
    )

    # 6) Build the LangDetectHead on top of Whisper's encoder
    class LangDetectHead(nn.Module):
        def __init__(self):
            super().__init__()
            whisper_model = WhisperModel.from_pretrained("openai/whisper-small")
            # Freeze all Whisper weights
            for p in whisper_model.parameters():
                p.requires_grad = False
            self.encoder = whisper_model.encoder
            self.classifier = nn.Sequential(
                nn.Linear(whisper_model.config.d_model, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, 2),
            )
            print(f"[Info] Whisper encoder d_model = {whisper_model.config.d_model}")

        def forward(self, input_features, labels=None):
            # Pass through frozen encoder
            with torch.no_grad():
                hidden = self.encoder(input_features).last_hidden_state  # (B, T, D)
            pooled = hidden.mean(dim=1)  # (B, D)
            logits = self.classifier(pooled)  # (B, 2)
            loss = None
            if labels is not None:
                loss = nn.CrossEntropyLoss()(logits, labels)
            return {"loss": loss, "logits": logits}

    print("Initializing LangDetectHead model...")
    model = LangDetectHead()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Info] Total parameters: {total_params:,}")
    print(f"[Info] Trainable parameters: {trainable_params:,}")

    # 7) Define metrics
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc}

    # 8) Construct TrainingArguments using command-line hyperparameters
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        #warmup_steps=100,
        warmup_ratio=0.1,
        save_total_limit=args.save_total_limit,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        #early_stopping_patience=3,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        fp16=args.fp16,
        report_to=args.report_to if args.report_to != "none" else None,
    )

    print(f"[Info] TrainingArguments:\n  output_dir = {args.output_dir}\n" +
          f"  learning_rate = {args.learning_rate}\n" +
          f"  num_train_epochs = {args.num_train_epochs}\n" +
          f"  save_total_limit = {args.save_total_limit}\n" +
          f"  per_device_train_batch_size = {args.per_device_train_batch_size}\n" +
          f"  per_device_eval_batch_size = {args.per_device_eval_batch_size}\n" +
          f"  fp16 = {args.fp16}\n" +
          f"  report_to = {args.report_to}\n")

    # 9) Instantiate Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["test"],
        compute_metrics=compute_metrics,
    )

    print("[Info] Starting training …")
    trainer.train()

    print("\n=== Final Evaluation ===")
    eval_results = trainer.evaluate()
    for key, value in eval_results.items():
        print(f"{key}: {value}")

    print(f"eval_accuracy = {eval_results['accuracy']:.4f}")

    print(f"\n[Info] Saving model to {args.output_dir} …")
    trainer.save_model()

if __name__ == "__main__":
    main()
