import argparse
from datasets import load_from_disk
from solve.models.load_whisper_pipeline import get_nn_pipeline, inference, detect_language_vanilla
from sklearn.metrics import classification_report
from transformers import WhisperProcessor
import evaluate
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on a dataset for language detection and transcription.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory (e.g., 'data/en_hi/').")
    parser.add_argument("--whisper_path", type=str, default="models/whisper-small-enhi-out", help="Path to the Whisper model directory.")
    parser.add_argument("--cld_path", type=str, default="models/en_hi_nn", help="Path to the language classifier model directory.")
    parser.add_argument("--cld_type", type=str, default="nn", choices=["nn", "cvx", "vanilla"], help="Detection head architecture.")
    parser.add_argument("--lang1", type=str, default="en", help="First language code (e.g., 'en' for English).")
    parser.add_argument("--lang2", type=str, default="hi", help="Second language code (e.g., 'hi' for Hindi).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()

def main():
    args = parse_args()

    wandb.init()

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    ds = load_from_disk(args.dataset_path)

    model, processor = get_nn_pipeline(
        whisper_path=args.whisper_path,
        cld_path=args.cld_path,
        cld_type=args.cld_type,
        lang1=args.lang1,
        lang2=args.lang2
    )

    # Use test split consistently
    test_ds = ds["test"]
    true_language_ids = [x["lang"] for x in test_ds]
    true_texts = [x["text"] for x in test_ds]

    pred_language_ids = []
    pred_texts = []
    
    batch_temp = []
    for i, sample in enumerate(test_ds):
        batch_temp.append(sample["audio"]["array"])
        if len(batch_temp) == args.batch_size or i == len(test_ds) - 1:
            language_ids_batch, texts_batch = inference(model, processor, batch_temp)
            pred_language_ids.extend(language_ids_batch)
            pred_texts.extend(texts_batch)
            batch_temp = []

    wer = 100.0 * wer_metric.compute(predictions=pred_texts, references=true_texts)
    cer = 100.0 * cer_metric.compute(predictions=pred_texts, references=true_texts)

    # Get classification report as dict
    report = classification_report(true_language_ids, pred_language_ids, output_dict=True)

    # Log WER and CER
    wandb.log({
        "eval/wer": wer,
        "eval/cer": cer,
    })

    # Log overall metrics
    wandb.log({
        "eval/accuracy": report['accuracy'],
        "macro_precision": report['macro avg']['precision'],
        "macro_recall": report['macro avg']['recall'],
        "macro_f1": report['macro avg']['f1-score'],
        "macro_support": report['macro avg']['support'],
        "weighted_precision": report['weighted avg']['precision'],
        "weighted_recall": report['weighted avg']['recall'],
        "weighted_f1": report['weighted avg']['f1-score'],
        "weighted_support": report['weighted avg']['support'],
    })

    # Log per-class metrics
    for lang in [args.lang1, args.lang2]:
        if lang in report:
            wandb.log({
                f"{lang}_precision": report[lang]['precision'],
                f"{lang}_recall": report[lang]['recall'],
                f"{lang}_f1": report[lang]['f1-score'],
                f"{lang}_support": report[lang]['support'],
            })

    # Create a wandb.Table for the classification report
    data = []
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            data.append([label, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']])
        elif label == 'accuracy':
            data.append([label, None, None, metrics, None])  # Accuracy is a single value

    columns = ["label", "precision", "recall", "f1-score", "support"]
    table = wandb.Table(data=data, columns=columns)

    # Log the table as an artifact
    artifact = wandb.Artifact("classification_report", type="report")
    artifact.add(table, "classification_report_table")
    wandb.log_artifact(artifact)

    # Per-accent analysis
    if 'accent' in test_ds.features:
        true_accents = [x["lang"]+"-"+x["accent"] for x in test_ds]
        unique_accents = set(true_accents)
        
        per_accent_data = []
        
        for accent in sorted(unique_accents):
            mask = [a == accent for a in true_accents]
            
            filtered_true_lang = [t for t, m in zip(true_language_ids, mask) if m]
            filtered_pred_lang = [p for p, m in zip(pred_language_ids, mask) if m]
            filtered_true_text = [t for t, m in zip(true_texts, mask) if m]
            filtered_pred_text = [p for p, m in zip(pred_texts, mask) if m]
            
            if len(filtered_true_lang) == 0:
                continue
            
            acc_wer = 100.0 * wer_metric.compute(predictions=filtered_pred_text, references=filtered_true_text)
            acc_cer = 100.0 * cer_metric.compute(predictions=filtered_pred_text, references=filtered_true_text)
            acc_report = classification_report(filtered_true_lang, filtered_pred_lang, output_dict=True)
            
            wandb.log({
                f"eval/accent/{accent}/wer": acc_wer,
                f"eval/accent/{accent}/cer": acc_cer,
                f"eval/accent/{accent}/accuracy": acc_report['accuracy'],
                f"eval/accent/{accent}/macro_precision": acc_report['macro avg']['precision'],
                f"eval/accent/{accent}/macro_recall": acc_report['macro avg']['recall'],
                f"eval/accent/{accent}/macro_f1": acc_report['macro avg']['f1-score'],
                f"eval/accent/{accent}/macro_support": acc_report['macro avg']['support'],
            })
            
            # Per-class per-accent (optional, but for completeness)
            for lang in [args.lang1, args.lang2]:
                if lang in acc_report:
                    wandb.log({
                        f"eval/accent/{accent}/{lang}_precision": acc_report[lang]['precision'],
                        f"eval/accent/{accent}/{lang}_recall": acc_report[lang]['recall'],
                        f"eval/accent/{accent}/{lang}_f1": acc_report[lang]['f1-score'],
                        f"eval/accent/{accent}/{lang}_support": acc_report[lang]['support'],
                    })
            
            # Collect for table
            per_accent_data.append([accent, acc_wer, acc_cer, acc_report['accuracy'], acc_report['macro avg']['f1-score'], acc_report['macro avg']['support']])
        
        if per_accent_data:
            per_accent_columns = ["accent", "wer", "cer", "accuracy", "macro_f1", "support"]
            per_accent_table = wandb.Table(data=per_accent_data, columns=per_accent_columns)
            wandb.log({"per_accent_metrics": per_accent_table})

    # Optional: Print for local output
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print("\nLanguage Classification Report:")
    print(classification_report(true_language_ids, pred_language_ids))

    wandb.finish()

if __name__ == "__main__":
    main()