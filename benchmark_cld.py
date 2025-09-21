import argparse
from datasets import load_from_disk
from solve.models.load_whisper_pipeline import get_nn_pipeline, inference
from sklearn.metrics import classification_report
from transformers import WhisperProcessor
import evaluate

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Whisper model on a dataset for language detection and transcription.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the dataset directory (e.g., 'data/en_hi/').")
    parser.add_argument("--whisper_path", type=str, default="models/whisper-small-enhi-out", help="Path to the Whisper model directory.")
    parser.add_argument("--cld_path", type=str, default="models/en_hi_nn", help="Path to the language classifier model directory.")
    parser.add_argument("--cld_type", type=str, default="models/en_hi_nn", choices=["nn", "cvx"], help="Detection head architecture.")
    parser.add_argument("--lang1", type=str, default="en", help="First language code (e.g., 'en' for English).")
    parser.add_argument("--lang2", type=str, default="hi", help="Second language code (e.g., 'hi' for Hindi).")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    return parser.parse_args()

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def main():
    # Parse command-line arguments
    args = parse_args()

    # Load evaluation metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Load dataset
    ds = load_from_disk(args.dataset_path)

    # Load model and processor
    model = get_nn_pipeline(
        whisper_path=args.whisper_path,
        cld_path=args.cld_path,
        lang1=args.lang1,
        lang2=args.lang2
    )
    processor = WhisperProcessor.from_pretrained(args.whisper_path)

    # Get true labels and texts
    true_language_ids = [x["lang"] for x in ds["train"]]
    true_texts = [x["text"] for x in ds["train"]]

    # Initialize lists for predictions
    pred_language_ids = []
    pred_texts = []

    # Perform batched inference
    for ba in batch(ds["train"], args.batch_size):
        audios = [b["audio"] for b in ba]
        language_ids_batch, texts_batch = inference(model, processor, audios)

        pred_language_ids.extend(language_ids_batch)
        pred_texts.extend(texts_batch)  # Fixed: was incorrectly extending pred_texts

    # Compute metrics
    wer = 100.0 * wer_metric.compute(predictions=pred_texts, references=true_texts)
    cer = 100.0 * cer_metric.compute(predictions=pred_texts, references=true_texts)

    # Print results
    print(f"WER: {wer:.2f}%")
    print(f"CER: {cer:.2f}%")
    print("\nLanguage Classification Report:")
    print(classification_report(true_language_ids, pred_language_ids))

if __name__ == "__main__":
    main()