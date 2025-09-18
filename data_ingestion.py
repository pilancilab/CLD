import os
import sys
import csv
import uuid
import shutil
import random
from dataclasses import dataclass, field
from typing import Dict, List, Iterable, Optional, Tuple

import soundfile as sf

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None


# -----------------------
# Config structures
# -----------------------

@dataclass
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1


@dataclass
class IngestionParams:
    balanced_count: Optional[int] = None
    split: SplitRatios = field(default_factory=SplitRatios)
    target_sr: int = 22050


# -----------------------
# Utility helpers
# -----------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_wav_bytes_to_path(audio_array, sampling_rate: int, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    sf.write(out_path, audio_array, sampling_rate)


def save_rows_to_csv(rows: List[Dict[str, str]], csv_path: str) -> None:
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_file", "text", "lang", "accent"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# -----------------------
# Dataset loaders
# -----------------------

def load_common_voice(lang_code: str, split: str = "train", streaming: bool = False):
    if load_dataset is None:
        raise RuntimeError("datasets library not available. pip install datasets")
    ds = load_dataset("mozilla-foundation/common_voice_17_0", lang_code, split=split, streaming=streaming) #TODO: CHANGE TO 17.0
    return ds


def iter_common_voice_samples(ds, accent_name, accent_code) -> Iterable[Dict]:
    for ex in ds:
        # Each example has ex["audio"] dict with array and sampling_rate when accessed
        try:
            audio = ex["audio"]
            # Accessing triggers decode/resample to dataset.features["audio"].sampling_rate
            array = audio["array"]
            sr = audio["sampling_rate"]
            text = ex.get("sentence", "")
            lang = ex.get("locale", "")
            accent = ex.get("accent", "")
            if(accent != accent_name):
                continue

            yield {
                "array": array,
                "sr": sr,
                "text": text,
                "lang": lang,
                "accent": accent_code,
            }
        except Exception:
            continue


def load_lahaja(split: str = "test"):
    if load_dataset is None:
        raise RuntimeError("datasets library not available. pip install datasets")
    ds = load_dataset("ai4bharat/Lahaja", split=split)
    return ds


def iter_lahaja_samples(ds, accent_name, accent_code) -> Iterable[Dict]:
    for ex in ds:
        try:
            audio_data = ex.get("audio_filepath", None)
            if audio_data is not None and isinstance(audio_data, dict):
                array = audio_data.get("array")
                sr = audio_data.get("sampling_rate")
                if array is None or sr is None:
                    continue
            else:
                # Some variants expose raw bytes in a column (e.g., 'bytes' or 'audio_filepath').
                # Prefer decoded column if available; otherwise skip.
                continue
            text = ex.get("text", "")
            lang = ex.get("lang", "")
            native_language = ex.get("native_language", "")
            
            if(native_language != accent_name):
                continue
                
            yield {
                "array": array,
                "sr": sr,
                "text": text,
                "lang": lang,
                "accent": accent_code,
            }
        except Exception:
            continue


# -----------------------
# Cleaning pipeline hooks (reuse existing script)
# -----------------------

from process_input_three_ways import (
    normalize_audio,
    process_audio_file as noise_reduce_file,
    resample_audio,
    TARGET_SAMPLE_RATE,
)


def run_cleaning_pipeline(wav_path: str, work_dir: str) -> str:
    # Apply normalize -> noise reduction -> resample in place into work_dir
    ensure_dir(work_dir)
    base_name = os.path.basename(wav_path)
    temp_path = os.path.join(work_dir, base_name)
    # Start by copying the original
    if os.path.abspath(wav_path) != os.path.abspath(temp_path):
        shutil.copyfile(wav_path, temp_path)
    # Normalize
    normalize_audio(temp_path, work_dir)
    # Noise reduction (reads from and writes to work_dir)
    noise_reduce_file(os.path.join(work_dir, base_name), work_dir)
    # Resample
    resample_audio(os.path.join(work_dir, base_name), work_dir, target_sample_rate=TARGET_SAMPLE_RATE)
    return os.path.join(work_dir, base_name)


# -----------------------
# Core ingestion
# -----------------------


def split_rows(rows: List[Dict[str, str]], ratios: SplitRatios) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    random.shuffle(rows)
    n = len(rows)
    n_train = int(n * ratios.train)
    n_val = int(n * ratios.val)
    train = rows[:n_train]
    val = rows[n_train:n_train + n_val]
    test = rows[n_train + n_val:]
    return train, val, test


def ingest(config: Dict, out_dir: str) -> None:
    ensure_dir(out_dir)
    audio_dir = os.path.join(out_dir, "audio")
    ensure_dir(audio_dir)
    tmp_clean_dir = os.path.join(out_dir, ".clean")
    ensure_dir(tmp_clean_dir)

    params = config.get("params", {})
    balanced_count = params.get("balanced_count")
    split_cfg = params.get("split", None)
    split = SplitRatios(**split_cfg) if isinstance(split_cfg, dict) else SplitRatios()

    rows: List[Dict[str, str]] = []

    languages = config.get("langauges", {}) or config.get("languages", {})
    for lang_key, lang_spec in languages.items():
        accents = lang_spec.get("accents", [])

        for accent_code, meta in accents.items():
            dataset_name = meta.get("dataset")
            accent_name = meta.get("name", accent_code)
            if dataset_name is None:
                continue

            if dataset_name == "common_voice":
                ds = load_common_voice(lang_key, split="train", streaming=False)
                iterable = iter_common_voice_samples(ds, accent_name, accent_code)
            elif dataset_name in ("lahaja"):
                ds = load_lahaja(split="test")
                iterable = iter_lahaja_samples(ds, accent_name, accent_code)
            else:
                # Unknown dataset; skip for now
                continue

            taken = list(iterable)[:balanced_count]
            for sample in taken:
                uid = str(uuid.uuid4())[:8]
                file_name = f"{lang_key}_{accent_code}_{uid}.wav"
                raw_path = os.path.join(audio_dir, file_name)

                # Save original
                write_wav_bytes_to_path(sample["array"], sample["sr"], raw_path)

                # Clean in place to match your process_input_three_ways
                cleaned_path = run_cleaning_pipeline(raw_path, tmp_clean_dir)

                # Move cleaned back into audio_dir (overwrite original)
                shutil.copyfile(cleaned_path, raw_path)

                rows.append({
                    "audio_file": os.path.relpath(raw_path, out_dir),
                    "text": sample.get("text", ""),
                    "lang": sample.get("lang", lang_key),
                    "accent": sample.get("accent", accent_code),
                })

    # Optional augmentation stub (placeholder)
    # TODO: integrate MUSAN/pyannote pipeline here.

    train_rows, val_rows, test_rows = split_rows(rows, split)

    save_rows_to_csv(train_rows, os.path.join(out_dir, "train.csv"))
    save_rows_to_csv(val_rows, os.path.join(out_dir, "val.csv"))
    save_rows_to_csv(test_rows, os.path.join(out_dir, "test.csv"))

    # Clean up temp dir
    try:
        shutil.rmtree(tmp_clean_dir)
    except Exception:
        pass


def parse_json_config(cfg_path: str) -> Dict:
    import json
    with open(cfg_path, "r") as f:
        data = json.load(f)
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Data ingestion pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--out", type=str, required=True, help="Output directory")
    args = parser.parse_args()

    cfg = parse_json_config(args.config)
    ingest(cfg, args.out)


if __name__ == "__main__":
    main()


