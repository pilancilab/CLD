import os
import glob
import numpy as np
import torch
from datasets import load_from_disk

import pytest

from solve.models.asr_model import Whisper
from solve.models.lang_detect_head import NNLangDetectHead, CVXNNLangDetectHead


PROJECT_ROOT = "/Users/williamtan/Projects/CLD"
DATASET_PATH = os.path.join(PROJECT_ROOT, "data/test_data/dataset")
NN_DIR = os.path.join(PROJECT_ROOT, "data/test_data/nn")
CVX_DIR = os.path.join(PROJECT_ROOT, "data/test_data/cvx")

LANG1 = "en"
LANG2 = "zh"
MODEL_NAME = "openai/whisper-small"


def _get_one_or_two_audio_arrays(ds_split):
    arrays = []
    for sample in ds_split:
        audio = sample["audio"]
        if isinstance(audio, dict) and audio.get("array") is not None:
            arrays.append(audio["array"])
        elif isinstance(audio, dict) and audio.get("path") is not None and os.path.exists(audio["path"]):
            import torchaudio
            waveform, _ = torchaudio.load(audio["path"])
            arrays.append(waveform.mean(0).numpy())
        if len(arrays) >= 2:
            break
    if not arrays:
        raise RuntimeError("No valid audio in dataset")
    return arrays


def _collect_audio_and_labels(ds_split, max_n=2):
    """Return (audio_batch, true_langs, true_texts) from the split."""
    audio_batch = []
    true_langs = []
    true_texts = []
    for sample in ds_split:
        audio = sample["audio"]
        if isinstance(audio, dict) and audio.get("array") is not None:
            audio_arr = audio["array"]
        elif isinstance(audio, dict) and audio.get("path") is not None and os.path.exists(audio["path"]):
            import torchaudio
            waveform, _ = torchaudio.load(audio["path"])
            audio_arr = waveform.mean(0).numpy()
        else:
            continue
        audio_batch.append(audio_arr)
        true_langs.append(sample.get("lang"))
        true_texts.append(sample.get("text") or sample.get("transcription") or "")
        if len(audio_batch) >= max_n:
            break
    if not audio_batch:
        raise RuntimeError("No valid audio in dataset")
    return audio_batch, true_langs, true_texts


def _display_truth_and_preds(true_langs, true_texts, pred_langs, pred_texts):
    for i, (t_lang, t_txt, p_lang, p_txt) in enumerate(zip(true_langs, true_texts, pred_langs, pred_texts)):
        print(f"[sample {i}] true_lang={t_lang} pred_lang={p_lang} true_text={t_txt} pred_text={p_txt}")


def test_load_data_real_dataset_encoder_features():
    w = Whisper(MODEL_NAME, config={"lang1": LANG1, "lang2": LANG2})
    # Use test split to keep it small
    A, y = w.load_data(DATASET_PATH, target_lang=LANG1, dataset_split="test", shuffle=False)
    assert isinstance(A, np.ndarray) and isinstance(y, np.ndarray)
    assert A.ndim == 2 and A.shape[1] > 0
    assert y.ndim == 1 and y.shape[0] == A.shape[0]


def test_default_head_vanilla_language_detection_on_one_sample():
    ds = load_from_disk(DATASET_PATH)
    test_ds = ds["test"]
    audio_batch, true_langs, true_texts = _collect_audio_and_labels(test_ds, max_n=1)
    audio = audio_batch[0]

    w = Whisper(MODEL_NAME, config={"lang1": LANG1, "lang2": LANG2})
    langs, texts = w.predict(audio)
    _display_truth_and_preds(true_langs, true_texts, langs, texts)
    assert isinstance(langs, list)
    assert len(langs) == 1
    # vanilla detection should return one of model languages; for our test expect en/hi
    assert isinstance(texts, list) and len(texts) == 1 and isinstance(texts[0], str)


def test_nn_head_predicts_from_hidden_states_real_audio():
    ds = load_from_disk(DATASET_PATH)
    test_ds = ds["test"]
    audio_batch, true_langs, true_texts = _collect_audio_and_labels(test_ds, max_n=2)

    w = Whisper(MODEL_NAME, config={"lang1": LANG1, "lang2": LANG2})
    
    # Load NN head
    nn_files = glob.glob(os.path.join(NN_DIR, "*.safetensors"))
    assert nn_files, f"No .safetensors found in {NN_DIR}"
    nn_head = NNLangDetectHead.load(nn_files[0], w)
    w.set_lang_detect_head(nn_head)
    
    langs, texts = w.predict(audio_batch)
    _display_truth_and_preds(true_langs, true_texts, langs, texts)
    assert isinstance(langs, list) and len(langs) == len(audio_batch)
    assert set(langs).issubset({LANG1, LANG2})
    assert isinstance(texts, list) and len(texts) == len(audio_batch)


def test_cvx_head_predicts_from_hidden_states_real_audio():
    ds = load_from_disk(DATASET_PATH)
    test_ds = ds["test"]
    audio_batch, true_langs, true_texts = _collect_audio_and_labels(test_ds, max_n=2)

    w = Whisper(MODEL_NAME, config={"lang1": LANG1, "lang2": LANG2})
    
    # Load CVX head
    cvx_files = glob.glob(os.path.join(CVX_DIR, "*.pkl"))
    assert cvx_files, f"No .pkl found in {CVX_DIR}"
    cvx_head = CVXNNLangDetectHead.load(cvx_files[0], asr_model=w)
    w.set_lang_detect_head(cvx_head)
    
    langs, texts = w.predict(audio_batch)
    _display_truth_and_preds(true_langs, true_texts, langs, texts)
    assert isinstance(langs, list) and len(langs) == len(audio_batch)
    assert set(langs).issubset({LANG1, LANG2})
    assert isinstance(texts, list) and len(texts) == len(audio_batch)


