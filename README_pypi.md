<table border="0" cellpadding="0" cellspacing="0">
<tr>
<td width="70%" valign="middle">

<h1>CLD</h1>

<b>Convex Low-resource Accent-Robust Language Detection in Speech Recognition</b><br>
A lightweight language-detection module for multilingual ASR, optimized via ADMM in JAX.

<p>
  <a href="https://icml.cc/virtual/2026/poster/64615"><img alt="paper" src="https://img.shields.io/badge/paper-ICML%202026-blue.svg"></a>
  <a href="https://huggingface.co/papers/2605.23235"><img alt="huggingface" src="https://img.shields.io/badge/🤗%20HF-models-yellow.svg"></a>
  <a href="https://pypi.org/project/jaxcld/"><img alt="pypi" src="https://img.shields.io/badge/pip-jaxcld-3775A9.svg?logo=pypi&logoColor=white"></a>
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue.svg">
  <img alt="jax" src="https://img.shields.io/badge/jax-0.4%2B-orange.svg">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green.svg">
</p>

</td>
<td width="30%" valign="middle" align="right">
<img src="https://raw.githubusercontent.com/pilancilab/CLD/main/assets/CLD_logo.png" alt="CLD" width="240">
</td>
</tr>
</table>

---

This repository provides the official implementation of **CLD**, a lightweight language-detection module for multilingual ASR. This codebase contains our pip-installable Python package (`jaxcld/`) including our training/benchmark scripts implemented in JAX and optimized via ADMM for high performance in low-resource settings. Simply, the package attaches a small language detection head (Convex NN / small NN / linear SVM) to ASR encoder representations, and use it to select the language token (Whisper) or adapter (MMS) before decoding.

## Highlights

- High Accuracy: Excels in binary and multiclass language detection.
- Low-Resource Robustness: Effective with limited data.
- Efficient: 13x training speedup from traditional NNs due to ADMM optimization and JAX.

## Requirements

The package is published on PyPI as [`jaxcld`](https://pypi.org/project/jaxcld/). JAX (CPU) is included by default. For GPU support, install the matching extra:

```bash
# CPU
pip install jaxcld

# GPU — CUDA 12 (most modern systems)
pip install "jaxcld[cuda12]"

# GPU — CUDA 11
pip install "jaxcld[cuda11]"
```

If you've cloned this repo, you can instead install from source:

- **Package-only install** (inference usage):

```bash
pip install -e .                  # CPU
pip install -e ".[cuda12]"        # GPU — CUDA 12
```

- **Full training/benchmark environment** (recommended if you run the scripts in this repo):

```bash
pip install -e ".[train]"         # CPU
pip install -e ".[train,cuda12]"  # GPU — CUDA 12
```

If you prefer installing from the pinned dependency list instead:

```bash
pip install -r requirements.txt
```

## Using the package

### Minimal inference example (Whisper)

```python
import numpy as np

from jaxcld import ASRModel, CVXNNLangDetectHead, NNLangDetectHead, SVMLangDetectHead

# 1) Load the base ASR model
languages = ["en", "hi", "id", "ms", "zh"]
asr = ASRModel.from_pretrained("openai/whisper-small", config={"languages": languages})

# 2) Load a language detection head artifact (choose ONE)
# head = CVXNNLangDetectHead.load("path/to/whisper-small_trained_cvx_mlp.pkl", asr)
# head = NNLangDetectHead.load("path/to/openai_whisper-small_nn_head.pkl", asr)
# head = SVMLangDetectHead.load("path/to/openai_whisper-small_linear_svm.pkl", asr)

# 3) Attach head and run inference
asr.set_lang_detect_head(head)

audio_16k_mono: np.ndarray = ...  # shape (T,), sampling rate 16kHz
pred_langs, pred_texts = asr.predict(audio_16k_mono)
print(pred_langs[0], pred_texts[0])
```

## Pre-trained models

Trained convex heads are published on the [Hugging Face Hub](https://huggingface.co/papers/2605.23235):

| Model | Backbone | Languages | Det. Acc | WER ↓ | CER ↓ | HF Hub |
| --- | --- | --- | --- | --- | --- | --- |
| `cld-whisper-small-5lang` | whisper-small | en/hi/id/ms/zh | 0.98 | 48.23 | 27.47 | [🤗](https://huggingface.co/williamhtan/cld-whisper-small-5lang) |
| `cld-whisper-large-v3-5lang` | whisper-large-v3 | en/hi/id/ms/zh | 0.98 | 31.11 | 19.81 | [🤗](https://huggingface.co/williamhtan/cld-whisper-large-v3-5lang) |
| `cld-mms-1b-5lang` | mms-1b-all | en/hi/id/ms/zh | 0.96 | 48.10 | 23.47 | [🤗](https://huggingface.co/williamhtan/cld-mms-1b-5lang) |
| `cld-whisper-small-enzh` (100–10000 samples/class) | whisper-small | en/zh | 0.99–1.00 | — | — | [🤗](https://huggingface.co/williamhtan/cld-whisper-small-enzh) |

### Loading from the Hub

```python
import numpy as np
from huggingface_hub import hf_hub_download
from jaxcld import ASRModel, CVXNNLangDetectHead

languages = ["en", "hi", "id", "ms", "zh"]

# 1) Load the frozen base ASR model
asr = ASRModel.from_pretrained("openai/whisper-small", config={"languages": languages})

# 2) Download the convex head from the Hub and load it
head_path = hf_hub_download("williamhtan/cld-whisper-small-5lang", "model.pkl")
head = CVXNNLangDetectHead.load(head_path, asr)

# 3) Attach and run
asr.set_lang_detect_head(head)
audio_16k_mono: np.ndarray = ...   # shape (T,), 16 kHz mono
pred_langs, pred_texts = asr.predict(audio_16k_mono)
print(pred_langs[0], pred_texts[0])
```

For the low-resource binary model, specify the samples-per-class subfolder:

```python
head_path = hf_hub_download("williamhtan/cld-whisper-small-enzh", "1000/model.pkl")
```

## Citation

If you use this code in your work, please cite the paper:

```bibtex
@inproceedings{feng2026cld,
  title     = {Convex Low-resource Accent-Robust Language Detection in Speech Recognition},
  author    = {Feng, Miria and Tan, William and Pilanci, Mert},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026},
  series    = {Proceedings of Machine Learning Research},
  publisher = {PMLR},
  url       = {https://icml.cc/virtual/2026/poster/64615}
}
```
