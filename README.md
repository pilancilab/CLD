## TODO
- [x] Planning
- [x] Get dry data
- [x] Tests
- [x] Diff multi class
- [x] ASRModel
- [x] Train new Linear SVC
  - [x] Train script
  - [x] LangDetectHead
- [x] Evaluate
- [ ] Add & migrate other files
  - [x] train_nn.py
  - [ ] train_whisper.py
- [x] Package (`pyproject.toml`)
- [x] Documentation (basic install + usage)

## Install

Editable install (recommended for local dev):

```bash
pip install -e .
```

If you're running the training / benchmarking scripts in this repo:

```bash
pip install -e ".[train]"
```

## Usage

```python
from cld import ASRModel, CVXNNLangDetectHead

asr_model = ASRModel.from_pretrained(
    "openai/whisper-small",
    config={"languages": []},  # provide the language list your head was trained on
)
lang_detect_head = CVXNNLangDetectHead.load("whisper-small_trained_cvx_mlp.pkl", asr_model)
asr_model.set_lang_detect_head(lang_detect_head)

language_id, text = asr_model.predict(audio)
```

## Tests

- LangDetectHead
  - Test loading for nn, cvxnn
  - Test prediction from WhisperModel
- ASR (repeat for all ASR types: WhisperModel, MMSModel)
  - Init model
  - Load data
  - Predict (all lang detect heads: vanilla, nn, cvxnn)

Languages: en,hi,id,ms,zh
