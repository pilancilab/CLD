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
- [ ] Package
- [ ] Documentation

## Tests

- LangDetectHead
  - Test loading for nn, cvxnn
  - Test prediction from WhisperModel
- ASR (repeat for all ASR types: WhisperModel, MMSModel)
  - Init model
  - Load data
  - Predict (all lang detect heads: vanilla, nn, cvxnn)

Languages: en, hi, id, ms, zh
