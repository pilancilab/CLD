'''
Tests detection head and finetuned whisper in full pipeline
select: model and detect_head state
'''

import os, time, torch, types
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, pipeline, WhisperProcessor
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

state = load_file("/home/ubuntu/arizonafiles/voice_clone/detection_head_lr3e-5_ep5/checkpoint-10008/model.safetensors")
classifier_state = {k.replace('classifier.', ''): v for k, v in state.items() if k.startswith('classifier.')}

#AUDIO_DIR = "/home/ubuntu/arizonafiles/voice_clone/audio_test/zh/"
#model_id = "/home/ubuntu/arizonafiles/voice_clone/bilingual_whisper_14x"
model_id = "/home/ubuntu/arizonafiles/voice_clone/bilingual_whisper_0528e9_14x"

# /home/ubuntu/arizonafiles/voice_clone/bilingual_whisper_full
# /home/ubuntu/arizonafiles/voice_clone/bilingual_whisper4
# /home/ubuntu/arizonafiles/voice_clone/bilingual_whisper_0526v1_4gpu

AUDIO_DIRS = [
    "/home/ubuntu/arizonafiles/voice_clone/audio_test/zh/",
    "/home/ubuntu/arizonafiles/voice_clone/audio_test/en/"]


# Load Whisper
whisper = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
whisper.config.forced_decoder_ids = None
whisper.generation_config.forced_decoder_ids = None

# 2) Build and load tiny detection head
class LangDetectHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2),
        )
    
    def forward(self, hidden_states):
        pooled = hidden_states.mean(dim=1)  # (B, D)
        return self.classifier(pooled)      # (B, 2)

d_model = whisper.config.d_model
head = LangDetectHead(d_model).to(torch.cuda.current_device(), dtype=torch.float16)
head.classifier.load_state_dict(classifier_state)
whisper.lang_detect_head = head

# Override the token retrieval instead
def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
    # Use custom detection head
    encoder_outputs = self.model.encoder(input_features, return_dict=True)
    hidden = encoder_outputs.last_hidden_state
    logits = self.lang_detect_head(hidden)
    class_id = logits.argmax(dim=-1).item()
    
    # CHECK THIS, 0 = en and 1 = zh
    lang_token = 50259 if class_id == 0 else 50260
    lang_name = "English" if class_id == 0 else "Chinese"
    print(f"Language Detection: {lang_name} (token: {lang_token})")
    
    # Return init tokens sequence: [start_token, lang_token, task_token]
    init_tokens = [50258, lang_token, 50359]  # start, language, transcribe
    
    # Convert to tensor shape: (batch_size, sequence_length)
    init_tokens_tensor = torch.tensor([init_tokens] * batch_size, 
                                    dtype=torch.long, 
                                    device=input_features.device)
    
    return init_tokens_tensor

# Attach the override head
whisper._retrieve_init_tokens = types.MethodType(_custom_retrieve_init_tokens, whisper)

whisper.save_pretrained(model_id)

processor = WhisperProcessor.from_pretrained(model_id)

# Instatiate pipeline
pipe = pipeline("automatic-speech-recognition", model=whisper, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device_map="auto")

# Inference Loop
for AUDIO_DIR in AUDIO_DIRS:
    print(f"\n=== Processing directory: {AUDIO_DIR} ===")
    for fname in os.listdir(AUDIO_DIR):
        if not fname.endswith(".wav"):
            continue
        path = os.path.join(AUDIO_DIR, fname)
        t0 = time.time()
        out = pipe(path)
        print(f"\n▶ {fname} → {out['text']}  (took {time.time()-t0:.2f}s)")


# # Inference Loop 
# for fname in os.listdir(AUDIO_DIR):
#     if not fname.endswith(".wav"):
#         continue
#     path = os.path.join(AUDIO_DIR, fname)
#     t0 = time.time()
#     out = pipe(path)
#     print(f"\n▶ {fname} → {out['text']}  (took {time.time()-t0:.2f}s)")




# # debug token mapping
# processor = WhisperProcessor.from_pretrained(model_id)
# tokenizer = processor.tokenizer

# print("Checking language tokens...")

# # Method 1: Check if tokenizer has language mapping
# if hasattr(tokenizer, 'lang_to_id'):
#     print("Language to ID mapping:")
#     for lang, token_id in tokenizer.lang_to_id.items():
#         print(f"{lang}: {token_id}")
# else:
#     print("No lang_to_id attribute found")

# # Method 2: Check generation config for language tokens
# if hasattr(whisper.generation_config, 'lang_to_id'):
#     print("\nGeneration config language mapping:")
#     for lang, token_id in whisper.generation_config.lang_to_id.items():
#         print(f"{lang}: {token_id}")

# # Method 3: Manually check common language tokens
# print(f"\nManual token checks:")
# try:
#     en_tokens = tokenizer.encode('<|en|>')
#     print(f"English token '<|en|>': {en_tokens}")
# except:
#     print("Could not encode '<|en|>'")

# try:
#     zh_tokens = tokenizer.encode('<|zh|>')
#     print(f"Chinese token '<|zh|>': {zh_tokens}")
# except:
#     print("Could not encode '<|zh|>'")

# # Method 4: Check vocab for language tokens
# print(f"\nVocab size: {len(tokenizer.get_vocab())}")
# vocab = tokenizer.get_vocab()

# # Look for language tokens (they typically start with <| and end with |>)
# lang_tokens = {token: id for token, id in vocab.items() if token.startswith('<|') and token.endswith('|>') and len(token) <= 6}
# print("Found language-like tokens:")
# for token, token_id in sorted(lang_tokens.items(), key=lambda x: x[1]):
#     print(f"{token}: {token_id}")

# exit()

#   "bos_token_id": 50257,
#   "decoder_start_token_id": 50258,
#   "eos_token_id": 50257,
#   "is_multilingual": true,
#   "lang_to_id": {
#     "<|af|>": 50327,
#     "<|am|>": 50334,
#     "<|ar|>": 50272,
#     "<|as|>": 50350,
#     "<|az|>": 50304,
#     "<|ba|>": 50355,
#     "<|be|>": 50330,
#     "<|bg|>": 50292,
#     "<|bn|>": 50302,
#     "<|bo|>": 50347,
#     "<|br|>": 50309,
#     "<|bs|>": 50315,
#     "<|ca|>": 50270,
#     "<|cs|>": 50283,
#     "<|cy|>": 50297,
#     "<|da|>": 50285,
#     "<|de|>": 50261,
#     "<|el|>": 50281,
#     "<|en|>": 50259,
#     "<|es|>": 50262,
#     "<|et|>": 50307,
#     "<|eu|>": 50310,
#     "<|fa|>": 50300,
#     "<|fi|>": 50277,
#     "<|fo|>": 50338,
#     "<|fr|>": 50265,
#     "<|gl|>": 50319,
#     "<|gu|>": 50333,
#     "<|haw|>": 50352,
#     "<|ha|>": 50354,
#     "<|he|>": 50279,
#     "<|hi|>": 50276,
#     "<|hr|>": 50291,
#     "<|ht|>": 50339,
#     "<|hu|>": 50286,
#     "<|hy|>": 50312,
#     "<|id|>": 50275,
#     "<|is|>": 50311,
#     "<|it|>": 50274,
#     "<|ja|>": 50266,
#     "<|jw|>": 50356,
#     "<|ka|>": 50329,
#     "<|kk|>": 50316,
#     "<|km|>": 50323,
#     "<|kn|>": 50306,
#     "<|ko|>": 50264,
#     "<|la|>": 50294,
#     "<|lb|>": 50345,
#     "<|ln|>": 50353,
#     "<|lo|>": 50336,
#     "<|lt|>": 50293,
#     "<|lv|>": 50301,
#     "<|mg|>": 50349,
#     "<|mi|>": 50295,
#     "<|mk|>": 50308,
#     "<|ml|>": 50296,
#     "<|mn|>": 50314,
#     "<|mr|>": 50320,
#     "<|ms|>": 50282,
#     "<|mt|>": 50343,
#     "<|my|>": 50346,
#     "<|ne|>": 50313,
#     "<|nl|>": 50271,
#     "<|nn|>": 50342,
#     "<|no|>": 50288,
#     "<|oc|>": 50328,
#     "<|pa|>": 50321,
#     "<|pl|>": 50269,
#     "<|ps|>": 50340,
#     "<|pt|>": 50267,
#     "<|ro|>": 50284,
#     "<|ru|>": 50263,
#     "<|sa|>": 50344,
#     "<|sd|>": 50332,
#     "<|si|>": 50322,
#     "<|sk|>": 50298,
#     "<|sl|>": 50305,
#     "<|sn|>": 50324,
#     "<|so|>": 50326,
#     "<|sq|>": 50317,
#     "<|sr|>": 50303,
#     "<|su|>": 50357,
#     "<|sv|>": 50273,
#     "<|sw|>": 50318,
#     "<|ta|>": 50287,
#     "<|te|>": 50299,
#     "<|tg|>": 50331,
#     "<|th|>": 50289,
#     "<|tk|>": 50341,
#     "<|tl|>": 50348,
#     "<|tr|>": 50268,
#     "<|tt|>": 50351,
#     "<|uk|>": 50280,
#     "<|ur|>": 50290,
#     "<|uz|>": 50337,
#     "<|vi|>": 50278,
#     "<|yi|>": 50335,
#     "<|yo|>": 50325,
#     "<|zh|>": 50260
