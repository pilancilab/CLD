import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

dtype = torch.float16

lang_tokens_glob = []
time_taken_glob = []

def load_whisper(whisper_path):
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_path, device_map="auto", dtype=dtype)
    whisper.config.forced_decoder_ids = None
    # whisper.generation_config.forced_decoder_ids = None
    # whisper.generation_config.suppress_tokens = None  # or None
    # whisper.generation_config.begin_suppress_tokens = None  # or None
    return whisper

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

def lang_to_id(whisper, lang):
    lang_code = f"<|{lang}|>"
    return whisper.generation_config.lang_to_id[lang_code]

def id_to_lang(whisper, tid):
    id_to_lang_mapping =  dict(zip(whisper.generation_config.lang_to_id.values(), whisper.generation_config.lang_to_id.keys()))
    return id_to_lang_mapping.get(tid, "    ")[2:-2]

def custom_retrieve_init_tokens_creator(processor, lang1, lang2, cld_type):
    

    def head_caller(self, hidden):
        if cld_type == "nn":
            logits = self.lang_detect_head(hidden)
            return logits.argmax(dim=-1).tolist()
        elif cld_type == "cvx":
            pooled = hidden.mean(dim=1).cpu().detach().numpy()  # Move to CPU and numpy for predict
            logits = self.lang_detect_head.predict(pooled, self.lang_detect_head.theta1, self.lang_detect_head.theta2)
            return [0 if x > 0 else 1 for x in logits]

    def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
        global lang_tokens_glob, time_taken_glob
        encoder_outputs = self.model.encoder(input_features, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        class_ids = []
        # Measure per-sample language detection time
        for i in range(hidden.shape[0]):
            start_time = time.perf_counter()
            ids = head_caller(self, hidden[i:i+1])
            end_time = time.perf_counter()
            time_taken_glob.append(end_time - start_time)
            class_ids.append(ids[0])

        # Assuming 0 = lang1, 1 = lang2
        lang_tokens = [lang_to_id(self, lang1) if class_id == 0 else lang_to_id(self, lang2) for class_id in class_ids]
        lang_tokens_glob.extend([lang1 if class_id == 0 else lang2 for class_id in class_ids])
        
        # Return init tokens: [start, lang, transcribe]
        init_tokens = [[50258, lang_token, 50359] for lang_token in lang_tokens]
        
        init_tokens_tensor = torch.tensor(init_tokens, 
                                          dtype=torch.long, 
                                          device=input_features.device)
        
        return init_tokens_tensor

    return _custom_retrieve_init_tokens

def detect_language_vanilla(model, input_features):
    # 50258 is the token for transcribing
    batch_size = input_features.shape[0]
    device = input_features.device
    decoder_input_ids = torch.full((batch_size, 1), 50258, dtype=torch.long, device=device)
    model_output = model(input_features, decoder_input_ids=decoder_input_ids)
    logits = model_output.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
    
    # Language tokens in Whisper multilingual models are IDs 50263 to 50361 (99 languages)
    # Compute probabilities and detect the most likely language per batch item
    language_probs = torch.softmax(logits, dim=-1)
    language_indices = torch.argmax(language_probs, dim=-1)  # Shape: (batch_size,)
    
    # Map indices to language codes (sorted list of Whisper's 99 supported languages)
    detected_languages = [id_to_lang(model, x.item()) for x in language_indices]
    
    # Return list of detected languages (one per batch item); also return probs if needed
    return detected_languages  # e.g., ['en'] for batch_size=1


def get_nn_pipeline(whisper_path, cld_path, cld_type, lang1, lang2):
    try:
        processor = WhisperProcessor.from_pretrained(whisper_path)
    except Exception:
        processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper = load_whisper(whisper_path)
    whisper.cld_type = cld_type
    d_model = whisper.config.d_model

    change_head = True
    if cld_type == "nn":
        state = load_file(cld_path)
        classifier_state = {k.replace('classifier.', ''): v for k, v in state.items() if k.startswith('classifier.')}
        
        head = LangDetectHead(d_model)
        head.classifier.load_state_dict(classifier_state)
        
        # Move head to match encoder's last layer device
        encoder_last_device = next(whisper.model.encoder.layers[-1].parameters()).device
        head = head.to(encoder_last_device, dtype=dtype)
        
    elif cld_type == "cvx":
        with open(cld_path, 'rb') as f:
            head = pickle.load(f)
    else:
        # vanilla
        change_head = False

    if change_head:
        whisper.lang_detect_head = head
        whisper._retrieve_init_tokens = types.MethodType(custom_retrieve_init_tokens_creator(processor, lang1, lang2, cld_type), whisper)

    return whisper, processor

def inference(model, processor, audio):
    global lang_tokens_glob, time_taken_glob
    lang_tokens_glob = []
    time_taken_glob = []
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Place on model's entry device
    first_device = next(model.parameters()).device
    input_features = input_features.to(first_device, dtype=dtype)
    
    predicted_ids = model.generate(input_features)
    # For vanilla, measure per-sample detection separately to collect timings per item
    if model.cld_type == "vanilla":
        lang_tokens = []
        for i in range(input_features.shape[0]):
            start_time = time.perf_counter()
            langs = detect_language_vanilla(model, input_features[i:i+1])
            end_time = time.perf_counter()
            time_taken_glob.append(end_time - start_time)
            lang_tokens.append(langs[0])
        lang_tokens_glob = lang_tokens
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return lang_tokens_glob, transcription, time_taken_glob