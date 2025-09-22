import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

dtype = torch.float16

def load_whisper(whisper_path):
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_path, device_map="auto", dtype=dtype)
    whisper.config.forced_decoder_ids = None
    whisper.generation_config.forced_decoder_ids = None
    # No .to() - device_map handles it
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

def custom_retrieve_init_tokens_creator(processor, lang1, lang2, cld_type):

    def head_caller(self, hidden):
        if cld_type == "nn":
            logits = self.lang_detect_head(hidden)
            return logits.argmax(dim=-1).tolist()
        elif cld_type == "cvx":
            pooled = hidden.mean(dim=1).cpu().numpy()  # Move to CPU and numpy for predict
            logits = self.lang_detect_head.predict(pooled, self.lang_detect_head.theta1, self.lang_detect_head.theta2)
            return [0 if x < 0 else 1 for x in logits]

    def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
        encoder_outputs = self.model.encoder(input_features, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        class_ids = head_caller(self, hidden)
        
        # Assuming 0 = lang1, 1 = lang2
        lang_tokens = [lang_to_id(self, lang1) if class_id == 0 else lang_to_id(self, lang2) for class_id in class_ids]
        
        # Return init tokens: [start, lang, transcribe]
        init_tokens = [[50258, lang_token, 50359] for lang_token in lang_tokens]
        
        init_tokens_tensor = torch.tensor(init_tokens, 
                                          dtype=torch.long, 
                                          device=input_features.device)
        
        return init_tokens_tensor

    return _custom_retrieve_init_tokens

def get_nn_pipeline(whisper_path, cld_path, cld_type, lang1, lang2):
    processor = WhisperProcessor.from_pretrained(whisper_path)
    whisper = load_whisper(whisper_path)
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

    return whisper

def inference(model, processor, audio):
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    
    # Place on model's entry device
    first_device = next(model.parameters()).device
    input_features = input_features.to(first_device, dtype=dtype)
    
    predicted_ids = model.generate(input_features)
    language_tokens = [processor.decode([pred[1]], skip_special_tokens=False)[2:-2] for pred in predicted_ids]
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return language_tokens, transcription