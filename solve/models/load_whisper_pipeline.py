
import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, pipeline, WhisperProcessor

def load_whisper(whisper_path):
    whisper = WhisperForConditionalGeneration.from_pretrained(whisper_path, device_map="auto", torch_dtype=torch.float16)
    whisper.config.forced_decoder_ids = None
    whisper.generation_config.forced_decoder_ids = None

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

def custom_retrieve_init_tokens_creator(processor, lang1, lang2, cld_type):

    def head_caller(self, hidden):
        if cld_type == "nn":
            logits = self.lang_detect_head(hidden)
            return logits.argmax(dim=-1).item()
        elif cld_type == "cvx":
            logits = self.lang_detect_head.predict(hidden, self.lang_detect_head.theta1, self.lang_detect_head.theta2)
            return [0 if x < 0 else 1 for x in logits]

    # Override the token retrieval instead
    def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
        # Use custom detection head
        encoder_outputs = self.model.encoder(input_features, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        class_ids = head_caller(hidden)
        
        # CHECK THIS, 0 = en and 1 = zh
        lang_tokens = [processor.tokenizer.lang_to_id[lang1] if class_id == 0 else processor.tokenizer.lang_to_id[lang2] for class_id in class_ids]
        # lang_name = "English" if class_id == 0 else "Chinese"
        # print(f"Language Detection: {lang_name} (token: {lang_token})")
        
        # Return init tokens sequence: [start_token, lang_token, task_token]
        init_tokens = [[50258, lang_token, 50359] for lang_token in lang_tokens]  # start, language, transcribe
        
        # Convert to tensor shape: (batch_size, sequence_length)
        init_tokens_tensor = torch.tensor(init_tokens, 
                                        dtype=torch.long, 
                                        device=input_features.device)
        
        return init_tokens_tensor

    return _custom_retrieve_init_tokens

def get_nn_pipeline(whisper_path, cld_path, cld_type, lang1, lang2):
    whisper = load_whisper(whisper_path)
    d_model = whisper.config.d_model
    if cld_path == "nn":
        head = LangDetectHead(d_model).to(torch.cuda.current_device(), dtype=torch.float16)
        head.classifier.load_state_dict(cld_path)
    elif cld_path == "cvx":
        with open(cld_path, 'rb') as f:
            head = pickle.load(f)

    whisper.lang_detect_head = head

    processor = WhisperProcessor.from_pretrained(whisper_path)

    # Attach the override head
    whisper._retrieve_init_tokens = types.MethodType(custom_retrieve_init_tokens_creator(processor, lang1, lang2, cld_type), whisper)



    # Instatiate pipeline
    # pipe = pipeline("automatic-speech-recognition", model=whisper, tokenizer=processor.tokenizer, feature_extractor=processor.feature_extractor, device_map="auto")

    return whisper

def inference(model, processor, audio):
    input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt").input_features 
    predicted_ids = model.generate(input_features)
    language_token_id = predicted_ids[1]
    language_token = processor.decode([language_token_id], skip_special_tokens=False)[2:-2]
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return language_token, transcription
    





