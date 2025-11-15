import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor

import jax.numpy as jnp
import numpy as np
from datasets import load_from_disk
import torch
import torchaudio
from typing import Tuple
from abc import ABC, abstractmethod

dtype = torch.float16



class ASRModel(ABC):
    def __init__(self, model_name, config):
        """Load model here"""
        self.model_name = model_name
        self.config = config
    
    @classmethod
    def from_pretrained(self, model_name, config={}):
        if model_name.startswith("openai/whisper"):
            return Whisper(model_name, config)
        elif model_name.startswith("nvidia/"):
            pass
        elif model_name.startswith("omniASR"):
            pass

    @abstractmethod
    def load_data(self, dataset_path: str, target_lang: str = 'en', caller_script: str = None, data_seed: int = 42, dataset_split: str = "train") -> Tuple[np.array, np.array]:
        pass
    
    def load_data_jax(self, dataset_path: str, target_lang: str = 'en', caller_script: str = None, data_seed: int = 42, dataset_split: str = "train") -> Tuple[jnp.ndarray, jnp.ndarray]:
        A, y = self.load_data(dataset_path, target_lang, caller_script, data_seed, dataset_split)
        A = jnp.array(A)  # (n, 768)
        y = jnp.array(y)    # (n,)
        return A, y

    @abstractmethod
    def set_lang_detect_head(lang_detect_head):
        pass

    @abstractmethod
    def predict(self, audio):
        """Runs transcription on the audio, returns list of language tokens and transcriptions"""
        pass

    @abstractmethod
    def get_dimensions(self):
        pass

    @abstractmethod
    def get_device(self):
        pass
    


def whisper_custom_retrieve_init_tokens_creator(asr_model, lang1, lang2):
    def _custom_retrieve_init_tokens(self, input_features, batch_size, generation_config=None, **kwargs):
        def lang_to_id(_, lang):
            return self.generation_config.lang_to_id[f"<|{lang}|>"]

        encoder_outputs = self.model.encoder(input_features, return_dict=True)
        hidden = encoder_outputs.last_hidden_state
        class_ids = asr_model.head.predict(hidden)
        
        # Assuming 0 = lang1, 1 = lang2
        lang_tokens = [lang_to_id(self, lang1) if class_id == 0 else lang_to_id(self, lang2) for class_id in class_ids]
        asr_model.lang_tokens.extend([lang1 if class_id == 0 else lang2 for class_id in class_ids])
        
        # Return init tokens: [start, lang, transcribe]
        init_tokens = [[50258, lang_token, 50359] for lang_token in lang_tokens]
        
        init_tokens_tensor = torch.tensor(init_tokens, 
                                          dtype=torch.long, 
                                          device=input_features.device)
        
        return init_tokens_tensor

    return _custom_retrieve_init_tokens


class Whisper(ASRModel):
    def __init__(self, model_name, config={}):
        super().__init__(model_name, config)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_name, device_map="auto", dtype=dtype)
        self.model.config.forced_decoder_ids = None
        self.processor = WhisperProcessor.from_pretrained(model_name)
        self.head = None # default head


    def load_data(self, dataset_path: str, target_lang: str = 'en', caller_script: str = None, data_seed: int = 42, dataset_split: str = "train", shuffle=True, negative_label=-1.0) -> Tuple[np.array, np.array]:
        """
        Load HF dataset, extract pooled model hidden states, return train/test splits.
        
        Args:
            dataset_path (str): Path to local HF dataset dir (splits: train, valid, test).
            target_lang (str): POS language code (e.g., 'en').
            caller_script (str): 'defrun' for 90% data (convex training); else full.
            data_seed (int): Seed for shuffle/split.
        
        Returns:
            Atr, ytr, Atst, ytst, ntr, ntst: JAX arrays for features/labels (pooled to 768 dim).
        """
        np.random.seed(data_seed)
        
        # Load train split (main data for training)
        dataset = load_from_disk(dataset_path)
        train_data = dataset[dataset_split]
        print(f"Loaded {len(train_data)} train samples")
        
        # Load Whisper encoder
        self.model.eval()
        
        def extract_pooled_hidden(audio) -> np.ndarray:
            """Extract and pool last hidden states to (768,)."""
            # Handle audio dict or path
            if isinstance(audio, dict):
                if audio.get('array') is not None:
                    audio_arr = audio['array']
                    sr = audio['sampling_rate']
                else:
                    audio_path = audio['path']
                    if not os.path.exists(audio_path):
                        return None
                    waveform, sr = torchaudio.load(audio_path)
                    audio_arr = waveform.mean(0).numpy()
            else:
                # Assume path if not dict
                if not os.path.exists(audio):
                    return None
                waveform, sr = torchaudio.load(audio)
                audio_arr = waveform.mean(0).numpy()
            
            # Resample to 16kHz
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(sr, 16000)
                audio_arr = resampler(torch.tensor(audio_arr)).numpy()
            
            # Process to input_features
            inputs = self.processor(audio_arr, sampling_rate=16000, return_tensors='pt').to(self.get_device(), dtype=dtype)
            
            # Encoder last hidden
            with torch.no_grad():
                encoder_outputs = self.model.model.encoder(inputs.input_features, output_hidden_states=True)
                hidden = encoder_outputs.last_hidden_state.squeeze(0)  # (seq_len, 768)
            
            # Pool: Mean over seq_len
            pooled = hidden.mean(0).cpu().numpy()  # (768,)
            return pooled
        
        # Extract features and labels for all train samples
        features = []
        labels = []
        valid_count = 0
        for sample in train_data:
            hidden = extract_pooled_hidden(sample['audio'])
            if hidden is None:
                continue  # Skip invalid audio
            
            label = 1.0 if sample['lang'] == target_lang else negative_label
            features.append(hidden)
            labels.append(label)
            valid_count += 1
        
        if valid_count == 0:
            raise ValueError("No valid audio samples found")
        
        print(f"Extracted {valid_count} valid samples: {np.sum(np.array(labels) == 1)} POS, {np.sum(np.array(labels) == negative_label)} NEG")
        
        # Convert to arrays
        A = np.array(features)
        y = np.array(labels)
        
        # Shuffle
        if shuffle:
            perm = np.random.permutation(A.shape[0])
            A = A[perm]
            y = y[perm]

        return A, y

    def set_lang_detect_head(self, lang_detect_head):
        self.head = lang_detect_head
        print("sdlfks")
        if self.head:
            print(self.head)
            self.model._retrieve_init_tokens = types.MethodType(whisper_custom_retrieve_init_tokens_creator(self, self.config.get("lang1"), self.config.get("lang2")), self.model)
        
    def _detect_language_vanilla(self, input_features):
        # 50258 is the token for transcribing
        batch_size = input_features.shape[0]
        device = input_features.device
        decoder_input_ids = torch.full((batch_size, 1), 50258, dtype=torch.long, device=device)
        model_output = self.model(input_features, decoder_input_ids=decoder_input_ids)
        logits = model_output.logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        
        # Language tokens in Whisper multilingual models are IDs 50263 to 50361 (99 languages)
        # Compute probabilities and detect the most likely language per batch item
        language_probs = torch.softmax(logits, dim=-1)
        language_indices = torch.argmax(language_probs, dim=-1)  # Shape: (batch_size,)
        
        # Map indices to language codes (sorted list of Whisper's 99 supported languages)
        detected_languages = [self.id_to_lang(x.item()) for x in language_indices]
        
        # Return list of detected languages (one per batch item); also return probs if needed
        return detected_languages  # e.g., ['en'] for batch_size=1
    
    def predict(self, audio):
        input_features = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        input_features = input_features.to(self.get_device(), dtype=dtype)

        self.lang_tokens = []
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if(self.head is None or getattr(self.head, "SKIP", False)):
            self.lang_tokens = self._detect_language_vanilla(input_features)
        return self.lang_tokens, transcription
    
    def get_dimensions(self):
        return self.model.config.d_model

    def get_device(self):
        return next(self.model.model.encoder.layers[-1].parameters()).device

    def lang_to_id(self, lang):
        lang_code = f"<|{lang}|>"
        return self.model.generation_config.lang_to_id[lang_code]

    def id_to_lang(self, tid):
        id_to_lang_mapping =  dict(zip(self.model.generation_config.lang_to_id.values(), self.model.generation_config.lang_to_id.keys()))
        return id_to_lang_mapping.get(tid, "    ")[2:-2]

