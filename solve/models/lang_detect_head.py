import os, time, torch, types, pickle
import torch.nn as nn
from safetensors.torch import load_file
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from abc import ABC, abstractmethod

dtype = torch.float16

class LangDetectHead(ABC):
    def __init__(self, head):
        self.head = head

    @abstractmethod
    def load(filepath, asr_model):
        """Load detection head from filepath."""
        pass

    @abstractmethod
    def predict(self, hidden):
        """Runs classification on the hidden layer"""
        pass

class NNLangDetectHeadModule(nn.Module):
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
    

class NNLangDetectHead(LangDetectHead):
    @staticmethod
    def load(filepath, asr_model):
        state = load_file(filepath)
        classifier_state = {k.replace('classifier.', ''): v for k, v in state.items() if k.startswith('classifier.')}
        
        head = NNLangDetectHeadModule(asr_model.get_dimensions())
        head.classifier.load_state_dict(classifier_state)
        
        # Move head to match encoder's last layer device
        head = head.to(asr_model.get_device(), dtype=dtype)
        return NNLangDetectHead(head)
    
    def predict(self, hidden):
        logits = self.head(hidden)
        return logits.argmax(dim=-1).tolist()
    
class CVXNNLangDetectHead(LangDetectHead):
    @staticmethod
    def load(filepath, asr_model):
        with open(filepath, 'rb') as f:
            head = pickle.load(f)
        
        return CVXNNLangDetectHead(head)
    
    def predict(self, hidden):
        pooled = hidden.mean(dim=1).cpu().detach().numpy()  # Move to CPU and numpy for predict
        logits = self.head.predict(pooled, self.head.theta1, self.head.theta2)
        return [0 if x > 0 else 1 for x in logits]
        

