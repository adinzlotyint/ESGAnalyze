import json
import numpy as np
import torch
import os
from pathlib import Path
from transformers import LongformerForSequenceClassification, LongformerTokenizerFast

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")

class ESGInference:
    def __init__(self):
        self.model_path = Path(MODEL_DIR)
        self.criteria_names = [
            'c1_transition_plan',
            'c2_risk_management', 
            'c4_boundaries',
            'c6_historical_data',
            'c7_intensity_metrics',
            'c8_targets_credibility'
        ]
        self.model = LongformerForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = LongformerTokenizerFast.from_pretrained(self.model_path)
        with open(self.model_path / "thresholds.json", "r") as f:
          self.thresholds = np.array(json.load(f))
        
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
    
    def predict(self, text) -> dict:
        inputs = self.tokenizer(
            text,
            max_length=4096,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits.cpu().numpy().squeeze()
            probabilities = 1 / (1 + np.exp(-logits))
            predictions = (probabilities >= self.thresholds).astype(int)
        
        results = {}
        for i, criterion in enumerate(self.criteria_names):
            results[criterion] = {
                "prediction": int(predictions[i]),
                "probability": float(probabilities[i]),
                "threshold": float(self.thresholds[i])
            }
        
        return results