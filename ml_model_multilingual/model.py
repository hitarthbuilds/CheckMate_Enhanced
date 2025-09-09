# ml_model_multilingual/model.py

from transformers import XLMRobertaForSequenceClassification, XLMRobertaTokenizer
import torch

class TruthModel:
    def __init__(self, model_name="joeddav/xlm-roberta-large-xnli", device=None):
        """
        Multilingual fact-checking model with propaganda check.
        """
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device

        # Load tokenizer and model
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Labels for NLI
        self.labels = ["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

    def predict(self, claim: str, evidence: str):
        """
        Fact-check prediction between claim and evidence.
        """
        inputs = self.tokenizer(claim, evidence, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            score, pred_idx = torch.max(probs, dim=1)
            label = self.labels[pred_idx.item()]

        return {"label": label, "score": score.item()}

    def propaganda_check(self, text: str):
        """
        Simple heuristic propaganda detection. Replace with real model if available.
        """
        propaganda_words = ["always", "never", "fake news", "conspiracy", "enemy"]
        score = sum(word in text.lower() for word in propaganda_words) / max(len(text.split()), 1)
        is_propaganda = score > 0.1  # threshold
        return {"is_propaganda": is_propaganda, "score": score}
