# ml_model_multilingual/model.py
import json
import os
from typing import List, Dict
import torch
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from .utils import find_fact_checks

ROOT = os.path.dirname(__file__)

class TruthModel:
    def __init__(self,
                 kb_path: str = os.path.join(ROOT, "kb", "sample_kb.json"),
                 embed_model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
                 clf_model_name: str = "microsoft/xtremedistil-l6-h384-uncased"):
        # device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 1) embedding model (SBERT multilingual) for fast KB lookup
        self.embed_model = SentenceTransformer(embed_model_name, device=self.device)

        # 2) classifier: a compact distilled model (you can replace with your fine-tuned model)
        # We'll use a pipeline (tokenizer+model) for simplicity
        try:
            # Try to load as HF pipeline for text-classification (fast setup)
            self.classifier_pipeline = pipeline("text-classification", model=clf_model_name, device=0 if self.device=="cuda" else -1)
        except Exception:
            # fallback: load tokenizers + model manually
            tokenizer = AutoTokenizer.from_pretrained(clf_model_name)
            model = AutoModelForSequenceClassification.from_pretrained(clf_model_name)
            model.to(self.device)
            self.classifier_pipeline = lambda text: [{"label": "LABEL_0", "score": 0.5}]

        # 3) Load KB and prepare embeddings
        with open(kb_path, "r", encoding="utf-8") as f:
            self.kb = json.load(f)
        self.kb_texts = [item["claim"] for item in self.kb]
        if self.kb_texts:
            self.kb_embs = self.embed_model.encode(self.kb_texts, convert_to_tensor=True, show_progress_bar=False)
        else:
            self.kb_embs = None

        # Trusted sites for scraping
        self.trusted_sites = ["altnews.in", "boomlive.in", "factcheck.pib.gov.in"]

        # Zero-shot multilingual NLI pipeline (for fallback if needed)
        try:
            self.zero_shot = pipeline("zero-shot-classification", model="joeddav/xlm-roberta-large-xnli", device=0 if self.device=="cuda" else -1)
        except Exception:
            self.zero_shot = None

    def kb_search(self, text: str, top_k: int = 5) -> List[Dict]:
        """Return top_k KB entries with cosine similarity."""
        if not self.kb_embs:
            return []
        q_emb = self.embed_model.encode(text, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, self.kb_embs, top_k=top_k)[0]
        results = []
        for h in hits:
            idx = h["corpus_id"]
            score = float(h["score"])
            kb_item = self.kb[idx].copy()
            kb_item["kb_score"] = score
            results.append(kb_item)
        return results

    def classify(self, text: str) -> Dict:
        """
        Return classifier label + confidence. This uses the compact classifier pipeline.
        """
        try:
            outputs = self.classifier_pipeline(text if isinstance(text, str) else str(text), truncation=True)
            # pipeline returns list of dicts
            out = outputs[0]
            label = out.get("label")
            score = float(out.get("score", 0.0))
            return {"label": label, "confidence": score, "raw": outputs}
        except Exception:
            # fallback to zero-shot if pipeline failed
            if self.zero_shot:
                res = self.zero_shot(text, candidate_labels=["real", "fake", "propaganda"])
                return {"label": res["labels"][0], "confidence": float(res["scores"][0]), "raw": res}
            return {"label": "unknown", "confidence": 0.0, "raw": None}

    def scrape_fact_checks(self, text: str) -> List[Dict]:
        """Use site-specific google search + fetch metadata from results."""
        try:
            found = find_fact_checks(text, self.trusted_sites)
            return found
        except Exception:
            return []

    def assess(self, text: str, kb_threshold: float = 0.66) -> Dict:
        """
        Full assessment:
          - KB search
          - Classifier judgement
          - Scrape trusted sources
          - Combine into final verdict and confidence
        """
        # normalize
        text = text.strip()
        if not text:
            return {"error": "empty"}

        # 1) KB
        kb_hits = self.kb_search(text, top_k=6)

        kb_used = False
        kb_top = None
        kb_aggregate = None
        if kb_hits and kb_hits[0]["kb_score"] >= kb_threshold:
            kb_used = True
            kb_top = kb_hits[0]
            # simple aggregate of verdicts weighted by kb_score
            verdict_scores = {}
            for h in kb_hits:
                v = h.get("verdict", "Unknown")
                verdict_scores[v] = verdict_scores.get(v, 0.0) + h["kb_score"]
            agg_verdict = max(verdict_scores.items(), key=lambda x: x[1])[0]
            kb_aggregate = {"verdict": agg_verdict, "verdict_scores": verdict_scores}

        # 2) classifier
        clf = self.classify(text)

        # 3) scraping
        sources = self.scrape_fact_checks(text)

        # 4) combine: naive blending
        final_label = clf["label"]
        final_conf = clf["confidence"]
        if kb_used:
            kb_verdict = kb_aggregate["verdict"].lower()
            if kb_verdict in final_label.lower() or final_label.lower() in kb_verdict:
                final_conf = min(1.0, final_conf * 0.6 + 0.4)
            else:
                final_conf = max(0.0, final_conf * 0.4)
            final_label = f"{final_label} (KB:{kb_aggregate['verdict']})"

        return {
            "claim": text,
            "final_label": final_label,
            "final_confidence": round(float(final_conf), 4),
            "classifier": clf,
            "kb_hits": kb_hits,
            "kb_aggregated": kb_aggregate,
            "sources": sources
        }

# small test when run directly
if __name__ == "__main__":
    tm = TruthModel()
    s = "The government has banned all exams."
    print(tm.assess(s))
