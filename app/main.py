# app/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from ml_model_multilingual.model import TruthModel
from typing import List, Dict
import uvicorn
import os
import datetime

app = FastAPI(title="TruthLens Backend (Multilingual)")

MODEL = TruthModel()  # instantiate model once at server start

# In-memory history (simple for demo; swap with DB for production)
HISTORY: List[Dict] = []

class PredictRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    claim: str
    correct_label: str
    notes: str = None

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.datetime.utcnow().isoformat()}

@app.post("/predict")
def predict(req: PredictRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    res = MODEL.assess(text)
    record = {
        "claim": text,
        "final_label": res.get("final_label"),
        "final_confidence": res.get("final_confidence"),
        "ts": datetime.datetime.utcnow().isoformat()
    }
    HISTORY.append(record)
    # keep history size bounded for demo
    if len(HISTORY) > 1000:
        HISTORY.pop(0)
    return res

@app.get("/history")
def get_history(limit: int = 50):
    return {"count": len(HISTORY), "history": HISTORY[-limit:]}

@app.post("/feedback")
def feedback(req: FeedbackRequest):
    # For hackathon demo, append feedback to history store
    entry = {
        "claim": req.claim,
        "correct_label": req.correct_label,
        "notes": req.notes,
        "ts": datetime.datetime.utcnow().isoformat()
    }
    HISTORY.append({"feedback": entry})
    return {"status": "ok", "saved": entry}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=False)
