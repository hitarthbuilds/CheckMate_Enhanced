# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from ml_model_multilingual.model import TruthModel
from ml_model_multilingual.evidence import fetch_evidence

app = FastAPI(title="CheckMate: Fact-Checker + Propaganda Detection")

# Initialize model once
MODEL = TruthModel()

# Input schema
class ClaimInput(BaseModel):
    claim: str

@app.get("/")
async def root():
    return {"message": "CheckMate API running. POST to /predict with claim only."}

@app.post("/predict")
async def predict_endpoint(data: ClaimInput):
    try:
        # Auto-fetch evidence
        evidence = fetch_evidence(data.claim)

        # Fact-check
        fact_result = MODEL.predict(data.claim, evidence)

        # Propaganda check
        propaganda_result = MODEL.propaganda_check(data.claim)

        # Return combined result
        return {
            "success": True,
            "claim": data.claim,
            "evidence": evidence,
            "fact_check": fact_result,
            "propaganda_check": propaganda_result
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
