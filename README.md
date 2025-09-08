# TruthLens Backend (Multilingual) - Hackathon Prototype

This repo contains a hackathon-ready multilingual backend for real-time misinformation detection.
It includes:
- SBERT-based KB search (fast)
- Compact classifier (distilled) for label + confidence
- Site-targeted search + scraping for trusted fact-check sources
- FastAPI endpoints for integration with Chrome extension or frontend

## Project structure
See the `truthlens-backend/` tree.

## Quick start (local)
```bash
# 1. create venv
python -m venv venv
source venv/bin/activate    # Windows: venv\\Scripts\\activate

# 2. install
pip install -r requirements.txt

# 3. run
uvicorn app.main:app --reload --port 8000
