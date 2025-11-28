# AI4Life — Combined Interview Assessment API

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE) ![Python](https://img.shields.io/badge/python-3.8%2B-blue)

AI4Life provides a lightweight API that evaluates interview transcripts using an ensemble of local BERT models and optional Gemini LLM scoring. The service produces per-trait scores (1–5) and a combined recommendation to help staffing teams assess candidates quickly.

Download model from [text](https://www.kaggle.com/code/dra00n/traits-week-7) and put it into the models folder

Quick links
- Main service: [app.py](app.py)
- Tests / example client: [test_api.py](test_api.py)
- Requirements: [requirements.txt](requirements.txt)
- Environment sample: [.env.example](.env.example)
- License: [LICENSE](LICENSE)

Why this project is useful
- Fast, reproducible offline scoring with BERT models.
- Optional LLM (Gemini) ensemble for richer context when an API key is provided.
- Simple HTTP API with endpoints for ensemble or BERT-only assessments.
- Small, testable, and easy to extend for new traits or scoring logic.

Core components
- [`BERTPredictor`](app.py) — local BERT-based multi-trait predictor.
- [`EnsemblePredictor`](app.py) — merges BERT and Gemini results (70%/30% weighting).
- [`assess_interview`](app.py) — main POST endpoint ("/assess") that returns a full AssessmentResponse.
- [`assess_bert_only`](app.py) — POST endpoint ("/assess-bert-only") to run BERT-only scoring.
- Gemini helpers: [`build_gemini_prompt`](app.py), [`call_gemini_with_retry`](app.py), [`clean_json`](app.py), [`get_gemini_scores`](app.py).
- Request / response Pydantic models: [`AssessmentRequest`](app.py), [`AssessmentResponse`](app.py).

Get started (developer)
1. Clone the repo
   - git clone <your-repo>

2. Create a virtual environment and install
   ```sh
   python -m venv .venv
   source .venv/bin/activate  # macOS / Linux
   .venv\Scripts\activate     # Windows PowerShell
   pip install -r requirements.txt

3. Run app
   ```sh
   python app.py