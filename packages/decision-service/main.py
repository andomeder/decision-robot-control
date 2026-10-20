"""Fast decision service: wraps /v1/systemone with optimized image preprocessing.

Run:
  uvicorn main:app --host 0.0.0.0 --port 8090 --workers 1
Env:
  DECISION_BASE_URL  (default http://127.0.0.1:8091)
  DECISION_MODEL     (optional)
  DECISION_API_KEY   (optional)
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "decision-client"))
from decision_client import DecisionClient, decision_settings, make_image_data_url  # noqa: E402

_SETTINGS = decision_settings()
BASE_URL = _SETTINGS["base_url"]
MODEL = _SETTINGS["model"]
API_KEY = _SETTINGS["api_key"]

_client: Optional[DecisionClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _client
    _client = DecisionClient(BASE_URL, model=MODEL, api_key=API_KEY, max_image_side=224, image_quality=75)
    yield
    await _client.aclose()


app = FastAPI(lifespan=lifespan, title="decision-service")


@app.get("/health")
async def health():
    return {"status": "ok", "backend": BASE_URL, "model": MODEL}


@app.post("/decide")
async def decide(
    state: str = Form(...),  # JSON string
    questions: str = Form(...),  # JSON string
    images: Optional[List[UploadFile]] = File(None),
):
    t0 = time.perf_counter()
    try:
        state_obj = json.loads(state)
        q_obj = json.loads(questions)
    except json.JSONDecodeError as e:
        return JSONResponse({"error": f"bad json: {e}"}, status_code=400)

    imgs = None
    if images:
        imgs = [await f.read() for f in images]

    try:
        result = await _client.decide(state_obj, q_obj, images=imgs)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)

    return {
        "answers": result.answers,
        "input_tokens": result.input_tokens,
        "latency_s": round(result.latency_s, 4),
        "service_latency_s": round(time.perf_counter() - t0, 4),
    }


@app.post("/decide_json")
async def decide_json(body: Dict[str, Any]):
    """JSON body: {state, questions, images:[dataurl,...]} (no multipart overhead)."""
    t0 = time.perf_counter()
    state = body.get("state")
    questions = body.get("questions", {})
    images = body.get("images")
    try:
        result = await _client.decide(state, questions, images=images)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    return {
        "answers": result.answers,
        "input_tokens": result.input_tokens,
        "latency_s": round(result.latency_s, 4),
        "service_latency_s": round(time.perf_counter() - t0, 4),
    }
