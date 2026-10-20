"""Client for POST /v1/systemone.

Works with a local llama.cpp server and with any host that speaks the same route.
"""

from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import httpx
from PIL import Image

__all__ = ["DecisionClient", "DecisionResult", "decision_settings", "load_dotenv", "make_image_data_url"]


def load_dotenv() -> None:
    """Read ./.env without overriding variables already set."""
    path = os.path.join(os.getcwd(), ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def decision_settings() -> Dict[str, Optional[str]]:
    """Base URL, model, and key. DECISION_URL is an alias for the base URL."""
    load_dotenv()
    base = (os.environ.get("DECISION_BASE_URL") or os.environ.get("DECISION_URL") or "http://127.0.0.1:8091").strip()
    model = (os.environ.get("DECISION_MODEL") or "").strip() or None
    key = (os.environ.get("DECISION_API_KEY") or "").strip() or None
    return {"base_url": base.rstrip("/"), "model": model, "api_key": key}

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_IMAGE = 224
DEFAULT_IMAGE_QUALITY = 80


def make_image_data_url(img: Any, max_side: int = DEFAULT_MAX_IMAGE, quality: int = DEFAULT_IMAGE_QUALITY) -> str:
    """Convert PIL Image / bytes / path to a compact JPEG data URL."""
    if isinstance(img, (str, bytes, bytearray)):
        if isinstance(img, str) and not img.startswith("data:"):
            with open(img, "rb") as f:
                img = f.read()
        if isinstance(img, (bytes, bytearray)):
            im = Image.open(io.BytesIO(img))
        else:
            im = Image.open(img)
    else:
        im = img
    im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality, optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


@dataclass
class DecisionResult:
    answers: Dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    def choice(self, name: str) -> Optional[str]:
        a = self.answers.get(name, {})
        return a.get("choice")

    def score(self, name: str) -> Optional[float]:
        a = self.answers.get(name, {})
        return a.get("score")

    def noul(self, name: str) -> Optional[float]:
        a = self.answers.get(name, {})
        return a.get("noul")

    def confidence(self, name: str) -> Optional[float]:
        a = self.answers.get(name, {})
        return a.get("confidence")


class DecisionClient:
    """Async decision client with pooling + retries."""

    def __init__(
        self,
        base_url: str,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_connections: int = 10,
        max_keepalive: int = 10,
        max_image_side: int = DEFAULT_MAX_IMAGE,
        image_quality: int = DEFAULT_IMAGE_QUALITY,
        retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_image_side = max_image_side
        self.image_quality = image_quality
        self.retries = retries
        limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=timeout,
            limits=limits,
            headers=headers,
            http2=False,
        )

    @classmethod
    def from_env(cls, **kwargs: Any) -> "DecisionClient":
        cfg: Dict[str, Any] = dict(decision_settings())
        cfg.update(kwargs)
        return cls(**cfg)

    async def aclose(self):
        await self._client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.aclose()

    async def decide(
        self,
        state: Any,
        questions: Dict[str, Any],
        images: Optional[List[Any]] = None,
    ) -> DecisionResult:
        imgs = None
        if images:
            imgs = [make_image_data_url(im, self.max_image_side, self.image_quality) for im in images]

        payload: Dict[str, Any] = {"state": state, "questions": questions}
        if self.model:
            payload["model"] = self.model
        if imgs:
            payload["images"] = imgs

        url = "/v1/systemone"
        t0 = time.perf_counter()
        last_exc = None
        for attempt in range(self.retries + 1):
            try:
                resp = await self._client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                usage = data.get("usage", {})
                return DecisionResult(
                    answers=data.get("answers", {}),
                    input_tokens=usage.get("input_tokens", 0),
                    output_tokens=usage.get("output_tokens", 0),
                    latency_s=time.perf_counter() - t0,
                    raw=data,
                )
            except (httpx.HTTPError, httpx.TimeoutException) as e:
                last_exc = e
                if attempt < self.retries:
                    await _asleep(0.05 * (attempt + 1))
                else:
                    raise
        raise last_exc  # unreachable


async def _asleep(s: float):
    import asyncio
    await asyncio.sleep(s)
