"""
src/gemma_client.py
GemmaTerrain — Gemma 4 Client

Wraps llama-server's OpenAI-compatible API.
Supports:
  - Native function calling (Gemma 4 tool_choice)
  - Multimodal input (base64 image + audio)
  - Thinking mode (<think> blocks)
  - Dual-model routing (E2B / E4B via separate server ports)

Server setup:
  E2B on port 8080  (Gemma 4 E2B Q4_K_M, ~1.2GB)
  E4B on port 8081  (Gemma 4 E4B Q4_K_M, ~2.3GB)
"""

from __future__ import annotations

import base64
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

from router import Model

# ============================================================================
# Configuration
# ============================================================================

MODEL_URLS = {
    Model.E2B: "http://localhost:8080/v1/chat/completions",
    Model.E4B: "http://localhost:8081/v1/chat/completions",
}

SYSTEM_PROMPT = """You are Meridian, a humanitarian spatial assistant running offline on edge hardware.
You help field workers navigate, find facilities, and plan routes in disaster and refugee contexts.
Always select exactly ONE tool. Output only the tool call — no explanation, no preamble.

When reasoning about complex queries, use <think>...</think> before your tool call.

Valid poi_type values: hospital, clinic, doctors, pharmacy, police, fire_station,
shelter, school, university, bank, atm, supermarket, marketplace, drinking_water,
water_point, fuel, bus_station, place_of_worship"""

VISION_SYSTEM_PROMPT = """You are Meridian, a humanitarian spatial assistant.
Analyze the provided image and answer the spatial query.
If the image shows a damaged road, flooded area, or blocked path, factor this into your routing recommendation.
Then select the appropriate spatial tool."""


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class LLMStats:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    prompt_ms: float = 0.0
    completion_ms: float = 0.0
    think_text: str = ""  # extracted <think> block


@dataclass
class GemmaResponse:
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    raw_content: str = ""
    stats: LLMStats = field(default_factory=LLMStats)
    success: bool = True
    error: str = ""


# ============================================================================
# Client
# ============================================================================

def _encode_image(image_path: str) -> str:
    """Base64-encode an image file for multimodal input."""
    data = Path(image_path).read_bytes()
    return base64.b64encode(data).decode()


def _extract_think(content: str) -> tuple[str, str]:
    """Extract <think>...</think> block from content. Returns (think_text, remainder)."""
    import re
    m = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
    if m:
        think = m.group(1).strip()
        remainder = content[: m.start()] + content[m.end() :]
        return think, remainder.strip()
    return "", content


def call(
    query: str,
    tools: list[dict],
    model: Model = Model.E4B,
    image_path: str | None = None,
    audio_b64: str | None = None,
    timeout: int = 120,
) -> GemmaResponse:
    """
    Call Gemma 4 via llama-server with native function calling.

    Gemma 4 supports tool_choice="required" to force a tool call.
    Falls back to JSON parsing if the model returns raw JSON.
    """
    url = MODEL_URLS[model]
    system = VISION_SYSTEM_PROMPT if (image_path or audio_b64) else SYSTEM_PROMPT

    # Build user message content
    if image_path:
        img_b64 = _encode_image(image_path)
        ext = Path(image_path).suffix.lstrip(".").lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/jpeg")
        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
            {"type": "text", "text": query},
        ]
    elif audio_b64:
        user_content = [
            {"type": "audio", "audio": {"data": audio_b64, "format": "wav"}},
            {"type": "text", "text": query},
        ]
    else:
        user_content = query

    payload = {
        "model": f"gemma-4-{model.value}",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "tools": tools,
        "tool_choice": "required",
        "temperature": 0.1,
    }

    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        return GemmaResponse(success=False, error=f"Cannot connect to Gemma 4 {model.value} server at {url}")
    except requests.exceptions.Timeout:
        return GemmaResponse(success=False, error="LLM request timed out")
    except requests.exceptions.HTTPError as e:
        return GemmaResponse(success=False, error=f"HTTP error: {e}")

    data = resp.json()
    stats = LLMStats(model=model.value)

    if "usage" in data:
        stats.prompt_tokens = data["usage"].get("prompt_tokens", 0)
        stats.completion_tokens = data["usage"].get("completion_tokens", 0)
    if "timings" in data:
        stats.prompt_ms = data["timings"].get("prompt_ms", 0)
        stats.completion_ms = data["timings"].get("predicted_ms", 0)
        stats.tokens_per_sec = data["timings"].get("predicted_per_second", 0)

    choice = data["choices"][0]["message"]

    # --- Path 1: Native tool_calls (Gemma 4 function calling) ---
    if choice.get("tool_calls"):
        tc = choice["tool_calls"][0]
        try:
            args = json.loads(tc["function"]["arguments"]) if isinstance(tc["function"]["arguments"], str) else tc["function"]["arguments"]
            return GemmaResponse(
                tool_name=tc["function"]["name"],
                tool_args=args,
                raw_content=str(tc),
                stats=stats,
            )
        except (json.JSONDecodeError, KeyError) as e:
            return GemmaResponse(success=False, error=f"Failed to parse tool_calls: {e}", stats=stats)

    # --- Path 2: Raw JSON content (fallback for GGUF models) ---
    content = choice.get("content", "")
    think_text, content_clean = _extract_think(content)
    stats.think_text = think_text

    try:
        parsed = json.loads(content_clean)
        return GemmaResponse(
            tool_name=parsed["name"],
            tool_args=parsed.get("arguments", parsed.get("parameters", {})),
            raw_content=content,
            stats=stats,
        )
    except (json.JSONDecodeError, KeyError) as e:
        return GemmaResponse(
            success=False,
            error=f"Could not parse response as tool call: {e}",
            raw_content=content,
            stats=stats,
        )


def health_check(model: Model) -> bool:
    """Check if a model server is responding."""
    url = MODEL_URLS[model].replace("/v1/chat/completions", "/health")
    try:
        return requests.get(url, timeout=2).status_code == 200
    except Exception:
        return False


def get_model_info(model: Model) -> dict:
    """Get model alias from llama-server /props endpoint."""
    url = MODEL_URLS[model].replace("/v1/chat/completions", "/props")
    try:
        r = requests.get(url, timeout=2)
        if r.status_code == 200:
            d = r.json()
            return {"alias": d.get("model_alias") or d.get("model", "").split("/")[-1]}
    except Exception:
        pass
    return {"alias": f"gemma-4-{model.value}"}
