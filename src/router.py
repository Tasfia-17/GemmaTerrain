"""
src/router.py
GemmaTerrain — Cactus Model Router

Routes queries to the appropriate Gemma 4 model based on:
  - Input modality (text / image / audio)
  - Query complexity (simple lookup vs. multi-hop reasoning)
  - Battery state (low power → force E2B)

Models:
  E2B  — Gemma 4 2B  — intent classification, language detection, simple lookups
  E4B  — Gemma 4 4B  — spatial tool calling, compound queries, multimodal reasoning
"""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class Model(str, Enum):
    E2B = "e2b"   # Gemma 4 2B
    E4B = "e4b"   # Gemma 4 4B


@dataclass
class RouteDecision:
    model: Model
    reason: str


# Keywords that signal complex multi-hop reasoning → E4B
_COMPLEX_PATTERNS = [
    "along the route", "along the way",
    "which camps", "which areas", "which communities",
    "not within", "outside", "violate", "standard",
    "damaged", "flooded", "impassable", "collapsed",
    "and then", "after that", "also show",
    "isochrone", "walkable area", "minute radius",
    "supply", "vaccine", "medicine", "stock",
]

# Simple single-tool patterns → E2B is sufficient
_SIMPLE_PATTERNS = [
    "nearest", "closest", "find", "where is",
    "how far", "distance", "route from", "walk from",
    "list", "show me", "pharmacies", "hospitals", "clinics",
]


def route(
    query: str,
    has_image: bool = False,
    has_audio: bool = False,
    battery_pct: float = 100.0,
) -> RouteDecision:
    """
    Decide which Gemma 4 model to use for this query.

    Rules (in priority order):
    1. Low battery (<20%) → E2B always
    2. Image or audio input → E4B (multimodal)
    3. Complex multi-hop query → E4B
    4. Default → E2B
    """
    if battery_pct < 20.0:
        return RouteDecision(Model.E2B, f"low battery ({battery_pct:.0f}%)")

    if has_image or has_audio:
        modalities = []
        if has_image:
            modalities.append("image")
        if has_audio:
            modalities.append("audio")
        return RouteDecision(Model.E4B, f"multimodal input ({', '.join(modalities)})")

    q = query.lower()
    for pattern in _COMPLEX_PATTERNS:
        if pattern in q:
            return RouteDecision(Model.E4B, f"complex query (matched: '{pattern}')")

    return RouteDecision(Model.E2B, "simple spatial lookup")
