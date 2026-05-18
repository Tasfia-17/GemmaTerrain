"""
src/engine.py
GemmaTerrain — Query Engine

Orchestrates the full pipeline:
  geocode → route → Gemma 4 → spatial tool → result

This is the single entry point for both CLI and Streamlit.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Add src/ to path when run from project root
sys.path.insert(0, str(Path(__file__).parent))

import spatial_tools
import geocode_layer
import gemma_client
from router import route, RouteDecision, Model
from spatial_tools import TOOL_DEFINITIONS


# ============================================================================
# Result
# ============================================================================

@dataclass
class QueryResult:
    tool_name: str = ""
    tool_args: dict = field(default_factory=dict)
    result: dict = field(default_factory=dict)
    geocoded: dict = field(default_factory=dict)
    query_time: float = 0.0
    modified_query: str = ""
    route_decision: RouteDecision = None
    llm_stats: gemma_client.LLMStats = None
    think_text: str = ""
    success: bool = True
    error: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        if self.route_decision:
            d["route_decision"] = {"model": self.route_decision.model.value, "reason": self.route_decision.reason}
        if self.llm_stats:
            d["llm_stats"] = asdict(self.llm_stats)
        return d


# ============================================================================
# Location helpers
# ============================================================================

def list_locations() -> dict:
    locations = {}
    for cfg_path in Path("data").glob("*/config.json"):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
                locations[cfg["slug"]] = {
                    "name": cfg["name"],
                    "center": [cfg["center"]["lat"], cfg["center"]["lon"]],
                    "bounds": cfg["bounds"],
                    "nodes": cfg.get("nodes", 0),
                    "pois": cfg.get("pois", 0),
                    "places": cfg.get("places", 0),
                    "examples": cfg.get("examples", []),
                }
        except Exception:
            continue
    return locations


def load_location(slug: str) -> bool:
    try:
        spatial_tools.load_location(slug)
        geocode_layer.load_location(slug)
        return True
    except FileNotFoundError:
        return False


# ============================================================================
# Main query function
# ============================================================================

def query(
    user_query: str,
    location: str | None = None,
    image_path: str | None = None,
    audio_b64: str | None = None,
    battery_pct: float = 100.0,
) -> QueryResult:
    """
    Process a natural language spatial query through the full GemmaTerrain pipeline.

    Args:
        user_query:   Natural language question
        location:     Location slug (e.g. "coxs_bazar")
        image_path:   Optional path to image for multimodal queries
        audio_b64:    Optional base64-encoded WAV audio
        battery_pct:  Current battery level (0-100), affects model routing
    """
    t0 = time.time()

    # Load location
    if location and location != spatial_tools.current_location:
        if not load_location(location):
            return QueryResult(success=False, error=f"Location not found: {location}", query_time=time.time() - t0)

    if spatial_tools.G_nk is None:
        return QueryResult(success=False, error="No location loaded.", query_time=time.time() - t0)

    # Geocode place names in query
    try:
        modified_query, geocoded = geocode_layer.geocode_query(user_query)
    except Exception:
        modified_query, geocoded = user_query, {}

    # Route to appropriate model
    decision = route(
        modified_query,
        has_image=bool(image_path),
        has_audio=bool(audio_b64),
        battery_pct=battery_pct,
    )

    # Call Gemma 4
    gemma_resp = gemma_client.call(
        query=modified_query,
        tools=TOOL_DEFINITIONS,
        model=decision.model,
        image_path=image_path,
        audio_b64=audio_b64,
    )

    if not gemma_resp.success:
        return QueryResult(
            geocoded=geocoded,
            modified_query=modified_query,
            route_decision=decision,
            llm_stats=gemma_resp.stats,
            query_time=time.time() - t0,
            success=False,
            error=gemma_resp.error,
        )

    # Execute spatial tool
    try:
        result_json = spatial_tools.execute_tool(gemma_resp.tool_name, **gemma_resp.tool_args)
        result = json.loads(result_json)
    except Exception as e:
        return QueryResult(
            tool_name=gemma_resp.tool_name,
            tool_args=gemma_resp.tool_args,
            geocoded=geocoded,
            modified_query=modified_query,
            route_decision=decision,
            llm_stats=gemma_resp.stats,
            query_time=time.time() - t0,
            success=False,
            error=f"Tool execution failed: {e}",
        )

    return QueryResult(
        tool_name=gemma_resp.tool_name,
        tool_args=gemma_resp.tool_args,
        result=result,
        geocoded=geocoded,
        modified_query=modified_query,
        route_decision=decision,
        llm_stats=gemma_resp.stats,
        think_text=gemma_resp.stats.think_text if gemma_resp.stats else "",
        query_time=time.time() - t0,
        success=True,
    )


def health_check() -> dict:
    locations = list_locations()
    return {
        "e2b_server": gemma_client.health_check(Model.E2B),
        "e4b_server": gemma_client.health_check(Model.E4B),
        "e2b_info": gemma_client.get_model_info(Model.E2B),
        "e4b_info": gemma_client.get_model_info(Model.E4B),
        "locations_available": len(locations),
        "locations": list(locations.keys()),
        "current_location": spatial_tools.current_location,
        "spatial_loaded": spatial_tools.G_nk is not None,
    }
