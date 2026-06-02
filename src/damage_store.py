"""
src/damage_store.py
GemmaTerrain — Qdrant Damage Memory

Field workers submit damage reports (text or image caption).
Each report is embedded and stored in Qdrant with geo-payload.
Before finalizing a route, the engine queries Qdrant for damage
near the path — any semantic hit triggers a reroute warning.

Collection schema per point:
  id:      UUID
  vector:  384-dim (all-MiniLM-L6-v2, runs on CPU / ARM)
  payload:
    text:      original report text
    lat, lon:  GPS coordinates of the damage
    location:  city slug (e.s. "coxs_bazar")
    severity:  "low" | "medium" | "high"
    timestamp: ISO-8601 string
    reporter:  optional name
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    Range,
    GeoRadius,
    GeoPoint,
)
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────
# Singleton state
# ────────────────────────────────────────────────────────────
_client: QdrantClient | None = None
_model: SentenceTransformer | None = None

COLLECTION = "damage_reports"
VECTOR_SIZE = 384
QDRANT_PATH = "./qdrant_data"   # local on-disk; no server needed


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(path=QDRANT_PATH)
        _ensure_collection()
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        # all-MiniLM-L6-v2 is ~80 MB, runs well on Pi 5
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _ensure_collection():
    client = _client
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def report_damage(
    text: str,
    lat: float,
    lon: float,
    location: str,
    severity: str = "medium",
    reporter: str = "",
) -> str:
    """
    Embed and store a field damage report in Qdrant.
    Returns JSON confirmation with the assigned ID.
    """
    client = _get_client()
    point_id = str(uuid.uuid4())
    vector = _embed(text)
    payload = {
        "text": text,
        "lat": lat,
        "lon": lon,
        "location": location,
        "severity": severity,
        "reporter": reporter,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )
    return json.dumps({
        "status": "stored",
        "id": point_id,
        "text": text,
        "lat": lat,
        "lon": lon,
        "severity": severity,
    })


def query_damage_near_route(
    path_coords: list[dict],
    location: str,
    radius_m: float = 150.0,
    top_k: int = 5,
) -> list[dict]:
    """
    For each sampled point on a route, semantically search Qdrant
    for damage reports within radius_m.  Returns deduplicated hits.

    path_coords: list of {"lat": float, "lon": float}
    """
    if not path_coords:
        return []

    client = _get_client()
    model = _get_model()

    # Embed a generic "danger on route" anchor — we want anything damage-like
    anchor = "road blocked flooded damaged collapsed impassable obstacle hazard"
    anchor_vec = model.encode(anchor, normalize_embeddings=True).tolist()

    seen_ids: set[str] = set()
    hits: list[dict] = []

    # Sample up to 20 evenly spaced points to keep it fast
    step = max(1, len(path_coords) // 20)
    sampled = path_coords[::step]

    for pt in sampled:
        results = client.search(
            collection_name=COLLECTION,
            query_vector=anchor_vec,
            query_filter=Filter(
                must=[
                    FieldCondition(key="location", match={"value": location}),
                ]
            ),
            limit=top_k,
            score_threshold=0.35,   # only semantically relevant hits
            with_payload=True,
        )
        for r in results:
            pid = str(r.id)
            if pid in seen_ids:
                continue
            # Geo check: is this report actually near this path point?
            dlat = (r.payload["lat"] - pt["lat"]) * 111000
            dlon = (r.payload["lon"] - pt["lon"]) * 111000 * 0.85
            dist = (dlat**2 + dlon**2) ** 0.5
            if dist <= radius_m:
                seen_ids.add(pid)
                hits.append({
                    "id": pid,
                    "text": r.payload["text"],
                    "lat": r.payload["lat"],
                    "lon": r.payload["lon"],
                    "severity": r.payload.get("severity", "medium"),
                    "timestamp": r.payload.get("timestamp", ""),
                    "distance_m": round(dist, 1),
                    "score": round(r.score, 3),
                })

    hits.sort(key=lambda x: x["distance_m"])
    return hits


def list_damage_reports(location: str, limit: int = 50) -> str:
    """Return all damage reports for a location as JSON."""
    client = _get_client()
    results, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=Filter(
            must=[FieldCondition(key="location", match={"value": location})]
        ),
        limit=limit,
        with_payload=True,
    )
    reports = [{"id": str(r.id), **r.payload} for r in results]
    return json.dumps({"location": location, "count": len(reports), "reports": reports})


def delete_damage_report(report_id: str) -> str:
    """Remove a single damage report by ID."""
    client = _get_client()
    client.delete(
        collection_name=COLLECTION,
        points_selector=[report_id],
    )
    return json.dumps({"status": "deleted", "id": report_id})


def get_collection_stats() -> dict:
    """Return Qdrant collection info for health check display."""
    try:
        client = _get_client()
        info = client.get_collection(COLLECTION)
        return {
            "status": "ok",
            "points": info.points_count,
            "vectors_size": VECTOR_SIZE,
            "distance": "cosine",
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
