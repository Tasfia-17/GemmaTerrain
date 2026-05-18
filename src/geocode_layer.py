"""
src/geocode_layer.py
GemmaTerrain — Geocode Layer
Repurposed from DreamMeridian. Resolves OSM place names to coordinates
before the query reaches Gemma 4, so the LLM only handles tool selection.
"""

import duckdb
import re
from pathlib import Path

con = None
current_location = None
known_places = {}


def load_location(slug: str):
    global con, current_location, known_places

    if current_location == slug and con is not None:
        return

    db_path = Path("data") / slug / f"{slug}.duckdb"
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    if con is not None:
        con.close()
    con = duckdb.connect(str(db_path), read_only=True)
    current_location = slug
    _load_known_places()


def _load_known_places():
    global known_places
    known_places = {}
    if con is None:
        return
    tables = [t[0] for t in con.execute("SHOW TABLES").fetchall()]
    if "places" not in tables:
        return
    for name, name_lower, lat, lon, place_type in con.execute(
        "SELECT name, name_lower, lat, lon, place_type FROM places"
    ).fetchall():
        known_places[name_lower] = {"name": name, "lat": lat, "lon": lon, "place_type": place_type}


def geocode_query(query: str) -> tuple[str, dict]:
    """
    Replace known place names in query with (lat X, lon Y) coordinates.
    Returns (modified_query, geocode_info).
    """
    geocode_info = {}
    modified = query
    query_lower = query.lower()
    used_spans = []

    for name_lower, info in sorted(known_places.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(name_lower) + r"\b"
        for match in re.finditer(pattern, query_lower):
            start, end = match.span()
            if any(not (end <= us or start >= ue) for us, ue in used_spans):
                continue
            original = query[start:end]
            geocode_info[info["name"]] = {"lat": info["lat"], "lon": info["lon"], "place_type": info["place_type"]}
            modified = re.compile(re.escape(original), re.IGNORECASE).sub(
                f"(lat {info['lat']:.6f}, lon {info['lon']:.6f})", modified, count=1
            )
            used_spans.append((start, end))

    return modified, geocode_info
