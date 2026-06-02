"""
src/spatial_tools.py
GemmaTerrain — Spatial Tools (+ Qdrant damage-aware routing)

Six original OSM/routing tools, plus two Qdrant-backed tools:
  report_damage     — field workers log damage into Qdrant
  check_route_damage — query Qdrant for hazards along a proposed route
"""

import networkit as nk
import duckdb
import json
from pathlib import Path
from math import sqrt, cos, radians

# ============================================================================
# Global State
# ============================================================================
G_nk = None
node_mapping = {}
reverse_mapping = {}
con = None
current_location = None


def load_location(slug: str):
    global G_nk, node_mapping, reverse_mapping, con, current_location

    if current_location == slug and G_nk is not None:
        return

    base = Path("data") / slug
    if not base.exists():
        raise FileNotFoundError(f"Location not found: {base}")

    G_nk = nk.graphio.readGraph(str(base / f"{slug}.nkb"), nk.Format.NetworkitBinary)

    with open(base / f"{slug}_mappings.json") as f:
        m = json.load(f)
        node_mapping = {int(k): v for k, v in m["nx_to_nk"].items()}
        reverse_mapping = {int(k): v for k, v in m["nk_to_nx"].items()}

    if con is not None:
        con.close()
    con = duckdb.connect(str(base / f"{slug}.duckdb"), read_only=True)
    con.install_extension("spatial")
    con.load_extension("spatial")

    current_location = slug


def get_available_locations() -> dict:
    locations = {}
    for cfg_path in Path("data").glob("*/config.json"):
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
                locations[cfg["slug"]] = {
                    "name": cfg["name"],
                    "center": cfg["center"],
                    "nodes": cfg.get("nodes", 0),
                    "pois": cfg.get("pois", 0),
                    "places": cfg.get("places", 0),
                    "examples": cfg.get("examples", []),
                }
        except Exception:
            continue
    return locations


# ============================================================================
# Helpers
# ============================================================================

def _nearest_node(lat: float, lon: float):
    return con.execute(
        "SELECT node_id, lat, lon FROM nodes ORDER BY ST_Distance(geom, ST_Point(?,?)) LIMIT 1",
        [lon, lat],
    ).fetchone()


def _path_coords(path_nk: list, sample: int = 100) -> list:
    if not path_nk:
        return []
    step = max(1, len(path_nk) // sample)
    coords = []
    for i in range(0, len(path_nk), step):
        nx_id = reverse_mapping.get(path_nk[i])
        if nx_id:
            row = con.execute("SELECT lat, lon FROM nodes WHERE node_id=?", [nx_id]).fetchone()
            if row:
                coords.append({"lat": row[0], "lon": row[1]})
    # always include last
    nx_id = reverse_mapping.get(path_nk[-1])
    if nx_id:
        row = con.execute("SELECT lat, lon FROM nodes WHERE node_id=?", [nx_id]).fetchone()
        if row:
            last = {"lat": row[0], "lon": row[1]}
            if not coords or coords[-1] != last:
                coords.append(last)
    return coords


# ============================================================================
# Spatial Tools (6 functions)
# ============================================================================

def list_pois(poi_type: str, lat: float, lon: float, radius_m: int = 1000) -> str:
    total = con.execute(
        "SELECT COUNT(*) FROM osm_features WHERE tag_value=? AND ST_Distance(geom,ST_Point(?,?))*111000<?",
        [poi_type, lon, lat, radius_m],
    ).fetchone()[0]
    rows = con.execute(
        """SELECT name, lat, lon, ST_Distance(geom,ST_Point(?,?))*111000 AS dist
           FROM osm_features WHERE tag_value=? AND ST_Distance(geom,ST_Point(?,?))*111000<?
           ORDER BY dist LIMIT 50""",
        [lon, lat, poi_type, lon, lat, radius_m],
    ).fetchall()
    return json.dumps({
        "poi_type": poi_type, "count": total, "radius_m": radius_m,
        "center": {"lat": lat, "lon": lon},
        "pois": [{"name": r[0], "lat": r[1], "lon": r[2], "distance_m": r[3]} for r in rows],
    })


def find_nearest_poi_with_route(poi_type: str, lat: float, lon: float, limit: int = 3) -> str:
    pois = con.execute(
        """SELECT name, lat, lon FROM osm_features WHERE tag_value=? AND name IS NOT NULL
           ORDER BY ST_Distance(geom,ST_Point(?,?)) LIMIT ?""",
        [poi_type, lon, lat, limit],
    ).fetchall()
    if not pois:
        return json.dumps({"poi_type": poi_type, "found": 0, "nearest_pois": []})

    start_nk = node_mapping.get(_nearest_node(lat, lon)[0])
    results, nearest_path = [], []

    for idx, (name, plat, plon) in enumerate(pois):
        end_nk = node_mapping.get(_nearest_node(plat, plon)[0])
        if start_nk is None or end_nk is None:
            continue
        store = idx == 0
        d = nk.distance.Dijkstra(G_nk, start_nk, True, store, end_nk)
        d.run()
        dist = d.distance(end_nk)
        if dist < float("inf"):
            if store:
                nearest_path = _path_coords(d.getPath(end_nk))
            results.append({
                "name": name, "lat": plat, "lon": plon,
                "distance_m": round(dist, 1),
                "walk_minutes": round(dist / 83.33, 1),
            })

    results.sort(key=lambda x: x["walk_minutes"])
    return json.dumps({
        "poi_type": poi_type, "found": len(results),
        "nearest_pois": results, "path": nearest_path,
        "start": {"lat": lat, "lon": lon},
    })


def calculate_route(start_lat: float, start_lon: float, end_lat: float, end_lon: float) -> str:
    start_nk = node_mapping.get(_nearest_node(start_lat, start_lon)[0])
    end_nk = node_mapping.get(_nearest_node(end_lat, end_lon)[0])
    if start_nk is None or end_nk is None:
        return json.dumps({"error": "Could not find route nodes"})
    d = nk.distance.Dijkstra(G_nk, start_nk, True, True, end_nk)
    d.run()
    dist = d.distance(end_nk)
    if dist == float("inf"):
        return json.dumps({"error": "No route found"})
    path = d.getPath(end_nk)
    return json.dumps({
        "distance_km": round(dist / 1000, 2),
        "walk_minutes": round(dist / 83.33, 0),
        "num_nodes": len(path),
        "path": _path_coords(path),
    })


def find_along_route(
    start_lat: float, start_lon: float, end_lat: float, end_lon: float,
    poi_type: str = None, buffer_m: int = 100,
) -> str:
    start_nk = node_mapping.get(_nearest_node(start_lat, start_lon)[0])
    end_nk = node_mapping.get(_nearest_node(end_lat, end_lon)[0])
    if start_nk is None or end_nk is None:
        return json.dumps({"error": "Could not find route nodes"})
    d = nk.distance.Dijkstra(G_nk, start_nk, True, True, end_nk)
    d.run()
    dist = d.distance(end_nk)
    if dist == float("inf"):
        return json.dumps({"error": "No route found"})

    path_nk = d.getPath(end_nk)
    path_coords = []
    for nk_id in path_nk:
        nx_id = reverse_mapping.get(nk_id)
        if nx_id:
            row = con.execute("SELECT lat, lon FROM nodes WHERE node_id=?", [nx_id]).fetchone()
            if row:
                path_coords.append((row[0], row[1]))

    if len(path_coords) < 2:
        return json.dumps({"error": "Route too short"})

    sampled = path_coords[:: max(1, len(path_coords) // 40)]
    if path_coords[-1] not in sampled:
        sampled.append(path_coords[-1])

    lats = [p[0] for p in path_coords]
    lons = [p[1] for p in path_coords]
    buf = buffer_m / 111000 * 1.5
    params = [min(lons) - buf, max(lons) + buf, min(lats) - buf, max(lats) + buf]
    type_filter = "AND tag_value=?" if poi_type else ""
    if poi_type:
        params.append(poi_type)

    candidates = con.execute(
        f"SELECT name,lat,lon,tag_key,tag_value FROM osm_features "
        f"WHERE lon BETWEEN ? AND ? AND lat BETWEEN ? AND ? AND name IS NOT NULL {type_filter}",
        params,
    ).fetchall()

    cos_lat = cos(radians((min(lats) + max(lats)) / 2))

    def min_dist(plat, plon):
        best = float("inf")
        for la, lo in sampled:
            d2 = sqrt(((plat - la) * 111000) ** 2 + ((plon - lo) * 111000 * cos_lat) ** 2)
            if d2 < best:
                best = d2
        return best

    pois_along = []
    for name, plat, plon, tkey, tval in candidates:
        d2 = min_dist(plat, plon)
        if d2 <= buffer_m:
            pois_along.append({"name": name, "lat": plat, "lon": plon, "type": tval, "off_route_m": round(d2, 1)})

    return json.dumps({
        "distance_km": round(dist / 1000, 2),
        "walk_minutes": round(dist / 83.33, 0),
        "buffer_m": buffer_m, "poi_type": poi_type,
        "pois_found": len(pois_along), "pois": pois_along[:15],
        "path": _path_coords(path_nk),
    })


def generate_isochrone(lat: float, lon: float, max_minutes: int = 15) -> str:
    start_nk = node_mapping.get(_nearest_node(lat, lon)[0])
    if start_nk is None:
        return json.dumps({"error": "Could not find start node"})
    max_dist = max_minutes * 83.33
    d = nk.distance.Dijkstra(G_nk, start_nk, True, False)
    d.run()
    boundary, reachable = [], 0
    for nk_id in range(G_nk.numberOfNodes()):
        dist = d.distance(nk_id)
        if dist <= max_dist:
            reachable += 1
            if dist > max_dist * 0.8:
                nx_id = reverse_mapping.get(nk_id)
                if nx_id:
                    row = con.execute("SELECT lat, lon FROM nodes WHERE node_id=?", [nx_id]).fetchone()
                    if row:
                        boundary.append({"lat": row[0], "lon": row[1], "walk_minutes": round(dist / 83.33, 1)})
    if len(boundary) > 100:
        boundary = boundary[:: len(boundary) // 100]
    return json.dumps({"max_minutes": max_minutes, "reachable_nodes": reachable, "boundary_points": boundary})


def geocode_place(place_name: str) -> str:
    rows = con.execute(
        "SELECT lat, lon, name FROM osm_features WHERE name ILIKE ? LIMIT 10",
        [f"%{place_name}%"],
    ).fetchall()
    if not rows:
        return json.dumps({"error": f"Place not found: {place_name}"})
    return json.dumps({
        "place": place_name,
        "lat": sum(r[0] for r in rows) / len(rows),
        "lon": sum(r[1] for r in rows) / len(rows),
        "matches": len(rows),
    })


# ============================================================================
# Qdrant Damage Tools (imported lazily to avoid hard dep if qdrant not installed)
# ============================================================================

def _damage_store():
    import damage_store
    return damage_store


def report_damage_tool(
    text: str,
    lat: float,
    lon: float,
    severity: str = "medium",
    reporter: str = "",
) -> str:
    """Store a field damage report in Qdrant with location context."""
    return _damage_store().report_damage(
        text=text, lat=lat, lon=lon,
        location=current_location or "unknown",
        severity=severity, reporter=reporter,
    )


def check_route_damage(
    start_lat: float, start_lon: float,
    end_lat: float, end_lon: float,
    buffer_m: int = 150,
) -> str:
    """
    Calculate route then query Qdrant for damage reports near the path.
    Returns route + damage warnings so the caller can decide to reroute.
    """
    # First get the route
    start_nk = node_mapping.get(_nearest_node(start_lat, start_lon)[0])
    end_nk = node_mapping.get(_nearest_node(end_lat, end_lon)[0])
    if start_nk is None or end_nk is None:
        return json.dumps({"error": "Could not find route nodes"})
    d = nk.distance.Dijkstra(G_nk, start_nk, True, True, end_nk)
    d.run()
    dist = d.distance(end_nk)
    if dist == float("inf"):
        return json.dumps({"error": "No route found"})

    path_coords = _path_coords(d.getPath(end_nk))

    # Query Qdrant for damage near this path
    damage_hits = _damage_store().query_damage_near_route(
        path_coords, current_location or "unknown", radius_m=buffer_m
    )

    return json.dumps({
        "distance_km": round(dist / 1000, 2),
        "walk_minutes": round(dist / 83.33, 0),
        "path": path_coords,
        "damage_warnings": damage_hits,
        "damage_count": len(damage_hits),
        "route_safe": len(damage_hits) == 0,
    })


def list_damage_reports_tool() -> str:
    """List all damage reports for the current location."""
    return _damage_store().list_damage_reports(current_location or "unknown")


# ============================================================================
# Tool Registry
# ============================================================================
TOOLS = {
    "list_pois": list_pois,
    "find_nearest_poi_with_route": find_nearest_poi_with_route,
    "calculate_route": calculate_route,
    "find_along_route": find_along_route,
    "generate_isochrone": generate_isochrone,
    "geocode_place": geocode_place,
    "report_damage": report_damage_tool,
    "check_route_damage": check_route_damage,
    "list_damage_reports": list_damage_reports_tool,
}

TOOL_DEFINITIONS = [
    {
        "name": "list_pois",
        "description": "List points of interest of a given type within a radius. Use for 'find all X near Y' or 'how many X within Zkm'.",
        "parameters": {
            "type": "object",
            "properties": {
                "poi_type": {"type": "string", "enum": ["hospital","clinic","doctors","pharmacy","police","fire_station","shelter","school","university","bank","atm","supermarket","marketplace","drinking_water","water_point","fuel","bus_station","place_of_worship"]},
                "lat": {"type": "number", "description": "Latitude of center point"},
                "lon": {"type": "number", "description": "Longitude of center point"},
                "radius_m": {"type": "integer", "description": "Search radius in meters", "default": 1000},
            },
            "required": ["poi_type", "lat", "lon"],
        },
    },
    {
        "name": "find_nearest_poi_with_route",
        "description": "Find the nearest POI of a type and return the walking route. Use for 'nearest X to Y' or 'closest X'.",
        "parameters": {
            "type": "object",
            "properties": {
                "poi_type": {"type": "string", "enum": ["hospital","clinic","doctors","pharmacy","police","fire_station","shelter","school","university","bank","atm","supermarket","marketplace","drinking_water","water_point","fuel","bus_station","place_of_worship"]},
                "lat": {"type": "number"},
                "lon": {"type": "number"},
                "limit": {"type": "integer", "default": 3},
            },
            "required": ["poi_type", "lat", "lon"],
        },
    },
    {
        "name": "calculate_route",
        "description": "Calculate walking route and distance between two points. Use for 'how do I walk from A to B' or 'distance between X and Y'.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"}, "start_lon": {"type": "number"},
                "end_lat": {"type": "number"}, "end_lon": {"type": "number"},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon"],
        },
    },
    {
        "name": "find_along_route",
        "description": "Find POIs along a walking route between two points. Use for 'pharmacies along the way from A to B'.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"}, "start_lon": {"type": "number"},
                "end_lat": {"type": "number"}, "end_lon": {"type": "number"},
                "poi_type": {"type": "string"},
                "buffer_m": {"type": "integer", "default": 100},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon"],
        },
    },
    {
        "name": "generate_isochrone",
        "description": "Generate walkable area reachable within N minutes from a point. Use for 'what can I reach in X minutes' or 'X minute walking radius'.",
        "parameters": {
            "type": "object",
            "properties": {
                "lat": {"type": "number"}, "lon": {"type": "number"},
                "max_minutes": {"type": "integer", "default": 15},
            },
            "required": ["lat", "lon"],
        },
    },
    {
        "name": "geocode_place",
        "description": "Get coordinates for a place name. Use when you need to resolve a place name to coordinates.",
        "parameters": {
            "type": "object",
            "properties": {"place_name": {"type": "string"}},
            "required": ["place_name"],
        },
    },
    {
        "name": "report_damage",
        "description": "Report road or infrastructure damage. Use when a user describes a blocked road, flood, collapse, or hazard. Stores the report in persistent memory so future routes avoid it.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "Description of the damage"},
                "lat": {"type": "number", "description": "Latitude of the damaged location"},
                "lon": {"type": "number", "description": "Longitude of the damaged location"},
                "severity": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                "reporter": {"type": "string", "default": ""},
            },
            "required": ["text", "lat", "lon"],
        },
    },
    {
        "name": "check_route_damage",
        "description": "Calculate a walking route AND check Qdrant for damage reports near the path. Use when user asks to route somewhere safely, or mentions avoiding flooded/damaged/blocked roads.",
        "parameters": {
            "type": "object",
            "properties": {
                "start_lat": {"type": "number"}, "start_lon": {"type": "number"},
                "end_lat": {"type": "number"}, "end_lon": {"type": "number"},
                "buffer_m": {"type": "integer", "default": 150},
            },
            "required": ["start_lat", "start_lon", "end_lat", "end_lon"],
        },
    },
    {
        "name": "list_damage_reports",
        "description": "List all active damage reports for the current area.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def execute_tool(tool_name: str, **kwargs) -> str:
    if tool_name not in TOOLS:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    try:
        return TOOLS[tool_name](**kwargs)
    except Exception as e:
        return json.dumps({"error": str(e)})
