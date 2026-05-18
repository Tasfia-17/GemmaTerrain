#!/usr/bin/env python3
"""
build_location.py — GemmaTerrain Location Builder
Repurposed from DreamMeridian. Downloads OSM data for any location.

Usage:
    python build_location.py "Cox's Bazar, Bangladesh" coxs_bazar
    python build_location.py "Dhaka, Bangladesh" dhaka
    python build_location.py "Mandalay, Myanmar" mandalay --tiles
"""

import json
import math
import sys
import time
import warnings
from pathlib import Path

import networkit as nk
import osmnx as ox
import duckdb
import pandas as pd
import requests

warnings.filterwarnings("ignore", message=".*Geometry is in a geographic CRS.*")
warnings.filterwarnings("ignore", message=".*Overpass max query area.*")

PLACE_TAGS = {"place": ["neighbourhood","neighborhood","suburb","quarter","locality","hamlet","village","town","city","borough","district"]}

OSM_TAGS = {
    "amenity": ["hospital","clinic","doctors","pharmacy","dentist","nursing_home","police","fire_station","shelter","emergency_service","school","kindergarten","college","university","place_of_worship","community_centre","social_facility","refugee_site","townhall","bank","atm","money_transfer","marketplace","food_court","drinking_water","water_point","toilets","bus_station","ferry_terminal","fuel","post_office"],
    "healthcare": True,
    "emergency": True,
    "shop": ["supermarket","convenience","grocery","general","food","pharmacy","chemist","medical_supply","hardware","mobile_phone"],
    "building": ["hospital","school","university","college","government","civic","public","mosque","temple","church","religious","fire_station","police","warehouse"],
    "office": ["government","ngo","diplomatic","humanitarian","un","international_organization"],
    "man_made": ["water_tower","water_well","water_works","pumping_station","storage_tank","communications_tower"],
    "aeroway": ["aerodrome","helipad","heliport"],
    "railway": ["station","halt"],
    "natural": ["spring","water"],
}

TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TILE_ZOOMS = range(11, 17)


def _bounds(G):
    lats = [d["y"] for _, d in G.nodes(data=True) if "y" in d]
    lons = [d["x"] for _, d in G.nodes(data=True) if "x" in d]
    p = 0.005
    return {"north": max(lats)+p, "south": min(lats)-p, "east": max(lons)+p, "west": min(lons)-p}


def _center(G):
    lats = [d["y"] for _, d in G.nodes(data=True) if "y" in d]
    lons = [d["x"] for _, d in G.nodes(data=True) if "x" in d]
    return {"lat": (max(lats)+min(lats))/2, "lon": (max(lons)+min(lons))/2}


def _nx_to_nk(G_nx):
    nm = {n: i for i, n in enumerate(G_nx.nodes())}
    rm = {i: n for n, i in nm.items()}
    G_nk = nk.Graph(len(nm), weighted=True, directed=False)
    for u, v, d in G_nx.edges(data=True):
        G_nk.addEdge(nm[u], nm[v], d.get("length", 1.0))
    return G_nk, nm, rm


def _process_features(gdf):
    if len(gdf) == 0:
        return []
    gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())
    gdf["centroid"] = gdf_proj.geometry.centroid.to_crs("EPSG:4326")
    gdf["lat"] = gdf["centroid"].y
    gdf["lon"] = gdf["centroid"].x

    def tag_info(row):
        for k in OSM_TAGS:
            if k in row.index and pd.notna(row[k]):
                return k, str(row[k])
        return "other", "yes"

    ti = gdf.apply(tag_info, axis=1)
    gdf["tag_key"] = [t[0] for t in ti]
    gdf["tag_value"] = [t[1] for t in ti]
    cols = [c for c in ["lat","lon","name","tag_key","tag_value"] if c in gdf.columns]
    result = gdf[cols].to_dict("records")
    seen, deduped = set(), []
    for r in result:
        key = (r.get("name"), round(r["lat"],4), round(r["lon"],4), r["tag_value"])
        if key not in seen:
            seen.add(key); deduped.append(r)
    return deduped


def download_tiles(bounds, slug):
    tile_dir = Path("static/tiles") / slug
    tile_dir.mkdir(parents=True, exist_ok=True)

    def ll_to_tile(lat, lon, z):
        lr = math.radians(lat); n = 2**z
        return int((lon+180)/360*n), int((1-math.asinh(math.tan(lr))/math.pi)/2*n)

    all_tiles = []
    for z in TILE_ZOOMS:
        x0, y1 = ll_to_tile(bounds["south"], bounds["west"], z)
        x1, y0 = ll_to_tile(bounds["north"], bounds["east"], z)
        all_tiles += [(z,x,y) for x in range(x0,x1+1) for y in range(y0,y1+1)]

    print(f"Downloading {len(all_tiles)} tiles...")
    headers = {"User-Agent": "GemmaTerrain/1.0 (Humanitarian AI)"}
    dl = sk = 0
    for z, x, y in all_tiles:
        p = tile_dir / f"{z}/{x}/{y}.png"
        if p.exists():
            sk += 1; continue
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(TILE_URL.format(z=z,x=x,y=y), headers=headers, timeout=10)
            if r.status_code == 200:
                p.write_bytes(r.content); dl += 1
        except Exception:
            pass
        if dl > 0:
            time.sleep(0.1)
    print(f"  ✓ {dl} downloaded, {sk} cached")


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)

    place_name = sys.argv[1]
    slug = sys.argv[2].lower().replace(" ","_").replace("'","")
    do_tiles = "--tiles" in sys.argv

    out = Path("data") / slug
    out.mkdir(parents=True, exist_ok=True)
    db_path = out / f"{slug}.duckdb"

    print(f"\n{'='*60}\nGemmaTerrain Location Builder\n{'='*60}")
    print(f"Location: {place_name}\nSlug:     {slug}\n{'='*60}")
    t0 = time.time()

    # [1/5] Street network
    print("\n[1/5] Street network...")
    G_nx = ox.graph_from_place(place_name, network_type="all", simplify=True)
    print(f"  ✓ {G_nx.number_of_nodes():,} nodes, {G_nx.number_of_edges():,} edges")

    # [2/5] Convert to NetworKit
    print("\n[2/5] Converting to NetworKit...")
    G_nk, nm, rm = _nx_to_nk(G_nx)
    ox.save_graphml(G_nx, out / f"{slug}.graphml")
    nk.graphio.writeGraph(G_nk, str(out / f"{slug}.nkb"), nk.Format.NetworkitBinary)
    with open(out / f"{slug}_mappings.json", "w") as f:
        json.dump({"nx_to_nk": nm, "nk_to_nx": rm}, f)
    print(f"  ✓ Saved .nkb + mappings")

    # [3/5] Nodes to DuckDB
    print("\n[3/5] Building spatial database...")
    con = duckdb.connect(str(db_path))
    con.install_extension("spatial"); con.load_extension("spatial")
    nodes = [(n, d["y"], d["x"], d["x"], d["y"]) for n, d in G_nx.nodes(data=True) if "y" in d]
    con.execute("DROP TABLE IF EXISTS nodes")
    con.execute("CREATE TABLE nodes (node_id BIGINT PRIMARY KEY, lat DOUBLE, lon DOUBLE, geom GEOMETRY)")
    con.executemany("INSERT INTO nodes VALUES (?,?,?,ST_Point(?,?))", nodes)
    con.execute("CREATE INDEX nodes_geom_idx ON nodes USING RTREE (geom)")
    print(f"  ✓ {len(nodes):,} nodes")

    # [4/5] POI features
    print("\n[4/5] Downloading POI features...")
    try:
        gdf = ox.features_from_place(place_name, tags=OSM_TAGS)
        features = _process_features(gdf)
    except Exception:
        features = []
        for tag_key, tag_values in OSM_TAGS.items():
            try:
                gdf = ox.features_from_place(place_name, tags={tag_key: tag_values if tag_values is not True else True})
                features += _process_features(gdf)
            except Exception:
                pass

    con.execute("DROP TABLE IF EXISTS osm_features")
    if features:
        df = pd.DataFrame(features)
        if "name" not in df.columns:
            df["name"] = None
        con.execute("CREATE TABLE osm_features AS SELECT * FROM df")
        con.execute("ALTER TABLE osm_features ADD COLUMN geom GEOMETRY")
        con.execute("UPDATE osm_features SET geom=ST_Point(lon,lat) WHERE lon IS NOT NULL")
    else:
        con.execute("CREATE TABLE osm_features (lat DOUBLE, lon DOUBLE, name VARCHAR, tag_key VARCHAR, tag_value VARCHAR, geom GEOMETRY)")
    con.execute("CREATE INDEX osm_features_geom_idx ON osm_features USING RTREE (geom)")
    con.execute("CREATE INDEX osm_features_tag_idx ON osm_features(tag_key,tag_value)")
    poi_count = con.execute("SELECT COUNT(*) FROM osm_features").fetchone()[0]
    print(f"  ✓ {poi_count:,} POIs")

    # [5/5] Place names
    print("\n[5/5] Downloading place names...")
    try:
        pgdf = ox.features_from_place(place_name, tags=PLACE_TAGS)
        pgdf_proj = pgdf.to_crs(pgdf.estimate_utm_crs())
        pgdf["centroid"] = pgdf_proj.geometry.centroid.to_crs("EPSG:4326")
        pgdf["lat"] = pgdf["centroid"].y; pgdf["lon"] = pgdf["centroid"].x
        pgdf["place_type"] = pgdf.apply(lambda r: str(r["place"]) if "place" in r.index and pd.notna(r["place"]) else "unknown", axis=1)
        pgdf = pgdf[pgdf["name"].notna()].copy()
        seen, places = set(), []
        for _, row in pgdf.iterrows():
            k = row["name"].lower()
            if k not in seen:
                seen.add(k); places.append({"name": row["name"], "lat": row["lat"], "lon": row["lon"], "place_type": row["place_type"]})
    except Exception:
        places = []

    con.execute("DROP TABLE IF EXISTS places")
    if places:
        pdf = pd.DataFrame(places); pdf["name_lower"] = pdf["name"].str.lower()
        con.execute("CREATE TABLE places AS SELECT * FROM pdf")
        con.execute("CREATE INDEX places_name_idx ON places(name_lower)")
    else:
        con.execute("CREATE TABLE places (name VARCHAR, lat DOUBLE, lon DOUBLE, place_type VARCHAR, name_lower VARCHAR)")
    place_count = len(places)
    print(f"  ✓ {place_count:,} places")
    con.close()

    # Optional tiles
    if do_tiles:
        print("\n[+] Downloading map tiles...")
        download_tiles(_bounds(G_nx), slug)

    # Config
    config = {
        "slug": slug, "name": place_name,
        "center": _center(G_nx), "bounds": _bounds(G_nx),
        "nodes": G_nx.number_of_nodes(), "edges": G_nx.number_of_edges(),
        "pois": poi_count, "places": place_count,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "examples": [
            f"Find nearest hospital to {place_name.split(',')[0]}",
            f"15 minute walking radius from {place_name.split(',')[0]}",
            f"Clinics within 2km of {place_name.split(',')[0]}",
        ],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    print(f"\n{'='*60}\nBUILD COMPLETE — {(time.time()-t0)/60:.1f} min\n{'='*60}")
    print(f"  data/{slug}/  ({G_nk.numberOfNodes():,} nodes · {poi_count:,} POIs · {place_count:,} places)")
    print(f"\n  Run: python gemmaterrain.py -l {slug} \"Find nearest hospital\"")


if __name__ == "__main__":
    main()
