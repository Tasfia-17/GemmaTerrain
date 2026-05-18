#!/usr/bin/env python3
"""
gemmaterrain.py — CLI entry point

Usage:
    python gemmaterrain.py -l coxs_bazar "Find nearest hospital to Camp 6"
    python gemmaterrain.py -l san_juan --image road.jpg "Is this road passable?"
    python gemmaterrain.py --list
    python gemmaterrain.py --health
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from engine import query, list_locations, health_check, QueryResult
from router import Model


# ============================================================================
# Terminal Colors
# ============================================================================
class C:
    BOLD = "\033[1m"; DIM = "\033[2m"; RESET = "\033[0m"
    CYAN = "\033[36m"; GREEN = "\033[32m"; YELLOW = "\033[33m"
    MAGENTA = "\033[35m"; BLUE = "\033[34m"; RED = "\033[31m"
    WHITE = "\033[97m"


MODEL_COLOR = {Model.E2B: C.CYAN, Model.E4B: C.MAGENTA}
MODEL_LABEL = {Model.E2B: "Gemma 4 E2B", Model.E4B: "Gemma 4 E4B"}


# ============================================================================
# Formatting
# ============================================================================
def format_result(result: QueryResult, loc_info: dict = None) -> str:
    lines = []

    if not result.success:
        lines.append(f"{C.RED}✗ {result.error}{C.RESET}")
        return "\n".join(lines)

    # Geocoding
    if result.geocoded:
        lines.append(f"{C.DIM}GEOCODING{C.RESET}")
        for place, info in result.geocoded.items():
            lines.append(f"  📍 {C.YELLOW}{place}{C.RESET} → {C.DIM}({info['lat']:.6f}, {info['lon']:.6f}){C.RESET}")
        lines.append("")

    # Model routing
    if result.route_decision:
        rd = result.route_decision
        mc = MODEL_COLOR.get(rd.model, C.CYAN)
        lines.append(f"{C.DIM}MODEL ROUTING{C.RESET}")
        lines.append(f"  {mc}{MODEL_LABEL[rd.model]}{C.RESET}  {C.DIM}← {rd.reason}{C.RESET}")
        lines.append("")

    # Thinking
    if result.think_text:
        lines.append(f"{C.DIM}REASONING{C.RESET}")
        for line in result.think_text.split("\n")[:6]:
            lines.append(f"  {C.DIM}{line}{C.RESET}")
        lines.append("")

    # Tool call
    lines.append(f"{C.DIM}TOOL CALL{C.RESET}")
    lines.append(f"  {C.BOLD}Tool:{C.RESET}  {C.CYAN}{result.tool_name}{C.RESET}")
    lines.append(f"  {C.BOLD}Args:{C.RESET}  {C.DIM}{json.dumps(result.tool_args, separators=(', ', ': '))}{C.RESET}")
    lines.append("")

    # Results
    data = result.result
    lines.append(f"{C.DIM}RESULTS{C.RESET}")

    if "error" in data:
        lines.append(f"  {C.RED}{data['error']}{C.RESET}")
    elif result.tool_name == "list_pois":
        lines.append(f"  Found {C.GREEN}{C.BOLD}{data['count']}{C.RESET} {data['poi_type']}(s) within {C.CYAN}{data['radius_m']}m{C.RESET}")
        for p in data.get("pois", [])[:6]:
            lines.append(f"    • {p.get('name','Unnamed')} {C.DIM}({p['distance_m']:.0f}m){C.RESET}")
        if data["count"] > 6:
            lines.append(f"    {C.DIM}... and {data['count'] - 6} more{C.RESET}")
    elif result.tool_name == "find_nearest_poi_with_route":
        lines.append(f"  Nearest {data.get('poi_type','POI')}(s): {C.GREEN}{data['found']} found{C.RESET}")
        for p in data.get("nearest_pois", [])[:5]:
            lines.append(f"    🚶 {C.BOLD}{p.get('name','Unnamed')}{C.RESET} — {C.CYAN}{p['walk_minutes']:.1f} min{C.RESET} {C.DIM}({p['distance_m']:.0f}m){C.RESET}")
    elif result.tool_name == "calculate_route":
        lines.append(f"  📏 Distance:  {C.CYAN}{data['distance_km']:.2f} km{C.RESET}")
        lines.append(f"  🚶 Walk time: {C.GREEN}{data['walk_minutes']:.0f} min{C.RESET}")
    elif result.tool_name == "generate_isochrone":
        lines.append(f"  ⏱️  {C.CYAN}{data['max_minutes']} min{C.RESET} walkable area")
        lines.append(f"  🔗 Reachable nodes: {C.GREEN}{data['reachable_nodes']:,}{C.RESET}")
    elif result.tool_name == "geocode_place":
        lines.append(f"  📍 {C.YELLOW}{data['place']}{C.RESET} → ({data['lat']:.6f}, {data['lon']:.6f})")
    else:
        lines.append(f"  {json.dumps(data, indent=2)}")

    lines.append("")

    # Performance
    lines.append(f"{C.DIM}PERFORMANCE{C.RESET}")
    lines.append(f"  ⏱️  Total:  {C.GREEN}{result.query_time:.2f}s{C.RESET}")
    if result.llm_stats and result.llm_stats.tokens_per_sec > 0:
        s = result.llm_stats
        lines.append(f"  🧠 {s.prompt_tokens}+{s.completion_tokens} tokens @ {C.CYAN}{s.tokens_per_sec:.1f} tok/s{C.RESET}")
    if loc_info:
        lines.append(f"  🗺️  {C.DIM}{loc_info.get('nodes',0):,} nodes · {loc_info.get('pois',0):,} POIs{C.RESET}")

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="🌍 GemmaTerrain — Multimodal GeoAI for the Unconnected",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -l coxs_bazar "Find nearest hospital to Camp 6"
  %(prog)s -l san_juan --image road.jpg "Is this road passable? Route to nearest shelter."
  %(prog)s -l jakarta "What can I reach in 15 minutes from Gelora?"
  %(prog)s --list
  %(prog)s --health
        """,
    )
    parser.add_argument("query", nargs="?", help="Natural language spatial query")
    parser.add_argument("-l", "--location", help="Location slug")
    parser.add_argument("--image", help="Image file for multimodal query")
    parser.add_argument("--battery", type=float, default=100.0, help="Battery level 0-100 (affects model routing)")
    parser.add_argument("--list", action="store_true", help="List available locations")
    parser.add_argument("--health", action="store_true", help="System health check")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--model", choices=["e2b", "e4b"], help="Force specific model (overrides router)")
    args = parser.parse_args()

    if args.list:
        locs = list_locations()
        if not locs:
            print("No locations found. Run build_location.py first.")
            sys.exit(1)
        print(f"\n{C.BOLD}{C.BLUE}🌍 GemmaTerrain{C.RESET} — Available locations:\n")
        for slug, info in locs.items():
            print(f"  {C.CYAN}{slug}{C.RESET}: {info['name']}")
            print(f"    {C.DIM}{info['nodes']:,} nodes · {info['pois']:,} POIs · {info['places']:,} places{C.RESET}")
        print()
        sys.exit(0)

    if args.health:
        h = health_check()
        if args.json:
            print(json.dumps(h, indent=2))
        else:
            print(f"\n{C.BOLD}{C.BLUE}🌍 GemmaTerrain{C.RESET} Health\n")
            for model_key, label in [("e2b", "Gemma 4 E2B"), ("e4b", "Gemma 4 E4B")]:
                online = h[f"{model_key}_server"]
                status = f"{C.GREEN}✓ Online{C.RESET}" if online else f"{C.RED}✗ Offline{C.RESET}"
                alias = h.get(f"{model_key}_info", {}).get("alias", "")
                print(f"  {label}: {status}  {C.DIM}{alias}{C.RESET}")
            print(f"  Locations: {h['locations_available']} ({', '.join(h['locations'])})")
            print()
        sys.exit(0)

    if not args.query:
        parser.print_help()
        sys.exit(1)

    locs = list_locations()
    location = args.location or (list(locs.keys())[0] if locs else None)
    if not location:
        print("Error: No locations found.")
        sys.exit(1)

    # If --model flag given, patch battery to force routing
    battery = args.battery
    if args.model == "e2b":
        battery = 10.0  # forces E2B
    elif args.model == "e4b":
        battery = 100.0

    result = query(
        args.query,
        location=location,
        image_path=args.image,
        battery_pct=battery,
    )

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print()
        print(format_result(result, locs.get(location, {})))

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
