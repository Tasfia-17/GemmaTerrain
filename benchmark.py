#!/usr/bin/env python3
"""
benchmark.py — GemmaTerrain Benchmark Suite

Runs 30 queries across all 3 locations, measures accuracy and latency.
Saves results to benchmarks/benchmark_{device}.json.

Usage:
    python benchmark.py
    python benchmark.py --model e4b   # force E4B for all queries
"""

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engine import query, list_locations

QUERIES = [
    # Cox's Bazar
    ("coxs_bazar", "Find the nearest hospital to Camp 6",           "find_nearest_poi_with_route"),
    ("coxs_bazar", "How do I walk from Camp 3 to Camp 9",           "calculate_route"),
    ("coxs_bazar", "Show 15 minute walkable area from Camp 8W",     "generate_isochrone"),
    ("coxs_bazar", "Clinics within 2km of Camp 10",                 "list_pois"),
    ("coxs_bazar", "Pharmacies along the route from Camp 3 to Camp 8W", "find_along_route"),
    ("coxs_bazar", "Nearest shelter to Camp 12",                    "find_nearest_poi_with_route"),
    ("coxs_bazar", "Schools within 1km of Camp 6",                  "list_pois"),
    ("coxs_bazar", "20 minute walking radius from Camp 5",          "generate_isochrone"),
    ("coxs_bazar", "Walking route from Camp 8W to Camp 15",         "calculate_route"),
    ("coxs_bazar", "Water points near Camp 6",                      "list_pois"),
    # San Juan
    ("san_juan",   "Find the nearest clinic to Condado",            "find_nearest_poi_with_route"),
    ("san_juan",   "Walking route from Santurce to Miramar",        "calculate_route"),
    ("san_juan",   "List pharmacies within 1km of Ocean Park",      "list_pois"),
    ("san_juan",   "20 minute walking radius from Condado",         "generate_isochrone"),
    ("san_juan",   "Nearest hospital to Río Piedras",               "find_nearest_poi_with_route"),
    ("san_juan",   "Shelters within 2km of Santurce",               "list_pois"),
    ("san_juan",   "How do I walk from Old San Juan to Condado?",   "calculate_route"),
    ("san_juan",   "15 minute walkable area from Miramar",          "generate_isochrone"),
    ("san_juan",   "Clinics along the route from Condado to Santurce", "find_along_route"),
    ("san_juan",   "Banks within 500m of Ocean Park",               "list_pois"),
    # Jakarta
    ("jakarta",    "Find nearest hospital to Menteng",              "find_nearest_poi_with_route"),
    ("jakarta",    "How far to walk from Gambir to Kemang",         "calculate_route"),
    ("jakarta",    "Schools within 2km of Gelora",                  "list_pois"),
    ("jakarta",    "15 minute walking radius from Serdang",         "generate_isochrone"),
    ("jakarta",    "Nearest pharmacy to Cipulir",                   "find_nearest_poi_with_route"),
    ("jakarta",    "Hospitals within 3km of Menteng",               "list_pois"),
    ("jakarta",    "Walking route from Cipulir to Lebak Bulus",     "calculate_route"),
    ("jakarta",    "20 minute walkable area from Gambir",           "generate_isochrone"),
    ("jakarta",    "Clinics along the route from Gambir to Kemang", "find_along_route"),
    ("jakarta",    "Water points near Gelora",                      "list_pois"),
]


def detect_device() -> tuple[str, str]:
    system, machine = platform.system(), platform.machine()
    if system == "Linux":
        kernel = platform.release().lower()
        if "neptune" in kernel or "valve" in kernel:
            return "steamdeck", "x86-64 Zen 2"
        pi = Path("/proc/device-tree/model")
        if pi.exists():
            m = pi.read_text().strip()
            return ("pi5", "ARM Cortex-A76") if "Pi 5" in m else ("pi4", "ARM Cortex-A72")
        return "linux", machine
    if system == "Darwin":
        try:
            r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=2)
            for chip in ["M4", "M3", "M2", "M1"]:
                if chip in r.stdout:
                    return f"mac_{chip.lower()}", f"Apple {chip}"
        except Exception:
            pass
        return "mac_silicon", "Apple Silicon"
    return "unknown", machine


def run(force_model: str | None = None):
    device, arch = detect_device()
    locs = list_locations()

    print(f"{'='*60}\nGemmaTerrain Benchmark\n{'='*60}")
    print(f"  Device:   {device} ({arch})")
    print(f"  Platform: {platform.platform()}")
    print(f"  Queries:  {len(QUERIES)}")
    print(f"{'='*60}\n")

    # Warmup
    print("Warming up...", flush=True)
    for loc in ["coxs_bazar", "san_juan", "jakarta"]:
        query("Find nearest hospital", location=loc)
    print("Done.\n")

    results = []
    t_total = time.time()

    for i, (loc, q, expected_tool) in enumerate(QUERIES, 1):
        print(f"[{i:2}/{len(QUERIES)}] {loc}: {q[:45]:<45}", end=" ", flush=True)
        t0 = time.time()
        r = query(q, location=loc)
        elapsed = time.time() - t0
        correct = r.success and r.tool_name == expected_tool
        print(f"{'✓' if correct else '✗'} {elapsed:.2f}s  {r.tool_name or 'ERROR'}")

        results.append({
            "location": loc, "query": q,
            "expected_tool": expected_tool,
            "actual_tool": r.tool_name,
            "correct": correct,
            "success": r.success,
            "time": round(elapsed, 3),
            "model_used": r.route_decision.model.value if r.route_decision else None,
            "tokens_per_sec": r.llm_stats.tokens_per_sec if r.llm_stats else 0,
            "error": r.error if not r.success else None,
        })

    total_time = time.time() - t_total
    times = [r["time"] for r in results]
    correct_count = sum(1 for r in results if r["correct"])
    tok_speeds = [r["tokens_per_sec"] for r in results if r["tokens_per_sec"] > 0]
    avg_tok = sum(tok_speeds) / len(tok_speeds) if tok_speeds else 0

    print(f"\n{'='*60}\nRESULTS\n{'='*60}")
    print(f"  Accuracy:  {correct_count}/{len(QUERIES)} ({correct_count/len(QUERIES)*100:.1f}%)")
    print(f"  Mean time: {sum(times)/len(times):.2f}s")
    print(f"  Min/Max:   {min(times):.2f}s / {max(times):.2f}s")
    if avg_tok:
        print(f"  Avg tok/s: {avg_tok:.1f}")
    print(f"  Total:     {total_time:.1f}s")

    # Per-location breakdown
    for loc_name in ["coxs_bazar", "san_juan", "jakarta"]:
        loc_results = [r for r in results if r["location"] == loc_name]
        loc_correct = sum(1 for r in loc_results if r["correct"])
        print(f"  {loc_name}: {loc_correct}/{len(loc_results)}")

    out = {
        "device": device, "architecture": arch, "platform": platform.platform(),
        "total_queries": len(QUERIES),
        "correct": correct_count,
        "accuracy": round(correct_count / len(QUERIES), 3),
        "total_time": round(total_time, 2),
        "mean_time": round(sum(times) / len(times), 2),
        "min_time": round(min(times), 2),
        "max_time": round(max(times), 2),
        "avg_tokens_per_sec": round(avg_tok, 1),
        "results": results,
    }

    Path("benchmarks").mkdir(exist_ok=True)
    out_path = Path("benchmarks") / f"benchmark_{device}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["e2b", "e4b"], help="Force model (overrides router)")
    args = parser.parse_args()
    run(args.model)
