#!/usr/bin/env python3
"""
demo_damage_data.py

Prepopulates Qdrant with sample damage reports for demo purposes.
Run this after installing Qdrant to see the damage-aware routing in action.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from damage_store import report_damage

# Sample damage reports for Cox's Bazar
COXS_BAZAR_REPORTS = [
    {
        "text": "Main road flooded after monsoon, water up to knee level, impassable for pedestrians",
        "lat": 21.434300,
        "lon": 92.001790,
        "severity": "high",
        "reporter": "Field Worker A",
    },
    {
        "text": "Bridge partially collapsed, structural damage visible, unsafe for crossing",
        "lat": 21.450120,
        "lon": 92.012340,
        "severity": "high",
        "reporter": "Engineer B",
    },
    {
        "text": "Road blocked by fallen tree, passable on foot but difficult",
        "lat": 21.423560,
        "lon": 92.015670,
        "severity": "medium",
        "reporter": "Field Worker C",
    },
    {
        "text": "Large pothole on main path, minor hazard",
        "lat": 21.440200,
        "lon": 92.008900,
        "severity": "low",
        "reporter": "Community Member",
    },
]

# Sample damage reports for San Juan
SAN_JUAN_REPORTS = [
    {
        "text": "Street flooded from hurricane damage, water covering entire road surface",
        "lat": 18.465320,
        "lon": -66.105740,
        "severity": "high",
        "reporter": "Response Team 1",
    },
    {
        "text": "Power lines down across pathway, dangerous to cross",
        "lat": 18.468900,
        "lon": -66.112300,
        "severity": "high",
        "reporter": "Safety Inspector",
    },
    {
        "text": "Debris from damaged buildings blocking sidewalk",
        "lat": 18.462100,
        "lon": -66.108500,
        "severity": "medium",
        "reporter": "Cleanup Crew",
    },
]

# Sample damage reports for Jakarta
JAKARTA_REPORTS = [
    {
        "text": "Jalan tergenang banjir, tidak bisa dilewati pejalan kaki",
        "lat": -6.200000,
        "lon": 106.816666,
        "severity": "high",
        "reporter": "Petugas Lapangan",
    },
    {
        "text": "Road subsidence after heavy rain, unstable ground",
        "lat": -6.195000,
        "lon": 106.823000,
        "severity": "medium",
        "reporter": "City Engineer",
    },
]


def populate_location(location: str, reports: list):
    print(f"\n📍 Populating {location}...")
    for r in reports:
        result = report_damage(
            text=r["text"],
            lat=r["lat"],
            lon=r["lon"],
            location=location,
            severity=r["severity"],
            reporter=r.get("reporter", ""),
        )
        print(f"  ✓ {r['severity'].upper()}: {r['text'][:60]}...")
    print(f"✅ Added {len(reports)} reports to {location}")


def main():
    print("🚧 GemmaTerrain - Demo Damage Data Generator")
    print("=" * 60)

    populate_location("coxs_bazar", COXS_BAZAR_REPORTS)
    populate_location("san_juan", SAN_JUAN_REPORTS)
    populate_location("jakarta", JAKARTA_REPORTS)

    print("\n" + "=" * 60)
    print("✅ Demo data loaded!")
    print("\nTry these queries to see damage-aware routing:")
    print("  • python gemmaterrain.py -l coxs_bazar 'Route from Camp 3 to Camp 8W avoiding damage'")
    print("  • python gemmaterrain.py -l san_juan 'Safe route from Condado to Miramar'")
    print("  • python gemmaterrain.py -l jakarta 'Check route from Menteng to Gelora for hazards'")


if __name__ == "__main__":
    main()
