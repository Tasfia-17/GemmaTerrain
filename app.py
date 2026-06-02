#!/usr/bin/env python3
"""
app.py — GemmaTerrain Streamlit Dashboard

Multimodal GeoAI for the Unconnected.
Powered by Gemma 4 E2B/E4B on ARM edge hardware.
"""

import json
import math
import platform
import re
import subprocess
from datetime import timedelta
from pathlib import Path

import folium
import psutil
import streamlit as st
from streamlit_folium import st_folium

import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))

from engine import query, list_locations, health_check as engine_health
from router import Model

# ============================================================================
# CSS
# ============================================================================
CSS = """
<style>
@font-face { font-family:'Clan'; src:url('app/static/fonts/ClanOT-Book.woff2') format('woff2'); font-weight:400; }
@font-face { font-family:'Clan'; src:url('app/static/fonts/ClanOT-Medium.woff2') format('woff2'); font-weight:500; }
@font-face { font-family:'Clan'; src:url('app/static/fonts/ClanOT-Bold.woff2') format('woff2'); font-weight:700; }
@font-face { font-family:'JetBrains Mono'; src:url('app/static/fonts/JetBrainsMono-Regular.woff2') format('woff2'); }

html,body,[class*="css"],.stApp,.stApp *,.stMarkdown,.stTextInput input,
.stSelectbox,.stButton button,.stMetric,section[data-testid="stSidebar"],
section[data-testid="stSidebar"] *,div[data-testid="stMarkdownContainer"] * {
    font-family:'Clan',-apple-system,BlinkMacSystemFont,sans-serif !important;
}
h1,h2,h3,.stApp h1,.stApp h2,.stApp h3 { font-family:'Clan',sans-serif !important; font-weight:700 !important; }
#MainMenu,footer,header { visibility:hidden; }
.block-container { padding-top:1.5rem; padding-bottom:1rem; }
section[data-testid="stSidebar"] { min-width:300px !important; width:300px !important; }
section[data-testid="stSidebar"] > div { width:300px !important; }

.sys-card { background:rgba(30,41,59,.6); border:1px solid rgba(71,85,105,.4); border-radius:8px; padding:.75rem; margin-bottom:.5rem; }
.sys-header { display:flex; justify-content:space-between; align-items:center; margin-bottom:.5rem; }
.sys-device { font-weight:600; font-size:.9rem; color:#f1f5f9; }
.sys-os { font-size:.75rem; color:#94a3b8; }
.sys-uptime { font-size:.7rem; color:#64748b; background:rgba(100,116,139,.2); padding:.2rem .5rem; border-radius:4px; }
.stats-row { display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin-top:.5rem; }
.stat-box { text-align:center; padding:.4rem; background:rgba(15,23,42,.5); border-radius:6px; }
.stat-val { font-size:1rem; font-weight:700; color:#f1f5f9; line-height:1.2; }
.stat-val.warm { color:#fbbf24; } .stat-val.hot { color:#f87171; } .stat-val.ok { color:#4ade80; }
.stat-lbl { font-size:.65rem; color:#64748b; text-transform:uppercase; letter-spacing:.05em; }

.model-badge { display:inline-flex; align-items:center; gap:.4rem; padding:.3rem .7rem; border-radius:6px; font-size:.8rem; font-weight:600; margin:.2rem 0; }
.model-e2b { background:rgba(6,182,212,.15); border:1px solid rgba(6,182,212,.3); color:#22d3ee; }
.model-e4b { background:rgba(168,85,247,.15); border:1px solid rgba(168,85,247,.3); color:#c084fc; }
.model-offline { background:rgba(239,68,68,.15); border:1px solid rgba(239,68,68,.3); color:#f87171; }
.model-dot { width:7px; height:7px; border-radius:50%; }
.model-e2b .model-dot { background:#22d3ee; } .model-e4b .model-dot { background:#c084fc; }
.model-offline .model-dot { background:#f87171; }

.think-block { background:rgba(99,102,241,.08); border-left:3px solid rgba(99,102,241,.4); padding:.6rem .8rem; border-radius:0 6px 6px 0; font-size:.8rem; color:#a5b4fc; font-family:'JetBrains Mono',monospace; white-space:pre-wrap; margin:.5rem 0; }
.route-reason { font-size:.75rem; color:#64748b; margin-top:.2rem; }
.geo-badge { display:inline-flex; align-items:center; gap:.3rem; background:rgba(251,191,36,.15); border:1px solid rgba(251,191,36,.3); color:#fbbf24; padding:.2rem .4rem; border-radius:4px; font-size:.75rem; font-weight:500; }
.geo-coords { font-size:.65rem; color:#64748b; font-family:'JetBrains Mono',monospace; }
.result-item { padding:.35rem 0; border-bottom:1px solid rgba(71,85,105,.2); font-size:.85rem; }
.result-item:last-child { border-bottom:none; }
.result-name { font-weight:500; color:#f1f5f9; }
.result-detail { font-size:.8rem; color:#94a3b8; }
.section-hdr { font-size:.65rem; font-weight:600; text-transform:uppercase; letter-spacing:.08em; color:#64748b; margin:.6rem 0 .4rem 0; }
iframe { border-radius:10px !important; border:1px solid rgba(71,85,105,.3) !important; }
</style>
"""

# ============================================================================
# Hardware stats (repurposed from DreamMeridian)
# ============================================================================
@st.cache_data(ttl=60)
def _hw_info() -> dict:
    info = {"device_short": "Unknown", "os_name": "Unknown", "os_version": "",
            "cpu_arch": platform.machine(), "cpu_cores": psutil.cpu_count(logical=True) or 0,
            "mem_total_gb": psutil.virtual_memory().total / 1024**3}
    if Path("/proc/device-tree/model").exists():
        m = Path("/proc/device-tree/model").read_text().strip().replace("\x00", "")
        short = re.sub(r" Rev [\d.]+", "", re.sub(r" Model ([A-Z])", r" \1", m.replace("Raspberry Pi ", "Pi ")))
        info["device_short"] = short.strip()
    elif platform.system() == "Darwin":
        info["device_short"] = "Mac"
    if Path("/boot/dietpi/.version").exists():
        info["os_name"] = "DietPi"
    elif platform.system() == "Darwin":
        info["os_name"] = "macOS"; info["os_version"] = platform.mac_ver()[0]
    elif Path("/etc/os-release").exists():
        for line in Path("/etc/os-release").read_text().split("\n"):
            if line.startswith("ID="):
                info["os_name"] = line.split("=")[1].strip('"').title()
    return info


def _dyn_stats() -> dict:
    s = {"cpu_temp": None, "cpu_pct": psutil.cpu_percent(interval=0.1),
         "mem_pct": psutil.virtual_memory().percent, "disk_pct": 0, "uptime": "—"}
    try:
        r = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=2)
        if r.returncode == 0:
            s["cpu_temp"] = float(r.stdout.split("=")[1].replace("'C", ""))
    except Exception:
        t = Path("/sys/class/thermal/thermal_zone0/temp")
        if t.exists():
            s["cpu_temp"] = int(t.read_text().strip()) / 1000
    try:
        s["disk_pct"] = psutil.disk_usage("/").percent
    except Exception:
        pass
    try:
        up = float(Path("/proc/uptime").read_text().split()[0])
        td = timedelta(seconds=int(up))
        s["uptime"] = f"{td.days}d {td.seconds//3600}h" if td.days else f"{td.seconds//3600}h {(td.seconds%3600)//60}m"
    except Exception:
        pass
    return s


# ============================================================================
# Map
# ============================================================================
POI_COLORS = {"hospital": "red", "clinic": "red", "doctors": "red", "pharmacy": "green",
              "school": "blue", "university": "blue", "shelter": "orange",
              "police": "darkblue", "bank": "purple", "atm": "purple"}


def _build_map(result, loc: dict) -> folium.Map:
    m = folium.Map(location=loc["center"], zoom_start=13, tiles="cartodbdark_matter")
    if not result or not result.success:
        return m

    for place, info in result.geocoded.items():
        folium.CircleMarker([info["lat"], info["lon"]], radius=8, color="#fbbf24",
                            fill=True, fillColor="#fbbf24", fillOpacity=0.9,
                            popup=f"📍 {place}").add_to(m)

    data = result.result
    tool = result.tool_name
    color = POI_COLORS.get(data.get("poi_type", ""), "gray")

    if tool == "list_pois":
        for p in data.get("pois", []):
            folium.Marker([p["lat"], p["lon"]], popup=f"<b>{p.get('name','')}</b><br>{p.get('distance_m',0):.0f}m",
                          icon=folium.Icon(color=color, icon="info-sign")).add_to(m)

    elif tool == "find_nearest_poi_with_route":
        path = data.get("path", [])
        if path:
            folium.PolyLine([[p["lat"], p["lon"]] for p in path], weight=4, color="#3b82f6", opacity=0.8).add_to(m)
            if "start" in data:
                folium.Marker([data["start"]["lat"], data["start"]["lon"]],
                              icon=folium.Icon(color="green", icon="play"), popup="Start").add_to(m)
        for i, p in enumerate(data.get("nearest_pois", [])):
            folium.Marker([p["lat"], p["lon"]], popup=f"<b>{p.get('name','')}</b><br>🚶 {p['walk_minutes']:.0f} min",
                          icon=folium.Icon(color=color if i == 0 else "lightgray", icon="info-sign")).add_to(m)

    elif tool in ("calculate_route", "find_along_route"):
        path = data.get("path", [])
        if path:
            coords = [[p["lat"], p["lon"]] for p in path]
            folium.PolyLine(coords, weight=4, color="#3b82f6", opacity=0.8).add_to(m)
            folium.Marker(coords[0], icon=folium.Icon(color="green", icon="play")).add_to(m)
            folium.Marker(coords[-1], icon=folium.Icon(color="red", icon="stop")).add_to(m)
        for p in data.get("pois", []):
            folium.Marker([p["lat"], p["lon"]], popup=p.get("name"),
                          icon=folium.Icon(color="orange")).add_to(m)

    elif tool == "generate_isochrone":
        boundary = data.get("boundary_points", [])
        args = result.tool_args
        cx, cy = args.get("lat"), args.get("lon")
        if boundary and cx and cy:
            boundary.sort(key=lambda p: math.atan2(p["lat"] - cx, p["lon"] - cy))
            folium.Polygon([[p["lat"], p["lon"]] for p in boundary],
                           color="#a855f7", fill=True, fillOpacity=0.2).add_to(m)
            folium.Marker([cx, cy], icon=folium.Icon(color="purple", icon="user")).add_to(m)

    # Fit bounds
    pts = ([[i["lat"], i["lon"]] for i in result.geocoded.values()]
           + [[p["lat"], p["lon"]] for p in data.get("nearest_pois", []) + data.get("pois", [])]
           + [[p["lat"], p["lon"]] for p in data.get("path", [])]
           + [[p["lat"], p["lon"]] for p in data.get("boundary_points", [])])
    if "start" in data:
        pts.append([data["start"]["lat"], data["start"]["lon"]])
    if len(pts) >= 2:
        m.fit_bounds(pts, padding=[30, 30])
    return m


# ============================================================================
# Sidebar system panel
# ============================================================================
@st.fragment(run_every="2s")
def _system_panel(health: dict):
    hw = _hw_info()
    s = _dyn_stats()

    temp_cls = "ok"
    if s["cpu_temp"] is not None:
        temp_cls = "hot" if s["cpu_temp"] > 70 else ("warm" if s["cpu_temp"] > 55 else "ok")
        temp_disp = f"{s['cpu_temp']:.0f}°"
    else:
        temp_disp = "—"

    st.markdown(f"""
<div class="sys-card">
  <div class="sys-header">
    <div>
      <div class="sys-device">{hw['device_short']}</div>
      <div class="sys-os">{hw['os_name']} {hw['os_version']} · {hw['cpu_arch']} · {hw['cpu_cores']} cores</div>
    </div>
    <div class="sys-uptime">⏱ {s['uptime']}</div>
  </div>
  <div class="stats-row">
    <div class="stat-box"><div class="stat-val {temp_cls}">{temp_disp}</div><div class="stat-lbl">Temp</div></div>
    <div class="stat-box"><div class="stat-val">{s['cpu_pct']:.0f}%</div><div class="stat-lbl">CPU</div></div>
    <div class="stat-box"><div class="stat-val">{s['mem_pct']:.0f}%</div><div class="stat-lbl">RAM</div></div>
    <div class="stat-box"><div class="stat-val">{s['disk_pct']:.0f}%</div><div class="stat-lbl">Disk</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    # Model status badges
    for model_key, label, css_cls in [("e2b", "Gemma 4 E2B (2B)", "model-e2b"), ("e4b", "Gemma 4 E4B (4B)", "model-e4b")]:
        online = health.get(f"{model_key}_server", False)
        alias = health.get(f"{model_key}_info", {}).get("alias", "")
        if online:
            st.markdown(f'<div class="model-badge {css_cls}"><div class="model-dot"></div>{label} <span style="opacity:.6;font-size:.7rem">{alias}</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="model-badge model-offline"><div class="model-dot"></div>{label} offline</div>', unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================
def main():
    st.set_page_config(page_title="GemmaTerrain", page_icon="🌍", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    locations = list_locations()
    if not locations:
        st.error("No locations found. Run `python build_location.py` first.")
        return

    health = engine_health()

    # Session state
    for k, v in [("result", None), ("current_loc", list(locations.keys())[0]),
                 ("query_text", ""), ("uploaded_image", None)]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.selectbox("Location", options=list(locations.keys()),
                     format_func=lambda x: locations[x]["name"], key="loc_select")
        selected = st.session_state.loc_select

        if selected != st.session_state.current_loc:
            st.session_state.current_loc = selected
            st.session_state.result = None
            st.session_state.query_text = ""
            st.rerun()

        loc = locations[selected]
        st.caption(f"{loc['nodes']:,} nodes · {loc['pois']:,} POIs · {loc['places']:,} places")
        st.divider()

        _system_panel(health)
        st.divider()

        st.markdown('<div class="section-hdr">Try asking</div>', unsafe_allow_html=True)
        for i, ex in enumerate(loc.get("examples", [])[:4]):
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.query_text = ex
                st.rerun()

        st.divider()
        battery = st.slider("Battery %", 0, 100, 100, help="Simulates low-power routing to Gemma 4 E2B")

    # ── Main content ─────────────────────────────────────────────────────────
    loc = locations[selected]

    col_h1, col_h2 = st.columns([3, 2])
    with col_h1:
        st.markdown("## 🌍 GemmaTerrain")
        st.caption(f"Multimodal GeoAI for the Unconnected · **{loc['name']}**")
    with col_h2:
        st.markdown("""
<div style="display:flex;gap:.5rem;flex-wrap:wrap;justify-content:flex-end;margin-top:.5rem;font-size:.8rem;">
  <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);color:#94a3b8;padding:.3rem .6rem;border-radius:5px">🧠 Gemma 4</span>
  <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);color:#94a3b8;padding:.3rem .6rem;border-radius:5px">🗄️ DuckDB</span>
  <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);color:#94a3b8;padding:.3rem .6rem;border-radius:5px">🛤️ NetworKit</span>
  <span style="background:rgba(59,130,246,.1);border:1px solid rgba(59,130,246,.2);color:#94a3b8;padding:.3rem .6rem;border-radius:5px">📡 Offline</span>
</div>""", unsafe_allow_html=True)

    # Query input + image upload
    col_q, col_img = st.columns([4, 1])
    with col_q:
        query_text = st.text_input("Query", value=st.session_state.query_text,
                                   placeholder=f"Ask about {loc['name'].split(',')[0]}...",
                                   label_visibility="collapsed")
        st.session_state.query_text = query_text
    with col_img:
        uploaded = st.file_uploader("📷", type=["jpg", "jpeg", "png"], label_visibility="collapsed",
                                    help="Upload image for multimodal query")

    # Save uploaded image to temp file
    image_path = None
    if uploaded:
        tmp = Path("/tmp/gt_upload") / uploaded.name
        tmp.parent.mkdir(exist_ok=True)
        tmp.write_bytes(uploaded.read())
        image_path = str(tmp)
        st.image(str(tmp), width=120, caption="Multimodal input")

    if st.button("Query", type="primary", disabled=not query_text):
        with st.spinner("Processing..."):
            result = query(query_text, location=selected, image_path=image_path, battery_pct=battery)
            st.session_state.result = result
            if not result.success:
                st.error(result.error)

    # Map
    st_folium(_build_map(st.session_state.result, loc), height=440, use_container_width=True)

    # Results panel
    result = st.session_state.result
    if result and result.success:
        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            st.metric("Query Time", f"{result.query_time:.2f}s")

            if result.route_decision:
                rd = result.route_decision
                css = "model-e2b" if rd.model == Model.E2B else "model-e4b"
                label = "Gemma 4 E2B" if rd.model == Model.E2B else "Gemma 4 E4B"
                st.markdown(f'<div class="model-badge {css}"><div class="model-dot"></div>{label}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="route-reason">{rd.reason}</div>', unsafe_allow_html=True)

            if result.geocoded:
                st.markdown('<div class="section-hdr">Geocoded</div>', unsafe_allow_html=True)
                for place, info in result.geocoded.items():
                    st.markdown(f'<div class="geo-badge">📍 {place}</div><div class="geo-coords">{info["lat"]:.4f}, {info["lon"]:.4f}</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="section-hdr">Tool</div>', unsafe_allow_html=True)
            st.code(result.tool_name, language=None)
            data = result.result
            if "error" in data:
                st.error(data["error"])
            if "count" in data:
                st.metric(data.get("poi_type", "POIs").replace("_", " ").title(), data["count"])
            if "distance_km" in data and "pois_found" not in data:
                st.metric("Route", f"{data['distance_km']:.1f} km · {int(data['walk_minutes'])} min")
            if "reachable_nodes" in data:
                st.metric(f"Reachable ({data['max_minutes']}m)", f"{data['reachable_nodes']:,}")
            if result.tool_name == "check_route_damage":
                dmg = data.get("damage_count", 0)
                safe = data.get("route_safe", True)
                if safe:
                    st.success("✅ Route is clear")
                else:
                    st.error(f"⚠️ {dmg} damage report(s) on this route")
                    for w in data.get("damage_warnings", [])[:3]:
                        st.markdown(f"- **{w['severity'].upper()}** {w['text']} ({w['distance_m']:.0f}m off-route)")

            if result.llm_stats and result.llm_stats.tokens_per_sec > 0:
                s = result.llm_stats
                st.caption(f"⚡ {s.tokens_per_sec:.1f} tok/s · {s.prompt_tokens}+{s.completion_tokens} tokens")

            # Show damage warnings if present
            if result.tool_name == "check_route_damage":
                dmg = data.get("damage_count", 0)
                safe = data.get("route_safe", True)
                if safe:
                    st.success("✅ Route is clear")
                else:
                    st.error(f"⚠️ {dmg} damage report(s) near route")
                    for w in data.get("damage_warnings", [])[:3]:
                        st.markdown(f"- **{w['severity'].upper()}** {w['text']} ({w['distance_m']:.0f}m off)")

        with col3:
            # Thinking block
            if result.think_text:
                st.markdown('<div class="section-hdr">Gemma 4 Reasoning</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="think-block">{result.think_text[:600]}</div>', unsafe_allow_html=True)

            pois = result.result.get("nearest_pois") or result.result.get("pois", [])
            if pois:
                st.markdown('<div class="section-hdr">Results</div>', unsafe_allow_html=True)
                for p in pois[:6]:
                    detail = f"· {p['walk_minutes']:.0f} min" if "walk_minutes" in p else (f"· {p['distance_m']:.0f}m" if "distance_m" in p else "")
                    st.markdown(f'<div class="result-item"><span class="result-name">{p.get("name","Unnamed")}</span> <span class="result-detail">{detail}</span></div>', unsafe_allow_html=True)
                if len(pois) > 6:
                    st.caption(f"+{len(pois)-6} more")

            if st.checkbox("Show JSON"):
                st.json(result.result)

    # ── Damage Report Panel ──────────────────────────────────────────────────
    st.divider()
    with st.expander("🚧 Report Field Damage (Qdrant Memory)", expanded=False):
        st.caption("Submit infrastructure damage reports. They are embedded and stored in Qdrant so future route queries automatically avoid them.")
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        from damage_store import report_damage, list_damage_reports, delete_damage_report, get_collection_stats

        stats = get_collection_stats()
        qdrant_pts = stats.get("points", 0)
        st.markdown(f"**Qdrant collection:** `damage_reports` · {qdrant_pts} reports stored")

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            dmg_text = st.text_area("Damage description", placeholder="e.g. Road flooded after heavy rain, impassable", height=80, key="dmg_text")
        with dc2:
            dmg_lat = st.number_input("Latitude", value=float(loc["center"][0]), format="%.6f", key="dmg_lat")
            dmg_lon = st.number_input("Longitude", value=float(loc["center"][1]), format="%.6f", key="dmg_lon")
        with dc3:
            dmg_sev = st.selectbox("Severity", ["low", "medium", "high"], index=1, key="dmg_sev")
            dmg_rep = st.text_input("Reporter (optional)", key="dmg_rep")

        if st.button("Submit Damage Report", key="dmg_submit", disabled=not dmg_text):
            res = json.loads(report_damage(dmg_text, dmg_lat, dmg_lon, selected, dmg_sev, dmg_rep))
            st.success(f"✅ Stored in Qdrant (ID: `{res['id'][:8]}...`)")
            st.rerun()

        # List existing reports
        reports_json = json.loads(list_damage_reports(selected))
        if reports_json["count"] > 0:
            st.markdown(f"**Active reports for {loc['name'].split(',')[0]}:** {reports_json['count']}")
            for r in reports_json["reports"][:10]:
                rcol1, rcol2 = st.columns([5, 1])
                with rcol1:
                    sev_color = {"low": "🟡", "medium": "🟠", "high": "🔴"}.get(r.get("severity","medium"), "⚪")
                    st.markdown(f"{sev_color} **{r.get('severity','?').upper()}** — {r['text']} _(lat {r['lat']:.4f}, lon {r['lon']:.4f})_")
                with rcol2:
                    if st.button("🗑️", key=f"del_{r['id']}"):
                        delete_damage_report(r["id"])
                        st.rerun()


if __name__ == "__main__":
    main()
