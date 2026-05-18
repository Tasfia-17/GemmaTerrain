# 🌍 GemmaTerrain
**Multimodal GeoAI for the Unconnected - Gemma 4 on ARM Edge Hardware**

GemmaTerrain is a fully offline multimodal spatial query system that runs on ARM edge devices (Raspberry Pi 5, Steam Deck). It combines **Gemma 4 E2B/E4B** (quantized GGUF) with high-performance graph routing and spatial databases to answer natural language questions about geographic data - with image and audio input support - without any cloud connectivity.

Built for humanitarian scenarios where internet access is unreliable: refugee camp navigation, disaster response coordination, and field operations planning.


[![Gemma 4](https://img.shields.io/badge/Gemma%204-E2B%20%2F%20E4B-blue)](https://huggingface.co/google/gemma-4-e4b-it)
[![LoRA](https://img.shields.io/badge/LoRA-tasfuuu19%2Fgemma--4--e2b--spatial--lora-purple)](https://huggingface.co/tasfuuu19/gemma-4-e2b-spatial-lora)
[![ARM](https://img.shields.io/badge/ARM-Cortex--A76-orange)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Offline](https://img.shields.io/badge/Offline-100%25-green)](https://github.com/ggerganov/llama.cpp)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow)](LICENSE)

---

## 📑 Table of Contents
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Model Routing](#-model-routing-cactus)
- [Pre-Built Datasets](#-pre-built-datasets)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running GemmaTerrain](#-running-gemmaterrain)
- [Recommended Queries](#-recommended-queries)
- [Spatial Tools](#-spatial-tools)
- [Fine-Tuned LoRA](#-fine-tuned-lora)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Acknowledgments](#-acknowledgments)
- [License](#-license)

---

## 🎯 Key Features

- **100% Offline** - All AI inference runs locally via llama.cpp - no cloud, no API keys
- **Dual-Model Routing** - Gemma 4 E2B for fast simple lookups, E4B for multimodal and complex queries
- **Multimodal Input** - Upload a photo of a damaged road and ask "Is this passable?" alongside spatial queries
- **Natural Language Queries** - Ask in English, Bangla, Spanish, or Indonesian
- **Real Routing** - Actual walking routes on road network graphs via NetworKit Dijkstra (not straight-line)
- **Three Disaster Response Scenarios** - Pre-built datasets for Cox's Bazar, San Juan, and Jakarta
- **Thinking Mode** - Gemma 4 `<think>` blocks visible in the UI for complex reasoning
- **Battery-Aware** - Routes to E2B automatically when battery < 20%
- **Fine-Tuned** - Custom LoRA adapter trained on humanitarian spatial query → tool call pairs

---

## 🏗️ Architecture

```
User Query (text / image / audio)
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Geocoding Layer                                            │
│  "Camp 6" → (lat 21.200000, lon 92.160000)                  │
│  Regex-replaces known place names before LLM sees query     │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Cactus Model Router                                        │
│  Low battery → E2B  |  Image/audio → E4B                    │
│  Complex query → E4B  |  Simple lookup → E2B                │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Gemma 4 E2B (port 8080) or E4B (port 8081)                 │
│  via llama-server OpenAI-compatible API                     │
│  Native function calling + thinking mode + multimodal       │
└─────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  Spatial Tools (6 functions)                                │
│  list_pois │ find_nearest_poi_with_route │ calculate_route  │
│  find_along_route │ generate_isochrone │ geocode_place      │
└─────────────────────────────────────────────────────────────┘
        │
        ┌──────────────┴──────────────┐
        ▼                             ▼
┌──────────────────┐    ┌──────────────────────────────────┐
│  DuckDB Spatial  │    │  NetworKit Graph Engine          │
│  POI queries     │    │  Dijkstra routing on OSM network │
│  R-tree indexed  │    │  ~25K–208K nodes per city        │
└──────────────────┘    └──────────────────────────────────┘
```

---

## 🔀 Model Routing (Cactus)

| Condition | Model | Reason |
|---|---|---|
| Battery < 20% | Gemma 4 E2B | Power efficiency |
| Image or audio input | Gemma 4 E4B | Multimodal required |
| Complex multi-hop query | Gemma 4 E4B | Reasoning depth |
| Simple POI / route lookup | Gemma 4 E2B | Speed (~15 tok/s on Pi 5) |

**Complex query signals:** `along the route`, `isochrone`, `walkable area`, `damaged`, `flooded`, `impassable`, `which camps`, `not within`, `and then`...

---

## 📍 Pre-Built Datasets

Three disaster response scenarios with pre-built offline data:

| Location | Context | Nodes | Edges | POIs | Places |
|---|---|---|---|---|---|
| `coxs_bazar` | Rohingya refugee camps, Bangladesh | 27,551 | 71,530 | 6,509 | 464 |
| `san_juan` | Hurricane response, Puerto Rico | 24,602 | 61,055 | 11,351 | 405 |
| `jakarta` | Urban flood response, Indonesia | 208,281 | 508,954 | 41,028 | 331 |

Data sourced from OpenStreetMap via OSMnx. Includes hospitals, clinics, pharmacies, schools, shelters, banks, markets, fuel stations, police stations, and places of worship.

Build your own location:
```bash
python build_location.py "Dhaka, Bangladesh" dhaka
python build_location.py "Mandalay, Myanmar" mandalay --tiles
```

---

## 📋 Requirements

### Hardware
- Raspberry Pi 5 (8GB+ RAM recommended) or any ARM64/x86 device
- ~12GB storage for models + data
- Tested on: Pi 5 16GB (DietPi), M3 MacBook Air, Kaggle T4

### Models (download separately)
```
models/
├── google_gemma-4-E2B-it-Q4_K_M.gguf      (~3.5 GB)
├── google_gemma-4-E4B-it-Q4_K_M.gguf      (~5.4 GB)
├── mmproj-google_gemma-4-E2B-it-f16.gguf  (~986 MB)  ← for multimodal
└── mmproj-google_gemma-4-E4B-it-f16.gguf  (~990 MB)  ← for multimodal
```

### Software Prerequisites
```bash
sudo apt update
sudo apt install -y build-essential cmake git python3 python3-venv libopenblas-dev

# uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Tasfia-17/GemmaTerrain.git
cd GemmaTerrain
```

### 2. Install Python Dependencies
```bash
uv sync
```

### 3. Build llama.cpp (ARM-Optimized)
```bash
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp && mkdir build && cd build

# Raspberry Pi 5 (Cortex-A76)
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-mcpu=cortex-a76 -O3 -ffast-math -fno-finite-math-only" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-a76 -O3 -ffast-math -fno-finite-math-only" \
    -DGGML_NATIVE=ON \
    -DGGML_LTO=ON \
    -DLLAMA_CURL=OFF

cmake --build . -j4 --config Release
cd ../..
```

### 4. Download Models
```bash
mkdir -p models
# Download from HuggingFace - see INSTALL.md for direct links
```

### 5. Verify
```bash
ls -la llama.cpp/build/bin/llama-server
ls -la models/*.gguf
ls -la data/
```

---

## 🎮 Running GemmaTerrain

### Start Model Servers

**Terminal 1 - E2B (simple queries, ~3.5GB)**
```bash
./llama.cpp/build/bin/llama-server \
    -m ./models/google_gemma-4-E2B-it-Q4_K_M.gguf \
    --mmproj ./models/mmproj-google_gemma-4-E2B-it-f16.gguf \
    -c 2048 -t 4 --mlock \
    --host 0.0.0.0 --port 8080
```

**Terminal 2 - E4B (complex/multimodal queries, ~5.4GB)**
```bash
./llama.cpp/build/bin/llama-server \
    -m ./models/google_gemma-4-E4B-it-Q4_K_M.gguf \
    --mmproj ./models/mmproj-google_gemma-4-E4B-it-f16.gguf \
    -c 2048 -t 4 --mlock \
    --host 0.0.0.0 --port 8081
```

### Command Line Interface
```bash
# Basic query
python gemmaterrain.py -l coxs_bazar "Find nearest hospital to Camp 6"

# With image (multimodal)
python gemmaterrain.py -l san_juan --image road.jpg "Is this road passable?"

# Force specific model
python gemmaterrain.py -l jakarta --model e4b "What can I reach in 15 minutes from Gelora?"

# Simulate low battery (forces E2B)
python gemmaterrain.py -l coxs_bazar --battery 15 "Find nearest clinic"

# List locations
python gemmaterrain.py --list

# Health check
python gemmaterrain.py --health

# JSON output
python gemmaterrain.py -l coxs_bazar --json "Find nearest hospital to Camp 6"
```

### Streamlit Dashboard
```bash
uv run streamlit run app.py --server.port 8501
```
Open `http://[device-ip]:8501` in your browser.

---

## 💬 Recommended Queries

### Cox's Bazar, Bangladesh (Rohingya Refugee Camps)
```bash
python gemmaterrain.py -l coxs_bazar "Find the nearest hospital to Camp 6"
python gemmaterrain.py -l coxs_bazar "How do I walk from Camp 3 to Camp 8W?"
python gemmaterrain.py -l coxs_bazar "Show me everywhere I can walk to in 15 minutes from Camp 8W"
python gemmaterrain.py -l coxs_bazar "Clinics within 2km of Camp 10"
python gemmaterrain.py -l coxs_bazar "Pharmacies along the route from Camp 3 to Camp 8W"
# Bangla transliterated
python gemmaterrain.py -l coxs_bazar "Camp 6 er kache hospital kothay?"
python gemmaterrain.py -l coxs_bazar "Camp 8W theke 15 minute hatar jaiga"
```

### San Juan, Puerto Rico (Hurricane Response)
```bash
python gemmaterrain.py -l san_juan "Where is the closest hospital to Condado?"
python gemmaterrain.py -l san_juan "Walking route from Santurce to Miramar"
python gemmaterrain.py -l san_juan "Show me a 20 minute walking radius from Condado"
# Spanish
python gemmaterrain.py -l san_juan "¿Dónde está el hospital más cercano a Condado?"
python gemmaterrain.py -l san_juan "Farmacias dentro de 1km de Ocean Park"
```

### Jakarta, Indonesia (Urban Flood Response)
```bash
python gemmaterrain.py -l jakarta "Find nearest hospital to Menteng"
python gemmaterrain.py -l jakarta "What can I reach in 15 minutes from Gelora?"
python gemmaterrain.py -l jakarta "How far to walk from Gambir to Kemang?"
# Indonesian
python gemmaterrain.py -l jakarta "Rumah sakit terdekat dari Menteng"
python gemmaterrain.py -l jakarta "Apotek dalam radius 1km dari Gelora"
```

---

## 🔧 Spatial Tools

| Tool | Description | Example |
|---|---|---|
| `list_pois` | List POIs of a type within radius | "Clinics within 2km of Camp 6" |
| `find_nearest_poi_with_route` | Nearest POI + walking route | "Find nearest hospital to Condado" |
| `calculate_route` | Walking route between two points | "Walk from Camp 3 to Camp 9" |
| `find_along_route` | POIs along a walking path | "Pharmacies along route from A to B" |
| `generate_isochrone` | Walkable area from a point in N minutes | "15 minute walking radius from Camp 6" |
| `geocode_place` | Place name → coordinates | "Where is Camp 8W?" |

**Supported POI types:** `hospital`, `clinic`, `doctors`, `pharmacy`, `police`, `fire_station`, `shelter`, `school`, `university`, `bank`, `atm`, `supermarket`, `marketplace`, `drinking_water`, `water_point`, `fuel`, `bus_station`, `place_of_worship`

---

## 🤖 Fine-Tuned LoRA

A custom LoRA adapter is available for the E2B model, fine-tuned on humanitarian spatial query → tool call pairs:

**[tasfuuu19/gemma-4-e2b-spatial-lora](https://huggingface.co/tasfuuu19/gemma-4-e2b-spatial-lora)**

- 28 unique examples across 3 locations and 4 languages, augmented to 5,000
- QLoRA (r=16, alpha=32) with Unsloth, 3 epochs on Kaggle T4
- Languages: English, Bangla (transliterated), Spanish, Indonesian

Fine-tune your own adapter:
```bash
# Install fine-tuning deps
pip install unsloth torch transformers datasets trl

# Train
python src/finetune.py --model e2b --output ./output/lora_e2b
python src/finetune.py --model e4b --output ./output/lora_e4b --push-to-hub your-hf-id/gemma-4-e4b-spatial-lora
```

---

## 📁 Project Structure

```
GemmaTerrain/
├── gemmaterrain.py        # CLI entry point
├── app.py                 # Streamlit dashboard
├── build_location.py      # OSM dataset builder
├── benchmark.py           # Accuracy + latency benchmark suite
├── pyproject.toml         # Python dependencies (uv)
├── uv.lock                # Locked dependency tree
├── README.md
├── INSTALL.md             # Full installation instructions
├── WRITEUP.md             # Technical deep-dive and humanitarian context
├── src/
│   ├── engine.py          # Query orchestration pipeline
│   ├── gemma_client.py    # llama-server HTTP client (multimodal)
│   ├── router.py          # Cactus dual-model router
│   ├── spatial_tools.py   # 6 spatial tools + tool registry
│   ├── geocode_layer.py   # Pre-LLM place name → coordinates
│   └── finetune.py        # Unsloth QLoRA fine-tuning pipeline
├── data/
│   ├── coxs_bazar/        # Cox's Bazar dataset (27K nodes, 6K POIs)
│   ├── san_juan/          # San Juan dataset (24K nodes, 11K POIs)
│   └── jakarta/           # Jakarta dataset (208K nodes, 41K POIs)
└── models/                # GGUF models (gitignored, ~11GB total)
```

---

## 🛠️ Troubleshooting

**LLM server won't start**
```bash
lsof -i :8080              # Check if port is in use
pkill -f llama-server      # Kill existing process
```

**Out of memory on Pi 5**
```bash
# Reduce context size
./llama.cpp/build/bin/llama-server \
    -m ./models/google_gemma-4-E2B-it-Q4_K_M.gguf \
    -c 1024 -t 4 --host 0.0.0.0 --port 8080
```

**Slow queries**
```bash
# Set CPU governor to performance
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

**No location data**
```bash
python build_location.py "Cox's Bazar, Bangladesh" coxs_bazar
```

---

## 🙏 Acknowledgments

- [Gemma 4](https://huggingface.co/google/gemma-4-e4b-it) by Google DeepMind - Multimodal LLM with native function calling
- [llama.cpp](https://github.com/ggerganov/llama.cpp) by Georgi Gerganov - Efficient ARM inference engine
- [Unsloth](https://github.com/unslothai/unsloth) - 2× faster QLoRA fine-tuning
- [NetworKit](https://networkit.github.io/) - High-performance graph algorithms
- [DuckDB](https://duckdb.org/) - Embedded analytical database with spatial extension
- [OSMnx](https://github.com/gboeing/osmnx) by Geoff Boeing - Street network retrieval
- [OpenStreetMap](https://www.openstreetmap.org/) - Geographic data

---

## 📄 License

Apache 2.0 - See [LICENSE](LICENSE) for details.
