## Installation

### Raspberry Pi 5 (Primary Target)

**1. System preparation**
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y git cmake build-essential python3-venv libopenblas-dev
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.bashrc
```

**2. Clone and install**
```bash
git clone https://github.com/your-org/gemmaterrain.git
cd gemmaterrain
uv sync
```

**3. Build llama.cpp with Cortex-A76 optimizations**
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-mcpu=cortex-a76+dotprod -O3 -ffast-math -fno-finite-math-only" \
    -DCMAKE_CXX_FLAGS="-mcpu=cortex-a76+dotprod -O3 -ffast-math -fno-finite-math-only" \
    -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_ARM_DOTPROD=ON -DLLAMA_CURL=OFF

cmake --build . -j4 --config Release
cd ../..
```

**4. Download Gemma 4 models**
```bash
mkdir -p models

# Gemma 4 E2B — ~1.2GB (port 8080, fast routing)
uv run --with huggingface-hub hf download \
    bartowski/google_gemma-4-E2B-it-GGUF \
    gemma-4-E2B-it-Q4_K_M.gguf \
    --local-dir ./models

# Gemma 4 E4B — ~2.3GB (port 8081, multimodal + complex queries)
uv run --with huggingface-hub hf download \
    bartowski/google_gemma-4-E4B-it-GGUF \
    gemma-4-E4B-it-Q4_K_M.gguf \
    --local-dir ./models
```

**5. Start model servers**

Terminal 1 — E2B:
```bash
./llama.cpp/build/bin/llama-server \
    -m models/gemma-4-E2B-it-Q4_K_M.gguf \
    -c 4096 -t 4 --mlock --host 0.0.0.0 --port 8080
```

Terminal 2 — E4B:
```bash
./llama.cpp/build/bin/llama-server \
    -m models/gemma-4-E4B-it-Q4_K_M.gguf \
    -c 4096 -t 4 --mlock --host 0.0.0.0 --port 8081
```

**6. Run**
```bash
# CLI
python gemmaterrain.py -l coxs_bazar "Find nearest hospital to Camp 6"

# Web dashboard
uv run streamlit run app.py --server.port 8501
```

---

### Apple Silicon (macOS) — Development

```bash
git clone https://github.com/your-org/gemmaterrain.git
cd gemmaterrain && uv sync

git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON
cmake --build . -j --config Release
cd ../..

# Download models (same as above)
# Start servers on ports 8080 and 8081 (same commands, Metal accelerates automatically)
```

---

### Memory-Constrained Setup (Pi 5 4GB or 8GB)

If running both models simultaneously is tight, run only E4B and set E2B URL to the same port:

```bash
# Edit src/gemma_client.py — point both MODEL_URLS to port 8081
# Then run only the E4B server
```

Or run models sequentially (swap on demand) — the router will still select the right model, but both calls go to the same server.
