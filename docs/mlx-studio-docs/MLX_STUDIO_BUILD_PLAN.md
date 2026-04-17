# MLX Studio — LTX Desktop Fork for Mac with Local MLX Generation

## Context

LTX Desktop (https://github.com/Lightricks/LTX-Desktop) is Lightricks' official open-source AI video production suite. It has a full NLE timeline, T2V/I2V/A2V generation, image gen, retake, gap fill, transitions, color correction, export — everything we need. It's built with **React/TypeScript + Python/FastAPI + Electron** (Apache 2.0 license).

**The problem:** LTX Desktop only runs local generation on Windows (NVIDIA CUDA). On Mac, it falls back to cloud API — no local generation.

**Our goal:** Fork LTX Desktop, replace the CUDA inference backend with **MLX (mlx-video)** for native Apple Silicon local generation. Rebrand as MLX Studio. Ship a Mac-native version that generates locally without any cloud dependency.

**Developer:** JS/TS developer, Mac M5 Pro 64GB RAM
**Timeline:** ~2-3 weeks to local Mac generation working

---

## What We Get For Free (From the Fork)

Everything in the `frontend/` directory — untouched:
- Full NLE timeline editor (trim, slip, slide, roll, ripple)
- Text-to-Video UI
- Image-to-Video UI
- Audio-to-Video UI
- Image generation UI (Z-Image Turbo)
- Retake (regenerate sections)
- Timeline AI (multiple takes per clip)
- Gap fill (context-aware)
- Text/Subtitles with SRT
- Timeline import/export (Premiere Pro, DaVinci, FCP)
- Color correction
- Transitions
- H.264 + ProRes export via FFmpeg
- Customizable keyboard shortcuts
- Storyboard-to-Timeline

## What We Build

Only the backend ML inference layer — replacing CUDA with MLX:

1. **MLX inference engine** — replace NVIDIA backend with mlx-video
2. **Model management for Mac** — download/cache LTX 2.3 GGUF/MLX models
3. **Apple Silicon optimization** — VRAM monitoring, memory management
4. **Mac packaging** — DMG instead of Windows installer
5. **(Optional) Electron → Tauri migration** — smaller footprint (phase 2)

---

## Architecture (Modified)

```
LTX Desktop (current)              MLX Studio (our fork)
=====================              ====================

Frontend (React/TS)                Frontend (React/TS)
  │ (keep 100% as-is)               │ (identical)
  ▼                                  ▼
Electron                           Electron (phase 1) → Tauri (phase 2)
  │                                  │
  ▼                                  ▼
Python FastAPI (:8000)             Python FastAPI (:8000)
  │                                  │
  ▼                                  ▼
CUDA/NVIDIA inference    ──→       MLX inference (mlx-video)
  │                                  │
  ▼                                  ▼
GPU (NVIDIA RTX)                   Apple Silicon (M1-M5 GPU + ANE)
```

**Key principle:** Touch the frontend as little as possible. All changes happen in the Python backend. The FastAPI endpoints stay the same — only the inference implementation changes.

---

## Implementation Plan

### Phase 0: Setup & Exploration (Day 1-2)

#### 0.1 Fork and clone
```bash
git clone https://github.com/Lightricks/LTX-Desktop.git mlx-studio
cd mlx-studio
git remote rename origin upstream
git remote add origin <your-github-repo>
```

#### 0.2 Understand the codebase structure
```
mlx-studio/
├── frontend/           # React/TS — DO NOT MODIFY (phase 1)
│   ├── src/
│   │   ├── components/ # All UI components
│   │   ├── pages/      # App pages
│   │   ├── hooks/      # React hooks
│   │   ├── services/   # API client (calls localhost:8000)
│   │   └── stores/     # State management
│   └── package.json
│
├── electron/           # Electron main process
│   ├── main.ts         # Starts Python server, manages app
│   └── preload.ts      # Bridge between renderer and main
│
├── backend/            # Python FastAPI — THIS IS WHAT WE MODIFY
│   ├── server.py       # FastAPI app
│   ├── routes/         # API endpoints
│   ├── services/       # Business logic
│   ├── inference/      # ← CUDA code lives here (REPLACE THIS)
│   └── requirements.txt
│
├── shared/             # Shared types/constants
└── docs/
```

#### 0.3 Map the API contract
- Read every endpoint in `backend/routes/`
- Document the request/response schemas
- Identify which endpoints call CUDA inference
- These are the ONLY files we need to change

#### 0.4 Set up Mac dev environment
```bash
# Frontend
cd frontend && npm install

# Backend  
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Add MLX dependencies
pip install mlx>=0.20.0 mlx-video>=0.2.0 mflux>=0.9.0

# Electron
cd .. && npm install
```

---

### Phase 1: Replace CUDA Backend with MLX (Day 3-7)

#### 1.1 Create MLX inference engine
Replace the CUDA inference module with an MLX equivalent that implements the same interface.

**File:** `backend/inference/mlx_engine.py`

```python
# Pseudocode — actual implementation depends on existing interface
class MLXVideoEngine:
    """Drop-in replacement for CUDA engine using mlx-video"""
    
    def __init__(self):
        self.model = None
        self.model_name = None
    
    async def load_model(self, model_id: str):
        """Load LTX 2.3 model via mlx-video"""
        # Use mlx_video to load the model
        pass
    
    async def generate_video(self, request: GenerateRequest) -> GenerateResult:
        """Generate video — T2V, I2V, or A2V depending on request"""
        # Route to appropriate generation mode
        if request.mode == "t2v":
            return await self._text_to_video(request)
        elif request.mode == "i2v":
            return await self._image_to_video(request)
        elif request.mode == "a2v":
            return await self._audio_to_video(request)
    
    async def generate_image(self, request: ImageRequest) -> ImageResult:
        """Generate image using mflux (Flux)"""
        pass
    
    def get_progress(self) -> ProgressInfo:
        """Return current generation progress"""
        pass
    
    def get_vram_usage(self) -> VRAMInfo:
        """Return Apple Silicon memory usage via Metal API"""
        pass
```

#### 1.2 Replace inference imports
Find every file that imports the CUDA engine and swap to MLX:

```python
# Before (in routes or services)
from inference.cuda_engine import CUDAVideoEngine

# After
from inference.mlx_engine import MLXVideoEngine
```

#### 1.3 Update model management
- Change model download URLs to MLX-compatible models (GGUF/MLX format)
- Update model paths for macOS (`~/Library/Application Support/MLXStudio/models/`)
- Add model catalog:
  - LTX 2.3 Distilled Q4 (~19GB) — fast preview
  - LTX 2.3 Dev Q5 (~25GB) — high quality
  - Flux Schnell (~8GB) — image generation

#### 1.4 Update requirements.txt
```
# Remove CUDA-specific packages
# - torch+cuda
# - xformers
# - etc.

# Add MLX packages
mlx>=0.20.0
mlx-video>=0.2.0
mflux>=0.9.0
fastapi>=0.115.0
uvicorn>=0.34.0
websockets>=14.0
Pillow>=10.0
pydantic>=2.0
```

#### 1.5 Test basic T2V generation
- Start the backend: `python server.py`
- Start the frontend: `npm run dev`  
- Start Electron: `npm run electron:dev`
- Enter a prompt → click Generate → verify video generates via MLX
- Confirm progress updates flow to the UI via WebSocket

---

### Phase 2: Mac-Specific Optimizations (Day 8-10)

#### 2.1 Apple Silicon memory management
```python
# backend/services/mac_utils.py
import subprocess
import json

def get_metal_memory():
    """Get Apple Silicon GPU memory usage"""
    # Use Metal API via ctypes or subprocess
    # Return: { used_mb, total_mb, percent }
    pass

def get_optimal_settings(total_ram_gb: int):
    """Return recommended generation settings based on available RAM"""
    if total_ram_gb >= 64:
        return {"max_resolution": "1080p", "model": "q5", "batch_frames": 97}
    elif total_ram_gb >= 32:
        return {"max_resolution": "720p", "model": "q4", "batch_frames": 65}
    else:
        return {"max_resolution": "480p", "model": "q4", "batch_frames": 49}
```

#### 2.2 Progress streaming
Ensure mlx-video progress callbacks properly stream to the WebSocket:
- Stage updates (loading, encoding, sampling, decoding)
- Step-by-step percentage
- ETA estimation
- Live frame previews (encode intermediate frames as base64)

#### 2.3 Model auto-detection
On first launch:
- Detect Apple Silicon chip (M1/M2/M3/M4/M5)
- Detect total unified memory
- Recommend appropriate model variant
- Auto-download if user agrees

---

### Phase 3: Branding & Mac Packaging (Day 11-13)

#### 3.1 Rebrand
- App name: "MLX Studio"
- Update `electron/main.ts` — window title, app name
- Update `frontend/` — logo, about page, branding
- New app icon (can use AI-generated icon)
- Update `package.json` — name, description, author

#### 3.2 Mac DMG packaging
- Update Electron Builder config for macOS:
  - `.dmg` output with drag-to-Applications
  - Code signing (optional for dev, required for distribution)
  - Universal binary (arm64 for Apple Silicon)
- Update build scripts in `package.json`

#### 3.3 First-run experience
- Python environment setup (auto-detect or bundle)
- Model download wizard
- Hardware capability check
- Quick tutorial / sample prompt

---

### Phase 4: Test & Polish (Day 14-16)

#### 4.1 Feature testing matrix

| Feature | Test |
|---------|------|
| T2V | Enter prompt → generates video → plays in timeline |
| I2V | Upload image → generates video from it |
| A2V | Upload audio → generates matching video |
| Image gen | Enter prompt → generates image |
| Retake | Select clip section → regenerate → replaces in timeline |
| Timeline | Cut, trim, slip, slide, roll, ripple |
| Export | H.264 and ProRes export work |
| Import | XML from Premiere/DaVinci/FCP loads |
| Queue | Multiple generations queue properly |
| Progress | WebSocket progress shows in UI |
| Memory | VRAM monitoring accurate on Apple Silicon |

#### 4.2 Performance benchmarks
- T2V: 4-second clip at 720p — target < 10 min on M5 Pro
- I2V: Same with reference image
- Image gen: 1024x1024 — target < 15 sec on M5 Pro
- Memory usage during generation — stay under 90% of unified memory

#### 4.3 Error handling
- Model not downloaded → show download prompt
- Out of memory → suggest lower resolution/fewer frames
- Python not found → installation guide
- Generation failure → meaningful error message

---

### Phase 5: Release (Day 17-18)

#### 5.1 GitHub repository
- Create `github.com/<your-username>/mlx-studio`
- Add README with:
  - What it is (LTX Desktop for Mac with local MLX generation)
  - Screenshots/demo video
  - System requirements (Apple Silicon, macOS 14+, 16GB+ RAM)
  - Installation instructions
  - Feature list
  - Credits to Lightricks/LTX-Desktop
- Add CHANGELOG.md
- Add LICENSE (Apache 2.0, same as upstream)

#### 5.2 Release artifacts
- `.dmg` for macOS (arm64)
- GitHub Release with release notes
- Installation script for Python dependencies

#### 5.3 Documentation
- Getting started guide
- Supported models
- Keyboard shortcuts
- Troubleshooting (common Mac issues)

---

## Phase 2 Roadmap (Post-MVP)

After the initial release, extend beyond LTX Desktop's features:

### v1.1 — Enhanced Mac Experience (~2 weeks)
- [ ] Electron → Tauri migration (5MB vs 200MB app size)
- [ ] Native macOS menu bar integration
- [ ] Dock progress indicator
- [ ] Spotlight integration for searching generations

### v1.2 — Multi-Model Support (~2 weeks)
- [ ] Wan2.2 TI2V-5B support (text+image to video)
- [ ] Wan2.2 I2V-14B support
- [ ] Model switcher in UI
- [ ] Per-model parameter presets

### v1.3 — Advanced Features (~3 weeks)
- [ ] Multi-frame conditioning (first/mid/last frame)
- [ ] Character consistency tools
- [ ] Video-to-Video (style transfer)
- [ ] LoRA support for LTX 2.3
- [ ] Batch generation with variations

### v1.4 — Image Studio (~2 weeks)
- [ ] Enhanced Flux integration (Dev + Schnell)
- [ ] LoRA browser + loader
- [ ] ControlNet support
- [ ] Inpainting / outpainting

---

## Key Files to Modify

| File | Change | Priority |
|------|--------|----------|
| `backend/inference/*` | Replace CUDA → MLX | Critical |
| `backend/requirements.txt` | Remove CUDA, add MLX deps | Critical |
| `backend/services/model_manager.py` | Mac model paths + MLX models | Critical |
| `electron/main.ts` | Mac-specific Python startup | High |
| `package.json` | Branding, Mac build config | High |
| `frontend/src/components/Header.tsx` | Logo/branding | Medium |
| `frontend/src/pages/Settings.tsx` | Mac-specific settings | Medium |
| `docs/*` | Mac installation guide | Medium |

**Do NOT modify:**
- `frontend/src/components/timeline/*` — full NLE, works as-is
- `frontend/src/components/editor/*` — editing tools, works as-is
- `frontend/src/components/export/*` — export pipeline, works as-is
- Any UI component that doesn't touch inference directly

---

## Verification

1. **Fork & build**: Clone → install deps → app launches on Mac
2. **T2V works**: Enter prompt → video generates locally via MLX → appears in timeline
3. **I2V works**: Upload image → generates video from it
4. **Timeline works**: Cut, arrange, transition — all NLE features functional
5. **Export works**: H.264 + ProRes export produces valid video files
6. **Performance**: 4-sec 720p video generates in < 10 min on M5 Pro 64GB
7. **Packaging**: `npm run build` produces working `.dmg`
8. **Memory**: App + model fit in 64GB unified memory comfortably

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| mlx-video API doesn't match LTX Desktop's interface | Write adapter layer, minimal changes |
| Some frontend features depend on CUDA-specific outputs | Test each feature, stub if needed |
| Electron Python startup differs on Mac vs Windows | Study ltx-video-mac's approach for reference |
| Model download slow (19-48GB) | Show progress, support resume, offer smaller model first |
| Memory pressure on 16GB Macs | Detect RAM, recommend appropriate model/settings |
