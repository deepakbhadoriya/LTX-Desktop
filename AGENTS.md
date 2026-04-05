# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an Electron app for AI video generation using LTX models. Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export
- **Backend** (`backend/`): Python FastAPI server (port 8000) handling ML model orchestration and generation

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only |
| `pnpm backend:test` | Run Python pytest tests |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build` | Full platform build (auto-detects platform) |
| `pnpm setup:dev` | One-time dev environment setup (auto-detects platform) |

Run a single backend test file via pnpm: `pnpm backend:test -- tests/test_ic_lora.py`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*`
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`)
- **Backend calls**: Always use `backendFetch` from `frontend/lib/backend.ts` for app backend HTTP requests (it attaches auth/session details). Do not call `fetch` directly for backend endpoints.
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- **No frontend tests** currently exist

## Backend Architecture

Request flow: `_routes/* (thin) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock→read/validate→unlock→heavy work→lock→write. Never hold lock during heavy compute/IO.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode is also enforced as a test (`test_pyright.py`)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS

## Python Config

- Python 3.13+ (per `.python-version`), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`)
- Dependencies in `backend/pyproject.toml`

---

## MLX Studio — Fork-Specific Guidelines

> This is a fork of LTX Desktop. Goal: Replace CUDA backend with MLX for native Apple Silicon local generation.

### What We Change vs What Stays

| Layer | Modify? | Reason |
|-------|---------|--------|
| `frontend/*` | **NO** | UI is device-agnostic |
| `electron/*` | **MINIMAL** | Only GPU detection in `gpu.ts` |
| `backend/_routes/*` | **NO** | Routes are thin plumbing |
| `backend/handlers/*` | **NO** | Business logic is device-agnostic |
| `backend/services/*` | **YES** | Replace CUDA pipeline implementations with MLX |
| `backend/runtime_config/*` | **YES** | Update model specs for MLX models |
| `backend/app_handler.py` | **YES** | Update `build_default_service_bundle()` |

### New MLX Service Files to Create

```
backend/services/
├── fast_video_pipeline/
│   └── mlx_video_pipeline.py      # Replaces LTXFastVideoPipeline
├── gpu_cleaner/
│   └── mlx_cleaner.py             # Replaces TorchCleaner
├── gpu_info/
│   └── gpu_info_impl.py           # Extend with MLX detection (modify existing)
├── text_encoder/
│   └── mlx_text_encoder.py        # Replaces LTXTextEncoder
├── a2v_pipeline/
│   └── mlx_a2v_pipeline.py        # Replaces LTXa2vPipeline
├── retake_pipeline/
│   └── mlx_retake_pipeline.py     # Replaces LTXRetakePipeline
├── ic_lora_pipeline/
│   └── mlx_ic_lora_pipeline.py    # Replaces LTXIcLoraPipeline
└── image_generation_pipeline/
    └── mlx_image_pipeline.py      # Replaces ZitImageGenerationPipeline (uses mflux)
```

### MLX Implementation Rules

1. **Protocol compliance**: Every new service MUST implement the same Protocol from `interfaces.py`
2. **Lazy imports**: Import `mlx`, `mlx_video`, `mflux` inside `__init__` or method body, NEVER at module top level
3. **Same API contract**: Frontend/handlers don't know the backend changed — same request/response schemas
4. **Device abstraction**: Extend `services_utils.py` with MLX device type support
5. **Memory monitoring**: Use `psutil.virtual_memory()` for Apple Silicon unified memory reporting
6. **No torch dependency**: MLX pipelines must NOT import torch (remove CUDA/MPS code paths)
7. **Seed handling**: Use `mlx.core.random.seed()` instead of `torch.Generator`
8. **Cleanup**: Use `gc.collect()` for MLX memory cleanup (no `torch.cuda.empty_cache()`)

### ServiceBundle Wiring

All service swaps happen in ONE place — `app_handler.py::build_default_service_bundle()`:

```python
# Before (CUDA)
ServiceBundle(
    fast_video_pipeline_class=LTXFastVideoPipeline,
    gpu_cleaner=TorchCleaner(device=config.device),
    ...
)

# After (MLX)
ServiceBundle(
    fast_video_pipeline_class=MLXVideoPipeline,
    gpu_cleaner=MLXCleaner(),
    ...
)
```

### Testing New MLX Services

1. Create `FakeMLXVideoPipeline` in `tests/fakes/services.py`
2. Wire fake in `conftest.py` via `ServiceBundle`
3. Run existing tests — they should pass with fakes (no real MLX needed)
4. Add MLX-specific integration tests for new edge cases

### Model Download Specs (MLX)

Update `runtime_config/model_download_specs.py`:
- LTX 2.3 Distilled Q4 MLX (~19GB) — fast preview
- LTX 2.3 Dev Q5 MLX (~25GB) — high quality
- Flux Schnell MLX (~8GB) — image generation (via mflux)

### Environment Variables for MLX

```bash
MLX_GPU_MEMORY_FRACTION=0.9    # Use 90% of unified memory
PYTORCH_ENABLE_MPS_FALLBACK=1  # Keep for any remaining torch ops
```

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`
