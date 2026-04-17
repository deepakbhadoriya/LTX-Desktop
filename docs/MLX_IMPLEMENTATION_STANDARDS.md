# MLX Studio — Implementation Standards & Coding Guide

> Reference document for implementing MLX Apple Silicon support in the LTX Desktop fork.
> Read `backend/architecture.md` first for the base architecture.
> Read `CLAUDE.md` for quick-reference rules.

---

## Table of Contents

1. [Python Backend Standards](#python-backend-standards)
2. [Frontend Standards](#frontend-standards)
3. [Electron Standards](#electron-standards)
4. [MLX-Specific Rules](#mlx-specific-rules)
5. [File Creation Checklist](#file-creation-checklist)
6. [Testing Standards](#testing-standards)
7. [Common Patterns Reference](#common-patterns-reference)

---

## Python Backend Standards

### Import Organization

```python
"""Module docstring describing purpose."""

from __future__ import annotations

# 1. Standard library (alphabetical within group)
import gc
import logging
import os
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Final, Protocol

# 2. Third-party (alphabetical)
from pydantic import BaseModel, Field

# 3. Local imports (absolute from backend root, alphabetical)
from _routes._errors import HTTPError
from api_types import GenerateVideoRequest
from services.interfaces import FastVideoPipeline

# 4. TYPE_CHECKING guard for circular deps only
if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)
```

**Rules:**
- Always use `from __future__ import annotations` (PEP 563)
- Absolute imports from repo root — never relative `../`
- Heavy ML imports (`mlx`, `mlx_video`, `mflux`) go inside `__init__` or method bodies, NOT at module top
- Module-level `logger = logging.getLogger(__name__)` — always

### Naming Conventions

| What | Convention | Example |
|------|-----------|---------|
| **Protocol interfaces** | PascalCase, no prefix | `FastVideoPipeline`, `GpuInfo` |
| **Concrete implementations** | Specific name or `*Impl` | `MLXVideoPipeline`, `GpuInfoImpl` |
| **Test fakes** | `Fake*` prefix | `FakeGpuInfo`, `FakeVideoPipeline` |
| **Handlers** | `*Handler` | `VideoGenerationHandler` |
| **State types** | PascalCase frozen dataclass | `GenerationRunning`, `GenerationComplete` |
| **Functions/methods** | snake_case | `generate_video()`, `load_model()` |
| **Private methods** | `_` prefix | `_prepare_image()`, `_run_inference()` |
| **Module constants** | SCREAMING_SNAKE | `DEFAULT_MODELS_DIR`, `MAX_FRAMES` |
| **Type aliases** | PascalCase | `GenerationState`, `ModelFileType` |
| **Request models** | `*Request` | `GenerateVideoRequest` |
| **Response models** | `*Response` | `GenerateVideoResponse` |
| **DTO/TypedDict** | `*Payload` | `GpuTelemetryPayload` |
| **Structural types** | `*Like` | `HttpResponseLike`, `VideoCaptureLike` |

### Type Annotation Style

```python
# Use pipe operator (PEP 604) — NOT Union or Optional
value: str | None           # ✅ not Optional[str]
result: int | float         # ✅ not Union[int, float]

# Discriminated unions for state machines
GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled

# Pattern matching for exhaustive handling
match state:
    case GenerationRunning() as running:
        handle_running(running)
    case GenerationComplete(result=result):
        handle_complete(result)
    case _:
        raise ValueError("Unexpected state")

# Annotated for constraints
from typing import Annotated
from pydantic import StringConstraints
NonEmptyPrompt = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

# Protocol for interface definitions
class FastVideoPipeline(Protocol):
    @staticmethod
    def create(checkpoint_path: str, ...) -> "FastVideoPipeline": ...
    def generate(self, prompt: str, ...) -> None: ...
```

**Pyright strict mode is enforced** — all parameters and returns must be annotated. No implicit `Any`.

### Class Structure Order

```python
class MLXVideoPipeline:
    """One-line summary.

    Detailed description of what this class does and why.
    """

    # 1. Class-level constants
    pipeline_kind: Final = "fast"

    # 2. Static factory method
    @staticmethod
    def create(checkpoint_path: str, device: str) -> "MLXVideoPipeline":
        return MLXVideoPipeline(checkpoint_path=checkpoint_path, device=device)

    # 3. Constructor
    def __init__(self, checkpoint_path: str, device: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._device = device
        # Lazy import heavy deps inside __init__
        from mlx_video import VideoGenerator
        self._generator = VideoGenerator(checkpoint_path)

    # 4. Public interface methods (match Protocol)
    def generate(self, prompt: str, seed: int, ...) -> None:
        """Generate video from text/image input."""
        ...

    def warmup(self, output_path: str) -> None:
        """Warmup pipeline with dummy generation."""
        ...

    # 5. Private helper methods
    def _encode_output(self, frames: list, output_path: str) -> None:
        ...
```

### Error Handling

```python
# In handlers: raise HTTPError with exception chaining
try:
    result = pipeline.generate(prompt=req.prompt, seed=seed)
except SpecificMLXError as e:
    raise HTTPError(400, "User-readable error message") from e
except Exception as e:
    raise HTTPError(500, "Generation failed") from e

# Status codes:
# 4xx = client error → logged as warning, no traceback
# 5xx = server error → logged with full traceback
# Don't call logger.exception() in handlers — app_factory.py owns logging
```

### Logging

```python
logger = logging.getLogger(__name__)

# Info: expected operations
logger.info("Loading MLX model: %s", model_path)
logger.info("[%s] Generation complete: %.2fs", gen_id, elapsed)

# Warning: unusual but recoverable
logger.warning("MLX memory pressure at %.1f%%, reducing batch", mem_pct)

# Error: with traceback
logger.error("Generation %s failed", gen_id, exc_info=True)

# Use %-formatting (not f-strings) for lazy evaluation
logger.info("Step %d/%d complete", current, total)  # ✅
logger.info(f"Step {current}/{total} complete")      # ❌
```

### Docstrings

```python
def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
    """Generate video from text/image conditioning.

    Orchestrates model loading, text encoding, inference, and output.

    Args:
        req: Generation request with prompt, resolution, duration.

    Returns:
        Response with video_path on success, status on cancel.

    Raises:
        HTTPError: On validation or execution failure.
    """
```

### Thread Safety

```python
# Pattern: lock → read/validate → unlock → heavy work → lock → write

# Step 1: Read under lock
with self.lock:
    generation_id = self._make_generation_id()
    seed = self._resolve_seed(req.seed)
    if not self._models_available():
        raise HTTPError(400, "Models not downloaded")

# Step 2: Heavy work WITHOUT lock
output_path = self._run_inference(prompt=req.prompt, seed=seed)

# Step 3: Write under lock (re-check state hasn't changed)
with self.lock:
    if self._generation.is_generation_cancelled():
        output_path.unlink()
        raise RuntimeError("Cancelled during generation")
    self._generation.complete_generation(str(output_path))

# Or use the decorator for simple atomic mutations
@with_state_lock
def start_generation(self, generation_id: str) -> None:
    self.state.active_generation = GpuGeneration(...)
```

### Pydantic Models

```python
from pydantic import BaseModel, ConfigDict, Field

class GenerateVideoRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")  # Reject unknown fields

    prompt: NonEmptyPrompt
    resolution: Literal["540p", "720p", "1080p"]
    duration: int = Field(ge=6, le=120, default=10)
    fps: int = Field(ge=8, le=60, default=24)
    seed: int | None = None

    @field_validator("duration", mode="before")
    @classmethod
    def _clamp_duration(cls, value: Any) -> int:
        return max(6, min(120, int(value)))
```

---

## Frontend Standards

### Component Pattern

```typescript
interface ComponentProps {
  prop1: string
  prop2?: number
  onEvent?: (data: SomeType) => void
  disabled?: boolean
}

function Component({ prop1, prop2 = 10, onEvent, disabled = false }: ComponentProps) {
  // 1. State
  const [value, setValue] = useState<string>('')
  // 2. Refs
  const ref = useRef<HTMLInputElement>(null)
  // 3. Derived values
  const isValid = value.length > 0 && !disabled
  // 4. Effects
  useEffect(() => { return () => {} }, [deps])
  // 5. Callbacks
  const handleClick = useCallback(() => {}, [deps])
  // 6. Render
  return <div className="...">{/* JSX */}</div>
}

export { Component }
```

### State Management

- React Context only — **NO Redux, NO Zustand** for app state
- Memoize context values with `useMemo` to prevent re-renders
- Custom hooks validate provider presence with `throw new Error()`

### API Communication

```typescript
// ALWAYS use backendFetch — NEVER raw fetch for backend endpoints
import { backendFetch } from './backend'

// ApiClient uses OpenAPI-generated types for type safety
export class ApiClient {
  static generateVideo(body, init?) {
    return this.requestJson('/api/generate', 'post', this.buildJsonRequestInit(body, init))
  }
}
```

### Styling

```typescript
// Tailwind utility classes + cn() helper for conditionals
import { cn } from '@/lib/utils'
<div className={cn('bg-zinc-900 rounded-xl', isActive && 'ring-2 ring-blue-500')} />

// shadcn components via class-variance-authority (CVA)
const buttonVariants = cva('inline-flex items-center ...', {
  variants: { variant: { default: '...', outline: '...' } }
})
```

### Naming (Frontend)

| What | Convention | Example |
|------|-----------|---------|
| Component files | PascalCase.tsx | `SettingsPanel.tsx` |
| Hook files | kebab-case.ts | `use-generation.ts` |
| Utility files | kebab-case.ts | `api-client.ts` |
| Hooks | camelCase with `use` | `useGeneration` |
| Booleans | `is/has/should/can` | `isLoading`, `hasError` |
| Constants | SCREAMING_SNAKE | `MIN_DURATION` |

### TypeScript Config

- Strict mode: `noUnusedLocals`, `noUnusedParameters`
- Use `type` for unions/DTOs, `interface` for extensible shapes
- Type guards at boundaries with `unknown`
- `AbortController` for cancellable operations

---

## Electron Standards

### IPC Pattern

```typescript
// Main: register handlers
ipcMain.handle('check-gpu', handleCheckGpu)

// Preload: expose to renderer (CommonJS required)
contextBridge.exposeInMainWorld('electronAPI', {
  checkGpu: () => ipcRenderer.invoke('check-gpu'),
})

// Renderer: call through electronAPI
const gpuInfo = await window.electronAPI.checkGpu()
```

### Python Backend Lifecycle

- Auto-start on app launch via `electron/python-backend.ts`
- Health check polling to detect crashes
- Crash recovery with debounce (`CRASH_DEBOUNCE_MS = 10_000`)
- Auth token generated per session, passed via env var

---

## MLX-Specific Rules

### 1. No torch in MLX pipelines

```python
# ❌ WRONG — don't import torch in MLX service files
import torch
@torch.inference_mode()
def generate(self): ...

# ✅ CORRECT — use pure MLX
import mlx.core as mx
def generate(self):
    mx.eval(output)  # Force evaluation
```

### 2. Lazy imports for heavy ML dependencies

```python
# ❌ WRONG — top-level import slows all tests
import mlx_video
import mflux

class MLXVideoPipeline:
    ...

# ✅ CORRECT — import inside constructor
class MLXVideoPipeline:
    def __init__(self, checkpoint_path: str) -> None:
        from mlx_video import VideoGenerator
        self._generator = VideoGenerator(checkpoint_path)
```

### 3. Device abstraction

```python
# Extend services_utils.py for MLX support
def get_device_type(device: str) -> str:
    if "cuda" in str(device): return "cuda"
    if "mps" in str(device): return "mps"
    if "mlx" in str(device): return "mlx"
    return "cpu"

def empty_device_cache(device: str) -> None:
    dt = get_device_type(device)
    if dt == "cuda": torch.cuda.empty_cache()
    elif dt == "mps": torch.mps.empty_cache()
    elif dt == "mlx": gc.collect()
```

### 4. Memory management

```python
import psutil

def get_available_memory_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)

def get_total_memory_gb() -> float:
    return psutil.virtual_memory().total / (1024 ** 3)
```

### 5. Seed handling

```python
# Replace torch.Generator with MLX seeding
import mlx.core as mx
mx.random.seed(seed)
```

---

## File Creation Checklist

When creating a new MLX service implementation:

- [ ] File follows naming: `mlx_<service_name>.py`
- [ ] `from __future__ import annotations` at top
- [ ] Module docstring present
- [ ] Class implements the matching Protocol from `interfaces.py`
- [ ] Has `@staticmethod create()` factory method (if Protocol requires it)
- [ ] Heavy imports (`mlx`, `mlx_video`) are lazy (inside `__init__` or methods)
- [ ] No `torch` imports (unless absolutely necessary for compatibility)
- [ ] All methods have type annotations (Pyright strict)
- [ ] Docstrings on class and public methods
- [ ] Private methods prefixed with `_`
- [ ] Logger created: `logger = logging.getLogger(__name__)`
- [ ] Corresponding `Fake*` class created in `tests/fakes/services.py`
- [ ] Wired in `app_handler.py::build_default_service_bundle()`
- [ ] Existing tests still pass with fakes

---

## Testing Standards

### No Mocks — Fakes Only

```python
# ❌ WRONG
from unittest.mock import MagicMock, patch
@patch('services.gpu_info.GpuInfoImpl')
def test_something(mock_gpu): ...

# ✅ CORRECT — swap via ServiceBundle
class FakeGpuInfo:
    def get_gpu_info(self) -> GpuTelemetryPayload:
        return {"name": "Fake Apple M5", "vram": 65536, "vramUsed": 0}

# In conftest.py:
bundle = ServiceBundle(gpu_info=FakeGpuInfo(), ...)
```

### Integration Test Pattern

```python
def test_generate_video(client, create_fake_model_files):
    create_fake_model_files()  # Setup model stubs
    response = client.post("/api/generate", json={
        "prompt": "a cat walking",
        "resolution": "720p",
        "duration": 6,
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "complete"
    assert "video_path" in data
```

### Behavioral Fakes

```python
class FakeMLXVideoPipeline:
    """Behavioral fake that mimics real pipeline contract."""
    def __init__(self) -> None:
        self.generate_count = 0
        self.last_prompt: str | None = None

    @staticmethod
    def create(checkpoint_path: str, **kwargs) -> "FakeMLXVideoPipeline":
        return FakeMLXVideoPipeline()

    def generate(self, prompt: str, seed: int, **kwargs) -> None:
        self.generate_count += 1
        self.last_prompt = prompt
        # Create a dummy output file
        Path(kwargs.get("output_path", "/tmp/fake.mp4")).touch()
```

---

## Common Patterns Reference

### Service Bundle Wiring (app_handler.py)

```python
def build_default_service_bundle(config: RuntimeConfig) -> ServiceBundle:
    """Lazy-import real implementations for runtime."""
    # Lazy imports to keep tests fast
    from services.fast_video_pipeline.mlx_video_pipeline import MLXVideoPipeline
    from services.gpu_cleaner.mlx_cleaner import MLXCleaner
    from services.gpu_info.gpu_info_impl import GpuInfoImpl

    return ServiceBundle(
        gpu_cleaner=MLXCleaner(),
        gpu_info=GpuInfoImpl(),
        fast_video_pipeline_class=MLXVideoPipeline,
        # ... other services
    )
```

### Route → Handler → Service Flow

```python
# _routes/generation.py (thin)
@router.post("/api/generate")
def route_generate(req: GenerateVideoRequest, handler = Depends(get_state_service)):
    return handler.video_generation.generate(req)

# handlers/video_generation_handler.py (logic)
def generate(self, req: GenerateVideoRequest) -> GenerateVideoResponse:
    with self.lock:
        self._generation.start_generation(gen_id)
    output = self._pipelines.run_fast_video(req)  # Heavy work, no lock
    with self.lock:
        self._generation.complete_generation(output)
    return GenerateVideoCompleteResponse(status="complete", video_path=output)

# services/fast_video_pipeline/mlx_video_pipeline.py (side effect)
def generate(self, prompt, seed, height, width, ...):
    from mlx_video import generate_av
    generate_av(prompt=prompt, seed=seed, ...)
```

### State Machine Transitions

```python
@dataclass(frozen=True)
class GenerationRunning:
    id: str
    progress: GenerationProgress

@dataclass(frozen=True)
class GenerationComplete:
    id: str
    result: str | list[str]

GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled

# Exhaustive matching
match self.state.active_generation:
    case GpuGeneration(state=GenerationRunning() as running):
        return {"status": "running", "progress": running.progress}
    case GpuGeneration(state=GenerationComplete(result=path)):
        return {"status": "complete", "video_path": path}
    case _:
        return {"status": "idle"}
```
