# Switch to dgrauet/ltx-2-mlx Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Blaizzy/mlx-video inference library with dgrauet/ltx-2-mlx to enable Q8 quantized model support, fitting LTX 2.3 comfortably on 64 GB Apple Silicon Macs.

**Architecture:** Each MLX pipeline adapter wraps a persistent dgrauet pipeline class with `low_memory=True`. Pipelines lazy-load on first `generate()` call and persist across requests. All dgrauet imports are lazy (inside `__init__` or methods) per CLAUDE.md rules.

**Tech Stack:** Python 3.13, ltx-core-mlx, ltx-pipelines-mlx, MLX 0.31+, FastAPI

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/pyproject.toml` | Modify | Swap mlx-video dep for ltx-core-mlx + ltx-pipelines-mlx |
| `backend/runtime_config/model_download_specs.py` | Modify | Point to dgrauet/ltx-2.3-mlx-q8 repo |
| `backend/services/fast_video_pipeline/mlx_video_pipeline.py` | Rewrite | T2V/I2V via ImageToVideoPipeline |
| `backend/services/a2v_pipeline/mlx_a2v_pipeline.py` | Rewrite | A2V via AudioToVideoPipeline |
| `backend/services/retake_pipeline/mlx_retake_pipeline.py` | Rewrite | Retake via RetakePipeline |
| `backend/services/ic_lora_pipeline/mlx_ic_lora_pipeline.py` | Rewrite | IC-LoRA via ICLoraPipeline |

**No changes needed:** `app_handler.py`, Protocol files, fakes, tests, `mlx_text_encoder.py`, `safetensors_metadata_fix.py`

---

### Task 1: Update pyproject.toml Dependencies

**Files:**
- Modify: `backend/pyproject.toml:34-36,44-54`

- [ ] **Step 1: Replace mlx-video with ltx-core-mlx and ltx-pipelines-mlx**

In `backend/pyproject.toml`, replace the MLX dependencies block (lines 34-36):

```python
# OLD:
    "mlx>=0.20.0; sys_platform == 'darwin'",
    "mlx-video; sys_platform == 'darwin'",
    "mflux>=0.16.0; sys_platform == 'darwin'",

# NEW:
    "mlx>=0.31.0; sys_platform == 'darwin'",
    "ltx-core-mlx; sys_platform == 'darwin'",
    "ltx-pipelines-mlx; sys_platform == 'darwin'",
    "mflux>=0.16.0; sys_platform == 'darwin'",
```

In the `[tool.uv.sources]` section, replace the mlx-video source (line 54):

```toml
# OLD:
mlx-video = { git = "https://github.com/Blaizzy/mlx-video.git" }

# NEW:
ltx-core-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx.git", subdirectory = "packages/ltx-core-mlx" }
ltx-pipelines-mlx = { git = "https://github.com/dgrauet/ltx-2-mlx.git", subdirectory = "packages/ltx-pipelines-mlx" }
```

- [ ] **Step 2: Install new dependencies**

Run:
```bash
cd backend && uv sync
```

Expected: dependencies resolve and install successfully. `ltx_pipelines_mlx` module becomes importable.

- [ ] **Step 3: Verify import**

Run:
```bash
cd backend && uv run python -c "from ltx_pipelines_mlx import TextToVideoPipeline; print('OK')"
```

Expected: prints `OK`

- [ ] **Step 4: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock
git commit -m "deps: switch from mlx-video to ltx-core-mlx + ltx-pipelines-mlx"
```

---

### Task 2: Update Model Download Specs for Q8

**Files:**
- Modify: `backend/runtime_config/model_download_specs.py:113-151`

- [ ] **Step 1: Update MLX_MODEL_DOWNLOAD_SPECS checkpoint to Q8 repo**

In `backend/runtime_config/model_download_specs.py`, replace the `MLX_MODEL_DOWNLOAD_SPECS` dict (lines 113-151):

```python
MLX_MODEL_DOWNLOAD_SPECS: dict[ModelFileType, ModelFileDownloadSpec] = {
    "checkpoint": ModelFileDownloadSpec(
        relative_path=Path("mlx-ltx-2.3-q8"),
        expected_size_bytes=59_000_000_000,
        is_folder=True,
        repo_id="dgrauet/ltx-2.3-mlx-q8",
        description="LTX 2.3 (MLX Q8 quantized)",
    ),
    "upsampler": ModelFileDownloadSpec(
        relative_path=Path("spatial_upscaler_x2_v1_1.safetensors"),
        expected_size_bytes=996_000_000,
        is_folder=False,
        repo_id="dgrauet/ltx-2.3-mlx-q8",
        description="2x Spatial Upscaler (MLX)",
    ),
    "text_encoder": ModelFileDownloadSpec(
        relative_path=Path("mlx-gemma-3-12b-it-q4"),
        expected_size_bytes=7_000_000_000,
        is_folder=True,
        repo_id="mlx-community/gemma-3-12b-it-4bit",
        description="Gemma text encoder (MLX 4-bit)",
    ),
    "zit": ModelFileDownloadSpec(
        relative_path=Path("flux2-klein-4b-mlx-4bit"),
        expected_size_bytes=4_610_000_000,
        is_folder=True,
        repo_id="themindstudio/flux2-klein-4b-mlx-4bit",
        description="Flux.2 Klein 4B (mflux 4-bit)",
        ignore_patterns=("*.md", "*.png", "*.jpg", ".gitattributes", ".DS_Store"),
    ),
    "ic_lora": ModelFileDownloadSpec(
        relative_path=Path("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"),
        expected_size_bytes=654_465_352,
        is_folder=False,
        repo_id="Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control",
        description="Union IC-LoRA control model",
    ),
}
```

Key changes from the old spec:
- `repo_id` changed from `dgrauet/ltx-2.3-mlx` to `dgrauet/ltx-2.3-mlx-q8`
- `relative_path` changed from `mlx-ltx-2.3` to `mlx-ltx-2.3-q8` (distinct folder name)
- `expected_size_bytes` updated to ~59 GB (full repo including both transformers)
- Removed `ignore_patterns` for `transformer-dev.safetensors` (needed for A2V and Retake pipelines)
- `zit`, `text_encoder`, `ic_lora` specs unchanged

- [ ] **Step 2: Run tests to verify download specs don't break anything**

Run:
```bash
cd backend && uv run pytest tests/test_models.py -v
```

Expected: all tests pass (tests use `DEFAULT_MODEL_DOWNLOAD_SPECS`, not MLX specs)

- [ ] **Step 3: Commit**

```bash
git add backend/runtime_config/model_download_specs.py
git commit -m "config: point MLX checkpoint to dgrauet/ltx-2.3-mlx-q8"
```

---

### Task 3: Rewrite MLX Video Pipeline (T2V/I2V)

**Files:**
- Rewrite: `backend/services/fast_video_pipeline/mlx_video_pipeline.py`

- [ ] **Step 1: Rewrite mlx_video_pipeline.py**

Replace the entire file with:

```python
"""MLX fast video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Final

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    from ltx_pipelines_mlx import ImageToVideoPipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXVideoPipeline:
    """Fast video generation pipeline using ltx-pipelines-mlx on Apple Silicon.

    Wraps ImageToVideoPipeline (handles both T2V and I2V).
    Pipeline persists across calls; low_memory=True manages component lifecycle.
    """

    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
    ) -> "MLXVideoPipeline":
        del device  # MLX uses Metal GPU automatically
        return MLXVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._pipeline: ImageToVideoPipeline | None = None

        # Resolve model directory: dgrauet pipelines expect a directory
        # containing all safetensors files.
        checkpoint_p = Path(checkpoint_path)
        if checkpoint_p.is_dir():
            self._model_dir = str(checkpoint_p)
        elif checkpoint_p.parent.exists():
            self._model_dir = str(checkpoint_p.parent)
        else:
            self._model_dir = checkpoint_path

        self._gemma_repo = (
            str(gemma_root)
            if gemma_root and Path(gemma_root).exists()
            else _DEFAULT_GEMMA_REPO
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx import ImageToVideoPipeline  # type: ignore[import-untyped]

        self._pipeline = ImageToVideoPipeline(
            model_dir=self._model_dir,
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX ImageToVideoPipeline loaded from %s", self._model_dir)

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        output_path: str,
    ) -> None:
        """Generate video from text/image input via MLX."""
        logger.info(
            "MLX generate: prompt=%r seed=%d %dx%d %d frames @ %.1f fps",
            prompt[:50], seed, width, height, num_frames, frame_rate,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        image_arg: str | None = images[0].path if images else None

        self._pipeline.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            image=image_arg,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        gc.collect()
        logger.info("MLX generation complete: %s", output_path)

    def warmup(self, output_path: str) -> None:
        """Warmup pipeline with a small test generation."""
        logger.info("MLX pipeline warmup starting")
        try:
            self._ensure_loaded()
            assert self._pipeline is not None
            self._pipeline.generate_and_save(
                prompt="test warmup",
                output_path=output_path,
                height=256,
                width=384,
                num_frames=9,
                seed=42,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            gc.collect()
        logger.info("MLX pipeline warmup complete")

    def compile_transformer(self) -> None:
        """No-op — MLX compiles Metal shaders on first use."""
        pass
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd backend && uv run pytest tests/ -v -x
```

Expected: all tests pass (tests use FakeFastVideoPipeline, never import the real MLX pipeline)

- [ ] **Step 3: Commit**

```bash
git add backend/services/fast_video_pipeline/mlx_video_pipeline.py
git commit -m "feat: rewrite MLX video pipeline to use ltx-pipelines-mlx"
```

---

### Task 4: Rewrite MLX A2V Pipeline

**Files:**
- Rewrite: `backend/services/a2v_pipeline/mlx_a2v_pipeline.py`

- [ ] **Step 1: Rewrite mlx_a2v_pipeline.py**

Replace the entire file with:

```python
"""MLX Audio-to-Video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    from ltx_pipelines_mlx import AudioToVideoPipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXA2VPipeline:
    """Audio-to-video generation pipeline using ltx-pipelines-mlx on Apple Silicon.

    Wraps AudioToVideoPipeline (two-stage with dev transformer + CFG).
    Pipeline persists across calls; low_memory=True manages component lifecycle.
    """

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
    ) -> "MLXA2VPipeline":
        del device
        return MLXA2VPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._pipeline: AudioToVideoPipeline | None = None

        checkpoint_p = Path(checkpoint_path)
        if checkpoint_p.is_dir():
            self._model_dir = str(checkpoint_p)
        elif checkpoint_p.parent.exists():
            self._model_dir = str(checkpoint_p.parent)
        else:
            self._model_dir = checkpoint_path

        self._gemma_repo = (
            str(gemma_root)
            if gemma_root and Path(gemma_root).exists()
            else _DEFAULT_GEMMA_REPO
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx import AudioToVideoPipeline  # type: ignore[import-untyped]

        self._pipeline = AudioToVideoPipeline(
            model_dir=self._model_dir,
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX AudioToVideoPipeline loaded from %s", self._model_dir)

    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        output_path: str,
    ) -> None:
        """Generate video from audio input via MLX."""
        logger.info(
            "MLX A2V generate: prompt=%r seed=%d %dx%d %d frames",
            prompt[:50], seed, width, height, num_frames,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        image_arg: str | None = images[0].path if images else None

        self._pipeline.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            audio_path=audio_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=float(frame_rate),
            seed=seed,
            stage1_steps=num_inference_steps,
            image=image_arg,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
        )

        gc.collect()
        logger.info("MLX A2V generation complete: %s", output_path)
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd backend && uv run pytest tests/ -v -x
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/services/a2v_pipeline/mlx_a2v_pipeline.py
git commit -m "feat: rewrite MLX A2V pipeline to use ltx-pipelines-mlx"
```

---

### Task 5: Rewrite MLX Retake Pipeline

**Files:**
- Rewrite: `backend/services/retake_pipeline/mlx_retake_pipeline.py`

- [ ] **Step 1: Rewrite mlx_retake_pipeline.py**

Replace the entire file with:

```python
"""MLX Retake pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltx_pipelines_mlx import RetakePipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXRetakePipeline:
    """Retake (partial video regeneration) pipeline using ltx-pipelines-mlx on Apple Silicon.

    Wraps RetakePipeline for real segment regeneration using the dev transformer with CFG.
    Pipeline persists across calls; low_memory=True manages component lifecycle.
    """

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        device: object,
        *,
        loras: list[object] | None = None,
        quantization: object | None = None,
    ) -> "MLXRetakePipeline":
        del device, quantization, loras
        return MLXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
    ) -> None:
        self._pipeline: RetakePipeline | None = None

        checkpoint_p = Path(checkpoint_path)
        if checkpoint_p.is_dir():
            self._model_dir = str(checkpoint_p)
        elif checkpoint_p.parent.exists():
            self._model_dir = str(checkpoint_p.parent)
        else:
            self._model_dir = checkpoint_path

        self._gemma_repo = (
            str(gemma_root)
            if gemma_root and Path(gemma_root).exists()
            else _DEFAULT_GEMMA_REPO
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx import RetakePipeline  # type: ignore[import-untyped]

        self._pipeline = RetakePipeline(
            model_dir=self._model_dir,
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX RetakePipeline loaded from %s", self._model_dir)

    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: object | None = None,
        audio_guider_params: object | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
    ) -> None:
        """Regenerate a section of an existing video via MLX."""
        del negative_prompt, video_guider_params, audio_guider_params
        del enhance_prompt, regenerate_video, distilled

        logger.info(
            "MLX retake: prompt=%r seed=%d %.2f-%.2fs",
            prompt[:50], seed, start_time, end_time,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        # Convert time range to frame indices (24 fps)
        start_frame = int(start_time * 24)
        end_frame = int(end_time * 24)

        video_latent, audio_latent = self._pipeline.retake_from_video(
            prompt=prompt,
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            seed=seed,
            num_steps=num_inference_steps,
            regenerate_audio=regenerate_audio,
        )

        self._pipeline._decode_and_save_video(  # pyright: ignore[reportPrivateUsage]
            video_latent, audio_latent, output_path
        )

        gc.collect()
        logger.info("MLX retake complete: %s", output_path)
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd backend && uv run pytest tests/ -v -x
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/services/retake_pipeline/mlx_retake_pipeline.py
git commit -m "feat: rewrite MLX retake pipeline to use ltx-pipelines-mlx"
```

---

### Task 6: Rewrite MLX IC-LoRA Pipeline

**Files:**
- Rewrite: `backend/services/ic_lora_pipeline/mlx_ic_lora_pipeline.py`

- [ ] **Step 1: Rewrite mlx_ic_lora_pipeline.py**

Replace the entire file with:

```python
"""MLX IC-LoRA pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    from ltx_pipelines_mlx import ICLoraPipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXIcLoraPipeline:
    """IC-LoRA controlled video generation using ltx-pipelines-mlx on Apple Silicon.

    Wraps ICLoraPipeline for reference-conditioned generation with LoRA weights.
    Pipeline persists across calls; low_memory=True manages component lifecycle.
    """

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: object,
    ) -> "MLXIcLoraPipeline":
        del device
        return MLXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._lora_path = lora_path
        self._pipeline: ICLoraPipeline | None = None

        checkpoint_p = Path(checkpoint_path)
        if checkpoint_p.is_dir():
            self._model_dir = str(checkpoint_p)
        elif checkpoint_p.parent.exists():
            self._model_dir = str(checkpoint_p.parent)
        else:
            self._model_dir = checkpoint_path

        self._gemma_repo = (
            str(gemma_root)
            if gemma_root and Path(gemma_root).exists()
            else _DEFAULT_GEMMA_REPO
        )

    def _ensure_loaded(self) -> None:
        """Lazy-load the pipeline on first use."""
        if self._pipeline is not None:
            return
        from ltx_pipelines_mlx import ICLoraPipeline  # type: ignore[import-untyped]

        self._pipeline = ICLoraPipeline(
            model_dir=self._model_dir,
            lora_paths=[(self._lora_path, 1.0)],
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX ICLoraPipeline loaded from %s", self._model_dir)

    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
    ) -> None:
        """Generate video with IC-LoRA control via MLX."""
        logger.info(
            "MLX IC-LoRA generate: prompt=%r seed=%d %dx%d %d frames",
            prompt[:50], seed, width, height, num_frames,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        self._pipeline.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
        )

        gc.collect()
        logger.info("MLX IC-LoRA generation complete: %s", output_path)
```

- [ ] **Step 2: Run tests**

Run:
```bash
cd backend && uv run pytest tests/ -v -x
```

Expected: all tests pass

- [ ] **Step 3: Commit**

```bash
git add backend/services/ic_lora_pipeline/mlx_ic_lora_pipeline.py
git commit -m "feat: rewrite MLX IC-LoRA pipeline to use ltx-pipelines-mlx"
```

---

### Task 7: Run Full Test Suite and Typecheck

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run:
```bash
cd backend && uv run pytest tests/ -v
```

Expected: all tests pass. Tests use fake services so the library swap should be invisible to them.

- [ ] **Step 2: Run pyright typecheck**

Run:
```bash
cd backend && uv run pyright
```

Expected: no new errors from the pipeline rewrites. Existing `# type: ignore[import-untyped]` and `# pyright: ignore[reportAssignmentType]` suppressions in `app_handler.py` remain valid since the class names haven't changed.

- [ ] **Step 3: Verify no stale mlx_video imports remain**

Run:
```bash
grep -r "mlx_video" backend/services/ --include="*.py"
```

Expected: no matches. All `from mlx_video.models.ltx_2.generate import generate_video` imports should be gone.

- [ ] **Step 4: Commit any fixes and final commit**

If pyright or tests caught issues, fix and commit. Then:

```bash
git add -A
git commit -m "chore: verify tests and typecheck pass after library switch"
```
