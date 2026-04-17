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
