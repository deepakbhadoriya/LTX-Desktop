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

        images_arg: list[tuple[str, int, float]] | None = None
        if images:
            images_arg = [(img.path, img.frame_idx, img.strength) for img in images]

        self._pipeline.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            images=images_arg,
        )

        gc.collect()
        logger.info("MLX IC-LoRA generation complete: %s", output_path)
