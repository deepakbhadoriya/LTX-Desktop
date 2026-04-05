"""MLX Retake pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


class MLXRetakePipeline:
    """Retake (partial video regeneration) pipeline using MLX on Apple Silicon.

    Replaces LTXRetakePipeline which uses CUDA-based ltx_pipelines.
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
        del device, quantization  # MLX uses Metal GPU automatically; no FP8 quantization
        return MLXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            loras=loras,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        loras: list[object] | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._loras = loras or []
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        """Lazily load the mlx_video retake pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading MLX retake pipeline from: %s", self._checkpoint_path)

        from mlx_video import LTXRetake  # type: ignore[import-untyped]

        self._pipeline = LTXRetake(
            checkpoint_path=self._checkpoint_path,
            gemma_root=self._gemma_root,
        )

        logger.info("MLX retake pipeline loaded successfully")
        return self._pipeline

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
        import mlx.core as mx  # type: ignore[import-untyped]

        logger.info(
            "MLX retake: prompt=%r seed=%d %.2f-%.2fs",
            prompt[:50], seed, start_time, end_time,
        )

        pipeline = self._ensure_pipeline()

        mx.random.seed(seed)
        pipeline(  # type: ignore[operator]
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            output_path=output_path,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            distilled=distilled,
        )

        gc.collect()
        logger.info("MLX retake complete: %s", output_path)
