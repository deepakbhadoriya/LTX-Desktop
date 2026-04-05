"""MLX Audio-to-Video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os
from typing import Any

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class MLXa2vPipeline:
    """Audio-to-video generation pipeline using MLX on Apple Silicon.

    Replaces LTXa2vPipeline which uses CUDA-based ltx_pipelines.
    """

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        device: object,
    ) -> "MLXa2vPipeline":
        del device  # MLX uses Metal GPU automatically
        return MLXa2vPipeline(
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
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        """Lazily load the mlx_video A2V pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading MLX A2V pipeline from: %s", self._checkpoint_path)

        from mlx_video import LTXA2V  # type: ignore[import-untyped]

        self._pipeline = LTXA2V(
            checkpoint_path=self._checkpoint_path,
            gemma_root=self._gemma_root,
            upsampler_path=self._upsampler_path,
        )

        logger.info("MLX A2V pipeline loaded successfully")
        return self._pipeline

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
        import mlx.core as mx  # type: ignore[import-untyped]

        logger.info(
            "MLX A2V generate: prompt=%r seed=%d %dx%d %d frames",
            prompt[:50], seed, width, height, num_frames,
        )

        pipeline = self._ensure_pipeline()

        image_inputs = [
            {"path": img.path, "frame_idx": img.frame_idx, "strength": img.strength}
            for img in images
        ]

        mx.random.seed(seed)
        pipeline(  # type: ignore[operator]
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            images=image_inputs,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            output_path=output_path,
        )

        gc.collect()
        logger.info("MLX A2V generation complete: %s", output_path)
