"""MLX IC-LoRA pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class MLXIcLoraPipeline:
    """IC-LoRA controlled video generation pipeline using MLX on Apple Silicon.

    Replaces LTXIcLoraPipeline which uses CUDA-based ltx_pipelines.
    """

    @staticmethod
    def create(
        checkpoint_path: str,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: object,
    ) -> "MLXIcLoraPipeline":
        del device  # MLX uses Metal GPU automatically
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
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._lora_path = lora_path
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        """Lazily load the mlx_video IC-LoRA pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading MLX IC-LoRA pipeline from: %s", self._checkpoint_path)

        from mlx_video import LTXICLora  # type: ignore[import-untyped]

        self._pipeline = LTXICLora(
            checkpoint_path=self._checkpoint_path,
            gemma_root=self._gemma_root,
            upsampler_path=self._upsampler_path,
            lora_path=self._lora_path,
        )

        logger.info("MLX IC-LoRA pipeline loaded successfully")
        return self._pipeline

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
        import mlx.core as mx  # type: ignore[import-untyped]

        logger.info(
            "MLX IC-LoRA generate: prompt=%r seed=%d %dx%d %d frames",
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
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=image_inputs,
            video_conditioning=video_conditioning,
            output_path=output_path,
        )

        gc.collect()
        logger.info("MLX IC-LoRA generation complete: %s", output_path)
