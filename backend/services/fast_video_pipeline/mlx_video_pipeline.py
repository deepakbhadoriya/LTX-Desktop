"""MLX fast video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os
from typing import Final

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)


class MLXVideoPipeline:
    """Fast video generation pipeline using MLX on Apple Silicon.

    Drop-in replacement for LTXFastVideoPipeline that uses mlx_video
    instead of CUDA-based ltx_pipelines for local inference on Mac.
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
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        """Lazily load the mlx_video pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading MLX video pipeline from: %s", self._checkpoint_path)

        from mlx_video import LTXVideo  # type: ignore[import-untyped]

        self._pipeline = LTXVideo(
            checkpoint_path=self._checkpoint_path,
            gemma_root=self._gemma_root,
            upsampler_path=self._upsampler_path,
        )

        logger.info("MLX video pipeline loaded successfully")
        return self._pipeline

    def _run_inference(
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
        """Run inference via mlx_video and write output to file."""
        import mlx.core as mx  # type: ignore[import-untyped]

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
            output_path=output_path,
        )

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

        self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            output_path=output_path,
        )

        gc.collect()
        logger.info("MLX generation complete: %s", output_path)

    def warmup(self, output_path: str) -> None:
        """Warmup pipeline with a small test generation."""
        logger.info("MLX pipeline warmup starting")
        try:
            self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=9,
                frame_rate=8.0,
                images=[],
                output_path=output_path,
            )
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)
            gc.collect()
        logger.info("MLX pipeline warmup complete")

    def compile_transformer(self) -> None:
        """No-op — MLX compiles Metal shaders on first use."""
        pass
