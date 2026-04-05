"""MLX image generation pipeline using mflux (Flux Schnell) on Apple Silicon."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass

from services.services_utils import ImagePipelineOutputLike, PILImageType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MfluxOutput:
    images: Sequence[PILImageType]


class MLXImageGenerationPipeline:
    """Image generation pipeline using mflux (Flux) on Apple Silicon.

    Replaces ZitImageGenerationPipeline which uses CUDA-based diffusers.
    Uses the mflux library for Flux Schnell inference on MLX.
    """

    @staticmethod
    def create(
        model_path: str,
        device: str | None = None,
    ) -> "MLXImageGenerationPipeline":
        del device  # MLX uses Metal GPU automatically
        return MLXImageGenerationPipeline(model_path=model_path)

    def __init__(self, model_path: str) -> None:
        self._model_path = model_path
        self._pipeline: object | None = None

    def _ensure_pipeline(self) -> object:
        """Lazily load the mflux pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading mflux image pipeline from: %s", self._model_path)

        from mflux import Flux1  # type: ignore[import-untyped]

        self._pipeline = Flux1.from_alias("schnell", path=self._model_path)

        logger.info("mflux image pipeline loaded successfully")
        return self._pipeline

    def generate(
        self,
        prompt: str,
        height: int,
        width: int,
        guidance_scale: float,
        num_inference_steps: int,
        seed: int,
    ) -> ImagePipelineOutputLike:
        """Generate an image from a text prompt via mflux."""
        import mlx.core as mx  # type: ignore[import-untyped]

        _ = guidance_scale  # Flux Schnell ignores guidance_scale

        logger.info("MLX image generate: prompt=%r %dx%d seed=%d", prompt[:50], width, height, seed)

        pipeline = self._ensure_pipeline()

        mx.random.seed(seed)
        image = pipeline.generate_image(  # type: ignore[union-attr]
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        gc.collect()
        logger.info("MLX image generation complete")
        return _MfluxOutput(images=[image])

    def to(self, device: str) -> None:
        """No-op — MLX manages device placement automatically."""
        pass
