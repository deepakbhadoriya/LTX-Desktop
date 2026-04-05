"""MLX image generation pipeline using mflux (Z-Image-Turbo) on Apple Silicon."""

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
    """Image generation pipeline using mflux Z-Image-Turbo on Apple Silicon.

    Replaces ZitImageGenerationPipeline which uses CUDA-based diffusers.
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
        """Lazily load the mflux Z-Image-Turbo pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading mflux Z-Image-Turbo from: %s", self._model_path)

        from mflux.models.z_image import ZImageTurbo  # type: ignore[import-untyped]

        self._pipeline = ZImageTurbo(model_path=self._model_path)

        logger.info("mflux Z-Image-Turbo loaded successfully")
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
        _ = guidance_scale  # Z-Image-Turbo ignores guidance_scale

        logger.info("MLX image generate: prompt=%r %dx%d seed=%d", prompt[:50], width, height, seed)

        pipeline = self._ensure_pipeline()

        image = pipeline.generate_image(  # type: ignore[union-attr]
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
        )

        gc.collect()
        logger.info("MLX image generation complete")
        return _MfluxOutput(images=[image])

    def to(self, device: str) -> None:
        """No-op — MLX manages device placement automatically."""
        pass
