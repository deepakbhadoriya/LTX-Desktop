"""MLX image generation pipeline using mflux (Flux.2 Klein 4B) on Apple Silicon."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from services.services_utils import ImagePipelineOutputLike, PILImageType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _MfluxOutput:
    images: Sequence[PILImageType]


class MLXImageGenerationPipeline:
    """Image generation pipeline using mflux Flux.2 Klein 4B on Apple Silicon.

    Replaces ZitImageGenerationPipeline which uses CUDA-based diffusers.
    Uses Flux.2 Klein 4B (4-bit quantized) for fast, memory-efficient
    image generation on Apple Silicon.
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
        self._pipeline: Any | None = None

    def _get_pipeline(self) -> Any:
        """Lazily load the mflux Flux.2 Klein pipeline on first use."""
        if self._pipeline is not None:
            return self._pipeline

        logger.info("Loading mflux Flux.2 Klein 4B from: %s", self._model_path)

        from mflux.models.flux2 import Flux2Klein  # type: ignore[import-untyped]

        pipeline = Flux2Klein(model_path=self._model_path)
        self._pipeline = pipeline

        logger.info("mflux Flux.2 Klein 4B loaded successfully")
        return pipeline

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
        logger.info("MLX image generate: prompt=%r %dx%d seed=%d", prompt[:50], width, height, seed)

        pipeline = self._get_pipeline()

        # Flux2Klein.generate_image returns a GeneratedImage wrapper;
        # extract the PIL image via .image attribute.
        result = pipeline.generate_image(
            prompt=prompt,
            seed=seed,
            num_inference_steps=num_inference_steps,
            width=width,
            height=height,
            guidance=guidance_scale,
        )

        gc.collect()
        logger.info("MLX image generation complete")
        return _MfluxOutput(images=[result.image])

    def to(self, device: str) -> None:
        """No-op — MLX manages device placement automatically."""
        pass
