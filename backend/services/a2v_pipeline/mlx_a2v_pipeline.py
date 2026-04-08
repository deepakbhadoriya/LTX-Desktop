"""MLX Audio-to-Video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_REPO = "dgrauet/ltx-2.3-mlx"
_DEFAULT_TEXT_ENCODER_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXA2VPipeline:
    """Audio-to-video generation pipeline using mlx_video on Apple Silicon.

    Uses generate_video() with audio_path parameter for A2V generation.
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
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path

        checkpoint_p = Path(checkpoint_path)
        if checkpoint_p.is_dir():
            self._model_repo = str(checkpoint_p)
        elif checkpoint_p.parent.exists():
            self._model_repo = str(checkpoint_p.parent)
        else:
            self._model_repo = _DEFAULT_MODEL_REPO
        self._text_encoder_repo = str(gemma_root) if gemma_root and Path(gemma_root).exists() else _DEFAULT_TEXT_ENCODER_REPO

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
        from mlx_video.models.ltx_2.generate import generate_video, PipelineType  # type: ignore[import-untyped]

        logger.info(
            "MLX A2V generate: prompt=%r seed=%d %dx%d %d frames",
            prompt[:50], seed, width, height, num_frames,
        )

        image_arg: str | None = None
        image_strength_arg: float = 1.0
        if images:
            image_arg = images[0].path
            image_strength_arg = images[0].strength

        generate_video(
            model_repo=self._model_repo,
            text_encoder_repo=self._text_encoder_repo,
            prompt=prompt,
            negative_prompt=negative_prompt,
            pipeline=PipelineType.DISTILLED,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            seed=seed,
            fps=int(frame_rate),
            output_path=output_path,
            spatial_upscaler=self._upsampler_path,
            image=image_arg,
            image_strength=image_strength_arg,
            audio_file=audio_path,
            audio_start_time=audio_start_time,
            audio=True,
        )

        gc.collect()
        logger.info("MLX A2V generation complete: %s", output_path)
