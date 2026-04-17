"""MLX Audio-to-Video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    from ltx_pipelines_mlx import AudioToVideoPipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXA2VPipeline:
    """Audio-to-video generation pipeline using ltx-pipelines-mlx on Apple Silicon.

    Wraps AudioToVideoPipeline (two-stage with dev transformer + CFG).
    Pipeline persists across calls; low_memory=True manages component lifecycle.
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
        self._upsampler_path = upsampler_path
        self._pipeline: AudioToVideoPipeline | None = None

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
        from ltx_pipelines_mlx import AudioToVideoPipeline  # type: ignore[import-untyped]

        self._pipeline = AudioToVideoPipeline(
            model_dir=self._model_dir,
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX AudioToVideoPipeline loaded from %s", self._model_dir)

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
        logger.info(
            "MLX A2V generate: prompt=%r seed=%d %dx%d %d frames",
            prompt[:50], seed, width, height, num_frames,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        image_arg: str | None = images[0].path if images else None

        self._pipeline.generate_and_save(
            prompt=prompt,
            output_path=output_path,
            audio_path=audio_path,
            height=height,
            width=width,
            num_frames=num_frames,
            fps=float(frame_rate),
            seed=seed,
            stage1_steps=num_inference_steps,
            image=image_arg,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
        )

        gc.collect()
        logger.info("MLX A2V generation complete: %s", output_path)
