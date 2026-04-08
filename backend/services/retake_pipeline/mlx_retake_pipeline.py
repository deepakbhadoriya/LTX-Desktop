"""MLX Retake pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_REPO = "dgrauet/ltx-2.3-mlx"
_DEFAULT_TEXT_ENCODER_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXRetakePipeline:
    """Retake (partial video regeneration) pipeline using mlx_video on Apple Silicon.

    Uses generate_video() with video conditioning for retake generation.
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
        del device, quantization, loras
        return MLXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
        )

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root

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
        from mlx_video.models.ltx_2.generate import generate_video, PipelineType  # type: ignore[import-untyped]

        logger.info(
            "MLX retake: prompt=%r seed=%d %.2f-%.2fs",
            prompt[:50], seed, start_time, end_time,
        )

        del video_guider_params, audio_guider_params, enhance_prompt

        # mlx_video does not yet support partial retake natively.
        # Generate a full replacement clip matching the retake region duration.
        duration_secs = end_time - start_time
        num_frames = max(9, int(duration_secs * 24))
        # Snap to 8k+1 (required by LTX VAE)
        num_frames = ((num_frames - 1) // 8) * 8 + 1

        generate_video(
            model_repo=self._model_repo,
            text_encoder_repo=self._text_encoder_repo,
            prompt=prompt,
            negative_prompt=negative_prompt,
            pipeline=PipelineType.DISTILLED if distilled else PipelineType.DEV,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            seed=seed,
            fps=24,
            output_path=output_path,
        )

        gc.collect()
        logger.info("MLX retake complete: %s", output_path)
