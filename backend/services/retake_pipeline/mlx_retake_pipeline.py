"""MLX Retake pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_REPO = "Lightricks/LTX-Video-2.3-distilled"
_DEFAULT_TEXT_ENCODER_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"


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

        checkpoint_dir = Path(checkpoint_path).parent
        self._model_repo = str(checkpoint_dir) if checkpoint_dir.exists() else _DEFAULT_MODEL_REPO
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

        generate_video(
            model_repo=self._model_repo,
            text_encoder_repo=self._text_encoder_repo,
            prompt=prompt,
            negative_prompt=negative_prompt,
            pipeline=PipelineType.DISTILLED if distilled else PipelineType.DEV,
            num_inference_steps=num_inference_steps,
            seed=seed,
            output_path=output_path,
            # Retake-specific: source video with temporal region
            retake_video_path=video_path,
            retake_start_time=start_time,
            retake_end_time=end_time,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
        )

        gc.collect()
        logger.info("MLX retake complete: %s", output_path)
