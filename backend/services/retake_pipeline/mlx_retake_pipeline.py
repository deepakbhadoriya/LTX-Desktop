"""MLX Retake pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ltx_pipelines_mlx import RetakePipeline  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

_DEFAULT_GEMMA_REPO = "mlx-community/gemma-3-12b-it-4bit"


class MLXRetakePipeline:
    """Retake (partial video regeneration) pipeline using ltx-pipelines-mlx on Apple Silicon.

    Wraps RetakePipeline for real segment regeneration using the dev transformer with CFG.
    Pipeline persists across calls; low_memory=True manages component lifecycle.
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
        self._pipeline: RetakePipeline | None = None

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
        from ltx_pipelines_mlx import RetakePipeline  # type: ignore[import-untyped]

        self._pipeline = RetakePipeline(
            model_dir=self._model_dir,
            gemma_model_id=self._gemma_repo,
            low_memory=True,
        )
        self._pipeline.load()
        logger.info("MLX RetakePipeline loaded from %s", self._model_dir)

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
        del negative_prompt, video_guider_params, audio_guider_params
        del enhance_prompt, regenerate_video, distilled

        logger.info(
            "MLX retake: prompt=%r seed=%d %.2f-%.2fs",
            prompt[:50], seed, start_time, end_time,
        )

        self._ensure_loaded()
        assert self._pipeline is not None

        # Convert time range to frame indices (24 fps)
        start_frame = int(start_time * 24)
        end_frame = int(end_time * 24)

        video_latent, audio_latent = self._pipeline.retake_from_video(
            prompt=prompt,
            video_path=video_path,
            start_frame=start_frame,
            end_frame=end_frame,
            seed=seed,
            num_steps=num_inference_steps,
            regenerate_audio=regenerate_audio,
        )

        self._pipeline._decode_and_save_video(  # pyright: ignore[reportPrivateUsage]
            video_latent, audio_latent, output_path
        )

        gc.collect()
        logger.info("MLX retake complete: %s", output_path)
