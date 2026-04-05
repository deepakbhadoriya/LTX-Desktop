"""MLX fast video pipeline for Apple Silicon inference."""

from __future__ import annotations

import gc
import logging
import os
from pathlib import Path
from typing import Final

from api_types import ImageConditioningInput

logger = logging.getLogger(__name__)

# Default HF repo IDs for mlx_video — overridden if local paths exist.
_DEFAULT_MODEL_REPO = "Lightricks/LTX-Video-2.3-distilled"
_DEFAULT_TEXT_ENCODER_REPO = "Lightricks/gemma-3-12b-it-qat-q4_0-unquantized"


class MLXVideoPipeline:
    """Fast video generation pipeline using mlx_video on Apple Silicon.

    Uses the mlx_video.models.ltx_2.generate.generate_video() function
    which handles model loading, text encoding, inference, and video output.
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

        # Resolve model repo: use parent dir as HF repo if local, else default.
        checkpoint_dir = Path(checkpoint_path).parent
        self._model_repo = str(checkpoint_dir) if checkpoint_dir.exists() else _DEFAULT_MODEL_REPO
        self._text_encoder_repo = str(gemma_root) if gemma_root and Path(gemma_root).exists() else _DEFAULT_TEXT_ENCODER_REPO

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
        """Run inference via mlx_video generate_video() and write output."""
        from mlx_video.models.ltx_2.generate import generate_video, PipelineType  # type: ignore[import-untyped]

        image_arg: str | None = None
        image_strength_arg: float = 1.0
        image_frame_idx_arg: int = 0
        if images:
            image_arg = images[0].path
            image_strength_arg = images[0].strength
            image_frame_idx_arg = images[0].frame_idx

        generate_video(
            model_repo=self._model_repo,
            text_encoder_repo=self._text_encoder_repo,
            prompt=prompt,
            pipeline=PipelineType.DISTILLED,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            fps=int(frame_rate),
            output_path=output_path,
            spatial_upscaler=self._upsampler_path,
            image=image_arg,
            image_strength=image_strength_arg,
            image_frame_idx=image_frame_idx_arg,
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
