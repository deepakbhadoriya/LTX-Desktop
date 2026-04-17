"""No-op pose processor for Darwin — pose conditioning is unused on this platform."""

from __future__ import annotations

import logging

import numpy as np

from services.services_utils import FrameArray

logger = logging.getLogger(__name__)


class MLXPosePipeline:
    """No-op pose processor for Darwin — pose conditioning is unused on this platform."""

    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: object,
    ) -> "MLXPosePipeline":
        del pose_model_path, person_detector_model_path, device
        return MLXPosePipeline()

    def apply(self, frame: FrameArray) -> FrameArray:
        """Return a zero-filled frame — pose conditioning is not supported on MLX."""
        result: FrameArray = np.zeros_like(frame)  # pyright: ignore[reportAssignmentType]
        return result
