"""No-op pose processor for Darwin — pose conditioning is unused."""

from __future__ import annotations

import numpy as np

from services.services_utils import FrameArray


class NoopPosePipeline:
    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: object,
    ) -> "NoopPosePipeline":
        del pose_model_path, person_detector_model_path, device
        return NoopPosePipeline()

    def apply(self, frame: FrameArray) -> FrameArray:
        result: FrameArray = np.zeros_like(frame)  # pyright: ignore[reportAssignmentType]
        return result
