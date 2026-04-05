"""Pose processor protocol definitions."""

from __future__ import annotations

from typing import Protocol

from services.services_utils import FrameArray


class PoseProcessorPipeline(Protocol):
    @staticmethod
    def create(
        pose_model_path: str,
        person_detector_model_path: str,
        device: object,
    ) -> "PoseProcessorPipeline":
        ...

    def apply(self, frame: FrameArray) -> FrameArray:
        ...
