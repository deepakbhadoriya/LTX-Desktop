"""Depth processor protocol definitions."""

from __future__ import annotations

from typing import Protocol

from services.services_utils import FrameArray


class DepthProcessorPipeline(Protocol):
    @staticmethod
    def create(
        model_path: str,
        device: object,
    ) -> "DepthProcessorPipeline":
        ...

    def apply(self, frame: FrameArray) -> FrameArray:
        ...
