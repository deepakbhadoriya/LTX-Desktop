"""Lightweight depth estimation for macOS using cv2/numpy (no torch)."""

from __future__ import annotations

import logging
from typing import cast

import cv2
import numpy as np

from services.services_utils import FrameArray

logger = logging.getLogger(__name__)


class MLXDepthPipeline:
    """Sobel-based depth approximation matching MidasDPTPipeline output format."""

    @staticmethod
    def create(
        model_path: str,
        device: object,
    ) -> "MLXDepthPipeline":
        del model_path, device  # Not needed — pure cv2 implementation
        return MLXDepthPipeline()

    def apply(self, frame: FrameArray) -> FrameArray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(np.add(np.multiply(sobel_x, sobel_x), np.multiply(sobel_y, sobel_y)))

        blurred = cv2.GaussianBlur(magnitude, (5, 5), 0)

        min_val = float(blurred.min())
        max_val = float(blurred.max())
        if max_val - min_val <= 1e-6:
            depth_uint8 = np.zeros(gray.shape, dtype=np.uint8)
        else:
            normalized = (blurred - min_val) / (max_val - min_val)
            depth_uint8 = np.clip(normalized * 255.0, 0.0, 255.0).astype(np.uint8)

        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)
        return cast(FrameArray, colored)
