"""MLX text encoder — no-op implementation for Apple Silicon.

On the MLX path, text encoding is handled internally by mlx_video.
This stub satisfies the TextEncoder Protocol without touching torch or ltx_pipelines.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.app_state_types import AppState, TextEncodingResult

logger = logging.getLogger(__name__)


class MLXTextEncoder:
    """No-op text encoder for the MLX pipeline path.

    The mlx_video pipeline handles prompt encoding internally using its own
    text encoder, so the patching and API encoding facilities of LTXTextEncoder
    are not needed.
    """

    def install_patches(self, state_getter: Callable[[], AppState]) -> None:
        """No-op — MLX pipeline does not use ltx_pipelines PromptEncoder."""
        pass

    def encode_via_api(
        self,
        prompt: str,
        api_key: str,
        checkpoint_path: str,
        enhance_prompt: bool,
    ) -> TextEncodingResult | None:
        """No-op — MLX pipeline encodes text locally."""
        return None
