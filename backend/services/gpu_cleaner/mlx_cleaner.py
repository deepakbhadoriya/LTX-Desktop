"""MLX memory cleanup for Apple Silicon."""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


class MLXCleaner:
    """GPU memory cleanup using gc.collect() for MLX on Apple Silicon.

    Replaces TorchCleaner which uses torch.cuda.empty_cache() / torch.mps.empty_cache().
    """

    def cleanup(self) -> None:
        """Release MLX memory via Python garbage collection."""
        gc.collect()
