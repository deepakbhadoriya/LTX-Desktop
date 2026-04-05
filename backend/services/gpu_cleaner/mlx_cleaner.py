"""MLX memory cleanup for Apple Silicon."""

from __future__ import annotations

import gc


class MLXCleaner:
    """GPU memory cleanup using gc.collect() for MLX on Apple Silicon.

    Replaces TorchCleaner which uses torch.cuda.empty_cache() / torch.mps.empty_cache().
    """

    def cleanup(self) -> None:
        gc.collect()
