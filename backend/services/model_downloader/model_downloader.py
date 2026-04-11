"""Model downloader service protocol definitions."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol


class ModelDownloader(Protocol):
    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
        initial_bytes: int = 0,
    ) -> Path: ...

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
        ignore_patterns: list[str] | None = None,
        initial_bytes: int = 0,
    ) -> Path: ...
