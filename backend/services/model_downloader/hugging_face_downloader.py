"""Hugging Face model download service wrapper."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from threading import Lock
from typing import Any
from unittest.mock import patch

# Enable huggingface_hub's high-throughput xet transfer mode before the
# library is imported (its constants.py reads this env var at module load).
# xet-backed files (which large Lightricks model shards use) will transfer
# via parallel chunked downloads for much higher throughput on fast links.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from huggingface_hub import file_download, hf_hub_download, snapshot_download  # noqa: E402  # type: ignore[reportUnknownVariableType]
from tqdm.auto import tqdm as tqdm_auto  # noqa: E402  # type: ignore[reportUnknownVariableType]

# Number of parallel file workers for snapshot (multi-file) downloads.
# ``huggingface_hub.snapshot_download`` defaults to 8; we bump to 16 to
# saturate fast links on modern multi-core machines. For single-file
# downloads via ``hf_hub_download``, parallelism is handled internally by
# xet's chunked transfer — there is no public worker knob.
DOWNLOAD_MAX_WORKERS = 16


def _make_progress_tqdm_class(callback: Callable[[int], None], initial_bytes: int = 0) -> type:
    """Return a tqdm subclass that reports aggregated progress via *callback*.

    Used for both single-file and snapshot downloads.  Snapshot downloads
    spawn one tqdm instance per file; all instances share mutable state so
    the callback reports total progress across every file in the download.

    ``initial_bytes`` seeds the counter so that resumed downloads report the
    correct cumulative progress — ``huggingface_hub`` only calls tqdm.update
    for bytes transferred *this session*, not pre-existing .incomplete bytes.
    """
    lock = Lock()
    shared: dict[str, int] = {"downloaded": initial_bytes}

    class _ProgressTqdm(tqdm_auto):  # type: ignore[reportUntypedBaseClass]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)  # type: ignore[reportUnknownMemberType]

        def update(self, n: float | int | None = 1) -> bool | None:  # type: ignore[reportIncompatibleMethodOverride]
            result = super().update(n)
            if n is not None:
                with lock:
                    shared["downloaded"] += int(n)
                callback(shared["downloaded"])
            return result

    return _ProgressTqdm


@contextlib.contextmanager
def _patch_download_progress(callback: Callable[[int], None], initial_bytes: int = 0) -> Iterator[None]:
    """Temporarily monkey-patch ``huggingface_hub.file_download.http_get``
    and ``xet_get`` to inject a custom tqdm bar that forwards progress to
    *callback*.

    ``hf_hub_download`` does not expose a ``tqdm_class`` parameter (unlike
    ``snapshot_download``), but its internal ``http_get`` and ``xet_get``
    both accept a private ``_tqdm_bar`` kwarg.  We wrap them to inject our
    own bar when the caller hasn't already provided one.

    See ``test_http_get_accepts_tqdm_bar`` — if that test breaks after a
    huggingface_hub upgrade, this patch needs to be revisited.
    """
    tqdm_cls = _make_progress_tqdm_class(callback, initial_bytes=initial_bytes)
    original_http_get: Callable[..., Any] = file_download.http_get  # type: ignore[reportUnknownMemberType]

    def _wrapped_http_get(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("_tqdm_bar") is None:
            kwargs["_tqdm_bar"] = tqdm_cls(disable=True)
        return original_http_get(*args, **kwargs)

    xet_get_fn: Callable[..., Any] | None = getattr(file_download, "xet_get", None)

    def _wrapped_xet_get(*args: Any, **kwargs: Any) -> None:
        if kwargs.get("_tqdm_bar") is None:
            kwargs["_tqdm_bar"] = tqdm_cls(disable=True)
        assert xet_get_fn is not None
        return xet_get_fn(*args, **kwargs)

    with patch.object(file_download, "http_get", _wrapped_http_get):
        if xet_get_fn is not None:
            with patch.object(file_download, "xet_get", _wrapped_xet_get):
                yield
        else:
            yield


class HuggingFaceDownloader:
    """Wraps huggingface_hub download functions."""

    def download_file(
        self,
        repo_id: str,
        filename: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
        initial_bytes: int = 0,
    ) -> Path:
        ctx = (
            _patch_download_progress(on_progress, initial_bytes=initial_bytes)
            if on_progress is not None
            else contextlib.nullcontext()
        )
        with ctx:
            path: str = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=local_dir)
        return Path(path)

    def download_snapshot(
        self,
        repo_id: str,
        local_dir: str,
        on_progress: Callable[[int], None] | None = None,
        ignore_patterns: list[str] | None = None,
        initial_bytes: int = 0,
    ) -> Path:
        ctx = (
            _patch_download_progress(on_progress, initial_bytes=initial_bytes)
            if on_progress is not None
            else contextlib.nullcontext()
        )
        with ctx:
            path: str = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                ignore_patterns=ignore_patterns,
                max_workers=DOWNLOAD_MAX_WORKERS,
            )
        return Path(path)
