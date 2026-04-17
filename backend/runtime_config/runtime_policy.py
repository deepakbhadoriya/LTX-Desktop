"""Runtime policy decisions for forced API mode."""

from __future__ import annotations


def decide_force_api_generations(
    system: str,
    cuda_available: bool,
    vram_gb: int | None,
    *,
    total_ram_gb: float | None = None,
) -> bool:
    """Return whether API-only generation must be forced for this runtime."""
    if system == "Darwin":
        if total_ram_gb is None:
            try:
                import psutil
                total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                return True
        return total_ram_gb < 16  # LTX 2.3 Q4 needs ~19GB; 16GB minimum for local gen

    if system in ("Windows", "Linux"):
        if not cuda_available:
            return True
        if vram_gb is None:
            return True
        return vram_gb < 15

    # Fail closed for non-target platforms unless explicitly relaxed.
    return True
