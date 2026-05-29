"""System resource detection and configuration utilities."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def detect_accelerator(force_cpu: bool = False) -> str:
    """Detect the best available compute accelerator.

    Parameters
    ----------
    force_cpu:
        When *True*, always returns ``"cpu"`` regardless of GPU availability.

    Returns
    -------
    str
        ``"gpu"`` if a CUDA device is available and *force_cpu* is False,
        otherwise ``"cpu"``.
    """
    if force_cpu:
        logger.info("Accelerator forced to CPU by configuration")
        return "cpu"

    try:
        import torch  # type: ignore[import-untyped]

        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info("GPU detected: %s", device_name)
            return "gpu"
    except ImportError:
        logger.debug("PyTorch not installed — falling back to CPU")

    logger.info("No GPU available — using CPU")
    return "cpu"


def configure_torch_threads(max_threads: int | str = "auto") -> int:
    """Set the number of intra-op threads used by PyTorch.

    Parameters
    ----------
    max_threads:
        Number of threads.  ``"auto"`` resolves to ``max(1, cpu_count // 2)``.

    Returns
    -------
    int
        The number of threads that was actually set.
    """
    cpu_count = os.cpu_count() or 4

    if max_threads == "auto":
        n_threads = max(1, cpu_count // 2)
    else:
        n_threads = int(max_threads)

    try:
        import torch  # type: ignore[import-untyped]

        torch.set_num_threads(n_threads)
        logger.info("PyTorch intra-op threads set to %d", n_threads)
    except ImportError:
        logger.debug("PyTorch not installed — thread setting skipped")

    return n_threads


def resolve_max_workers(max_workers: int | str = "auto", has_gpu: bool = False) -> int:
    """Determine the number of parallel workers.

    Parameters
    ----------
    max_workers:
        Desired worker count.  ``"auto"`` resolves heuristically:

        * **GPU present** → ``max(1, cpu_count - 2)``
        * **CPU only**    → ``max(1, cpu_count - 1)``
    has_gpu:
        Whether a GPU accelerator is available (affects the ``"auto"``
        heuristic).

    Returns
    -------
    int
        Resolved number of workers (always ≥ 1).
    """
    if max_workers != "auto":
        resolved = max(1, int(max_workers))
        logger.info("Max workers set to %d (explicit)", resolved)
        return resolved

    cpu_count = os.cpu_count() or 4

    if has_gpu:
        resolved = max(1, cpu_count - 2)
    else:
        resolved = max(1, cpu_count - 1)

    logger.info(
        "Max workers auto-resolved to %d (cpu_count=%d, has_gpu=%s)",
        resolved,
        cpu_count,
        has_gpu,
    )
    return resolved
