from __future__ import annotations

import gc
import logging
import sys

logger = logging.getLogger(__name__)


def collect_model_memory() -> None:
    """Release Python cycles and caches owned by already-imported ML runtimes."""
    gc.collect()

    mx = sys.modules.get("mlx.core")
    if mx is not None:
        try:
            # MLX model finalizers can expose another set of unreachable objects
            # after Metal completes the pending releases.
            for _ in range(2):
                mx.synchronize()
                gc.collect()
                if hasattr(mx, "clear_cache"):
                    mx.clear_cache()
        except Exception:
            logger.debug("MLX memory cleanup failed", exc_info=True)

    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
                torch.mps.synchronize()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            logger.debug("Torch memory cleanup failed", exc_info=True)
