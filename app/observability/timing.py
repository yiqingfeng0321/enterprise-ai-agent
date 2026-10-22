from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    result = {"elapsed_ms": 0.0}
    started = time.perf_counter()
    try:
        yield result
    finally:
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
