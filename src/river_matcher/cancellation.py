from __future__ import annotations

import threading


class OperationCancelled(RuntimeError):
    """Raised when a cooperative cancellation request is observed."""


class CancellationToken:
    """Thread-safe cancellation flag shared by UI workers and matching code."""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()

    def check(self) -> None:
        if self._event.is_set():
            raise OperationCancelled("Matching was cancelled.")
