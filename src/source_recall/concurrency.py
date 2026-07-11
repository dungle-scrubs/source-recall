"""Concurrency primitives shared across source-recall modules."""

from __future__ import annotations

import threading


class ReaderWriterLock:
    """Minimal reader/writer lock built on a condition variable.

    Multiple readers may hold the lock concurrently; a writer waits until
    no readers (or writers) are active.  Used to let concurrent
    ``query``/``status`` calls proceed in parallel while an exclusive
    write lock swaps the underlying connection (M-2 fix).

    Writers are preferred once waiting to avoid starving refreshes under
    heavy read load.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._readers = 0
        self._writers = 0
        self._writer_active = False

    def acquire_read(self) -> None:
        with self._cond:
            while self._writers > 0 or self._writer_active:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers += 1
            while self._readers > 0 or self._writer_active:
                self._cond.wait()
            self._writers -= 1
            self._writer_active = True

    def release_write(self) -> None:
        with self._cond:
            self._writer_active = False
            self._cond.notify_all()
