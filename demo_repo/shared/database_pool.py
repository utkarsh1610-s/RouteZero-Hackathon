"""Thread-safe database connection pooling shared by every StreamCo service."""

import logging
import queue
import threading
from typing import Any

import psycopg2

logger = logging.getLogger(__name__)


class PoolExhaustedError(RuntimeError):
    """Raised when no pooled connection becomes available within the timeout."""


class ConnectionPool:
    """Fixed-size pool of PostgreSQL connections.

    Every StreamCo service (payments, auth, ML, profiles) shares this module
    so that connection limits are enforced consistently across the fleet.
    """

    def __init__(self, dsn: str, max_size: int = 10, acquire_timeout_seconds: float = 5.0) -> None:
        self.dsn = dsn
        self.max_size = max_size
        self.acquire_timeout_seconds = acquire_timeout_seconds
        self._idle: "queue.Queue[Any]" = queue.Queue(maxsize=max_size)
        self._created = 0
        self._lock = threading.Lock()

    def acquire(self) -> Any:
        """Return an open connection, creating one if under the size limit."""
        try:
            return self._idle.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self.max_size:
                self._created += 1
                logger.debug("Opening connection %d/%d", self._created, self.max_size)
                return psycopg2.connect(self.dsn)
        try:
            return self._idle.get(timeout=self.acquire_timeout_seconds)
        except queue.Empty as exc:
            raise PoolExhaustedError(
                f"No database connection available after {self.acquire_timeout_seconds}s "
                f"(pool size {self.max_size})"
            ) from exc

    def release(self, conn: Any) -> None:
        """Return a connection to the idle queue for reuse."""
        try:
            self._idle.put_nowait(conn)
        except queue.Full:
            logger.warning("Idle queue full; closing surplus connection")
            conn.close()

    def close_all(self) -> None:
        """Close every idle connection, e.g. during service shutdown."""
        while True:
            try:
                conn = self._idle.get_nowait()
            except queue.Empty:
                return
            conn.close()
