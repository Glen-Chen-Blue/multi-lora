"""Discrete simulation clock with event scheduling."""

import heapq
from typing import Callable, List, Tuple, Dict


class SimClock:
    """Global simulation clock. Tracks current time in milliseconds."""

    def __init__(self):
        self.now_ms: int = 0
        self._events: List[Tuple[int, int, Callable]] = []  # (time_ms, seq, callback)
        self._seq: int = 0  # tie-breaker for same-time events
        self._periodic: Dict[int, Tuple[int, Callable, bool]] = {}  # handle -> (interval_ms, cb, active)
        self._next_handle: int = 0

    def schedule_at(self, time_ms: int, callback: Callable) -> None:
        """Schedule a one-shot callback at an absolute time."""
        heapq.heappush(self._events, (time_ms, self._seq, callback))
        self._seq += 1

    def schedule_periodic(self, interval_ms: int, callback: Callable, start_ms: int = 0) -> int:
        """Schedule a repeating callback. Returns handle for cancellation."""
        handle = self._next_handle
        self._next_handle += 1
        self._periodic[handle] = (interval_ms, callback, True)
        # Schedule first occurrence
        first_time = start_ms + interval_ms
        self.schedule_at(first_time, lambda: self._fire_periodic(handle))
        return handle

    def cancel_periodic(self, handle: int) -> None:
        """Cancel a periodic schedule."""
        if handle in self._periodic:
            interval, cb, _ = self._periodic[handle]
            self._periodic[handle] = (interval, cb, False)

    def _fire_periodic(self, handle: int) -> None:
        if handle not in self._periodic:
            return
        interval, cb, active = self._periodic[handle]
        if not active:
            del self._periodic[handle]
            return
        cb()
        # Re-schedule
        next_time = self.now_ms + interval
        self.schedule_at(next_time, lambda: self._fire_periodic(handle))

    def advance(self, delta_ms: int = 1) -> None:
        """Advance clock by delta_ms, firing all due callbacks."""
        target = self.now_ms + delta_ms
        self.now_ms = target
        while self._events and self._events[0][0] <= target:
            _, _, cb = heapq.heappop(self._events)
            cb()

    def now(self) -> int:
        return self.now_ms

    def now_s(self) -> float:
        return self.now_ms / 1000.0
