"""A tiny jitter buffer for smooth playback over lossy networks.

Real internet audio arrives out of order and clumped. Playing bytes the instant
they land makes the voice sound robotic. This buffer reorders by sequence number
and only releases audio once a small ``target_ms`` cushion has built up, then
keeps the cushion topped up. Missing packets past the wait budget are skipped so
playback never stalls forever.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass(order=True)
class _Packet:
    seq: int
    data: bytes = field(compare=False)


class JitterBuffer:
    def __init__(self, frame_ms: int = 20, target_ms: int = 60, max_ms: int = 400) -> None:
        self.frame_ms = max(1, frame_ms)
        self.target_frames = max(1, target_ms // self.frame_ms)
        self.max_frames = max(self.target_frames, max_ms // self.frame_ms)
        self._heap: list[_Packet] = []
        self._next_seq = 0
        self._primed = False
        self.dropped = 0

    def push(self, seq: int, data: bytes) -> None:
        if seq < self._next_seq:
            self.dropped += 1  # too late, already played past this point
            return
        heapq.heappush(self._heap, _Packet(seq, data))
        if len(self._heap) > self.max_frames:
            self._primed = True  # over budget: stop waiting, start draining

    def pop(self) -> bytes | None:
        """Return the next in-order frame, or None if we should keep buffering."""
        if not self._primed:
            if len(self._heap) < self.target_frames:
                return None
            self._primed = True

        if not self._heap:
            self._primed = False
            return None

        head = self._heap[0]
        if head.seq == self._next_seq:
            heapq.heappop(self._heap)
            self._next_seq += 1
            return head.data
        # Gap: if we've waited long enough, skip the missing packet.
        if len(self._heap) >= self.max_frames:
            self._next_seq = head.seq
            heapq.heappop(self._heap)
            self._next_seq += 1
            self.dropped += 1
            return head.data
        return None

    def flush(self) -> None:
        self._heap.clear()
        self._primed = False

    def __len__(self) -> int:
        return len(self._heap)
