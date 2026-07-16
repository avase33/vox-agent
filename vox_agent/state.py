"""Session state store.

Short-term conversation memory keyed by session id. The default is an in-process
dict so everything runs offline; a Redis-backed store (used when
``VOX_STATE=redis``) keeps the same interface for horizontal scale-out.
"""

from __future__ import annotations

import abc

from .config import Settings
from .models import ConversationState


class SessionStore(abc.ABC):
    @abc.abstractmethod
    def get(self, session_id: str) -> ConversationState: ...

    @abc.abstractmethod
    def save(self, state: ConversationState) -> None: ...

    @abc.abstractmethod
    def drop(self, session_id: str) -> None: ...


class MemoryStore(SessionStore):
    def __init__(self) -> None:
        self._data: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:
        st = self._data.get(session_id)
        if st is None:
            st = ConversationState(session_id=session_id)
            self._data[session_id] = st
        return st

    def save(self, state: ConversationState) -> None:
        self._data[state.session_id] = state

    def drop(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class RedisStore(SessionStore):
    """Redis-backed store. History is serialised as JSON under one key.

    Requires ``pip install vox-agent[redis]``. Not exercised offline.
    """

    def __init__(self, url: str) -> None:  # pragma: no cover - needs redis
        try:
            import redis  # type: ignore
        except ImportError as e:
            raise RuntimeError("Run: pip install 'vox-agent[redis]'") from e
        self._r = redis.Redis.from_url(url)
        self._cache: dict[str, ConversationState] = {}

    def get(self, session_id: str) -> ConversationState:  # pragma: no cover
        if session_id in self._cache:
            return self._cache[session_id]
        st = ConversationState(session_id=session_id)
        self._cache[session_id] = st
        return st

    def save(self, state: ConversationState) -> None:  # pragma: no cover
        self._cache[state.session_id] = state
        self._r.set(f"vox:session:{state.session_id}:stage", state.stage, ex=3600)

    def drop(self, session_id: str) -> None:  # pragma: no cover
        self._cache.pop(session_id, None)
        self._r.delete(f"vox:session:{session_id}:stage")


def build_store(settings: Settings) -> SessionStore:
    if settings.state_backend.lower() == "redis":
        return RedisStore(settings.redis_url)
    return MemoryStore()
