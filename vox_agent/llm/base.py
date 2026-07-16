"""LLM provider interface.

An LLM streams a response for the current :class:`ConversationState`, yielding
either :class:`Token` deltas (speak these) or a :class:`ToolCall` (run this, then
call ``stream`` again with the tool result appended to history). The pipeline
drives that loop. Real providers express tool use natively; the mock delegates to
the deterministic dialog policy.
"""

from __future__ import annotations

import abc
from typing import Any, AsyncIterator, Union

from ..models import ConversationState, Token, ToolCall

LLMEvent = Union[Token, ToolCall]


class BaseLLM(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    async def stream(
        self, state: ConversationState, tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        raise NotImplementedError
        yield  # pragma: no cover

    async def aclose(self) -> None:  # pragma: no cover
        return None
