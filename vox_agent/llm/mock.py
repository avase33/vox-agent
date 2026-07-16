"""Offline mock LLM.

Streams tokens and emits tool calls exactly like a hosted model would, but its
"reasoning" is the deterministic :class:`ReceptionistPolicy`. This keeps the
whole agent runnable with zero API keys while exercising the real streaming +
function-calling control flow the pipeline depends on.
"""

from __future__ import annotations

import asyncio
from typing import Any, AsyncIterator

from ..dialog.state_machine import ReceptionistPolicy
from ..models import ConversationState, Role, Token
from .base import BaseLLM, LLMEvent


class MockLLM(BaseLLM):
    name = "mock"

    def __init__(self, token_delay_s: float = 0.006) -> None:
        self.policy = ReceptionistPolicy()
        self.token_delay_s = token_delay_s

    async def stream(
        self, state: ConversationState, tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:
        # Opening line: agent greets before the caller has said anything.
        has_user = any(t.role is Role.USER for t in state.history)
        has_assistant = any(t.role is Role.ASSISTANT for t in state.history)
        if not has_user and not has_assistant:
            async for tok in self._say(self.policy.greeting()):
                yield tok
            state.stage = "greeted"
            return

        decision = self.policy.next(state)
        state.stage = decision.stage or state.stage
        if decision.end_call:
            state.slots["end_call"] = True

        if decision.kind == "tool" and decision.tool is not None:
            yield decision.tool
            return

        async for tok in self._say(decision.text):
            yield tok

    async def _say(self, text: str) -> AsyncIterator[Token]:
        for word in text.split():
            await asyncio.sleep(self.token_delay_s)
            yield Token(text=word + " ")
