"""OpenAI-compatible streaming LLM adapter (covers OpenAI and Groq).

Groq's SDK mirrors OpenAI's chat-completions API, so both share this
implementation — only the client construction differs (see the factory in
``vox_agent.llm.__init__``). Streams token deltas and assembles ``tool_calls``
across chunks, yielding a :class:`ToolCall` once a function call completes.

Not exercised by the offline test suite (needs a network + key), but written to
be correct against the documented streaming schema.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..models import ConversationState, Role, Token, ToolCall
from .base import BaseLLM, LLMEvent
from .prompts import SYSTEM_PROMPT


def _to_messages(state: ConversationState) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if state.summary:
        messages.append({"role": "system", "content": state.summary})
    for t in state.history:
        if t.role is Role.USER:
            messages.append({"role": "user", "content": t.content})
        elif t.role is Role.ASSISTANT:
            if t.tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": t.content or None,
                        "tool_calls": [
                            {
                                "id": c.call_id or "call_%d" % i,
                                "type": "function",
                                "function": {
                                    "name": c.name,
                                    "arguments": json.dumps(c.arguments),
                                },
                            }
                            for i, c in enumerate(t.tool_calls)
                        ],
                    }
                )
            else:
                messages.append({"role": "assistant", "content": t.content})
        elif t.role is Role.TOOL and t.tool_result is not None:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": t.tool_result.call_id or "call_0",
                    "content": json.dumps(t.tool_result.content),
                }
            )
    return messages


class OpenAICompatLLM(BaseLLM):
    def __init__(self, client: Any, model: str, name: str = "openai") -> None:
        self._client = client
        self._model = model
        self.name = name

    async def stream(
        self, state: ConversationState, tools: list[dict[str, Any]]
    ) -> AsyncIterator[LLMEvent]:  # pragma: no cover - needs network + key
        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=_to_messages(state),
            tools=tools or None,
            stream=True,
            temperature=0.4,
            max_tokens=200,
        )
        # Accumulate any streamed tool call across chunks.
        pending: dict[int, dict[str, Any]] = {}
        async for chunk in stream:
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                yield Token(text=delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = pending.setdefault(tc.index, {"name": "", "args": "", "id": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] += tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments
        for slot in pending.values():
            try:
                args = json.loads(slot["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield ToolCall(name=slot["name"], arguments=args, call_id=slot["id"])
