"""Tool registry: JSON schemas + dispatch.

The schemas are the exact ``tools`` payload you would hand to Groq/OpenAI for
function calling. The registry also executes a :class:`~vox_agent.models.ToolCall`
against the backing implementation, returning a :class:`ToolResult`. This is the
layer that lets the agent *do things*, not just talk.
"""

from __future__ import annotations

from typing import Any, Callable

from ..models import ToolCall, ToolResult
from .calendar_db import CalendarDB

# OpenAI/Groq-compatible function schemas.
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check which appointment slots are open on a given day, "
            "optionally testing a specific time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {
                        "type": "string",
                        "description": "Day such as 'tuesday', 'tomorrow', or an ISO date.",
                    },
                    "time": {
                        "type": "string",
                        "description": "Optional specific time like '3pm' or '15:00'.",
                    },
                },
                "required": ["day"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_booking",
            "description": "Book an appointment in an open slot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "day": {"type": "string"},
                    "time": {"type": "string"},
                    "name": {"type": "string", "description": "Caller's name."},
                    "reason": {"type": "string", "description": "Reason for the visit."},
                },
                "required": ["day", "time", "name"],
            },
        },
    },
]


class ToolRegistry:
    def __init__(self, db: CalendarDB | None = None) -> None:
        self.db = db or CalendarDB()
        self._impls: dict[str, Callable[..., Any]] = {
            "check_availability": self._check_availability,
            "create_booking": self._create_booking,
        }

    @property
    def schemas(self) -> list[dict[str, Any]]:
        return TOOL_SCHEMAS

    def run(self, call: ToolCall) -> ToolResult:
        impl = self._impls.get(call.name)
        if impl is None:
            return ToolResult(call.call_id, call.name, None, ok=False, error="unknown_tool")
        try:
            content = impl(**call.arguments)
            return ToolResult(call.call_id, call.name, content, ok=True)
        except TypeError as e:
            return ToolResult(call.call_id, call.name, None, ok=False, error=f"bad_args: {e}")
        except Exception as e:  # pragma: no cover - defensive
            return ToolResult(call.call_id, call.name, None, ok=False, error=str(e))

    # -- implementations --------------------------------------------------
    def _check_availability(self, day: str, time: str | None = None) -> dict:
        return self.db.check_availability(day, time)

    def _create_booking(
        self, day: str, time: str, name: str, reason: str = ""
    ) -> dict:
        return self.db.book(day, time, name, reason)
