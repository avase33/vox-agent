"""The receptionist dialog policy — a slot-filling state machine.

Rather than trusting a free-form prompt to remember what it still needs, the
policy tracks explicit slots (day, time, name) and drives the conversation until
they are filled, then acts. It decides, on every turn, either to *say* something
or to *call a tool*. This same object is what the offline :class:`MockLLM`
delegates to; a hosted LLM would express the same policy implicitly through
function calling.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Optional

from ..models import ConversationState, Role, ToolCall
from . import nlu


@dataclass
class Decision:
    kind: str                     # "say" | "tool"
    text: str = ""
    tool: Optional[ToolCall] = None
    stage: str = ""
    end_call: bool = False


def _cid() -> str:
    return uuid.uuid4().hex[:8]


class ReceptionistPolicy:
    """Books appointments over voice. Deterministic and fully testable."""

    GREETING = (
        "Thanks for calling Vox Clinic, this is the automated assistant. "
        "Would you like to book an appointment?"
    )

    def greeting(self) -> str:
        return self.GREETING

    def next(self, state: ConversationState) -> Decision:
        slots = state.slots
        last = state.history[-1] if state.history else None

        # -- Phase 2: we were just handed a tool result --------------------
        if last is not None and last.role is Role.TOOL and last.tool_result is not None:
            return self._after_tool(state, last.tool_result)

        # -- Phase 1: respond to the latest user utterance -----------------
        user_text = state.last_user_text()
        parsed = nlu.parse(user_text)

        # merge freshly-heard entities
        if parsed.day:
            slots["day"] = parsed.day
        if parsed.time:
            slots["time"] = parsed.time
        if parsed.name:
            slots["name"] = parsed.name

        # Wrap-up: caller says goodbye (unless we're mid-booking).
        if "goodbye" in parsed.intents and not slots.get("booking_pending"):
            return Decision("say", "Thanks for calling. Have a great day!", "done", end_call=True)

        # Already booked on this call: start a fresh request or close out.
        if slots.get("booked"):
            if parsed.intent in ("book", "check") or parsed.day or parsed.time:
                for k in ("booked", "availability_checked", "slot_available",
                          "ready_to_book", "open_slots"):
                    slots.pop(k, None)
            elif parsed.intent == "deny":
                return Decision(
                    "say", "Thanks for calling. Have a great day!", "done", end_call=True
                )
            else:
                return Decision("say", "Is there anything else I can help with?", "collecting")

        wants_booking = (
            parsed.intent in ("book", "check")
            or slots.get("day")
            or slots.get("intent_book")
        )
        if parsed.intent in ("book", "check"):
            slots["intent_book"] = True

        if not wants_booking:
            return Decision(
                "say",
                "I can help you book an appointment. What day works for you?",
                "collecting",
            )

        # Need a day first.
        if not slots.get("day"):
            return Decision("say", "Sure — what day would you like to come in?", "collecting")

        # Have a day but no confirmed availability yet -> check the calendar.
        if not slots.get("availability_checked"):
            slots["availability_checked"] = True
            args = {"day": slots["day"]}
            if slots.get("time"):
                args["time"] = slots["time"]
            return Decision(
                "tool", tool=ToolCall("check_availability", args, _cid()), stage="checking"
            )

        # Availability known. If a specific time is open, collect name / confirm.
        if slots.get("time") and slots.get("slot_available"):
            if not slots.get("name"):
                return Decision("say", "Great, that time is open. Can I get your name?", "collecting")
            if parsed.intent == "affirm" or slots.get("ready_to_book"):
                slots["booking_pending"] = True
                return Decision(
                    "tool",
                    tool=ToolCall(
                        "create_booking",
                        {
                            "day": slots["day"],
                            "time": slots["time"],
                            "name": slots["name"],
                            "reason": slots.get("reason", "appointment"),
                        },
                        _cid(),
                    ),
                    stage="booking",
                )
            slots["ready_to_book"] = True
            return Decision(
                "say",
                f"I'll book {slots['name']} for {slots['day']} at {slots['time']}. Shall I confirm?",
                "confirming",
            )

        # We have a day but the requested time is taken / unspecified: offer slots.
        open_slots = slots.get("open_slots") or []
        if not open_slots:
            return Decision(
                "say",
                f"I'm sorry, we're fully booked on {slots['day']}. Would another day work?",
                "collecting",
            )
        # Ask the caller to pick from the open slots; reset the check for the new time.
        slots["availability_checked"] = False
        pretty = ", ".join(open_slots[:4])
        return Decision(
            "say",
            f"On {slots['day']} I have {pretty}. Which time works best?",
            "collecting",
        )

    # ---------------------------------------------------------------------
    def _after_tool(self, state: ConversationState, result) -> Decision:
        slots = state.slots
        if result.name == "check_availability":
            content = result.content or {}
            slots["open_slots"] = content.get("open_slots", [])
            if slots.get("time"):
                slots["slot_available"] = bool(content.get("available"))
                if slots["slot_available"]:
                    if not slots.get("name"):
                        return Decision("say", "That time is available! Can I get your name?", "collecting")
                    slots["ready_to_book"] = True
                    return Decision(
                        "say",
                        f"{slots['time']} on {slots['day']} is open. "
                        f"Shall I book it under {slots['name']}?",
                        "confirming",
                    )
                # requested time taken -> offer alternatives
                slots["availability_checked"] = False
                pretty = ", ".join(slots["open_slots"][:4]) or "no open times"
                return Decision(
                    "say",
                    f"{slots['time']} is taken. I do have {pretty}. Which would you like?",
                    "collecting",
                )
            # no specific time requested: list options
            slots["availability_checked"] = False
            pretty = ", ".join(slots["open_slots"][:4]) or "no open times today"
            return Decision(
                "say",
                f"On {slots['day']} I have {pretty}. Which time works best?",
                "collecting",
            )

        if result.name == "create_booking":
            content = result.content or {}
            slots["booking_pending"] = False
            if content.get("ok"):
                slots["booked"] = True
                return Decision(
                    "say",
                    f"You're all set for {content.get('day')} at {content.get('time')}. "
                    "Anything else?",
                    "done",
                )
            slots["availability_checked"] = False
            slots["slot_available"] = False
            return Decision(
                "say",
                "Hmm, that slot was just taken. Would another time work?",
                "collecting",
            )

        return Decision("say", "Okay.", "collecting")
