"""Lightweight natural-language understanding for the offline brain.

Regex/keyword intent detection and entity extraction (day, time, name). A real
deployment lets the LLM do this implicitly via function calling; here it powers
the deterministic mock policy and doubles as a fast pre-parser you could feed the
LLM as hints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
         "today", "tomorrow"]
_TIME_RE = re.compile(r"\b(\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{1,2}:\d{2}|\d{1,2}\s*o'?clock)\b", re.I)
_NAME_RE = re.compile(r"\b(?:my name is|i am|i'm|this is|it's|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", re.I)

_INTENT_KEYWORDS = {
    "book": ["book", "schedule", "appointment", "reserve", "set up a", "make an"],
    "check": ["available", "availability", "free", "open", "any slots", "what times"],
    "cancel": ["cancel", "reschedule", "change my"],
    "affirm": ["yes", "yeah", "yep", "sure", "correct", "that works", "sounds good", "please do"],
    "deny": ["no", "nope", "not really", "different", "another"],
    "greet": ["hello", "hi", "hey", "good morning", "good afternoon"],
    "goodbye": ["bye", "goodbye", "that's all", "thanks", "thank you", "nothing else"],
}


@dataclass
class NLU:
    intent: str = "other"
    day: Optional[str] = None
    time: Optional[str] = None
    name: Optional[str] = None
    intents: list[str] = field(default_factory=list)


def parse(text: str) -> NLU:
    low = text.lower().strip()
    intents: list[str] = []
    for intent, kws in _INTENT_KEYWORDS.items():
        if any(k in low for k in kws):
            intents.append(intent)

    day = next((d for d in _DAYS if re.search(r"\b" + d + r"\b", low)), None)

    tm = _TIME_RE.search(text)
    time = tm.group(1).strip() if tm else None

    nm = _NAME_RE.search(text)
    name = nm.group(1).strip() if nm else None

    # priority: booking-ish intents win over generic greet/goodbye
    primary = "other"
    for pref in ("book", "check", "cancel", "affirm", "deny", "goodbye", "greet"):
        if pref in intents:
            primary = pref
            break
    return NLU(intent=primary, day=day, time=time, name=name, intents=intents)
