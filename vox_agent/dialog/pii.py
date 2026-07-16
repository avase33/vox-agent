"""PII redaction middleware.

Callers say credit-card and phone numbers out loud. This scrubs them from the
transcript *before* it reaches the LLM (or any log), replacing them with typed
placeholders while remembering the real value locally so the agent can still
confirm the last four digits. Order matters: match the most specific patterns
(cards, SSNs) before generic long-digit runs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Spoken numbers frequently arrive with spaces between digit groups, so the
# patterns tolerate spaces/dashes.
_D = r"[\s-]?"
_CARD = re.compile(r"\b(?:\d%s){12,15}\d\b" % _D)          # 13-16 digits
_SSN = re.compile(r"\b\d{3}%s\d{2}%s\d{4}\b" % (_D, _D))    # xxx-xx-xxxx
_PHONE = re.compile(r"\b(?:\+?\d{1,2}%s)?\(?\d{3}\)?%s\d{3}%s\d{4}\b" % (_D, _D, _D))
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")


@dataclass
class RedactionResult:
    text: str
    redactions: list[str] = field(default_factory=list)
    vault: dict[str, str] = field(default_factory=dict)  # placeholder -> raw


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def redact(text: str) -> RedactionResult:
    vault: dict[str, str] = {}
    found: list[str] = []
    counters = {"CARD": 0, "SSN": 0, "PHONE": 0, "EMAIL": 0}

    def _sub(pattern: re.Pattern[str], label: str, s: str) -> str:
        def repl(m: re.Match[str]) -> str:
            counters[label] += 1
            token = f"[{label}_{counters[label]}]"
            raw = m.group(0)
            vault[token] = raw
            if label in ("CARD", "SSN", "PHONE"):
                last4 = _digits(raw)[-4:]
                vault[token + ":last4"] = last4
            found.append(label)
            return token

        return pattern.sub(repl, s)

    out = _sub(_CARD, "CARD", text)
    out = _sub(_SSN, "SSN", out)
    out = _sub(_PHONE, "PHONE", out)
    out = _sub(_EMAIL, "EMAIL", out)
    return RedactionResult(text=out, redactions=found, vault=vault)
