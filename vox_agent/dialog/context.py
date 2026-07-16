"""Context window management with summarisation-based pruning.

Long voice calls pile up turns fast, which slows the LLM. We keep the most recent
``keep_recent`` turns verbatim and compress everything older into a one-line
extractive summary (key entities + decisions). The compressor here is heuristic
so it needs no model; swap :func:`summarise` for an LLM call in production.
"""

from __future__ import annotations

from ..models import ConversationState, Role, Turn


def summarise(turns: list[Turn]) -> str:
    """Extractive summary: collect slots mentioned + tool outcomes."""
    facts: list[str] = []
    for t in turns:
        if t.role is Role.TOOL and t.tool_result is not None and t.tool_result.ok:
            facts.append(f"tool {t.tool_result.name} -> {t.tool_result.content}")
        elif t.role is Role.USER and t.content:
            snippet = t.content.strip()
            if len(snippet) > 60:
                snippet = snippet[:57] + "..."
            facts.append(f"user: {snippet}")
    if not facts:
        return ""
    return "Earlier in the call: " + "; ".join(facts[-6:])


def build_prompt_turns(state: ConversationState, keep_recent: int = 10) -> list[Turn]:
    """Return a pruned list of turns suitable for the LLM context.

    If history is short, return it unchanged. Otherwise summarise the old part
    into a single system turn and keep the recent tail verbatim.
    """
    history = state.history
    if len(history) <= keep_recent:
        return list(history)

    old, recent = history[:-keep_recent], history[-keep_recent:]
    summary = state.summary
    fresh = summarise(old)
    if fresh:
        summary = (summary + " " + fresh).strip() if summary else fresh
    state.summary = summary

    pruned: list[Turn] = []
    if summary:
        pruned.append(Turn(role=Role.SYSTEM, content=summary))
    pruned.extend(recent)
    return pruned
