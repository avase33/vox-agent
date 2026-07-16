from vox_agent.dialog.context import build_prompt_turns
from vox_agent.dialog.nlu import parse
from vox_agent.models import ConversationState, Role, Turn


def test_parse_booking_entities():
    n = parse("I'd like to book an appointment on Tuesday at 3pm")
    assert n.intent == "book"
    assert n.day == "tuesday"
    assert n.time and "3" in n.time


def test_parse_name():
    assert parse("My name is Jordan Lee").name == "Jordan Lee"
    assert parse("this is Sam").name == "Sam"


def test_parse_affirm_deny_goodbye():
    assert parse("yes please").intent == "affirm"
    assert parse("no, a different time").intent in ("deny", "check")
    assert parse("that's all, thank you").intent == "goodbye"


def test_context_pruning_creates_summary():
    st = ConversationState("s")
    for i in range(30):
        st.add(Turn(Role.USER, f"line {i}"))
    pruned = build_prompt_turns(st, keep_recent=10)
    assert len(pruned) <= 12
    assert st.summary
    # recent turns are preserved verbatim
    assert pruned[-1].content == "line 29"


def test_context_short_history_untouched():
    st = ConversationState("s")
    st.add(Turn(Role.USER, "hello"))
    pruned = build_prompt_turns(st, keep_recent=10)
    assert len(pruned) == 1
    assert not st.summary
