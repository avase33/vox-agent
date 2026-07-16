from vox_agent.models import ToolCall
from vox_agent.tools import CalendarDB, ToolRegistry


def test_seeded_slots_excluded_from_availability():
    reg = ToolRegistry()
    r = reg.run(ToolCall("check_availability", {"day": "tuesday"}, "1"))
    assert r.ok
    # 10:00 and 14:00 are seeded as taken on the next Tuesday
    assert "10:00" not in r.content["open_slots"]
    assert "09:00" in r.content["open_slots"]


def test_specific_time_availability():
    reg = ToolRegistry()
    r = reg.run(ToolCall("check_availability", {"day": "tuesday", "time": "10am"}, "1"))
    assert r.content["available"] is False
    r2 = reg.run(ToolCall("check_availability", {"day": "tuesday", "time": "9am"}, "2"))
    assert r2.content["available"] is True


def test_create_booking_persists_and_blocks_double_book():
    reg = ToolRegistry()
    b = reg.run(ToolCall("create_booking", {"day": "wednesday", "time": "11am", "name": "Jo"}, "1"))
    assert b.ok and b.content["ok"] and b.content["time"] == "11:00"
    b2 = reg.run(ToolCall("create_booking", {"day": "wednesday", "time": "11:00", "name": "Al"}, "2"))
    assert b2.content["ok"] is False and b2.content["reason"] == "slot_taken"


def test_unknown_tool_and_bad_args():
    reg = ToolRegistry()
    assert reg.run(ToolCall("nope", {}, "x")).ok is False
    assert reg.run(ToolCall("check_availability", {}, "y")).ok is False


def test_time_normalisation():
    db = CalendarDB(seed=False)
    assert db.book("friday", "3pm", "X")["time"] == "15:00"
    assert db.book("friday", "9:30am", "Y")["time"] == "09:30"
    assert db.book("friday", "16:00", "Z")["time"] == "16:00"
