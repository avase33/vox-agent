from vox_agent.dialog.pii import redact


def test_redacts_credit_card_keeps_last4():
    r = redact("my card is 4111 1111 1111 1111 thanks")
    assert "[CARD_1]" in r.text
    assert "4111 1111 1111 1111" not in r.text
    assert any(k.endswith(":last4") and v == "1111" for k, v in r.vault.items())
    assert "CARD" in r.redactions


def test_redacts_phone_and_email():
    r = redact("call me at 415-555-0198 or email a.b+x@example.com")
    assert "[PHONE_1]" in r.text
    assert "[EMAIL_1]" in r.text
    assert "415-555-0198" not in r.text


def test_redacts_ssn():
    r = redact("my social is 123-45-6789")
    assert "[SSN_1]" in r.text
    assert "123-45-6789" not in r.text


def test_leaves_ordinary_text_untouched():
    r = redact("book tuesday at 3pm for two people")
    assert r.text == "book tuesday at 3pm for two people"
    assert not r.redactions
