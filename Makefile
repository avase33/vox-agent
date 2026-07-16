.PHONY: install dev test verify demo chat bench serve lint clean

install:
	pip install -e .

dev:
	pip install -e ".[server,dev]"

test:
	pytest -q

verify:
	python scripts/verify_full.py

verify-tiny:
	python scripts/verify_tiny.py

demo:
	python -m vox_agent.cli demo

chat:
	python -m vox_agent.cli chat

bench:
	python -m vox_agent.cli bench -n 20

serve:
	python -m vox_agent.cli serve

lint:
	ruff check vox_agent tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__ *.wav *.db
