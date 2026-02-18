.PHONY: help setup run run-chainlit test lint format build check eval validate-groq

PYTHON ?= python

help:
	@echo "Auto-Analyst harness commands:"
	@echo "  make setup        Install dependencies"
	@echo "  make run          Run Streamlit UI"
	@echo "  make run-chainlit Run Chainlit UI"
	@echo "  make lint         Run syntax checks (py_compile)"
	@echo "  make test         Run full pytest suite"
	@echo "  make check        Run lint + test"
	@echo "  make build        Run build-equivalent integrity check"
	@echo "  make format       Formatter status (informational)"
	@echo "  make eval         Run evaluation script"
	@echo "  make validate-groq Validate Groq key/model accessibility"

setup:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

run:
	$(PYTHON) -m streamlit run ui/app.py

run-chainlit:
	chainlit run ui/chainlit_app.py -w

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m py_compile $$(find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*")

format:
	@echo "No repo-managed formatter is enforced. 'black'/'ruff' are documented but not pinned in this repository."

build: lint
	@echo "Build check complete via syntax integrity (no packaging build is configured)."

check: lint test

eval:
	$(PYTHON) evaluation/run_evaluation.py --dataset data/sample_eval.json --model all-MiniLM-L6-v2

validate-groq:
	$(PYTHON) -m tools.provider_health --provider groq --timeout 8
