# IrisAI Development Makefile
PYTEST = python -m pytest
PROJ_DIR = $(shell pwd)
PYTHONPATH = $(PROJ_DIR)

.PHONY: test test-unit test-runtime test-e2e test-all help

help:
	@echo "IrisAI Test Targets:"
	@echo "  make test        — Unit tests only (fast, no LLM)"
	@echo "  make test-runtime — Runtime + PEL + protocol tests (no LLM)"
	@echo "  make test-e2e    — Live E2E tests (REQUIRED after functional changes)"
	@echo "  make test-all    — Full suite (unit + runtime + e2e)"
	@echo ""
	@echo "LITELLM_VIRTUAL_KEY must be set for e2e tests (always available in IrisAI session)"

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_unit.py -v

test-runtime:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_runtime.py tests/test_pel_functional.py tests/test_pel_approval.py tests/test_protocol.py -v

test-e2e:
	@echo "🔴 Running LIVE E2E tests with real Claude on Bedrock..."
	@echo "   Requires: LITELLM_VIRTUAL_KEY (always available in IrisAI session)"
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/test_e2e.py tests/test_worker_agent_runtime.py tests/test_sub_agent.py -v -m "e2e"

test-all:
	PYTHONPATH=$(PYTHONPATH) $(PYTEST) tests/ -q
