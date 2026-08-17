# RAG 2.0 测试与质量门禁
# 本地: make test | make lint
# CI:   make test-ci   (带覆盖率门禁, 低于 80% 失败)

.PHONY: test test-ci lint format test-frontend

test:
	pytest -p no:cacheprovider

test-ci:
	pytest -m "not slow and not integration" --cov=api --cov=engines --cov-report=term-missing --cov-fail-under=80 -p no:cacheprovider

test-frontend:
	node tests/test_frontend_safety.mjs

lint:
	ruff check .

format:
	ruff format .
