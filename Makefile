# RAG 2.0 测试与质量门禁
# 本地: make test | make lint | make validate
# CI:   make test-ci   (带覆盖率门禁, 低于 80% 失败)

.PHONY: test test-ci eval-routing eval-preguard eval-preguard-real eval-fanout-ab eval-gate eval-ab lint format test-frontend validate check-size all

test:
	pytest -p no:cacheprovider

# 本地快跑: 跳过 slow/integration (与 CI 同口径, 不含覆盖率, 秒级反馈)
test-fast:
	pytest -m "not slow and not integration" -p no:cacheprovider

test-ci:
	pytest -m "not slow and not integration" --cov=api --cov=engines --cov-report=term-missing --cov-fail-under=80 -p no:cacheprovider

# 路由回归门禁: 纯规则、零外部依赖, 固定 12 道黄金路由题 (data/_eval_p1_subset.json),
# 断言 gold_kb ∈ 候选集。防路由/关键词/数据归属改动打回已知正确路由。
eval-routing:
	pytest tests/test_routing_eval_regression_pytest.py -p no:cacheprovider -q

# 生成前护栏(low_relevance)离线误拒/拦截量化 — 语义兜底阈值(0.50)的复现依据,
# 见 scripts/eval_preguard.py 与 api/config.py answerability_preguard_enabled 注释。
eval-preguard:
	python scripts/eval_preguard.py

# 残余风险实证: 真实检索链路重放, 确认 low_relevance 无独有新增误拒
# (G1"完美命中假设"下的边缘样本在真实检索中被 low_confidence 覆盖/本属正确拒答)。
# 依赖本地模型+语料, 不进 CI; 结论变化时人工复核 scripts/eval_preguard_realretrieval.py。
eval-preguard-real:
	python scripts/eval_preguard_realretrieval.py

# #3 跨库 rerank 端到端 A/B: 12 道真实路由题 × OFF(重排关)/GLOBAL(#3 全局重排)/PER-KB(旧逐库)
# 三策略 gold 命中对比; --cap 可覆盖候选池预算(验证 10 vs 20 对跨库重排价值的影响)。
eval-fanout-ab:
	python scripts/eval_fanout_rerank_ab.py

# 召回回归门禁: 首阶段(--no-rerank) 390 问约 45s, 比对基线, 指标回退超容差
# (比率 1.0pp / MRR 0.01) 则 exit 1。全量重排档 CPU 上约 70min, 不进 CI。
#
# 语料(corpus_chunks.json, 19MB 文档派生数据)未入库 —— 由自托管 runner 挂载提供, 或本地自备。
# 路径可用变量覆盖, 适配挂载点:
#   make eval-gate EVAL_DIR=/mnt/rag-eval            # 语料挂载目录(需含 corpus_chunks.json + eval_dataset.json)
#   make eval-gate EVAL_DIR=/mnt/rag-eval CHUNKS=my_corpus.json
# 基线 BASELINE 默认走仓库内路径 —— 它是随代码演进的契约, 必须版本控制, 不随挂载点漂移。
EVAL_DIR ?= data/eval
CHUNKS ?= corpus_chunks.json
DATASET ?= eval_dataset.json
BASELINE ?= data/eval/baseline_retrieval_first_stage.json
GATE_OUT ?= .eval-gate-current.json

eval-gate:
	@test -f "$(EVAL_DIR)/$(CHUNKS)" || \
		{ echo "缺少语料 $(EVAL_DIR)/$(CHUNKS) — 召回门禁无法运行(语料未入库, 需挂载或本地自备)"; exit 1; }
	python scripts/eval_retrieval.py --no-rerank \
		--eval-dir "$(EVAL_DIR)" --chunks "$(CHUNKS)" --dataset "$(DATASET)" \
		--out "$(GATE_OUT)"
	python scripts/eval_gate.py --baseline "$(BASELINE)" --current "$(GATE_OUT)"

# P2#10 通用评估 A/B 对比: 两份 evaluate.py 产出的 eval-summary.json 指标 diff,
# 回退指标打 ↓ 标记并汇总。用法: make eval-ab A=data/eval-A.json B=data/eval-B.json
eval-ab:
	python scripts/ab_eval.py --a $(A) --b $(B)

test-frontend:
	node tests/test_frontend_safety.mjs

check-size:
	@total=$$(wc -c web/index.html web/admin.html web/common.js web/icons.js web/common.css | tail -1 | awk '{print $$1}'); \
	echo "前端总大小: $$total bytes"; \
	if [ "$$total" -gt 153600 ]; then \
		echo "FAIL: 前端资源超过 150KB 预算 ($${total} bytes)"; exit 1; \
	else \
		echo "PASS: 在预算内 ($${total}/153600 bytes)"; \
	fi

validate: lint format test-frontend check-size
	@echo ""
	@echo "=== 完整质量门禁通过 ==="

# lint 范围=产品核心(api/engines/tests), 与 CI 门禁一致并阻断; scripts/web 辅助代码自查即可
lint:
	ruff check api engines tests

format:
	ruff format .

all: test-ci eval-routing test-frontend check-size
	@echo "=== 全部检查通过 ==="
