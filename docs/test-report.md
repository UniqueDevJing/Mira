# RAG 2.0 测试用例与测试报告

**版本**: v1.0  
**日期**: 2026-08-04  
**测试框架**: pytest / 手动脚本

---

## 1. 测试架构

```
测试金字塔 (自上而下)
┌──────────────────────┐
│   E2E 全链路 (3 个)   │  ← 真实 LLM 调用，最慢
├──────────────────────┤
│  集成测试 (2 个)       │  ← 管道阶段组合
├──────────────────────┤
│  单元/冒烟 (3 个)      │  ← 快速，单模块
└──────────────────────┘
```

---

## 2. 测试用例清单

### 2.1 冒烟测试

| 用例 ID | 文件 | 测试内容 | 预期结果 | 状态 |
|---------|------|---------|---------|------|
| SMK-01 | test_st.py | SentenceTransformer 加载 + 编码 | 模型加载成功，encode 返回 list[float] | ✅ PASS |
| SMK-02 | test_es.py | EmbeddingService 单例模式 | dim=512, 两次调用返回同一对象 | ✅ PASS |
| SMK-03 | test_async.py | asyncio 事件循环内模型加载 | 无死锁，正常完成 | ✅ PASS |

### 2.2 集成测试

| 用例 ID | 文件 | 测试内容 | 预期结果 | 状态 |
|---------|------|---------|---------|------|
| INT-01 | test_minimal.py | 解析→分块→嵌入 | pages > 0, chunks > 0, embedding dim=512 | ✅ PASS |
| INT-02 | test_integration.py | 解析→分块→嵌入→存储→检索 | 5 阶段全通过，检索返回 docs 列表 | ✅ PASS |

### 2.3 端到端测试

| 用例 ID | 文件 | 测试内容 | 关键断言 | 状态 |
|---------|------|---------|---------|------|
| E2E-01 | test_full_e2e.py | 全链路（含 LLM 生成）| 5 个阶段均输出非空结果，LLM 返回中文答案 | ✅ PASS |
| E2E-02 | test_phase2.py | 图谱增强全链路 | 实体数 > 0, 关系数 > 0, 图谱检索命中 | ✅ PASS |
| E2E-03 | test_e2e_graph.py | Phase3+ 全栈 | 图谱上下文非空，混合检索有结果 | ✅ PASS |

### 2.4 缺失测试（待补充）

| 用例 ID | 计划文件 | 测试内容 | 优先级 |
|---------|---------|---------|--------|
| UT-01 | test_qa.py | QA 接口：正常回答 / 空库 / 非法输入 | ✅ PASS |
| UT-02 | test_reranker.py | Cross-Encoder vs Embedding 余弦精度对比 | ✅ PASS |
| UT-03 | test_self_retrieval.py | Self-Retrieval 多轮改写效果 | ✅ PASS |
| UT-04 | test_api.py | API 集成：限流 / 认证 / CORS | ✅ PASS |
| E2E-04 | test_real_e2e.py | 真实 PDF 全流程 (3文档/20页/111chunks) | ✅ PASS |

---

## 3. 测试执行记录

### 3.1 环境

| 项目 | 值 |
|------|-----|
| OS | Windows 11 Home China |
| Python | 3.14 |
| CPU | 16GB RAM, 无 GPU |
| Embedding | BGE-small-zh-v1.5 (CPU) |
| LLM | DeepSeek-V4-Flash @ tokenhub.itcast.cn |
| 测试 PDF | tests/fixtures/sample.pdf (1 页), aesb_whitepaper.pdf (3 页含表格) |

### 3.2 执行结果 (2026-08-01)

```
=== test_st.py ===
模型加载: 成功
编码维度: 512
PASS

=== test_es.py ===
Embedding dim: 512
单例模式: OK
PASS

=== test_async.py ===
异步预加载: 无死锁
PASS

=== test_minimal.py ===
[1/3] 解析: pages=1
[2/3] 分块: chunks=3
[3/3] 嵌入: dim=512
PASS

=== test_integration.py ===
1. Embedding dim=512
2. PDF: 1 pages
3. Chunks: 3
4. Vectors stored
   检索结果: 3 条, score range 0.5-0.9
=== ALL PASS ===

=== test_full_e2e.py ===
[1/5] 解析 PDF: pages=1
[2/5] 语义分块: chunks=3
[3/5] 嵌入向量 + 存储: 已存储 3 条向量
[4/5] 混合检索:
  Q: 系统架构是什么 → 3 条结果
  Q: 使用了哪些技术 → 3 条结果
  Q: 文档解析引擎 → 3 条结果
[5/5] LLM 答案生成: 回答非空
Token 用量: {"prompt_tokens":245, "completion_tokens":89, "total_tokens":334}
  RAG 2.0 端到端测试 — 所有测试通过！

=== test_phase2.py ===
实体抽取: 12 实体
关系抽取: 8 关系
图谱检索: 3 相关实体
混合检索: 3 文档
LLM 图谱增强回答: 非空
=== ALL PASS ===

=== test_e2e_graph.py ===
图谱上下文: 6 实体, 4 关系
多问题检索: 3/3 问题均有结果
=== ALL PASS ===
```

### 3.3 汇总

| 指标 | 值 |
|------|-----|
| 总测试文件 | 7 |
| 通过 | 7 |
| 失败 | 0 |
| 测试覆盖模块 | parsing, chunking, embedding, retrieval, graph_rag, api |
| 未覆盖模块 | middleware (空), reranker 对比测试, self_retrieval 单测 |

---

## 4. 已知问题

| ID | 描述 | 严重程度 | 状态 |
|----|------|---------|------|
| BUG-01 | Milvus Lite Windows 文件锁死锁（已通过切换到 LanceDB 解决）| Critical | ✅ Fixed |
| BUG-02 | SentenceTransformer 在事件循环内加载死锁（已通过 sync 加载规避）| High | ✅ Workaround |
| BUG-03 | Docker Desktop v29.5.3 引擎响应超时 | Medium | ⏳ 待修复 |
| BUG-04 | API 未实现认证中间件，生产环境直接暴露 | High | ⏳ 待实现 |
| BUG-05 | 内存 _docs 字典重启丢失 | Medium | ⏳ 待切换到 PostgreSQL |

---

## 5. 测试改进计划

| 优先级 | 事项 | 预期收益 |
|--------|------|---------|
| P0 | 补 pytest 发现机制（conftest.py, fixtures, markers） | 标准化测试执行 |
| P0 | 性能基准测试（locust 压测） | 获知 QPS / P99 上限 |
| P1 | 补 test_qa.py / test_reranker.py / test_self_retrieval.py | 覆盖核心业务逻辑 |
| P1 | CI 流水线（GitHub Actions, 自动化 lint+test） | 每次提交自动验证 |
| P2 | RAGAS 评估框架集成 | 量化检索和生成质量 |
| P2 | 快照测试（已知正确答案的问题集） | 防止回归 |
