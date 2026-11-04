# M4：GraphRAG 可证伪化 —— 关系路径正确率评测与收益开关

## 为什么做这件事

关系推理（GraphRAG）是 RAG 系统里最容易"变玄学"的部分：架构图上画着多跳推理，
但"图谱到底带来了什么增量"如果不拿数字证明，它就只是 PPT 上的一格。
**敢对自家功能做可证伪评测、并保留"把它关掉"的开关，比堆功能更有信号量。**

M4 的交付物不是一个新功能，而是三个：

1. **`RAG_GRAPH_ENABLED` 总开关**（`api/config.py`，默认 `true`）：
   关闭后 hybrid 跳过图谱腿、请求级 `mode=graph` 降级为 hybrid——零代码改动即可关闭图谱。
2. **关系路径正确率评测**（`scripts/eval_graph_path.py` + `scripts/graph_path_set.json`）：
   纯离线、零 LLM 成本，量化"图谱路径是否真的被走通、带来了哪些增量召回"。
3. **一条写死的结论规则**：多跳问题增量召回 > 0 且路径正确率 ≥ 0.5 → 图谱默认保留；
   否则建议 `RAG_GRAPH_ENABLED=false`。结论由数字给出，不由感觉给出。

## 评测设计

评测集（`graph_path_set.json`）刻意做成规则抽取友好、零 LLM 依赖：

```
c1: TokenHub 网关使用 Redis 做限流计数。      → TokenHub -uses→ Redis
c2: Redis 的快照持久化依赖 MinIO。            → Redis -depends_on→ MinIO
c3: MinIO 集群使用 Docker Compose 部署。      → MinIO -uses→ Docker
```

四类问题各司其职：

| 问题 | 类型 | 验证什么 |
|---|---|---|
| q1/q2（多跳） | 证据 chunk（c2）与问题**词面零重叠** | 关键词基线只能命中种子 chunk，必须靠 `TokenHub→Redis→MinIO` 两跳才能召回——图谱的增量在这里才成立 |
| q3（单跳同 chunk） | 答案与问题词面共现 | 图谱增量应为 **0**（诚实对照：单跳共现本就够用，图谱不抢功） |
| q4（负控制） | 与语料无关 | 图谱不应有任何实体/chunk 产出（多管闲事按 0 计） |

## 实测结果（本仓库评测集，离线可复现）

```bash
python scripts/eval_graph_path.py
# 路径正确率        : 1.0        （预期三元组被遍历精确还原）
# macro 召回 图开/基线: 1.0 / 0.5
# 多跳增量 gain     : 0.75       （q1 +0.5，q2 +1.0）
# 单跳增量 gain     : 0.0
# 结论              : RAG_GRAPH_ENABLED 保持 true
```

**结论：图谱腿保留，默认开启。** 多跳问题的增量是真实的（2/2 问题靠图谱召回原本够不到的
证据 chunk），路径机制本身可被精确还原（不是随机遍历）。

## 口径边界（必须读）

- 离线基线是**词面重叠的近似下界**；线上真实基线是向量 + BM25（语义上"依赖什么存储"与
  "快照持久化"可能被向量召回拉近）。**在部署环境用真实向量库重测后，若多跳增量消失，
  就按结论规则关闭开关**——这正是这套评测的用途：它给出的是可复现的判定程序，不是一次性结论。
- 评测语料刻意用规则抽取友好的句子（技术词实体 + 动词模板）。LLM 抽取模式下的路径质量
  应另行抽样验证（`EntityExtractor.extract()` 路径），本评测只钉住规则兜底层的正确性——
  那是线上零成本运行的那条路径。

## 文件清单

| 文件 | 说明 |
|---|---|
| `api/config.py` | `graph_enabled: bool = True`（env `RAG_GRAPH_ENABLED`）+ 结论规则注释 |
| `api/core/retrieval.py` | hybrid 图谱腿 / `mode=graph` 均受总开关门控 |
| `scripts/graph_path_set.json` | 评测集：6 条语料 + 4 类问题（多跳/单跳/负控制） |
| `scripts/eval_graph_path.py` | 评测器：建图（零 LLM）→ 路径正确率 + 开/关召回差 → verdict |
| `tests/test_graph_path_eval_pytest.py` | 5 个用例：多跳正增量 / 单跳零增量 / 夹具性质 / 负控制 / 开关 |
