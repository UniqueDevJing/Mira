# Mira 技术选型审计报告（2026-09-06）

> 结论基于 390 问全量实测对比，非理论推断。全部实验可复现。

## 结论：Embedding 模型是当前召回/精度的最大杠杆，bge-base-zh-v1.5 是明确更优解

### Embedding 对比（完整生产链路：融合+自适应+rerank）

| 模型 | 维度 | R@1 | R@3 | R@10 | MRR | 嵌入耗时(1531块) |
|---|---|---|---|---|---|---|
| **bge-small-zh-v1.5（当前）** | 512 | 0.390 | 0.746 | 0.979 | 0.772 | ~30s |
| **bge-base-zh-v1.5（推荐切换）** | 768 | **0.405** | **0.756** | **0.983** | **0.789** | ~3min |
| bge-m3 | 1024 | 0.356* | 0.768* | 0.984* | 0.762* | ~9min |

*bge-m3 为 rerank-off 基线（与上两行口径差异见 docs/EVALUATION.md §1）；CPU 重嵌入成本 3 倍且指标不优于 bge-base，中文短 QA 场景不推荐。

**⚠️ 2026-09-06 晚间实测撤回该建议**：上表对比在**无前缀协议**下得出（离线评测 query 不带 "query: " 前缀）。生产协议是 `embed_query` 带 `"query: "` 前缀 + chunk 无前缀（bge-zh 官方检索约定）。在生产协议下 KB 级实测（80 题真实库）：bge-base MRR **0.815→0.62**、R@10 0.975→0.775，**大幅变差**——bge-base 对前缀协议的匹配不对称远比 bge-small 敏感。已全量回滚（.env/config/向量库恢复 bge-small 512 维原版），KB 评测恢复 MRR 0.8153。

**教训**：embedding 模型对比必须在**生产完全一致的编码协议**（前缀/归一化/截断）下做，离线裸文本协议的结论不可迁移。

### 已完成并证伪的方向（勿重复尝试）

| 方向 | 结论 |
|---|---|
| rerank 融合权重 alpha | 0.9 已是 R@3 最优（全网格） |
| rerank 候选深度 | 10 已是平衡点，越深越差 |
| CE 输入长度 | 384 已应用（R@3 +1.45pp） |
| 查询改写 | 已启用，A/B 实测仅 +0.77%（饱和） |

### 切换 bge-base 的落地步骤

1. `api/config.py` → `embedding_model: str = "BAAI/bge-base-zh-v1.5"`
2. **10 个知识库全量重建向量**（维度 512→768，LanceDB 表结构不变）：对每库 `documents.rebuild_index(kb)` 或重跑入库脚本
3. 重启服务 + 跑 `make eval-routing` / KB 评测回归
4. 模型权重已就位：`data/hf_cache/models--BAAI--bge-base-zh-v1.5`（`HF_HUB_CACHE` 指向此目录）

### 需要硬件才能解锁的方向（当前不推荐）

| 方向 | 预期收益 | 门槛 |
|---|---|---|
| bge-reranker-v2-m3（5.5 亿） | R@1/precision 显著提升 | GPU（CPU 延迟 ~4×） |
| bge-reranker-large | R@3 突破 0.78 | GPU |
| bge-m3 稠密+稀疏+多向量混合 | 长文档场景 | 部署复杂度高 |

### 其余选型判定（合理，无需更换）

- **LanceDB**：轻量嵌入式，10 库规模合适（Lance 列存 + 标量过滤）
- **BM25+RRF 融合**：已是混合检索标准做法，自适应 alpha 有实测收益
- **规则+LLM 路由**：top2 73.8% 配合扇出，架构合理
- **qwen-plus/qwen-vl-plus**：生成端 faithfulness 0.914 已证明够用
