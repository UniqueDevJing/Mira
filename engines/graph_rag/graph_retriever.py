"""GraphRAG 检索器 — 多跳推理 + 图谱增强检索"""

from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore


class GraphRAGRetriever:
    def __init__(
        self, entity_extractor: EntityExtractor, relation_extractor: RelationExtractor, graph_store: GraphStore
    ):
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.graph_store = graph_store

    def build_from_chunks(self, chunks: list):
        """从文档 chunks 构建知识图谱 — 批量抽取 (每批合并 LLM 调用, 失败逐 chunk 规则兜底)"""
        batch = []
        for chunk in chunks:
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            chunk_id = chunk.chunk_id if hasattr(chunk, "chunk_id") else ""
            batch.append((text, chunk_id))

        entity_map = self.entity_extractor.extract_batch(batch)
        relation_map = self.relation_extractor.extract_batch(batch, entity_map)

        for _, chunk_id in batch:
            for e in entity_map.get(chunk_id, []):
                self.graph_store.upsert_entity(e.name, e.type, chunk_id, e.aliases)
            for r in relation_map.get(chunk_id, []):
                self.graph_store.add_relation(r.subject, r.predicate, r.object, chunk_id)

        self.graph_store.save()  # 构建完成落盘 (persist_path=None 时为空操作)

        return {"entities": self.graph_store.stats()["nodes"], "relations": self.graph_store.stats()["edges"]}

    def retrieve(self, question: str, top_k: int = 5, rule_only: bool = True) -> dict:
        """图谱增强检索 — 提取问题实体 → 多跳遍历 → 返回相关上下文

        rule_only=True (默认): 查询侧只用规则抽取实体, 不触发 LLM
        (原实现走 extract() 会为每个 QA 请求付一次 LLM 调用, 与路由预算冲突)。
        """
        # 1. 从问题中提取实体 (规则, 零 LLM 成本)
        q_entities = (
            self.entity_extractor.extract_rules(question) if rule_only else self.entity_extractor.extract(question)
        )

        # 2. 对每个实体执行多跳遍历
        graph_context = []
        entities_found = []

        def _find_node(name):
            """精确 → 大小写不敏感 → 别名反查 (规则抽取保留原文大小写, "fastapi" vs "FastAPI" 需归一)"""
            return self.graph_store.find_node(name)

        hop_targets = []
        for entity in q_entities:
            canonical, node = _find_node(entity.name)
            if node is None:
                continue
            entities_found.append(canonical)
            # 多跳遍历
            hops = self.graph_store.multi_hop(
                canonical,
                relations=["uses", "depends_on", "contains", "supplies", "signs", "references", "employs", "owns"],
            )
            for hop in hops[:5]:
                # 取与起点相对的端点 (双向遍历时 hop["to"] 可能等于起点, C2)
                other = hop["from"] if hop["to"] == canonical else hop["to"]
                if other != canonical:
                    hop_targets.append(other)
                target = self.graph_store.get_entity(other)
                if target:
                    graph_context.append(f"{hop['from']} {hop['relation']} {hop['to']}")

            # 实体上下文
            ctx = self.graph_store.get_context_for_entity(canonical)
            if ctx:
                graph_context.append(ctx)

        # 3. 收集相关 chunk IDs (种子实体 + 多跳目标实体 — 原实现只收种子, 多跳结果不进检索)
        source_chunks = set()
        collect = list(entities_found)
        for to in dict.fromkeys(hop_targets):
            canonical, _ = _find_node(to)
            if canonical and canonical not in collect:
                collect.append(canonical)
        for name in collect:
            node = self.graph_store.get_entity(name)
            if node and "chunks" in node:
                source_chunks.update(node["chunks"])

        return {
            "entities": entities_found,
            "graph_context": graph_context[:10],
            "source_chunks": list(source_chunks)[:top_k],
        }
