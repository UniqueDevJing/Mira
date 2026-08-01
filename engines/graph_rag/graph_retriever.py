"""GraphRAG 检索器 — 多跳推理 + 图谱增强检索"""
from engines.graph_rag.entity_extractor import EntityExtractor, RelationExtractor
from engines.graph_rag.graph_store import GraphStore


class GraphRAGRetriever:
    def __init__(self, entity_extractor: EntityExtractor,
                 relation_extractor: RelationExtractor,
                 graph_store: GraphStore):
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.graph_store = graph_store

    def build_from_chunks(self, chunks: list):
        """从文档 chunks 构建知识图谱"""
        for chunk in chunks:
            text = chunk.content if hasattr(chunk, 'content') else str(chunk)
            chunk_id = chunk.chunk_id if hasattr(chunk, 'chunk_id') else ""

            # 实体抽取
            entities = self.entity_extractor.extract(text, chunk_id)

            # 关系抽取
            relations = self.relation_extractor.extract(text, entities, chunk_id)

            # 写入图谱
            for e in entities:
                self.graph_store.upsert_entity(
                    e.name, e.type, chunk_id, e.aliases
                )
            for r in relations:
                self.graph_store.add_relation(
                    r.subject, r.predicate, r.object, chunk_id
                )

        return {"entities": self.graph_store.stats()["nodes"],
                "relations": self.graph_store.stats()["edges"]}

    def retrieve(self, question: str, top_k: int = 5) -> dict:
        """图谱增强检索 — 提取问题实体 → 多跳遍历 → 返回相关上下文"""
        # 1. 从问题中提取实体
        q_entities = self.entity_extractor.extract(question)

        # 2. 对每个实体执行多跳遍历
        graph_context = []
        entities_found = []
        for entity in q_entities:
            node = self.graph_store.get_entity(entity.name)
            if node:
                entities_found.append(entity.name)
                # 多跳遍历
                hops = self.graph_store.multi_hop(
                    entity.name,
                    relations=["uses", "depends_on", "contains", "supplies",
                               "signs", "references", "employs", "owns"]
                )
                for hop in hops[:5]:
                    target = self.graph_store.get_entity(hop["to"])
                    if target:
                        graph_context.append(
                            f"{hop['from']} {hop['relation']} {hop['to']}"
                        )

                # 实体上下文
                ctx = self.graph_store.get_context_for_entity(entity.name)
                if ctx:
                    graph_context.append(ctx)

        # 3. 收集相关 chunk IDs
        source_chunks = set()
        for name in entities_found:
            node = self.graph_store.get_entity(name)
            if node and "chunks" in node:
                source_chunks.update(node["chunks"])

        return {
            "entities": entities_found,
            "graph_context": graph_context[:10],
            "source_chunks": list(source_chunks)[:top_k],
        }
