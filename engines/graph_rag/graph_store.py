"""知识图谱存储 — 内存版（Phase 2 开发用，生产切 Neo4j）"""
from typing import List, Dict
from collections import defaultdict


class GraphStore:
    def __init__(self):
        self.nodes: Dict[str, dict] = {}       # {name: {type, aliases, chunks, properties}}
        self.edges: List[dict] = []             # [{subject, predicate, object, chunk_id}]
        self.adj_out = defaultdict(list)        # {subject: [(object, predicate)]}
        self.adj_in = defaultdict(list)          # {object: [(subject, predicate)]}

    def upsert_entity(self, name: str, etype: str, chunk_id: str = "",
                       aliases: List[str] = None):
        if name in self.nodes:
            self.nodes[name]["chunks"].add(chunk_id)
            if aliases:
                self.nodes[name]["aliases"].extend(aliases)
        else:
            self.nodes[name] = {
                "type": etype, "aliases": aliases or [],
                "chunks": {chunk_id}, "properties": {}
            }

    def add_relation(self, subject: str, predicate: str, object: str,
                     chunk_id: str = ""):
        edge = {"subject": subject, "predicate": predicate,
                "object": object, "chunk_id": chunk_id}
        self.edges.append(edge)
        self.adj_out[subject].append((object, predicate))
        self.adj_in[object].append((subject, predicate))

    def get_entity(self, name: str) -> dict:
        return self.nodes.get(name)

    def get_relations(self, subject: str = None, predicate: str = None,
                      object: str = None) -> List[dict]:
        results = []
        for e in self.edges:
            if subject and e["subject"] != subject: continue
            if predicate and e["predicate"] != predicate: continue
            if object and e["object"] != object: continue
            results.append(e)
        return results

    def multi_hop(self, start: str, relations: List[str], max_depth: int = 3) -> List[dict]:
        """多跳遍历：从 start 出发，沿 relations 路径遍历"""
        results = []
        current = [start]
        for rel in relations[:max_depth]:
            next_nodes = []
            for node in current:
                for obj, pred in self.adj_out.get(node, []):
                    if pred == rel:
                        results.append({"from": node, "relation": rel, "to": obj})
                        next_nodes.append(obj)
            current = next_nodes
            if not current:
                break
        return results

    def get_context_for_entity(self, name: str) -> str:
        """获取实体的图谱上下文"""
        node = self.nodes.get(name)
        if not node:
            return ""

        parts = [f"实体: {name} (类型: {node['type']})"]

        # 出边
        outs = self.adj_out.get(name, [])[:10]
        if outs:
            parts.append("关联到: " + ", ".join(
                f"{obj}({pred})" for obj, pred in outs
            ))

        # 入边
        ins = self.adj_in.get(name, [])[:10]
        if ins:
            parts.append("被关联: " + ", ".join(
                f"{subj}({pred})" for subj, pred in ins
            ))

        return " | ".join(parts)

    def stats(self) -> dict:
        return {
            "nodes": len(self.nodes),
            "edges": len(self.edges),
            "node_types": {n["type"] for n in self.nodes.values()},
            "relation_types": {e["predicate"] for e in self.edges},
        }
