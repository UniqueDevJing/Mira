"""知识图谱存储 — 内存版（Phase 2 开发用，生产切 Neo4j）。

持久化: persist_path 指定时, build_from_chunks 完成后落盘 pickle, 进程重启不丢图
(与 BM25 的 data/bm25_<kb>.pkl 对称)。默认 persist_path=None 保持纯内存 (行为不变)。
"""

import logging
import os
import pickle
import threading
from collections import defaultdict

from engines.interfaces import GraphStoreInterface

logger = logging.getLogger(__name__)


class GraphStore(GraphStoreInterface):
    def __init__(
        self,
        persist_path: str | None = None,
        redis_url: str | None = None,
        redis_key: str | None = None,
    ):
        self.nodes: dict[str, dict] = {}  # {name: {type, aliases, chunks, properties}}
        self.edges: list[dict] = []  # [{subject, predicate, object, chunk_id}]
        self.adj_out = defaultdict(list)  # {subject: [(object, predicate)]}
        self.adj_in = defaultdict(list)  # {object: [(subject, predicate)]}
        self._edge_keys: set[tuple] = set()  # 三元组去重: 同一关系被多 chunk 提及不重复累积
        # 小写归一化索引: lower(name/alias) -> canonical, 供 find_node O(1) 反查 (C4)
        self._lower_index: dict[str, str] = {}
        # RLock: 上传线程 build_from_chunks 与检索线程 retrieve 并发读写, 无锁会抛
        # "dictionary changed size during iteration" (nodes/adj_* 迭代时被改)
        self._lock = threading.RLock()
        self._persist_path = persist_path
        # Redis 共享后端 (可选): 整图 pickle 写入 Redis, 多 worker 共享同一份, 省去重复 LLM 抽取。
        # 懒加载客户端; 未安装 redis 或连接失败 → _redis=None, 回退内存/文件, 不阻断启动。
        self._redis = None
        self._redis_key = redis_key or (os.path.basename(persist_path) if persist_path else "graph")
        if redis_url:
            try:
                import redis as _redis_mod

                self._redis = _redis_mod.from_url(redis_url, decode_responses=False)
            except Exception as e:  # noqa: BLE001 — Redis 不可用回退内存/文件
                logger.warning("图谱 Redis 后端初始化失败, 回退内存/文件: %s", str(e)[:120])
                self._redis = None
        # 恢复优先级: Redis (共享, 多 worker 一致) → pickle 文件 (重启) → 空图
        # redis_loaded: True=已从 Redis 恢复; False=Redis 缺失/损坏需回退文件; None=无 Redis 走文件
        redis_loaded = self._load_redis() if self._redis is not None else None
        if redis_loaded is not True and persist_path and os.path.exists(persist_path):
            self._load(persist_path)

    def _save(self) -> None:
        """持久化图谱 (build_from_chunks 完成后调用; 不每步落盘避免大图 I/O 抖动)。

        双写: (1) 本地 pickle 文件 (重启恢复, 与 BM25 对称); (2) Redis (多 worker 共享同一份,
        首个 ingest 的 worker 写入, 其余 worker 直接加载, 省去重复 LLM 抽取 + 保证图谱一致)。
        """
        blob = pickle.dumps(
            {
                "nodes": self.nodes,
                "edges": self.edges,
                "adj_out": dict(self.adj_out),
                "adj_in": dict(self.adj_in),
                "edge_keys": self._edge_keys,
                "lower_index": self._lower_index,
            }
        )
        if self._persist_path:
            try:
                os.makedirs(os.path.dirname(self._persist_path) or ".", exist_ok=True)
                with open(self._persist_path, "wb") as f:
                    f.write(blob)
            except OSError as e:
                logger.warning("图谱持久化写入失败: %s", str(e)[:120])
        if self._redis is not None:
            try:
                self._redis.set(self._redis_key, blob)
            except Exception as e:  # noqa: BLE001 — Redis 写入失败不阻断 (内存态仍可用)
                logger.warning("图谱 Redis 写入失败: %s", str(e)[:120])

    def _load(self, path: str) -> None:
        """从 pickle 恢复图谱; 损坏则回退空图 (不阻断启动)。"""
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self.nodes = data["nodes"]
            self.edges = data["edges"]
            self.adj_out = defaultdict(list, data.get("adj_out", {}))
            self.adj_in = defaultdict(list, data.get("adj_in", {}))
            self._edge_keys = set(data.get("edge_keys", set()))
            self._lower_index = data.get("lower_index", {})
            logger.info("图谱从 %s 恢复: %d 节点 %d 边", path, len(self.nodes), len(self.edges))
        except Exception as e:  # noqa: BLE001 — 持久化损坏回退空图
            logger.warning("图谱持久化加载失败, 回退空图: %s", str(e)[:120])

    def _load_redis(self) -> bool:
        """从 Redis 恢复整图 (二进制 pickle)。成功返回 True; 缺失/损坏返回 False (调用方回退文件/空图)。"""
        try:
            blob = self._redis.get(self._redis_key)
            if not blob:
                return False
            data = pickle.loads(blob)
            self.nodes = data["nodes"]
            self.edges = data["edges"]
            self.adj_out = defaultdict(list, data.get("adj_out", {}))
            self.adj_in = defaultdict(list, data.get("adj_in", {}))
            self._edge_keys = set(data.get("edge_keys", set()))
            self._lower_index = data.get("lower_index", {})
            logger.info("图谱从 Redis(%s) 恢复: %d 节点 %d 边", self._redis_key, len(self.nodes), len(self.edges))
            return True
        except Exception as e:  # noqa: BLE001 — Redis 损坏/不可达回退
            logger.warning("图谱 Redis 加载失败, 回退: %s", str(e)[:120])
            return False

    def save(self) -> None:
        """显式落盘 (build_from_chunks 结束时调用)。"""
        self._save()

    def upsert_entity(self, name: str, etype: str, chunk_id: str = "", aliases: list[str] | None = None):
        with self._lock:
            if name in self.nodes:
                if chunk_id and chunk_id not in self.nodes[name]["chunks"]:
                    self.nodes[name]["chunks"].append(chunk_id)
                if aliases:
                    self.nodes[name]["aliases"].extend(aliases)
                    for a in aliases:  # 新别名同步进索引
                        self._lower_index[a.lower()] = name
            else:
                self.nodes[name] = {
                    "type": etype,
                    "aliases": aliases or [],
                    "chunks": [chunk_id] if chunk_id else [],
                    "properties": {},
                }
                # 名 + 别名统一进小写索引
                self._lower_index[name.lower()] = name
                for a in aliases or []:
                    self._lower_index[a.lower()] = name

    def _canonical(self, name: str) -> str:
        """归一化实体名到已存在节点 key (大小写/别名反查)。节点不存在时保留原样。"""
        canonical, _ = self.find_node(name)
        return canonical if canonical else name

    def add_relation(self, subject: str, predicate: str, object: str, chunk_id: str = ""):
        with self._lock:
            # 归一化: LLM 关系抽取 subject/object 大小写漂移 ("fastapi" vs 节点 "FastAPI") 时,
            # 原样入库会让多跳遍历沿错误 key 查找, 整条边不可见
            subj = self._canonical(subject)
            obj = self._canonical(object)
            key = (subj, predicate, obj)
            if key in self._edge_keys:
                return  # 已记录的唯一边, 跳过 (原无条件 append, edges/adj 随 chunk 反复膨胀)
            self._edge_keys.add(key)
            edge = {"subject": subj, "predicate": predicate, "object": obj, "chunk_id": chunk_id}
            self.edges.append(edge)
            self.adj_out[subj].append((obj, predicate))
            self.adj_in[obj].append((subj, predicate))

    def get_entity(self, name: str) -> dict:
        return self.nodes.get(name)

    def find_node(self, name: str) -> tuple[str | None, dict | None]:
        """精确 → 大小写不敏感 → 别名反查。线程安全: nodes 迭代持锁, 防并发 upsert 时 RuntimeError。"""
        with self._lock:
            node = self.nodes.get(name)
            if node is not None:
                return name, node
            lower = name.lower()
            # O(1) 索引命中 (C4): 大多数场景无需线性扫描
            canon = self._lower_index.get(lower)
            if canon is not None and canon in self.nodes:
                return canon, self.nodes[canon]
            # 兜底线性扫描 (索引未覆盖的极端竞态)
            for n, nd in self.nodes.items():
                if n.lower() == lower:
                    return n, nd
                if any(a.lower() == lower for a in nd.get("aliases", [])):
                    return n, nd
            return None, None

    def get_relations(
        self, subject: str | None = None, predicate: str | None = None, object: str | None = None
    ) -> list[dict]:
        with self._lock:
            results = []
            for e in self.edges:
                if subject and e["subject"] != subject:
                    continue
                if predicate and e["predicate"] != predicate:
                    continue
                if object and e["object"] != object:
                    continue
                results.append(e)
            return results

    def multi_hop(self, start: str, relations: list[str], max_depth: int = 3, bidirectional: bool = True) -> list[dict]:
        """多跳遍历：从 start 出发，沿指定关系类型遍历 max_depth 跳。

        每一跳遍历所有指定的 relations 类型，而非只检查前 N 个。
        bidirectional=True（默认）时同时沿出边与入边遍历，解决反向关系漏召回 (C2)；
        入边返回 {"from": 邻节点, "relation": pred, "to": node}，语义方向保留。
        """
        with self._lock:
            results = []
            visited = {start}
            current = [start]

            for depth in range(max_depth):
                next_nodes = []
                for node in current:
                    for obj, pred in self.adj_out.get(node, []):
                        if pred in relations and obj not in visited:
                            results.append({"from": node, "relation": pred, "to": obj})
                            next_nodes.append(obj)
                            visited.add(obj)
                    if bidirectional:
                        for subj, pred in self.adj_in.get(node, []):
                            if pred in relations and subj not in visited:
                                results.append({"from": subj, "relation": pred, "to": node})
                                next_nodes.append(subj)
                                visited.add(subj)
                current = next_nodes
                if not current:
                    break
            return results

    def get_context_for_entity(self, name: str) -> str:
        """获取实体的图谱上下文"""
        with self._lock:
            node = self.nodes.get(name)
            if not node:
                return ""

            parts = [f"实体: {name} (类型: {node['type']})"]

            # 出边
            outs = self.adj_out.get(name, [])[:10]
            if outs:
                parts.append("关联到: " + ", ".join(f"{obj}({pred})" for obj, pred in outs))

            # 入边
            ins = self.adj_in.get(name, [])[:10]
            if ins:
                parts.append("被关联: " + ", ".join(f"{subj}({pred})" for subj, pred in ins))

            return " | ".join(parts)

    def stats(self) -> dict:
        with self._lock:
            return {
                "nodes": len(self.nodes),
                "edges": len(self.edges),
                "node_types": {n["type"] for n in self.nodes.values()},
                "relation_types": {e["predicate"] for e in self.edges},
            }
