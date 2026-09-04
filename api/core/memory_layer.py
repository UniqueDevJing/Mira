"""长期记忆层 — 用户历史问答向量存储 (框架"记忆层: 长期"部分)。

- 存储: lancedb_data/rag_memory 表 (向量 512 维 bge-small-zh, user_id 过滤)。
- remember(): 每次问答后异步写入 (fire-and-forget, 失败仅记日志不影响主链路)。
- recall(): 提问时检索该用户最相关的历史问答, 注入为个性化上下文。
- 空表/组件缺失/异常一律静默降级返回空, 绝不阻断问答主流程。
"""

from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger(__name__)

_TABLE = "rag_memory"
_URI = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "lancedb_data")

_conn = None


def _get_table():
    """懒加载并确保表存在; 失败返回 None (降级无记忆)。"""
    global _conn
    if _conn is not None:
        try:
            return _conn.open_table(_TABLE)
        except Exception as e:  # noqa: BLE001 — 表不存在则走下方创建
            logger.debug("rag_memory 表未就绪, 尝试创建: %s", str(e)[:120])
    try:
        import lancedb

        from api.state import get_embedder

        _conn = lancedb.connect(_URI)
        if _TABLE not in _conn.table_names():
            dim = len(get_embedder().embed_query("dim"))
            _conn.create_table(
                _TABLE,
                data=[{"id": "init", "user_id": "", "question": "", "answer": "",
                       "ts": 0.0, "vector": [0.0] * dim}],
                mode="overwrite",
            )
            # 移除占位行: 重建为空表
            t = _conn.open_table(_TABLE)
            t.delete("id = 'init'")
        return _conn.open_table(_TABLE)
    except Exception as e:  # noqa: BLE001 — 记忆层整体降级
        logger.warning("长期记忆层不可用(降级为无记忆): %s", str(e)[:160])
        _conn = None
        return None


def remember(user_id: str, question: str, answer: str) -> bool:
    """写入一条长期记忆。失败返回 False, 不抛异常。"""
    if not user_id or not question:
        return False
    try:
        from api.state import get_embedder

        table = _get_table()
        if table is None:
            return False
        vec = get_embedder().embed_query(question[:500])
        table.add([{
            "id": f"m{int(time.time()*1000)}{user_id[:8]}",
            "user_id": user_id[:64],
            "question": question[:1000],
            "answer": answer[:2000],
            "ts": time.time(),
            "vector": vec,
        }])
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("长期记忆写入失败(忽略): %s", str(e)[:160])
        return False


def recall(user_id: str, query: str, top_k: int = 3) -> list[dict]:
    """检索该用户最相关的历史问答。失败/无记录返回 []。"""
    if not user_id or not query:
        return []
    try:
        from api.state import get_embedder

        table = _get_table()
        if table is None:
            return []
        vec = get_embedder().embed_query(query[:500])
        rows = (
            table.search(vec)
            .metric("cosine")
            .where(f"user_id = '{user_id[:64]}'")
            .limit(top_k)
            .to_list()
        )
        out = []
        for r in rows:
            q, a = (r.get("question") or "").strip(), (r.get("answer") or "").strip()
            if not q:
                continue
            out.append({
                "question": q,
                "answer": a[:300],
                "score": round(max(0.0, 1.0 - float(r.get("_distance", 1.0))), 4),
                "ts": r.get("ts", 0.0),
            })
        return out
    except Exception as e:  # noqa: BLE001
        logger.warning("长期记忆检索失败(忽略): %s", str(e)[:160])
        return []
