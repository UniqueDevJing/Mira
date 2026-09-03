"""通用父子文档机制端到端验证 (P1#6)

StructureChunker 对长段落产出 child(检索单元) + parent(整段大上下文);
入库后 HybridRetriever 检索命中 child, _expand_parents 按 parent_id 回拉 parent 全文喂 LLM。
验证该机制对通用文本(非代码)生效, 且 parent_content 为大段上下文。
"""

import tempfile

import numpy as np

from engines.chunking.structure_chunker import StructureChunker
from engines.retrieval.hybrid_retriever import HybridRetriever
from engines.retrieval.vector_store import VectorStore

DIM = 64


def _doc(blocks, doc_id="d1"):
    return type(
        "Doc",
        (),
        {"doc_id": doc_id, "pages": [{"page_num": 1, "blocks": blocks}], "source": {"path": "doc.md"}, "update_time": 0},
    )()


def _title(t, level=None):
    md = {"heading_level": level} if level is not None else {}
    return {"type": "title", "content": t, "page_num": 1, "metadata": md}


def _para(t):
    return {"type": "paragraph", "content": t, "page_num": 1, "metadata": {}}


class _FakeEmbedder:
    def __init__(self, vec):
        self.vec = vec

    def embed_query(self, q):
        return self.vec


def test_generic_parent_child_expansion():
    # 长段落 (>800 字) 触发 parent + child
    long = _para("通用文档正文内容。" * 200)
    doc = _doc([_title("第一章", 1), long])
    chunks = StructureChunker().chunk(doc)
    children = [c for c in chunks if not c.metadata.get("is_parent")]
    parents = [c for c in chunks if c.metadata.get("is_parent")]
    assert parents and children

    # 赋确定性稠密 embedding (child[0] 作为查询目标)
    rng = np.random.default_rng(0)
    for c in chunks:
        v = rng.standard_normal(DIM)
        c.embedding = (v / np.linalg.norm(v)).tolist()
    target = children[0]
    target_vec = target.embedding

    with tempfile.TemporaryDirectory() as d:
        store = VectorStore(uri=d, dim=DIM)
        store.insert(chunks)
        retriever = HybridRetriever(vector_store=store, embedder=_FakeEmbedder(target_vec))
        res = retriever.retrieve("查询", top_k=10)
        docs = res["documents"]
        # 命中 child 后至少一条带 parent_content (父块大上下文)
        assert any(d.get("parent_content") for d in docs), "未展开父块上下文"
        # parent_content 应为父块整段内容 (>800 字, 即原始长段落)
        assert any(len(d.get("parent_content", "")) > 800 for d in docs)
        # parent_id 链路正确: 展开出的 parent_content 等于对应父块内容
        pid = target.metadata["parent_id"]
        parent_content = next(d["parent_content"] for d in docs if d.get("parent_content"))
        assert parent_content == next(c.content for c in parents if c.chunk_id == pid)
