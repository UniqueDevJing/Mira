"""质检报告修复回归测试 — 标题链 / 查询去重 / 批量嵌入 / rerank 不改输入"""


# ── semantic_chunker 标题链 (P0-2.2) ──
def _make_uir():
    from engines.parsing.pdf_parser import UIRDocument

    return UIRDocument(
        doc_id="t1",
        source={"type": "pdf", "path": "x"},
        tables=[],
        pages=[
            {
                "page_num": 1,
                "blocks": [
                    {"type": "title", "content": "一级标题", "metadata": {"font_size": 22}, "page_num": 1},
                    {"type": "title", "content": "二级标题", "metadata": {"font_size": 18}, "page_num": 1},
                    {"type": "paragraph", "content": "段落A", "page_num": 1},
                    {"type": "title", "content": "另一节", "metadata": {"font_size": 22}, "page_num": 1},
                    {"type": "paragraph", "content": "段落B", "page_num": 1},
                ],
            }
        ],
    )


def _chain_of(chunks, keyword: str):
    return next(c for c in chunks if keyword in c.content).context["title_chain"]


def test_title_chain_hierarchy():
    """结构切分: 段落应携带其所在标题层级链 (替代原语义 title tree)。"""
    from engines.chunking.structure_chunker import StructureChunker

    chunks = StructureChunker().chunk(_make_uir())
    # 段落A 在"一级标题>二级标题"下
    assert _chain_of(chunks, "段落A") == ["一级标题", "二级标题"]
    # 段落B 在顶级"另一节"下, 父链不串到上一节
    assert _chain_of(chunks, "段落B") == ["另一节"]


def test_title_chain_excludes_sibling():
    """兄弟标题段落链应独立, 不继承上一节标题。"""
    from engines.chunking.structure_chunker import StructureChunker

    chunks = StructureChunker().chunk(_make_uir())
    assert "一级标题" not in _chain_of(chunks, "段落B")


# ── query_rewriter 模板去重防膨胀 (P0-2.6) ──
def _eval_result():
    from engines.retrieval.evaluator import EvalResult

    return EvalResult(
        relevance_score=0.4, coverage_score=0.4, confidence_score=0.0, need_rewrite=True, reason="相关性不足"
    )


def test_template_rewrite_dedup():
    """无 LLM 时模板改写应去重, 不产生重复候选。"""
    from engines.retrieval.query_rewriter import QueryRewriter

    out = QueryRewriter().rewrite("系统架构是什么", _eval_result())
    assert out[0] == "系统架构是什么 相关文档"
    assert len(set(out)) == len(out)


def test_template_rewrite_no_expand():
    """已含模板后缀的查询应原样返回, 不逐轮膨胀 (原 bug: 无限拼接)。"""
    from engines.retrieval.query_rewriter import QueryRewriter

    polluted = "系统架构是什么 相关文档 详细信息"
    out = QueryRewriter().rewrite(polluted, _eval_result())
    assert out == [polluted]


# ── evaluator 批量嵌入 (P1-3.4) ──
class _FakeEmbedder:
    def __init__(self):
        self.batch_calls = 0
        self.query_calls = 0

    def embed_query(self, text):
        self.query_calls += 1
        return [1.0, 0.0]

    def embed_batch(self, texts):
        self.batch_calls += 1
        return [[1.0, 0.0] for _ in texts]


def test_calc_relevance_uses_embed_batch():
    """评估器应一次 embed_batch 而非逐条 embed_query (原 10 次调用)。"""
    from engines.retrieval.evaluator import RetrievalEvaluator

    fake = _FakeEmbedder()
    ev = RetrievalEvaluator(embedder=fake)
    docs = [{"content": "测试内容" * 5} for _ in range(5)]
    score = ev._calc_relevance("查询", docs)
    assert score > 0.0
    assert fake.query_calls == 1
    assert fake.batch_calls == 1


# ── 忠实度护栏 (幻觉降级) ──
def test_faithfulness_from_context():
    """答案复用上下文词 → 高忠实; 无关答案 → 低忠实 (护栏信号)。"""
    from api.core.qa_metrics import _faithfulness

    ctx = ["退款流程：买家申请退款，商家48小时内处理，逾期系统自动退款。"]
    assert _faithfulness("买家申请退款，商家48小时内处理", ctx) > 0.5
    assert _faithfulness("今天天气很好，适合外出散步", ctx) < 0.2


def test_faithfulness_below_threshold_triggers_guard():
    """忠实度低于 RAG_FIDELITY_THRESHOLD (0.4) 时护栏应触发。"""
    from api.config import settings
    from api.core.qa_metrics import _faithfulness

    ctx = ["退款流程：买家申请退款，商家48小时内处理。"]
    assert _faithfulness("小行星撞击地球导致恐龙灭绝", ctx) < settings.fidelity_threshold


# ── document_store 启动重置 (服务重启中断恢复) ──
def test_reset_stale_processing(tmp_path):
    """启动时应把卡死的 processing 重置为 failed, ready 不受影响。"""
    from api.core.document_store import DocumentStore

    store = DocumentStore(db_path=str(tmp_path / "doc.db"))
    store.save("a1", "x.pdf", status="processing")
    store.save("a2", "y.pdf", status="ready")
    n = store.reset_stale_processing()
    assert n == 1
    assert store.get("a1")["status"] == "failed"
    assert store.get("a2")["status"] == "ready"


# ── reranker 不改输入 (P1-3.9) ──
def test_rerank_does_not_mutate_input(embedder):
    """rerank 应返回新对象, 不修改传入文档的 score (原: 直接 dict 写入)。"""
    from engines.retrieval.reranker import Reranker

    docs = [
        {"id": "a", "content": "FastAPI 是高性能 Web 框架", "score": 0.5},
        {"id": "b", "content": "Django 是 web 框架", "score": 0.6},
        {"id": "c", "content": "Flask 轻量级框架", "score": 0.4},
    ]
    original_scores = [d["score"] for d in docs]
    r = Reranker(embedder=embedder)
    out = r.rerank("FastAPI 框架", docs, top_k=3)
    assert [d["score"] for d in docs] == original_scores  # 输入未变
    assert out is not docs
