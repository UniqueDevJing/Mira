"""QA 质量指标单测 — 验证忠实度与综合指标计算 (纯函数, 无 async/state)。"""

from api.core.qa_metrics import EMBED_FAITHFUL_MIN, _calc_qa_metrics, _cosine, _faithfulness


def test_faithfulness_high_when_answer_reuses_context():
    ctx = ["退款流程：买家申请退款，商家48小时内处理，逾期系统自动退款。"]
    assert _faithfulness("买家申请退款，商家48小时内处理", ctx) > 0.5


def test_faithfulness_low_when_unrelated():
    ctx = ["退款流程：买家申请退款，商家48小时内处理。"]
    assert _faithfulness("今天天气很好，适合外出散步", ctx) < 0.2


def test_faithfulness_empty_answer_zero():
    assert _faithfulness("", ["任意上下文"]) == 0.0


def test_faithfulness_empty_context_zero():
    assert _faithfulness("有内容但无上下文", []) == 0.0


def test_faithfulness_no_embed_fn_equals_word_only():
    ctx = ["退款流程：买家申请退款。"]
    a = _faithfulness("买家申请退款", ctx)
    b = _faithfulness("买家申请退款", ctx, embed_fn=None)
    assert a == b


def test_faithfulness_embedding_lifts_paraphrase():
    # 词重合为 0 (无共享 2 字 token), 但语义一致 → 旧实现会误拒; embedding 抬升避免误拒
    ctx = ["香蕉属于热带作物，生长在温暖地区。"]
    embed_fn = lambda t: [1.0, 0.0]  # 任意固定向量使 answer/context 余弦=1
    faith = _faithfulness("苹果是一种水果，长在温暖的地方", ctx, embed_fn=embed_fn)
    assert faith >= EMBED_FAITHFUL_MIN


def test_faithfulness_embedding_weak_does_not_lift():
    # 词重合 0 且 embedding 几乎正交 → 护栏不松, 仍判低忠实
    ctx = ["香蕉属于热带作物，生长在温暖地区。"]
    embed_fn = lambda t: [1.0, 0.0] if "苹果" in t else [0.0, 1.0]
    faith = _faithfulness("苹果是一种水果", ctx, embed_fn=embed_fn)
    assert faith < 0.4


def test_faithfulness_embedding_failure_falls_back_to_word():
    # 嵌入不可用 → 退回词重合信号, 不崩溃为 0
    ctx = ["水果富含维生素，对身体有益。"]
    embed_fn = lambda t: (_ for _ in ()).throw(RuntimeError("model down"))
    faith = _faithfulness("水果对身体有益", ctx, embed_fn=embed_fn)
    assert faith > 0.3


def test_cosine_identical_and_orthogonal():
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_calc_qa_metrics_no_answer():
    m = _calc_qa_metrics("", ["ctx"], top1=0.9)
    assert m["faithfulness"] == 0.0
    assert m["confidence"] == 0.0
    assert m["retrieval_rounds"] == 1


def test_calc_qa_metrics_confidence_blend():
    # 高忠实 + 高 top1 → 高置信; 0.6*faith + 0.4*rel
    ctx = ["部署架构说明：使用 FastAPI 与 Docker 构建服务。"]
    m = _calc_qa_metrics("使用 FastAPI 与 Docker 构建服务", ctx, top1=1.0)
    assert m["faithfulness"] > 0.5
    assert m["retrieval_relevance"] == 1.0
    assert 0.5 < m["confidence"] <= 1.0


def test_calc_qa_metrics_rounds_passthrough():
    m = _calc_qa_metrics("答案", ["ctx"], top1=0.5, retrieval_rounds=3)
    assert m["retrieval_rounds"] == 3
    assert m["top1_score"] == 0.5
