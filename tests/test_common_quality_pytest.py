"""engines/common/quality.py 共享原语单测 (OPT-C4)。

锁定三件事:
- is_refusal 识别标准拒答话术 (evaluate 与运行时共享单一 regex)
- word_overlap_faithfulness 保留单字否定词 ("不支持" ≠ "支持")
- cosine 边界 (空/不等长/零向量返回 0)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engines.common.quality import cosine, is_refusal, word_overlap_faithfulness


def test_is_refusal_matches_standard_phrases():
    for text in (
        "知识库中未找到与您问题相关的内容",
        "无法回答该问题",
        "检索到的最佳片段相关度过低",
        "请换一种表述方式后重试",
        "文档中未提及该事项",
    ):
        assert is_refusal(text), text
    assert not is_refusal("退款需要 7 个工作日到账")
    assert not is_refusal("")
    assert not is_refusal(None)  # type: ignore[arg-type]


def test_word_overlap_preserves_negations():
    # "不支持"与上下文"不支持"高重合; 与"支持"也共享"支持"词 ——
    # 关键不变量: 否定词参与分母, 语义差异体现为分数差
    ctx_no = ["七天无理由不支持退换"]
    ctx_yes = ["七天无理由支持退换"]
    ans = "不支持退换"
    assert word_overlap_faithfulness(ans, ctx_no) > word_overlap_faithfulness(ans, ctx_yes)


def test_word_overlap_empty_cases():
    assert word_overlap_faithfulness("", ["ctx"]) == 0.0
    assert word_overlap_faithfulness("答案", []) == 0.0
    assert word_overlap_faithfulness("答案", ["", "  "]) == 0.0


def test_cosine_edges():
    assert cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([], [1.0]) == 0.0
    assert cosine([1.0], [1.0, 2.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0
