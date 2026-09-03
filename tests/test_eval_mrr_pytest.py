"""evaluate.py MRR 计算单元测试 (P2#10 补全)"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.evaluate import _first_relevant_rank

SRC = [
    {"chunk_id": "x2"},
    {"chunk_ids": ["x3", "x9"]},
    {"chunk_id": "x1"},
]


def test_rank_of_first_expected():
    assert _first_relevant_rank(SRC, {"x1"}) == 3
    assert _first_relevant_rank(SRC, {"x3"}) == 2
    assert _first_relevant_rank(SRC, {"x9"}) == 2
    assert _first_relevant_rank(SRC, {"x2"}) == 1


def test_rank_none_when_no_hit():
    assert _first_relevant_rank(SRC, {"zz"}) == 0


def test_mrr_value():
    # MRR = 1/rank; 无命中为 0
    assert round(1 / _first_relevant_rank(SRC, {"x1"}), 4) == 0.3333
    assert _first_relevant_rank(SRC, {"zz"}) == 0
