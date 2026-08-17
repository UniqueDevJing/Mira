"""Query 规则预处理单元测试。"""

from engines.retrieval.query_preprocessor import expand_query, normalize_query, preprocess_query


def test_normalize_strips_polite():
    assert normalize_query("你好，请问退款流程是什么") == "退款流程是什么"
    assert normalize_query("  退货 怎么 处理 ") == "退货 怎么 处理"


def test_normalize_keeps_plain():
    assert normalize_query("退款什么时候到账") == "退款什么时候到账"


def test_expand_adds_synonyms():
    out = expand_query("怎么退款")
    assert "退款" in out and "退货" in out  # 追加同义词
    assert "快递" in expand_query("物流怎么查")


def test_expand_does_not_duplicate():
    out = expand_query("退货退款怎么处理")
    # 已含同义词时不重复追加
    assert out.count("退货") == 1
    assert "退货退款" not in out.split()  # 原文已含, 不重复追加


def test_expand_adds_api_synonym():
    out = expand_query("接口报错")
    assert "API" in out


def test_preprocess_returns_pair():
    vq, bq = preprocess_query("你好，部署系统")
    assert vq == "部署系统"  # 向量: 原义
    assert bq != vq and "安装" in bq  # BM25: 扩展
