"""表格感知扩展验证 (路线图 1.3 补全): hr/marketing/training 现已开启 parse.table。

_TableAware 仅当文档确实含表格 (uir_doc.tables 非空) 时追加表格块;
无表文档零影响。本测试验证三类的表格块产出 + 无表文档不退化。
"""

from api.config import settings
from engines.chunking.strategies import get_chunker
from engines.doc_types import DOC_TYPES


def _doc(blocks, tables=None):
    return type(
        "Doc",
        (),
        {
            "doc_id": "d1",
            "pages": [{"page_num": 1, "blocks": blocks}],
            "source": {"path": "doc.pdf"},
            "tables": tables or [],
            "update_time": 0,
        },
    )()


def _para(text):
    return {"type": "paragraph", "content": text, "page_num": 1, "metadata": {}}


_TABLE_MD = "| 项目 | 数值 |\n|---|---|\n| 基本工资 | 10000 |\n| 绩效 | 3000 |"


def test_hr_marketing_training_enable_table_parse():
    for t in ("hr", "marketing", "training"):
        assert DOC_TYPES[t].parse.get("table") is True, f"{t} 应开启 table 感知"


def test_table_chunks_produced_for_enabled_types():
    doc = _doc([_para("正文段落内容。")], tables=[{"markdown": _TABLE_MD}])
    for t in ("hr", "marketing", "training"):
        chunks = get_chunker(DOC_TYPES[t], settings).chunk(doc)
        assert any(c.context.get("is_table") for c in chunks), f"{t} 未产出表格块"


def test_no_table_chunks_when_doc_has_no_tables():
    # 无表文档: 即便类型开启 table 感知, 也不应凭空产生表格块 (零退化)
    doc = _doc([_para("普通正文段落, 无任何表格。")])
    for t in ("hr", "marketing", "training"):
        chunks = get_chunker(DOC_TYPES[t], settings).chunk(doc)
        assert not any(c.context.get("is_table") for c in chunks), f"{t} 无表文档不应有表格块"
