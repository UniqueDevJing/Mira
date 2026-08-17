"""解析层共享数据结构 — 所有格式 parser 统一产出"""

from dataclasses import dataclass, field


@dataclass
class TextBlock:
    type: str
    bbox: list[float]
    content: str
    page_num: int
    metadata: dict = field(default_factory=dict)


@dataclass
class UIRDocument:
    doc_id: str
    source: dict
    pages: list[dict]
    tables: list[dict]
