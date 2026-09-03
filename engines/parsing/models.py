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
    update_time: int = 0  # 文件 mtime (秒); 用于 chunk 元数据携带"更新时间"以支撑新鲜度展示

    def __post_init__(self):
        # 解析后统一去噪(水印/页码/跨页重复/重复声明); 懒加载避免循环依赖, 异常不影响入库
        try:
            from engines.parsing.denoise import denoise_document

            denoise_document(self)
        except Exception:  # noqa: S110, BLE001 -- 去噪异常绝不该阻断入库
            pass
