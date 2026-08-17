"""文本读取工具 — 编码自适应 (UTF-8 BOM → GBK → 兜底 replace)。

Windows 记事本默认 GBK 的中文 txt/md 直接按 utf-8 读会乱码入库 (U+FFFD),
污染向量库与 BM25 索引。按顺序尝试编码, 全失败才用 replace 兜底。
"""

from pathlib import Path


def read_text_auto(path: str, fallbacks: tuple[str, ...] = ("utf-8-sig", "gbk")) -> str:
    """按候选编码顺序读取文本文件, 首个成功解码者胜出。"""
    for enc in fallbacks:
        try:
            return Path(path).read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return Path(path).read_text(encoding="utf-8", errors="replace")
