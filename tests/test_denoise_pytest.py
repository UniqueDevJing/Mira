"""去噪后处理单测 — 跨页重复/水印/页码/联系方式/整块重复声明。

验证 denoise 在 UIRDocument.__post_init__ 自动生效, 且不误伤正常正文。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engines.parsing.models import UIRDocument


def _doc(pages_blocks: list[list[str]]) -> UIRDocument:
    """构造多页 UIRDocument; pages_blocks[i] = 第 i 页的块文本列表。"""
    pages = []
    for pi, blocks in enumerate(pages_blocks, 1):
        pages.append({
            "page_num": pi,
            "blocks": [
                {"type": "paragraph", "bbox": [], "content": b, "page_num": pi, "metadata": {}}
                for b in blocks
            ],
        })
    return UIRDocument(doc_id="test", source={"type": "txt", "path": "t.txt"}, pages=pages, tables=[])


def _all_contents(doc: UIRDocument) -> list[str]:
    out = []
    for page in doc.pages:
        for b in page["blocks"]:
            out.append(b["content"])
    return out


def test_cross_page_watermark_removed():
    # 5 页, 每页含相同水印行 + 一条独有正文
    pages = [
        ["版权所有 © 2024 示例公司 保留所有权利", f"第 {i} 页独有内容: 关于产品 {i} 的说明。"]
        for i in range(1, 6)
    ]
    doc = _doc(pages)
    contents = _all_contents(doc)
    assert all("版权所有" not in c for c in contents), contents
    # 5 条独有正文应保留
    assert sum("独有内容" in c for c in contents) == 5


def test_page_indicator_removed():
    doc = _doc([["第 3 页", "这是真实正文段落, 描述系统架构。"], ["第 4 页", "另一段正文。"]])
    contents = _all_contents(doc)
    assert all("第 3 页" not in c and "第 4 页" not in c for c in contents), contents
    assert any("真实正文段落" in c for c in contents)
    assert any("另一段正文" in c for c in contents)


def test_contact_line_removed():
    doc = _doc([["support@example.com 客服邮箱", "正文内容 A。"], ["https://www.example.com 官网", "正文内容 B。"]])
    contents = _all_contents(doc)
    assert all("example.com" not in c for c in contents), contents
    assert any("正文内容 A" in c for c in contents)
    assert any("正文内容 B" in c for c in contents)


def test_intra_block_watermark_line_stripped():
    # 多行块内混有水印行 → 仅剔除该行, 块保留
    doc = _doc([["正常第一段。\n版权所有 © 2024 公司\n正常第二段, 描述实现细节。"]])
    contents = _all_contents(doc)
    assert len(contents) == 1
    assert "版权所有" not in contents[0]
    assert "正常第一段" in contents[0] and "正常第二段" in contents[0]


def test_normal_doc_unchanged():
    # 无噪声的正常文档 → 块数与内容完全不变
    pages = [
        ["引言: 项目背景与目标。", "方法: 采用 RAG 架构。"],
        ["实验: 在 390 条业务集上评测。", "结论: Recall@3 达 74%。"],
    ]
    doc = _doc(pages)
    contents = _all_contents(doc)
    assert contents == ["引言: 项目背景与目标。", "方法: 采用 RAG 架构。",
                        "实验: 在 390 条业务集上评测。", "结论: Recall@3 达 74%。"]


def test_duplicate_short_block_collapsed():
    # 同文档内短块完全重复 (重复声明) → 仅保留首块
    doc = _doc([
        ["本文档最终解释权归本公司所有。"],
        ["正文段落一。"],
        ["本文档最终解释权归本公司所有。"],  # 重复声明副本
    ])
    contents = _all_contents(doc)
    assert contents.count("本文档最终解释权归本公司所有。") == 1
    assert "正文段落一。" in contents


def test_long_duplicate_preserved():
    # 长块(>200 字)即便重复也保留, 避免误合并正常长文
    long_a = "长短句" * 100  # 300 字, 远超 200 阈值
    doc = _doc([[long_a], ["其他内容。"], [long_a]])
    contents = _all_contents(doc)
    assert contents.count(long_a) == 2
