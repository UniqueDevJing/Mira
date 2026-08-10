# 多格式解析 + 统一结构分块 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 MD / TXT / DOCX 解析，分块从"语义嵌入切分"改为"结构边界 + 递归字符回退"。

**Architecture:** 每格式一个 parser（`engines/parsing/`）产出统一 `UIRDocument`；`StructureChunker` 按标题边界分组、超长段用 `RecursiveCharacterTextSplitter`（中文分隔符）递归切。parser 注册表按扩展名路由，上传路由加扩展名校验。

**Tech Stack:** FastAPI, langchain-text-splitters, python-docx, pytest, ruff。

**Spec:** `docs/superpowers/specs/2026-08-10-multiformat-parsing-design.md`

## Global Constraints

- Python >= 3.12；命令用 `venv/Scripts/python -m pytest`（bash）运行，不激活全局环境
- 所有配置走 `api/config.py` Pydantic Settings，`RAG_` 前缀 env 覆盖；分块参数 `chunk_max_chars=800`、`chunk_overlap=128`
- 依赖装国内镜像: `pip install <pkg> -i https://pypi.tuna.tsinghua.edu.cn/simple`
- 中文分隔符: `["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`
- 测试文件命名 `test_*_pytest.py`（pytest 只收集此模式）；旧脚本 `test_*.py` 不被收集
- 提交用中文 message，附 `Co-Authored-By: Claude <noreply@anthropic.com>`
- 完成标准: 全 pytest 绿 + `ruff check` 0 错误
- 状态枚举（与现网一致）: `processing` / `ready` / `empty` / `failed` / `not_found`

---

### Task 1: UIRDocument 抽取到 models.py

**Files:**
- Create: `engines/parsing/models.py`
- Modify: `engines/parsing/pdf_parser.py:11-25`（删本地 dataclass，改 import）
- Modify: `engines/parsing/ocr.py:20`（改 import 来源）

**Interfaces:**
- Produces: `engines.parsing.models.UIRDocument`（dataclass: `doc_id: str`, `source: dict`, `pages: List[dict]`, `tables: List[dict]`）— 所有 parser 统一产出

- [ ] **Step 1: 写 models.py**

```python
"""解析层共享数据结构 — 所有格式 parser 统一产出"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TextBlock:
    type: str
    bbox: List[float]
    content: str
    page_num: int
    metadata: dict = field(default_factory=dict)


@dataclass
class UIRDocument:
    doc_id: str
    source: dict
    pages: List[dict]
    tables: List[dict]
```

- [ ] **Step 2: 改 pdf_parser.py** — 删除第 11-25 行本地 `TextBlock`/`UIRDocument` 定义，文件头加 `from engines.parsing.models import UIRDocument`（`TextBlock` 若全文件无引用则不 import）

- [ ] **Step 3: 改 ocr.py:20** — `from engines.parsing.pdf_parser import UIRDocument` 改为 `from engines.parsing.models import UIRDocument`

- [ ] **Step 4: 验证** — 跑现有 PDF 解析回归，确认导入不挂

Run: `venv/Scripts/python -c "from engines.parsing.pdf_parser import PDFParser, UIRDocument; from engines.parsing.ocr import OCRProcessor; from engines.parsing.models import TextBlock; print('ok')"`
Expected: `ok`（若 TextBlock 报未引用警告则从 import 去掉 TextBlock）

- [ ] **Step 5: Commit**

```bash
git add engines/parsing/models.py engines/parsing/pdf_parser.py engines/parsing/ocr.py
git commit -m "refactor: UIRDocument 抽取到 parsing/models.py 共享"
```

---

### Task 2: StructureChunker 核心（标题边界 + 递归回退）

**Files:**
- Create: `engines/chunking/structure_chunker.py`
- Create: `tests/test_structure_chunker_pytest.py`
- Modify: `api/config.py`（加两个分块参数）

**Interfaces:**
- Consumes: `UIRDocument`（duck-typed: `doc_id`, `pages[i]["blocks"]`，block 需有 `type`/`content`/`page_num`/`metadata`）
- Produces: `engines.chunking.structure_chunker.StructureChunker(max_chars=800, overlap=128).chunk(uir_doc) -> List[Chunk]`；`api.config.Settings.chunk_max_chars` / `chunk_overlap`

- [ ] **Step 1: config.py 加分块参数**（`api/config.py` 加在 OCR 段后）

```python
    # 分块
    chunk_max_chars: int = 800
    chunk_overlap: int = 128
```

- [ ] **Step 2: 写失败测试** `tests/test_structure_chunker_pytest.py`

```python
"""StructureChunker: 标题边界 + 递归字符回退"""
import pytest

from engines.chunking.structure_chunker import StructureChunker


def _doc(blocks):
    return type("Doc", (), {"doc_id": "d1", "pages": [{"page_num": 1, "blocks": blocks}]})


def _title(text, level=None):
    md = {"heading_level": level} if level is not None else {}
    return {"type": "title", "content": text, "page_num": 1, "metadata": md}


def _para(text):
    return {"type": "paragraph", "content": text, "page_num": 1, "metadata": {}}


def test_empty_doc_returns_no_chunks():
    assert StructureChunker().chunk(_doc([])) == []


def test_title_is_segment_boundary():
    blocks = [_title("第一章 概述", 1), _para("正文A"), _title("第二章", 1), _para("正文B")]
    chunks = StructureChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert "正文A" in chunks[0].content and "正文B" not in chunks[0].content
    assert chunks[1].context["title_chain"] == ["第二章"]


def test_nested_title_chain_keeps_parents():
    blocks = [_title("H1", 1), _para("p1"), _title("H2", 2), _para("p2")]
    chunks = StructureChunker().chunk(_doc(blocks))
    assert len(chunks) == 2
    assert chunks[1].context["title_chain"] == ["H1", "H2"]


def test_oversized_segment_recursively_split():
    long = _para("测试内容。" * 300)
    chunks = StructureChunker(max_chars=200, overlap=20).chunk(_doc([_title("T", 1), long]))
    assert len(chunks) >= 2
    assert all(len(c.content) <= 200 + 40 for c in chunks)  # chunk_size + overlap 容差
    assert all(c.context["title_chain"] == ["T"] for c in chunks)


def test_no_title_single_segment_split():
    long = _para("无标题文本。" * 200)
    chunks = StructureChunker(max_chars=100, overlap=0).chunk(_doc([long]))
    assert len(chunks) >= 2


def test_chinese_separator_break_no_mid_sentence():
    text = "第一句。" + "第二句。" + "第三句。"
    chunks = StructureChunker(max_chars=12, overlap=0).chunk(_doc([_para(text)]))
    # 递归切在中文标点断句, 不在字符中间腰斩
    assert all(c.content.endswith("。") for c in chunks)
```

- [ ] **Step 3: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_structure_chunker_pytest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'engines.chunking.structure_chunker'`

- [ ] **Step 4: 写实现** `engines/chunking/structure_chunker.py`

```python
"""基于文档结构的统一分块: 标题边界 + 递归字符回退"""
import logging
import re
from typing import List, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter

from engines.interfaces import Chunk

logger = logging.getLogger(__name__)

_CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


class StructureChunker:
    def __init__(self, max_chars: int = 800, overlap: int = 128):
        self.max_chars = max_chars
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars, chunk_overlap=overlap, separators=_CHINESE_SEPARATORS
        )

    def chunk(self, uir_doc) -> List[Chunk]:
        blocks = self._flatten_blocks(uir_doc)
        if not blocks:
            return []
        chains = self._heading_chains(blocks)
        segments = self._split_segments(blocks)
        chunks = []
        for start_idx, seg in segments:
            title_chain = chains[start_idx]
            content = "\n\n".join(b["content"] for b in seg)
            if len(content) <= self.max_chars:
                chunks.append(self._make_chunk(uir_doc.doc_id, content, seg, title_chain, len(chunks)))
            else:
                for part in self._splitter.split_text(content):
                    chunks.append(self._make_chunk(uir_doc.doc_id, part, seg, title_chain, len(chunks)))
        return chunks

    @staticmethod
    def _flatten_blocks(uir_doc) -> List[dict]:
        blocks = []
        for page in uir_doc.pages:
            for block in page["blocks"]:
                if block["type"] in ("paragraph", "title"):
                    blocks.append(block)
        return blocks

    @staticmethod
    def _heading_chains(blocks: List[dict]) -> List[List[str]]:
        stack = []  # [(level, text)]
        chains = []
        for block in blocks:
            if block["type"] == "title":
                level = block.get("metadata", {}).get("heading_level")
                if level is None:
                    level = StructureChunker._estimate_level(block)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, block["content"]))
            chains.append([t for _, t in stack])
        return chains

    @staticmethod
    def _split_segments(blocks: List[dict]) -> List[Tuple[int, List[dict]]]:
        segments = []
        start = 0
        for i, block in enumerate(blocks):
            if i > 0 and block["type"] == "title":
                segments.append((start, blocks[start:i]))
                start = i
        segments.append((start, blocks[start:]))
        return segments

    @staticmethod
    def _estimate_level(block: dict) -> int:
        """PDF 无 heading_level 时的兜底: font_size + 加粗 + 编号正则"""
        md = block.get("metadata", {})
        size = md.get("font_size", 12)
        if md.get("is_bold") or re.match(r"^第[一二三四五六七八九十百千]+[章节篇]", block["content"]):
            size = max(size, 16)
        if size >= 22:
            return 1
        elif size >= 18:
            return 2
        elif size >= 14:
            return 3
        return 4

    @staticmethod
    def _make_chunk(doc_id: str, content: str, segment: List[dict],
                    title_chain: List[str], index: int) -> Chunk:
        first, last = segment[0], segment[-1]
        return Chunk(
            chunk_id=f"{doc_id}_chunk_{index:04d}",
            doc_id=doc_id,
            content=content.strip(),
            context={"title_chain": title_chain, "doc_title": doc_id},
            metadata={
                "page_range": [first.get("page_num", 1), last.get("page_num", 1)],
                "char_count": len(content),
                "paragraph_count": len(segment),
            },
        )
```

> 若 `from langchain_text_splitters import ...` 报 ImportError，改为 `from langchain.text_splitter import RecursiveCharacterTextSplitter`（langchain 0.3 兼容路径）。

- [ ] **Step 5: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_structure_chunker_pytest.py -v`
Expected: PASS（6 个用例全绿）

- [ ] **Step 6: Commit**

```bash
git add engines/chunking/structure_chunker.py tests/test_structure_chunker_pytest.py api/config.py
git commit -m "feat: StructureChunker 标题边界 + 递归字符回退分块"
```

---

### Task 3: 删除 semantic_chunker + 更新全部引用

**Files:**
- Delete: `engines/chunking/semantic_chunker.py`
- Modify: `engines/chunking/__init__.py`
- Modify: `tests/conftest.py:56`
- Modify: `tests/test_vector_store_pytest.py:5`
- Modify: `tests/test_e2e_graph.py:25`、`tests/test_full_e2e.py:14`、`tests/test_integration.py:9`、`tests/test_minimal.py:19`、`tests/test_phase2.py:23`、`tests/test_real_e2e.py:33`（旧脚本 import 一行替换）

**Interfaces:**
- Consumes: Task 2 的 `StructureChunker`
- Produces: 移除 `engines.chunking.semantic_chunker` 模块，`engines.chunking` 只导出 `StructureChunker` / `Chunk`

- [ ] **Step 1: 改 `engines/chunking/__init__.py`**

```python
from engines.chunking.structure_chunker import StructureChunker, Chunk
```

- [ ] **Step 2: 更新活代码引用**
- `tests/conftest.py:56`: `from engines.chunking.semantic_chunker import Chunk` → `from engines.chunking.structure_chunker import Chunk`
- `tests/test_vector_store_pytest.py:5`: 同上替换
- 6 个旧脚本（test_e2e_graph/test_full_e2e/test_integration/test_minimal/test_phase2/test_real_e2e）: `from engines.chunking.semantic_chunker import SemanticChunker` → `from engines.chunking.structure_chunker import StructureChunker`，并把后续 `SemanticChunker(` 用法同步替换为 `StructureChunker(`

- [ ] **Step 3: 删除旧模块** — `git rm engines/chunking/semantic_chunker.py`

- [ ] **Step 4: 验证**

Run: `venv/Scripts/python -m pytest tests/test_vector_store_pytest.py tests/test_structure_chunker_pytest.py -v`
Expected: PASS（旧脚本不被 pytest 收集，仅需 grep 确认无残留引用）

- [ ] **Step 5: 无残留引用确认**

Run: `venv/Scripts/python -c "import engines.chunking; print('ok')"`
Expected: `ok`。再 Grep `semantic_chunker|SemanticChunker` 于 `*.py`，Expected: 0 命中

- [ ] **Step 6: Commit**

```bash
git add -A engines/chunking tests/conftest.py tests/test_vector_store_pytest.py tests/test_e2e_graph.py tests/test_full_e2e.py tests/test_integration.py tests/test_minimal.py tests/test_phase2.py tests/test_real_e2e.py
git commit -m "refactor: semantic_chunker 更名 structure_chunker, 更新全部引用"
```

---

### Task 4: MarkdownParser

**Files:**
- Create: `engines/parsing/markdown_parser.py`
- Create: `tests/test_markdown_parser_pytest.py`

**Interfaces:**
- Consumes: `engines.parsing.models.UIRDocument`（Task 1）
- Produces: `MarkdownParser.supported_types = [".md", ".markdown"]`；`MarkdownParser.parse(file_path: str) -> UIRDocument`

- [ ] **Step 1: 写失败测试** `tests/test_markdown_parser_pytest.py`

```python
"""MarkdownParser: # 标题 → title block, 空行分段"""
import os

import pytest

from engines.parsing.markdown_parser import MarkdownParser


@pytest.fixture
def md_path(tmp_path):
    p = tmp_path / "sample.md"
    p.write_text("# 第一章 概述\n\n这是第一段。\n\n## 1.1 背景\n\n这是背景段落。", encoding="utf-8")
    return str(p)


def test_heading_level_detected(md_path):
    uir = MarkdownParser().parse(md_path)
    titles = [b for b in uir.pages[0]["blocks"] if b["type"] == "title"]
    assert [t["content"] for t in titles] == ["第一章 概述", "1.1 背景"]
    assert [t["metadata"]["heading_level"] for t in titles] == [1, 2]


def test_paragraph_block(md_path):
    uir = MarkdownParser().parse(md_path)
    paras = [b for b in uir.pages[0]["blocks"] if b["type"] == "paragraph"]
    assert any("这是第一段" in b["content"] for b in paras)


def test_doc_id_deterministic(md_path):
    assert MarkdownParser().parse(md_path).doc_id == MarkdownParser().parse(md_path).doc_id
```

- [ ] **Step 2: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_markdown_parser_pytest.py -v`
Expected: FAIL — `No module named 'engines.parsing.markdown_parser'`

- [ ] **Step 3: 写实现** `engines/parsing/markdown_parser.py`

```python
"""Markdown 解析: # 标题 → title block, 空行分段"""
import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument

_TITLE_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


class MarkdownParser:
    supported_types = [".md", ".markdown"]

    def parse(self, file_path: str) -> UIRDocument:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        raw_paras = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = []
        for para in raw_paras:
            cleaned = "\n".join(line.strip() for line in para.splitlines() if line.strip())
            m = _TITLE_RE.match(cleaned)
            if m:
                blocks.append({
                    "type": "title", "bbox": [], "content": m.group(2),
                    "page_num": 1, "metadata": {"heading_level": len(m.group(1))},
                })
            else:
                blocks.append({
                    "type": "paragraph", "bbox": [], "content": cleaned,
                    "page_num": 1, "metadata": {},
                })
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "markdown", "path": file_path},
                           pages=[{"page_num": 1, "blocks": blocks}], tables=[])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_markdown_parser_pytest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engines/parsing/markdown_parser.py tests/test_markdown_parser_pytest.py
git commit -m "feat: MarkdownParser 标题层级 + 空行分段"
```

---

### Task 5: TxtParser

**Files:**
- Create: `engines/parsing/txt_parser.py`
- Create: `tests/test_txt_parser_pytest.py`

**Interfaces:**
- Produces: `TxtParser.supported_types = [".txt"]`；`TxtParser.parse(file_path: str) -> UIRDocument`

- [ ] **Step 1: 写失败测试** `tests/test_txt_parser_pytest.py`

```python
"""TxtParser: 空行分段, 全为 paragraph"""
import pytest

from engines.parsing.txt_parser import TxtParser


@pytest.fixture
def txt_path(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("第一段文本。\n\n第二段文本。", encoding="utf-8")
    return str(p)


def test_paragraphs_split_by_blank_line(txt_path):
    uir = TxtParser().parse(txt_path)
    blocks = uir.pages[0]["blocks"]
    assert len(blocks) == 2
    assert all(b["type"] == "paragraph" for b in blocks)
    assert "第一段" in blocks[0]["content"]


def test_no_title_blocks(txt_path):
    uir = TxtParser().parse(txt_path)
    assert not any(b["type"] == "title" for b in uir.pages[0]["blocks"])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_txt_parser_pytest.py -v`
Expected: FAIL — `No module named 'engines.parsing.txt_parser'`

- [ ] **Step 3: 写实现** `engines/parsing/txt_parser.py`

```python
"""TXT 解析: 空行分段, 全为 paragraph 块"""
import hashlib
import re
from pathlib import Path

from engines.parsing.models import UIRDocument


class TxtParser:
    supported_types = [".txt"]

    def parse(self, file_path: str) -> UIRDocument:
        text = Path(file_path).read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        blocks = [
            {"type": "paragraph", "bbox": [], "content": p, "page_num": 1, "metadata": {}}
            for p in paragraphs
        ]
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "txt", "path": file_path},
                           pages=[{"page_num": 1, "blocks": blocks}], tables=[])
```

- [ ] **Step 4: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_txt_parser_pytest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engines/parsing/txt_parser.py tests/test_txt_parser_pytest.py
git commit -m "feat: TxtParser 空行分段"
```

---

### Task 6: DocxParser

**Files:**
- Create: `engines/parsing/docx_parser.py`
- Create: `tests/test_docx_parser_pytest.py`
- Modify: `pyproject.toml`（加 python-docx）

**Interfaces:**
- Produces: `DocxParser.supported_types = [".docx"]`；`DocxParser.parse(file_path: str) -> UIRDocument`

- [ ] **Step 1: 装依赖**

Run: `venv/Scripts/python -m pip install "python-docx>=1.1.0" -i https://pypi.tuna.tsinghua.edu.cn/simple`
Expected: `Successfully installed python-docx-...`

- [ ] **Step 2: pyproject.toml 加依赖** — 在 `"pillow>=11.0.0",` 行后插入 `"python-docx>=1.1.0",`

- [ ] **Step 3: 写失败测试** `tests/test_docx_parser_pytest.py`

```python
"""DocxParser: Heading 样式 → title block, 表格提取, 非段落内容降级"""
from docx import Document

from engines.parsing.docx_parser import DocxParser


def _make_docx(tmp_path) -> str:
    doc = Document()
    doc.add_heading("第一章 概述", level=1)
    doc.add_paragraph("这是正文段落。")
    doc.add_heading("1.1 背景", level=2)
    table = doc.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text = "列A"
    table.rows[0].cells[1].text = "列B"
    table.rows[1].cells[0].text = "v1"
    table.rows[1].cells[1].text = "v2"
    p = tmp_path / "sample.docx"
    doc.save(str(p))
    return str(p)


def test_heading_levels(tmp_path):
    uir = DocxParser().parse(_make_docx(tmp_path))
    titles = [b for b in uir.pages[0]["blocks"] if b["type"] == "title"]
    assert [t["content"] for t in titles] == ["第一章 概述", "1.1 背景"]
    assert [t["metadata"]["heading_level"] for t in titles] == [1, 2]


def test_paragraph_block(tmp_path):
    uir = DocxParser().parse(_make_docx(tmp_path))
    paras = [b for b in uir.pages[0]["blocks"] if b["type"] == "paragraph"]
    assert any("正文段落" in b["content"] for b in paras)


def test_tables_extracted(tmp_path):
    uir = DocxParser().parse(_make_docx(tmp_path))
    assert len(uir.tables) == 1
    assert uir.tables[0]["matrix"][0] == ["列A", "列B"]
```

- [ ] **Step 4: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_docx_parser_pytest.py -v`
Expected: FAIL — `No module named 'engines.parsing.docx_parser'`

- [ ] **Step 5: 写实现** `engines/parsing/docx_parser.py`

```python
"""DOCX 解析: Heading 样式 → title block; 文本框/SmartArt 优雅降级"""
import hashlib
import re
from pathlib import Path

from docx import Document as DocxDocument

from engines.parsing.models import UIRDocument


class DocxParser:
    supported_types = [".docx"]

    def parse(self, file_path: str) -> UIRDocument:
        doc = DocxDocument(file_path)
        blocks = []
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            style = p.style.name if p.style else ""
            if style.startswith("Heading") or style.startswith("标题"):
                blocks.append({
                    "type": "title", "bbox": [], "content": text, "page_num": 1,
                    "metadata": {"heading_level": self._heading_level(style)},
                })
            else:
                blocks.append({
                    "type": "paragraph", "bbox": [], "content": text, "page_num": 1,
                    "metadata": {},
                })
        tables = []
        for t in doc.tables:
            rows = [[cell.text.strip() for cell in row.cells] for row in t.rows]
            if rows:
                tables.append({"page_num": 1, "bbox": [], "matrix": rows, "headers": rows[0]})
        # 文本框/SmartArt/嵌套表格: doc.paragraphs 不返回, 自动跳过 (graceful degradation)
        doc_id = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()[:16]
        return UIRDocument(doc_id=doc_id, source={"type": "docx", "path": file_path},
                           pages=[{"page_num": 1, "blocks": blocks}], tables=tables)

    @staticmethod
    def _heading_level(style: str) -> int:
        m = re.search(r"(\d+)", style)
        return min(int(m.group(1)), 6) if m else 1
```

- [ ] **Step 6: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_docx_parser_pytest.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add engines/parsing/docx_parser.py tests/test_docx_parser_pytest.py pyproject.toml
git commit -m "feat: DocxParser Heading 样式解析 + 表格提取"
```

---

### Task 7: Parser 注册表

**Files:**
- Create: `engines/parsing/registry.py`
- Create: `tests/test_registry_pytest.py`

**Interfaces:**
- Consumes: 四个 parser 的 `supported_types` 类属性（Task 4/5/6 + 现有 PDFParser）
- Produces: `engines.parsing.registry.get_parser(ext: str) -> parser|None`；`SUPPORTED_EXTENSIONS: set[str]`；`SUPPORTED_MIME: dict[str, set[str]]`

- [ ] **Step 1: 写失败测试** `tests/test_registry_pytest.py`

```python
"""Parser 注册表: 扩展名路由 + 大小写归一 + MIME 表"""
from engines.parsing.registry import (
    SUPPORTED_EXTENSIONS, SUPPORTED_MIME, get_parser,
)


def test_supported_extensions():
    assert ".pdf" in SUPPORTED_EXTENSIONS
    assert ".md" in SUPPORTED_EXTENSIONS
    assert ".txt" in SUPPORTED_EXTENSIONS
    assert ".docx" in SUPPORTED_EXTENSIONS
    assert ".doc" not in SUPPORTED_EXTENSIONS


def test_get_parser_case_insensitive():
    from engines.parsing.docx_parser import DocxParser
    assert isinstance(get_parser(".DOCX"), DocxParser)
    assert isinstance(get_parser(".docx"), DocxParser)


def test_get_parser_unknown_returns_none():
    assert get_parser(".doc") is None
    assert get_parser("") is None


def test_mime_table_covers_all_extensions():
    assert SUPPORTED_EXTENSIONS <= set(SUPPORTED_MIME)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_registry_pytest.py -v`
Expected: FAIL — `No module named 'engines.parsing.registry'`

- [ ] **Step 3: 写实现** `engines/parsing/registry.py`

```python
"""解析器注册表 — 扩展名 → parser; 扩展名大小写归一后作权威路由"""
from engines.parsing.pdf_parser import PDFParser
from engines.parsing.markdown_parser import MarkdownParser
from engines.parsing.txt_parser import TxtParser
from engines.parsing.docx_parser import DocxParser

_PARSERS = (PDFParser, MarkdownParser, TxtParser, DocxParser)

SUPPORTED_EXTENSIONS = {ext for cls in _PARSERS for ext in cls.supported_types}
SUPPORTED_PARSERS = {ext: cls for cls in _PARSERS for ext in cls.supported_types}

# MIME 软校验表 (仅日志警告, 不阻断; HTTP header 客户端可控, 不可作信任源)
SUPPORTED_MIME = {
    ".pdf": {"application/pdf"},
    ".md": {"text/markdown", "text/plain"},
    ".markdown": {"text/markdown", "text/plain"},
    ".txt": {"text/plain"},
    ".docx": {"application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
}


def get_parser(ext: str):
    cls = SUPPORTED_PARSERS.get((ext or "").lower())
    return cls() if cls else None
```

- [ ] **Step 4: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_registry_pytest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engines/parsing/registry.py tests/test_registry_pytest.py
git commit -m "feat: parser 注册表按扩展名路由"
```

---

### Task 8: 上传路由改造（扩展名校验 + 真实后缀 + 统一分块 + empty 状态）

**Files:**
- Modify: `api/routes/documents.py`
- Create: `tests/test_documents_multiformat_pytest.py`

**Interfaces:**
- Consumes: `engines.parsing.registry.get_parser` / `SUPPORTED_MIME`（Task 7），`StructureChunker`（Task 2），`api.config.settings`（Task 2）
- Produces: upload 路由拒绝未知扩展名（400）；pipeline 按扩展名选 parser、真实后缀写 temp、0 chunk 返回且后台状态为 `empty`

- [ ] **Step 1: 写失败测试** `tests/test_documents_multiformat_pytest.py`

```python
"""上传路由: 扩展名校验 + 空文档状态"""
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_upload_rejects_unknown_extension():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.doc", b"content", "application/msword")})
    assert r.status_code == 400


def test_upload_accepts_markdown():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.md", b"# 标题\n\n正文", "text/markdown")})
    assert r.status_code == 200
    assert r.json()["status"] == "processing"


def test_upload_rejects_uppercase_unknown():
    r = client.post("/api/v1/documents/upload",
                    files={"file": ("a.XLSX", b"x", "application/octet-stream")})
    assert r.status_code == 400
```

> 注: 若 `api.main` 引入触发外部依赖（Redis/LLM 等），测试前确认 `conftest.py` 的环境变量已设置（当前已设置 HF/transformers 离线变量）。若 TestClient 启动失败，改为直接测 `_process_document_pipeline` 的扩展名分支（见 Step 5 fallback）。

- [ ] **Step 2: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_documents_multiformat_pytest.py -v`
Expected: FAIL — 上传 `.doc` 返回 200（现无校验）

- [ ] **Step 3: 改 upload_document**（`api/routes/documents.py`）

import 区加:
```python
import os
from fastapi import HTTPException
from engines.parsing.registry import get_parser, SUPPORTED_MIME
```

`upload_document` 内、`doc_store.save` 之前加:
```python
    ext = os.path.splitext(file.filename or "")[1].lower()
    if get_parser(ext) is None:
        raise HTTPException(status_code=400, detail=f"不支持的格式: {ext or '未知'}")
    # MIME 软校验: 与扩展名不符仅告警, 扩展名仍权威
    if file.content_type:
        expected = SUPPORTED_MIME.get(ext, set())
        if expected and file.content_type.lower() not in expected:
            logger.warning("[%s] MIME 与扩展名不符: filename=%s, content_type=%s",
                           doc_id, file.filename, file.content_type)
```

- [ ] **Step 4: 改 `_process_document_background`** — `ready` 状态分叉出 `empty`:

```python
        result = await asyncio.to_thread(_process_document_pipeline, doc_id, filename, content, kb)
        status = "empty" if result.get("chunks", 0) == 0 else "ready"
        doc_store.update_status(doc_id, status, page_count=result.get("pages", 0), chunk_count=result.get("chunks", 0))
```

- [ ] **Step 5: 改 `_process_document_pipeline`** — import 区替换 + 真实后缀 + 0 chunk 早退:

```python
    import tempfile, os
    from engines.parsing.registry import get_parser
    from engines.chunking.structure_chunker import StructureChunker
    from engines.embedding.embedder import EmbeddingService
    from api.config import settings

    tmp_path = None
    ext = os.path.splitext(filename)[1].lower()
    parser = get_parser(ext)
    if parser is None:
        raise ValueError(f"不支持的格式: {ext}")
    try:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        tmp_path = tmp.name
        tmp.write(content)
        tmp.close()

        uir = parser.parse(tmp_path)
        logger.info("[%s] 解析完成: %d 页", doc_id, len(uir.pages))

        chunker = StructureChunker(max_chars=settings.chunk_max_chars, overlap=settings.chunk_overlap)
        chunks = chunker.chunk(uir)
        logger.info("[%s] 分块完成: %d chunks", doc_id, len(chunks))

        if not chunks:
            return {"pages": len(uir.pages), "chunks": 0}
```

（embedding / 向量库 / 图谱三段保持原样不变）

- [ ] **Step 6: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_documents_multiformat_pytest.py -v`
Expected: PASS（3 用例绿）

- [ ] **Step 7: Commit**

```bash
git add api/routes/documents.py tests/test_documents_multiformat_pytest.py
git commit -m "feat: 上传扩展名校验 + 按格式路由 + empty 状态"
```

---

### Task 9: PDFParser 标题检测升级（font_size + 加粗 + 编号）

**Files:**
- Modify: `engines/parsing/pdf_parser.py`
- Create: `tests/test_pdf_title_pytest.py`

**Interfaces:**
- Consumes: 现有 `PDFParser._classify_block`
- Produces: block metadata 含 `is_bold`；`_classify_block` 返回 `title` 当 font_size>16 或加粗或编号正则命中

- [ ] **Step 1: 写失败测试** `tests/test_pdf_title_pytest.py`

```python
"""PDF 标题检测: 字号 + 加粗 + 编号组合"""
from engines.parsing.pdf_parser import PDFParser


def _block(text, size=12, flags=0, y=100):
    return {
        "bbox": [50, y, 400, y + 30],
        "lines": [{"spans": [{"text": text, "size": size, "flags": flags}]}],
    }


def test_bold_is_title():
    assert PDFParser()._classify_block(_block("摘要", size=12, flags=16), 842) == "title"


def test_large_font_is_title():
    assert PDFParser()._classify_block(_block("章节标题", size=20, flags=0), 842) == "title"


def test_numbered_chinese_title():
    assert PDFParser()._classify_block(_block("第一章 概述", size=12, flags=0), 842) == "title"


def test_numbered_arabic_title():
    assert PDFParser()._classify_block(_block("1.1 背景", size=12, flags=0), 842) == "title"


def test_plain_paragraph_not_title():
    assert PDFParser()._classify_block(_block("普通正文文本", size=12, flags=0), 842) == "paragraph"
```

- [ ] **Step 2: 跑测试确认失败**

Run: `venv/Scripts/python -m pytest tests/test_pdf_title_pytest.py -v`
Expected: FAIL — 加粗/编号用例返回 `paragraph`

- [ ] **Step 3: 改 pdf_parser.py**

文件头加 `import re`。`_parse_native` 的 metadata 加 `is_bold`:
```python
                            blocks.append({
                                "type": self._classify_block(block, page_height),
                                "bbox": list(block["bbox"]),
                                "content": text.strip(),
                                "page_num": page_num,
                                "metadata": {
                                    "font_size": line["spans"][0]["size"] if line["spans"] else 0,
                                    "is_bold": bool(line["spans"][0]["flags"] & 16) if line["spans"] else False,
                                }
                            })
```

`_classify_block` 替换为:
```python
    _NUMBERED_TITLE_RE = re.compile(r"^(第[一二三四五六七八九十百千]+[章节篇]|(\d+(\.\d+)*)[、\s.])")

    def _classify_block(self, block: dict, page_height: float = 842) -> str:
        """根据位置/字号/加粗/编号分类文本块。

        page_height 默认 A4 (842pt)，由调用方传入实际页面高度。
        """
        bbox = block["bbox"]
        if bbox[1] < 50:
            return "header"
        elif bbox[1] > page_height - 50:
            return "footer"
        spans = block["lines"][0]["spans"] if block.get("lines") else []
        size = spans[0]["size"] if spans else 0
        bold = bool(spans[0]["flags"] & 16) if spans else False
        text = "".join(s["text"] for line in block.get("lines", []) for s in line["spans"])
        if size > 16 or bold or self._NUMBERED_TITLE_RE.match(text):
            return "title"
        return "paragraph"
```

- [ ] **Step 4: 跑测试确认通过**

Run: `venv/Scripts/python -m pytest tests/test_pdf_title_pytest.py tests/test_structure_chunker_pytest.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add engines/parsing/pdf_parser.py tests/test_pdf_title_pytest.py
git commit -m "feat: PDF 标题检测升级 字号+加粗+编号正则"
```

---

### Task 10: 回归 + 设计文档更新 + 完成标准

**Files:**
- Modify: `docs/design-decisions.md`（分块策略选型变更记录）
- Modify: `docs/superpowers/specs/2026-08-10-multiformat-parsing-design.md`（状态枚举修正，见下）

**Interfaces:**
- 无新接口

- [ ] **Step 1: 修正 spec 状态枚举** — spec 原文写 `processing/completed/error/empty`，与现网实际 `processing/ready/empty/failed/not_found` 不符。改为后者并同步到 Global Constraints（已如此写）

- [ ] **Step 2: 更新 design-decisions.md** — 定位"分块策略"行，选择列改为「标题树硬切 + 递归字符回退（含中文分隔符）」，并在该表下方追加变更记录段落:

```markdown
### 2026-08-10 变更: PDF 分块从语义切分改为结构切分

原方案「语义相似度 + 标题树动态切分」废除，理由:
1. 语义边界检测每段落一次 embedding 调用，约为存储 embedding 的 2x 额外成本
2. 标题树（font_size/加粗/编号）纯规则零成本，已保留主要结构信息
3. 递归字符切用中文分隔符（。！？；，）断句，质量与语义切差距小

替代排除: RecursiveCharacterTextSplitter 单独使用（无标题上下文，弃）、全格式语义切（TXT 无结构白烧 embedding，弃）。
```

- [ ] **Step 3: 全量测试**

Run: `venv/Scripts/python -m pytest -m "not slow and not integration"`
Expected: 全部 PASS，旧语义分块相关断言如不适用则按测试实际调整（保留回归意图）

- [ ] **Step 4: ruff**

Run: `venv/Scripts/python -m ruff check .`
Expected: `All checks passed!`（0 错误）

- [ ] **Step 5: 提交设计文档更新**

```bash
git add docs/design-decisions.md docs/superpowers/specs/2026-08-10-multiformat-parsing-design.md
git commit -m "docs: 分块策略选型变更记录 + spec 状态枚举修正"
```

- [ ] **Step 6: 手工验证**（用户侧，非自动）— 各上传一份: 原生 PDF / 扫描 PDF / `.md` / `.txt` / `.docx`，确认解析分块入库，0 chunk 的文档状态为 `empty`

---

## Self-Review 记录

**Spec 覆盖:** 统一分块策略 → Task 2；中文分隔符 → Task 2；PDF 标题组合 + 扫描版降级 → Task 9（OCR 全 paragraph 无 font_size，自动降级无标题，由 Task 2 chunker 处理）；注册表大小写 + MIME → Task 7/8；空文档 empty → Task 8；DOCX 降级 → Task 6；参数配置化 → Task 2；CI 全绿 → Task 10。覆盖完整。

**占位符检查:** 无 TBD/TODO；每步含完整代码。Task 10 Step 2 的 design-decisions.md 定位依赖该文件现有结构，已在步骤中说明定位方式。

**类型一致性:** `StructureChunker(max_chars, overlap)` 在 Task 2 定义、Task 8 以 `settings` 传参调用，签名一致；`get_parser(ext)` Task 7 定义、Task 8 消费，一致；`Chunk.context["title_chain"]` Task 2 产出、测试断言一致；`UIRDocument` Task 1 定义、Task 4/5/6 消费，字段 `doc_id/source/pages/tables` 一致。
