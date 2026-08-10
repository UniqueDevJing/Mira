# 多格式解析 + 统一结构分块 — 设计

日期: 2026-08-10
状态: 已批准

## 目标

RAG 2.0 现仅支持 PDF。本设计新增 MD / TXT / DOCX 三种格式，并将分块策略从"语义嵌入切分"改为"结构边界 + 递归字符回退"，砍掉分块阶段的 embedding 开销。

## 分块策略（统一）

所有格式一套逻辑:

```
段落扫描 → 标题栈维护 title_chain → 标题即段边界
  → 段长度 ≤ max_chars ? 单 chunk : RecursiveCharacterTextSplitter 递归切
```

- 标题层级来源: block metadata `heading_level`（MD `#` / DOCX Heading 样式提供真实层级）→ 缺失则 PDF font_size 兜底
- 顺带修复现有 `_find_title_chain` 的 "取最后一个标题" bug — 改为扫描时维护实时标题栈，多章节不串链
- 递归切复用 langchain `RecursiveCharacterTextSplitter`（依赖已在），中文分隔符:
  `["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]`
- 参数配置化（`api/config.py`，`RAG_` 前缀 env 覆盖）:
  - `chunk_max_chars: int = 800`
  - `chunk_overlap: int = 128`

> 行为变更: 现有 PDF chunk 默认 800 tokens×4 = 3200 字符，改为 800 字符后粒度变细、embedding 调用变多。800 字符远在 BGE-small-zh 512 token 上下文内，检索更精准；可用 `RAG_CHUNK_MAX_CHARS=3200` 恢复。

## 架构

```
engines/parsing/
  pdf_parser.py        (改: 标题检测升级 font_size + 加粗 + 编号正则)
  markdown_parser.py   (新: # 标题 → title block，空行分段)
  txt_parser.py        (新: 空行分段 → paragraph block)
  docx_parser.py       (新: Heading 样式 → title block，python-docx)
  registry.py          (新: 扩展名 → parser 注册表)
engines/chunking/
  semantic_chunker.py  → structure_chunker.py (改名 + 重写)
api/routes/documents.py (改: 扩展名校验 + 真实后缀 temp 文件 + 状态区分)
api/config.py          (改: 加分块参数)
pyproject.toml         (+python-docx)
```

**Parser 注册表**: `{".pdf": PDFParser, ".md": MarkdownParser, ".txt": TxtParser, ".docx": DocxParser}`。每个 parser 带 `supported_types` 属性（现有 PDFParser 已有）。选注册表而非 if/elif — 新增格式零改调用方。

## 各格式细节

### PDF
- 标题检测: 字号 ≥ 阈值 + 加粗（span 字体 flags）+ 编号正则（`第[一二三四五六七八九十百千]+章` / `^\d+(\.\d+)*`）组合判断
- 扫描版（OCR）: OCR 输出全为 `paragraph` 块、无 font_size → 自动降级为无标题树 + 递归切

### Markdown
- 行首 `#{1,6} ` → title block（`heading_level` = # 数量）
- 空行分隔的段落 → paragraph block
- 代码块/引用/列表一律按段落处理（保持简单）

### TXT
- 空行分隔段落 → paragraph block
- 无标题 → 全文档单段 → 递归字符切分

### DOCX
- python-docx; `paragraph.style` 以 `Heading `（英文 Word）或 `标题 `（中文 Word）开头 → title block（heading_level 取自样式编号）
- 表格走 `doc.tables` 提取到 `UIRDocument.tables`
- 文本框/SmartArt（`w:drawing`）→ warning 日志跳过，不崩（graceful degradation）

## 数据流

```
upload → 扩展名校验（400 拒绝）→ tmp 文件（真实后缀）
  → registry[ext].parse() → UIRDocument（统一结构）
  → StructureChunker.chunk() → Chunk[]（title_chain + page_range）
  → embedding → 向量库
```

## 错误处理

- 不支持的扩展名 → 400，不碰解析
- 扩展名归一: `ext.lower()` 大小写兼容
- MIME 软校验: `UploadFile.content_type` 与预期不符 → warning 日志；扩展名仍权威（HTTP header 客户端可控，不作信任源）。不引入 python-magic
- python-docx 解析损坏/加密 docx → 捕获 → 文档状态 `error`，不崩
- 空文档（0 段落）→ 状态 `empty`，与 `error` 区分
- 状态枚举: `processing` / `completed` / `error` / `empty`

## 测试 (`tests/test_multiformat_pytest.py`)

- 每格式 parser: 标题/段落产出正确
- Chunker: 标题边界切分、超长递归回退、title_chain 实时栈、中文分隔符断句
- DOCX 用 python-docx 现场生成临时文件测
- Upload: 扩展名拒绝 400、大小写兼容、空文档 `empty`
- 回归: 现有 PDF 测试不挂

## 完成标准

1. 全 pytest 绿
2. ruff 0 错误
3. 四格式（PDF 原生 / PDF 扫描 / MD / TXT / DOCX）各手工上传验证一次

## 风险与缓解

| 风险 | 缓解 |
|:--|:--|
| PDF 语义切→结构切质量回退 | 标题硬边界 + 递归切，标题上下文保留；对比新旧 chunk 边界 |
| `semantic_chunker` rename 破坏 imports | Grep 全量更新 + 跑全套测试 |
| chunk_size 3200→800 行为变更 | 配置化，可 `RAG_CHUNK_MAX_CHARS` 恢复；800 字符在模型上下文内 |
| python-docx 新依赖 | 已说明必要性: Word 结构提取无现成替代在依赖里 |

## 选型变更记录

`docs/design-decisions.md` 需更新: PDF 分块从"语义相似度 + 标题树"改为"标题树硬切 + 递归字符回退"，理由:
1. 语义边界检测每段落一次 embedding 调用，约等于存储 embedding 的 2x 额外成本
2. 标题树（font_size/加粗/编号）纯规则、零成本，已保留主要结构信息
3. 递归字符切用中文分隔符断句，质量与语义切差距小
