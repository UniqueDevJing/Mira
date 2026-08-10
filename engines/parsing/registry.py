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
