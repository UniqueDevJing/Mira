"""Parser 注册表: 扩展名路由 + 大小写归一 + MIME 表"""
from engines.parsing.registry import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_MIME,
    get_parser,
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
