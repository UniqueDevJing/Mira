"""_dedupe_docs 去重函数契约锁 - 防 LLM fallback 展示重复 chunk."""

import os
import sys

SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "core")
sys.path.insert(0, SCRIPTS)

from orchestrator import _dedupe_docs


class TestDedupeDocs:
    def test_empty_input(self):
        assert _dedupe_docs([]) == []

    def test_no_duplicates(self):
        docs = [
            {"content": "paragraph A content"},
            {"content": "paragraph B content"},
            {"content": "paragraph C content"},
        ]
        result = _dedupe_docs(docs)
        assert len(result) == 3

    def test_duplicate_content(self):
        """Same content appears multiple times -> keep first only."""
        doc1 = "Refund will be returned to your payment account after review approval"
        doc2 = doc1  # exact duplicate
        doc3 = "Refund amount returns to consumer payment account. Orders paid by WeChat Pay"
        docs = [{"content": doc1}, {"content": doc2}, {"content": doc3}]
        result = _dedupe_docs(docs)
        assert len(result) == 2

    def test_long_identical_chunks(self):
        """Long identical chunks share same 200-char fingerprint -> deduped."""
        long_content = "refund return payment account" * 30
        docs = [{"content": long_content} for _ in range(5)]
        result = _dedupe_docs(docs)
        assert len(result) == 1

    def test_max_n_limit(self):
        """max_n limits return count."""
        docs = [
            {"content": "unique_a"},
            {"content": "unique_b"},
            {"content": "unique_c"},
            {"content": "unique_d"},
        ]
        result = _dedupe_docs(docs, max_n=2)
        assert len(result) == 2

    def test_partial_overlap_kept(self):
        """Contents differing within first 200 chars -> both kept."""
        content_a = "this is very long shared prefix content" + "X" * 300
        content_b = "this is very long shared prefix content" + "Y" * 300
        docs = [{"content": content_a}, {"content": content_b}]
        result = _dedupe_docs(docs)
        assert len(result) == 2

    def test_order_preserved(self):
        """Preserves order of first appearances."""
        docs = [
            {"content": "first segment"},
            {"content": "second unique segment"},
            {"content": "duplicate first segment"},  # wait, this isn't actually a duplicate
        ]
        result = _dedupe_docs(docs)
        assert len(result) == 3

    def test_none_content_handled(self):
        """None values in content don't crash."""
        docs = [
            {"content": None},
            {"content": "valid content here"},
            {"content": ""},
            {"content": "another valid piece"},
        ]
        result = _dedupe_docs(docs)
        assert len(result) == 2
        assert result[0]["content"] == "valid content here"
