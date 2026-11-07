"""多模态检索基座测试（W3）。

覆盖四类约束：
1. **溯源硬要求**：没有 source 的图不允许入库；
2. **内容寻址去重**：同一张图重复入库不产生第二条记录；
3. **检索可解释**：分数能拆成关键词与向量两部分，中文 2-gram 能命中；
4. **召回可评测**：recall@k / MRR 的边界（空期望集、未命中、命中在第 2 位）。
"""

from __future__ import annotations

import struct
import zlib

import pytest

from engines.multimodal import (
    ByteHistogramEncoder,
    ImageIndex,
    ImageRetriever,
    MissingProvenance,
    cosine,
    mrr,
    png_size,
    recall_at_k,
    tokenize,
)


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def build_index() -> ImageIndex:
    index = ImageIndex()
    index.add(
        make_png(8, 8, (200, 30, 30)),
        kb="policy",
        caption="高血压分级示意图 收缩压 舒张压",
        source="内科学 第9版",
        page=112,
        fetched_at="2026-11-01T00:00:00Z",
    )
    index.add(
        make_png(8, 8, (30, 30, 200)),
        kb="policy",
        caption="退款流程图 客服处理时效",
        source="售后政策.pdf",
        page=7,
    )
    index.add(
        make_png(16, 8, (30, 200, 30)),
        kb="finance",
        caption="资产负债表结构图",
        source="财报 2025Q3",
        page=3,
    )
    return index


# ───────────── 溯源与去重 ─────────────

def test_missing_provenance_is_rejected():
    index = ImageIndex()
    with pytest.raises(MissingProvenance):
        index.add(make_png(4, 4, (1, 2, 3)), kb="policy", caption="无来源", source="")


def test_same_bytes_deduped_by_content_hash():
    index = build_index()
    before = len(index)
    entry, created = index.add(
        make_png(8, 8, (200, 30, 30)),
        kb="policy",
        caption="重复入库",
        source="另一个来源",
    )
    assert created is False
    assert len(index) == before
    assert entry.source == "内科学 第9版", "去重后应保留首次入库的溯源信息"


def test_asset_id_is_content_hash():
    data = make_png(4, 4, (9, 9, 9))
    index = ImageIndex()
    entry, _ = index.add(data, kb="k", caption="c", source="s")
    import hashlib

    assert entry.asset_id == hashlib.sha256(data).hexdigest()


# ───────────── 编码器 ─────────────

def test_encoder_is_deterministic_and_normalized():
    encoder = ByteHistogramEncoder()
    data = make_png(8, 8, (10, 20, 30))
    first = encoder.encode(data)
    second = encoder.encode(data)
    assert first == second
    assert len(first) == 32
    assert abs(cosine(first, first) - 1.0) < 1e-9


def test_encoder_distinguishes_different_images():
    encoder = ByteHistogramEncoder()
    a = encoder.encode(make_png(8, 8, (255, 0, 0)))
    b = encoder.encode(make_png(8, 8, (0, 0, 255)))
    assert cosine(a, b) < 0.999


def test_encoder_rejects_empty():
    with pytest.raises(ValueError):
        ByteHistogramEncoder().encode(b"")


def test_png_size_reads_header_and_tolerates_non_png():
    assert png_size(make_png(16, 8, (0, 0, 0))) == (16, 8)
    assert png_size(b"not-a-png") == (0, 0)


# ───────────── 检索 ─────────────

def test_tokenize_cjk_bigrams_and_words():
    tokens = tokenize("高血压 CT 检查")
    assert "高血" in tokens and "血压" in tokens
    assert "ct" in tokens


def test_retriever_finds_by_caption_keyword():
    hits = ImageRetriever(build_index()).search("高血压 示意图", k=3)
    assert hits, "应至少命中一条"
    assert "高血压" in hits[0].entry.caption
    assert hits[0].keyword > 0
    assert hits[0].score > 0


def test_retriever_respects_kb_filter():
    tool = ImageRetriever(build_index())
    assert all(h.entry.kb == "finance" for h in tool.search("结构图", k=5, kb="finance"))
    assert tool.search("结构图", k=5, kb="不存在的库") == []


def test_hit_payload_is_json_ready_with_provenance():
    hit = ImageRetriever(build_index()).search("退款流程图", k=1)[0]
    payload = hit.to_json()
    assert payload["type"] == "image"
    assert payload["source"] == "售后政策.pdf"
    assert payload["page"] == 7
    assert set(payload) >= {"asset_id", "caption", "score", "source"}


def test_search_by_similar_image():
    index = build_index()
    hits = index.search_similar(make_png(8, 8, (200, 30, 30)), k=1)
    assert hits and hits[0][1] > 0.99


# ───────────── 评测指标 ─────────────

def test_recall_at_k_boundaries():
    index = build_index()
    hits = ImageRetriever(index).search("高血压", k=3)
    target = hits[0].entry.asset_id
    assert recall_at_k(hits, {target}, k=1) == 1.0
    assert recall_at_k(hits, {"not-there"}, k=3) == 0.0
    assert recall_at_k(hits, set(), k=3) == 1.0, "空期望集视为无需召回"
    assert recall_at_k([], {target}, k=3) == 0.0


def test_mrr_rewards_higher_rank():
    index = build_index()
    hits = ImageRetriever(index).search("高血压", k=3)
    assert mrr(hits, {hits[0].entry.asset_id}) == 1.0
    assert mrr(hits, {"not-there"}) == 0.0
    assert mrr([], {"x"}) == 0.0


# ───────────── 持久化 ─────────────

def test_save_and_load_roundtrip(tmp_path):
    index = build_index()
    path = tmp_path / "image-index.json"
    index.save(path)

    loaded = ImageIndex.load(path)
    assert len(loaded) == len(index)
    assert loaded.encoder.name == "byte-histogram"
    entry = loaded.all()[0]
    assert entry.vector and len(entry.vector) == 32
    # 载入后仍可检索
    assert ImageRetriever(loaded).search("资产负债表", k=1)[0].entry.kb == "finance"