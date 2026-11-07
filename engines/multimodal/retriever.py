"""图文混合检索 + 召回评测（W3）。

检索策略（显式、可解释，不用黑盒）：
1. **字幕关键词命中**（中文按 2-gram 切分，避免依赖分词器）——这是当前唯一有语义成分的信号；
2. **描述子余弦相似**——只作同分时的次序细化；
3. 分数 = 0.7 * 关键词 + 0.3 * 余弦，最终按分数排序。

为什么不假装这是向量检索：编码器是非语义的（见 index.py 的说明）。**把口径写清楚，
比包装成"多模态语义检索"更可信**；接上 CLIP 后只需把权重调成 0.3/0.7 即可。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .index import ImageEntry, ImageIndex, cosine

KEYWORD_WEIGHT = 0.7
VECTOR_WEIGHT = 0.3
_CJK = re.compile(r"[\u4e00-\u9fa5]+")
_WORD = re.compile(r"[a-zA-Z0-9]+")


def tokenize(text: str) -> set[str]:
    """中文 2-gram + 英文/数字单词，够用且无第三方依赖。"""
    tokens: set[str] = set()
    for chunk in _CJK.findall(text or ""):
        if len(chunk) == 1:
            tokens.add(chunk)
        for i in range(len(chunk) - 1):
            tokens.add(chunk[i : i + 2])
    tokens.update(w.lower() for w in _WORD.findall(text or ""))
    return tokens


def keyword_score(query_tokens: set[str], caption: str) -> float:
    if not query_tokens:
        return 0.0
    caption_tokens = tokenize(caption)
    if not caption_tokens:
        return 0.0
    return len(query_tokens & caption_tokens) / len(query_tokens)


@dataclass
class ImageHit:
    entry: ImageEntry
    score: float
    keyword: float
    vector: float

    def to_json(self) -> dict:
        return {
            "asset_id": self.entry.asset_id,
            "caption": self.entry.caption,
            "kb": self.entry.kb,
            "score": round(self.score, 4),
            "keyword_score": round(self.keyword, 4),
            "vector_score": round(self.vector, 4),
            "source": self.entry.source,
            "page": self.entry.page,
            "width": self.entry.width,
            "height": self.entry.height,
            "type": "image",
        }


class ImageRetriever:
    def __init__(self, index: ImageIndex) -> None:
        self.index = index

    def search(self, query: str, k: int = 5, kb: str | None = None) -> list[ImageHit]:
        query_tokens = tokenize(query)
        # 文本落在与图片同一空间的方式：把 query 文本的哈希当作"伪图像向量"的分母，
        # 仅用于次序细化；不宣称语义相似。
        pseudo = _text_vector(query)
        hits: list[ImageHit] = []
        for entry in self.index.all():
            if kb is not None and entry.kb != kb:
                continue
            kw = keyword_score(query_tokens, entry.caption)
            vec = max(0.0, cosine(pseudo, entry.vector))
            score = KEYWORD_WEIGHT * kw + VECTOR_WEIGHT * vec
            if score <= 0:
                continue
            hits.append(ImageHit(entry, score, kw, vec))
        hits.sort(key=lambda h: (-h.score, h.entry.asset_id))
        return hits[: max(0, k)]


def _text_vector(text: str) -> list[float]:
    digest = hashlib.sha256((text or "").encode()).digest()
    from .index import DIM

    raw = [(digest[i % len(digest)] / 255.0) for i in range(DIM)]
    norm = sum(v * v for v in raw) ** 0.5 or 1.0
    return [v / norm for v in raw]


def recall_at_k(hits: list[ImageHit] | list[str], expected: set[str], k: int = 5) -> float:
    """expected 里有多少被前 k 命中（以 asset_id 比对）。空期望集返回 1.0。"""
    if not expected:
        return 1.0
    ids = [h if isinstance(h, str) else h.entry.asset_id for h in hits[:k]]
    return len(expected & set(ids)) / len(expected)


def mrr(hits: list[ImageHit], expected: set[str]) -> float:
    """第一个命中的倒数排名；未命中为 0。"""
    if not expected:
        return 1.0
    for rank, hit in enumerate(hits, start=1):
        if hit.entry.asset_id in expected:
            return 1.0 / rank
    return 0.0