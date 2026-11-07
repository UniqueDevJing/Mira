"""多模态检索基座（W3）。

**这一层的诚实定位**：编码器是可插拔的，默认实现是**非语义的字节描述子**，
只用于打通「入库 → 去重 → 检索 → 评测」的链路，不承诺语义匹配质量。
生产环境把 `ImageEncoder` 换成 CLIP/SigLIP 即可，其余代码不动。

为什么先做链路而不是先接模型：
- 语义模型需要权重与 GPU，CI 跑不动，评测也无法复现；
- 而**溯源（provenance）、去重、召回指标、门禁**这些工程约束与模型无关，
  先立起来，换模型时才不会推翻重来。
"""

from __future__ import annotations

import hashlib
import json
import math
import pathlib
import struct
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from typing import Protocol

DIM = 32


class ImageEncoder(Protocol):
    def encode(self, image_bytes: bytes) -> list[float]: ...


class ByteHistogramEncoder:
    """确定性字节描述子：32 维、L2 归一化。

    维度构成：16 个字节直方图桶 + 3 个通道均值近似 + 宽高比 + 尺寸对数 + 4 个熵/游程特征。
    它能区分「结构不同的图」，但**不理解画面内容**——这正是它只作占位的理由。
    """

    name = "byte-histogram"

    def encode(self, image_bytes: bytes) -> list[float]:
        if not image_bytes:
            raise ValueError("empty image")
        digest = hashlib.sha256(image_bytes).digest()
        hist = [0] * 16
        for byte in image_bytes[:65536]:
            hist[byte >> 4] += 1
        total = sum(hist) or 1
        hist = [h / total for h in hist]

        width, height = png_size(image_bytes)
        aspect = (width / height) if height else 1.0
        size_feature = math.log10(len(image_bytes) + 10) / 6.0
        entropy = -sum(p * math.log2(p) for p in hist if p > 0) / 4.0
        ch_mean = [b / 255.0 for b in digest[:3]]
        tail = [digest[3] / 255.0, digest[4] / 255.0, aspect / 2.0, min(1.0, size_feature)]

        vector = hist + ch_mean + tail + [entropy]
        return _l2_normalize(vector[:DIM] + [0.0] * max(0, DIM - len(vector)))


def png_size(data: bytes) -> tuple[int, int]:
    """从 PNG 头读宽高；非 PNG 返回 (0,0)（不抛，避免一张坏图毁掉整批入库）。"""
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or len(data) < 24:
        return 0, 0
    return struct.unpack(">II", data[16:24])


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


@dataclass
class ImageEntry:
    asset_id: str
    path: str
    kb: str
    caption: str
    width: int
    height: int
    timeout_s: float = 0.0
    vector: list[float] = field(default_factory=list)
    # 溯源是硬要求：没有来源的图不允许进索引
    source: str = ""
    page: int = 0
    fetched_at: str = ""

    def to_json(self) -> dict:
        return asdict(self)


class MissingProvenance(ValueError):
    """图片缺少溯源信息——不允许入库。"""


class ImageIndex:
    """JSON 持久化的小型图片索引：内容寻址 + 余弦检索。

    为什么不用 LanceDB：这一层要能在 CI 里零依赖跑通；接口保持窄（add/search/save/load），
    将来换向量库只替换本类实现，retriever 与评测脚本不受影响。
    """

    def __init__(self, encoder: ImageEncoder | None = None, path: str | pathlib.Path | None = None) -> None:
        self.encoder = encoder or ByteHistogramEncoder()
        self.path = pathlib.Path(path) if path else None
        self._entries: dict[str, ImageEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    def add(
        self,
        image_bytes: bytes,
        *,
        kb: str,
        caption: str = "",
        source: str,
        page: int = 0,
        fetched_at: str = "",
        path_hint: str = "",
    ) -> tuple[ImageEntry, bool]:
        """返回 (条目, 是否新增)。同一内容重复入库 → 复用已有条目（内容寻址去重）。"""
        if not source:
            raise MissingProvenance("图片必须带 source（不能有来路不明的图）")
        asset_id = hashlib.sha256(image_bytes).hexdigest()
        if asset_id in self._entries:
            return self._entries[asset_id], False

        width, height = png_size(image_bytes)
        entry = ImageEntry(
            asset_id=asset_id,
            path=path_hint,
            kb=kb,
            caption=caption,
            width=width,
            height=height,
            vector=self.encoder.encode(image_bytes),
            source=source,
            page=page,
            fetched_at=fetched_at,
        )
        self._entries[asset_id] = entry
        return entry, True

    def get(self, asset_id: str) -> ImageEntry | None:
        return self._entries.get(asset_id)

    def all(self) -> list[ImageEntry]:
        return list(self._entries.values())

    def search(self, vector: list[float], k: int = 5, kb: str | None = None) -> list[tuple[ImageEntry, float]]:
        candidates: Iterable[ImageEntry] = self._entries.values()
        scored = [
            (entry, cosine(vector, entry.vector))
            for entry in candidates
            if kb is None or entry.kb == kb
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(0, k)]

    def search_similar(self, image_bytes: bytes, k: int = 5) -> list[tuple[ImageEntry, float]]:
        return self.search(self.encoder.encode(image_bytes), k=k)

    def save(self, path: str | pathlib.Path | None = None) -> str:
        target = pathlib.Path(path) if path else self.path
        if target is None:
            raise ValueError("没有指定保存路径")
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "encoder": getattr(self.encoder, "name", "unknown"),
            "entries": [e.to_json() for e in self._entries.values()],
        }
        target.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return str(target)

    @classmethod
    def load(cls, path: str | pathlib.Path, encoder: ImageEncoder | None = None) -> ImageIndex:
        data = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        index = cls(encoder=encoder, path=path)
        for raw in data.get("entries", []):
            entry = ImageEntry(**raw)
            index._entries[entry.asset_id] = entry
        return index