"""召回评测脚本的端到端测试（W3）。

脚本是给 CI/人工使用的门禁入口，如果它本身不可靠，指标就不可信：
这里直接跑 `main()`，断言落盘 JSON、退出码与冒烟模式行为。
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import struct
import sys
import zlib


def make_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    raw = b"".join(b"\x00" + bytes(rgb) * width for _ in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(
            ">I", zlib.crc32(tag + data) & 0xFFFFFFFF
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def load_script():
    root = pathlib.Path(__file__).resolve().parents[1]
    path = root / "scripts" / "eval_image_recall.py"
    sys.path.insert(0, str(root))
    spec = importlib.util.spec_from_file_location("eval_image_recall_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_index_file(tmp_path) -> pathlib.Path:
    from engines.multimodal import ImageIndex

    index = ImageIndex()
    index.add(
        make_png(8, 8, (200, 30, 30)),
        kb="policy",
        caption="高血压分级示意图",
        source="内科学 第9版",
        page=112,
    )
    index.add(
        make_png(8, 8, (30, 30, 200)),
        kb="policy",
        caption="退款流程图",
        source="售后政策.pdf",
        page=7,
    )
    path = tmp_path / "index.json"
    index.save(path)
    return path


def test_smoke_mode_writes_metric(tmp_path, monkeypatch, capsys):
    script = load_script()
    index_path = build_index_file(tmp_path)
    out = tmp_path / "image-recall.json"

    monkeypatch.setattr(
        sys, "argv",
        ["eval_image_recall.py", "--index", str(index_path), "--smoke", "--out", str(out)],
    )
    assert script.main() == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["tool"] == "eval_image_recall"
    assert payload["k"] == 5
    assert payload["mode"] == "smoke"
    assert payload["image_recall_at_5"] == 1.0, "字幕自问自答必须能命中自己"
    assert payload["cases"] == 2
    assert "recall@5=1.000" in capsys.readouterr().out


def test_eval_mode_with_explicit_cases(tmp_path, monkeypatch):
    script = load_script()
    index_path = build_index_file(tmp_path)
    from engines.multimodal import ImageIndex

    index = ImageIndex.load(index_path)
    target = next(e.asset_id for e in index.all() if "退款" in e.caption)

    cases = tmp_path / "cases.json"
    cases.write_text(
        json.dumps([{"query": "退款流程", "expected": [target], "kb": "policy"}], ensure_ascii=False),
        encoding="utf-8",
    )
    out = tmp_path / "recall.json"
    monkeypatch.setattr(
        sys, "argv",
        ["eval_image_recall.py", "--index", str(index_path), "--cases", str(cases), "--out", str(out)],
    )
    assert script.main() == 0

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["mode"] == "eval"
    assert payload["image_recall_at_5"] == 1.0


def test_missing_index_returns_2(tmp_path, monkeypatch):
    script = load_script()
    monkeypatch.setattr(
        sys, "argv",
        ["eval_image_recall.py", "--index", str(tmp_path / "nope.json"), "--smoke"],
    )
    assert script.main() == 2


def test_no_cases_and_no_smoke_returns_2(tmp_path, monkeypatch):
    script = load_script()
    index_path = build_index_file(tmp_path)
    monkeypatch.setattr(sys, "argv", ["eval_image_recall.py", "--index", str(index_path)])
    assert script.main() == 2