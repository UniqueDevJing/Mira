"""P1-3 实体歧义型评测集构建器的单元测试 — 验证代码质量与产出可行性。

锁定不变量：
  - 纯函数正确性 (_jaccard / _entities / _pick_disambiguator)
  - 已产出数据集的 schema 与语义正确性
  - 小型端到端构建能产出合法 JSON 且满足同样不变量
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from engines.embedding.embedder import EmbeddingService
from engines.retrieval.bm25_index import Bm25Index
from engines.retrieval.fusion import rrf_fuse

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_ROOT, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import build_entity_ambig_set as bam

_EVAL_DIR = os.path.join(_ROOT, "data", "eval")
_DATASET = os.path.join(_EVAL_DIR, "entity_ambig_dataset.json")
_CORPUS = os.path.join(_EVAL_DIR, "corpus_chunks.json")


# ---------- 纯函数 ----------

def test_jaccard_pure():
    assert bam._jaccard({1, 2}, {2, 3}) == pytest.approx(1 / 3)
    assert bam._jaccard({1}, {1}) == 1.0
    assert bam._jaccard(set(), set()) == 1.0
    assert bam._jaccard({1, 2, 3}, set()) == 0.0


def test_entities_pure():
    ents = bam._entities("张伟在2023年参加了会议并获得了奖项")
    surf = {e for e, _ in ents}
    assert "张伟" in surf
    assert "2023年" in surf
    # 动词/普通名词不应进入专名集合（不被 nr/nt/ns/nz 或 year 命中）
    assert "参加" not in surf
    assert "获得" not in surf


# ---------- 已产出数据集不变量 ----------

def _load():
    with open(_DATASET, encoding="utf-8") as f:
        ds = json.load(f)
    with open(_CORPUS, encoding="utf-8") as f:
        ck = {c["chunk_id"]: c for c in json.load(f)}
    return ds, ck


def test_dataset_schema_and_semantics():
    ds, ck = _load()
    assert ds, "entity_ambig_dataset.json 不应为空"
    for it in ds:
        for key in ("id", "kb", "category", "question", "reference_answer",
                    "golden_doc_ids", "expected_chunk_ids", "entity",
                    "disambiguator", "competitor_chunk_ids",
                    "ca_cb_cosine", "ca_cb_ent_overlap",
                    "bm25_golden_rank", "hard", "hard_negatives"):
            assert key in it, f"缺字段 {key}: {it.get('id')}"
        # golden / competitor 必须真实存在于语料
        gid = it["expected_chunk_ids"][0]
        cid = it["competitor_chunk_ids"][0]
        assert gid in ck and cid in ck
        g, c = ck[gid], ck[cid]
        ent = it["entity"]
        # 二者都必须含有共享实体（否则不构成歧义）
        assert ent in g["content"] and ent in c["content"], it["id"]
        # competitor 不应含消歧短语（golden 独有，定义合法 golden）
        assert it["disambiguator"] in g["content"], it["id"]
        assert it["disambiguator"] not in c["content"], it["id"]
        # 不同文档（真正的多文档歧义）
        assert g["doc_id"] != c["doc_id"], it["id"]
        # 主题确实不同
        assert it["ca_cb_ent_overlap"] < 0.5, it["id"]
        # 参考答案非空且来自 golden chunk
        assert it["reference_answer"].strip() and it["reference_answer"] in g["content"], it["id"]


def test_dataset_competitor_in_pool():
    """自过滤保证：竞争者必须真实进入融合候选池（rerank 实际输入）。

    通过重跑融合检索验证 cb 在 top-15；抽样 20 条即可（全量太慢）。"""
    ds, ck = _load()

    chunks = list(ck.values())
    embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs_n = embs / np.clip(norms, 1e-9, None)
    bm = Bm25Index()
    bm.add_documents([{"id": c["chunk_id"], "chunk_id": c["chunk_id"],
                      "doc_id": c["doc_id"], "content": c["content"]} for c in chunks])
    emb = EmbeddingService()

    sample = ds[:20]
    bad = 0
    for it in sample:
        q = it["question"]
        q_emb = np.array(emb.embed_query(q), dtype=np.float32)
        sims = embs_n @ q_emb
        v_docs = []
        for idx in np.argsort(-sims)[:40]:
            c = chunks[int(idx)]
            v_docs.append({"chunk_id": c["chunk_id"], "doc_id": c["doc_id"],
                           "content": c["content"], "score": float(sims[idx])})
        b_docs = bm.search(q, 40)
        fused = rrf_fuse(v_docs, b_docs)
        pool = {d.get("chunk_id") for d in fused[:15]}
        if it["competitor_chunk_ids"][0] not in pool:
            bad += 1
    assert bad == 0, f"{bad}/20 样本的竞争者不在融合候选池 top-15"


# ---------- 端到端小构建 ----------

def test_builder_small_run(tmp_path):
    # 构建器从 eval-dir 读 corpus_chunks.json，需先把语料复制进临时目录
    shutil.copy(_CORPUS, tmp_path / "corpus_chunks.json")
    out = tmp_path / "entity_ambig_dataset.json"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run(
        [sys.executable, os.path.join(_SCRIPTS, "build_entity_ambig_set.py"),
         "--eval-dir", str(tmp_path), "--target", "12", "--max-pairs-per-entity", "2"],
        capture_output=True, text=True, timeout=240, env=env,
        cwd=_ROOT, check=False,
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert out.exists()
    with open(out, encoding="utf-8") as f:
        ds = json.load(f)
    assert len(ds) >= 1, "小构建应至少产出 1 条"
    # 复用不变量
    with open(os.path.join(str(tmp_path), "corpus_chunks.json"), encoding="utf-8") as f:
        corpus = json.load(f)
    ck = {c["chunk_id"]: c for c in corpus}
    for it in ds:
        g, c = ck[it["expected_chunk_ids"][0]], ck[it["competitor_chunk_ids"][0]]
        assert it["entity"] in g["content"] and it["entity"] in c["content"]
        assert it["disambiguator"] in g["content"] and it["disambiguator"] not in c["content"]
        assert g["doc_id"] != c["doc_id"]
