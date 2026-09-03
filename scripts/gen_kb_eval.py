"""生成 KB 匹配的评测集（P0' 质量地基）。

从 lancedb 各业务表读 chunk，调用 qwen-plus 为每个 chunk 生成
(question, reference_answer)，并记录 expected_chunk_ids 用于召回计算。
按 KB 分层抽样，输出 tests/eval_dataset_kb.json。

用法:
  python scripts/gen_kb_eval.py                 # 全量生成
  python scripts/gen_kb_eval.py --smoke         # 每库 1 条, 验证连通与解析
  python scripts/gen_kb_eval.py --max-per-kb 20 # 每库上限 20 条
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import httpx
import lancedb


def _load_env():
    """尽量让脚本独立读到 .env（不依赖 import config）。"""
    if os.getenv("RAG_LLM_API_KEY"):
        return
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(p):
        return
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _openai_client():
    key = os.getenv("RAG_LLM_API_KEY")
    base = os.getenv("RAG_LLM_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    model = os.getenv("RAG_LLM_MODEL", "qwen-plus")
    if not key:
        raise SystemExit("缺少 RAG_LLM_API_KEY，无法调用 LLM")
    return LLMClient(key=key, base=base, model=model)


class LLMClient:
    def __init__(self, key: str, base: str, model: str):
        self._key = key
        self._base = base
        self.model = model

    def generate_qa(self, doc_title: str, content: str) -> dict | None:
        prompt = (
            "你是一个客服知识库评测集生成器。给定一段企业知识库文本，请生成 1 个该文本"
            "能回答的自然用户问题，以及一段仅基于该文本、简洁准确的参考答案。\n"
            "要求:\n"
            "- 问题要像真实客户会问的口语化问法，不要照抄原文标题。\n"
            "- 参考答案必须只来自给定文本，不得编造。\n"
            "- 只输出 JSON: {\"question\": \"...\", \"reference_answer\": \"...\"}\n"
            f"文本标题: {doc_title}\n文本内容: {content}\n"
        )
        try:
            resp = httpx.post(
                f"{self._base}/chat/completions",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.3,
                },
                headers={"Authorization": f"Bearer {self._key}"},
                timeout=60,
                trust_env=False,
            )
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, str):
                data = json.loads(data)
            content = data["choices"][0]["message"].get("content") or ""
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] LLM 调用失败: {e}", file=sys.stderr)
            return None
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _chunks(table_name: str, db_path: str):
    db = lancedb.connect(db_path)
    t = db.open_table(table_name)
    for row in t.search().limit(10000).to_list():
        yield row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="每库仅 1 条, 验证用")
    ap.add_argument("--max-per-kb", type=int, default=0, help="每库上限条数, 0=全部")
    ap.add_argument("--out", default="tests/eval_dataset_kb.json")
    args = ap.parse_args()

    _load_env()
    client = _openai_client()

    kb_tables = {
        "policy": "rag_policy",
        "service": "rag_service",
        "tech": "rag_tech",
    }
    db_path = "lancedb_data"
    out = []
    per_kb = 1 if args.smoke else args.max_per_kb

    for kb, tbl in kb_tables.items():
        cnt = 0
        for row in _chunks(tbl, db_path):
            if per_kb and cnt >= per_kb:
                break
            content = (row.get("content") or "").strip()
            if len(content) < 20:
                continue
            doc_title = row.get("doc_title") or row.get("doc_id") or ""
            qa = client.generate_qa(doc_title, content[:1500])
            if not qa or not qa.get("question"):
                continue
            out.append({
                "kb": kb,
                "question": qa["question"],
                "reference_answer": qa.get("reference_answer", ""),
                "expected_chunk_ids": [row.get("id")],
            })
            cnt += 1
            print(f"  [{kb}] 已生成 {cnt} 条: {qa['question'][:40]}")
        print(f"=> {kb}: 共 {cnt} 条")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n写出 {len(out)} 条 -> {args.out}")


if __name__ == "__main__":
    main()
