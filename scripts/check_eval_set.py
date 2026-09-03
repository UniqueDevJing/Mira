"""QA 评测集质量检查 — 确保样本格式完整、无重复、无空值。

作为评测运行前的 gates，避免脏数据导致基线失真。

用法:
  python scripts/check_eval_set.py tests/eval_dataset.json
  # 通过输出 exit code 0 / 1
"""
import argparse
import json
import sys
from difflib import SequenceMatcher


def check(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    ok = True
    issues: list[str] = []

    if not isinstance(data, list):
        print("FAIL: 文件必须是非空 JSON 数组")
        return False
    if len(data) == 0:
        print("FAIL: 评测集为空")
        return False

    required_keys = {"kb", "question", "reference_answer"}
    for i, item in enumerate(data):
        missing = required_keys - set(item.keys())
        if missing:
            issues.append(f"  第 {i+1} 行: 缺少字段 {missing}")
        q = str(item.get("question", "")).strip()
        a = str(item.get("reference_answer", "")).strip()
        if not q:
            issues.append(f"  第 {i+1} 行: question 为空")
        if not a:
            issues.append(f"  第 {i+1} 行: reference_answer 为空")

    # 重复问题检测 (相似度 > 0.9)
    questions = [str(x["question"]).strip() for x in data]
    for i in range(len(questions)):
        for j in range(i + 1, len(questions)):
            ratio = SequenceMatcher(None, questions[i], questions[j]).ratio()
            if ratio > 0.9 and ratio < 1.0:
                issues.append(f"  疑似重复 ({ratio:.2f}): '{questions[i][:40]}...' vs '{questions[j][:40]}...'")

    if issues:
        print("WARN:")
        for issue in issues:
            print(issue)
        return False

    cats: dict[str, int] = {}
    for x in data:
        k = x.get("kb", "?")
        cats[k] = cats.get(k, 0) + 1

    print(f"PASS: {len(data)} 条样本，{len(cats)} 个类别")
    for k, v in sorted(cats.items()):
        print(f"  [{k}] {v}")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="tests/eval_dataset.json")
    args = ap.parse_args()
    success = check(args.file)
    sys.exit(0 if success else 1)
