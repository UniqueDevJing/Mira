"""模型下载器 — 直连 HF 镜像, 绕开 huggingface_hub 缓存机制。

为什么不用 huggingface_hub:
  它在下载中断/失败时会删除 .incomplete 临时文件。在受限环境 (沙箱 / 回收站不可用)
  该删除操作被拦截并抛 OSError, 导致整个下载失败。本脚本用 requests 流式写盘,
  完成后 os.replace 原子改名, 全程不需要删除任何文件。

下载到 models/<repo_id 最后一段>/, sentence-transformers 可直接按本地路径加载。

用法:
    python scripts/download_model.py BAAI/bge-reranker-base
    python scripts/download_model.py BAAI/bge-reranker-base --mirror https://hf-mirror.com
    python scripts/download_model.py BAAI/bge-reranker-base --only-weights   # 只下权重

下载后把 config 指向本地路径即可, 例如:
    RAG_RERANKER_MODEL=./models/bge-reranker-base
"""

import argparse
import os
import sys
import time

import requests

MIRROR = "https://hf-mirror.com"

# sentence-transformers CrossEncoder 需要的最小文件集
CORE_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]
WEIGHT_FILES = ["model.safetensors", "pytorch_model.bin", "onnx/model.onnx"]


def head_size(url: str) -> int:
    """取远端文件大小; 失败返回 0 (不阻断下载)。"""
    try:
        r = requests.head(url, allow_redirects=True, timeout=20)
        return int(r.headers.get("Content-Length", 0) or 0)
    except Exception:  # noqa: BLE001 — 取不到大小不阻断, 仅影响进度显示
        return 0


def download(url: str, dest: str, label: str) -> bool:
    """流式下载到 dest.tmp 后原子改名。已存在且大小一致则跳过。"""
    tmp = dest + ".tmp"
    want = head_size(url)

    if os.path.exists(dest) and want and os.path.getsize(dest) == want:
        print(f"  ✓ {label} 已存在 ({want / 1024 / 1024:.1f} MB), 跳过")
        return True

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    t0 = time.time()
    got = 0
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            total = int(r.headers.get("Content-Length", 0) or 0) or want
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    got += len(chunk)
                    if total:
                        pct = got / total * 100
                        print(f"\r  ↓ {label} {got / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.0f}%)", end="", flush=True)
    except Exception as e:  # noqa: BLE001 — 下载失败是可预期的网络问题, 降级为返回 False
        print(f"\n  ✗ {label} 下载失败: {type(e).__name__}: {str(e)[:120]}")
        return False

    os.replace(tmp, dest)  # 原子改名, 不触发删除
    dt = time.time() - t0
    speed = got / 1024 / 1024 / dt if dt > 0 else 0
    print(f"\r  ✓ {label} 完成 {got / 1024 / 1024:.1f} MB  用时 {dt:.1f}s ({speed:.1f} MB/s)      ")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="从 HF 镜像下载模型到本地 models/ 目录")
    ap.add_argument("repo", help="仓库 ID, 如 BAAI/bge-reranker-base")
    ap.add_argument("--mirror", default=MIRROR, help=f"镜像地址 (默认 {MIRROR})")
    ap.add_argument("--dest", default="models", help="保存根目录 (默认 models/)")
    ap.add_argument("--only-weights", action="store_true", help="只下载权重文件")
    args = ap.parse_args()

    name = args.repo.split("/")[-1]
    outdir = os.path.join(args.dest, name)
    base = f"{args.mirror}/{args.repo}/resolve/main"

    print(f"仓库: {args.repo}")
    print(f"镜像: {args.mirror}")
    print(f"目标: {outdir}")
    print()

    files = (WEIGHT_FILES if args.only_weights else WEIGHT_FILES + CORE_FILES)
    ok, weight_done = 0, False

    for fname in files:
        url = f"{base}/{fname}"
        dest = os.path.join(outdir, fname)
        # 权重文件有多个候选, 成功下一个即可
        if fname in WEIGHT_FILES and weight_done:
            continue
        print(f"[{fname}]")
        if download(url, dest, fname):
            ok += 1
            if fname in WEIGHT_FILES:
                weight_done = True

    print()
    if not weight_done:
        print(f"✗ 未取得任何权重文件, 下载不完整: {outdir}")
        return 1
    print(f"✓ 完成 {ok} 个文件 -> {outdir}")
    print()
    print("使用方式: 把配置指向本地路径")
    print(f"  RAG_RERANKER_MODEL=./{outdir.replace(chr(92), '/')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
