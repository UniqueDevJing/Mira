"""启动 Cloudflare 临时隧道 — 把本机 8000 端口暴露到公网 URL。

用法: python scripts/tunnel.py
前置: RAG 服务已在本机 8000 运行 (IDEA rag_api 或 uvicorn)。
注意: 每次启动 URL 随机; API 已有 X-API-Key 鉴权保护, 勿公网扩散 Key。
"""

import re
import subprocess
import sys
import urllib.error
import urllib.request

CLOUDFLARED = r"C:\Program Files (x86)\cloudflared\cloudflared.exe"
TARGET = "http://127.0.0.1:8000"
HEALTH = "http://127.0.0.1:8000/health"
URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
URL_TIMEOUT_S = 30


def check_server() -> bool:
    try:
        urllib.request.urlopen(HEALTH, timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def main():
    if not check_server():
        print("[错误] 本机 8000 服务未启动, 请先启动 RAG 服务 (IDEA: rag_api)", file=sys.stderr)
        sys.exit(1)

    print("[cloudflared] 正在建立公网隧道, 等待 URL...", flush=True)
    proc = subprocess.Popen(
        [CLOUDFLARED, "tunnel", "--url", TARGET, "--protocol", "http2", "--no-autoupdate"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    url = None
    start = __import__("time").time()
    try:
        for line in proc.stdout:
            line = line.strip()
            m = URL_RE.search(line)
            if m and not url:
                url = m.group(0)
                print("=" * 62, flush=True)
                print(f"  公网地址: {url}", flush=True)
                print("  手机/他人浏览器打开, 右上角填访问 Key 后即可问答", flush=True)
                print("  关闭本窗口 = 断开隧道", flush=True)
                print("=" * 62, flush=True)
            if not url and (__import__("time").time() - start) > URL_TIMEOUT_S:
                print("[错误] 30s 内未获取到公网 URL, 检查网络或 cloudflared", file=sys.stderr)
                proc.terminate()
                sys.exit(1)
    except KeyboardInterrupt:
        print("\n[cloudflared] 手动中断, 关闭隧道")
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
