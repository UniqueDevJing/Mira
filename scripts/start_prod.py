"""RAG 2.0 生产守护启动器(单实例 / 崩溃自愈 / 端口自愈 / 全链路可观测)。

- 登录后由计划任务(或手动双击)拉起, 全程无控制台窗口(pythonw)。
- 同时拉起 rag_api(:8000) 与命名隧道(rag.uniquejingclaudecoding.top)。
- 隧道立即启动, 不阻塞等待模型加载; API 就绪后写 ready 标志。
- 单实例锁: 已运行则直接退出, 避免重复拉起。
- 端口自愈: 若 8000 被残留进程占用则自动释放。
- 任一子进程崩溃自动重启(API 限次, 隧道不限)。
- 全链路日志: API 与隧道输出均写入 start_prod.log; 并周期性探测公网可达性。
"""
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

try:
    import ctypes
except Exception:  # 非 Windows 时降级(本脚本仅在 Windows 运行)
    ctypes = None

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.expanduser(r"C:\Users\Dominion\.cloudflared\rag-tunnel-token.txt")
LOG_FILE = os.path.join(SCRIPTS, "start_prod.log")
LOCK_FILE = os.path.join(SCRIPTS, "start_prod.lock")
READY_FILE = os.path.join(SCRIPTS, "start_prod.ready")

HEALTH = "http://127.0.0.1:8000/health"
PUBLIC_URL = "https://rag.uniquejingclaudecoding.top/"
PORT = 8000
API_RESTART_DELAY = 3
TUNNEL_RESTART_DELAY = 5
MAX_API_RESTARTS = 10
PROBE_EVERY = 5  # 每 N 轮探测一次公网可达性

_CLOUDFLARED = [
    r"C:\Program Files (x86)\cloudflared\cloudflared.exe",
    r"C:\Program Files\cloudflared\cloudflared.exe",
    r"C:\cloudflared\cloudflared.exe",
]

# 进程存活期间持有互斥体句柄, 保证 "Global\RAG2_Prod_Singleton" 对象一直存在
_MUTEX_HANDLE = None


class TeeWriter:
    """把输出同时写到日志文件(pythonw 下 stdout 被丢弃, 必须落盘)。"""

    def __init__(self, f):
        self.f = f

    def write(self, s):
        try:
            self.f.write(s)
            self.f.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.f.flush()
        except Exception:
            pass


def find_cloudflared() -> str | None:
    for p in _CLOUDFLARED:
        if os.path.exists(p):
            return p
    return shutil.which("cloudflared")


def api_env() -> dict:
    e = os.environ.copy()
    e.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    e.setdefault("HF_HUB_OFFLINE", "1")
    e.setdefault("TRANSFORMERS_OFFLINE", "1")
    e.setdefault("RAG_API_KEY_ENABLED", "true")
    api_key = os.environ.get("RAG_API_KEY")
    if not api_key:
        raise RuntimeError("RAG_API_KEY environment variable is required for production")
    e["RAG_API_KEY"] = api_key
    return e


def check_server() -> bool:
    try:
        urllib.request.urlopen(HEALTH, timeout=3)
        return True
    except (urllib.error.URLError, OSError):
        return False


def probe_public() -> tuple[bool, str]:
    """探测公网可达性(禁用代理, 排除系统代理干扰; 返回详情便于定位 400 vs 超时)。"""
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(PUBLIC_URL + "health", timeout=6) as resp:
            return resp.status == 200, f"HTTP {resp.status}"
    except urllib.error.HTTPError as e:
        return False, f"HTTP {e.code}"
    except (urllib.error.URLError, OSError) as e:
        return False, f"{type(e).__name__}: {e}"


def pid_alive(pid: int) -> bool:
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True,
        ).stdout
        return str(pid) in out
    except Exception:
        return False


def port_in_use() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def free_port() -> None:
    """释放占用 PORT 的残留进程(本机专用于该服务)。"""
    try:
        out = subprocess.run(["netstat", "-ano"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{PORT}" in line and "LISTENING" in line:
                pid = line.split()[-1]
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
    except Exception:
        pass


def write_lock() -> None:
    with open(LOCK_FILE, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def is_another_instance_running() -> bool:
    """全局命名互斥体保证跨进程单实例(防止 schtasks 重复触发拉起多个 cloudflared)。

    注意: 错误码必须用 ctypes.WinDLL(use_last_error=True) + ctypes.get_last_error()
    读取 —— 直接用 ctypes.GetLastError() 会在运行时抛 AttributeError, 互斥体失效。
    """
    global _MUTEX_HANDLE
    if ctypes is None:
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        _MUTEX_HANDLE = kernel32.CreateMutexW(None, 1, "Global\\RAG2_Prod_Singleton")
        if not _MUTEX_HANDLE:
            return False
        return ctypes.get_last_error() == 183  # ERROR_ALREADY_EXISTS
    except Exception:
        return False


def kill_stale_daemons() -> None:
    """异步杀掉所有残留的 start_prod.py 守护实例(排除自身), 不阻塞启动路径。

    防: 旧版本代码(无互斥体)的僵尸实例仍在抢 8000 端口 / 抢隧道 token。
    用 PowerShell -EncodedCommand 传 UTF-16LE base64, 避免命令行引号转义地狱。
    Popen 不等待: PowerShell 冷启动可达数秒, 同步等待会拖慢服务启动。
    """
    try:
        import base64

        ps = (
            "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe' or Name='python.exe'\" | "
            f"Where-Object {{ $_.CommandLine -like '*start_prod.py*' -and $_.ProcessId -ne {os.getpid()} }} | "
            "ForEach-Object { Stop-Process -Id $_.ProcessId -Force }"
        )
        enc = base64.b64encode(ps.encode("utf-16-le")).decode("ascii")
        subprocess.Popen(
            ["powershell", "-NoProfile", "-EncodedCommand", enc],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        pass


def acquire_lock() -> None:
    """接管模式: 杀掉旧守护实例(锁文件 PID 进程树 + 残留守护 + 残留隧道), 不退出。

    解决 schtasks /end 只终止调度引用、未杀子进程导致锁残留、
    新实例永远跳过的死局; 同时清理多余 cloudflared 实例避免抢同一 token。
    每步打日志, 便于在 start_prod.log 中定位卡点。
    """
    # 1) 锁文件记录的旧守护 PID 存活 -> 杀整棵进程树(uvicorn/cloudflared 子进程一并消失)
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE, "r", encoding="utf-8") as f:
                old = int(f.read().strip())
            if old and pid_alive(old):
                print(f"[接管] 终止旧实例 (PID {old}) 及其子进程...", flush=True)
                subprocess.run(["taskkill", "/PID", str(old), "/T", "/F"],
                               capture_output=True, text=True, timeout=15)
                time.sleep(2)
        except Exception:
            pass
    # 2) 兜底: 异步杀所有残留 start_prod.py 守护(含旧版无互斥体的僵尸), 不阻塞
    kill_stale_daemons()
    # 3) 清理可能残留的隧道进程(避免多实例抢同一 token 导致 400)
    subprocess.run(["taskkill", "/IM", "cloudflared.exe", "/F"],
                   capture_output=True, text=True, timeout=15)
    time.sleep(1)
    write_lock()


def tail_log(lines: int = 40) -> str:
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-lines:])
    except OSError:
        return "(无日志)"


def start_api(log_f) -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app",
         "--host", "0.0.0.0", "--port", str(PORT)],
        cwd=PROJECT, env=api_env(),
        stdout=log_f, stderr=subprocess.STDOUT,
    )


def start_tunnel(cf: str, token: str, log_f) -> subprocess.Popen:
    # 隧道输出必须落盘: pythonw 下 stdout 被丢弃, 否则连不上也看不到原因
    return subprocess.Popen(
        [cf, "tunnel", "--no-prechecks", "run",
         "--token", token, "--protocol", "http2"],
        stdout=log_f, stderr=subprocess.STDOUT,
    )


def free_port_if_stale() -> None:
    if port_in_use() and not check_server():
        free_port()
        time.sleep(2)


def cleanup(api_proc, tunnel_proc) -> None:
    for p in (api_proc, tunnel_proc):
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass
    for f in (LOCK_FILE, READY_FILE):
        try:
            os.remove(f)
        except OSError:
            pass


def main() -> None:
    log_f = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stdout = TeeWriter(log_f)
    sys.stderr = sys.stdout
    print(f"[启动] 日志文件: {LOG_FILE}", flush=True)

    # 全局单实例: 若已有实例持有互斥体则直接退出(避免多个 cloudflared 抢同一 token)
    if is_another_instance_running():
        print("[跳过] 已有实例运行中(互斥体占用), 退出。", flush=True)
        sys.exit(0)

    cf = find_cloudflared()
    if cf is None:
        print("[错误] 找不到 cloudflared, 请先安装。", file=sys.stderr)
        sys.exit(1)
    print(f"[预检] cloudflared: {cf}", flush=True)

    if not os.path.exists(TOKEN_FILE):
        print(f"[错误] 缺少隧道 token: {TOKEN_FILE}", file=sys.stderr)
        sys.exit(1)
    token = open(TOKEN_FILE, encoding="utf-8").read().strip()
    if not token:
        print("[错误] 隧道 token 为空", file=sys.stderr)
        sys.exit(1)
    print(f"[预检] 隧道 token 长度: {len(token)}", flush=True)

    acquire_lock()

    # 端口自愈: 占用且非本服务 -> 释放
    if port_in_use() and not check_server():
        print("[端口自愈] 8000 被残留进程占用, 正在释放...", flush=True)
        free_port()
        time.sleep(2)

    api_proc = None
    tunnel_proc = None
    try:
        # API 已在跑则复用, 否则拉起(后台, 不等就绪)
        if check_server():
            print("[复用] 本机 :8000 已在运行", flush=True)
        else:
            api_proc = start_api(log_f)

        # 隧道立即启动, 不阻塞模型加载
        tunnel_proc = start_tunnel(cf, token, log_f)
        print(f"[OK] 隧道已启动 -> {PUBLIC_URL}", flush=True)
        print("[提示] API 首次加载模型可能较慢, 稍候访问即可。", flush=True)

        api_restarts = 0
        ready = check_server()
        probe_cnt = 0
        while True:
            if api_proc is not None and api_proc.poll() is not None:
                api_restarts += 1
                if api_restarts > MAX_API_RESTARTS:
                    print(f"[错误] rag_api 连续崩溃 {MAX_API_RESTARTS} 次, 放弃。日志尾迹:", file=sys.stderr)
                    print(tail_log(), file=sys.stderr)
                    break
                print(f"[看门狗] rag_api 退出(第{api_restarts}次), 重启中...", flush=True)
                free_port_if_stale()
                api_proc = start_api(log_f)

            if tunnel_proc.poll() is not None:
                print("[看门狗] 隧道退出, 重启中...", flush=True)
                time.sleep(TUNNEL_RESTART_DELAY)
                tunnel_proc = start_tunnel(cf, token, log_f)
                print("[OK] 隧道已恢复", flush=True)

            if not ready and check_server():
                ready = True
                open(READY_FILE, "w", encoding="utf-8").close()
                print("[OK] rag_api 就绪", flush=True)

            probe_cnt += 1
            if probe_cnt % PROBE_EVERY == 0:
                ok, detail = probe_public()
                print(f"[探测] 公网 -> {'OK' if ok else 'FAIL'} ({detail})", flush=True)

            time.sleep(3)
    finally:
        cleanup(api_proc, tunnel_proc)
        try:
            log_f.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
