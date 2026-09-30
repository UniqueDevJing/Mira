#!/usr/bin/env bash
# ============================================================================
# Mira 金融校验演示服务 · 服务器一键安装/升级（root 执行）
#
# 与跨境项目同机部署：第二个 systemd 服务（127.0.0.1:8001），
# nginx 以 /mira/ 路径前缀反代 —— 一个 IP 承载两个项目。
#
# 用法（服务器上）：
#   bash demo/server_setup.sh /tmp/mira_demo.tgz
#
# 部署后访问：
#   http://<服务器IP>/mira/
# ============================================================================
set -euo pipefail

APP_DIR=/opt/mira-demo
SERVICE_USER=www-data
NGINX_CONF=${NGINX_CONF:-/etc/nginx/sites-enabled/crossborder}
PKG="${1:-/tmp/mira_demo.tgz}"

[ -f "$PKG" ] || { echo "✗ 找不到安装包：$PKG"; exit 1; }

echo "==> [1/6] 解压到 $APP_DIR"
mkdir -p "$APP_DIR"
tar xzf "$PKG" -C "$APP_DIR"

echo "==> [2/6] venv + 依赖（仅 fastapi/uvicorn，engines/finance 纯标准库）"
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install -q --upgrade pip
./venv/bin/pip install -q fastapi "uvicorn[standard]" "mcp>=1.9,<2" "mcp>=1.9,<2"

echo "==> [3/6] 修正属主（root 解压后 www-data 会写不进去）"
chown -R "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"

echo "==> [4/6] systemd 服务（127.0.0.1:8001，不直接暴露公网）"
sed "s#/opt/mira-demo#$APP_DIR#g" demo/finance-demo.service \
  > /etc/systemd/system/finance-demo.service
systemctl daemon-reload
systemctl enable --now finance-demo
systemctl restart finance-demo   # 升级场景：已在运行也要重启加载新代码
sleep 2
if curl -fsS http://127.0.0.1:8001/api/meta >/dev/null 2>&1; then
  echo "    ✓ 服务已起，/api/meta 正常"
else
  echo "    ⚠ 健康检查未通过，查看日志： journalctl -u finance-demo -n 30"
  exit 1
fi

echo "==> [5/6] nginx 追加 /mira/ 反代（幂等）"
if [ ! -f "$NGINX_CONF" ]; then
  echo "    ⚠ 未找到 $NGINX_CONF，请手动把 demo/nginx-mira.conf 的内容并入你的 server 块后 reload nginx"
else
  python3 - "$NGINX_CONF" <<'PY'
import sys
conf = sys.argv[1]
s = open(conf, encoding="utf-8").read()
if "/mira/" in s:
    print("    已存在 /mira/ 配置，跳过")
else:
    block = ("    location /mira/ {\n"
             "        proxy_pass http://127.0.0.1:8001/;\n"
             "        proxy_set_header Host $host;\n"
             "        proxy_set_header X-Real-IP $remote_addr;\n"
             "    }\n\n")
    anchor = "    location / {"
    if anchor not in s:
        print("    ✗ 未找到 location / 锚点，请手动并入 demo/nginx-mira.conf"); sys.exit(1)
    s = s.replace(anchor, block + anchor, 1)
    open(conf, "w", encoding="utf-8").write(s)
    print("    已插入 /mira/ 反代")
PY
  nginx -t
  systemctl reload nginx
  echo "    ✓ nginx 已加载"
fi

echo "==> [6/6] 自测（走 nginx 全链路）"
printf "  %-28s %s\n" "直连 8001 /api/meta" \
  "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1:8001/api/meta)"
printf "  %-28s %s\n" "nginx /mira/api/meta" \
  "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 http://127.0.0.1/mira/api/meta)"
printf "  %-28s " "nginx /mira/ 页面"
PAGE=$(curl -sS -o /tmp/_mira_page.html -w '%{http_code}' --max-time 10 http://127.0.0.1/mira/)
echo "$PAGE"
grep -q "Mira 金融校验" /tmp/_mira_page.html && echo "  ✓ 页面内容正确"

echo ""
echo "==================== 部署完成 ===================="
echo "  演示页： http://$(curl -s ifconfig.me 2>/dev/null || echo '<服务器IP>')/mira/"
echo "  服务日志： journalctl -u finance-demo -f"
echo "  常用命令： systemctl restart finance-demo"
echo "==================================================="
