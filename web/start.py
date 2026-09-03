"""轻量前端服务器 — 静态文件本地 serve，API 代理到后端。自动打开浏览器。"""

import http.server
import socketserver
import urllib.request
import json
import webbrowser
import os

WEB = os.path.dirname(os.path.abspath(__file__))
BACKEND = "http://127.0.0.1:8000"

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/") or self.path.startswith("/docs"):
            return self._proxy()
        # 静态文件
        path = self.path
        if path == "/": path = "/index.html"
        fp = os.path.join(WEB, path.lstrip("/"))
        if not os.path.isfile(fp):
            self.send_error(404)
            return
        ct = self.guess_type(fp)[0] or "application/octet-stream"
        data = open(fp, "rb").read()
        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if not (self.path.startswith("/api/") or self.path.startswith("/docs")):
            self.send_error(405); return
        clen = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(clen) if clen else None
        r = urllib.request.Request(f"{BACKEND}{self.path}", data=body, method="POST")
        for h in ("Authorization","X-API-Key","Content-Type"):
            if h in self.headers: r.add_header(h, self.headers[h])
        try:
            resp = urllib.request.urlopen(r, timeout=300)
            rb = resp.read()
            self.send_response(resp.status)
            for k,v in resp.headers.items():
                if k.lower() not in ("transfer-encoding","connection"):
                    self.send_header(k,v)
            self.end_headers()
            self.wfile.write(rb)
        except Exception as e:
            self.send_error(502, str(e))

    def do_OPTIONS(self):
        if self.path.startswith("/api/"): return self._proxy()
        self.send_response(200); self.end_headers()
    def _proxy(self): pass  # handled by caller
    def log_message(self,*a): pass

port = 9876
socketserver.TCPServer.allow_reuse_address = True
httpd = socketserver.TCPServer(("127.0.0.1", port), H)
webbrowser.open(f"http://127.0.0.1:{port}")
print(f"Frontend: http://127.0.0.1:{port}")
print(f"Backend: {BACKEND}")
httpd.serve_forever()
