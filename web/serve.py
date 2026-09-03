"""Local frontend dev server — serves web/ static + proxies /api/* to backend."""

import http.server
import socketserver
import urllib.request
import urllib.parse
import os
import sys
import json

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND = "http://127.0.0.1:8000"
PORT = 8080
HOST = "127.0.0.1"

# Headers to forward from client to backend
FORWARD_HEADERS = ("Authorization", "X-API-Key", "Content-Type", "Accept", "Origin", "Accept-Encoding")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if self.path.startswith(("/api/", "/docs", "/openapi.json")):
            return self._proxy("GET", parsed)

        # Serve static files from web/ directory
        if self.path.startswith("/web"):
            rel = self.path[5:]  # strip "/web" -> e.g. /common.js
        elif self.path == "/":
            rel = "/index.html"
        else:
            rel = self.path

        # Normalize to proper OS path
        rel = rel.replace("/", os.sep)
        filepath = os.path.normpath(os.path.join(WEB_DIR, rel.lstrip(os.sep)))

        if not os.path.isfile(filepath):
            self.send_error_response(404, "Not Found", {"Content-Type": "text/html; charset=utf-8"})
            return

        try:
            with open(filepath, "rb") as f:
                content = f.read()
        except Exception:
            self.send_error_response(500, "Internal Error")
            return

        ctype = self.guess_type(filepath)[0] or "application/octet-stream"
        length = len(content)
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if self.path.startswith(("/api/", "/docs")):
            return self._proxy("POST", parsed)
        self.send_error_response(405, "Method Not Allowed")

    def do_OPTIONS(self):
        parsed = urllib.parse.urlparse(self.path)
        if self.path.startswith(("/api/",)):
            return self._proxy("OPTIONS", parsed)
        self.send_response(200)
        self.send_header("Connection", "close")
        self.end_headers()

    def _proxy(self, method, parsed):
        backend_url = f"{BACKEND}{self.path}"
        headersToSend = {}
        for h in FORWARD_HEADERS:
            val = self.headers.get(h)
            if val:
                headersToSend[h] = val

        body = None
        if method == "POST":
            clen = int(self.headers.get("Content-Length", 0))
            if clen > 0:
                body = self.rfile.read(clen)

        req = urllib.request.Request(backend_url, data=body, method=method)
        for k, v in headersToSend.items():
            req.add_header(k, v)

        try:
            resp = urllib.request.urlopen(req, timeout=300)
            resp_body = resp.read()
            resp_headers = dict(resp.headers)

            self.send_response(resp.status)
            for k, v in resp_headers.items():
                kl = k.lower()
                if kl in ("transfer-encoding", "connection", "hop-by-hop"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(resp_body)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            err_headers = dict(e.headers)
            self.send_response(e.code)
            for k, v in err_headers.items():
                kl = k.lower()
                if kl in ("transfer-encoding", "connection", "hop-by-hop"):
                    continue
                self.send_header(k, v)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            self.send_error_response(502, f"Backend unreachable: {e}")

    def send_error_response(self, code, message, extra_headers=None):
        body = json.dumps({"detail": message}, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if extra_headers:
            for k, v in extra_headers.items():
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def guess_type(self, path):
        import mimetypes
        return mimetypes.guess_type(path)

    def log_message(self, fmt, *args):
        pass


socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer((HOST, PORT), Handler) as httpd:
    print(f"Frontend at http://{HOST}:{PORT}")
    print(f"Backend at http://127.0.0.1:8000")
    httpd.serve_forever()
