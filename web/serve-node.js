const http = require('http')
const fs = require('fs')
const path = require('path')
const { URL } = require('url')

const WEB = __dirname
const BACKEND = 'http://127.0.0.1:8000'
const PORT = 3100

const TYPES = {
  '.html': 'text/html', '.js': 'application/javascript',
  '.css': 'text/css', '.json': 'application/json',
  '.png': 'image/png', '.jpg': 'image/jpeg',
  '.svg': 'image/svg+xml', '.ico': 'image/x-icon',
}

function serveFile(res, filepath) {
  const ext = path.extname(filepath).toLowerCase()
  const type = TYPES[ext] || 'application/octet-stream'
  try {
    const data = fs.readFileSync(filepath)
    res.writeHead(200, { 'Content-Type': type, 'Content-Length': data.length })
    res.end(data)
  } catch (e) {
    res.writeHead(404); res.end('Not found')
  }
}

function proxy(req, res, urlStr) {
  const u = new URL(urlStr, BACKEND)
  const opts = {
    hostname: '127.0.0.1', port: 8000,
    path: u.pathname + u.search,
    method: req.method,
    headers: {}, timeout: 300000,
  }
  for (const k of ['Authorization','X-API-Key','Content-Type','Accept']) {
    if (req.headers[k.toLowerCase()]) opts.headers[k] = req.headers[k.toLowerCase()]
  }
  const p = http.request(opts, (pResp) => {
    res.writeHead(pResp.statusCode, pResp.headers)
    pResp.pipe(res)
  })
  p.on('error', (e) => { res.writeHead(502); res.end(e.message) })
  req.pipe(p)
}

const server = http.createServer((req, res) => {
  const u = new URL(req.url, `http://localhost:${PORT}`)

  // API / docs proxy
  if (u.pathname.startsWith('/api/') || u.pathname.startsWith('/docs')) {
    return proxy(req, res, u.pathname + u.search)
  }

  // Static files - map /web/x -> web/x, root -> index.html
  let filePath
  if (u.pathname.startsWith('/web')) {
    filePath = path.join(WEB, u.pathname.slice(5))
  } else if (u.pathname === '/' || !u.pathname.includes('.')) {
    filePath = path.join(WEB, 'index.html')
  } else {
    filePath = path.join(WEB, u.pathname)
  }
  filePath = path.normalize(filePath)

  // Security: don't escape WEB dir
  if (!filePath.startsWith(WEB)) {
    res.writeHead(403); res.end('Forbidden')
    return
  }

  serveFile(res, filePath)
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`Frontend: http://127.0.0.1:${PORT}`)
  console.log(`Backend: ${BACKEND}`)
  require('child_process').exec(`start "" "http://127.0.0.1:${PORT}"`)
})
