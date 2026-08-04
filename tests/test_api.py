"""API 集成测试 — 限流 / 认证 / CORS / 健康检查"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print('=' * 60)
print('API 集成测试')
print('=' * 60)

from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)

# ── 1. 健康检查 ──
print('\n[1/6] 健康检查...')
resp = client.get('/health')
print(f'  status={resp.status_code}, body={resp.json()}')
assert resp.status_code == 200
assert resp.json()['status'] == 'healthy'
assert resp.json()['version'] == '1.0.0'
print('  PASS')

# ── 2. 根路径 (Web UI) ──
print('\n[2/6] Web UI 根路径...')
resp = client.get('/')
print(f'  status={resp.status_code}, content_type={resp.headers.get("content-type", "")[:30]}')
assert resp.status_code == 200
print('  PASS')

# ── 3. 文档上传（无文件）──
print('\n[3/6] 文档上传 — 缺少文件...')
resp = client.post('/api/v1/documents/upload')
print(f'  status={resp.status_code}')
assert resp.status_code == 422, f'期望 422, 实际 {resp.status_code}'
print('  PASS: 正确拒绝缺少文件的请求')

# ── 4. 文档列表（空库）──
print('\n[4/6] 文档列表...')
resp = client.get('/api/v1/documents')
print(f'  status={resp.status_code}')
data = resp.json()
print(f'  items={len(data.get("items", []))}, total={data.get("total")}')
assert resp.status_code == 200
assert 'items' in data and 'total' in data
print('  PASS')

# ── 5. 文档状态（不存在的文档）──
print('\n[5/6] 文档状态 — 不存在...')
resp = client.get('/api/v1/documents/nonexistent/status')
print(f'  status={resp.status_code}')
data = resp.json()
print(f'  body={data}')
assert resp.status_code == 200
assert data['status'] == 'not_found'
print('  PASS')

# ── 6. CORS 头检查 ──
print('\n[6/6] CORS 头检查...')
resp = client.options('/health', headers={
    'Origin': 'http://localhost:3000',
    'Access-Control-Request-Method': 'GET',
})
print(f'  status={resp.status_code}')
cors_headers = {
    k: v for k, v in resp.headers.items()
    if k.lower().startswith('access-control')
}
for k, v in cors_headers.items():
    print(f'  {k}: {v}')
assert 'access-control-allow-origin' in (k.lower() for k in resp.headers.keys()), \
    '缺少 Access-Control-Allow-Origin 头'
print('  PASS')

# ── 7 (额外). API Key 认证 ──
print('\n[Bonus] API Key 认证...')
os.environ['RAG_API_KEY_ENABLED'] = 'true'
os.environ['RAG_API_KEY'] = 'test-secret-key'

# 不带 Key 应被拒绝
resp = client.post('/api/v1/qa/ask', json={'question': 'test'})
print(f'  无Key: status={resp.status_code}')
assert resp.status_code == 401, f'期望 401, 实际 {resp.status_code}'

# 错误 Key 应被拒绝
resp = client.post('/api/v1/qa/ask', json={'question': 'test'}, headers={'X-API-Key': 'wrong'})
print(f'  错误Key: status={resp.status_code}')
assert resp.status_code == 403, f'期望 403, 实际 {resp.status_code}'

# 正确 Key 应通过
resp = client.post('/api/v1/qa/ask', json={'question': 'test'}, headers={'X-API-Key': 'test-secret-key'})
print(f'  正确Key: status={resp.status_code}')
assert resp.status_code == 200, f'期望 200, 实际 {resp.status_code}'

# Bearer 方式
resp = client.post('/api/v1/qa/ask', json={'question': 'test'}, headers={'Authorization': 'Bearer test-secret-key'})
print(f'  Bearer: status={resp.status_code}')
assert resp.status_code == 200, f'期望 200, 实际 {resp.status_code}'

# 健康检查始终免认证
resp = client.get('/health')
print(f'  健康检查(无Key): status={resp.status_code}')
assert resp.status_code == 200

os.environ.pop('RAG_API_KEY_ENABLED', None)
os.environ.pop('RAG_API_KEY', None)
print('  PASS')

print('\n' + '=' * 60)
print('API 集成测试 — 全部通过')
print('=' * 60)
