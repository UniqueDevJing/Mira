"""QA 接口测试 — 正常/空库/非法输入"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'

print('=' * 60)
print('QA 接口测试')
print('=' * 60)

from fastapi.testclient import TestClient

# 设置 API Key（如果有的话）
os.environ.pop('RAG_API_KEY_ENABLED', None)

from api.main import app
client = TestClient(app)

# ── 1. 空库提问 ──
print('\n[1/3] 空知识库提问...')
resp = client.post('/api/v1/qa/ask', json={
    'question': '系统用了哪些技术？',
    'mode': 'hybrid',
    'top_k': 3,
})
print(f'  status={resp.status_code}')
data = resp.json()
print(f'  answer={data.get("answer", "")[:120]}')
assert resp.status_code == 200, f'期望 200, 实际 {resp.status_code}'
assert data.get('answer'), 'answer 不应为空'
print('  PASS')

# ── 2. Pydantic 校验（非法输入）──
print('\n[2/3] 输入校验...')

# 空问题
resp = client.post('/api/v1/qa/ask', json={'question': ''})
print(f'  空问题: status={resp.status_code}')
assert resp.status_code == 422, f'期望 422, 实际 {resp.status_code}'

# 问题过长 (>2000 chars)
resp = client.post('/api/v1/qa/ask', json={'question': 'x' * 2001})
print(f'  过长问题: status={resp.status_code}')
assert resp.status_code == 422, f'期望 422, 实际 {resp.status_code}'

# 非法 mode
resp = client.post('/api/v1/qa/ask', json={'question': 'test', 'mode': 'invalid'})
print(f'  非法mode: status={resp.status_code}')
assert resp.status_code == 422, f'期望 422, 实际 {resp.status_code}'

# top_k 超出范围
resp = client.post('/api/v1/qa/ask', json={'question': 'test', 'top_k': 100})
print(f'  top_k=100: status={resp.status_code}')
assert resp.status_code == 422, f'期望 422, 实际 {resp.status_code}'

print('  PASS')

# ── 3. 响应结构验证 ──
print('\n[3/3] 响应结构验证...')
resp = client.post('/api/v1/qa/ask', json={
    'question': '测试问题',
    'mode': 'hybrid',
    'enable_self_retrieval': False,
    'top_k': 5,
})
print(f'  status={resp.status_code}')
data = resp.json()
required_fields = ['answer', 'sources', 'graph_context', 'retrieval_rounds', 'latency_ms']
for field in required_fields:
    assert field in data, f'缺少响应字段: {field}'
    print(f'  {field}: OK ({"..." if field == "sources" else data[field] if not isinstance(data[field], (list, dict)) else f"len={len(data[field])}"})')
print('  PASS')

print('\n' + '=' * 60)
print('QA 接口测试 — 全部通过')
print('=' * 60)
