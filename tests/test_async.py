"""最小异步测试 — 模型预加载"""
import os, asyncio
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 在事件循环外加载模型
print('Loading...')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('Loaded')

async def test():
    loop = asyncio.get_running_loop()
    emb = await loop.run_in_executor(None, lambda: model.encode(['测试文本'], normalize_embeddings=True))
    print(f'dim={emb.shape[1]}')
    print('DONE')

asyncio.run(test())
