"""测试 - 纯同步不涉及asyncio"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print('Import...')
from engines.embedding.embedder import EmbeddingService

print('Init...')
svc = EmbeddingService()

print('Embed...')
emb = svc.embed_query('test')
print(f'dim={len(emb)}')

print('DONE')
