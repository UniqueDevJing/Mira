"""最小同步测试 - SentenceTransformer"""
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

print('Loading model...')
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('BAAI/bge-small-zh-v1.5')
print('Encoding...')
emb = model.encode(['测试文本'])
print(f'Shape: {emb.shape}')
print('DONE')
