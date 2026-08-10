"""最小测试"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

async def test():
    print('1. Embedding...')
    from engines.embedding.embedder import EmbeddingService
    svc = EmbeddingService()
    emb = svc.embed_query('test')
    print(f'   dim={len(emb)}')

    print('2. PDF Parse...')
    from engines.parsing.pdf_parser import PDFParser
    uir = PDFParser().parse('tests/fixtures/sample.pdf')
    print(f'   pages={len(uir.pages)}')

    print('3. Chunking...')
    from engines.chunking.structure_chunker import StructureChunker
    chunks = StructureChunker().chunk(uir)
    print(f'   chunks={len(chunks)}')

    print('DONE')
asyncio.run(test())
