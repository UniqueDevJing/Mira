"""实体抽取引擎 — 规则 + LLM 混合抽取"""
import re, json
import httpx
from typing import List
from dataclasses import dataclass, field


@dataclass
class Entity:
    name: str
    type: str
    aliases: List[str] = field(default_factory=list)
    source_chunk_id: str = ""


@dataclass
class Relation:
    subject: str
    predicate: str
    object: str
    source_chunk_id: str = ""


ENTITY_PROMPT = """从文本中抽取所有实体，返回JSON数组。每个实体包含name(名称)、type(类型: Person/Organization/Product/Technology/Date/Location)、aliases(别名列表)。

文本：{text}

重要：只返回JSON数组，不要其他文字。示例格式：
[{"name":"张三","type":"Person","aliases":["张总"]},{"name":"Python","type":"Technology","aliases":[]}]"""

RELATION_PROMPT = """已知实体列表: {entities}

从文本中抽取实体间的关系，返回JSON数组。每个关系包含subject(主体)、predicate(关系类型)、object(客体)。

关系类型：uses(使用), supplies(供应), signs(签署), references(引用), contains(包含), depends_on(依赖), employs(雇佣), owns(拥有)

文本：{text}

重要：只返回JSON数组，不要其他文字。示例格式：
[{"subject":"系统","predicate":"uses","object":"Python"},{"subject":"FastAPI","predicate":"depends_on","object":"Uvicorn"}]"""


class EntityExtractor:
    def __init__(self, llm_url: str = "", llm_model: str = "", llm_key: str = ""):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_key = llm_key

    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        rules = self._rule_extract(text, chunk_id)
        llm_ents = []
        if self.llm_url and self.llm_key:
            try:
                llm_ents = self._llm_extract(text, chunk_id)
            except Exception:
                pass
        seen = {e.name.lower(): e for e in rules}
        for e in llm_ents:
            k = e.name.lower()
            if k in seen:
                seen[k].aliases.extend(e.aliases)
            else:
                seen[k] = e
        return list(seen.values())

    def _rule_extract(self, text: str, chunk_id: str) -> List[Entity]:
        ents = []
        techs = [
            r'\b(FastAPI|Flask|Django|Spring)\b',
            r'\b(Milvus|Pinecone|Chroma|Weaviate|LanceDB)\b',
            r'\b(PostgreSQL|MySQL|MongoDB|Redis|Neo4j)\b',
            r'\b(Docker|Kubernetes|K8s)\b',
            r'\b(Python|Java|Go|Rust|TypeScript)\b',
            r'\b(React|Vue|Next\.js|Nuxt)\b',
            r'\b(BGE|SentenceTransformer|PyTorch|TensorFlow)\b',
            r'\b(Llama|GPT|Claude|Qwen|DeepSeek)\b',
            r'\b(PaddleOCR|MinerU|Docling)\b',
            r'\b(RAG|LLM|GraphRAG|OCR)\b',
            r'\b(NATS|RabbitMQ|Kafka)\b',
            r'\b(TokenHub|MinIO)\b',
        ]
        for pat in techs:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ents.append(Entity(name=m.group(), type="Technology", source_chunk_id=chunk_id))
        return ents

    def _llm_extract(self, text: str, chunk_id: str) -> List[Entity]:
        resp = httpx.post(
            f"{self.llm_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.llm_key}"},
            json={"model": self.llm_model, "messages": [
                {"role": "user", "content": ENTITY_PROMPT.format(text=text[:3000])}
            ], "max_tokens": 800, "temperature": 0.1},
            timeout=30
        )
        data = resp.json()
        if isinstance(data, str): data = json.loads(data)
        content = data["choices"][0]["message"]["content"]
        m = re.search(r'\[.*\]', content, re.DOTALL)
        if m:
            items = json.loads(m.group())
            return [Entity(name=i["name"], type=i["type"],
                          aliases=i.get("aliases", []), source_chunk_id=chunk_id)
                    for i in items]
        return []


class RelationExtractor:
    def __init__(self, llm_url: str = "", llm_model: str = "", llm_key: str = ""):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_key = llm_key

    def extract(self, text: str, entities: List[Entity], chunk_id: str = "") -> List[Relation]:
        if not self.llm_url or not self.llm_key:
            return []
        names = ", ".join([e.name for e in entities[:20]])
        try:
            resp = httpx.post(
                f"{self.llm_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.llm_key}"},
                json={"model": self.llm_model, "messages": [
                    {"role": "user", "content": RELATION_PROMPT.format(
                        entities=names, text=text[:3000])}
                ], "max_tokens": 800, "temperature": 0.1},
                timeout=30
            )
            data = resp.json()
            if isinstance(data, str): data = json.loads(data)
            content = data["choices"][0]["message"]["content"]
            m = re.search(r'\[.*\]', content, re.DOTALL)
            if m:
                items = json.loads(m.group())
                return [Relation(subject=i["subject"], predicate=i["predicate"],
                                object=i["object"], source_chunk_id=chunk_id)
                        for i in items]
        except Exception:
            pass
        return []
