"""实体抽取引擎 — 规则 + LLM 混合抽取"""
import re, json, logging
from typing import List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
[{{"name":"张三","type":"Person","aliases":["张总"]}},{{"name":"Python","type":"Technology","aliases":[]}}]"""

RELATION_PROMPT = """已知实体列表: {entities}

从文本中抽取实体间的关系，返回JSON数组。每个关系包含subject(主体)、predicate(关系类型)、object(客体)。

关系类型：uses(使用), supplies(供应), signs(签署), references(引用), contains(包含), depends_on(依赖), employs(雇佣), owns(拥有)

文本：{text}

重要：只返回JSON数组，不要其他文字。示例格式：
[{{"subject":"系统","predicate":"uses","object":"Python"}},{{"subject":"FastAPI","predicate":"depends_on","object":"Uvicorn"}}]"""


def _extract_json_array(content: str) -> list:
    """从 LLM 响应中稳健提取 JSON 数组，支持多种格式"""
    # 策略 1: 直接匹配 JSON 数组（支持嵌套）
    m = re.search(r'\[.*\]', content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass  # 尝试修复后重试

    # 策略 2: 尝试修复常见 JSON 错误
    # 去掉 markdown 代码块
    cleaned = re.sub(r'```(?:json)?\s*|```', '', content).strip()
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        json_str = m.group()
        # 修复尾部多余逗号
        json_str = re.sub(r',\s*]', ']', json_str)
        # 修复单引号
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        # 尝试替换单引号为双引号
        try:
            return json.loads(json_str.replace("'", '"'))
        except json.JSONDecodeError:
            pass

    # 策略 3: 逐行匹配 JSON 对象
    objs = re.findall(r'\{[^}]+\}', content)
    if objs:
        result = []
        for obj_str in objs:
            try:
                result.append(json.loads(obj_str))
            except json.JSONDecodeError:
                continue
        if result:
            return result

    logger.warning(f"无法从 LLM 响应中提取 JSON: {content[:300]}")
    return []


class EntityExtractor:
    def __init__(self, llm_url: str = "", llm_model: str = "", llm_key: str = ""):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_key = llm_key
        self._llm_available = bool(llm_url and llm_key)
        self._fail_count = 0
        self._max_fails = 3  # 连续失败 N 次后禁用 LLM
        self._llm_client = None

    def _get_llm_client(self):
        """获取同步 LLM 客户端"""
        if self._llm_client is None and self._llm_available:
            from api.core.llm_client import SyncLLMClient
            self._llm_client = SyncLLMClient(
                base_url=self.llm_url,
                model=self.llm_model,
                api_key=self.llm_key,
                max_retries=2,
                timeout=15.0,
                circuit_breaker_threshold=3,
                circuit_breaker_recovery_time=60.0,
            )
        return self._llm_client

    def extract(self, text: str, chunk_id: str = "") -> List[Entity]:
        rules = self._rule_extract(text, chunk_id)
        llm_ents = []
        if self._llm_available and self._fail_count < self._max_fails:
            try:
                llm_ents = self._llm_extract(text, chunk_id)
            except Exception as e:
                self._fail_count += 1
                if self._fail_count >= self._max_fails:
                    logger.info("LLM 实体抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
                logger.debug("LLM 实体抽取失败: %s", str(e)[:100])
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
        client = self._get_llm_client()
        if not client:
            return []

        try:
            from api.core.llm_client import CircuitBreakerOpenError
            response = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个JSON提取工具，只输出JSON数组，不输出其他任何文字。"},
                    {"role": "user", "content": ENTITY_PROMPT.format(text=text[:3000])}
                ],
                max_tokens=800,
                temperature=0.1,
            )
            content = response.content
            # DeepSeek reasoning_content 不是有效 JSON，跳过
            if not content or not content.strip():
                return []
            items = _extract_json_array(content)
            if items:
                return [Entity(name=i["name"], type=i.get("type", "Unknown"),
                              aliases=i.get("aliases", []), source_chunk_id=chunk_id)
                        for i in items if "name" in i]
        except CircuitBreakerOpenError:
            logger.debug("LLM 熔断中，跳过实体抽取")
        except Exception as e:
            logger.debug("LLM 实体抽取跳过: %s", type(e).__name__)
        return []


class RelationExtractor:
    def __init__(self, llm_url: str = "", llm_model: str = "", llm_key: str = ""):
        self.llm_url = llm_url
        self.llm_model = llm_model
        self.llm_key = llm_key
        self._llm_available = bool(llm_url and llm_key)
        self._fail_count = 0
        self._max_fails = 3
        self._llm_client = None

    def _get_llm_client(self):
        """获取同步 LLM 客户端"""
        if self._llm_client is None and self._llm_available:
            from api.core.llm_client import SyncLLMClient
            self._llm_client = SyncLLMClient(
                base_url=self.llm_url,
                model=self.llm_model,
                api_key=self.llm_key,
                max_retries=2,
                timeout=15.0,
                circuit_breaker_threshold=3,
                circuit_breaker_recovery_time=60.0,
            )
        return self._llm_client

    def extract(self, text: str, entities: List[Entity], chunk_id: str = "") -> List[Relation]:
        if not self._llm_available or self._fail_count >= self._max_fails:
            return []
        if not entities:
            return []

        client = self._get_llm_client()
        if not client:
            return []

        names = ", ".join([e.name for e in entities[:20]])
        try:
            from api.core.llm_client import CircuitBreakerOpenError
            response = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个JSON提取工具，只输出JSON数组。"},
                    {"role": "user", "content": RELATION_PROMPT.format(
                        entities=names, text=text[:3000])}
                ],
                max_tokens=800,
                temperature=0.1,
            )
            content = response.content
            if not content or not content.strip():
                return []
            items = _extract_json_array(content)
            if items:
                return [Relation(subject=i.get("subject", i.get("s", "")),
                                predicate=i.get("predicate", i.get("p", "")),
                                object=i.get("object", i.get("o", "")),
                                source_chunk_id=chunk_id)
                        for i in items if "subject" in i or "s" in i]
        except CircuitBreakerOpenError:
            logger.debug("LLM 熔断中，跳过关系抽取")
        except Exception as e:
            self._fail_count += 1
            if self._fail_count >= self._max_fails:
                logger.info("LLM 关系抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
            logger.debug("LLM 关系抽取失败: %s", str(e)[:100])
        return []
