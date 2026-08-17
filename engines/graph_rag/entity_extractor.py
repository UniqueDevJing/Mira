"""实体抽取引擎 — 规则 + LLM 混合抽取"""

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    name: str
    type: str
    aliases: list[str] = field(default_factory=list)
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


BATCH_ENTITY_PROMPT = """从多段文本中抽取所有实体。每段用 [c0] 文本 [/c0]、[c1] 文本 [/c1] ... 标记。
返回 JSON 对象: 键为段落编号(c0/c1/...), 值为该段实体数组。
每个实体: {{"name":名称,"type":Person/Organization/Product/Technology/Date/Location,"aliases":[]}}
只输出 JSON 对象，不要其他文字。

文本：
{texts}"""

BATCH_RELATION_PROMPT = """对多段文本抽取实体间关系。每段用 [c0] 已知实体 + 文本 [/c0] 标记。
返回 JSON 对象: 键为段落编号(c0/c1/...), 值为该段关系数组。
每个关系: {{"subject":主体,"predicate":关系类型,"object":客体}}
关系类型：uses(使用), supplies(供应), signs(签署), references(引用), contains(包含), depends_on(依赖), employs(雇佣), owns(拥有)
只输出 JSON 对象，不要其他文字。

{texts}"""


def _extract_json_object(content: str) -> dict:
    """从 LLM 响应中稳健提取 JSON 对象 (批量抽取用), 支持修复常见错误。"""
    cleaned = re.sub(r"```(?:json)?\s*|```", "", content or "").strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return {}
    obj_str = m.group()
    for candidate in (obj_str, re.sub(r",\s*}", "}", obj_str), obj_str.replace("'", '"')):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    logger.warning("无法从 LLM 响应中提取 JSON 对象: %s", cleaned[:300])
    return {}


def _extract_json_array(content: str) -> list:
    """从 LLM 响应中稳健提取 JSON 数组，支持多种格式"""
    # 策略 1: 直接匹配 JSON 数组（支持嵌套）
    m = re.search(r"\[.*\]", content, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass  # 尝试修复后重试

    # 策略 2: 尝试修复常见 JSON 错误
    # 去掉 markdown 代码块
    cleaned = re.sub(r"```(?:json)?\s*|```", "", content).strip()
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        json_str = m.group()
        # 修复尾部多余逗号
        json_str = re.sub(r",\s*]", "]", json_str)
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
    objs = re.findall(r"\{[^}]+\}", content)
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


class _BaseLLMExtractor:
    """共享 LLM 抽取底座 — 客户端构造 + 熔断降级计数。

    Entity/Relation 抽取器此前逐字重复实现 __init__/_get_llm_client (50 行), 提取为基类。
    客户端由全局单例持有 (state._shared_extractors), 生命周期与应用一致。
    """

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
            from engines.common.llm_client import SyncLLMClient

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

    def close(self):
        """释放 LLM 客户端连接 (进程关闭时调用; 全局单例场景生命周期与应用一致)。"""
        if self._llm_client is not None:
            self._llm_client.close()
            self._llm_client = None


class EntityExtractor(_BaseLLMExtractor):
    def extract_rules(self, text: str, chunk_id: str = "") -> list[Entity]:
        """纯规则实体抽取 — 查询侧用 (检索问题不求 LLM, 避免每个 QA 一次 LLM 往返)。"""
        return self._rule_extract(text, chunk_id)

    def extract(self, text: str, chunk_id: str = "") -> list[Entity]:
        rules = self._rule_extract(text, chunk_id)
        llm_ents = []
        if self._llm_available and self._fail_count < self._max_fails:
            try:
                llm_ents = self._llm_extract(text, chunk_id)
            except Exception as e:  # noqa: BLE001 — 降级边界: LLM 失败走规则兜底
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

    def extract_batch(self, batch: list[tuple[str, str]], batch_size: int = 4) -> dict[str, list[Entity]]:
        """批量实体抽取 — 每批合并 1 次 LLM 调用, 返回 {chunk_id: [Entity]}。

        与 extract() 一致: 规则 + LLM 实体合并去重; 批量调用失败 → 该批全部走规则兜底。
        batch_size 越大调用越少, 但单次 prompt 越长, 默认 4 折中。
        """
        result: dict[str, list[Entity]] = {}
        for i in range(0, len(batch), batch_size):
            group = batch[i : i + batch_size]
            rules_map = {cid: self._rule_extract(text, cid) for text, cid in group}
            llm_map: dict[str, list[Entity]] = {}
            if self._llm_available and self._fail_count < self._max_fails:
                try:
                    llm_map = self._llm_extract_batch(group)
                except Exception as e:  # noqa: BLE001 — 降级边界: LLM 失败走规则兜底
                    self._fail_count += 1
                    if self._fail_count >= self._max_fails:
                        logger.info("LLM 实体抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
                    logger.debug("批量实体抽取失败: %s", str(e)[:100])
            for _, cid in group:
                seen = {e.name.lower(): e for e in rules_map[cid]}
                for e in llm_map.get(cid, []):
                    k = e.name.lower()
                    if k in seen:
                        seen[k].aliases.extend(e.aliases)
                    else:
                        seen[k] = e
                result[cid] = list(seen.values())
        return result

    def _llm_extract_batch(self, group: list[tuple[str, str]]) -> dict[str, list[Entity]]:
        """单批 LLM 抽取: 提示词按 [c0]...[/c0] 分段, 解析后按 chunk 映射。失败抛异常由调用方兜底。"""
        client = self._get_llm_client()
        if not client:
            raise RuntimeError("LLM 客户端不可用")
        block = "\n".join(f"[c{i}]{(text[:2000])}[/c{i}]" for i, (text, _) in enumerate(group))
        response = client.chat(
            messages=[
                {"role": "system", "content": "你是一个JSON提取工具，只输出JSON对象，不输出其他任何文字。"},
                {"role": "user", "content": BATCH_ENTITY_PROMPT.format(texts=block)},
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        content = response.content
        if not content or not content.strip():
            raise ValueError("批量实体抽取: 空响应")
        obj = _extract_json_object(content)
        if not obj:
            raise ValueError("批量实体抽取: JSON 解析失败")
        result: dict[str, list[Entity]] = {}
        for i, (_, cid) in enumerate(group):
            result[cid] = [
                Entity(
                    name=it["name"],
                    type=it.get("type", "Unknown"),
                    aliases=it.get("aliases", []),
                    source_chunk_id=cid,
                )
                for it in obj.get(f"c{i}", [])
                if isinstance(it, dict) and "name" in it
            ]
        return result

    def _rule_extract(self, text: str, chunk_id: str) -> list[Entity]:
        ents = []
        techs = [
            r"\b(FastAPI|Flask|Django|Spring)\b",
            r"\b(Milvus|Pinecone|Chroma|Weaviate|LanceDB)\b",
            r"\b(PostgreSQL|MySQL|MongoDB|Redis|Neo4j)\b",
            r"\b(Docker|Kubernetes|K8s)\b",
            r"\b(Python|Java|Go|Rust|TypeScript)\b",
            r"\b(React|Vue|Next\.js|Nuxt)\b",
            r"\b(BGE|SentenceTransformer|PyTorch|TensorFlow)\b",
            r"\b(Llama|GPT|Claude|Qwen|DeepSeek)\b",
            r"\b(PaddleOCR|MinerU|Docling)\b",
            r"\b(RAG|LLM|GraphRAG|OCR)\b",
            r"\b(NATS|RabbitMQ|Kafka)\b",
            r"\b(TokenHub|MinIO)\b",
        ]
        for pat in techs:
            for m in re.finditer(pat, text, re.IGNORECASE):
                ents.append(Entity(name=m.group(), type="Technology", source_chunk_id=chunk_id))
        return ents

    def _llm_extract(self, text: str, chunk_id: str) -> list[Entity]:
        client = self._get_llm_client()
        if not client:
            return []

        try:
            from engines.common.llm_client import CircuitBreakerOpenError

            response = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个JSON提取工具，只输出JSON数组，不输出其他任何文字。"},
                    {"role": "user", "content": ENTITY_PROMPT.format(text=text[:3000])},
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
                return [
                    Entity(
                        name=i["name"],
                        type=i.get("type", "Unknown"),
                        aliases=i.get("aliases", []),
                        source_chunk_id=chunk_id,
                    )
                    for i in items
                    if "name" in i
                ]
        except CircuitBreakerOpenError:
            logger.debug("LLM 熔断中，跳过实体抽取")
        except Exception as e:  # noqa: BLE001 — 降级边界: 计数并失败返回空走规则
            self._fail_count += 1
            if self._fail_count >= self._max_fails:
                logger.info("LLM 实体抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
            logger.debug("LLM 实体抽取跳过: %s", str(e)[:100])
        return []


# 关系动词模式: (predicate, 触发词) — 规则兜底用, 命中即定向建边
_RELATION_VERBS = [
    ("uses", ["使用", "采用", "应用", "运用", "借助"]),
    ("depends_on", ["依赖", "依赖于", "基于", "构建于"]),
    ("contains", ["包含", "包括", "含", "集成", "内置"]),
    ("references", ["参考", "引用", "参照"]),
    ("supplies", ["提供", "供应", "供给"]),
    ("owns", ["拥有", "隶属", "属于"]),
]
_RELATION_CO_OCCUR = "related_to"  # 共现兜底关系 (LLM 不可用时的弱关系)


class RelationExtractor(_BaseLLMExtractor):
    def extract(self, text: str, entities: list[Entity], chunk_id: str = "") -> list[Relation]:
        if not entities:
            return []
        # LLM 可用且未熔断 → LLM 优先; 返回空或异常 → 降级规则兜底
        if self._llm_available and self._fail_count < self._max_fails:
            llm_rels = self._llm_extract(text, entities, chunk_id)
            if llm_rels:
                return llm_rels
        return self._rule_extract(text, entities, chunk_id)

    def extract_batch(
        self, batch: list[tuple[str, str]], entities_map: dict[str, list[Entity]], batch_size: int = 4
    ) -> dict[str, list[Relation]]:
        """批量关系抽取 — 每批合并 1 次 LLM 调用, 返回 {chunk_id: [Relation]}。

        与 extract() 一致: LLM 优先, 失败/无实体 → 规则兜底。批量失败只降级该批。
        """
        result: dict[str, list[Relation]] = {cid: [] for _, cid in batch}
        for i in range(0, len(batch), batch_size):
            group = batch[i : i + batch_size]
            ents_group = [entities_map.get(cid, []) for _, cid in group]
            llm_map: dict[str, list[Relation]] = {}
            if any(ents_group) and self._llm_available and self._fail_count < self._max_fails:
                try:
                    llm_map = self._llm_extract_batch(group, ents_group)
                except Exception as e:  # noqa: BLE001 — 降级边界: LLM 失败走规则兜底
                    self._fail_count += 1
                    if self._fail_count >= self._max_fails:
                        logger.info("LLM 关系抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
                    logger.debug("批量关系抽取失败: %s", str(e)[:100])
            for (text, cid), ents in zip(group, ents_group):
                got = llm_map.get(cid)
                result[cid] = got if got is not None else (self._rule_extract(text, ents, cid) if ents else [])
        return result

    def _llm_extract_batch(
        self, group: list[tuple[str, str]], ents_group: list[list[Entity]]
    ) -> dict[str, list[Relation]]:
        """单批 LLM 抽取: 每段带已知实体, 解析后按 chunk 映射。失败抛异常由调用方兜底。"""
        client = self._get_llm_client()
        if not client:
            raise RuntimeError("LLM 客户端不可用")
        lines = []
        for i, ((text, _), ents) in enumerate(zip(group, ents_group)):
            names = ", ".join(e.name for e in ents[:20])
            lines.append(f"[c{i}] 实体: {names}\n{text[:2000]} [/c{i}]")
        block = "\n".join(lines)
        response = client.chat(
            messages=[
                {"role": "system", "content": "你是一个JSON提取工具，只输出JSON对象。"},
                {"role": "user", "content": BATCH_RELATION_PROMPT.format(texts=block)},
            ],
            max_tokens=1200,
            temperature=0.1,
        )
        content = response.content
        if not content or not content.strip():
            raise ValueError("批量关系抽取: 空响应")
        obj = _extract_json_object(content)
        if not obj:
            raise ValueError("批量关系抽取: JSON 解析失败")
        result: dict[str, list[Relation]] = {}
        for i, (_, cid) in enumerate(group):
            result[cid] = [
                Relation(
                    subject=it.get("subject", it.get("s", "")),
                    predicate=it.get("predicate", it.get("p", "")),
                    object=it.get("object", it.get("o", "")),
                    source_chunk_id=cid,
                )
                for it in obj.get(f"c{i}", [])
                if isinstance(it, dict) and ("subject" in it or "s" in it)
            ]
        return result

    def _llm_extract(self, text: str, entities: list[Entity], chunk_id: str) -> list[Relation]:
        client = self._get_llm_client()
        if not client:
            return []

        names = ", ".join([e.name for e in entities[:20]])
        try:
            from engines.common.llm_client import CircuitBreakerOpenError

            response = client.chat(
                messages=[
                    {"role": "system", "content": "你是一个JSON提取工具，只输出JSON数组。"},
                    {"role": "user", "content": RELATION_PROMPT.format(entities=names, text=text[:3000])},
                ],
                max_tokens=800,
                temperature=0.1,
            )
            content = response.content
            if not content or not content.strip():
                return []
            items = _extract_json_array(content)
            if items:
                return [
                    Relation(
                        subject=i.get("subject", i.get("s", "")),
                        predicate=i.get("predicate", i.get("p", "")),
                        object=i.get("object", i.get("o", "")),
                        source_chunk_id=chunk_id,
                    )
                    for i in items
                    if "subject" in i or "s" in i
                ]
        except CircuitBreakerOpenError:
            logger.debug("LLM 熔断中，跳过关系抽取")
        except Exception as e:  # noqa: BLE001 — 降级边界: LLM 失败走规则兜底
            self._fail_count += 1
            if self._fail_count >= self._max_fails:
                logger.info("LLM 关系抽取连续失败 %d 次，降级为纯规则模式", self._max_fails)
            logger.debug("LLM 关系抽取失败: %s", str(e)[:100])
        return []

    def _rule_extract(self, text: str, entities: list[Entity], chunk_id: str) -> list[Relation]:
        """规则兜底: 动词模式定向建边, 无动词则同句共现弱关系。

        LLM Key 缺失或熔断时仍产出边, 保证 Graph RAG 多跳推理可用。
        """
        relations = []
        seen = set()
        for sent in re.split(r"[。！？；\n]+", text):
            present = [e for e in entities if e.name and e.name.lower() in sent.lower()]
            if len(present) < 2:
                continue
            triples = self._match_verb_triples(sent, present)
            if triples:
                for s, pred, o in triples:
                    self._add_relation(relations, seen, s, pred, o, chunk_id)
            else:
                for i in range(len(present)):
                    for j in range(i + 1, len(present)):
                        self._add_relation(
                            relations, seen, present[i].name, _RELATION_CO_OCCUR, present[j].name, chunk_id
                        )
        return relations

    @staticmethod
    def _match_verb_triples(sent: str, present: list[Entity]) -> list[tuple]:
        """动词模式: 命中第一个动词 → [(subject, predicate, object)]; 未命中 → []。

        主体取动词前实体, 客体取动词后实体 (近似定向)。
        """
        lower = sent.lower()
        for pred, verbs in _RELATION_VERBS:
            for v in verbs:
                idx = lower.find(v)
                if idx < 0:
                    continue
                before = [e for e in present if lower.find(e.name.lower()) < idx]
                after = [e for e in present if lower.find(e.name.lower()) > idx]
                return [(s.name, pred, o.name) for s in before for o in after]
        return []

    @staticmethod
    def _add_relation(relations: list, seen: set, subject: str, predicate: str, obj: str, chunk_id: str):
        """去重后追加一条关系。"""
        key = (subject.lower(), predicate, obj.lower())
        if key in seen:
            return
        seen.add(key)
        relations.append(Relation(subject=subject, predicate=predicate, object=obj, source_chunk_id=chunk_id))
