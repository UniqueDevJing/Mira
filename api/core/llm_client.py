"""统一 LLM 客户端 (异步) — 重试、熔断、Token 统计。

同步版 SyncLLMClient 与共享底座 (_BaseLLMClient/熔断/解析) 已下沉
engines/common/llm_client.py — 引擎层实体抽取复用同步客户端, 消除 engines → api 反向依赖。
本模块保留 async LLMClient, 并 re-export 同步件保持调用方兼容。
"""

import asyncio
import json
import logging
import time
from threading import Lock

import httpx

from api.core.metrics import llm_errors_total  # re-export 自 engines/common.metrics
from engines.common.llm_client import (
    CircuitBreakerOpenError,
    CircuitBreakerState,
    LLMResponse,
    SyncLLMClient,
    _BaseLLMClient,
)

logger = logging.getLogger(__name__)


class LLMClient(_BaseLLMClient):
    """异步 LLM 客户端 — 事件循环内调用。

    特性：
    - 指数退避重试（4xx 不重试，429/5xx 重试）
    - 熔断机制（连续失败 N 次后快速失败）
    - Token 消耗统计（Prometheus 指标）
    - 连接池复用
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 连接池 (trust_env=False: tokenhub 国内直连, 避免 DevSidecar 等系统代理拖慢/超时)
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            trust_env=False,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=5,
            ),
        )

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """发送聊天请求。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            json_mode: 是否返回 JSON 格式
            timeout: 覆盖实例默认超时 (None 用 self.timeout)

        Returns:
            LLMResponse: 包含内容和 token 统计的响应

        Raises:
            httpx.HTTPStatusError: HTTP 错误（非重试后仍失败）
            CircuitBreakerOpenError: 熔断器打开
        """
        # 检查熔断器
        if self._is_circuit_open():
            raise CircuitBreakerOpenError("LLM 服务熔断中，请稍后重试")

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = await self._call_llm(messages, temperature, max_tokens, json_mode, timeout)
                elapsed_ms = (time.time() - start_time) * 1000

                # 统计 token
                self._track_tokens(response)

                # 重置熔断器
                self._reset_circuit_breaker()

                logger.info(
                    "LLM 调用成功: model=%s, tokens_in=%d, tokens_out=%d, latency=%.1fms",
                    self.model,
                    response.prompt_tokens,
                    response.completion_tokens,
                    elapsed_ms,
                )

                return LLMResponse(
                    content=response.content,
                    reasoning_content=response.reasoning_content,
                    prompt_tokens=response.prompt_tokens,
                    completion_tokens=response.completion_tokens,
                    total_tokens=response.total_tokens,
                    latency_ms=elapsed_ms,
                )

            except httpx.HTTPStatusError as e:
                status_code = e.response.status_code
                last_error = e

                # 4xx 错误（除 429）是请求侧永久错误, 不重试也不记熔断 — 熔断语义是"服务不可用",
                # 401/400 累计会打开熔断掩盖真实配置错误 (如 api_key 失效), 对上游也无保护意义
                if 400 <= status_code < 500 and status_code != 429:
                    logger.warning("LLM 调用失败 (HTTP %d): %s", status_code, str(e)[:200])
                    llm_errors_total.labels(error_type=f"http_{status_code}").inc()
                    raise

                # 429/5xx 可重试
                if attempt < self.max_retries:
                    wait_time = 2**attempt * 0.5
                    logger.warning(
                        "LLM 调用失败 (HTTP %d), 重试 %d/%d, 等待 %.1fs",
                        status_code,
                        attempt + 1,
                        self.max_retries,
                        wait_time,
                    )
                    llm_errors_total.labels(error_type=f"http_{status_code}").inc()
                    await asyncio.sleep(wait_time)
                else:
                    self._record_failure()
                    llm_errors_total.labels(error_type="max_retries").inc()
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2**attempt * 0.5
                    logger.warning("LLM 调用超时, 重试 %d/%d, 等待 %.1fs", attempt + 1, self.max_retries, wait_time)
                    llm_errors_total.labels(error_type="timeout").inc()
                    await asyncio.sleep(wait_time)
                else:
                    self._record_failure()
                    llm_errors_total.labels(error_type="timeout_max_retries").inc()
                    raise

            except Exception as e:
                last_error = e
                self._record_failure()
                llm_errors_total.labels(error_type="unknown").inc()
                logger.error("LLM 调用异常: %s", str(e)[:200])
                raise

        # 所有重试都失败
        self._record_failure()
        raise last_error

    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        timeout: float | None = None,
    ) -> LLMResponse:
        """实际调用 LLM API。timeout 覆盖实例默认 (None 用 self.timeout)。"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        resp = await self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=timeout or self.timeout,
        )
        resp.raise_for_status()

        return self._parse_completion(resp.json())

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
    ):
        """流式聊天。逐块产出 dict:
        - {"type": "delta", "content": str, "reasoning": str}
        - {"type": "usage", "usage": {...}}  (流结束时)
        熔断检查同 chat；异常时记录失败并抛出。
        """
        if self._is_circuit_open():
            raise CircuitBreakerOpenError("LLM 服务熔断中，请稍后重试")

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "") or ""
                    reasoning = delta.get("reasoning_content", "") or ""
                    if content or reasoning:
                        yield {"type": "delta", "content": content, "reasoning": reasoning}
                    if chunk.get("usage"):
                        yield {"type": "usage", "usage": chunk["usage"]}
            self._reset_circuit_breaker()
        except httpx.HTTPStatusError as e:
            # 4xx 永久错误不记熔断 (与 chat 对齐); 5xx 服务不可用才累计
            if e.response.status_code >= 500:
                self._record_failure()
            raise
        except Exception:
            self._record_failure()
            raise

    async def close(self):
        """关闭客户端"""
        await self._client.aclose()


# 共享熔断状态: async/sync 客户端共用同一熔断器, 上游持续失败时任一路径触发即全保护
# (原各建独立实例独立熔断, 主流程与 SelfRetrieval 改写路径保护各半)
_shared_circuit_breaker = CircuitBreakerState(half_open_time=30.0)

# 全局单例（线程安全）
_llm_client: LLMClient | None = None
_sync_llm_client: SyncLLMClient | None = None
_llm_lock = Lock()
_sync_llm_lock = Lock()


def get_llm_client() -> LLMClient:
    """获取全局异步 LLM 客户端单例"""
    global _llm_client
    if _llm_client is None:
        with _llm_lock:
            if _llm_client is None:
                from api.config import settings

                _llm_client = LLMClient(
                    base_url=settings.llm_base_url,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    circuit_breaker_state=_shared_circuit_breaker,
                )
    return _llm_client


def get_sync_llm_client() -> SyncLLMClient:
    """获取全局同步 LLM 客户端单例"""
    global _sync_llm_client
    if _sync_llm_client is None:
        with _sync_llm_lock:
            if _sync_llm_client is None:
                from api.config import settings

                _sync_llm_client = SyncLLMClient(
                    base_url=settings.llm_base_url,
                    model=settings.llm_model,
                    api_key=settings.llm_api_key,
                    circuit_breaker_state=_shared_circuit_breaker,
                )
    return _sync_llm_client
