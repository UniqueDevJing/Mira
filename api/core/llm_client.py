"""统一 LLM 客户端 — 重试、熔断、Token 统计。

所有 LLM 调用必须通过此客户端，禁止直接调用 httpx。
"""
import time
import asyncio
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from threading import Lock

import httpx

from api.core.metrics import llm_tokens_total, llm_errors_total

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """LLM 响应封装"""
    content: str
    reasoning_content: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class CircuitBreakerState:
    """熔断器状态"""
    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
    half_open_time: float = 30.0  # 熔断恢复时间（秒）


class LLMClient:
    """统一 LLM 客户端。

    特性：
    - 指数退避重试（4xx 不重试，429/5xx 重试）
    - 熔断机制（连续失败 N 次后快速失败）
    - Token 消耗统计（Prometheus 指标）
    - 连接池复用
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_time: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        # 熔断器
        self._circuit_breaker = CircuitBreakerState(
            half_open_time=circuit_breaker_recovery_time
        )
        self._cb_threshold = circuit_breaker_threshold
        self._cb_lock = Lock()

        # 连接池
        self._client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=20,
                max_keepalive_connections=5,
            ),
        )

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> LLMResponse:
        """发送聊天请求。

        Args:
            messages: 消息列表，格式 [{"role": "user", "content": "..."}]
            temperature: 温度参数
            max_tokens: 最大生成 token 数
            json_mode: 是否返回 JSON 格式

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
                response = await self._call_llm(
                    messages, temperature, max_tokens, json_mode
                )
                elapsed_ms = (time.time() - start_time) * 1000

                # 统计 token
                self._track_tokens(response)

                # 重置熔断器
                self._reset_circuit_breaker()

                logger.info(
                    "LLM 调用成功: model=%s, tokens_in=%d, tokens_out=%d, latency=%.1fms",
                    self.model, response.prompt_tokens, response.completion_tokens, elapsed_ms
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

                # 4xx 错误（除 429）不重试
                if 400 <= status_code < 500 and status_code != 429:
                    logger.warning("LLM 调用失败 (HTTP %d): %s", status_code, str(e)[:200])
                    self._record_failure()
                    llm_errors_total.labels(error_type=f"http_{status_code}").inc()
                    raise

                # 429/5xx 可重试
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt * 0.5
                    logger.warning(
                        "LLM 调用失败 (HTTP %d), 重试 %d/%d, 等待 %.1fs",
                        status_code, attempt + 1, self.max_retries, wait_time
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
                    wait_time = 2 ** attempt * 0.5
                    logger.warning(
                        "LLM 调用超时, 重试 %d/%d, 等待 %.1fs",
                        attempt + 1, self.max_retries, wait_time
                    )
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
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """实际调用 LLM API"""
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
        )
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, str):
            import json
            data = json.loads(data)

        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        # DeepSeek 系列模型 reasoning_content 可能包含内容
        if not content and reasoning_content:
            content = reasoning_content

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _track_tokens(self, response: LLMResponse):
        """统计 token 消耗到 Prometheus"""
        if response.prompt_tokens > 0:
            llm_tokens_total.labels(type="prompt").inc(response.prompt_tokens)
        if response.completion_tokens > 0:
            llm_tokens_total.labels(type="completion").inc(response.completion_tokens)

    def _is_circuit_open(self) -> bool:
        """检查熔断器是否打开"""
        with self._cb_lock:
            if not self._circuit_breaker.is_open:
                return False

            # 检查是否到了半开时间
            elapsed = time.time() - self._circuit_breaker.last_failure_time
            if elapsed >= self._circuit_breaker.half_open_time:
                logger.info("LLM 熔断器进入半开状态")
                self._circuit_breaker.is_open = False
                return False

            return True

    def _record_failure(self):
        """记录失败，可能触发熔断"""
        with self._cb_lock:
            self._circuit_breaker.failure_count += 1
            self._circuit_breaker.last_failure_time = time.time()

            if self._circuit_breaker.failure_count >= self._cb_threshold:
                if not self._circuit_breaker.is_open:
                    logger.warning(
                        "LLM 熔断器打开: 连续失败 %d 次, %d 秒后恢复",
                        self._circuit_breaker.failure_count,
                        self._circuit_breaker.half_open_time,
                    )
                self._circuit_breaker.is_open = True

    def _reset_circuit_breaker(self):
        """重置熔断器"""
        with self._cb_lock:
            if self._circuit_breaker.failure_count > 0:
                logger.info("LLM 熔断器重置")
            self._circuit_breaker.failure_count = 0
            self._circuit_breaker.is_open = False

    async def close(self):
        """关闭客户端"""
        await self._client.aclose()


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""
    pass


class SyncLLMClient:
    """同步 LLM 客户端 — 用于同步代码调用。

    内部使用 httpx 同步客户端，保持与 LLMClient 相同的重试和熔断逻辑。
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        max_retries: int = 3,
        timeout: float = 60.0,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_recovery_time: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        # 熔断器
        self._circuit_breaker = CircuitBreakerState(
            half_open_time=circuit_breaker_recovery_time
        )
        self._cb_threshold = circuit_breaker_threshold
        self._cb_lock = Lock()

        # 同步客户端
        self._client = httpx.Client(timeout=timeout)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
    ) -> LLMResponse:
        """同步发送聊天请求"""
        # 检查熔断器
        if self._is_circuit_open():
            raise CircuitBreakerOpenError("LLM 服务熔断中，请稍后重试")

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_llm_sync(
                    messages, temperature, max_tokens, json_mode
                )
                elapsed_ms = (time.time() - start_time) * 1000

                # 统计 token
                self._track_tokens(response)

                # 重置熔断器
                self._reset_circuit_breaker()

                logger.info(
                    "LLM 调用成功 (sync): model=%s, tokens_in=%d, tokens_out=%d, latency=%.1fms",
                    self.model, response.prompt_tokens, response.completion_tokens, elapsed_ms
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

                # 4xx 错误（除 429）不重试
                if 400 <= status_code < 500 and status_code != 429:
                    logger.warning("LLM 调用失败 (HTTP %d): %s", status_code, str(e)[:200])
                    self._record_failure()
                    llm_errors_total.labels(error_type=f"http_{status_code}").inc()
                    raise

                # 429/5xx 可重试
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt * 0.5
                    logger.warning(
                        "LLM 调用失败 (HTTP %d), 重试 %d/%d, 等待 %.1fs",
                        status_code, attempt + 1, self.max_retries, wait_time
                    )
                    llm_errors_total.labels(error_type=f"http_{status_code}").inc()
                    time.sleep(wait_time)
                else:
                    self._record_failure()
                    llm_errors_total.labels(error_type="max_retries").inc()
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                if attempt < self.max_retries:
                    wait_time = 2 ** attempt * 0.5
                    logger.warning(
                        "LLM 调用超时, 重试 %d/%d, 等待 %.1fs",
                        attempt + 1, self.max_retries, wait_time
                    )
                    llm_errors_total.labels(error_type="timeout").inc()
                    time.sleep(wait_time)
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

    def _call_llm_sync(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> LLMResponse:
        """同步调用 LLM API"""
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        headers = {"Authorization": f"Bearer {self.api_key}"}

        resp = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        resp.raise_for_status()

        data = resp.json()
        if isinstance(data, str):
            import json
            data = json.loads(data)

        choice = data["choices"][0]
        message = choice["message"]

        content = message.get("content", "")
        reasoning_content = message.get("reasoning_content", "")

        # DeepSeek 系列模型 reasoning_content 可能包含内容
        if not content and reasoning_content:
            content = reasoning_content

        usage = data.get("usage", {})

        return LLMResponse(
            content=content,
            reasoning_content=reasoning_content,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
        )

    def _track_tokens(self, response: LLMResponse):
        """统计 token 消耗到 Prometheus"""
        if response.prompt_tokens > 0:
            llm_tokens_total.labels(type="prompt").inc(response.prompt_tokens)
        if response.completion_tokens > 0:
            llm_tokens_total.labels(type="completion").inc(response.completion_tokens)

    def _is_circuit_open(self) -> bool:
        """检查熔断器是否打开"""
        with self._cb_lock:
            if not self._circuit_breaker.is_open:
                return False

            # 检查是否到了半开时间
            elapsed = time.time() - self._circuit_breaker.last_failure_time
            if elapsed >= self._circuit_breaker.half_open_time:
                logger.info("LLM 熔断器进入半开状态")
                self._circuit_breaker.is_open = False
                return False

            return True

    def _record_failure(self):
        """记录失败，可能触发熔断"""
        with self._cb_lock:
            self._circuit_breaker.failure_count += 1
            self._circuit_breaker.last_failure_time = time.time()

            if self._circuit_breaker.failure_count >= self._cb_threshold:
                if not self._circuit_breaker.is_open:
                    logger.warning(
                        "LLM 熔断器打开: 连续失败 %d 次, %d 秒后恢复",
                        self._circuit_breaker.failure_count,
                        self._circuit_breaker.half_open_time,
                    )
                self._circuit_breaker.is_open = True

    def _reset_circuit_breaker(self):
        """重置熔断器"""
        with self._cb_lock:
            if self._circuit_breaker.failure_count > 0:
                logger.info("LLM 熔断器重置")
            self._circuit_breaker.failure_count = 0
            self._circuit_breaker.is_open = False

    def close(self):
        """关闭客户端"""
        self._client.close()


# 全局单例（线程安全）
_llm_client: Optional[LLMClient] = None
_sync_llm_client: Optional[SyncLLMClient] = None
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
                )
    return _sync_llm_client
