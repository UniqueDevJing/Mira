"""共享 LLM 基础件 — 同步客户端 + 熔断 + 响应解析。

下沉原因: engines 层需同步 LLM 客户端 (实体/关系抽取用), 原从 engines 导入
api.core.llm_client 形成 engines → api 反向依赖, 破坏分层单向性。
api 层从本模块 re-export, 保持调用方兼容。
"""

import json
import logging
import time
from dataclasses import dataclass
from threading import Lock

import httpx

from engines.common.metrics import llm_errors_total, llm_tokens_total

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


class CircuitBreakerOpenError(Exception):
    """熔断器打开异常"""


class _BaseLLMClient:
    """共享底座: 熔断器状态 + token 统计 + 响应解析。

    chat 重试循环因 await/time.sleep 差异由子类分别实现, 其余逻辑不再重复。
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
        circuit_breaker_state: "CircuitBreakerState | None" = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.max_retries = max_retries
        self.timeout = timeout

        # 熔断器: 允许注入共享状态 (async/sync 客户端共用, 防上游持续失败时保护减半)
        self._circuit_breaker = circuit_breaker_state or CircuitBreakerState(
            half_open_time=circuit_breaker_recovery_time
        )
        self._cb_threshold = circuit_breaker_threshold
        self._cb_lock = Lock()

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

    @staticmethod
    def _parse_completion(data) -> LLMResponse:
        """解析 chat/completions 响应 (async/sync 共用)"""
        if isinstance(data, str):
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


class SyncLLMClient(_BaseLLMClient):
    """同步 LLM 客户端 — 用于同步代码调用（后台线程）。

    内部使用 httpx 同步客户端，保持与 LLMClient 相同的重试和熔断逻辑。
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 同步客户端 (trust_env=False: 避免系统代理)
        self._client = httpx.Client(timeout=self.timeout, trust_env=False)

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        json_mode: bool = False,
        timeout: float | None = None,
    ) -> LLMResponse:
        """同步发送聊天请求。timeout 覆盖实例默认 (None 用 self.timeout)。"""
        # 检查熔断器
        if self._is_circuit_open():
            raise CircuitBreakerOpenError("LLM 服务熔断中，请稍后重试")

        start_time = time.time()
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                response = self._call_llm_sync(messages, temperature, max_tokens, json_mode, timeout)
                elapsed_ms = (time.time() - start_time) * 1000

                # 统计 token
                self._track_tokens(response)

                # 重置熔断器
                self._reset_circuit_breaker()

                logger.info(
                    "LLM 调用成功 (sync): model=%s, tokens_in=%d, tokens_out=%d, latency=%.1fms",
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

                # 4xx 错误（除 429）是请求侧永久错误, 不重试也不记熔断 (见 async 版注释)
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
                    time.sleep(wait_time)
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
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
        timeout: float | None = None,
    ) -> LLMResponse:
        """同步调用 LLM API。timeout 覆盖实例默认 (None 用 self.timeout)。"""
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
            timeout=timeout or self.timeout,
        )
        resp.raise_for_status()

        return self._parse_completion(resp.json())

    def close(self):
        """关闭客户端"""
        self._client.close()
