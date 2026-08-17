"""查询改写器"""

import logging
from typing import ClassVar

logger = logging.getLogger(__name__)


class QueryRewriter:
    STRATEGIES: ClassVar[dict[str, str]] = {
        "keyword_expand": "关键词扩展",
        "decompose": "子问题拆分",
        "synonym": "同义替换",
        "abstract_adjust": "抽象层级调整",
    }

    def __init__(self, llm_client=None, rewrite_timeout_s: float = 3.0):
        """rewrite_timeout_s 可经 orchestrator 注入 config 值。

        改写挂起时及时释放线程 (to_thread 不可取消, 60s 默认会占满线程池)。
        """
        self.llm = llm_client
        self.rewrite_timeout_s = rewrite_timeout_s

    def rewrite(self, original_query: str, eval_result, max_rewrites: int = 3) -> list[str]:
        if self.llm:
            prompt = f"""原始查询无法获得满意结果，请改写。
原始查询: {original_query}
失败原因: {eval_result.reason}
请生成 {max_rewrites} 个改写后的查询，每行一个。"""
            try:
                response = self.llm.chat(
                    [{"role": "user", "content": prompt}], temperature=0.7, timeout=self.rewrite_timeout_s
                )
                # SyncLLMClient.chat 返回 LLMResponse 对象
                content = response.content if hasattr(response, "content") else str(response)
                lines = content.strip().split("\n")
                return [l.strip().lstrip("0123456789. -") for l in lines if l.strip()][:max_rewrites]
            except Exception as e:  # noqa: BLE001 — 降级边界: LLM 失败走模板改写
                logger.warning("LLM 查询改写失败，降级到模板改写: %s", str(e)[:200])

        # 无 LLM 时的简单改写: 去重 + 防逐轮膨胀
        # 若查询已被模板污染（含修饰后缀），直接返回原样, 避免第 N 轮重复拼接
        if any(t in original_query for t in ("相关文档", "详细信息", "关于 ", "的所有信息")):
            return [original_query]
        candidates = [
            f"{original_query} 相关文档",
            f"关于 {original_query} 的信息",
        ]
        out = []
        for c in candidates:
            if c not in out and c != original_query:
                out.append(c)
            if len(out) >= max_rewrites:
                break
        return out or [original_query]
