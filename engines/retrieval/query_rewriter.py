"""查询改写器"""
import logging
from typing import List

logger = logging.getLogger(__name__)


class QueryRewriter:
    STRATEGIES = {
        "keyword_expand": "关键词扩展",
        "decompose": "子问题拆分",
        "synonym": "同义替换",
        "abstract_adjust": "抽象层级调整",
    }

    def __init__(self, llm_client=None):
        self.llm = llm_client

    def rewrite(self, original_query: str, eval_result,
                max_rewrites: int = 3) -> List[str]:
        strategies = ["keyword_expand", "synonym"]
        if eval_result.relevance_score < 0.5:
            strategies = ["keyword_expand", "synonym"]
        elif eval_result.coverage_score < 0.5:
            strategies = ["decompose", "abstract_adjust"]

        if self.llm:
            prompt = f"""原始查询无法获得满意结果，请改写。
原始查询: {original_query}
失败原因: {eval_result.reason}
请生成 {max_rewrites} 个改写后的查询，每行一个。"""
            try:
                response = self.llm.chat(
                    [{"role": "user", "content": prompt}], temperature=0.7
                )
                # SyncLLMClient.chat 返回 LLMResponse 对象
                content = response.content if hasattr(response, 'content') else str(response)
                lines = content.strip().split("\n")
                return [l.strip().lstrip("0123456789. -") for l in lines if l.strip()][:max_rewrites]
            except Exception as e:
                logger.warning("LLM 查询改写失败，降级到模板改写: %s", str(e)[:200])

        # 无 LLM 时的简单改写
        return [
            f"{original_query} 相关文档 详细信息",
            f"关于 {original_query} 的所有信息",
        ][:max_rewrites]
