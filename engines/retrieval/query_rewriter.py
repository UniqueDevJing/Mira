"""查询改写器"""
from typing import List


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
            response = self.llm.chat([{"role": "user", "content": prompt}], temperature=0.7)
            lines = response.strip().split("\n")
            return [l.strip().lstrip("0123456789. -") for l in lines if l.strip()][:max_rewrites]

        # 无 LLM 时的简单改写
        return [
            f"{original_query} 相关文档 详细信息",
            f"关于 {original_query} 的所有信息",
        ][:max_rewrites]
