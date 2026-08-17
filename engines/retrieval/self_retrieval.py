"""Self-Retrieval 主循环"""

from dataclasses import dataclass, field


@dataclass
class RetrievalTrace:
    round_num: int
    query: str
    results: list[dict]
    eval_result: object
    rewritten: list[str] = field(default_factory=list)


class SelfRetrieval:
    def __init__(self, retriever, evaluator, rewriter, max_rounds: int = 3):
        self.retriever = retriever
        self.evaluator = evaluator
        self.rewriter = rewriter
        self.max_rounds = max_rounds

    def retrieve(self, query: str, top_k: int = 20) -> dict:
        trace = []
        current_query = query

        for round_num in range(1, self.max_rounds + 1):
            results = self.retriever.retrieve(current_query, top_k)
            eval_result = self.evaluator.evaluate(current_query, results.get("documents", []))
            trace.append(RetrievalTrace(round_num, current_query, results, eval_result))

            if not eval_result.need_rewrite:
                break

            rewritten = self.rewriter.rewrite(current_query, eval_result)
            trace[-1].rewritten = rewritten
            if not rewritten:
                break
            # 改写结果与当前查询相同 (如已含模板后缀的污染查询) → 无改进空间, 立即终止
            if rewritten[0] == current_query:
                break
            current_query = rewritten[0]

        all_docs = {}
        for t in trace:
            for doc in t.results.get("documents", []):
                key = doc.get("id") or doc.get("chunk_id")
                if key and key not in all_docs:
                    all_docs[key] = doc

        graph_context = trace[-1].results.get("graph_context") if trace else None
        return {
            "documents": list(all_docs.values())[:top_k],
            "graph_context": graph_context,
            "retrieval_rounds": len(trace),
            "trace": [
                {
                    "round": t.round_num,
                    "query": t.query,
                    "relevance": t.eval_result.relevance_score,
                    "need_rewrite": t.eval_result.need_rewrite,
                    "rewritten": t.rewritten,
                }
                for t in trace
            ],
        }
