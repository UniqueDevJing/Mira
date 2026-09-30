# -*- coding: utf-8 -*-
"""完整闭环流水线（离线微缩版）—— 提问 → 检索 → 生成 → 校验 → 响应。

为什么这样做：
  完整 RAG 需要嵌入模型 / 大语料 / LLM，演示服务器不具备；但闭环的**结构**
  可以在纯标准库内真实执行——检索是真实的 TF-IDF 打分排序（非随机模拟），
  生成采用确定性回放脑（与跨境电商项目同一模式：mock=确定性、真实 LLM 可替换），
  校验是真实 verifier，响应按三级裁决真实分级。
  每个阶段独立计时、输入输出可追溯——这就是"完整闭环"的演示形态。

确定性：语料 / 问题 / 评测时间全部固定（DEMO_NOW），任何人重跑结果一致。
"""
import re
import time
from datetime import datetime, timedelta, timezone
from math import log

MAX_AGE = timedelta(days=365)

# 自定义提问的相关度门槛（实测校准：相关问题 top1 ≥3.0，无关问题 ≤1.2）
MIN_REL_SCORE = 2.0

# ── 微缩语料：4 个真实结构的金融文档片段（事实互洽，带报告期） ──
# 语料设计约束（由校验器的锚定机制反推）：
#   1. 每个数字紧邻其年份（30 字符锚定窗口内），跨年对比才不会被误判为同指标矛盾；
#   2. 不同文件的同名指标必须数值互洽（同值=互证，异值=会被判跨源矛盾）；
#   3. 研报类背景 chunk 不含数字（避免与年报数字形成伪矛盾）。
CORPUS = [
    {"id": "annual2024", "name": "公司2024年年度报告", "as_of": "2025-03-28",
     "text": "2024年年报：2024年公司实现营业收入12.5亿元（同比增长25%），"
             "2024年归母净利润2.4亿元，2024年毛利率40%，2024年研发费用1.8亿元。"},
    {"id": "annual2023", "name": "公司2023年年度报告", "as_of": "2024-04-15",
     "text": "2023年年报：2023年公司实现营业收入10亿元，2023年归母净利润2亿元，"
             "2023年研发费用1.5亿元。"},
    {"id": "balance2024", "name": "公司2024年资产负债表", "as_of": "2025-03-28",
     "text": "2024年资产负债表：2024年末总资产15.8亿元，2024年末总负债6.3亿元，"
             "2024年资产负债率39.9%。"},
    {"id": "cashflow2024", "name": "公司2024年现金流量表", "as_of": "2025-03-28",
     "text": "2024年现金流量表：2024年经营活动产生的现金流量净额为3.1亿元。"},
    {"id": "dividend2024", "name": "公司2024年度利润分配方案", "as_of": "2025-04-10",
     "text": "2024年度利润分配方案：2024年分红总额为1.2亿元。"},
    {"id": "employees2024", "name": "公司2024年年报-员工情况", "as_of": "2025-03-28",
     "text": "截至2024年末，公司在册员工1860人。"},
    {"id": "rdteam2024", "name": "公司2024年年报-研发团队", "as_of": "2025-03-28",
     "text": "2024年研发团队：研发人员620人。"},
    {"id": "segment2024", "name": "公司2024年年报-分部数据", "as_of": "2025-03-28",
     "text": "2024年分部数据：智能硬件业务收入7.2亿元，软件服务业务收入5.3亿元。"},
    {"id": "research", "name": "行业研究报告(2025)", "as_of": "2025-06-01",
     "text": "2025年行业研究：公司盈利能力处于行业第一梯队，市场地位稳固，成长性良好。"},
    {"id": "news", "name": "财经媒体报道(2025)", "as_of": "2025-05-10",
     "text": "财经媒体：据公司年报披露，2024年公司营业收入12.5亿元，净利润2.4亿元，研发投入持续加大。"},
]

# 演示专用固定时间：让 2024 年报"新鲜"（pass 路径）、2023 年报自然超期（warn 路径）
DEMO_NOW = datetime(2025, 7, 1, tzinfo=timezone.utc)

# ── 预设问题：ok=正确回答；fault=(类型, 幻觉回答)。fault=None 的题永不注入 ──
QUESTIONS = [
    {"id": "q1", "q": "公司2024年营业收入是多少？",
     "ok": "根据公司2024年年度报告，公司2024年实现营业收入12.5亿元，同比增长25%。",
     "fault": ("数量级篡改", "根据年报，公司2024年实现营业收入125亿元，同比增长25%。")},
    {"id": "q2", "q": "2024年净利润是多少？",
     "ok": "根据2024年年度报告，公司2024年归母净利润为2.4亿元。",
     "fault": ("币种臆造", "根据年报，公司2024年归母净利润为2.4亿美元。")},
    {"id": "q3", "q": "公司毛利率处于什么水平？",
     "ok": "2024年公司毛利率40%，盈利能力处于行业第一梯队。",
     "fault": ("无来源数字", "2024年公司毛利率52%，遥遥领先行业。")},
    {"id": "q4", "q": "公司2023年的研发费用是多少？",
     "ok": "根据公司2023年年度报告，2023年研发费用为1.5亿元。",
     "fault": ("数量级篡改", "2023年研发费用为15亿元。")},
    {"id": "q5", "q": "研发费用占营业收入的比重是多少？",
     "ok": "2024年研发费用1.8亿元，占营业收入12.5亿元的14.4%。",
     "fault": ("无来源数字", "2024年研发费用占营业收入的24.4%。"),
     "calc": {"expr": "1.8/12.5", "label": "研发费用占营收比", "result": 0.144}},
    {"id": "q6", "q": "公司2024年营收和净利润分别是多少？",
     "ok": "2024年公司实现营业收入12.5亿元，归母净利润2.4亿元。",
     "fault": None},
    {"id": "q7", "q": "公司资产负债率是多少？",
     "ok": "2024年末公司资产负债率39.9%（总负债6.3亿元 / 总资产15.8亿元）。",
     "fault": ("计算错误", "2024年末公司资产负债率66.6%。"),
     "calc": {"expr": "6.3/15.8", "label": "资产负债率", "result": 0.3987341772151899}},
    {"id": "q8", "q": "公司经营活动现金流情况如何？",
     "ok": "2024年经营活动产生的现金流量净额为3.1亿元。",
     "fault": ("数量级篡改", "2024年经营活动现金流净额31亿元。")},
    {"id": "q9", "q": "2024年每股分红是多少？",
     "ok": "根据2024年度利润分配方案，2024年分红总额为1.2亿元。",
     "fault": ("数量级篡改", "2024年分红总额为12亿元。")},
    {"id": "q10", "q": "公司有多少研发人员？",
     "ok": "根据公司2024年年报，2024年研发人员620人。",
     "fault": ("数量级篡改", "2024年研发人员6200人。")},
]

# ── 检索：真实 TF-IDF 打分（中文 bigram + ASCII 词元），非随机模拟 ──

_TOKEN_RE = re.compile(r"[0-9A-Za-z.]+")

def _terms(text: str) -> list[str]:
    """中文按 bigram 切词，ASCII 连续串按词元；检索的最小语义单元。"""
    text = re.sub(r"\s+", "", text)
    terms, i = [], 0
    while i < len(text):
        m = _TOKEN_RE.match(text, i)
        if m:
            terms.append(m.group(0).lower())
            i = m.end()
        else:
            if i + 1 < len(text):
                terms.append(text[i:i + 2])
            i += 1
    return terms

_DF = {}
for _c in CORPUS:
    for _t in set(_terms(_c["text"] + _c["name"])):
        _DF[_t] = _DF.get(_t, 0) + 1
_N = len(CORPUS)


def retrieve(query: str, top_k: int = 2) -> list[dict]:
    """TF-IDF 打分排序，返回 top_k 个 chunk（含得分与命中词元，供前端展示）。"""
    q_terms = _terms(query)
    scored = []
    for c in CORPUS:
        c_terms = _terms(c["text"] + c["name"])
        c_set = {}
        for t in c_terms:
            c_set[t] = c_set.get(t, 0) + 1
        s, hits = 0.0, []
        for t in dict.fromkeys(q_terms):          # 去重保序
            df = _DF.get(t, 0)
            if df and t in c_set:
                idf = log(1 + _N / df)
                s += idf * (1 + log(c_set[t]))
                hits.append(t)
        # 不做长度归一：打分按 query 词元去重累计，天然有界；
        # sqrt(len) 归一会过度惩罚信息密度高的长文档（年报被短新闻挤到第二位）
        scored.append({"chunk": c, "score": round(s, 4), "hits": hits[:8]})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _best_sentence(chunk: dict, query: str) -> tuple[str, float]:
    """抽取式生成：返回 chunk 中与 query 最匹配的句子（优先含数字的证据句）。"""
    q_terms = set(_terms(query))
    best, best_score = "", -1.0
    for sent in re.split(r"[。；，]", chunk["text"]):  # 子句级抽取, 答案更精准
        sent = sent.strip()
        if not sent:
            continue
        overlap = len(q_terms & set(_terms(sent)))
        score = overlap + (0.5 if re.search(r"[0-9]", sent) else 0.0)
        if score > best_score:
            best, best_score = sent, score
    return best, best_score


def _inject(question: dict, mode: str, index: int) -> tuple[str, str | None]:
    """确定性注入决策：none=从不；half=奇数序号题注入；all=除 fault=None 外全部注入。"""
    if question["fault"] is None or mode == "none":
        return question["ok"], None
    if mode == "all":
        return question["fault"][1], question["fault"][0]
    if mode == "half" and index % 2 == 1:
        return question["fault"][1], question["fault"][0]
    return question["ok"], None


_REFUSE_TMPL = ("抱歉，该回答中的数字未能通过金融数据校验，为避免误导已拒绝回答。\n"
                "校验发现：{summary}\n"
                "以下为检索到的原文片段，供您自行核对：\n{snippets}")

def _snippets(docs: list[dict]) -> str:
    return "\n".join(f"· [{d['name']}] {d['text'][:80]}…" for d in docs[:2])


def run_pipeline(question_id: str, fault_mode: str = "none") -> dict:
    """执行完整闭环，返回逐阶段轨迹（每阶段含真实耗时与输入输出）。"""
    idx = next((i for i, q in enumerate(QUESTIONS) if q["id"] == question_id), None)
    if idx is None:
        raise KeyError(question_id)
    question = QUESTIONS[idx]
    stages = []

    # ── 阶段 1：检索（真实 TF-IDF 打分） ──
    t0 = time.perf_counter()
    retrieved = retrieve(question["q"])
    retrieve_ms = (time.perf_counter() - t0) * 1000
    docs = [{"id": r["chunk"]["id"], "name": r["chunk"]["name"],
             "text": r["chunk"]["text"], "as_of": r["chunk"]["as_of"],
             "score": r["score"], "hits": r["hits"]} for r in retrieved]
    stages.append({"stage": "检索", "ms": round(retrieve_ms, 2),
                   "detail": {"query": question["q"],
                              "top_k": [{"name": d["name"], "score": d["score"],
                                         "hits": d["hits"], "as_of": d["as_of"]}
                                        for d in docs]}})

    # ── 阶段 2：生成（确定性回放脑；真实 LLM 可无缝替换此层） ──
    t0 = time.perf_counter()
    answer, fault_type = _inject(question, fault_mode, idx)
    generate_ms = (time.perf_counter() - t0) * 1000
    stages.append({"stage": "生成", "ms": round(generate_ms, 2),
                   "detail": {"answer": answer,
                              "fault_injected": fault_type is not None,
                              "fault_type": fault_type}})

    # ── 阶段 3：校验（真实 verifier，与门禁同一套引擎） ──
    t0 = time.perf_counter()
    from engines.finance import Source, verify_finance_answer
    srcs = [Source(d["name"], d["text"], d["as_of"]) for d in docs]
    v = verify_finance_answer(answer, srcs, now=DEMO_NOW, max_age=MAX_AGE)
    verify_ms = (time.perf_counter() - t0) * 1000
    stages.append({"stage": "校验", "ms": round(verify_ms, 2),
                   "detail": {"verdict": v.verdict, "summary": v.summary(),
                              "findings": [{"kind": f.kind, "severity": f.severity,
                                            "message": f.message} for f in v.findings]}})

    # ── 阶段 4：响应（按裁决分级执行——闭环的"闭环"所在） ──
    t0 = time.perf_counter()
    warn_msgs = "；".join(f.message for f in v.findings if f.severity == "warn")
    if v.verdict == "fail":
        action = "拒答"
        response = _REFUSE_TMPL.format(summary=v.summary(), snippets=_snippets(docs))
    elif v.verdict == "warn":
        action = "放行 + 附数据提示"
        response = answer + "\n\n⚠️ 数据提示：" + warn_msgs
    else:
        action = "直接放行"
        response = answer
    respond_ms = (time.perf_counter() - t0) * 1000
    stages.append({"stage": "响应", "ms": round(respond_ms, 2),
                   "detail": {"action": action, "response": response}})

    return {"question_id": question_id, "question": question["q"],
            "fault_mode": fault_mode, "verdict": v.verdict, "action": action,
            "response": response, "total_ms": round(sum(s["ms"] for s in stages), 2),
            "stages": stages}


def run_batch(fault_mode: str = "none") -> dict:
    """全部预设问题过一遍完整闭环，输出闭环统计——批量验证防线的整体表现。"""
    results = [run_pipeline(q["id"], fault_mode) for q in QUESTIONS]
    by_verdict = {"pass": 0, "warn": 0, "fail": 0}
    for r in results:
        by_verdict[r["verdict"]] += 1
    n = len(results)
    return {"fault_mode": fault_mode, "total": n,
            "verdicts": by_verdict,
            "refuse_rate": round(by_verdict["fail"] / n, 4) if n else 0,
            "results": [{"question_id": r["question_id"], "question": r["question"],
                         "verdict": r["verdict"], "action": r["action"],
                         "total_ms": r["total_ms"]} for r in results]}


# ═══════════════════════ ReAct 决策循环（Agent 本体） ═══════════════════════
# 与四阶段流水线的区别：流水线是固定管道，Agent 是**自主决策循环**——
# 每一步由"脑"选择下一个工具、观察返回结果、再决定下一步；幻觉注入后
# Agent 会通过 verify_numbers 工具**自己发现错误、回到原文修正、复检通过**
# （自校正）——这是工具调用 + 观察驱动 + 自我修正的完整 Agent 能力。

AGENT_TOOLS = ["search_corpus", "get_chunk", "calc", "draft_answer", "verify_numbers", "finalize"]


def run_agent(question_id=None, fault_mode="none", question_text=None):
    """ReAct 决策循环：Thought → Action(工具) → Observation，逐步落轨迹。

    脑为确定性回放脑（mock）；真实 LLM 可无缝替换本函数的决策部分，
    工具集与轨迹结构不变。
    question_text 给定时走自由提问分支：真实检索 + 抽取式生成 + 相关度门槛，
    语料中无相关信息时 Agent 如实告知（不编造）。
    """
    if question_text:
        question = {"id": "custom", "q": question_text, "fault": None, "custom": True}
        custom_mode = True
    else:
        idx = next((i for i, q in enumerate(QUESTIONS) if q["id"] == question_id), None)
        if idx is None:
            raise KeyError(question_id)
        question = QUESTIONS[idx]
        custom_mode = False
    steps = []

    def step(thought, action, action_input, observation, ok=None):
        steps.append({"step": len(steps) + 1, "thought": thought, "action": action,
                      "action_input": action_input, "observation": observation,
                      "ok": ok})

    t0 = time.perf_counter()

    # 1) 检索工具
    retrieved = retrieve(question["q"])
    step(f"用户问：{question['q']}。先调用检索工具，从语料中找相关财务文档",
         "search_corpus", question["q"],
         "命中 " + "；".join(f"{r['chunk']['name']}(score {r['score']})" for r in retrieved))

    # 2) 读文档工具（读前两份：一份可能覆盖不全，交叉印证）
    top1 = retrieved[0]["chunk"]
    top2 = retrieved[1]["chunk"] if len(retrieved) > 1 else top1
    step(f"得分最高的是【{top1['name']}】，查看全文定位数字依据",
         "get_chunk", top1["id"], top1["text"])
    if top2 is not top1:
        step(f"再读【{top2['name']}】交叉印证，防止单一来源遗漏",
             "get_chunk", top2["id"], top2["text"])

    # 3) 计算工具（仅配置了 calc 的问题）
    if question.get("calc"):
        cfg = question["calc"]
        step("原文给出了分子与分母，用安全计算器复算比率，避免心算出错",
             "calc", cfg["expr"], f"= {cfg['result']}（{cfg['label']}）")

    from engines.finance.numeric import extract_numbers
    # 4) 草拟回答。自定义问题走抽取式生成：从最相关文档中提取与问题最匹配的子句，
    #    并按档位确定性注入幻觉（首个非年份数字 ×10，数量级篡改）——注入后 Agent
    #    会被自己的校验工具打回，触发自校正循环。
    if custom_mode:
        if retrieved[0]["score"] < MIN_REL_SCORE:
            step("检索得分低于相关度门槛——语料中没有能回答该问题的信息",
                 "search_corpus", question["q"],
                 f"最高分 {retrieved[0]['score']} < 门槛 {MIN_REL_SCORE}，无相关文档")
            honest = ("抱歉，当前语料库中未找到与该问题相关的财务信息"
                      "（语料范围为 2023/2024 年报、行业研究与相关报道），无法回答该问题。")
            step("Agent 如实告知无法回答，而不是编造一个答案",
                 "finalize", "", honest)
            total_ms = (time.perf_counter() - t0) * 1000
            return {"question_id": "custom", "question": question["q"],
                    "fault_mode": fault_mode, "verdict": "no_answer",
                    "self_corrected": False, "fault_type": None,
                    "answer": honest, "tool_calls": 1,
                    "total_ms": round(total_ms, 2), "steps": steps}
        top_sent, _ = _best_sentence(top1, question["q"])
        draft = f"根据《{top1['name']}》：{top_sent}。"
        fault_type = None
        if fault_mode != "none":
            want = fault_mode == "all" or sum(ord(c) for c in question["q"]) % 2 == 1
            if want:
                cand = [t for t in extract_numbers(draft)
                        if not (t.unit == "COUNT" and 1900 <= t.value <= 2100)]
                if cand:
                    t0n = cand[0]
                    m = re.match(r"([\d,\.]+)(.*)", t0n.raw)
                    if m:
                        scaled = float(m.group(1).replace(",", "")) * 10
                        fmt = str(int(scaled)) if scaled == int(scaled) else (f"{scaled:.1f}")
                        draft = draft.replace(t0n.raw, fmt + m.group(2), 1)
                        fault_type = "数量级篡改"
    else:
        draft, fault_type = _inject(question, fault_mode, idx)
    note = "（本次生成被注入了幻觉数字，看 Agent 能否自己发现）" if fault_type else ""
    step("基于原文证据组织回答" + note,
         "draft_answer", "", draft)

    # 5) 自检工具（校验依据 = Agent 实际读过的文档，而非全语料——它只引用自己看过的）
    from engines.finance import Source, verify_finance_answer
    srcs = [Source(top1["name"], top1["text"], top1["as_of"]),
            Source(top2["name"], top2["text"], top2["as_of"])]
    v = verify_finance_answer(draft, srcs, now=DEMO_NOW, max_age=MAX_AGE)
    step("回答里有数字断言，调用数值校验工具自检",
         "verify_numbers", draft[:40] + ("…" if len(draft) > 40 else ""),
         f"裁决 {v.verdict}：{v.summary()}", ok=(not v.failed))

    # 6) 观察驱动：校验失败 → 自校正
    self_corrected = False
    final_answer, final_v = draft, v
    if v.failed and fault_type:
        self_corrected = True
        step(f"校验未通过——{fault_type}的数字没有来源支撑。回到原文重新核对正确数值",
             "get_chunk", top1["id"], "原文关键句：" + top1["text"])
        if custom_mode:
            step("重新从原文提取证据子句，修正数字", "draft_answer", "", top_sent)
            final_answer = f"根据《{top1['name']}》：{top_sent}。"
        else:
            if question.get("calc"):
                cfg = question["calc"]
                step("用计算器按原文数字重新复算，得到正确比率",
                     "calc", cfg["expr"], f"= {cfg['result']}")
            final_answer = question["ok"]
            step("用修正后的数字重新草拟回答", "draft_answer", "", final_answer)
        final_v = verify_finance_answer(final_answer, srcs, now=DEMO_NOW, max_age=MAX_AGE)
        step("修正后再次调用校验工具确认", "verify_numbers", final_answer[:40],
             f"裁决 {final_v.verdict}：{final_v.summary()}", ok=(not final_v.failed))

    # 7) 收尾：附时效提示（warn 时 Agent 主动声明）
    warn_msgs = "；".join(f.message for f in final_v.findings if f.severity == "warn")
    if final_v.verdict == "warn":
        final_answer = final_answer + "\n\n⚠️ 数据提示：" + warn_msgs
    action_note = "校验通过，输出最终回答" + ("（含数据时效提示）" if final_v.verdict == "warn" else "")
    step(action_note, "finalize", "", final_answer)

    total_ms = (time.perf_counter() - t0) * 1000
    return {"question_id": question_id, "question": question["q"],
            "fault_mode": fault_mode, "verdict": final_v.verdict,
            "self_corrected": self_corrected, "fault_type": fault_type,
            "answer": final_answer, "tool_calls": sum(1 for s in steps if s["action"] != "finalize"),
            "total_ms": round(total_ms, 2), "steps": steps}


def run_agent_batch(fault_mode: str = "none") -> dict:
    """全部问题过 ReAct 循环，统计自校正拦截表现。"""
    results = [run_agent(q["id"], fault_mode) for q in QUESTIONS]
    corrected = sum(1 for r in results if r["self_corrected"])
    by_verdict = {"pass": 0, "warn": 0, "fail": 0}
    for r in results:
        by_verdict[r["verdict"]] += 1
    n = len(results)
    return {"fault_mode": fault_mode, "total": n, "verdicts": by_verdict,
            "self_corrected": corrected,
            "results": [{"question_id": r["question_id"], "question": r["question"],
                         "verdict": r["verdict"], "self_corrected": r["self_corrected"],
                         "fault_type": r["fault_type"], "steps": len(r["steps"]),
                         "tool_calls": r["tool_calls"], "total_ms": r["total_ms"]}
                        for r in results]}
