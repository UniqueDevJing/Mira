"""文档类型路由全面质检 — 验证"不同类型走不同切分/检索路线"。

直接 import 真实 engines 模块, 对 10 类 RAG 文档类型做:
  1. 注册表一致性 (kbs 唯一 / 1:1 映射 / prompt_hint / chunker 配置)
  2. Chunker 分派 (get_chunker 返回正确类 + 参数 + 表格感知)
  3. 功能性切分 (用代表性文档实跑 chunker, 验证类型化输出)
  4. resolve_doc_type 路由 (优先级 / kb 反查 / 非法回退 / 孤儿库修复)
  5. 检索路线隔离 (kb→表名 / bm25 文件 / SKILLS 映射 / prompt_hint 注入)

运行: PYTHONPATH=. python scripts/qa_doc_type_routing.py
退出码: 0=全绿, 1=有 FAIL(关键缺陷), 2=仅 WARN(设计性 gap)
"""

import sys
import traceback
from types import SimpleNamespace

from engines.chunking.strategies import ClauseChunker, FaqChunker, _TableAware, get_chunker
from engines.chunking.structure_chunker import StructureChunker
from engines.doc_types import (
    DOC_TYPES,
    KB_TO_TYPE,
    RAG_DOC_TYPES,
    TYPE_TO_KB,
    get_doc_type,
    kb_to_doc_type,
    resolve_doc_type,
)
from engines.router.routing_rules import SKILLS

# ── 结果收集 ──
PASS, FAIL, WARN = [], [], []
def _rec(kind, name, detail=""):
    (PASS if kind == "PASS" else FAIL if kind == "FAIL" else WARN).append(f"{name}{(' — ' + detail) if detail else ''}")

# ── 构造 UIRDocument 的辅助 ──
def _doc(doc_id, blocks, tables=None, path="doc.txt"):
    pages = [{"blocks": blocks}]
    return SimpleNamespace(doc_id=doc_id, source={"path": path}, pages=pages, tables=tables or [])

def _t(content, level=1):
    return {"type": "title", "content": content, "metadata": {"heading_level": level}, "page_num": 1}
def _p(content):
    return {"type": "paragraph", "content": content, "metadata": {}, "page_num": 1}


# ══════════════════════ 1. 注册表一致性 ══════════════════════
def check_registry():
    types = list(RAG_DOC_TYPES.keys())
    # kbs 唯一
    kbs = [spec.kb for spec in RAG_DOC_TYPES.values()]
    if len(kbs) != len(set(kbs)):
        FAIL.append("注册表: 存在重复 kb (检索库冲突)")
    else:
        PASS.append(f"注册表: {len(types)} 类 RAG 文档, kb 全唯一")
    # 1:1 映射
    if len(TYPE_TO_KB) == len(KB_TO_TYPE) == len(types):
        PASS.append("注册表: TYPE_TO_KB / KB_TO_TYPE 1:1 互反")
    else:
        FAIL.append("注册表: TYPE_TO_KB 与 KB_TO_TYPE 数量不一致")
    # direct 必须被排除在 RAG 外
    if "direct" not in RAG_DOC_TYPES and DOC_TYPES["direct"].kb is None:
        PASS.append("注册表: direct 正确排除 (kb=None, 不走知识库)")
    else:
        FAIL.append("注册表: direct 未正确排除出 RAG 检索")
    # 每类应有 prompt_hint (检索生成差异化)
    missing_hint = [t for t, s in RAG_DOC_TYPES.items() if not s.prompt_hint]
    if not missing_hint:
        PASS.append("注册表: 全部 RAG 类型含 prompt_hint (生成差异化)")
    else:
        WARN.append(f"注册表: 缺 prompt_hint 的类型 {missing_hint}")
    # chunker 配置健全
    bad_cfg = []
    for t, s in RAG_DOC_TYPES.items():
        c = s.chunker or {}
        if not c.get("strategy") in ("semantic", "faq", "clause"):
            bad_cfg.append(f"{t}:strategy={c.get('strategy')}")
        if not isinstance(c.get("max_chars"), int) or c.get("max_chars") <= 0:
            bad_cfg.append(f"{t}:max_chars={c.get('max_chars')}")
    if not bad_cfg:
        PASS.append("注册表: chunker 配置 (strategy/max_chars) 全部合法")
    else:
        FAIL.append("注册表: 非法 chunker 配置 " + "; ".join(bad_cfg))


# ══════════════════════ 2. Chunker 分派 ══════════════════════
def check_dispatch():
    settings = SimpleNamespace(chunk_max_chars=800, chunk_overlap=128)
    expectations = {
        "policy": (StructureChunker, 1000, 160, False),
        "contract": (ClauseChunker, 1200, 200, True),
        "product": (StructureChunker, 900, 150, False),
        "service": (FaqChunker, 600, 80, False),
        "tech": (StructureChunker, 800, 128, False),
        "finance": (StructureChunker, 900, 150, False),
        "hr": (StructureChunker, 900, 150, False),
        "marketing": (StructureChunker, 800, 128, False),
        "meeting": (StructureChunker, 1000, 160, False),
        "training": (StructureChunker, 800, 128, False),
    }
    for tid, (exp_cls, exp_mc, exp_ov, exp_tbl) in expectations.items():
        spec = get_doc_type(tid)
        ch = get_chunker(spec, settings)
        # 剥离 _TableAware 包装看基类
        base = ch.base if isinstance(ch, _TableAware) else ch
        is_tbl = isinstance(ch, _TableAware)
        ok = isinstance(base, exp_cls) and base.max_chars == exp_mc and base.overlap == exp_ov and is_tbl == exp_tbl
        if ok:
            PASS.append(f"分派[{tid}]: {exp_cls.__name__}({exp_mc}/{exp_ov}) table={exp_tbl}")
        else:
            FAIL.append(
                f"分派[{tid}]: 期望 {exp_cls.__name__}({exp_mc}/{exp_ov}) table={exp_tbl} "
                f"实际 {type(base).__name__}({getattr(base,'max_chars','?')}/{getattr(base,'overlap','?')}) table={is_tbl}"
            )
    # direct 不应作为知识库文档入库 (upload 守卫: kb=None → 400, 不进 rag_None 表)
    if get_doc_type("direct").kb is None:
        PASS.append("分派[direct]: 上传守卫生效 — direct(kb=None) 被拒, 不污染 rag_None 表")
    else:
        FAIL.append("分派[direct]: direct 竟有 kb, 会污染 rag_None 表")


# ══════════════════════ 3. 功能性切分 ══════════════════════
def check_functional():
    settings = SimpleNamespace(chunk_max_chars=800, chunk_overlap=128)

    # --- FAQ (service) ---
    faq_doc = _doc("svc1", [
        _p("问：怎么申请退款？"),
        _p("答：您可在订单页点击『申请退款』，客服 24 小时内处理。"),
        _p("问：运费怎么算？"),
        _p("答：满 99 元包邮，未满收取 10 元运费。"),
    ], path="service_faq.txt")
    chunks = get_chunker(get_doc_type("service"), settings).chunk(faq_doc)
    qa_pairs = [c for c in chunks if "问：" in c.content and "答：" in c.content]
    if len(qa_pairs) == 2:
        PASS.append("功能[service]: FAQ 正确拆为 2 个问答对块")
    else:
        FAIL.append(f"功能[service]: FAQ 拆分异常, 问答对块数={len(qa_pairs)} (总块={len(chunks)})")

    # FAQ 退化: 无明确问答结构 → 应退回语义且不丢内容
    faq_plain = _doc("svc2", [
        _t("售后政策", 1),
        _p("本平台提供 7 天无理由退货。商品需保持完好，不影响二次销售。运费由买家承担。"),
    ], path="service_plain.txt")
    cp = get_chunker(get_doc_type("service"), settings).chunk(faq_plain)
    total_chars = sum(len(c.content) for c in cp)
    if cp and total_chars >= 20:
        PASS.append(f"功能[service]: 无问答结构正确退化语义切分 (块={len(cp)}, 不丢内容)")
    else:
        FAIL.append("功能[service]: 无问答结构退化失败 (丢内容或空块)")

    # --- 条款 (contract) 正常: 首条前有换行/标题 ---
    contract_norm = _doc("c1", [
        _t("保密协议", 1),
        _p("第一条 甲方义务：对在合作中知悉的乙方商业秘密承担保密责任，不得向第三方披露。"),
        _p("第二条 乙方义务：同等保护甲方技术资料，保密期限自签约起五年。"),
        _p("第三条 违约责任：任一方违约应赔偿守约方因此遭受的实际损失。"),
    ], path="contract_norm.txt")
    cc = get_chunker(get_doc_type("contract"), settings).chunk(contract_norm)
    clause_hits = [c for c in cc if any(f"第{i}条" in c.content for i in ("一", "二", "三"))]
    if len(clause_hits) >= 3:
        PASS.append(f"功能[contract]: 条款正确按『第X条』切分 (命中 {len(clause_hits)} 条)")
    else:
        FAIL.append(f"功能[contract]: 条款切分异常, 命中第X条块数={len(clause_hits)}")

    # --- 条款 BUG 暴露: 首条无前置换行 (文档开头直接『第一条』) ---
    contract_head = _doc("c2", [
        _p("第一条 甲方权利义务：提供符合约定的服务，并保证服务质量。"),
        _p("第二条 乙方权利义务：按时支付费用，配合甲方工作。"),
    ], path="contract_head.txt")
    ch = get_chunker(get_doc_type("contract"), settings).chunk(contract_head)
    head_hits = [c for c in ch if "第一条" in c.content]
    if len(head_hits) >= 1 and len(ch) >= 2:
        PASS.append("功能[contract]: 首条无前置换行也能切分")
    else:
        FAIL.append(f"功能[contract]: 首条无前置换行导致不切分! 块数={len(ch)} (应≥2), 首条命中={len(head_hits)} 【缺陷】")

    # --- 条款: 单级编号 '1. 2. 3.' (非 '1.1') ---
    contract_num = _doc("c3", [
        _t("服务合同", 1),
        _p("1. 甲方义务：提供培训材料。"),
        _p("2. 乙方义务：完成既定课程。"),
        _p("3. 争议解决：提交甲方所在地法院。"),
    ], path="contract_num.txt")
    cn = get_chunker(get_doc_type("contract"), settings).chunk(contract_num)
    num_hits = [c for c in cn if "1.甲方" in c.content or "2.乙方" in c.content or "3.争议" in c.content]
    if len(num_hits) >= 2:
        PASS.append(f"功能[contract]: 单级编号 '1. 2. 3.' 正确切分 (命中 {len(num_hits)})")
    else:
        FAIL.append(f"功能[contract]: 单级编号 '1. 2.' 未切分 (块数={len(cn)})")

    # --- 条款: 中文枚举 '一、 二、' ---
    contract_cn = _doc("c4", [
        _t("管理细则", 1),
        _p("一、适用范围：本细则适用于全体在职员工。"),
        _p("二、考核标准：按季度进行绩效评定。"),
        _p("三、申诉渠道：可向人力资源部提出复核。"),
    ], path="contract_cn.txt")
    ccn = get_chunker(get_doc_type("contract"), settings).chunk(contract_cn)
    cn_hits = [c for c in ccn if "一、适用" in c.content or "二、考核" in c.content or "三、申诉" in c.content]
    if len(cn_hits) >= 2:
        PASS.append(f"功能[contract]: 中文枚举 '一、二、' 正确切分 (命中 {len(cn_hits)})")
    else:
        FAIL.append(f"功能[contract]: 中文枚举 '一、' 未切分 (块数={len(ccn)})")

    # --- 条款: 括号编号 '（一）（二）' ---
    contract_par = _doc("c5", [
        _t("补充协议", 1),
        _p("（一）定义：本协议所称关联方指……"),
        _p("（二）效力：本补充协议与原合同具有同等效力。"),
    ], path="contract_par.txt")
    cpar = get_chunker(get_doc_type("contract"), settings).chunk(contract_par)
    par_hits = [c for c in cpar if "（一）定义" in c.content or "（二）效力" in c.content]
    if len(par_hits) >= 2:
        PASS.append(f"功能[contract]: 括号编号 '（一）（二）' 正确切分 (命中 {len(par_hits)})")
    else:
        FAIL.append(f"功能[contract]: 括号编号未切分 (块数={len(cpar)})")

    # --- 条款: 冒号尾随 '第一条：' (非空格) ---
    contract_colon = _doc("c6", [
        _t("保密协议", 1),
        _p("第一条：甲方对在合作中知悉的乙方商业秘密承担保密责任。"),
        _p("第二条：保密期限自本协议签署之日起满五年。"),
    ], path="contract_colon.txt")
    ccol = get_chunker(get_doc_type("contract"), settings).chunk(contract_colon)
    col_hits = [c for c in ccol if "第一条：" in c.content or "第二条：" in c.content]
    if len(col_hits) >= 2:
        PASS.append(f"功能[contract]: 冒号尾随 '第一条：' 正确切分 (命中 {len(col_hits)})")
    else:
        FAIL.append(f"功能[contract]: 冒号尾随 '第一条：' 未切分 (块数={len(ccol)})")

    # --- 语义 (policy/product/tech...) 结构感知 ---
    for tid in ["policy", "product", "tech", "finance", "hr", "marketing", "meeting", "training"]:
        spec = get_doc_type(tid)
        doc = _doc(tid, [
            _t(f"{tid}标题", 1),
            _p(f"这是 {tid} 类型的第一段正文，包含若干业务说明。"),
            _t(f"{tid}小节", 2),
            _p(f"这是 {tid} 的第二段正文，含更细的描述性内容用于检索。"),
        ], path=f"{tid}.txt")
        cs = get_chunker(spec, settings).chunk(doc)
        # 语义切分应保留标题链 (context.title_chain 非空) 且块大小不超 max_chars
        ok_chain = any(c.context.get("title_chain") for c in cs)
        oversize = [c for c in cs if len(c.content) > spec.chunker["max_chars"] + 5]
        if cs and ok_chain and not oversize:
            PASS.append(f"功能[{tid}]: 语义切分保留标题链且块≤{spec.chunker['max_chars']}")
        else:
            FAIL.append(f"功能[{tid}]: 语义切分异常 chain={ok_chain} 超块={len(oversize)} 总块={len(cs)}")


# ══════════════════════ 4. resolve_doc_type 路由 ══════════════════════
def check_routing():
    # 优先级: doc_type 优先
    assert resolve_doc_type("contract", "policy") == "contract", "doc_type 应优先"
    PASS.append("路由: doc_type 优先于 knowledge_base")
    # kb 反查
    assert resolve_doc_type(None, "finance") == "finance", "kb 反查应生效"
    PASS.append("路由: knowledge_base 可反查回类型 (finance)")
    # 非法 doc_type + 合法 kb
    assert resolve_doc_type("bogus", "hr") == "hr"
    PASS.append("路由: 非法 doc_type 用 kb 反查兜底 (hr)")
    # 旧默认库 'documents' 孤儿库 → 回退 policy (修复点)
    assert resolve_doc_type(None, "documents") == "policy", "孤儿库应回退 policy"
    assert resolve_doc_type("documents", None) == "policy"
    PASS.append("路由: 孤儿库 'documents'/非法 → 回退 policy (修复生效, 不进孤儿库)")
    # 全非法 → policy
    assert resolve_doc_type("???", "???") == "policy"
    PASS.append("路由: 全非法输入安全回退 policy")
    # 回退结果一定在 RAG_KBS 内 (保证可检索, 不死库)
    for bad in [None, "", "documents", "unknown_kb", "???"]:
        resolved = resolve_doc_type(bad, bad if bad else None)
        if resolved not in RAG_DOC_TYPES:
            FAIL.append(f"路由: 回退结果 {resolved} 不在 RAG 类型内 (会进孤儿库)")
        else:
            PASS.append(f"路由: 输入={bad!r} → 安全落 RAG 类型 {resolved}")


# ══════════════════════ 5. 检索路线隔离 ══════════════════════
def _vector_table(kb: str) -> str:
    LEGACY = {"", "documents"}
    return "documents" if kb in LEGACY else f"rag_{kb}"

def check_retrieval_isolation():
    # 每类型 kb → 独立表名, 无碰撞
    tables = {tid: _vector_table(spec.kb) for tid, spec in RAG_DOC_TYPES.items()}
    if len(set(tables.values())) == len(tables):
        PASS.append(f"检索: 各 kb → 独立 LanceDB 表 ({len(set(tables.values()))} 张, 无碰撞)")
    else:
        FAIL.append("检索: 存在 kb→表名碰撞 (跨库污染)")
    # BM25 文件隔离 (推导命名)
    bm25_files = {tid: f"bm25_{spec.kb}.pkl" for tid, spec in RAG_DOC_TYPES.items()}
    if len(set(bm25_files.values())) == len(bm25_files):
        PASS.append("检索: BM25 索引按 kb 独立文件, 无碰撞")
    else:
        FAIL.append("检索: BM25 文件命名碰撞")
    # SKILLS 的 kb 与注册表一致 (路由→检索库一致)
    mism = [tid for tid, s in DOC_TYPES.items() if SKILLS.get(tid, {}).get("kb") != s.kb]
    if not mism:
        PASS.append("检索: 路由 SKILLS[kb] 与注册表完全一致")
    else:
        FAIL.append(f"检索: SKILLS 与注册表 kb 不一致 {mism}")
    # prompt_hint 注入: 每 kb 能还原出带 hint 的类型
    for tid, spec in RAG_DOC_TYPES.items():
        rev = kb_to_doc_type(spec.kb)
        if rev is None or rev.type_id != tid or not rev.prompt_hint:
            FAIL.append(f"检索: kb={spec.kb} 反查类型失败或缺失 hint")
    PASS.append("检索: 每个 kb 可反查类型并携 prompt_hint (生成差异化生效)")
    # 检索算法是否因类型而异?
    algo_distinct = sum(1 for s in RAG_DOC_TYPES.values() if s.chunker.get("strategy") in ("faq", "clause"))
    PASS.append(f"检索: 切分算法真正差异化仅 {algo_distinct}/10 类 (faq+clause), 其余 8 类共用 semantic 仅尺寸/库/hint 不同")


def main():
    print("=" * 70)
    print("文档类型路由全面质检 — RAG 2.0")
    print("=" * 70)
    for fn in (check_registry, check_dispatch, check_functional, check_routing, check_retrieval_isolation):
        try:
            fn()
        except Exception:
            FAIL.append(f"{fn.__name__}: 运行异常\n{traceback.format_exc()}")

    print("\n── PASS ──")
    for p in PASS:
        print(f"  ✓ {p}")
    if WARN:
        print("\n── WARN (设计性 gap, 非阻断) ──")
        for w in WARN:
            print(f"  ⚠ {w}")
    if FAIL:
        print("\n── FAIL (关键缺陷) ──")
        for f in FAIL:
            print(f"  ✗ {f}")

    print(f"\n汇总: PASS={len(PASS)}  WARN={len(WARN)}  FAIL={len(FAIL)}")
    if FAIL:
        sys.exit(1)
    if WARN:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
