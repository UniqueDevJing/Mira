"""导出 Cross-Encoder 为 ONNX 并做 INT8 动态量化, 加速 CPU 上的 rerank 推理。

背景: bge-reranker-base (base 级) 在 CPU 上重排 10 个 512 字候选约 1.1s, 是 QA 链路最大单点成本。
ONNX Runtime + INT8 动态量化通常带来 2~4× 加速, 精度损失可忽略。

设计原则 (不写死):
  - 模型输入由 model.forward 签名动态推导, 不假设架构 (bert/xlm-roberta/roberta 均可)
  - 路径走 api.state.resolve_model_path, 与生产加载路径一致
  - 导出后**强制做精度校验**: ONNX 输出必须与 PyTorch 对齐, 超阈值直接失败退出, 不产出坏模型

用法:
  python scripts/export_reranker_onnx.py                      # 用 settings.reranker_model
  python scripts/export_reranker_onnx.py --model BAAI/bge-reranker-base --out models/bge-reranker-base-onnx
  python scripts/export_reranker_onnx.py --no-int8            # 只导出 fp32, 不做量化
  python scripts/export_reranker_onnx.py --max-diff 0.02      # 自定义精度校验阈值
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.config import settings
from api.state import resolve_model_path

# 校验用的样本文本: 覆盖中英混排 / 长文本, 确保导出路径与实际输入分布一致
_SAMPLE_PAIRS = [
    ("这是什么产品的保修期限?", "本产品自购买之日起享受三年保修服务, 人为损坏不在保修范围内。"),
    ("How long is the warranty?", "The warranty period is 3 years from the date of purchase."),
    ("合同违约金如何计算?",
     "甲方逾期交付的, 每逾期一日按合同总价款的千分之五向乙方支付违约金, 累计不超过合同总价款的百分之十。"),
    ("报销流程需要哪些材料?", "员工报销需提交发票原件、费用明细表及审批单, 由部门负责人签字后交财务审核。"),
    ("", "空查询边界用例: 验证空字符串不会导致导出或推理崩溃。"),
]


def _forward_input_names(model) -> list[str]:
    """从 model.forward 签名推导需要的输入名, 只保留我们支持的张量输入。"""
    supported = ("input_ids", "attention_mask", "token_type_ids")
    try:
        params = list(inspect.signature(model.forward).parameters)
    except (TypeError, ValueError):
        return ["input_ids", "attention_mask"]
    names = [p for p in params if p in supported]
    return names or ["input_ids", "attention_mask"]


def _torch_scores(ce, pairs: list[tuple[str, str]]):
    import numpy as np

    return np.asarray(ce.predict(pairs), dtype=np.float64)


def _detect_activation(ce) -> str:
    """推导 CrossEncoder 输出激活函数。

    sentence-transformers 的 CrossEncoder.predict() 并非直接返回 logits: 单标签模型
    (num_labels==1) 默认套 Sigmoid。这是**非线性**变换, 而 rerank 融合用的是 min-max 归一化
    后的分数 —— min-max(sigmoid(x)) != min-max(x)。所以 ONNX 后端必须施加相同激活,
    否则融合权重会微妙偏移。激活类型从模型对象推导, 不硬编码。
    """
    fn = getattr(ce, "default_activation_function", None)
    name = type(fn).__name__.lower() if fn is not None else ""
    if "sigmoid" in name:
        return "sigmoid"
    if "tanh" in name:
        return "tanh"
    return "identity"


def _apply_activation(x, activation: str):
    import numpy as np

    if activation == "sigmoid":
        return 1.0 / (1.0 + np.exp(-x))
    if activation == "tanh":
        return np.tanh(x)
    return x


def _onnx_scores(path: str, tok, pairs: list[tuple[str, str]], input_names: list[str],
                 activation: str = "identity"):
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    enc = tok(list(pairs), padding=True, truncation=True, max_length=tok.model_max_length, return_tensors="np")
    feed = {}
    for name in input_names:
        if name in enc:
            feed[name] = enc[name].astype("int64")
        elif name == "token_type_ids":
            feed[name] = np.zeros_like(enc["input_ids"], dtype="int64")
    out = sess.run(None, feed)[0]
    out = np.asarray(out)
    # num_labels==1 时输出是 (B,1), 取最后一维; 多分类时取正类列
    logits = out.reshape(len(pairs), -1)[:, -1].astype(np.float64)
    return _apply_activation(logits, activation)


def _compare(tag: str, a, b, max_diff: float) -> bool:
    import numpy as np

    absd = np.abs(a - b)
    # 分数量纲敏感度低, 用相对序一致性(更贴近 rerank 实际需求) + 绝对误差双判据
    order_ok = bool(np.all(np.argsort(-a) == np.argsort(-b)))
    ok = bool(absd.max() <= max_diff) and order_ok
    print(f"  [{tag}] 最大绝对误差={absd.max():.6f} (阈值 {max_diff})   排序一致={order_ok}   -> {'✅' if ok else '❌'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=settings.reranker_model)
    ap.add_argument("--out", default="", help="输出目录, 默认 <model>-onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-int8", action="store_true", help="跳过 INT8 量化, 只导出 fp32")
    ap.add_argument("--max-diff", type=float, default=0.05, help="ONNX 与 PyTorch 输出的最大绝对误差阈值")
    ap.add_argument("--int8-max-diff", type=float, default=0.30, help="INT8 与 fp32 ONNX 的最大绝对误差阈值")
    args = ap.parse_args()

    import torch

    src = resolve_model_path(args.model)
    out_dir = args.out or (src.rstrip("/\\") + "-onnx")
    print(f"[export] 源模型: {src}")
    print(f"[export] 输出目录: {out_dir}")

    from sentence_transformers import CrossEncoder

    t0 = time.time()
    ce = CrossEncoder(src)
    model = ce.model
    tok = ce.tokenizer
    model.eval()
    print(f"[export] PyTorch 模型加载完成 ({time.time()-t0:.1f}s)")

    input_names = _forward_input_names(model)
    activation = _detect_activation(ce)
    print(f"[export] 推导出的输入: {input_names}")
    print(f"[export] 输出激活函数: {activation}  (须与 CrossEncoder.predict 一致, 否则融合权重会偏移)")

    # ---- 1) 导出 fp32 ONNX ----
    os.makedirs(out_dir, exist_ok=True)
    fp32_path = os.path.join(out_dir, "model.onnx")
    enc = tok(["query"], ["document"], padding=True, truncation=True,
              max_length=tok.model_max_length, return_tensors="pt")
    dummy = tuple(enc[n].to(torch.int64) if n in enc else torch.zeros_like(enc["input_ids"])
                  for n in input_names)
    seq_axes = {n: {0: "batch", 1: "sequence"} for n in input_names}

    print("[export] 导出 fp32 ONNX ...")
    t0 = time.time()
    try:
        torch.onnx.export(
            model, dummy, fp32_path,
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dict(seq_axes, logits={0: "batch"}),
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,  # 传统 TorchScript 导出器对 BERT 类模型更稳
        )
    except TypeError:
        # 旧版 torch 无 dynamo 参数
        torch.onnx.export(
            model, dummy, fp32_path,
            input_names=input_names, output_names=["logits"],
            dynamic_axes=dict(seq_axes, logits={0: "batch"}),
            opset_version=args.opset, do_constant_folding=True,
        )
    print(f"[export] fp32 导出完成 ({time.time()-t0:.1f}s) -> {fp32_path} "
          f"({os.path.getsize(fp32_path)/1e6:.0f} MB)")

    # ---- 2) 精度校验: ONNX vs PyTorch ----
    print("[verify] 对比 ONNX 与 PyTorch 输出 ...")
    ref = _torch_scores(ce, _SAMPLE_PAIRS)
    got = _onnx_scores(fp32_path, tok, _SAMPLE_PAIRS, input_names, activation)
    if not _compare("onnx-fp32", ref, got, args.max_diff):
        print("[export] ❌ fp32 ONNX 精度不达标, 中止 (不产出不可用模型)")
        sys.exit(1)

    results = {"fp32_path": fp32_path, "input_names": input_names, "activation": activation}

    # ---- 3) INT8 动态量化 ----
    if not args.no_int8:
        print("[export] INT8 动态量化 ...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("[export] ⚠️ 未安装 onnxruntime.quantization, 跳过量化 (fp32 仍可用)")
        else:
            int8_path = os.path.join(out_dir, "model_int8.onnx")
            t0 = time.time()
            # extra_options 关闭 MatMul 常量折叠可提升量化后精度稳定性
            quantize_dynamic(
                model_input=fp32_path, model_output=int8_path, weight_type=QuantType.QInt8,
                extra_options={"EnableSubgraph": False},
            )
            print(f"[export] INT8 量化完成 ({time.time()-t0:.1f}s) -> {int8_path} "
                  f"({os.path.getsize(int8_path)/1e6:.0f} MB, "
                  f"{os.path.getsize(int8_path)/os.path.getsize(fp32_path)*100:.0f}% of fp32)")

            print("[verify] 对比 INT8 与 fp32 ONNX 输出 ...")
            fp32_ref = got
            got8 = _onnx_scores(int8_path, tok, _SAMPLE_PAIRS, input_names, activation)
            if _compare("onnx-int8", fp32_ref, got8, args.int8_max_diff):
                results["int8_path"] = int8_path
            else:
                print("[export] ⚠️ INT8 精度不达标, 丢弃量化模型 (保留 fp32)")

    # ---- 4) 拷贝 tokenizer, 使 ONNX 目录自包含 ----
    for fn in ("config.json", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "sentencepiece.bpe.model", "vocab.txt"):
        p = os.path.join(src, fn)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, fn))

    with open(os.path.join(out_dir, "export_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "source_model": args.model,
            "source_path": src,
            "opset": args.opset,
            "input_names": input_names,
            "max_length": int(tok.model_max_length),
            "activation": activation,
            "files": {k: os.path.basename(v) for k, v in results.items() if k.endswith("_path")},
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[export] ✅ 完成 -> {out_dir}")
    for k, v in results.items():
        if k.endswith("_path"):
            print(f"  {k}: {v}")

    # ---- 5) 延迟对比 (给个直观结论) ----
    print("\n[bench] 延迟对比 (10 候选 x 512 字, n=5) ...")
    pairs = [("这是一个用于基准测试的查询句子", f"这是第 {i} 篇候选文档的内容, 包含若干细节信息用于测量推理延迟。" * 20)
             for i in range(10)]

    def bench(fn, n=5):
        for _ in range(2):
            fn()
        ts = []
        for _ in range(n):
            t = time.time()
            fn()
            ts.append(time.time() - t)
        return sum(ts) / len(ts) * 1000

    torch_ms = bench(lambda: ce.predict(pairs))
    print(f"  PyTorch      : {torch_ms:.0f}ms")
    for key in ("fp32_path", "int8_path"):
        if key in results:
            p = results[key]
            ms = bench(lambda p=p: _onnx_scores(p, tok, pairs, input_names, activation))
            print(f"  {key[:-5]:<12}: {ms:.0f}ms   ({torch_ms/ms:.2f}× 加速)")
    print("\n提示: 用 scripts/bench_rerank_backends.py 可做更严谨的多轮基准。")


if __name__ == "__main__":
    main()
