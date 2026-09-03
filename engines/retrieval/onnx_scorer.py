"""ONNX Runtime 版 Cross-Encoder — 加速 CPU 上的 rerank 推理。

接口对齐 sentence_transformers.CrossEncoder 的 predict(pairs), 因此 Reranker 无需改动
即可在 PyTorch / ONNX 两种后端间切换 (见 api/config.py: reranker_backend)。

模型由 scripts/export_reranker_onnx.py 导出, 目录内至少包含:
    model.onnx        或  model_int8.onnx   (INT8 动态量化版)
    tokenizer 相关文件                        (使目录自包含)
    export_meta.json                          (可选, 记录导出时的输入名等信息)

设计原则 (不写死):
  - 输入张量名从 ONNX 图的实际输入推导, 不假设 input_ids/attention_mask
  - 目录/精度(int8 与否)由构造参数决定, 不硬编码路径
  - 加载失败一律抛异常, 由上层 Reranker 决定降级 (不在本模块静默兜底)
"""
from __future__ import annotations

import json
import logging
import os
import threading

logger = logging.getLogger(__name__)

_FP32 = "model.onnx"
_INT8 = "model_int8.onnx"


class OnnxCrossEncoder:
    """用 ONNX Runtime 跑 Cross-Encoder 打分。"""

    def __init__(self, model_dir: str, prefer_int8: bool = True, max_length: int | None = None,
                 providers: list[str] | None = None, intra_op_threads: int = 0,
                 activation: str | None = None):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        if not os.path.isdir(model_dir):
            msg = f"ONNX 模型目录不存在: {model_dir}"
            raise FileNotFoundError(msg)

        path = None
        if prefer_int8:
            p = os.path.join(model_dir, _INT8)
            if os.path.exists(p):
                path = p
        if path is None:
            p = os.path.join(model_dir, _FP32)
            if os.path.exists(p):
                path = p
        if path is None:
            msg = f"ONNX 模型文件缺失: {model_dir} 内应有 {_FP32} 或 {_INT8}"
            raise FileNotFoundError(msg)

        so = ort.SessionOptions()
        if intra_op_threads > 0:
            so.intra_op_num_threads = intra_op_threads
        self.session = ort.InferenceSession(
            path, sess_options=so, providers=providers or ["CPUExecutionProvider"]
        )
        self.model_path = path
        self.model_dir = model_dir

        # 输入张量名从图里读, 不假设固定名称/数量
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_name = self.session.get_outputs()[0].name

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.max_length = max_length or int(self.tokenizer.model_max_length or 512)
        self._lock = threading.Lock()  # ONNX Runtime 会话非线程安全 (rerank 走线程池并发)
        self.activation = activation or self._resolve_activation(model_dir)

        logger.info("ONNX Cross-Encoder 已加载: %s (输入=%s, 激活=%s)",
                    path, self.input_names, self.activation)

    @staticmethod
    def _resolve_activation(model_dir: str) -> str:
        """推导输出激活函数。

        sentence-transformers 的 CrossEncoder.predict() 对单标签模型默认套 Sigmoid。
        这是非线性变换, 而 rerank 融合用 min-max 归一化分数 —— min-max(sigmoid(x)) != min-max(x),
        所以这里必须与 PyTorch 路径一致, 否则融合权重会偏移。
        优先取导出时记录的 export_meta.json, 缺失时按 config.json 的 num_labels 推导。
        """
        meta_p = os.path.join(model_dir, "export_meta.json")
        if os.path.exists(meta_p):
            try:
                with open(meta_p, encoding="utf-8") as f:
                    act = json.load(f).get("activation")
                if act in ("sigmoid", "tanh", "identity"):
                    return act
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("读取 export_meta.json 失败, 回退按 num_labels 推导: %s", e)
        cfg_p = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_p):
            try:
                with open(cfg_p, encoding="utf-8") as f:
                    num_labels = json.load(f).get("num_labels", 1)
                return "sigmoid" if num_labels == 1 else "identity"
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("读取 config.json 失败, 回退 identity: %s", e)
        return "identity"

    def predict(self, pairs: list, batch_size: int = 32, show_progress_bar: bool = False):
        """对 (query, document) 列表打分, 返回 list[float]。

        签名对齐 sentence_transformers.CrossEncoder.predict, 多余参数仅为兼容。
        """
        import numpy as np

        if not pairs:
            return []

        # 兼容 [(q, d)] 与 [{"query":..,"text":..}] 两种传入形式
        norm = []
        for p in pairs:
            if isinstance(p, dict):
                norm.append((p.get("query", ""), p.get("text", p.get("content", ""))))
            elif isinstance(p, (list, tuple)) and len(p) >= 2:
                norm.append((p[0], p[1]))
            else:
                norm.append((str(p), ""))

        out: list[float] = []
        with self._lock:
            for i in range(0, len(norm), batch_size):
                chunk = norm[i:i + batch_size]
                enc = self.tokenizer(
                    [q for q, _ in chunk], [d for _, d in chunk],
                    padding=True, truncation=True, max_length=self.max_length,
                    return_tensors="np",
                )
                feed = {}
                for name in self.input_names:
                    key = name.split(".")[-1]  # 兼容导出时带前缀的命名
                    if key in enc:
                        feed[name] = enc[key].astype("int64")
                    elif key == "token_type_ids":
                        # 该模型词表不含句段嵌入时 tokenizer 不产出, 补零即可 (与 PyTorch 路径一致)
                        feed[name] = np.zeros(enc["input_ids"].shape, dtype="int64")
                    else:
                        msg = f"ONNX 图需要输入 {name}, 但 tokenizer 未提供"
                        raise ValueError(msg)
                logits = self.session.run([self.output_name], feed)[0]
                arr = np.asarray(logits).reshape(len(chunk), -1)
                # 单输出取最后一列 (num_labels==1 时即回归分数)
                scores = arr[:, -1].astype(np.float64)
                out.extend(self._activate(scores).astype(float).tolist())
        return out

    def _activate(self, scores):
        """施加与 sentence-transformers 一致的输出激活。"""
        import numpy as np

        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-scores))
        if self.activation == "tanh":
            return np.tanh(scores)
        return scores

    def __repr__(self) -> str:
        meta_p = os.path.join(self.model_dir, "export_meta.json")
        src = ""
        if os.path.exists(meta_p):
            try:
                with open(meta_p, encoding="utf-8") as f:
                    src = json.load(f).get("source_model", "")
            except (OSError, json.JSONDecodeError):
                src = ""
        return f"OnnxCrossEncoder({os.path.basename(self.model_path)}, from={src or 'unknown'})"
