FROM python:3.12-slim

WORKDIR /app

# 运行时依赖: onnxruntime 需 libgomp(OpenMP); opencv-python(non-headless) 需 libxcb + libGL
# (RapidOCR 扫描件兜底依赖 opencv, 缺 libxcb 会 ImportError 致 OCR 不可用)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libgomp1 libxcb1 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# HuggingFace 国内镜像 (离线场景设 TRANSFORMERS_OFFLINE=1)
ENV HF_ENDPOINT=https://hf-mirror.com
# pip 国内镜像, 加速依赖拉取
ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple \
    PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

COPY . .
# 完整依赖安装, 错误不静默吞掉 (原 2>/dev/null 会掩盖缺依赖)
RUN pip install --no-cache-dir .

# 构建期预下载 embedding 模型到 app 可读路径 — 运行时 TRANSFORMERS_OFFLINE=1, 不进镜像则加载失败
# (HF_HOME 指向 /app/.cache, 非 root 用户可读; 默认 /root/.cache 对 appuser 无权限)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5')"

# 非 root 运行 (生产安全): 建 appuser + 数据目录所有权
RUN useradd --create-home --uid 1000 appuser \
    && mkdir -p /app/data /app/lancedb_data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
