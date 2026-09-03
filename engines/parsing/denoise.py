"""解析后去噪 — 剔除非内容噪声, 提升检索信噪比。

设计原则 (视频要求「自动去噪: 水印 / 重复声明」):
- 仅剔除高置信噪声, 绝不误伤正常正文 (保守启发式)。
- 三类噪声:
  1. 跨页复现行: 同一条内容出现在 ≥50% 页(至少 3 页) → 页眉/页脚/水印常每页复现。
  2. 水印/页码/联系方式行: 命中明确启发式(版权/保密/page N/纯 URL·邮箱·电话)。
  3. 同文档内整块完全重复: 重复声明段, 保留首块、丢弃后续副本。

入口: UIRDocument.__post_init__ 懒加载调用, 对所有 parser 统一生效。
任何异常均吞掉并原样返回, 保证去噪永不阻断入库。
"""

import re

_WATERMARK_RE = re.compile(
    r"(版权所有|copyright|confidential|保密|内部资料|内部使用|all\s+rights\s+reserved|"
    r"密\s*级|draft|草稿|review\s+copy|sample|样章|不得外传|请勿外传|仅供内部)",
    re.IGNORECASE,
)
_PAGE_RE = re.compile(r"^\s*(第\s*\d+\s*页|page\s*\d+\s*(of\s*\d+)?|\d+\s*/\s*\d+)\s*$", re.IGNORECASE)
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"(?<!\d)(\d{3,4}[-\s]?\d{3,4}[-\s]?\d{4}|\d{3,4}[-\s]?\d{7,8})(?!\d)")

_BLOCK_TYPES = ("paragraph", "title", "code")


def _is_boilerplate_line(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if _PAGE_RE.match(s):
        return True
    if _WATERMARK_RE.search(s):
        return True
    # 短行 + 纯联系方式/链接 → 页脚水印嫌疑
    return len(s) <= 60 and (
        _URL_RE.search(s) or _EMAIL_RE.search(s) or _PHONE_RE.search(s)
    )


def denoise_document(uir) -> "object":
    """原地清理 uir.pages 中的噪声块; 异常时原样返回。"""
    try:
        pages = uir.pages or []
        if not pages:
            return uir
        n_pages = max(1, len(pages))

        # 1) 跨页重复行统计
        line_pages: dict[str, set] = {}
        for pi, page in enumerate(pages, 1):
            seen: set[str] = set()
            for b in page.get("blocks", []):
                if b.get("type") not in _BLOCK_TYPES:
                    continue
                for ln in (b.get("content") or "").splitlines():
                    ln = ln.strip()
                    if ln and ln not in seen:
                        seen.add(ln)
                        line_pages.setdefault(ln, set()).add(pi)
        repeat_thresh = max(3, int(n_pages * 0.5))
        repeated = {ln for ln, ps in line_pages.items() if len(ps) >= repeat_thresh}

        # 2) 逐块过滤 + 同文档整块去重
        seen_blocks: set[str] = set()
        for page in pages:
            kept = []
            for b in page.get("blocks", []):
                content = (b.get("content") or "").strip()
                if not content:
                    continue
                lines = content.splitlines()
                # 仅单行块且为噪声(水印/页码/跨页重复)才整块丢弃; 多行块走逐行剔除, 避免误删正文
                if len(lines) == 1 and (_is_boilerplate_line(content) or content in repeated):
                    continue
                # 多行块: 剔除其中的噪声行, 清空则丢弃
                cleaned = [
                    ln for ln in lines
                    if not (_is_boilerplate_line(ln.strip()) or ln.strip() in repeated)
                ]
                new_content = "\n".join(cleaned).strip()
                if not new_content:
                    continue
                # 同文档整块完全重复 → 丢弃后续副本 (重复声明); 仅对短块生效, 避免误合并正常长文
                if len(new_content) < 200 and new_content in seen_blocks:
                    continue
                seen_blocks.add(new_content)
                if new_content != b.get("content"):
                    b["content"] = new_content
                kept.append(b)
            page["blocks"] = kept
    except Exception:  # noqa: S110, BLE001 -- 去噪绝不该阻断入库: 任何异常原样返回
        pass
    return uir
