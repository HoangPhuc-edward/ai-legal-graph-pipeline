"""Wrap fast-html2md — làm sạch content_html trước khi tách cấu trúc.

Lưu ý: fast-html2md build trên markdownify, sẽ giữ heading thật (<h1>-<h6>)
thành '#', nhưng HTML của vbpl.vn thường KHÔNG dùng heading thật cho
"Chương"/"Điều" (chỉ là <p> in đậm/căn giữa) — nên đừng dựa vào '#' markdown
để tách cấp, vẫn cần structure_parser riêng cho cấu trúc pháp lý.

Xử lý bảng:
- Biểu mẫu Word (checkbox clip_image) → bị xóa hoàn toàn
- Bảng rỗng (ô trống toàn bộ) → bị xóa hoàn toàn
- Bảng có nội dung → prose lines "Bảng [cột1 | cột2]: cột1: val1, cột2: val2"
  Cột trống vẫn giữ tên: "cột1: val1, cột2: " để không mất cấu trúc.
"""
from __future__ import annotations

import re

from fast_html2md import HTMLToMarkdown

_converter = HTMLToMarkdown()

_HAS_WORD = re.compile(r"\w", re.UNICODE)
# Strip markdown image ![alt](url), link [text](url), và <br> HTML inline
_MD_IMAGE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_BR_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Biểu mẫu Word: ảnh clipboard từ máy soạn thảo (không bao giờ là bảng dữ liệu thật)
_FORM_IMAGE = re.compile(r"!\[\]\(file:///[^)]*(?:clip_image|msohtmlclip)", re.IGNORECASE)
# Post-processing artifacts
_BR_GLOBAL = re.compile(r"<br\s*/?>", re.IGNORECASE)
_PIPE_SEP = re.compile(r"(?:\|\s*[-=: ]{2,}\s*)+\|")  # | --- | --- | còn sót
_DOTS_FILL = re.compile(r"\.{4,}")                     # ............ (form fill-in)


def _visible_text(cell: str) -> str:
    """Trả về text hiển thị của ô — loại bỏ markdown image/link và <br> HTML."""
    text = _MD_IMAGE_LINK.sub(r"\1", cell)
    text = _BR_TAG.sub(" ", text)
    return text.strip()


def _table_block_to_prose(block_lines: list[str]) -> list[str]:
    """Convert 1 Markdown pipe table block → prose lines.

    Biểu mẫu (clip_image/msohtmlclip) → [] (bỏ hoàn toàn).
    Bảng rỗng → [] (bỏ hoàn toàn).
    Bảng có nội dung → ["Bảng [h1 | h2]:","h1: v1, h2: v2, h3: ", ...]
    Cột có tên nhưng ô trống → vẫn hiển thị "tên_cột: " để giữ cấu trúc.
    """
    if not block_lines:
        return []

    # Lọc biểu mẫu Word — bảng có ảnh clipboard từ máy soạn thảo
    if any(
        _FORM_IMAGE.search(cell)
        for row in block_lines
        for cell in row.split("|")[1:-1]
    ):
        return []

    headers = [c.strip() for c in block_lines[0].split("|")[1:-1]]

    data_start = 1
    if len(block_lines) > 1:
        sep_cells = [c.strip(" -") for c in block_lines[1].split("|")[1:-1]]
        if all(not c for c in sep_cells):
            data_start = 2

    has_content = any(
        _HAS_WORD.search(_visible_text(cell).strip(" -"))
        for row in block_lines
        for cell in row.split("|")[1:-1]
    )
    if not has_content:
        return []

    header_str = " | ".join(_visible_text(h) for h in headers if _HAS_WORD.search(_visible_text(h)))
    result = [f"Bảng [{header_str}]:"] if header_str else []

    for row_line in block_lines[data_start:]:
        cells = [c.strip() for c in row_line.split("|")[1:-1]]
        parts = []
        has_row_content = False
        for header, cell in zip(headers, cells):
            visible = _visible_text(cell)
            h_visible = _visible_text(header)
            if _HAS_WORD.search(visible):
                has_row_content = True
                parts.append(f"{h_visible}: {visible}" if h_visible else visible)
            elif _HAS_WORD.search(h_visible):
                parts.append(f"{h_visible}: ")  # cột có tên, ô trống → giữ tên cột
        if has_row_content and parts:
            result.append(", ".join(parts))

    return result


def _process_tables(md: str) -> str:
    """Thay mỗi pipe table block bằng prose (hoặc xóa nếu rỗng)."""
    lines = md.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith("|"):
            result.append(lines[i])
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and lines[i].startswith("|"):
            block.append(lines[i])
            i += 1
        result.extend(_table_block_to_prose(block))
    return "\n".join(result)


def _post_clean(md: str) -> str:
    """Dọn dẹp markdown sau bước xử lý bảng — xóa artifact còn sót."""
    md = _BR_GLOBAL.sub(" ", md)   # <br> còn sót ngoài bảng
    md = _PIPE_SEP.sub("", md)     # | --- | nhúng trong dòng văn xuôi dài
    md = _DOTS_FILL.sub("", md)    # ............ (form fill-in dots)
    return md


def convert(content_html: str) -> str:
    """Chuyển content_html sang markdown sạch, bảng → prose lines."""
    if not content_html:
        return ""
    try:
        md = _converter.convert(content_html)
        md = _process_tables(md)
        return _post_clean(md)
    except Exception:
        return ""
