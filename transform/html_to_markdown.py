"""Wrap fast-html2md — làm sạch content_html trước khi tách cấu trúc.

Lưu ý: fast-html2md build trên markdownify, sẽ giữ heading thật (<h1>-<h6>)
thành '#', nhưng HTML của vbpl.vn thường KHÔNG dùng heading thật cho
"Chương"/"Điều" (chỉ là <p> in đậm/căn giữa) — nên đừng dựa vào '#' markdown
để tách cấp, vẫn cần structure_parser riêng cho cấu trúc pháp lý.
"""
from __future__ import annotations

from fast_html2md import HTMLToMarkdown

_converter = HTMLToMarkdown()


def convert(content_html: str) -> str:
    """Chuyển content_html sang markdown sạch."""
    if not content_html:
        return ""
    try:
        return _converter.convert(content_html)
    except Exception:
        return ""
