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
from itertools import zip_longest

from bs4 import BeautifulSoup
from fast_html2md import HTMLToMarkdown

_converter = HTMLToMarkdown()

_HAS_WORD = re.compile(r"\w", re.UNICODE)
# Phát hiện colspan/rowspan >= 2 trong HTML — dùng để fast-path bỏ qua BS4 parse
_CS_RS = re.compile(r'\b(colspan|rowspan)\s*=\s*["\']?([2-9]|\d{2,})', re.IGNORECASE)
# Strip markdown image ![alt](url), link [text](url), và <br> HTML inline
_MD_IMAGE_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_BR_TAG = re.compile(r"<br\s*/?>", re.IGNORECASE)
# Biểu mẫu Word: ảnh clipboard từ máy soạn thảo (không bao giờ là bảng dữ liệu thật)
_FORM_IMAGE = re.compile(r"!\[\]\(file:///[^)]*(?:clip_image|msohtmlclip)", re.IGNORECASE)
# Post-processing artifacts
_BR_GLOBAL = re.compile(r"<br\s*/?>", re.IGNORECASE)
_PIPE_SEP = re.compile(r"(?:\|\s*[-=: ]{2,}\s*)+\|")  # | --- | --- | còn sót
_DOTS_FILL = re.compile(r"\.{4,}")                     # ............ (form fill-in)


def _get_rows(table_tag) -> list:
    """Lấy <tr> trực tiếp của table (không vào nested table con)."""
    rows = []
    for child in table_tag.children:
        name = getattr(child, "name", None)
        if name == "tr":
            rows.append(child)
        elif name in ("tbody", "thead", "tfoot"):
            for sub in child.children:
                if getattr(sub, "name", None) == "tr":
                    rows.append(sub)
    return rows


def _expand_table_to_grid(table_tag) -> list[list[str]]:
    """5-bước expand colspan/rowspan → flat table với tên cột phân tầng.

    Bước 1: Xây grid vật lý dict (row,col)→text, đồng thời ghi nhận max colspan/hàng.
    Bước 2: Nhận diện N tầng header — rows có colspan>1 là "header cha"; row đầu tiên
            không có colspan ngay sau dãy header cha là "leaf header"; dừng tại đó.
    Bước 3: Tên cột phẳng = ghép text N tầng theo cột, dùng " > " làm ngăn cách;
            bỏ tầng trùng với tầng trước (rowspan điền cùng text xuống nhiều tầng).
    Bước 4: Data rows đã có rowspan fill từ bước 1 — không cần xử lý thêm.
    Bước 5: Trả [flat_headers] + data_rows.

    Fallback: nếu không phát hiện multi-level header → trả grid phẳng như cũ.
    """
    rows = _get_rows(table_tag)
    n = len(rows)
    if n == 0:
        return []

    # Bước 1: parse raw cells + build physical grid
    row_max_cs: list[int] = []  # max colspan của các ô gốc HTML trong mỗi hàng
    cells: dict[tuple[int, int], str] = {}

    for ri, tr in enumerate(rows):
        ci = 0
        max_cs_row = 1
        for td in tr.find_all(["td", "th"], recursive=False):
            while (ri, ci) in cells:
                ci += 1
            try:
                cs = max(1, int(td.get("colspan") or 1))
                rs = max(1, int(td.get("rowspan") or 1))
            except (ValueError, TypeError):
                cs, rs = 1, 1
            text = td.get_text(" ", strip=True)
            max_cs_row = max(max_cs_row, cs)
            for r_off in range(rs):
                for c_off in range(cs):
                    tr_idx = ri + r_off
                    if tr_idx < n:
                        cells[(tr_idx, ci + c_off)] = text
            ci += cs
        row_max_cs.append(max_cs_row)

    if not cells:
        return []
    max_r = max(r for r, _ in cells) + 1
    max_c = max(c for _, c in cells) + 1

    # Bước 2: nhận diện số tầng header
    # - Rows có colspan > 1 → header tầng cha
    # - Row đầu tiên không có colspan, đứng ngay sau dãy cha → leaf header, dừng
    n_header = 0
    found_parent = False
    for ri, max_cs in enumerate(row_max_cs):
        if max_cs > 1:
            n_header = ri + 1
            found_parent = True
        elif found_parent:
            n_header = ri + 1  # leaf header row
            break
        else:
            break  # chưa gặp header cha → bảng đơn tầng

    # Fallback: không có multi-level header, hoặc tất cả rows là header (bảng kỳ lạ)
    if n_header == 0 or n_header >= max_r:
        return [
            [cells.get((r, c), "") for c in range(max_c)]
            for r in range(max_r)
        ]

    # Bước 3: xây tên cột phẳng "Cha > Con > Lá"
    # Dedup: bỏ tầng có text trùng tầng ngay trên (rowspan điền cùng text xuống)
    flat_headers: list[str] = []
    for c in range(max_c):
        parts: list[str] = []
        prev_text: str | None = None
        for ri in range(n_header):
            text = cells.get((ri, c), "").strip()
            if text and text != prev_text:
                parts.append(text)
                prev_text = text
        flat_headers.append(" > ".join(parts) if parts else "")

    # Bước 4 & 5: data rows — rowspan đã được fill trong cells dict (bước 1)
    data_rows = [
        [cells.get((r, c), "") for c in range(max_c)]
        for r in range(n_header, max_r)
    ]

    return [flat_headers] + data_rows


def _preprocess_tables_html(html: str) -> str:
    """Expand colspan/rowspan trong tất cả <table> trước khi pass vào fast_html2md.

    Chỉ xử lý LEAF table (không chứa <table> con) — wrapper/layout table
    bị bỏ qua để tránh thay thế outer table làm mất inner tables.
    Fast path: nếu không có 'colspan'/'rowspan' trong HTML → bỏ qua parse BS4.
    """
    if "colspan" not in html and "rowspan" not in html:
        return html
    soup = BeautifulSoup(html, "html.parser")
    for tbl in soup.find_all("table"):
        if tbl.find("table"):  # wrapper/layout table — bỏ qua, chỉ process leaf
            continue
        if not _CS_RS.search(str(tbl)):  # không có merge cell — không cần expand
            continue
        grid = _expand_table_to_grid(tbl)
        if not grid:
            continue
        rows_html = "".join(
            "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
            for row in grid
        )
        tbl.replace_with(BeautifulSoup(f"<table>{rows_html}</table>", "html.parser"))
    return str(soup)


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
        for header, cell in zip_longest(headers, cells, fillvalue=""):
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
        preprocessed = _preprocess_tables_html(content_html)
        md = _converter.convert(preprocessed)
        md = _process_tables(md)
        return _post_clean(md)
    except Exception:
        return ""
