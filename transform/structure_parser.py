"""Markdown -> cây Component (Norm là gốc ảo). Thuật toán stack-based tree builder.

Xử lý đúng việc "không phải văn bản nào cũng đi đủ cấp" — pattern level sâu hơn
push làm con, bằng pop+push làm anh em, nông hơn pop liên tục tới đúng cha.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from config import LEVEL_RANK
from schema.enums import ComponentLevel
from schema.nodes import Component

# Thứ tự ưu tiên CỐ ĐỊNH — kiểm tra Phần trước, rồi Chương, Mục, Điều, Khoản, Điểm
# (pattern hẹp hơn như "Điều" có thể match nhầm bên trong dòng "Chương").
LEVEL_PATTERNS: list[tuple[ComponentLevel, re.Pattern]] = [
    (ComponentLevel.PHAN, re.compile(r"^\s*Phần\s+(thứ\s+)?([IVXLCDM\d]+)", re.IGNORECASE)),
    (ComponentLevel.CHUONG, re.compile(r"^\s*Chương\s+([IVXLCDM\d]+)", re.IGNORECASE)),
    (ComponentLevel.MUC, re.compile(r"^\s*(Mục|Tiểu mục)\s+(\d+)", re.IGNORECASE)),
    (ComponentLevel.DIEU, re.compile(r"^\s*Điều\s+(\d+[a-zđ]?)\s*[\.\:]", re.IGNORECASE)),
    (ComponentLevel.KHOAN, re.compile(r"^\s*(\d{1,2})\.\s+")),
    (ComponentLevel.DIEM, re.compile(r"^\s*([a-zđ])\)\s+")),
]

_LEVEL_LABEL = {
    ComponentLevel.PHAN: "Phần",
    ComponentLevel.CHUONG: "Chương",
    ComponentLevel.MUC: "Mục",
    ComponentLevel.TIEU_MUC: "Tiểu mục",
    ComponentLevel.DIEU: "Điều",
    ComponentLevel.KHOAN: "Khoản",
    ComponentLevel.DIEM: "Điểm",
}

# Cùng các pattern Phần/Chương/Mục/Điều ở LEVEL_PATTERNS nhưng KHÔNG neo `^\s*`
# — dùng để phát hiện marker bị "dính" giữa dòng (không phải đầu dòng) rồi cưỡng
# chế tách thành dòng riêng ở normalize_legal_markdown(). CỐ TÌNH bỏ Khoản/Điểm:
# pattern Khoản (`\d{1,2}\.`) quá lỏng, dễ khớp nhầm số liệu/ngày tháng giữa câu
# (vd "2.000 - 3.000 đồng") — chỉ an toàn khi giới hạn trong phạm vi 1 Điều cụ
# thể, để dành cho cải tiến sau (xem CLAUDE.md / README phần TODO).
_FORCE_BREAK_PATTERNS: list[re.Pattern] = [
    re.compile(r"Phần\s+(thứ\s+)?([IVXLCDM\d]+)", re.IGNORECASE),
    re.compile(r"Chương\s+([IVXLCDM\d]+)", re.IGNORECASE),
    re.compile(r"(Mục|Tiểu mục)\s+(\d+)", re.IGNORECASE),
    re.compile(r"Điều\s+(\d+[a-zđ]?)\s*[\.\:]", re.IGNORECASE),
]


def _split_line_at_headers(line: str) -> list[str]:
    """Tách 1 dòng thành nhiều dòng nếu có marker cấp (Phần/Chương/Mục/Điều) bị
    dính giữa dòng — ví dụ '...hết hiệu lực.Chương II QUY ĐỊNH CHUNG' (HTML
    nguồn không có <p> riêng cho từng marker) -> tách thành 2 dòng để
    `_match_level` (chỉ match đầu dòng) bắt được."""
    segments = [line]
    changed = True
    while changed:
        changed = False
        next_segments: list[str] = []
        for seg in segments:
            split_pos = None
            for pattern in _FORCE_BREAK_PATTERNS:
                m = pattern.search(seg)
                if m and m.start() > 0:
                    if split_pos is None or m.start() < split_pos:
                        split_pos = m.start()
            if split_pos:
                next_segments.append(seg[:split_pos])
                next_segments.append(seg[split_pos:])
                changed = True
            else:
                next_segments.append(seg)
        segments = next_segments
    return [s for s in segments if s.strip()]


def normalize_legal_markdown(markdown: str) -> str:
    """Chuẩn hoá markdown TRƯỚC khi tách dòng, để marker cấp (Phần/Chương/Mục/
    Điều) luôn nằm ở đầu dòng riêng — điều kiện bắt buộc để `_match_level` khớp
    được (chỉ dùng `pattern.match()`, neo `^\\s*`).

    2 lỗi phổ biến nhất khiến cả văn bản 0 Component (toàn bộ regex không khớp
    dòng nào):
      1. fast_html2md bọc marker trong markdown emphasis — '**Điều 1.** Phạm vi'
         -> dòng bắt đầu bằng '**', không phải 'Điều'.
      2. HTML nguồn không tách <p> riêng cho từng Phần/Chương/Điều — nhiều
         marker bị dính chung 1 dòng/đoạn.
    """
    if not markdown:
        return markdown

    # Khoảng trắng lạ (NBSP, zero-width space) -> bình thường, không thì
    # `\s*`/`.strip()` không nhận diện được là whitespace.
    text = markdown.replace("\xa0", " ").replace("​", "")
    # "**"/"__" chỉ là định dạng (bold), không mang nghĩa cấu trúc — bỏ thẳng
    # luôn cho sạch (đỡ lẫn vào raw_text lưu trữ/embedding sau này).
    text = text.replace("**", "").replace("__", "")

    out_lines: list[str] = []
    for raw_line in text.splitlines():
        out_lines.extend(_split_line_at_headers(raw_line))
    return "\n".join(out_lines)


@dataclass
class _StackEntry:
    comp_id: str
    level: ComponentLevel
    rank: int


@dataclass
class ParseResult:
    components: list[Component] = field(default_factory=list)
    # comp_id (leaf hoặc bất kỳ node nào nhận text trực tiếp) -> raw_text tích lũy
    raw_text: dict[str, str] = field(default_factory=dict)


def _match_level(line: str) -> Optional[tuple[ComponentLevel, re.Match]]:
    for level, pattern in LEVEL_PATTERNS:
        m = pattern.match(line)
        if m:
            return level, m
    return None


def _build_citation(level: ComponentLevel, match: re.Match) -> str:
    identifier = match.groups()[-1].strip()
    return f"{_LEVEL_LABEL[level]} {identifier}"


def _build_title_text(line: str, match: re.Match) -> Optional[str]:
    remainder = line[match.end():].strip(" .:-\t")
    return remainder or None


def _fallback_whole_document(norm_id: str, markdown: str, now: datetime) -> ParseResult:
    """Văn bản KHÔNG khớp được level pattern nào kể cả sau normalize — vd sắc
    lệnh bổ nhiệm nhân sự ngắn, văn phong cũ, không có cấu trúc Điều/Khoản rõ
    ràng. Đây KHÔNG phải lỗi parser, văn bản vẫn hợp lệ — nhưng nếu để 0
    Component thì 0 TextUnit, toàn bộ nội dung biến mất khỏi đồ thị. Gom toàn
    văn vào đúng 1 pseudo-Component (level=DIEU, citation="Điều 1") để giữ lại
    nội dung thay vì mất trắng."""
    text = markdown.strip()
    if not text:
        return ParseResult()

    comp_id = f"{norm_id}__c1"
    component = Component(
        comp_id=comp_id,
        norm_id=norm_id,
        level=ComponentLevel.DIEU,
        citation="Điều 1",
        order_index=1,
        parent_comp_id=None,
        title_text=None,
        updated_at=now,
    )
    return ParseResult(components=[component], raw_text={comp_id: text + "\n"})


def parse_structure(norm_id: str, markdown: str) -> ParseResult:
    """Phân tích markdown của 1 văn bản thành cây Component.

    Component gốc trực tiếp dưới Norm có parent_comp_id=None.
    """
    now = datetime.now(timezone.utc)
    result = ParseResult()
    markdown = normalize_legal_markdown(markdown)

    # stack[0] luôn là gốc ảo (Norm), rank = -1
    stack: list[_StackEntry] = [_StackEntry(comp_id="__ROOT__", level=None, rank=-1)]
    order_index = 0
    current_leaf_id = "__ROOT__"  # nơi nhận text không khớp level nào

    for raw_line in markdown.splitlines():
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        matched = _match_level(line)
        if matched is None:
            # Nối vào raw_text của Component lá gần nhất đang mở
            if current_leaf_id != "__ROOT__":
                result.raw_text[current_leaf_id] = (
                    result.raw_text.get(current_leaf_id, "") + line.strip() + "\n"
                )
            continue

        level, m = matched
        rank = LEVEL_RANK[level.value]

        # So độ sâu với đỉnh stack hiện tại
        while stack[-1].rank >= rank:
            stack.pop()
        parent_entry = stack[-1]

        order_index += 1
        comp_id = f"{norm_id}__c{order_index}"
        parent_comp_id = None if parent_entry.comp_id == "__ROOT__" else parent_entry.comp_id

        component = Component(
            comp_id=comp_id,
            norm_id=norm_id,
            level=level,
            citation=_build_citation(level, m),
            order_index=order_index,
            parent_comp_id=parent_comp_id,
            title_text=_build_title_text(line, m),
            updated_at=now,
        )
        result.components.append(component)
        if component.title_text:
            # Nội dung trên cùng dòng với marker (phổ biến ở Khoản/Điểm, và ở
            # Điều không có Khoản con) — seed làm raw_text ban đầu của chính nó.
            result.raw_text[comp_id] = component.title_text + "\n"

        stack.append(_StackEntry(comp_id=comp_id, level=level, rank=rank))
        current_leaf_id = comp_id

    # Chỉ giữ raw_text cho Component LÁ THẬT (không phải cha của component nào
    # khác) — nội dung của Component trung gian (vd Điều có Khoản con) đã được
    # phản ánh đầy đủ ở các Component lá bên dưới, giữ lại sẽ trùng lặp.
    parent_ids = {c.parent_comp_id for c in result.components if c.parent_comp_id is not None}
    result.raw_text = {
        comp_id: text for comp_id, text in result.raw_text.items() if comp_id not in parent_ids
    }

    if not result.components:
        return _fallback_whole_document(norm_id, markdown, now)

    return result
