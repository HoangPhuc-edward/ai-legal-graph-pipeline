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


def parse_structure(norm_id: str, markdown: str) -> ParseResult:
    """Phân tích markdown của 1 văn bản thành cây Component.

    Component gốc trực tiếp dưới Norm có parent_comp_id=None.
    """
    now = datetime.now(timezone.utc)
    result = ParseResult()

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

    return result
