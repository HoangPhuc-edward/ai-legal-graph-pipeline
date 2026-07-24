"""Kiểm tra và sửa lỗi cấu trúc sau bước dedup và ghost removal.

Áp dụng các rule từ Structural_Validation_Rules.md:

Tự sửa (severity="fix"):
  D2 — Component lá có raw_text < 10 ký tự: gộp nội dung vào node cha,
       không tạo TextUnit riêng cho lá này (tránh TextUnit gần rỗng).
  C2 — Điều chỉ có 1 Khoản: rescan raw_text tìm "2. ", "3. " bị nuốt,
       tách ra thành Component mới.
  B1 — Điều nhảy số: rescan raw_text của Điều liền trước gap tìm "Điều N."
       bị nuốt, tách ra thành Component mới.
  B4 — Điểm nhảy chữ cái: rescan raw_text của Điểm liền trước gap tìm
       "b. ", "c. " bị nuốt, tách ra thành Component mới.

  Tất cả rescan đều theo nguyên tắc: nếu marker không tìm thấy → warn (không
  sửa được); nếu tìm thấy → tạo Component mới và severity="fix". Nội dung không
  thuộc phần recover được giữ trong Component cha (không mất content).

Chỉ cảnh báo (severity="warn") — khi rescan không tìm thấy marker:
  B1, B4, C2 (xem trên)
  C1 — Văn bản dài (>3000 ký tự) nhưng chỉ có 1 Điều
  C3 — Chương/Mục/Phần không có Điều con
  D1 — TextUnit dài bất thường so với các Component cùng cấp (median+MAD)

Cảnh báo bug parser (severity="bug" — cần sửa code, không patch dữ liệu):
  E1 — Nhảy cấp bất hợp lệ (Chương → Khoản, bỏ qua tầng Điều)

Nguyên tắc:
  - KHÔNG bao giờ xóa Component node — cấu trúc cây luôn được giữ nguyên.
  - Trường hợp bất khả kháng (văn bản quá xấu): chấp nhận cấu trúc không đẹp
    nhưng vẫn bảo toàn toàn bộ nội dung để phục vụ retrieval.
  - Component được recover có order_index gần đúng (có thể âm hoặc trùng với
    Component lân cận). Thứ tự trong graph không hoàn hảo nhưng nội dung có mặt.
"""
from __future__ import annotations

import re
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import LEVEL_RANK
from schema.enums import ComponentLevel
from schema.nodes import Component


# ─── Pattern phân tích citation ──────────────────────────────────────────────

_DIEU_NUM = re.compile(r"Điều\s+(\d+)", re.IGNORECASE)
_KHOAN_NUM = re.compile(r"Khoản\s+(\d+)", re.IGNORECASE)
_DIEM_CHAR = re.compile(r"Điểm\s+([a-zđ])\b", re.IGNORECASE)

# Thứ tự alphabet tiếng Việt dùng cho Điểm
_VIET_ALPHA = list("abcdđeghiklmnoprstu")
_VIET_ALPHA_POS = {c: i for i, c in enumerate(_VIET_ALPHA)}

# Các cấp nhóm cần check C3 (Chương/Mục/Phần rỗng)
_GROUP_LEVELS = {ComponentLevel.PHAN, ComponentLevel.CHUONG, ComponentLevel.MUC, ComponentLevel.TIEU_MUC}

# D2: ngưỡng "gần rỗng" (ký tự) — dưới ngưỡng này sẽ gộp vào cha
_D2_EMPTY_THRESHOLD = 10

# C1: ngưỡng nội dung tối thiểu để cảnh báo (văn bản cực ngắn thì 1 Điều là bình thường)
_C1_CONTENT_THRESHOLD = 3_000

# D1: số lần MAD để coi là outlier
_D1_MAD_FACTOR = 6


# ─── Pattern tìm marker bị nuốt trong raw_text (dùng cho rescan) ─────────────

# "Điều N." hoặc "Điều N:" ở đầu dòng
_EMBEDDED_DIEU_RE = re.compile(r"(?m)^(Điều\s+(\d+)\s*[\.:])", re.IGNORECASE)

# "N. " (Khoản số) ở đầu dòng — "1. Nội dung khoản"
_EMBEDDED_KHOAN_RE = re.compile(r"(?m)^(\d+)\.\s", re.MULTILINE)

# "a. " (Điểm chữ) ở đầu dòng — "b. Nội dung điểm"
_EMBEDDED_DIEM_RE = re.compile(r"(?m)^([a-zđ])\.\s", re.MULTILINE | re.IGNORECASE)


# ─── Dataclass kết quả ───────────────────────────────────────────────────────

@dataclass
class StructuralWarning:
    """1 phát hiện từ validate_structure."""
    rule: str              # "B1", "B2", ..., "E1"
    norm_id: str
    comp_id: Optional[str]  # None nếu cảnh báo ở cấp Norm
    severity: str          # "fix" | "warn" | "bug"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Kết quả validate_structure — components/raw_text đã được sửa (nếu có fix)."""
    warnings: list[StructuralWarning] = field(default_factory=list)
    components: list[Component] = field(default_factory=list)
    raw_text: dict[str, str] = field(default_factory=dict)

    @property
    def fix_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "fix")

    @property
    def warn_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "warn")

    @property
    def bug_count(self) -> int:
        return sum(1 for w in self.warnings if w.severity == "bug")


# ─── Hàm chính ───────────────────────────────────────────────────────────────

def validate_structure(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
) -> ValidationResult:
    """Kiểm tra cấu trúc sau dedup + ghost removal. Tự sửa D2/C2/B1/B4, warn/bug các rule còn lại.

    Đầu vào: components đã qua _dedup_components + _remove_triplication_ghosts.
    Đầu ra: ValidationResult với components/raw_text đã được sửa.
    """
    if not components:
        return ValidationResult(components=components, raw_text=raw_text)

    warnings: list[StructuralWarning] = []

    def _build_maps(comps: list[Component]):
        by_id = {c.comp_id: c for c in comps}
        ch_map: dict[Optional[str], list[Component]] = defaultdict(list)
        for c in comps:
            ch_map[c.parent_comp_id].append(c)
        for lst in ch_map.values():
            lst.sort(key=lambda c: c.order_index)
        return by_id, ch_map

    comp_by_id, children_map = _build_maps(components)

    # Phase 1: Fix — thứ tự quan trọng (D2 trước để không tạo TextUnit rỗng;
    # C2 sửa cấu trúc Khoản trước B1 sửa cấu trúc Điều)
    raw_text = _fix_d2(norm_id, components, raw_text, comp_by_id, children_map, warnings)
    components, raw_text = _rescan_c2(norm_id, components, raw_text, children_map, warnings)
    components, raw_text = _rescan_b1(norm_id, components, raw_text, warnings)
    components, raw_text = _rescan_b4(norm_id, components, raw_text, children_map, warnings)

    # Rebuild maps sau tất cả rescan — components mới có thể đã được thêm vào
    comp_by_id, children_map = _build_maps(components)

    # Phase 2: Check — chỉ warn/bug, không sửa dữ liệu
    _check_c1(norm_id, components, raw_text, warnings)
    _check_c3(norm_id, components, comp_by_id, warnings)
    _check_d1(norm_id, components, raw_text, warnings)
    _check_e1(norm_id, components, children_map, warnings)

    return ValidationResult(warnings=warnings, components=components, raw_text=raw_text)


# ─── Helper: split text tại các marker ──────────────────────────────────────

def _split_by_pattern(
    text: str,
    pattern: re.Pattern,
) -> tuple[str, list[tuple[re.Match, str]]]:
    """Split text tại mỗi lần pattern khớp ở đầu dòng.

    Trả về (intro_trước_match_đầu_tiên, [(match, đoạn_text_từ_match_đến_match_kế), ...]).
    intro rỗng nếu match đầu tiên ở ngay đầu text.
    """
    matches = list(pattern.finditer(text))
    if not matches:
        return text, []
    intro = text[:matches[0].start()].rstrip()
    segments = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        segments.append((m, text[m.start():end].rstrip()))
    return intro, segments


# ─── D2: Component lá gần rỗng — tự sửa ────────────────────────────────────

def _fix_d2(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    comp_by_id: dict[str, Component],
    children_map: dict[Optional[str], list[Component]],
    warnings: list[StructuralWarning],
) -> dict[str, str]:
    """D2: Leaf Component có raw_text < 30 ký tự → gộp text vào cha.

    Sau khi gộp, leaf bị xóa khỏi raw_text → không tạo TextUnit riêng cho nó.
    Component node vẫn giữ nguyên trong cây (bảo toàn cấu trúc).
    """
    has_children: set[str] = {c.parent_comp_id for c in components if c.parent_comp_id is not None}

    to_fix = [
        (comp_id, text, comp_by_id[comp_id])
        for comp_id, text in raw_text.items()
        if comp_id not in has_children
        and len(text) < _D2_EMPTY_THRESHOLD
        and comp_id in comp_by_id
        and comp_by_id[comp_id].parent_comp_id is not None
    ]

    if not to_fix:
        return raw_text

    raw_text = dict(raw_text)

    for comp_id, text, comp in to_fix:
        parent_id = comp.parent_comp_id
        parent_comp = comp_by_id.get(parent_id)
        parent_level = parent_comp.level.value if parent_comp else "?"

        existing = raw_text.get(parent_id, "")
        raw_text[parent_id] = (existing + "\n" + text).lstrip("\n") if existing else text
        del raw_text[comp_id]

        warnings.append(StructuralWarning(
            rule="D2",
            norm_id=norm_id,
            comp_id=comp_id,
            severity="fix",
            message=(
                f"Leaf {comp.level.value} {comp.citation!r} chỉ có {len(text)} ký tự "
                f"→ gộp vào {parent_level} {(parent_comp.citation if parent_comp else parent_id)!r}"
            ),
            details={
                "comp_id": comp_id,
                "text_len": len(text),
                "parent_id": parent_id,
                "parent_level": parent_level,
            },
        ))

    return raw_text


# ─── C2: Điều có đúng 1 Khoản con — rescan + fix ────────────────────────────

def _rescan_c2(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    children_map: dict[Optional[str], list[Component]],
    warnings: list[StructuralWarning],
) -> tuple[list[Component], dict[str, str]]:
    """C2: Tìm Khoản 2+ bị nuốt vào raw_text của Khoản 1, tách thành Component mới.

    Nếu không tìm thấy marker → warn. Nếu tìm thấy → fix.
    """
    new_comps = list(components)
    raw_text = dict(raw_text)

    for comp in list(components):
        if comp.level != ComponentLevel.DIEU:
            continue
        khoan_children = sorted(
            [ch for ch in children_map.get(comp.comp_id, []) if ch.level == ComponentLevel.KHOAN],
            key=lambda c: c.order_index,
        )
        if len(khoan_children) != 1:
            continue

        sole = khoan_children[0]
        khoan_text = raw_text.get(sole.comp_id, "")
        if not khoan_text:
            warnings.append(StructuralWarning(
                rule="C2", norm_id=norm_id, comp_id=comp.comp_id, severity="warn",
                message=(
                    f"{comp.citation} chỉ có 1 Khoản ({sole.citation}) "
                    f"— Khoản không có raw_text để rescan"
                ),
                details={"dieu_id": comp.comp_id, "only_khoan": sole.citation},
            ))
            continue

        _, segments = _split_by_pattern(khoan_text, _EMBEDDED_KHOAN_RE)

        # Chỉ tách Khoản có số > 1 (Khoản 2, 3, ...)
        relevant = [(sm, st) for sm, st in segments if int(sm.group(1)) > 1]

        if not relevant:
            warnings.append(StructuralWarning(
                rule="C2", norm_id=norm_id, comp_id=comp.comp_id, severity="warn",
                message=(
                    f"{comp.citation} chỉ có 1 Khoản ({sole.citation}) "
                    f"— không tìm thấy marker 'N.' trong raw_text để phục hồi Khoản 2+"
                ),
                details={"dieu_id": comp.comp_id, "only_khoan": sole.citation},
            ))
            continue

        # Giữ sole Khoản chỉ phần text trước marker "2." đầu tiên
        first_relevant_pos = relevant[0][0].start()
        raw_text[sole.comp_id] = khoan_text[:first_relevant_pos].rstrip()

        for j, (seg_m, seg_text) in enumerate(relevant):
            k_num = int(seg_m.group(1))
            new_comp_id = f"{sole.comp_id}__k{k_num}_rec"
            new_comp = Component(
                comp_id=new_comp_id,
                norm_id=norm_id,
                level=ComponentLevel.KHOAN,
                citation=f"Khoản {k_num}",
                order_index=sole.order_index + j + 1,
                parent_comp_id=comp.comp_id,
                title_text=None,
                updated_at=comp.updated_at,
            )
            new_comps.append(new_comp)
            raw_text[new_comp_id] = seg_text

        warnings.append(StructuralWarning(
            rule="C2", norm_id=norm_id, comp_id=comp.comp_id, severity="fix",
            message=(
                f"{comp.citation}: Phục hồi {len(relevant)} Khoản bị nuốt "
                f"(Khoản {relevant[0][0].group(1)}–{relevant[-1][0].group(1)}) "
                f"từ raw_text của {sole.citation}"
            ),
            details={
                "dieu_id": comp.comp_id,
                "recovered": [int(sm.group(1)) for sm, _ in relevant],
            },
        ))

    return new_comps, raw_text


# ─── B1: Điều nhảy số — rescan + fix ────────────────────────────────────────

def _rescan_b1(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    warnings: list[StructuralWarning],
) -> tuple[list[Component], dict[str, str]]:
    """B1: Tìm Điều bị nuốt vào raw_text của Điều liền trước gap, tách thành Component mới.

    Nếu không tìm thấy marker → warn. Nếu tìm thấy → fix.
    Nội dung embedded Điều ngoài gap (không trong range) được giữ lại trong
    cur_comp để không mất content.
    """
    dieu_list = sorted(
        [
            (int(m.group(1)), c)
            for c in components
            if c.level == ComponentLevel.DIEU
            for m in [_DIEU_NUM.search(c.citation)]
            if m
        ],
        key=lambda x: x[0],
    )

    if not dieu_list:
        return components, raw_text

    new_comps = list(components)
    raw_text = dict(raw_text)

    for i in range(len(dieu_list) - 1):
        cur_num, cur_comp = dieu_list[i]
        nxt_num, _ = dieu_list[i + 1]
        gap = nxt_num - cur_num
        if gap <= 1:
            continue

        dieu_text = raw_text.get(cur_comp.comp_id, "")
        if not dieu_text:
            warnings.append(StructuralWarning(
                rule="B1", norm_id=norm_id, comp_id=cur_comp.comp_id, severity="warn",
                message=(
                    f"Điều nhảy số: {cur_num} → {nxt_num} (thiếu {gap - 1} Điều) "
                    f"— Điều không có raw_text để rescan"
                ),
                details={"from_dieu": cur_num, "to_dieu": nxt_num, "gap": gap - 1},
            ))
            continue

        intro, segments = _split_by_pattern(dieu_text, _EMBEDDED_DIEU_RE)

        # Phần recover: Điều có số trong khoảng gap (cur_num < n < nxt_num)
        relevant = [
            (sm, st) for sm, st in segments
            if cur_num < int(sm.group(2)) < nxt_num
        ]
        # Phần không recover: giữ lại trong cur_comp để không mất content
        non_relevant_parts = [intro] + [
            st for sm, st in segments
            if not (cur_num < int(sm.group(2)) < nxt_num)
        ]

        if not relevant:
            warnings.append(StructuralWarning(
                rule="B1", norm_id=norm_id, comp_id=cur_comp.comp_id, severity="warn",
                message=(
                    f"Điều nhảy số: {cur_num} → {nxt_num} (thiếu {gap - 1} Điều) "
                    f"— không tìm thấy marker 'Điều N.' trong raw_text để phục hồi"
                ),
                details={"from_dieu": cur_num, "to_dieu": nxt_num, "gap": gap - 1},
            ))
            continue

        raw_text[cur_comp.comp_id] = "\n".join(p for p in non_relevant_parts if p).strip()

        for j, (seg_m, seg_text) in enumerate(relevant):
            d_num = int(seg_m.group(2))
            new_comp_id = f"{cur_comp.comp_id}__d{d_num}_rec"
            new_comp = Component(
                comp_id=new_comp_id,
                norm_id=norm_id,
                level=ComponentLevel.DIEU,
                citation=f"Điều {d_num}",
                # Chèn sau cur_comp — có thể trùng với nxt_comp nếu index liền kề
                order_index=cur_comp.order_index + j + 1,
                parent_comp_id=cur_comp.parent_comp_id,
                title_text=None,
                updated_at=cur_comp.updated_at,
            )
            new_comps.append(new_comp)
            raw_text[new_comp_id] = seg_text

        warnings.append(StructuralWarning(
            rule="B1", norm_id=norm_id, comp_id=cur_comp.comp_id, severity="fix",
            message=(
                f"Điều nhảy số: {cur_num} → {nxt_num} — phục hồi {len(relevant)} Điều "
                f"(Điều {relevant[0][0].group(2)}–{relevant[-1][0].group(2)}) từ raw_text"
            ),
            details={
                "from_dieu": cur_num,
                "to_dieu": nxt_num,
                "recovered": [int(sm.group(2)) for sm, _ in relevant],
            },
        ))

    return new_comps, raw_text


# ─── B4: Điểm nhảy chữ cái — rescan + fix ───────────────────────────────────

def _rescan_b4(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    children_map: dict[Optional[str], list[Component]],
    warnings: list[StructuralWarning],
) -> tuple[list[Component], dict[str, str]]:
    """B4: Tìm Điểm bị nuốt vào raw_text của Điểm liền trước gap, tách thành Component mới.

    Chỉ xử lý gap đầu tiên tìm thấy trong mỗi Khoản (tránh cascade phức tạp).
    Nếu không tìm thấy marker → warn. Nếu tìm thấy → fix.
    """
    new_comps = list(components)
    raw_text = dict(raw_text)

    for comp in list(components):
        if comp.level != ComponentLevel.KHOAN:
            continue
        diem_children = sorted(
            [ch for ch in children_map.get(comp.comp_id, []) if ch.level == ComponentLevel.DIEM],
            key=lambda c: c.order_index,
        )
        if len(diem_children) < 2:
            continue

        # Build (alpha_pos, char, diem_comp) để tìm gap
        diem_seq: list[tuple[int, str, Component]] = []
        for diem in diem_children:
            m = _DIEM_CHAR.search(diem.citation)
            if m:
                char = m.group(1).lower()
                pos = _VIET_ALPHA_POS.get(char)
                if pos is not None:
                    diem_seq.append((pos, char, diem))

        # Tìm gap đầu tiên
        for i in range(len(diem_seq) - 1):
            cur_pos, cur_char, cur_diem = diem_seq[i]
            nxt_pos, nxt_char, _ = diem_seq[i + 1]
            if nxt_pos - cur_pos <= 1:
                continue

            # Gap phát hiện — rescan raw_text của cur_diem
            diem_text = raw_text.get(cur_diem.comp_id, "")
            if not diem_text:
                warnings.append(StructuralWarning(
                    rule="B4", norm_id=norm_id, comp_id=comp.comp_id, severity="warn",
                    message=(
                        f"{comp.citation}: Điểm nhảy chữ cái: {cur_char} → {nxt_char} "
                        f"(thiếu {nxt_pos - cur_pos - 1} Điểm) — Điểm không có raw_text"
                    ),
                    details={"khoan_id": comp.comp_id, "from_char": cur_char, "to_char": nxt_char},
                ))
                break

            intro, segments = _split_by_pattern(diem_text, _EMBEDDED_DIEM_RE)

            missing = {_VIET_ALPHA[p] for p in range(cur_pos + 1, nxt_pos)}
            relevant = [
                (sm, st) for sm, st in segments
                if sm.group(1).lower() in missing
            ]
            non_relevant_parts = [intro] + [
                st for sm, st in segments
                if sm.group(1).lower() not in missing
            ]

            if not relevant:
                warnings.append(StructuralWarning(
                    rule="B4", norm_id=norm_id, comp_id=comp.comp_id, severity="warn",
                    message=(
                        f"{comp.citation}: Điểm nhảy chữ cái: {cur_char} → {nxt_char} "
                        f"(thiếu {nxt_pos - cur_pos - 1} Điểm) "
                        f"— không tìm thấy marker trong raw_text để phục hồi"
                    ),
                    details={"khoan_id": comp.comp_id, "from_char": cur_char, "to_char": nxt_char},
                ))
                break

            raw_text[cur_diem.comp_id] = "\n".join(p for p in non_relevant_parts if p).strip()

            for j, (seg_m, seg_text) in enumerate(relevant):
                d_char = seg_m.group(1).lower()
                new_comp_id = f"{cur_diem.comp_id}__dp{d_char}_rec"
                new_comp = Component(
                    comp_id=new_comp_id,
                    norm_id=norm_id,
                    level=ComponentLevel.DIEM,
                    citation=f"Điểm {d_char}",
                    order_index=cur_diem.order_index + j + 1,
                    parent_comp_id=comp.comp_id,
                    title_text=None,
                    updated_at=comp.updated_at,
                )
                new_comps.append(new_comp)
                raw_text[new_comp_id] = seg_text

            warnings.append(StructuralWarning(
                rule="B4", norm_id=norm_id, comp_id=comp.comp_id, severity="fix",
                message=(
                    f"{comp.citation}: Phục hồi {len(relevant)} Điểm bị nuốt "
                    f"(Điểm {relevant[0][0].group(1)}–{relevant[-1][0].group(1)}) "
                    f"từ raw_text của Điểm {cur_char}"
                ),
                details={
                    "khoan_id": comp.comp_id,
                    "recovered": [sm.group(1).lower() for sm, _ in relevant],
                },
            ))
            break  # chỉ xử lý gap đầu tiên mỗi Khoản

    return new_comps, raw_text


# ─── C1: Văn bản dài nhưng chỉ có 1 Điều ────────────────────────────────────

def _check_c1(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    warnings: list[StructuralWarning],
) -> None:
    """C1: Chỉ có 1 Điều nhưng tổng nội dung > 3000 ký tự — có thể nhiều Điều bị nuốt."""
    dieu_list = [c for c in components if c.level == ComponentLevel.DIEU]
    if len(dieu_list) != 1:
        return
    total = sum(len(t) for t in raw_text.values())
    if total >= _C1_CONTENT_THRESHOLD:
        dieu = dieu_list[0]
        warnings.append(StructuralWarning(
            rule="C1",
            norm_id=norm_id,
            comp_id=dieu.comp_id,
            severity="warn",
            message=(
                f"Chỉ có 1 Điều ({dieu.citation}) nhưng tổng nội dung {total} ký tự "
                f"(> {_C1_CONTENT_THRESHOLD}) — có thể nhiều Điều bị nuốt thành plain text"
            ),
            details={"dieu_count": 1, "total_content_chars": total},
        ))


# ─── C3: Chương/Mục/Phần không có Điều con ──────────────────────────────────

def _check_c3(
    norm_id: str,
    components: list[Component],
    comp_by_id: dict[str, Component],
    warnings: list[StructuralWarning],
) -> None:
    """C3: Chương/Mục/Phần tồn tại nhưng không có Điều con (trực tiếp hay gián tiếp)."""
    has_dieu_descendant: set[str] = set()
    for c in components:
        if c.level != ComponentLevel.DIEU:
            continue
        cur = c
        while cur.parent_comp_id:
            has_dieu_descendant.add(cur.parent_comp_id)
            parent = comp_by_id.get(cur.parent_comp_id)
            if parent is None:
                break
            cur = parent

    for c in components:
        if c.level not in _GROUP_LEVELS:
            continue
        if c.comp_id not in has_dieu_descendant:
            warnings.append(StructuralWarning(
                rule="C3",
                norm_id=norm_id,
                comp_id=c.comp_id,
                severity="warn",
                message=(
                    f"{c.level.value} {c.citation!r} không có Điều con nào "
                    f"— có thể là ghost hoặc Điều bị gán nhầm parent"
                ),
                details={"comp_id": c.comp_id, "level": c.level.value},
            ))


# ─── D1: TextUnit dài bất thường (tương đối so với đồng cấp) ────────────────

def _check_d1(
    norm_id: str,
    components: list[Component],
    raw_text: dict[str, str],
    warnings: list[StructuralWarning],
) -> None:
    """D1: Dùng median + MAD để phát hiện Component có nội dung dài bất thường.

    Chạy sau tất cả rescan — nếu D1 vẫn trigger ở đây thì content thực sự dài,
    không phải do marker bị nuốt (đã được B1/B4/C2 xử lý trước).
    """
    by_level: dict[str, list[tuple[str, int]]] = defaultdict(list)
    comp_map = {c.comp_id: c for c in components}
    for comp_id, text in raw_text.items():
        if not text or comp_id not in comp_map:
            continue
        by_level[comp_map[comp_id].level.value].append((comp_id, len(text)))

    for level, items in by_level.items():
        if len(items) < 3:
            continue
        lengths = [length for _, length in items]
        med = statistics.median(lengths)
        mad = statistics.median([abs(l - med) for l in lengths])
        threshold = med + _D1_MAD_FACTOR * max(mad, 1)

        for comp_id, length in items:
            if length > threshold:
                comp = comp_map.get(comp_id)
                citation = comp.citation if comp else comp_id
                warnings.append(StructuralWarning(
                    rule="D1",
                    norm_id=norm_id,
                    comp_id=comp_id,
                    severity="warn",
                    message=(
                        f"{level} {citation!r}: nội dung {length} ký tự "
                        f"bất thường (ngưỡng {threshold:.0f}, median {med:.0f}) "
                        f"— có thể Điều/Khoản bên trong bị nuốt thành plain text"
                    ),
                    details={
                        "text_len": length,
                        "median": round(med),
                        "mad": round(mad),
                        "threshold": round(threshold),
                    },
                ))


# ─── E1: Nhảy cấp bất hợp lệ — bug parser ───────────────────────────────────

def _check_e1(
    norm_id: str,
    components: list[Component],
    children_map: dict[Optional[str], list[Component]],
    warnings: list[StructuralWarning],
) -> None:
    """E1: Parent và child nhảy quá 1 tầng trong hierarchy — bug logic parser."""
    comp_by_id = {c.comp_id: c for c in components}
    for parent_id, children in children_map.items():
        if parent_id is None:
            continue
        parent = comp_by_id.get(parent_id)
        if parent is None:
            continue
        p_rank = LEVEL_RANK.get(parent.level.value, 99)

        for child in children:
            c_rank = LEVEL_RANK.get(child.level.value, 99)
            if c_rank - p_rank > 2:
                warnings.append(StructuralWarning(
                    rule="E1",
                    norm_id=norm_id,
                    comp_id=child.comp_id,
                    severity="bug",
                    message=(
                        f"Nhảy cấp bất hợp lệ: {parent.level.value} {parent.citation!r} "
                        f"→ {child.level.value} {child.citation!r} "
                        f"(khoảng cách rank={c_rank - p_rank}, tối đa cho phép=2). "
                        f"Bug logic parser — sửa code, không patch dữ liệu."
                    ),
                    details={
                        "parent_id": parent_id,
                        "parent_level": parent.level.value,
                        "child_id": child.comp_id,
                        "child_level": child.level.value,
                        "rank_gap": c_rank - p_rank,
                    },
                ))
