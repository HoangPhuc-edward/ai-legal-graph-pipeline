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

# Thứ tự ưu tiên CỐ ĐỊNH — PHU_LUC kiểm tra trước, rồi Phần, Chương, Mục, Điều, Khoản, Điểm.
# PHU_LUC dùng named group pl_id để tách định danh số thứ tự sau "Phụ lục":
#   (?P<pl_id>...) — "số I", "Số 1", roman, arabic, hoặc chữ hoa đơn (A, B, C)
LEVEL_PATTERNS: list[tuple[ComponentLevel, re.Pattern]] = [
    # pl_id alternation (thứ tự ưu tiên quan trọng):
    #   1. "[Ss]ố + space + roman/digit"  → "số I", "Số 1" (phải match đủ cả "số X")
    #   2. Roman numeral                  → I, II, III, XXIV
    #   3. Arabic digit                   → 1, 2, 10
    #   4. Single uppercase letter + boundary → A, B, C (nhưng KHÔNG phải "S" từ "Số 1")
    (ComponentLevel.PHU_LUC, re.compile(
        r"^\s*(?:Phụ\s+lục|PHỤ\s+LỤC)\s*"
        r"(?P<pl_id>[Ss]ố\s+(?:[IVXLCDM]+|\d+)|[IVXLCDM]+|\d+|[A-ZĐ](?=\s|$))?",
        re.UNICODE,
    )),
    (ComponentLevel.PHAN, re.compile(r"^\s*Phần\s+(thứ\s+)?([IVXLCDM\d]+)\b", re.IGNORECASE)),
    (ComponentLevel.CHUONG, re.compile(r"^\s*Chương\s+([IVXLCDM\d]+)\b", re.IGNORECASE)),
    (ComponentLevel.MUC, re.compile(r"^\s*(Mục|Tiểu mục)\s+(\d+)", re.IGNORECASE)),
    (ComponentLevel.DIEU, re.compile(r"^\s*[Đđ]iều\s+(\d+[a-zđ]?)\s*[\.\:]?")),
    (ComponentLevel.KHOAN, re.compile(r"^\s*(\d{1,2})\.\s+")),
    (ComponentLevel.DIEM, re.compile(r"^\s*([a-zđ])\)\s+")),
]

_LEVEL_LABEL = {
    ComponentLevel.PHU_LUC: "Phụ lục",
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
    re.compile(r"(?:Phụ\s+lục|PHỤ\s+LỤC)\b", re.UNICODE),
    re.compile(r"Phần\s+(thứ\s+)?([IVXLCDM\d]+)\b", re.IGNORECASE),
    re.compile(r"Chương\s+([IVXLCDM\d]+)\b", re.IGNORECASE),
    re.compile(r"(Mục|Tiểu mục)\s+(\d+)", re.IGNORECASE),
    # Có dấu chấm/hai chấm sau số — an toàn kể cả nội dung câu ("tại Điều 5.")
    re.compile(r"Điều\s+(\d+[a-zđ]?)\s*[\.\:]", re.IGNORECASE),
    # Văn bản cũ KHÔNG có dấu câu sau số Điều — chỉ split khi ký tự TRƯỚC Điều
    # là non-space (CHUNGĐiều, kín.Điều) để tránh false positive "tại Điều 5 của
    # Luật" (có space trước Điều → lookbehind không khớp).
    re.compile(r"(?<=\S)Điều\s+(\d+[a-zđ]?)\s*[\.\:]?", re.IGNORECASE),
]

# Dùng để phát hiện trailing content là Khoản/Điểm — quyết định có tách sau marker không
_KHOAN_AT_START = re.compile(r"^\s*\d{1,2}\.\s+")
_DIEM_AT_START = re.compile(r"^\s*[a-zđ]\)\s+")

# Từ dẫn trích dẫn — nếu prefix kết thúc bằng một trong các từ này thì marker
# phía sau là citation (trích dẫn), KHÔNG phải structural header.
# Ví dụ: "theo quy định tại Điều 5" → "tại " ở cuối prefix → không tách.
_CITATION_LEADERS = re.compile(
    r"(?:tại|theo|trong|của|nêu\s+tại|quy\s+định|căn\s+cứ|đề\s+cập|"
    r"khoản\s+\d+(?:\s+[a-zđ]\))?)\s*$",
    re.IGNORECASE | re.UNICODE,
)
# Ranh giới câu ngay trước marker → marker là structural (khởi đầu câu mới).
# Ví dụ: "...Nhà nước. Điều 2." → "." trước "Điều" → tách.
_SENTENCE_END = re.compile(r"[.;:]\s*$")


def _split_line_at_headers(line: str) -> list[str]:
    """Tách 1 dòng thành nhiều dòng nếu có marker cấp (Phần/Chương/Mục/Điều) bị
    dính giữa dòng.

    Xử lý 2 case:
      A) marker bị dính SAU nội dung (m.start() > 0): tách TRƯỚC marker.
         Thu thập TẤT CẢ điểm split trong 1 lần scan qua 5 pattern — tách đồng
         loạt thay vì while-loop cũ chỉ tách 1 điểm mỗi vòng (O(N²) → O(N)).
         Tách trừ khi có từ dẫn trích dẫn ngay trước marker (tại/theo/trong/...).
      B) marker ở ĐẦU dòng nhưng trailing là Khoản/Điểm: tách SAU marker.
         Chỉ áp dụng khi không có case A nào (ưu tiên A > B).
         Ví dụ: 'Điều 2. 1. Thu ngân sách...' -> ['Điều 2.', '1. Thu ngân sách...']
         Không tách: 'Điều 1. Phạm vi điều chỉnh' (trailing là title_text hợp lệ).

    Quy tắc ngữ cảnh cho Case A:
      Từ dẫn trích dẫn (tại/theo/trong/quy định/...) → citation, KHÔNG tách.
      Mọi trường hợp khác → tách (giữ hành vi gốc, bổ sung bảo vệ citation).
    """
    split_points: set[int] = set()
    case_b_split: Optional[int] = None

    for pattern in _FORCE_BREAK_PATTERNS:
        pos = 0
        checked_b = False
        while True:
            m = pattern.search(line, pos)
            if not m:
                break
            if m.start() > 0:
                # Case A — marker giữa dòng: tách trừ khi có từ dẫn trích dẫn trước
                prefix_s = line[:m.start()].rstrip()
                if not _CITATION_LEADERS.search(prefix_s):
                    split_points.add(m.start())
            elif not checked_b:
                # Case B — marker ở đầu dòng, chỉ kiểm tra lần đầu mỗi pattern
                trailing = line[m.end():]
                if _KHOAN_AT_START.match(trailing) or _DIEM_AT_START.match(trailing):
                    case_b_split = m.end()
                checked_b = True
            pos = m.end() if m.end() > pos else pos + 1

    if split_points:
        prev = 0
        segments: list[str] = []
        for bp in sorted(split_points):
            seg = line[prev:bp]
            if seg.strip():
                segments.append(seg)
            prev = bp
        if line[prev:].strip():
            segments.append(line[prev:])
        return segments

    if case_b_split is not None:
        return [s for s in [line[:case_b_split], line[case_b_split:]] if s.strip()]

    return [line] if line.strip() else []


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
    # Bước 1: Chuyển **Điều X** → \nĐiều X TRƯỚC khi xóa ** — để marker nằm đầu
    # dòng riêng thay vì bị "nuốt" vào raw_text của Chương cha. Văn bản cũ (Luật,
    # Sắc lệnh trước 2000) bọc mỗi Điều trong bold: "**Điều 1**Nội dung..." — sau
    # khi xóa ** thành space chỉ còn " Điều 1 Nội dung" không tách được.
    text = re.sub(
        r"\*+\s*(Điều\s+\d+[a-zđ]?\s*\.?)\s*\*+",
        r"\n\1",
        text,
        flags=re.IGNORECASE,
    )
    # Bước 2: "**"/"__" còn lại chỉ là định dạng, thay bằng khoảng trắng (không
    # phải "") để tránh dán liền từ: "**Điều 49**1." → "Điều 49 1." thay vì "Điều 491."
    text = text.replace("**", " ").replace("__", " ")
    text = re.sub(r" {2,}", " ", text)

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
    if level == ComponentLevel.PHU_LUC:
        identifier = (match.group("pl_id") or "").strip()
        # Normalize "số X" / "Số X" → "X" để citation nhất quán giữa các văn bản
        identifier = re.sub(r"^[Ss]ố\s+", "", identifier).strip()
        return f"Phụ lục {identifier}".strip() if identifier else "Phụ lục"
    groups = match.groups()
    identifier = groups[-1].strip()
    # MUC pattern has 2 groups: ("Mục"|"Tiểu mục", number) — use matched keyword, not hardcoded label
    if level == ComponentLevel.MUC:
        return f"{groups[0]} {identifier}"
    return f"{_LEVEL_LABEL[level]} {identifier}"


def _build_title_text(line: str, match: re.Match) -> Optional[str]:
    remainder = line[match.end():].strip(" .:-\t")
    return remainder or None


_DIEU_NUM_PRESCAN = re.compile(r"^\s*[Đđ]iều\s+(\d+)")
_OUTLIER_JUMP = 20   # nhảy ≥ N so với Điều trước → nghi ngờ citation bị tách nhầm
_OUTLIER_RECOVER = 5  # Điều kế tiếp phải gần với Điều trước ≤ N → xác nhận outlier


def _scan_dieu_outliers(lines: list[str]) -> frozenset[int]:
    """Pre-scan các dòng đã normalize: tìm line index có số Điều nhảy bất thường.

    Pattern bắt được: [..., 3, 4, 47, 5, 6, ...] — 47 nhảy lớn rồi sequence trở về
    gần giá trị trước → 47 là citation "theo Điều 47" bị tách nhầm thành structural.

    Điều kiện outlier (cả 3 phải đúng):
      1. curr - prev > _OUTLIER_JUMP   (nhảy lớn)
      2. nxt < curr                    (sequence trở về)
      3. nxt - prev <= _OUTLIER_RECOVER (trở về gần prev)

    Không nhầm với reset hợp lệ ở Phụ lục (đã xử lý riêng bởi ghost removal).
    """
    hits: list[tuple[int, int]] = []
    for i, line in enumerate(lines):
        m = _DIEU_NUM_PRESCAN.match(line)
        if m:
            hits.append((i, int(m.group(1))))

    if len(hits) < 3:
        return frozenset()

    nums = [n for _, n in hits]
    outliers: set[int] = set()
    for i in range(1, len(hits) - 1):
        prev, curr, nxt = nums[i - 1], nums[i], nums[i + 1]
        if curr - prev > _OUTLIER_JUMP and nxt < curr and nxt - prev <= _OUTLIER_RECOVER:
            outliers.add(hits[i][0])
    return frozenset(outliers)


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
    preamble_lines: list[str] = []  # text trước Component đầu tiên

    all_lines = markdown.splitlines()
    # Pre-scan: tìm các dòng Điều có số nhảy bất thường (citation bị tách nhầm)
    skip_indices = _scan_dieu_outliers(all_lines)

    for line_idx, raw_line in enumerate(all_lines):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue

        # Điều số outlier → gom vào raw_text của lá hiện tại thay vì tạo Component
        if line_idx in skip_indices:
            if current_leaf_id != "__ROOT__":
                result.raw_text[current_leaf_id] = (
                    result.raw_text.get(current_leaf_id, "") + line.strip() + "\n"
                )
            else:
                preamble_lines.append(line.strip())
            continue

        matched = _match_level(line)
        if matched is None:
            # Nối vào raw_text của Component lá gần nhất đang mở
            if current_leaf_id != "__ROOT__":
                result.raw_text[current_leaf_id] = (
                    result.raw_text.get(current_leaf_id, "") + line.strip() + "\n"
                )
            else:
                preamble_lines.append(line.strip())
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
    # khác). Trước khi xóa: nếu parent có raw_text đáng kể (thường do toàn bộ
    # nội dung Điều nằm trên 1 dòng HTML, "title_text" nuốt cả body), recover
    # sang first child để không mất content.
    parent_ids = {c.parent_comp_id for c in result.components if c.parent_comp_id is not None}
    _MIN_PARENT_RECOVER = 100  # ngưỡng: bỏ qua title ngắn, chỉ recover khi thực sự là body
    for c in result.components:
        if c.comp_id not in parent_ids:
            continue
        parent_raw = result.raw_text.get(c.comp_id, "")
        if len(parent_raw.strip()) <= _MIN_PARENT_RECOVER:
            continue
        children = sorted(
            [x for x in result.components if x.parent_comp_id == c.comp_id],
            key=lambda x: x.order_index,
        )
        if not children:
            continue
        first_child_id = children[0].comp_id
        if first_child_id in result.raw_text:
            result.raw_text[first_child_id] = parent_raw + result.raw_text[first_child_id]
    result.raw_text = {
        comp_id: text for comp_id, text in result.raw_text.items() if comp_id not in parent_ids
    }

    # Nội dung preamble (trước Component đầu tiên) → prepend vào leaf đầu tiên
    if preamble_lines and result.raw_text:
        preamble_text = "\n".join(preamble_lines) + "\n"
        first_leaf_id = min(
            result.raw_text,
            key=lambda cid: next((c.order_index for c in result.components if c.comp_id == cid), 0),
        )
        result.raw_text[first_leaf_id] = preamble_text + result.raw_text[first_leaf_id]

    if not result.components:
        return _fallback_whole_document(norm_id, markdown, now)

    return result
