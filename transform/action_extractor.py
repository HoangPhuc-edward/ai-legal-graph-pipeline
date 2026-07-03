"""Tầng B — regex-first, LLM chỉ fallback. Không quyết định có tạo quan hệ hay
không (việc đó luôn xảy ra ở Tầng A) — chỉ cố khớp được Component B (đích) cụ
thể hay không, theo khuôn mẫu cố định của Nghị định 34/2016/NĐ-CP (sửa đổi bởi
154/2020/NĐ-CP).

Component A (nguồn) đã biết — chính là Component đang được duyệt qua trong
doc_id (văn bản sửa đổi luôn trình bày "Điều X. Sửa đổi ... như sau: ..." nên
Component A = chính Component chứa câu văn đó). Hàm này chỉ cần tìm citation
của Component B trong câu văn, rồi khớp với `component_index` đã build ở Pass 1
(xem transform/pipeline.py) — KHÔNG parse lại văn bản đích.

Quy tắc leo thang 3 tầng (mỗi tầng đều thử khớp `component_index` ngay sau khi
trích được citation — trích được nhưng không khớp được Component B nào ĐÃ BIẾT
trong corpus thì vẫn leo thang tiếp, vì có thể câu văn lệch khuôn mẫu regex chứ
không phải citation sai):
  [1] 4 regex pattern (SUA_DOI/BO_SUNG, BAI_BO, THAY_THE_CUM_TU, BO_CUM_TU) — 0 token
  [2] gemini-2.5-flash (LLM_MODEL_LIGHT) — câu rõ ràng nhưng lệch khuôn mẫu
  [3] gemini-3.5-flash (LLM_MODEL_HEAVY) — câu phức, nhiều mệnh đề lồng nhau
Không khớp được ở cả 3 bước KHÔNG phải lỗi — Tầng A vẫn đầy đủ, chỉ thiếu Action.
"""
from __future__ import annotations

import logging
import re
from typing import Iterator, Optional

from config import LLM_MODEL_HEAVY, LLM_MODEL_LIGHT

logger = logging.getLogger(__name__)

# Cụm vị trí "[Điểm x] [Khoản y] Điều z" — Điều luôn bắt buộc, Khoản/Điểm tùy chọn,
# đi TRƯỚC Điều trong câu văn pháp luật thật (vd "khoản 4 Điều 38").
_LOC_PHRASE = (
    r"(?:Điểm\s+(?P<diem>[a-zđ])\)?\s+)?"
    r"(?:Khoản\s+(?P<khoan>\d+[a-zđ]?)\s+)?"
    r"Điều\s+(?P<dieu>\d+[a-zđ]?)"
)

PATTERN_SUA_DOI_BO_SUNG = re.compile(
    rf"(?:Sửa đổi|Bổ sung|sửa đổi|bổ sung)\s+{_LOC_PHRASE}", re.IGNORECASE
)
PATTERN_BAI_BO = re.compile(rf"(?:Bãi bỏ|bãi bỏ)\s+{_LOC_PHRASE}", re.IGNORECASE)
PATTERN_THAY_THE_CUM_TU = re.compile(
    rf'(?:Thay thế|thay thế) cụm từ\s*["“][^"”]+["”]\s*bằng cụm từ\s*["“][^"”]+["”]\s+tại\s+{_LOC_PHRASE}',
    re.IGNORECASE,
)
PATTERN_BO_CUM_TU = re.compile(
    rf'(?:Bỏ|bỏ) cụm từ\s*["“][^"”]+["”]\s+tại\s+{_LOC_PHRASE}', re.IGNORECASE
)

REGEX_PATTERNS = (
    PATTERN_SUA_DOI_BO_SUNG,
    PATTERN_BAI_BO,
    PATTERN_THAY_THE_CUM_TU,
    PATTERN_BO_CUM_TU,
)


def _normalize_citation_path(match: re.Match) -> Optional[str]:
    """Build citation_path đúng định dạng key của component_index, vd
    "Khoản 4 > Điều 38" hoặc "Điểm a > Khoản 4 > Điều 38" — thứ tự cụ thể
    nhất trước (Điểm > Khoản > Điều), khớp với cách build ở pipeline.py
    (cùng nhãn "Điểm/Khoản/Điều {định danh}" như structure_parser._build_citation)."""
    dieu = match.group("dieu")
    if not dieu:
        return None
    parts = []
    if match.group("diem"):
        parts.append(f"Điểm {match.group('diem')}")
    if match.group("khoan"):
        parts.append(f"Khoản {match.group('khoan')}")
    parts.append(f"Điều {dieu}")
    return " > ".join(parts)


def _extract_with_regex(text: str) -> Optional[str]:
    for pattern in REGEX_PATTERNS:
        m = pattern.search(text)
        if not m:
            continue
        citation_path = _normalize_citation_path(m)
        if citation_path:
            return citation_path
    return None


def _extract_with_llm(text: str, model_name: str, other_doc_id: str) -> Optional[str]:
    """Gọi Vertex AI Gemini để trích citation_path của Component B. Trả None nếu mơ hồ/lỗi."""
    try:
        from google import genai
        from config import GCP_LOCATION, GCP_PROJECT

        client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
        prompt = (
            "Trích vị trí (Điều/Khoản/Điểm) bị thay đổi từ đoạn văn bản pháp luật sau, "
            f"thuộc văn bản đích có id '{other_doc_id}'. Trả lời đúng định dạng:\n"
            'CITATION: "Khoản X > Điều Y" (hoặc "Điểm x > Khoản X > Điều Y", hoặc chỉ "Điều Y")\n'
            "Nếu không xác định được, trả lời UNKNOWN.\n\n"
            f"Đoạn văn:\n{text}"
        )
        response = client.models.generate_content(model=model_name, contents=prompt)
        content = (response.text or "").strip()
        if "UNKNOWN" in content.upper():
            return None
        citation_match = re.search(r"CITATION:\s*\"?([^\"\n]+)\"?", content)
        if not citation_match:
            return None
        return citation_match.group(1).strip()
    except Exception:
        logger.exception("LLM fallback (%s) thất bại, coi như không trích được.", model_name)
        return None


def find_amendments(
    doc_id: str,
    component_text: dict[str, str],
    other_doc_id: str,
    component_index: dict[tuple[str, str], str],
    use_llm: bool = True,
    known_norm_ids: Optional[set[str]] = None,
) -> Iterator[tuple[str, str]]:
    """Duyệt qua text của từng Component LÁ trong doc_id (Component A ứng viên),
    cố khớp được citation_path của Component B đã tồn tại trong component_index
    (đã build ở Pass 1 cho other_doc_id). Yield (comp_a_id, citation_path) cho
    mỗi Component A khớp được — KHÔNG tự resolve thành comp_b_id (việc đó do
    relation_classifier làm, dùng cùng component_index).

    known_norm_ids: set các norm_id đã có Component được index (thường build 1
    lần từ component_index — xem relation_classifier). Nếu other_doc_id KHÔNG
    nằm trong tập này (ví dụ: chạy --sample nhỏ và văn bản đích không thuộc
    mẫu), thì KHÔNG THỂ khớp được dù trích citation đúng đến đâu — bỏ qua toàn
    bộ regex/LLM ngay, tránh tốn lệnh gọi LLM chắc chắn vô ích.
    """
    if known_norm_ids is not None and other_doc_id not in known_norm_ids:
        logger.info(
            "Văn bản đích %s không có Component nào trong component_index "
            "(ngoài phạm vi --sample hoặc chưa parse được) — bỏ qua Tầng B cho doc %s, chỉ giữ Tầng A.",
            other_doc_id, doc_id,
        )
        return

    for comp_a_id, text in component_text.items():
        if not text or not text.strip():
            continue

        citation_path = _extract_with_regex(text)
        if citation_path and (other_doc_id, citation_path) in component_index:
            yield comp_a_id, citation_path
            continue

        if not use_llm:
            continue

        citation_path = _extract_with_llm(text, LLM_MODEL_LIGHT, other_doc_id)
        if citation_path and (other_doc_id, citation_path) in component_index:
            yield comp_a_id, citation_path
            continue

        citation_path = _extract_with_llm(text, LLM_MODEL_HEAVY, other_doc_id)
        if citation_path and (other_doc_id, citation_path) in component_index:
            yield comp_a_id, citation_path
            continue

        logger.info(
            "Component %s (doc %s) có vẻ đề cập %s nhưng không khớp được Component B đã biết — chỉ giữ Tầng A.",
            comp_a_id,
            doc_id,
            other_doc_id,
        )