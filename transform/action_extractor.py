"""Tầng B — regex-first, LLM chỉ fallback async. Không quyết định có tạo quan hệ
hay không (việc đó luôn xảy ra ở Tầng A) — chỉ cố khớp được Component B (đích)
cụ thể hay không, theo khuôn mẫu cố định của Nghị định 34/2016/NĐ-CP (sửa đổi
bởi 154/2020/NĐ-CP).

Component A (nguồn) đã biết — chính là Component đang được duyệt qua trong
doc_id (văn bản sửa đổi luôn trình bày "Điều X. Sửa đổi ... như sau: ..." nên
Component A = chính Component chứa câu văn đó). Hàm này chỉ cần tìm citation
của Component B trong câu văn, rồi khớp với `component_index` đã build ở Pass 1
(xem transform/pipeline.py) — KHÔNG parse lại văn bản đích.

Quy tắc leo thang 2 tầng:
  [1] 4 regex pattern (SUA_DOI/BO_SUNG, BAI_BO, THAY_THE_CUM_TU, BO_CUM_TU) — 0 token
  [2] LLM_MODEL_HEAVY — câu phức, nhiều mệnh đề lồng nhau
Trong cùng 1 văn bản, TẤT CẢ Component cần LLM được gọi ĐỒNG THỜI qua
asyncio.gather() + Semaphore (tối đa MAX_CONCURRENT_LLM request bay cùng lúc)
thay vì tuần tự — giảm thời gian xử lý từ O(N*latency) xuống O(latency).
Không khớp được ở cả 2 bước KHÔNG phải lỗi — Tầng A vẫn đầy đủ, chỉ thiếu Action.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from typing import Any, Iterator, Optional

from config import LLM_MODEL_HEAVY

logger = logging.getLogger(__name__)

MAX_CONCURRENT_LLM = 10

# Stats tích lũy qua toàn bộ Pass 2 — để in ra sau khi transform xong.
# Key: "regex_hit", "regex_miss_no_llm", "llm_ok", "llm_unknown", "rate_limited", "llm_error"
_stats: Counter = Counter()
_llm_times: list[float] = []


def log_stats() -> None:
    """In stats tích lũy — gọi từ pipeline.py sau khi Pass 2 xong."""
    total = sum(_stats.values())
    if total == 0:
        logger.info("[stats] Không có component nào được xử lý.")
        return
    llm_total = _stats["llm_ok"] + _stats["llm_unknown"] + _stats["rate_limited"] + _stats["llm_error"]
    regex_pct = 100 * _stats["regex_hit"] / total if total else 0
    skip_pct = 100 * _stats["prefilter_skip"] / total if total else 0
    avg_llm = sum(_llm_times) / len(_llm_times) if _llm_times else 0
    logger.info(
        "[stats] Tổng component: %d | regex_hit: %d (%.0f%%) | "
        "prefilter_skip: %d (%.0f%%) | llm_gọi: %d | "
        "llm_ok: %d | llm_unknown: %d | rate_limited: %d | llm_error: %d | avg LLM: %.2fs",
        total,
        _stats["regex_hit"], regex_pct,
        _stats["prefilter_skip"], skip_pct,
        llm_total,
        _stats["llm_ok"],
        _stats["llm_unknown"],
        _stats["rate_limited"],
        _stats["llm_error"],
        avg_llm,
    )

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
    rf'(?:Thay thế|thay thế) cụm từ\s*[""][^""]+[""]\s*bằng cụm từ\s*[""][^""]+[""]\s+tại\s+{_LOC_PHRASE}',
    re.IGNORECASE,
)
PATTERN_BO_CUM_TU = re.compile(
    rf'(?:Bỏ|bỏ) cụm từ\s*[""][^""]+[""]\s+tại\s+{_LOC_PHRASE}', re.IGNORECASE
)

REGEX_PATTERNS = (
    PATTERN_SUA_DOI_BO_SUNG,
    PATTERN_BAI_BO,
    PATTERN_THAY_THE_CUM_TU,
    PATTERN_BO_CUM_TU,
)

# Pre-filter: text phải chứa keyword sửa đổi KÈM số Điều trong vòng 300 ký tự.
# Lọc trước khi gọi LLM — loại bỏ "bầu cử bổ sung", "bãi bỏ cả văn bản",
# "thay thế Quyết định số X" (không có Điều/Khoản cụ thể).
_AMENDMENT_PREFILTER = re.compile(
    r"(?:sửa đổi|bổ sung|bãi bỏ|thay thế|bỏ cụm).{0,300}Điều\s+\d",
    re.IGNORECASE | re.DOTALL,
)

# Client được tạo 1 lần, dùng lại cho mọi LLM call — tránh overhead init per-call.
_genai_client: Optional[Any] = None


def _get_client() -> Any:
    global _genai_client
    if _genai_client is None:
        from google import genai
        from config import GCP_LOCATION, GCP_PROJECT
        _genai_client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_LOCATION)
    return _genai_client


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


async def _extract_with_llm_async(
    semaphore: asyncio.Semaphore,
    text: str,
    model_name: str,
    other_doc_id: str,
) -> Optional[str]:
    """Gọi LLM async với semaphore — tối đa MAX_CONCURRENT_LLM request cùng lúc.
    Trả None nếu mơ hồ/lỗi."""
    async with semaphore:
        t0 = time.monotonic()
        try:
            client = _get_client()
            prompt = (
                "Trích vị trí (Điều/Khoản/Điểm) bị thay đổi từ đoạn văn bản pháp luật sau, "
                f"thuộc văn bản đích có id '{other_doc_id}'. Trả lời đúng định dạng:\n"
                'CITATION: "Khoản X > Điều Y" (hoặc "Điểm x > Khoản X > Điều Y", hoặc chỉ "Điều Y")\n'
                "Nếu không xác định được, trả lời UNKNOWN.\n\n"
                f"Đoạn văn:\n{text}"
            )
            response = await client.aio.models.generate_content(model=model_name, contents=prompt)
            elapsed = time.monotonic() - t0
            _llm_times.append(elapsed)

            content = (response.text or "").strip()
            if "UNKNOWN" in content.upper():
                _stats["llm_unknown"] += 1
                return None
            citation_match = re.search(r"CITATION:\s*\"?([^\"\n]+)\"?", content)
            if not citation_match:
                _stats["llm_unknown"] += 1
                return None
            _stats["llm_ok"] += 1
            return citation_match.group(1).strip()
        except Exception as exc:
            elapsed = time.monotonic() - t0
            _llm_times.append(elapsed)
            exc_str = str(exc)
            if "429" in exc_str or "quota" in exc_str.lower() or "rate" in exc_str.lower():
                _stats["rate_limited"] += 1
                logger.warning("LLM rate-limited (429) sau %.1fs — %s", elapsed, exc_str[:120])
            else:
                _stats["llm_error"] += 1
                logger.exception("LLM async fallback (%s) thất bại sau %.1fs.", model_name, elapsed)
            return None


async def find_amendments_async(
    semaphore: asyncio.Semaphore,
    doc_id: str,
    component_text: dict[str, str],
    other_doc_id: str,
    component_index: dict[tuple[str, str], str],
    use_llm: bool = True,
    known_norm_ids: Optional[set[str]] = None,
) -> list[tuple[str, str]]:
    """Async — tất cả LLM call trong 1 văn bản chạy đồng thời qua asyncio.gather(),
    giới hạn bởi semaphore dùng chung toàn bộ Pass 2.

    Luồng xử lý:
      1. Regex pass (sync, 0 token) — duyệt tuần tự qua từng Component.
      2. LLM pass (async concurrent) — chỉ những Component regex không khớp được.
         Tất cả LLM task được gather() cùng lúc, semaphore điều phối slot.
    """
    if known_norm_ids is not None and other_doc_id not in known_norm_ids:
        return []

    results: list[tuple[str, str]] = []
    needs_llm: list[tuple[str, str]] = []  # (comp_a_id, text)

    for comp_a_id, text in component_text.items():
        if not text or not text.strip():
            continue
        citation_path = _extract_with_regex(text)
        if citation_path and (other_doc_id, citation_path) in component_index:
            _stats["regex_hit"] += 1
            results.append((comp_a_id, citation_path))
        elif use_llm and _AMENDMENT_PREFILTER.search(text):
            # Pre-filter: chỉ gọi LLM khi text có keyword sửa đổi KÈM "Điều N"
            needs_llm.append((comp_a_id, text))
        else:
            _stats["prefilter_skip"] += 1

    if not needs_llm:
        return results

    llm_tasks = [
        _extract_with_llm_async(semaphore, text, LLM_MODEL_HEAVY, other_doc_id)
        for _, text in needs_llm
    ]
    llm_results = await asyncio.gather(*llm_tasks)

    for (comp_a_id, _), citation_path in zip(needs_llm, llm_results):
        if citation_path and (other_doc_id, citation_path) in component_index:
            results.append((comp_a_id, citation_path))

    return results


def find_amendments(
    doc_id: str,
    component_text: dict[str, str],
    other_doc_id: str,
    component_index: dict[tuple[str, str], str],
    use_llm: bool = True,
    known_norm_ids: Optional[set[str]] = None,
) -> Iterator[tuple[str, str]]:
    """Sync wrapper — dùng cho tests. Pipeline dùng find_amendments_async trực tiếp.

    Tạo semaphore + event loop riêng qua asyncio.run() nên không cần event loop
    đang chạy ở caller — an toàn với pytest sync."""
    async def _run() -> list[tuple[str, str]]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_LLM)
        return await find_amendments_async(
            semaphore=semaphore,
            doc_id=doc_id,
            component_text=component_text,
            other_doc_id=other_doc_id,
            component_index=component_index,
            use_llm=use_llm,
            known_norm_ids=known_norm_ids,
        )

    yield from asyncio.run(_run())
