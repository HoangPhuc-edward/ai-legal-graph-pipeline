"""Orchestrator Tầng A (luôn tạo) + Tầng B (chỉ khi khớp được cả 2 đầu).

Đọc relationships.parquet (cột doc_id, other_doc_id, relationship), xử lý
theo đúng thứ tự: tạo Tầng A trước, làm giàu bằng Tầng B sau (Pass 2 — cần
component_index đã build xong ở Pass 1, xem transform/pipeline.py).
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Callable, Iterator, Optional

from . import action_extractor
from schema.edges import NormRelation
from schema.enums import ELIGIBLE_FOR_LAYER_B, RelationType
from schema.nodes import Action, TextUnit

logger = logging.getLogger(__name__)

# Nhãn tiếng Việt -> RelationType canonical. Nhãn chiều ngược (vd "được sửa đổi
# bởi") được chuẩn hoá về đúng 1 chiều bằng cách đảo from/to khi xử lý, không
# tạo enum riêng cho chiều ngược.
RELATION_LABEL_MAP: dict[str, RelationType] = {
    "Văn bản sửa đổi": RelationType.AMENDS,
    "Văn bản bổ sung": RelationType.SUPPLEMENTS,
    "Văn bản hết hiệu lực": RelationType.TERMINATES,
    "Văn bản bị hết hiệu lực 1 phần": RelationType.PARTIALLY_TERMINATES,
    "Văn bản đình chỉ": RelationType.SUSPENDS,
    "Văn bản đình chỉ 1 phần": RelationType.PARTIALLY_SUSPENDS,
    "Văn bản HD, QĐ chi tiết": RelationType.IMPLEMENTS,
    "Văn bản dẫn chiếu": RelationType.REFERS_TO,
    "Văn bản liên quan khác": RelationType.RELATED_TO,
    "Văn bản căn cứ": RelationType.CITES,
}

# Nhãn chiều ngược -> RelationType canonical (đảo from/to)
REVERSE_RELATION_LABEL_MAP: dict[str, RelationType] = {
    "Văn bản được sửa đổi": RelationType.AMENDS,
    "Văn bản được bổ sung": RelationType.SUPPLEMENTS,
    "Văn bản bị hết hiệu lực": RelationType.TERMINATES,
    "Văn bản làm hết hiệu lực 1 phần": RelationType.PARTIALLY_TERMINATES,
    "Văn bản bị đình chỉ": RelationType.SUSPENDS,
    "Văn bản làm đình chỉ 1 phần": RelationType.PARTIALLY_SUSPENDS,
    "Văn bản được HD, QĐ chi tiết": RelationType.IMPLEMENTS,
    "Văn bản được dẫn chiếu": RelationType.REFERS_TO,
    "Văn bản căn cứ bởi": RelationType.CITES,
}


def _resolve_relation(doc_id: str, other_doc_id: str, label: str) -> Optional[tuple[str, str, RelationType]]:
    """Trả về (from_norm_id, to_norm_id, relation_type) đã chuẩn hoá chiều, hoặc None nếu nhãn lạ."""
    label = label.strip()
    if label in RELATION_LABEL_MAP:
        return doc_id, other_doc_id, RELATION_LABEL_MAP[label]
    if label in REVERSE_RELATION_LABEL_MAP:
        return other_doc_id, doc_id, REVERSE_RELATION_LABEL_MAP[label]
    logger.warning("Nhãn quan hệ không xác định: '%s' (doc_id=%s)", label, doc_id)
    return None


# Kết quả Tầng B: (Action, cache TextUnit, comp_a_id, comp_b_id)
ActionResult = tuple[Action, TextUnit, str, str]


def process_relationship_row(
    doc_id: str,
    other_doc_id: str,
    relationship_label: str,
    component_index: dict[tuple[str, str], str],
    get_component_text_map: Optional[Callable[[str], dict[str, str]]] = None,
    lookup_norm_number: Optional[Callable[[str], str]] = None,
    use_llm: bool = True,
) -> Iterator[NormRelation | ActionResult]:
    """Sinh 1 NormRelation (Tầng A, luôn có) và 0..n ActionResult (Tầng B — 1
    Action cho mỗi cặp Component A/Component B khớp được)."""
    resolved = _resolve_relation(doc_id, other_doc_id, relationship_label)
    if resolved is None:
        return
    from_norm_id, to_norm_id, relation_type = resolved

    # TẦNG A — luôn tạo, không cần regex/LLM, map 1:1
    yield NormRelation(from_norm_id=from_norm_id, to_norm_id=to_norm_id, relation_type=relation_type)

    # TẦNG B — chỉ thử khi loại quan hệ có khái niệm "Component cụ thể"
    if relation_type not in ELIGIBLE_FOR_LAYER_B:
        return
    if get_component_text_map is None:
        return

    component_text = get_component_text_map(from_norm_id)
    if not component_text:
        return

    for comp_a_id, citation_path in action_extractor.find_amendments(
        doc_id=from_norm_id,
        component_text=component_text,
        other_doc_id=to_norm_id,
        component_index=component_index,
        use_llm=use_llm,
    ):
        comp_b_id = component_index.get((to_norm_id, citation_path))
        if comp_b_id is None:
            continue  # không khớp được Component B -> bỏ qua, chỉ giữ Tầng A

        now = datetime.now(timezone.utc)
        amending_doc_number = lookup_norm_number(from_norm_id) if lookup_norm_number else from_norm_id

        cache_text_unit = TextUnit(
            unit_id=f"action-cache-{uuid.uuid4().hex}",
            accumulated_text=component_text[comp_a_id],
            type="cache_action",
            embedding=None,  # KHÔNG embed — bản sao y nguyên của TextUnit Component A
            updated_at=now,
        )
        action = Action(
            action_id=f"action-{uuid.uuid4().hex}",
            relation_type=relation_type,
            amending_doc_number=amending_doc_number,
            updated_at=now,
        )
        yield action, cache_text_unit, comp_a_id, comp_b_id
