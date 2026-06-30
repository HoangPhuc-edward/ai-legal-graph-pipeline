"""Build TextUnit.accumulated_text — rule-based, không cần LLM sinh context.

Khác Contextual Retrieval (Anthropic): hệ thống đã biết chính xác cây phân cấp
(không phải đoán), nên context ở đây là sự thật cấu trúc có sẵn — chính xác hơn
và không tốn token LLM cho bước này.
"""
from __future__ import annotations

from schema.nodes import Component, Norm


def build_accumulated_text(
    norm: Norm, ancestor_chain: list[Component], leaf_text: str
) -> str:
    """ancestor_chain: danh sách Component từ gốc tới cha trực tiếp của lá (không gồm lá)."""
    path = [norm.title] + [c.citation for c in ancestor_chain]
    context_header = " > ".join(path)
    return f"[{context_header}]\n{leaf_text}"


def build_ancestor_chain(
    leaf: Component, components_by_id: dict[str, Component]
) -> list[Component]:
    """Đi ngược parent_comp_id từ lá lên gốc, trả về danh sách tổ tiên theo thứ tự gốc -> cha trực tiếp."""
    chain: list[Component] = []
    current = leaf
    while current.parent_comp_id is not None:
        parent = components_by_id[current.parent_comp_id]
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain
