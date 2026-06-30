from pydantic import BaseModel

from .enums import RelationType


class Contains(BaseModel):
    parent_id: str
    child_id: str


class HasTextUnit(BaseModel):
    owner_id: str  # comp_id (TextUnit nội dung chính) HOẶC action_id (TextUnit cache của Action)
    unit_id: str


class HasAction(BaseModel):
    """Component A (nguồn — Điều/Khoản TRONG văn bản đang thực hiện sửa đổi)
    -> Action. Dùng để TRUY NGƯỢC tìm văn bản đã gây ra hành động:
    Action <-[:HAS_ACTION]- Component A <-[:CONTAINS]- ... <-[:CONTAINS]- Norm A.
    """

    comp_id: str  # Component A — nguồn
    action_id: str


class ApplyTo(BaseModel):
    """Action -> Component B (đích — Điều/Khoản BỊ tác động). Chiều truy vấn
    dùng nhiều nhất: (Component B)<-[:APPLY_TO]-(Action)."""

    action_id: str
    comp_id: str  # Component B — đích


class NormRelation(BaseModel):  # Tầng A — cạnh cơ bản, luôn được tạo
    from_norm_id: str
    to_norm_id: str
    relation_type: RelationType
