from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from .enums import ComponentLevel, RelationType


class Norm(BaseModel):
    norm_id: str
    title: str
    norm_number: str
    norm_type: str
    published_date: Optional[str] = None
    valid_from: Optional[str] = None
    valid_to: Optional[str] = None
    publisher: Optional[str] = None
    signer: Optional[str] = None
    validity_status: Optional[str] = None
    sector: Optional[str] = None
    field: Optional[str] = None
    updated_at: datetime


class Component(BaseModel):
    comp_id: str
    norm_id: str
    level: ComponentLevel
    citation: str
    order_index: int
    parent_comp_id: Optional[str] = None  # None CHỈ khi cha là Norm trực tiếp
    title_text: Optional[str] = None
    updated_at: datetime


class TextUnit(BaseModel):
    """Dùng chung cho 2 ngữ cảnh sở hữu khác nhau — KHÔNG có field version_id.

    Chủ sở hữu được xác định hoàn toàn qua cạnh HAS_TEXTUNIT lúc truy vấn,
    không cần TextUnit tự biết về mình thuộc ai.
    """

    unit_id: str
    accumulated_text: str
    type: str = "noi_dung"  # "noi_dung" (hiện hành, gắn Component) hoặc "chi_tiet_thay_doi" (gắn Action)
    language: str = "vi"
    embedding: Optional[list[float]] = None
    embedded_at: Optional[datetime] = None
    error_log: Optional[str] = None
    updated_at: datetime


class Action(BaseModel):
    """Tầng B — node CẦU NỐI giữa 2 Component (qua HasAction/ApplyTo trong
    schema/edges.py), KHÔNG sở hữu Component nào trực tiếp bằng field.

        (Component A)-[:HAS_ACTION]->(Action)-[:APPLY_TO]->(Component B)

    Component A = Điều/Khoản TRONG văn bản đang thực hiện sửa đổi (nguồn).
    Component B = Điều/Khoản TRONG văn bản đích bị tác động.

    2 field dưới đây là CACHE (bản sao bất biến, copy 1 lần lúc tạo Action) —
    tránh phải traversal dài (HAS_ACTION -> Component A -> CONTAINS -> Norm A)
    cho câu hỏi thường gặp "bị tác động bởi văn bản nào, nội dung gì":
      - amending_doc_number: copy từ Norm A.norm_number.
      - TextUnit riêng (qua HAS_TEXTUNIT từ Action, type="cache_action"):
        copy accumulated_text của Component A — KHÔNG embed riêng (đã embed
        ở TextUnit của Component A rồi).
    """

    action_id: str
    relation_type: RelationType  # dùng chung enum với NormRelation (Tầng A)
    amending_doc_number: str  # CACHE — không thay thế cạnh HAS_ACTION
    effective_date: Optional[str] = None
    description: Optional[str] = None
    updated_at: datetime
