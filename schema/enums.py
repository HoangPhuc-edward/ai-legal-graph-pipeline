from enum import Enum


class ComponentLevel(str, Enum):
    PHAN = "Phan"
    CHUONG = "Chuong"
    MUC = "Muc"
    TIEU_MUC = "TieuMuc"
    DIEU = "Dieu"
    KHOAN = "Khoan"
    DIEM = "Diem"


class RelationType(str, Enum):
    """10 loại quan hệ — DÙNG CHUNG cho cả Tầng A (NormRelation) và Tầng B (Action).

    Không còn 2 enum tách biệt (ActionType + DirectRelationType) — chỉ 1 enum
    duy nhất, vì A và B biểu diễn CÙNG MỘT quan hệ ở 2 mức độ chi tiết khác nhau.
    """

    AMENDS = "AMENDS"  # sửa đổi
    SUPPLEMENTS = "SUPPLEMENTS"  # bổ sung
    TERMINATES = "TERMINATES"  # hết hiệu lực (toàn bộ)
    PARTIALLY_TERMINATES = "PARTIALLY_TERMINATES"  # hết hiệu lực 1 phần
    SUSPENDS = "SUSPENDS"  # đình chỉ
    PARTIALLY_SUSPENDS = "PARTIALLY_SUSPENDS"  # đình chỉ 1 phần
    IMPLEMENTS = "IMPLEMENTS"  # hướng dẫn/quy định chi tiết
    REFERS_TO = "REFERS_TO"  # dẫn chiếu
    RELATED_TO = "RELATED_TO"  # liên quan khác
    CITES = "CITES"  # căn cứ


# 4/10 loại có khái niệm "nội dung thay đổi ở cấp Điều/Khoản/Điểm cụ thể".
# TERMINATES/SUSPENDS bị loại: chúng terminate/suspend cả văn bản, không có
# citation Điều/Khoản → LLM call lãng phí, luôn trả UNKNOWN.
# PARTIALLY_TERMINATES/PARTIALLY_SUSPENDS giữ lại vì "1 phần" có thể chỉ
# hết hiệu lực/đình chỉ những Điều/Khoản cụ thể.
ELIGIBLE_FOR_LAYER_B = {
    RelationType.AMENDS,
    RelationType.SUPPLEMENTS,
    RelationType.PARTIALLY_TERMINATES,
    RelationType.PARTIALLY_SUSPENDS,
}
