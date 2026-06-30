from schema.enums import ELIGIBLE_FOR_LAYER_B, RelationType
from transform.relation_classifier import (
    RELATION_LABEL_MAP,
    REVERSE_RELATION_LABEL_MAP,
    _resolve_relation,
)

# Bảng 17 nhãn thực tế quan sát trong relationships.parquet: (label, RelationType,
# is_reverse, eligible_for_layer_b) — đối chiếu trực tiếp với bảng map đã chốt.
EXPECTED_LABELS = [
    ("Văn bản căn cứ", RelationType.CITES, False, False),
    ("Văn bản dẫn chiếu", RelationType.REFERS_TO, False, False),
    ("Văn bản hết hiệu lực", RelationType.TERMINATES, False, True),
    ("Văn bản quy định hết hiệu lực", RelationType.TERMINATES, True, True),
    ("Văn bản được HD, QĐ chi tiết", RelationType.IMPLEMENTS, True, False),
    ("Văn bản HD, QĐ chi tiết", RelationType.IMPLEMENTS, False, False),
    ("Văn bản bổ sung", RelationType.SUPPLEMENTS, False, True),
    ("Văn bản bị hết hiệu lực 1 phần", RelationType.PARTIALLY_TERMINATES, True, True),
    ("Văn bản được sửa đổi", RelationType.AMENDS, True, True),
    ("Văn bản sửa đổi", RelationType.AMENDS, False, True),
    ("Văn bản được bổ sung", RelationType.SUPPLEMENTS, True, True),
    ("Văn bản quy định hết hiệu lực 1 phần", RelationType.PARTIALLY_TERMINATES, False, True),
    ("Văn bản liên quan khác", RelationType.RELATED_TO, False, False),
    ("Văn bản đình chỉ 1 phần", RelationType.PARTIALLY_SUSPENDS, False, True),
    ("Văn bản bị đình chỉ 1 phần", RelationType.PARTIALLY_SUSPENDS, True, True),
    ("Văn bản đình chỉ", RelationType.SUSPENDS, False, True),
    ("Văn bản bị đình chỉ", RelationType.SUSPENDS, True, True),
]


def test_label_map_has_exactly_17_entries():
    assert len(RELATION_LABEL_MAP) + len(REVERSE_RELATION_LABEL_MAP) == 17


def test_label_map_no_overlap_between_canonical_and_reverse():
    assert set(RELATION_LABEL_MAP) & set(REVERSE_RELATION_LABEL_MAP) == set()


def test_each_expected_label_resolves_to_correct_relation_type_and_direction():
    for label, expected_type, is_reverse, _ in EXPECTED_LABELS:
        from_norm, to_norm, relation_type = _resolve_relation("DOC_A", "DOC_B", label)
        assert relation_type == expected_type, f"{label!r} should map to {expected_type}"
        if is_reverse:
            assert (from_norm, to_norm) == ("DOC_B", "DOC_A"), f"{label!r} should reverse from/to"
        else:
            assert (from_norm, to_norm) == ("DOC_A", "DOC_B"), f"{label!r} should keep from/to as-is"


def test_layer_b_eligibility_matches_table():
    for label, expected_type, _, eligible in EXPECTED_LABELS:
        assert (expected_type in ELIGIBLE_FOR_LAYER_B) == eligible, (
            f"{label!r} ({expected_type}) Tầng B eligibility mismatch"
        )


def test_cites_refers_to_related_to_have_no_reverse_label():
    reverse_types = set(REVERSE_RELATION_LABEL_MAP.values())
    assert RelationType.CITES not in reverse_types
    assert RelationType.REFERS_TO not in reverse_types
    assert RelationType.RELATED_TO not in reverse_types


def test_unknown_label_returns_none():
    assert _resolve_relation("DOC_A", "DOC_B", "Nhãn không tồn tại") is None
