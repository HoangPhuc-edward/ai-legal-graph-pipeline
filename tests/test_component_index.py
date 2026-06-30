from pathlib import Path

from transform.pipeline import _build_component_index_entries
from transform.structure_parser import parse_structure

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_component_index_keys_use_citation_path_up_to_dieu():
    markdown = (FIXTURES_DIR / "full_depth.md").read_text(encoding="utf-8")
    result = parse_structure("ND_34_2016", markdown)
    index = _build_component_index_entries("ND_34_2016", result.components)

    by_citation = {c.citation: c for c in result.components}
    dieu1 = by_citation["Điều 1"]
    khoan1_of_dieu1 = next(
        c for c in result.components if c.parent_comp_id == dieu1.comp_id and c.citation == "Khoản 1"
    )
    diem_a = next(c for c in result.components if c.parent_comp_id == khoan1_of_dieu1.comp_id)

    # Bare Điều lookup
    assert index[("ND_34_2016", "Điều 1")] == dieu1.comp_id
    # Khoản luôn kèm Điều tổ tiên gần nhất (đánh số lại mỗi Điều nên không thể tra bare)
    assert index[("ND_34_2016", "Khoản 1 > Điều 1")] == khoan1_of_dieu1.comp_id
    # Điểm kèm cả Khoản lẫn Điều
    assert index[("ND_34_2016", f"{diem_a.citation} > Khoản 1 > Điều 1")] == diem_a.comp_id


def test_component_index_disambiguates_same_citation_under_different_dieu():
    markdown = (FIXTURES_DIR / "full_depth.md").read_text(encoding="utf-8")
    result = parse_structure("ND_34_2016", markdown)
    index = _build_component_index_entries("ND_34_2016", result.components)

    by_citation = {c.citation: c for c in result.components}
    dieu1 = by_citation["Điều 1"]
    dieu2 = by_citation["Điều 2"]
    khoan1_dieu1 = next(
        c for c in result.components if c.parent_comp_id == dieu1.comp_id and c.citation == "Khoản 1"
    )
    khoan1_dieu2 = next(
        c for c in result.components if c.parent_comp_id == dieu2.comp_id and c.citation == "Khoản 1"
    )

    assert index[("ND_34_2016", "Khoản 1 > Điều 1")] == khoan1_dieu1.comp_id
    assert index[("ND_34_2016", "Khoản 1 > Điều 2")] == khoan1_dieu2.comp_id
    assert khoan1_dieu1.comp_id != khoan1_dieu2.comp_id
