from pathlib import Path

from schema.enums import ComponentLevel
from transform.structure_parser import parse_structure

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(name: str) -> str:
    return (FIXTURES_DIR / name).read_text(encoding="utf-8")


def test_full_depth_builds_correct_tree():
    markdown = _load("full_depth.md")
    result = parse_structure("norm-1", markdown)
    components = result.components

    levels = [c.level for c in components]
    assert ComponentLevel.PHAN in levels
    assert ComponentLevel.CHUONG in levels
    assert ComponentLevel.MUC in levels
    assert ComponentLevel.DIEU in levels
    assert ComponentLevel.KHOAN in levels
    assert ComponentLevel.DIEM in levels

    by_citation = {c.citation: c for c in components}

    phan = by_citation["Phần I"]
    assert phan.parent_comp_id is None  # con trực tiếp của Norm

    chuong1 = by_citation["Chương I"]
    assert chuong1.parent_comp_id == phan.comp_id

    muc1 = by_citation["Mục 1"]
    assert muc1.parent_comp_id == chuong1.comp_id

    dieu1 = by_citation["Điều 1"]
    assert dieu1.parent_comp_id == muc1.comp_id
    assert dieu1.title_text == "Phạm vi điều chỉnh"

    # Citation "Khoản 1"/"Điểm a" lặp lại giữa các Điều khác nhau (đánh số lại
    # từ đầu mỗi Điều) — lọc theo parent_comp_id thay vì dùng citation toàn cục.
    children_of_dieu1 = [c for c in components if c.parent_comp_id == dieu1.comp_id]
    khoan1 = next(c for c in children_of_dieu1 if c.citation == "Khoản 1")

    diem_a = next(
        c for c in components if c.parent_comp_id == khoan1.comp_id and c.citation == "Điểm a"
    )

    # Điều 2 là anh em của Điều 1 (cùng cha Mục 1)
    dieu2 = next(c for c in components if c.citation == "Điều 2" and c.parent_comp_id == muc1.comp_id)
    assert dieu2.parent_comp_id == muc1.comp_id

    # Chương II không có Mục -> Điều 3 là con trực tiếp của Chương II
    chuong2 = by_citation["Chương II"]
    dieu3 = by_citation["Điều 3"]
    assert dieu3.parent_comp_id == chuong2.comp_id

    # raw_text phải được gán đúng cho component lá (Khoản 2 không có Điểm con)
    khoan2 = by_citation["Khoản 2"]
    assert khoan2.comp_id in result.raw_text


def test_missing_phan_muc_does_not_break_stack():
    markdown = _load("missing_phan_muc.md")
    result = parse_structure("norm-2", markdown)
    components = result.components

    levels_present = {c.level for c in components}
    assert ComponentLevel.PHAN not in levels_present
    assert ComponentLevel.MUC not in levels_present
    assert ComponentLevel.CHUONG in levels_present
    assert ComponentLevel.DIEU in levels_present

    by_citation = {c.citation: c for c in components}
    chuong1 = by_citation["Chương I"]
    dieu1 = by_citation["Điều 1"]
    assert dieu1.parent_comp_id == chuong1.comp_id  # nhảy thẳng Chương -> Điều

    chuong2 = by_citation["Chương II"]
    assert chuong2.parent_comp_id is None  # con trực tiếp của Norm (không có Phần)


def test_dieu_only_quyet_dinh_ngan():
    markdown = _load("dieu_only.md")
    result = parse_structure("norm-3", markdown)
    components = result.components

    assert len(components) == 3
    assert all(c.level == ComponentLevel.DIEU for c in components)
    # Không có Chương/Phần -> mọi Điều đều là con trực tiếp của Norm
    assert all(c.parent_comp_id is None for c in components)
    assert [c.citation for c in components] == ["Điều 1", "Điều 2", "Điều 3"]


def test_order_index_is_sequential():
    markdown = _load("full_depth.md")
    result = parse_structure("norm-1", markdown)
    indices = [c.order_index for c in result.components]
    assert indices == sorted(indices)
    assert indices == list(range(1, len(indices) + 1))
