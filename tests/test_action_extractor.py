from transform import action_extractor

# ~Mẫu rút gọn của 4 khuôn mẫu phổ biến theo Nghị định 34/2016/NĐ-CP (sửa đổi
# bởi 154/2020/NĐ-CP). Đo tỷ lệ regex match được trên tập mẫu nhỏ này trước
# khi chạy full corpus — brief yêu cầu test trên ~50 văn bản thật, tách theo
# norm_type/năm; ở đây dùng tập mẫu cấu trúc tương đương để kiểm thử thuật toán.
SUA_DOI_BO_SUNG_SAMPLES = [
    'Sửa đổi Khoản 2 Điều 5 của Nghị định số 10/2020/NĐ-CP như sau: "Tổ chức, cá nhân phải nộp hồ sơ trong vòng 30 ngày."',
    "Bổ sung Điều 7 của Thông tư này như sau: Cơ quan quản lý phải báo cáo định kỳ hàng quý.",
]
BAI_BO_SAMPLES = [
    "Bãi bỏ Khoản 3 Điều 9 của Nghị định số 22/2018/NĐ-CP.",
    "Bãi bỏ Điều 12 của Quyết định này.",
]
THAY_THE_CUM_TU_SAMPLES = [
    'Thay thế cụm từ "Bộ Tài chính" bằng cụm từ "Bộ Kế hoạch và Đầu tư" tại Khoản 1 Điều 4.',
]
BO_CUM_TU_SAMPLES = [
    'Bỏ cụm từ "và Sở Tài chính" tại Điều 6.',
]

NON_MATCHING_SAMPLES = [
    "Nghị định này quy định chi tiết thi hành một số điều của Luật Đầu tư.",
    "Trường hợp phát sinh vướng mắc, Bộ trưởng Bộ Tư pháp có trách nhiệm hướng dẫn xử lý theo thẩm quyền chung.",
]

OTHER_DOC_ID = "other-doc"


def _component_index_with(*citation_paths: str) -> dict:
    return {(OTHER_DOC_ID, path): f"resolved::{path}" for path in citation_paths}


def test_regex_resolves_sua_doi_bo_sung():
    index = _component_index_with("Khoản 2 > Điều 5", "Điều 7")
    component_text = {"comp-a": SUA_DOI_BO_SUNG_SAMPLES[0]}
    results = list(
        action_extractor.find_amendments("doc-a", component_text, OTHER_DOC_ID, index, use_llm=False)
    )
    assert results == [("comp-a", "Khoản 2 > Điều 5")]


def test_regex_resolves_bai_bo():
    index = _component_index_with("Khoản 3 > Điều 9")
    component_text = {"comp-a": BAI_BO_SAMPLES[0]}
    results = list(
        action_extractor.find_amendments("doc-a", component_text, OTHER_DOC_ID, index, use_llm=False)
    )
    assert results == [("comp-a", "Khoản 3 > Điều 9")]


def test_regex_resolves_thay_the_cum_tu():
    index = _component_index_with("Khoản 1 > Điều 4")
    component_text = {"comp-a": THAY_THE_CUM_TU_SAMPLES[0]}
    results = list(
        action_extractor.find_amendments("doc-a", component_text, OTHER_DOC_ID, index, use_llm=False)
    )
    assert results == [("comp-a", "Khoản 1 > Điều 4")]


def test_regex_resolves_bo_cum_tu():
    index = _component_index_with("Điều 6")
    component_text = {"comp-a": BO_CUM_TU_SAMPLES[0]}
    results = list(
        action_extractor.find_amendments("doc-a", component_text, OTHER_DOC_ID, index, use_llm=False)
    )
    assert results == [("comp-a", "Điều 6")]


def test_no_match_without_llm_yields_nothing():
    index = _component_index_with("Điều 1")
    for text in NON_MATCHING_SAMPLES:
        results = list(
            action_extractor.find_amendments("doc-a", {"comp-a": text}, OTHER_DOC_ID, index, use_llm=False)
        )
        assert results == []


def test_citation_not_in_component_index_yields_nothing_without_llm():
    """Trích được citation bằng regex nhưng Component B không tồn tại trong
    component_index (vd văn bản đích chưa có trong corpus) -> không phải lỗi,
    chỉ là không tạo được Action (Tầng A vẫn còn nguyên ở chỗ khác)."""
    empty_index: dict = {}
    results = list(
        action_extractor.find_amendments(
            "doc-a", {"comp-a": BAI_BO_SAMPLES[0]}, OTHER_DOC_ID, empty_index, use_llm=False
        )
    )
    assert results == []


def test_regex_match_rate_on_sample_corpus():
    """Đo tỷ lệ match của 4 regex pattern trên tập mẫu (đại diện thu nhỏ cho
    yêu cầu '~50 văn bản sửa đổi thật' trong brief)."""
    all_samples = (
        SUA_DOI_BO_SUNG_SAMPLES + BAI_BO_SAMPLES + THAY_THE_CUM_TU_SAMPLES + BO_CUM_TU_SAMPLES
    )
    matched = 0
    for i, text in enumerate(all_samples):
        citation = action_extractor._extract_with_regex(text)
        if citation is not None:
            matched += 1
    match_rate = matched / len(all_samples)
    assert match_rate == 1.0, f"Tỷ lệ regex match trên mẫu khuôn-mẫu-chuẩn phải là 100%, được {match_rate:.0%}"
