Liệt kê chính xác 3 chỗ đã sửa, so với bản gốc:
1. transform/action_extractor.py — hàm find_amendments()
Thêm tham số mới known_norm_ids: Optional[set[str]] = None, và thêm đoạn kiểm tra ở đầu hàm:
pythonif known_norm_ids is not None and other_doc_id not in known_norm_ids:
    logger.info(
        "Văn bản đích %s không có Component nào trong component_index "
        "(ngoài phạm vi --sample hoặc chưa parse được) — bỏ qua Tầng B cho doc %s, chỉ giữ Tầng A.",
        other_doc_id, doc_id,
    )
    return
→ Nếu văn bản đích (other_doc_id) không nằm trong tập known_norm_ids, hàm return ngay lập tức, không chạy vòng lặp regex/LLM nào cả — vì chắc chắn không thể khớp được (component_index không có entry nào cho văn bản đó).
2. transform/relation_classifier.py — hàm process_relationship_row()
Thêm tham số known_norm_ids: Optional[set[str]] = None, và truyền tiếp xuống lời gọi action_extractor.find_amendments(...):
pythonknown_norm_ids=known_norm_ids,
→ Chỉ đơn thuần là "ống dẫn" tham số, không có logic mới.
3. transform/pipeline.py — hàm transform_relationships()
Thêm 1 dòng build tập known_norm_ids ngay sau khi có component_index:
pythonknown_norm_ids = {norm_id for norm_id, _ in component_index}
(lấy tất cả norm_id duy nhất từ các key (norm_id, citation_path) của component_index)
Và truyền xuống lời gọi relation_classifier.process_relationship_row(...):
pythonknown_norm_ids=known_norm_ids,