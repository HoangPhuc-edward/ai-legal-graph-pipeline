"""Scratch diagnostic: tìm văn bản KHÔNG sinh được Component nào (structure_parser
khớp 0 level pattern trên toàn bộ markdown) -> 0 TextUnit -> nội dung văn bản
hoàn toàn không vào được đồ thị. Đây là lỗi ở structure_parser/transform_one,
KHÔNG phải lỗi LLM fallback (action_extractor không liên quan tới việc tạo
Component/TextUnit chính của 1 văn bản — LLM fallback chỉ tồn tại ở Tầng B
cho action_extractor, không tồn tại cho structure_parser).
"""
import sys

import pyarrow.parquet as pq

sys.path.insert(0, ".")

from transform.html_to_markdown import convert
from transform.structure_parser import parse_structure

SAMPLE_SIZE = 2000

metadata = pq.read_table("data/raw/metadata.parquet").to_pylist()[:SAMPLE_SIZE]
content_table = pq.read_table("data/raw/content.parquet")
content_by_id = {row["id"]: row["content_html"] for row in content_table.to_pylist()}

zero_component_docs = []
empty_html_docs = []
empty_markdown_docs = []

for row in metadata:
    norm_id = str(row["id"])
    content_html = content_by_id.get(row["id"]) or content_by_id.get(norm_id)

    if not content_html or not content_html.strip():
        empty_html_docs.append((norm_id, row.get("title")))
        continue

    markdown = convert(content_html)
    if not markdown.strip():
        empty_markdown_docs.append((norm_id, row.get("title")))
        continue

    result = parse_structure(norm_id, markdown)
    if len(result.components) == 0:
        zero_component_docs.append((norm_id, row.get("title"), markdown))

print(f"Sample: {len(metadata)} văn bản")
print(f"  content_html rỗng/thiếu: {len(empty_html_docs)}")
print(f"  markdown rỗng sau convert: {len(empty_markdown_docs)}")
print(f"  0 Component (regex không khớp dòng nào): {len(zero_component_docs)}")
print()

for norm_id, title, markdown in zero_component_docs[:8]:
    print("=" * 80)
    print(f"norm_id={norm_id} title={title!r}")
    print("--- markdown (800 ký tự đầu) ---")
    print(repr(markdown[:800]))
    print()
