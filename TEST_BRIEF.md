# Thiết kế Test Case — Kiểm tra dữ liệu Neo4j đáp ứng RAG

> Mục tiêu: xác nhận đồ thị đủ và đúng để RAG hoạt động được.
> Không kiểm tra số lượng tuyệt đối (vì trên cùng một Neo4j có thể có nhiều graph
> khác nhau). Chỉ kiểm tra **tính đúng đắn của cấu trúc và khả năng truy xuất.**

---

## Nhóm 1 — Cấu trúc đồ thị đúng chưa?

### TC-01 — Node lá phải có TextUnit

**Vấn đề:** không phải mọi Component đều cần TextUnit — chỉ node lá (không có Component con)
mới là đơn vị nội dung thực sự. Chương/Mục/Điều có Khoản con không cần lưu nội dung riêng.

```cypher
MATCH (c:Component)
WHERE NOT (c)-[:CONTAINS]->(:Component)
WITH count(c) AS tong_la
MATCH (la:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit)
WHERE NOT (la)-[:CONTAINS]->(:Component)
RETURN
  tong_la,
  count(DISTINCT la) AS la_co_textunit,
  round(100.0 * count(DISTINCT la) / tong_la, 1) AS phan_tram
```

**Kỳ vọng:** ≥ 95% node lá có TextUnit. Phần còn lại là văn bản quá ngắn (Sắc lệnh
bổ nhiệm, Quyết định hành chính không có Điều/Khoản) — chấp nhận được.

**Nếu thất bại:** lỗi ở `text_accumulator` — không nhận ra đúng node nào là lá,
hoặc `load_component_textunits()` bị bỏ qua.

---

### TC-02 — Cây phân cấp không bị đứt

**Vấn đề:** mỗi Component phải truy ngược lên được Norm — nếu đứt, RAG tìm được nội dung
nhưng không biết nó thuộc văn bản nào.

```cypher
MATCH (c:Component)
WHERE NOT (:Norm)-[:CONTAINS*]->(c)
RETURN count(c) AS component_mo_neo, collect(c.comp_id)[..5] AS vi_du
```

**Kỳ vọng:** = 0. Bất kỳ số nào lớn hơn là lỗi nghiêm trọng.

---

### TC-03 — Action đủ 2 đầu

**Vấn đề:** Action là cầu nối giữa Component A và Component B — thiếu 1 đầu thì
graph expansion trong RAG bị chết.

```cypher
MATCH (act:Action)
WITH act,
  exists((:Component)-[:HAS_ACTION]->(act)) AS co_nguon,
  exists((act)-[:APPLY_TO]->(:Component))   AS co_dich
RETURN
  count(CASE WHEN co_nguon AND co_dich THEN 1 END) AS hop_le,
  count(CASE WHEN NOT co_nguon THEN 1 END)          AS thieu_nguon,
  count(CASE WHEN NOT co_dich  THEN 1 END)          AS thieu_dich
```

**Kỳ vọng:** `thieu_nguon` = 0, `thieu_dich` = 0.

---

## Nhóm 2 — RAG truy xuất được không?

### TC-04 — Vector index tồn tại và đầy đủ

**Vấn đề:** không có index thì `db.index.vector.queryNodes()` không chạy được — đây là
điều kiện cứng để RAG hoạt động.

```cypher
SHOW VECTOR INDEXES
YIELD name, populationPercent, labelsOrTypes, properties
WHERE labelsOrTypes = ['TextUnit']
RETURN name, populationPercent
```

**Kỳ vọng:** có ít nhất 1 index trên `TextUnit`, `populationPercent` = 100.

Nếu `populationPercent` < 100: index đang được build, chờ thêm hoặc có TextUnit
được thêm sau khi index tạo mà chưa được cập nhật.

---

### TC-05 — Traversal ngược từ TextUnit lên Norm

**Vấn đề:** khi RAG tìm được TextUnit, bước tiếp theo là đi ngược lên Norm để lấy metadata
(tên văn bản, số hiệu). Nếu không đi được, câu trả lời thiếu nguồn trích dẫn.

```cypher
MATCH (tu:TextUnit)
WHERE tu.type <> 'cache_action' AND tu.embedding IS NOT NULL
WITH tu, rand() AS r ORDER BY r LIMIT 20
MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
OPTIONAL MATCH (n:Norm)-[:CONTAINS*]->(c)
RETURN
  count(tu)                                              AS tong_mau,
  count(CASE WHEN n IS NOT NULL THEN 1 END)              AS len_duoc_norm,
  count(CASE WHEN n.norm_number IS NOT NULL THEN 1 END)  AS co_so_hieu
```

**Kỳ vọng:** `len_duoc_norm` = 20, `co_so_hieu` = 20.

---

### TC-06 — Vector search end-to-end

**Vấn đề:** test thực tế duy nhất xác nhận toàn bộ pipeline tìm kiếm hoạt động — embed
câu hỏi, tìm trong index, trả về kết quả có nghĩa.

*Chạy bằng Python:*

```python
from eval.retriever import retrieve

ket_qua = retrieve(driver, "điều kiện để được bầu cử đại biểu Quốc hội", top_k=5)

assert len(ket_qua) >= 3, f"Chỉ tìm được {len(ket_qua)} kết quả"
assert any(
    "bầu cử" in r["text"].lower() or "ứng cử" in r["text"].lower()
    for r in ket_qua
), "Kết quả không liên quan đến câu hỏi"
assert ket_qua[0]["score"] > 0.7, f"Score cao nhất chỉ {ket_qua[0]['score']:.3f}"

for r in ket_qua:
    print(f"  score={r['score']:.3f}  {r['norm_title']} — {r['citation']}")
```

**Kỳ vọng:** ít nhất 3 kết quả, kết quả liên quan đến bầu cử, score cao nhất > 0.7.

---

## Nhóm 3 — Nội dung đúng không?

### TC-07 — Nội dung TextUnit không rỗng

**Vấn đề:** TextUnit rỗng sinh ra vector vô nghĩa, làm nhiễu kết quả tìm kiếm.

```cypher
MATCH (tu:TextUnit)
WHERE tu.type <> 'cache_action'
  AND (tu.accumulated_text IS NULL OR size(trim(tu.accumulated_text)) < 20)
WITH count(tu) AS so_rong
MATCH (tu2:TextUnit) WHERE tu2.type <> 'cache_action'
RETURN so_rong, count(tu2) AS tong,
       round(100.0 * so_rong / count(tu2), 2) AS phan_tram_rong
```

**Kỳ vọng:** `phan_tram_rong` < 1%.

---

### TC-08 — Citation path đúng chiều

**Vấn đề:** lỗi từng xuất hiện thực tế — citation bị đảo ngược ("Mục 1 > Chương V"
thay vì "Chương V > Mục 1"), khiến action_extractor không khớp được Component B.

```cypher
MATCH (c:Component)
WHERE c.citation =~ '.*Mục\\s+\\d+.*>.*Chương.*'
RETURN count(c) AS citation_nguoc_chieu, collect(c.citation)[..5] AS vi_du
```

**Kỳ vọng:** = 0.

---

## Bảng tóm tắt

| # | Kiểm tra | Chạy bằng | Chặn RAG? |
|---|---|---|---|
| TC-01 | Node lá có TextUnit (≥95%) | Cypher | **Có** |
| TC-02 | Cây không đứt gãy | Cypher | **Có** |
| TC-03 | Action đủ 2 đầu | Cypher | Có (mất graph expansion) |
| TC-04 | Vector index đầy đủ | Cypher | **Có** |
| TC-05 | Traversal ngược lên Norm | Cypher | Có (mất metadata) |
| TC-06 | Vector search end-to-end | Python | **Có** |
| TC-07 | TextUnit không rỗng | Cypher | Gián tiếp |
| TC-08 | Citation đúng chiều | Cypher | Có (mất Action mức B) |

**Chạy theo thứ tự:** TC-01 → TC-02 → TC-04 → TC-06 — 4 test này đủ xác nhận
hệ thống RAG có thể chạy hay không. Các test còn lại chạy khi cần debug sâu hơn.