# Thiết kế Test Case — Kiểm tra dữ liệu Neo4j đáp ứng RAG

> Mục tiêu: xác nhận đồ thị đã load đủ và đúng để hệ thống RAG có thể
> trả lời câu hỏi pháp luật một cách chính xác. Không phải test code —
> test **chất lượng dữ liệu**.

---

## Cách đọc tài liệu này

Mỗi nhóm test có:
- **Kiểm tra điều gì** — mục đích thực tế
- **Query Cypher** — chạy thẳng trên Neo4j Browser
- **Kỳ vọng** — kết quả thế nào là đạt
- **Nếu thất bại** — nguyên nhân thường gặp và bước xử lý

---

## Nhóm 1 — Kiểm tra độ phủ cơ bản (Sanity Check)

> Trả lời câu hỏi: "Dữ liệu đã lên chưa, lên đủ chưa?"

### TC-01 — Số lượng văn bản trong đồ thị

**Kiểm tra điều gì:** xác nhận toàn bộ văn bản đã được load, không bị thiếu do lỗi pipeline.

```cypher
MATCH (n:Norm)
RETURN count(n) AS tong_van_ban
```

**Kỳ vọng:** con số phải gần với số dòng trong `metadata.parquet` (~153.000). Chênh lệch dưới 1% là chấp nhận được (văn bản thiếu content_html).

**Nếu thất bại:** xem log của bước Load — tìm lỗi timeout hoặc batch bị bỏ qua.

---

### TC-02 — Văn bản có Component con

**Kiểm tra điều gì:** xác nhận Transform đã tách được cấu trúc Điều/Khoản, không phải chỉ load metadata trơn.

```cypher
MATCH (n:Norm)
WITH n,
     size([(n)-[:CONTAINS*]->(:Component) | 1]) AS so_component
RETURN
  count(CASE WHEN so_component > 0 THEN 1 END) AS co_component,
  count(CASE WHEN so_component = 0 THEN 1 END) AS khong_co_component,
  round(100.0 * count(CASE WHEN so_component = 0 THEN 1 END) / count(n), 1) AS ty_le_rong_phan_tram
```

**Kỳ vọng:** tỷ lệ "không có Component" dưới 15%. Văn bản chỉ có metadata mà không có content_html hợp lệ sẽ rơi vào nhóm này.

**Nếu thất bại (>15% không có Component):** chạy lại `test.py` với `--sample 500` để kiểm tra structure_parser trên dữ liệu thật.

---

### TC-03 — Component có TextUnit (nội dung để embed)

**Kiểm tra điều gì:** xác nhận nội dung đã được lưu và vector đã được tính — đây là điều kiện cứng để RAG tìm kiếm được.

```cypher
MATCH (c:Component)
WITH
  count(c) AS tong,
  count(CASE WHEN exists((c)-[:HAS_TEXTUNIT]->()) THEN 1 END) AS co_textunit,
  count(CASE WHEN (c)-[:HAS_TEXTUNIT]->(:TextUnit {embedding: null}) THEN 1 END) AS chua_embed
RETURN tong, co_textunit,
  round(100.0 * co_textunit / tong, 1) AS phan_tram_co_noi_dung,
  chua_embed AS chua_co_vector
```

**Kỳ vọng:**
- Ít nhất 85% Component có TextUnit.
- `chua_embed` = 0 (tất cả đã được vector hoá). Nếu > 0, RAG sẽ bỏ sót phần nội dung đó.

**Nếu thất bại:** chạy `tools/retry_failed_embeddings.py` cho phần chưa embed.

---

### TC-04 — Quan hệ giữa văn bản

**Kiểm tra điều gì:** xác nhận mức A (Norm → Norm) đã load đủ.

```cypher
MATCH ()-[r:AMENDS|SUPPLEMENTS|TERMINATES|PARTIALLY_TERMINATES|SUSPENDS|PARTIALLY_SUSPENDS|IMPLEMENTS|REFERS_TO|RELATED_TO|CITES]->()
RETURN type(r) AS loai_quan_he, count(r) AS so_luong
ORDER BY so_luong DESC
```

**Kỳ vọng:** tổng tất cả loại quan hệ phải gần ~900.000. CITES phải chiếm nhiều nhất (~580.000).

---

## Nhóm 2 — Kiểm tra tính đúng đắn cấu trúc

> Trả lời câu hỏi: "Dữ liệu lên rồi, nhưng đúng chưa?"

### TC-05 — Cây phân cấp không bị đứt gãy

**Kiểm tra điều gì:** mỗi Component phải biết mình thuộc Văn bản nào — nếu cây bị đứt, RAG sẽ không thể truy ngược lên Norm khi cần.

```cypher
// Tìm Component không có đường dẫn nào lên Norm
MATCH (c:Component)
WHERE NOT exists((:Norm)-[:CONTAINS*]->(c))
RETURN count(c) AS component_bi_mo_neo, collect(c.comp_id)[..10] AS vi_du
```

**Kỳ vọng:** kết quả = 0. Bất kỳ Component nào không có Norm cha là lỗi nghiêm trọng.

**Nếu thất bại:** kiểm tra thứ tự load — Component phải được load sau Norm.

---

### TC-06 — Action phải có đủ 2 đầu

**Kiểm tra điều gì:** Action là cầu nối giữa 2 Component — nếu thiếu 1 đầu, traversal sẽ bị chết.

```cypher
MATCH (act:Action)
WITH act,
  exists((:Component)-[:HAS_ACTION]->(act)) AS co_nguon,
  exists((act)-[:APPLY_TO]->(:Component))   AS co_dich
RETURN
  count(CASE WHEN co_nguon AND co_dich THEN 1 END) AS hop_le,
  count(CASE WHEN NOT co_nguon THEN 1 END) AS thieu_nguon,
  count(CASE WHEN NOT co_dich THEN 1 END)  AS thieu_dich
```

**Kỳ vọng:** `thieu_nguon` = 0, `thieu_dich` = 0.

---

### TC-07 — TextUnit có nội dung không rỗng

**Kiểm tra điều gì:** TextUnit rỗng sẽ sinh ra vector vô nghĩa, ảnh hưởng đến chất lượng tìm kiếm.

```cypher
MATCH (tu:TextUnit)
WHERE tu.type <> 'cache_action'
  AND (tu.accumulated_text IS NULL OR size(trim(tu.accumulated_text)) < 20)
RETURN count(tu) AS textunit_rong_hoac_qua_ngan, collect(tu.unit_id)[..5] AS vi_du
```

**Kỳ vọng:** dưới 1% tổng số TextUnit. Văn bản quá ngắn (Sắc lệnh bổ nhiệm 1 người) có thể nằm đây nhưng không đáng lo.

---

### TC-08 — Citation path đúng chiều (gốc → lá)

**Kiểm tra điều gì:** bước trước đã phát hiện lỗi citation path bị đảo ngược ("Mục 1 > Chương V" thay vì "Chương V > Mục 1"). Test này xác nhận lỗi đã được sửa.

```cypher
// Tìm Component có citation chứa "Mục X > Chương" — đây là chiều sai
MATCH (c:Component)
WHERE c.citation =~ '.*Mục\\s+\\d+.*>.*Chương.*'
RETURN count(c) AS citation_nguoc_chieu, collect(c.citation)[..5] AS vi_du
```

**Kỳ vọng:** kết quả = 0.

---

## Nhóm 3 — Kiểm tra khả năng RAG truy xuất

> Trả lời câu hỏi: "Dữ liệu đúng rồi, nhưng RAG tìm được không?"

### TC-09 — Vector Index tồn tại và có dữ liệu

**Kiểm tra điều gì:** Neo4j Vector Index phải tồn tại và đã index đủ TextUnit — đây là điều kiện bắt buộc để `db.index.vector.queryNodes()` chạy được.

```cypher
SHOW VECTOR INDEXES
```

Sau đó kiểm tra số lượng đã index:

```cypher
MATCH (tu:TextUnit)
WHERE tu.embedding IS NOT NULL AND tu.type <> 'cache_action'
RETURN count(tu) AS da_co_vector
```

**Kỳ vọng:** có ít nhất 1 vector index tên `textunit_embedding_index`. Số `da_co_vector` phải khớp với số trong index (xem cột `populationPercent` trong kết quả `SHOW VECTOR INDEXES` — phải = 100%).

---

### TC-10 — Vector search trả về kết quả

**Kiểm tra điều gì:** thực hiện 1 tìm kiếm thật để xác nhận toàn bộ pipeline tìm kiếm hoạt động end-to-end.

*Cách thực hiện: chạy bằng Python vì cần embed câu hỏi trước.*

```python
# Chạy trong môi trường đã cài đặt đầy đủ
from eval.retriever import retrieve

ket_qua = retrieve(driver, "điều kiện để được bầu cử đại biểu Quốc hội", top_k=5)

assert len(ket_qua) > 0, "Không tìm được kết quả nào"
assert any("bầu cử" in r["text"] or "ứng cử" in r["text"] for r in ket_qua), \
    "Kết quả không liên quan đến câu hỏi"

print(f"[OK] Tìm được {len(ket_qua)} kết quả")
for r in ket_qua:
    print(f"  score={r['score']:.3f}  {r['norm_title']} — {r['citation']}")
```

**Kỳ vọng:** ít nhất 3/5 kết quả thuộc Luật Bầu cử hoặc văn bản liên quan. Score cao nhất phải > 0.7.

---

### TC-11 — Traversal ngược từ TextUnit lên Norm

**Kiểm tra điều gì:** khi RAG tìm được 1 TextUnit, nó phải đi ngược lên được Norm để lấy thêm metadata (tên văn bản, số hiệu, ngày hiệu lực...).

```cypher
// Lấy 10 TextUnit ngẫu nhiên, xem có lên được Norm không
MATCH (tu:TextUnit)
WHERE tu.type <> 'cache_action' AND tu.embedding IS NOT NULL
WITH tu, rand() AS r ORDER BY r LIMIT 10
MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu)
MATCH (n:Norm)-[:CONTAINS*]->(c)
RETURN tu.unit_id, n.title AS norm_title, n.norm_number, c.citation
```

**Kỳ vọng:** tất cả 10 dòng đều có `norm_title` và `norm_number` (không null). Nếu có dòng null → cây bị đứt (liên quan TC-05).

---

### TC-12 — Action mở rộng context hoạt động

**Kiểm tra điều gì:** bước Graph Expansion trong RAG — từ 1 Component tìm được, phải tìm được các Action liên quan để bổ sung context về sửa đổi.

```cypher
// Tìm 5 Component có Action, xem traverse được không
MATCH (c:Component)<-[:APPLY_TO]-(act:Action)
WITH c, act LIMIT 5
OPTIONAL MATCH (act)-[:HAS_TEXTUNIT]->(tu_cache:TextUnit)
RETURN
  c.comp_id,
  c.citation,
  act.relation_type,
  act.amending_doc_number,
  tu_cache.accumulated_text IS NOT NULL AS co_noi_dung_cache
```

**Kỳ vọng:** tất cả 5 dòng có `amending_doc_number` không null. Cột `co_noi_dung_cache` = true (cache nội dung thay đổi đã được lưu).

---

## Nhóm 4 — Kiểm tra tính đúng của nội dung

> Trả lời câu hỏi: "RAG tìm được rồi, nhưng nội dung có đúng không?"

### TC-13 — Kiểm tra thủ công nội dung ngẫu nhiên

**Kiểm tra điều gì:** một mẫu nhỏ (10-20 Component) được đối chiếu thủ công với văn bản gốc trên vbpl.vn để xác nhận nội dung không bị mất, ghép nhầm, hay decode sai encoding.

```cypher
MATCH (n:Norm)-[:CONTAINS*]->(c:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit)
WHERE c.level IN ['Dieu', 'Khoan']
  AND tu.type <> 'cache_action'
  AND tu.embedding IS NOT NULL
WITH n, c, tu, rand() AS r ORDER BY r LIMIT 10
RETURN
  n.norm_number    AS so_hieu,
  n.title          AS ten_van_ban,
  c.citation       AS vi_tri,
  left(tu.accumulated_text, 300) AS preview_noi_dung
```

**Cách đánh giá:** mở vbpl.vn, tìm từng văn bản trong danh sách, so sánh đoạn preview với nội dung gốc.

**Kỳ vọng:** nội dung khớp, không có ký tự lạ, không bị cắt xén giữa câu.

---

### TC-14 — Kiểm tra văn bản đã bị sửa đổi

**Kiểm tra điều gì:** đây là điểm mạnh của GraphRAG so với RAG thường — nếu 1 văn bản có Action sửa đổi, hệ thống phải thấy được cả nội dung cũ lẫn thông tin đã sửa.

```cypher
// Tìm 1 cặp cụ thể: văn bản sửa đổi + điều bị sửa
MATCH (ca:Component)-[:HAS_ACTION]->(act:Action {relation_type: 'AMENDS'})-[:APPLY_TO]->(cb:Component)
MATCH (ca)-[:HAS_TEXTUNIT]->(tu_a:TextUnit)
MATCH (cb)-[:HAS_TEXTUNIT]->(tu_b:TextUnit)
MATCH (na:Norm)-[:CONTAINS*]->(ca)
MATCH (nb:Norm)-[:CONTAINS*]->(cb)
RETURN
  na.norm_number AS van_ban_sua,
  ca.citation    AS dieu_thuc_hien_sua,
  nb.norm_number AS van_ban_bi_sua,
  cb.citation    AS dieu_bi_sua,
  act.amending_doc_number AS ten_cache,
  left(tu_a.accumulated_text, 200) AS noi_dung_sua_doi,
  left(tu_b.accumulated_text, 200) AS noi_dung_bi_sua
LIMIT 3
```

**Kỳ vọng:** cả 3 dòng đều có đủ thông tin ở cả 2 đầu. `ten_cache` không null. Nội dung `tu_a` phải có câu kiểu "sửa đổi, bổ sung..." — xác nhận transformer đã trích đúng đoạn.

---

## Nhóm 5 — Kiểm tra hiệu năng

> Trả lời câu hỏi: "Chạy được rồi, nhưng có đủ nhanh cho người dùng không?"

### TC-15 — Tốc độ vector search

**Kiểm tra điều gì:** độ trễ của bước tìm kiếm vector — ngưỡng chấp nhận được là dưới 500ms cho top-5.

*Chạy bằng Python, đo thời gian thực tế:*

```python
import time
from eval.retriever import embed_question, VECTOR_SEARCH_CYPHER

vector = embed_question("điều kiện bầu cử")

start = time.time()
with driver.session() as s:
    result = s.run(VECTOR_SEARCH_CYPHER, top_k=5, query_vector=vector).data()
elapsed = time.time() - start

print(f"Vector search top-5: {elapsed*1000:.0f}ms — {'OK' if elapsed < 0.5 else 'CHẬM'}")
assert elapsed < 1.0, f"Quá chậm: {elapsed:.2f}s"
```

**Kỳ vọng:** dưới 500ms. Nếu 500ms–1s: có thể chấp nhận nhưng cần theo dõi. Nếu >1s: kiểm tra vector index có được tạo đúng chưa.

---

### TC-16 — Tốc độ Graph Expansion

**Kiểm tra điều gì:** bước mở rộng sang Action sau khi đã có top-k Component — phải nhanh vì không dùng vector, chỉ là traversal.

```cypher
// Neo4j Browser tự hiển thị thời gian thực thi
PROFILE
MATCH (c:Component {comp_id: $comp_id})<-[:APPLY_TO]-(act:Action)-[:HAS_TEXTUNIT]->(tu:TextUnit)
RETURN act.relation_type, act.amending_doc_number, tu.accumulated_text
```

**Kỳ vọng:** dưới 50ms. Nếu chậm hơn, kiểm tra index trên `comp_id`.

---

## Tóm tắt — Bảng check nhanh

| # | Tên | Nhóm | Chạy bằng | Chặn RAG nếu thất bại? |
|---|---|---|---|---|
| TC-01 | Số lượng văn bản | Phủ | Cypher | Không |
| TC-02 | Văn bản có Component | Phủ | Cypher | Có — nội dung rỗng |
| TC-03 | Component có TextUnit + vector | Phủ | Cypher | **Có — không tìm kiếm được** |
| TC-04 | Quan hệ đã load | Phủ | Cypher | Không |
| TC-05 | Cây không đứt gãy | Đúng | Cypher | Có — không truy ngược được |
| TC-06 | Action đủ 2 đầu | Đúng | Cypher | Có — traversal chết |
| TC-07 | TextUnit không rỗng | Đúng | Cypher | Gián tiếp |
| TC-08 | Citation đúng chiều | Đúng | Cypher | Có — Action không khớp |
| TC-09 | Vector index tồn tại | RAG | Cypher | **Có — không chạy được** |
| TC-10 | Vector search end-to-end | RAG | Python | **Có — không tìm được** |
| TC-11 | Traversal ngược lên Norm | RAG | Cypher | Có — thiếu metadata |
| TC-12 | Action mở rộng context | RAG | Cypher | Không (Mức A vẫn dùng được) |
| TC-13 | Nội dung chính xác | Nội dung | Thủ công | Gián tiếp — câu trả lời sai |
| TC-14 | Văn bản sửa đổi | Nội dung | Cypher | Không (chỉ mất Mức B) |
| TC-15 | Tốc độ vector search | Hiệu năng | Python | Không (chỉ ảnh hưởng UX) |
| TC-16 | Tốc độ traversal | Hiệu năng | Cypher | Không (chỉ ảnh hưởng UX) |

**Ưu tiên chạy trước:** TC-03 → TC-09 → TC-10 → TC-05 → TC-06. Đây là 5 test có thể chặn hoàn toàn khả năng hoạt động của RAG.
