# legal_graph_pipeline

ETL pipeline: dữ liệu thô [`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents)
→ đồ thị hợp lệ trên Neo4j AuraDB (Vietnamese Legal GraphRAG). Thiết kế đầy đủ ở
[`../Brief_ETL_Legal_Graph.md`](../Brief_ETL_Legal_Graph.md).

## Kiến trúc tóm tắt

```
extract/  →  transform/ (2 pass)  →  embed/  →  load/
(tải thô)    (HTML → object Schema)    (vector hoá TextUnit)    (object → Neo4j)
```

- **Tầng A** (`(Norm)-[:RELATION_TYPE]->(Norm)`): luôn tạo cho mọi dòng `relationships.parquet`.
- **Tầng B** — `Action` là node CẦU NỐI thật giữa 2 `Component`:
  `(Component A)-[:HAS_ACTION]->(Action)-[:APPLY_TO]->(Component B)`, với Component A = Điều/Khoản
  TRONG văn bản đang sửa đổi, Component B = Điều/Khoản TRONG văn bản đích bị tác động. Chỉ tạo khi
  `action_extractor` khớp được **cả 2 đầu** (regex 3 bước, LLM fallback, tra `component_index`).
- **Cache trên `Action`** — `amending_doc_number` (copy `Norm A.norm_number`) + `TextUnit` riêng
  (`type="cache_action"`, copy `accumulated_text` của Component A, **không embed**) — tránh phải đi
  đường dài `HAS_ACTION → Component A → CONTAINS → Norm A` cho câu hỏi thường gặp.

Vì `Action` nối 2 `Component` thuộc 2 văn bản khác nhau, `transform/pipeline.py` chạy **2 pass**:
- **Pass 1** — `structure_parser` cho mọi văn bản, build `Component`/`TextUnit`, đồng thời build
  `component_index: dict[(norm_id, citation_path), comp_id]` (vd `("ND_34_2016", "Khoản 4 > Điều 38")`).
- **Pass 2** — đọc `relationships.parquet`, `action_extractor` khớp Component B qua `component_index`
  đã build ở Pass 1 — **không** parse lại văn bản đích.

Chi tiết đầy đủ về mô hình quan hệ, schema Pydantic, thuật toán `structure_parser`
(stack-based tree builder) và `action_extractor` (regex-first, LLM-fallback 3 tầng) — xem brief.

## Cài đặt

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt
cp .env.example .env            # điền NEO4J_URI / NEO4J_PASSWORD / GCP_PROJECT...
```

## Chạy pipeline

```bash
# Kiểm tra môi trường trước khi chạy (khuyến nghị):
python tools/health_check.py             # kiểm tra 5 điểm: lib, data files, Neo4j, LLM, Embedding
python tools/health_check.py --skip-llm  # bỏ qua check LLM + Embedding (nhanh hơn)

# Các stage ETL:
python run_pipeline.py --stage extract                              # tải parquet về data/raw/
python run_pipeline.py --stage transform --sample 200              # test trước khi full
python run_pipeline.py --stage transform --sample 200 --no-llm     # chỉ dùng regex, không gọi LLM
python run_pipeline.py --stage transform --sample 200 --workers 4  # song song Pass 1 (Colab)
python run_pipeline.py --stage transform --input-dir data/filtered  # dùng parquet đã lọc keyword
python run_pipeline.py --stage embed                               # vector hoá TextUnit (tốn phí)
python run_pipeline.py --stage load                                # nạp vào Neo4j
python run_pipeline.py --stage load --limit-aura                   # giới hạn 200k node / 400k edge
python run_pipeline.py --stage all                                 # chạy toàn bộ

# Ước tính kích thước đồ thị trước khi load:
python tools/estimate_graph_size.py             # đọc data/transformed/*.jsonl
python tools/estimate_graph_size.py --dir PATH  # thư mục transformed tuỳ chọn

# Xoá sạch Neo4j (có xác nhận):
python tools/clean_neo4j.py          # hỏi "yes" trước khi xoá
python tools/clean_neo4j.py --force  # xoá không hỏi
```

Mỗi stage đọc/ghi qua file trung gian dưới `./data/` (`raw/`, `transformed/`, `embedded/`)
để có thể dừng/chạy lại từng giai đoạn riêng — quan trọng vì `embed` tốn phí.

### Xoá sạch đồ thị (`tools/clean_neo4j.py`)

Dùng khi muốn **reload từ đầu** (thay đổi schema, re-transform toàn bộ, hoặc môi trường dev).

```bash
python tools/clean_neo4j.py          # in số node/edge hiện tại, hỏi "yes" trước khi xoá
python tools/clean_neo4j.py --force  # xoá không hỏi (dùng trong script/Colab)
```

**Những gì script làm:**
1. Đếm và hiển thị số node/edge hiện có.
2. Xoá toàn bộ node + edge theo batch 5,000 — tránh AuraDB timeout (không dùng `DETACH DELETE` trực tiếp
   vì với 200k+ node AuraDB sẽ timeout).
3. Recreate schema từ `load/schema_init.cypher` — gồm 4 unique constraint, 2 index thường,
   và 1 vector index `textunit_embedding_index` (768-dim, cosine).

**Sau khi clean, chạy lại từ load:**
```bash
python run_pipeline.py --stage load   # load lại từ data/embedded/textunits.jsonl đã có
```
Không cần re-transform hay re-embed — file `data/embedded/textunits.jsonl` vẫn còn nguyên.

**Lưu ý vector index:** `clean_neo4j.py` recreate lại vector index ngay sau xoá, nhưng Neo4j
cần thời gian populate (vài phút với corpus lớn) sau khi `--stage load` nạp xong embedding.
Chạy `pytest tests/test_neo4j_rag_quality.py -v -m critical` để xác nhận index sẵn sàng.

### Tra cứu dữ liệu thô an toàn (`extract/hf_dataset.py`)

Đọc parquet **theo batch** (không bao giờ load full file — xem [`CLAUDE.md`](CLAUDE.md) Mục 1).
Chạy lại nhiều lần **không bị duplicate** — file JSON ghi đè, Neo4j dùng `MERGE`.

```bash
# Lấy mẫu an toàn để chạy thử pipeline: N metadata đầu (mặc định 100)
# -> content tương ứng -> tối đa N relationships liên quan
python -m extract.hf_dataset sample           # N=100, ghi ra data/samples/sample.json
python -m extract.hf_dataset sample --n 20   # giảm xuống 20 nếu máy yếu

# Lọc 3 parquet theo từ khoá → ghi data/filtered/*.parquet (giảm input cho transform)
# Sau đó dùng --input-dir data/filtered để transform chỉ xử lý subset này
python -m extract.hf_dataset keyword                              # quét full, ghi data/filtered/
python -m extract.hf_dataset keyword --limit 5000                 # giới hạn 5000 văn bản khớp
python -m extract.hf_dataset keyword --keywords-file PATH --output-dir PATH  # tuỳ chỉnh

# Tải parquet (dành cho Colab / lưu vào Google Drive):
python -m extract.hf_dataset download                               # tải cả 3, bỏ qua file đã có
python -m extract.hf_dataset download --output-dir /content/drive/MyDrive/legal/raw
python -m extract.hf_dataset download --configs metadata relationships  # chỉ 2 config
python -m extract.hf_dataset download --force                       # tải lại dù đã có sẵn
```

Kết quả `keyword` ghi ra `data/filtered/*.parquet` — dùng với `--input-dir data/filtered` ở bước transform. Nếu parquet chưa có cục bộ, lệnh tự tải về `data/raw/` trước khi dùng.

## Test

### Unit tests (không cần kết nối ngoài)

```bash
pytest                              # chạy tất cả unit test
pytest tests/test_action_extractor.py -v
```

- `tests/test_structure_parser.py` — fixture đủ 7 cấp, thiếu Phần/Mục, chỉ có Điều
  (Quyết định ngắn) — đảm bảo thuật toán stack không lỗi khi nhảy cấp.
- `tests/test_component_index.py` — `component_index` (Pass 1) trả đúng `comp_id` theo
  `citation_path`, và phân biệt được "Khoản 1" trùng số ở 2 Điều khác nhau nhờ luôn kèm Điều tổ tiên.
- `tests/test_action_extractor.py` — `find_amendments` khớp được cả 2 đầu (Component A + citation
  Component B) cho 4 khuôn mẫu (SUA_DOI/BO_SUNG, BAI_BO, THAY_THE_CUM_TU, BO_CUM_TU), và không tạo
  Action khi citation không khớp được `component_index` (chỉ giữ Tầng A).
- `tests/test_relation_label_map.py` — tất cả 17 nhãn thực tế trong `relationships.parquet`
  đều có trong `RELATION_LABEL_MAP` hoặc `REVERSE_RELATION_LABEL_MAP`.

### Integration tests — Neo4j RAG quality (chạy sau `--stage load`)

**Yêu cầu**: `.env` đã có `NEO4J_URI` + `NEO4J_PASSWORD`. Nếu chưa set hoặc không kết nối
được, toàn bộ test tự động **skip** (không fail).

```bash
# Điền thông tin Neo4j vào .env trước:
# NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
# NEO4J_USER=neo4j
# NEO4J_PASSWORD=<password từ AuraDB console>

# Chạy toàn bộ (8 test case):
pytest tests/test_neo4j_rag_quality.py -v

# Chỉ chạy 6 test chặn RAG:
pytest tests/test_neo4j_rag_quality.py -v -m critical
```

**8 test case** (xem `TEST_BRIEF.md` để biết logic chi tiết):

| TC | Kiểm tra | Chặn RAG? |
|---|---|---|
| TC-01 | Node lá có TextUnit (≥95%) | Có |
| TC-02 | Cây không đứt gãy (Component → Norm) | Có |
| TC-03 | Action đủ 2 đầu (HAS_ACTION + APPLY_TO) | Có |
| TC-04 | Vector index tồn tại và đầy đủ | Có |
| TC-05 | Traversal ngược TextUnit → Norm lấy được số hiệu | Có |
| TC-06 | Vector search end-to-end (embed → index → kết quả hợp lý) | Có |
| TC-07 | TextUnit không rỗng (< 1% quá ngắn) | Gián tiếp |
| TC-08 | Citation path đúng chiều (không bị đảo ngược) | Có |

**Thứ tự ưu tiên** khi debug: `TC-01 → TC-02 → TC-04 → TC-06` — 4 test này đủ xác nhận
RAG chạy được hay không. TC-03 / TC-05 / TC-07 / TC-08 chạy khi cần debug sâu hơn.

**TC-04 / TC-05 / TC-06** cần tạo vector index thủ công sau khi `embed` xong — chưa có
trong `schema_init.cypher`:

```cypher
CREATE VECTOR INDEX textunit_embedding_index IF NOT EXISTS
FOR (t:TextUnit) ON (t.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 3072, `vector.similarity_function`: 'cosine'}};
```

Chạy lệnh trên trong **Neo4j Browser** (`https://<id>.databases.neo4j.io`) hoặc
`cypher-shell`. Sau khi tạo, chờ Neo4j populate index (có thể vài phút với corpus lớn)
rồi mới chạy lại test. TC-04 / TC-05 / TC-06 tự **skip** nếu index chưa tồn tại.

## Lưu ý khi chạy full corpus

1. Model `gemini-3.5-flash` (`LLM_MODEL_HEAVY` trong `.env`) là tên gọi theo brief gốc —
   kiểm tra model thật đang khả dụng trên Vertex AI tại thời điểm chạy và cập nhật `.env`
   nếu cần (không cần sửa code, mọi tên model đều đọc từ config).
2. Cột thật của dataset (đã verify qua HF API) khác tên field so với bản rút gọn trong
   brief — `transform/pipeline.py` đã map đúng theo cột thật (`so_ky_hieu`, `loai_van_ban`,
   `ngay_ban_hanh`, `tinh_trang_hieu_luc`, ...).
3. Chạy `--stage transform --sample 200` trước, kiểm tra vài chục `Component`/`Action`
   sinh ra bằng mắt, trước khi chạy `--stage all` trên toàn bộ 
