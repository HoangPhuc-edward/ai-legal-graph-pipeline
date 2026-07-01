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

### Tra cứu dữ liệu thô an toàn (`extract/hf_dataset.py`)

Đọc parquet **theo batch** (không bao giờ load full file — xem [`CLAUDE.md`](CLAUDE.md) Mục 1).
Chạy lại nhiều lần **không bị duplicate** — file JSON ghi đè, Neo4j dùng `MERGE`.

```bash
# Lấy mẫu an toàn để chạy thử pipeline: N metadata đầu (mặc định 100)
# -> content tương ứng -> tối đa N relationships liên quan
python -m extract.hf_dataset sample           # N=100, ghi ra data/samples/sample.json
python -m extract.hf_dataset sample --n 20   # giảm xuống 20 nếu máy yếu

# Tìm content_html khớp từ khoá trong extract/keywords.txt
# -> content + metadata + relationships liên quan, ghi ra JSON
python -m extract.hf_dataset keyword                  # quét full, lấy tất cả kết quả
python -m extract.hf_dataset keyword --limit 50       # giới hạn 50 văn bản khớp
python -m extract.hf_dataset keyword --keywords-file PATH --output PATH  # tuỳ chỉnh file

# Tải parquet (dành cho Colab / lưu vào Google Drive):
python -m extract.hf_dataset download                               # tải cả 3, bỏ qua file đã có
python -m extract.hf_dataset download --output-dir /content/drive/MyDrive/legal/raw
python -m extract.hf_dataset download --configs metadata relationships  # chỉ 2 config
python -m extract.hf_dataset download --force                       # tải lại dù đã có sẵn
```

Kết quả ghi ra `data/samples/sample.json` / `data/samples/keyword_search.json`
(tự tạo thư mục). Nếu parquet chưa có cục bộ, lệnh tự tải về `data/raw/` trước khi dùng.

## Test

```bash
pytest
```

- `tests/test_structure_parser.py` — fixture đủ 7 cấp, thiếu Phần/Mục, chỉ có Điều
  (Quyết định ngắn) — đảm bảo thuật toán stack không lỗi khi nhảy cấp.
- `tests/test_component_index.py` — `component_index` (Pass 1) trả đúng `comp_id` theo
  `citation_path`, và phân biệt được "Khoản 1" trùng số ở 2 Điều khác nhau nhờ luôn kèm Điều tổ tiên.
- `tests/test_action_extractor.py` — `find_amendments` khớp được cả 2 đầu (Component A + citation
  Component B) cho 4 khuôn mẫu (SUA_DOI/BO_SUNG, BAI_BO, THAY_THE_CUM_TU, BO_CUM_TU), và không tạo
  Action khi citation không khớp được `component_index` (chỉ giữ Tầng A). Khi chạy trên corpus thật,
  nên đo lại tỷ lệ tách theo `norm_type`/năm ban hành (văn bản trước 2016 hoặc cấp địa phương dự
  kiến tỷ lệ match thấp hơn).

## Lưu ý khi chạy full corpus

1. Model `gemini-3.5-flash` (`LLM_MODEL_HEAVY` trong `.env`) là tên gọi theo brief gốc —
   kiểm tra model thật đang khả dụng trên Vertex AI tại thời điểm chạy và cập nhật `.env`
   nếu cần (không cần sửa code, mọi tên model đều đọc từ config).
2. Cột thật của dataset (đã verify qua HF API) khác tên field so với bản rút gọn trong
   brief — `transform/pipeline.py` đã map đúng theo cột thật (`so_ky_hieu`, `loai_van_ban`,
   `ngay_ban_hanh`, `tinh_trang_hieu_luc`, ...).
3. Chạy `--stage transform --sample 200` trước, kiểm tra vài chục `Component`/`Action`
   sinh ra bằng mắt, trước khi chạy `--stage all` trên toàn bộ corpus.
