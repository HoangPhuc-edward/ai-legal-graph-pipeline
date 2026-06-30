# CLAUDE.md — Quy tắc bắt buộc cho dự án này

## ⚠️ QUY TẮC CỨNG SỐ 1 — KHÔNG bao giờ load full file parquet để "xem kiến trúc/dữ liệu"

Máy đang chạy **RAM giới hạn**. Việc gọi `pd.read_parquet(path)` hoặc `datasets.load_dataset(...)` trên `content.parquet` (178,665 dòng HTML thô) hoặc `relationships.parquet` (897,890 dòng) để "xem thử cấu trúc/vài dòng đầu" **gây tràn RAM và crash máy đang dùng để soạn code**.

**Toàn bộ schema của 3 file đã được liệt kê đầy đủ ở Mục 2 dưới đây.** Trước khi viết bất kỳ code nào động đến `content_html`/`metadata`/`relationships`, đọc Mục 2 trước — **không tự ý load file để "kiểm tra cho chắc"**.

Nếu thực sự cần xem dữ liệu thật (không phải schema — schema đã có sẵn), **bắt buộc dùng cách lấy mẫu an toàn ở Mục 3**, không dùng `pd.read_parquet()` trống tay.

---

## 2. Schema đầy đủ — KHÔNG cần load file để biết lại

Nguồn: [`th1nhng0/vietnamese-legal-documents`](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents), đã xác minh trực tiếp từ dataset card (không phải suy đoán).

### Config `metadata` — 153,420 dòng, 16 cột

| Cột | Mô tả |
|---|---|
| `id` | ID văn bản (int) — khoá join với `content.id` và `relationships.doc_id` |
| `title` | Tiêu đề đầy đủ |
| `so_ky_hieu` | Số hiệu chính thức (vd: `115/NQ-HĐBCQG`) |
| `ngay_ban_hanh` | Ngày ban hành (`DD/MM/YYYY`) |
| `loai_van_ban` | Loại văn bản — Quyết định, Nghị quyết, Thông tư... |
| `ngay_co_hieu_luc` | Ngày có hiệu lực |
| `ngay_het_hieu_luc` | Ngày hết hiệu lực (rỗng nếu còn hiệu lực) |
| `nguon_thu_thap` | Nguồn thu thập (vd: Công báo) |
| `ngay_dang_cong_bao` | Ngày đăng Công báo |
| `nganh` | Ngành — Tài chính, Y tế... |
| `linh_vuc` | Lĩnh vực pháp luật |
| `co_quan_ban_hanh` | Cơ quan ban hành (551 cơ quan khác nhau) |
| `chuc_danh` | Chức danh người ký |
| `nguoi_ky` | Tên người ký |
| `pham_vi` | Phạm vi địa lý áp dụng |
| `thong_tin_ap_dung` | Ghi chú thi hành |
| `tinh_trang_hieu_luc` | Trạng thái hiệu lực — Còn hiệu lực, Hết hiệu lực toàn bộ... |

### Config `content` — 178,665 dòng, 2 cột (⚠️ FILE NẶNG NHẤT, dễ tràn RAM nhất)

| Cột | Mô tả |
|---|---|
| `id` | ID văn bản — khoá join với `metadata.id` |
| `content_html` | HTML thô toàn văn |

Lưu ý: không phải `id` nào trong `metadata` cũng có dòng tương ứng trong `content` (một số văn bản chỉ có bản scan PDF, không có HTML).

### Config `relationships` — 897,890 dòng, 3 cột

| Cột | Mô tả |
|---|---|
| `doc_id` | ID văn bản nguồn — khoá join với `metadata.id` |
| `other_doc_id` | ID văn bản đích |
| `relationship` | Nhãn quan hệ (17 giá trị — xem `Relationship_Label_Mapping.md` nếu có trong repo) |

### Config `legacy` (518k dòng, field tiếng Anh) — KHÔNG dùng trong v1, bỏ qua hoàn toàn trừ khi được yêu cầu rõ ràng.

---

## 3. Cách xem dữ liệu thật an toàn — khi schema ở Mục 2 không đủ

Dùng `pyarrow` đọc **schema only** (không tải dữ liệu) hoặc đọc **đúng N dòng** (không tải cả file):

```python
import pyarrow.parquet as pq

# Xem schema thật của file cục bộ — KHÔNG tải bất kỳ dòng dữ liệu nào
pf = pq.ParquetFile("data_cache/content.parquet")
print(pf.schema_arrow)
print("Tổng số dòng:", pf.metadata.num_rows)

# Lấy đúng 5 dòng đầu để xem mẫu thật — CHỈ tải 5 dòng, không tải cả file
batch = next(pf.iter_batches(batch_size=5))
df_sample = batch.to_pandas()
print(df_sample)
```

**Cấm tuyệt đối** các lệnh sau khi chưa có `nrows`/`batch_size`/`--sample`:
```python
pd.read_parquet("data_cache/content.parquet")          # ❌ tải cả 178k dòng HTML
datasets.load_dataset("th1nhng0/vietnamese-legal-documents", "content")  # ❌ tải cả + lỗi ArrowInvalid đã biết
```

---

## 4. Quy tắc làm việc chung trong dự án này

- Mọi lệnh chạy thử pipeline **bắt buộc** kèm `--sample N` (vd. `--sample 200`) — không chạy full corpus khi đang debug code.
- Trước khi thêm logic mới động vào `content_html`/`metadata`/`relationships`, kiểm tra lại Mục 2 — đừng tự load file để "xác nhận lại" tên cột, tên cột đã đúng 100% như liệt kê ở trên.
- Nếu cần biết phân phối giá trị (vd. đếm theo `loai_van_ban`, theo `relationship`) — dùng `pyarrow.compute` trên `ParquetFile` theo batch, hoặc hỏi người dùng trước (số liệu thật của 17 nhãn `relationship` đã có sẵn, xem `Relationship_Label_Mapping.md`), **không** load full file chỉ để `.value_counts()`.