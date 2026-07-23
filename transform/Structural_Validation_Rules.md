# Bộ Rule Phát hiện Bất thường Cấu trúc (Structural Sanity Rules)

> Nguyên tắc thiết kế: mọi rule chỉ nhìn **cấu trúc/con số/thống kê** — không đọc hiểu
> nội dung pháp lý. Rule dựa trên 2 quy luật nền tảng của văn bản pháp luật Việt Nam:
> 1. **Điều đánh số liên tục toàn văn bản**, không reset theo Chương/Mục/Phần.
> 2. **Khoản/Điểm luôn reset về đầu (1/a) mỗi khi có cha mới** (Điều mới → Khoản lại từ 1).

Mỗi rule có: mô tả, cách phát hiện (Cypher hoặc pseudocode), và hành động đề xuất.

---

## Nhóm A — Trùng lặp (Duplication)

### Rule A1 — 2 Component cùng citation, cùng cha *(rule gốc của bạn)*
```cypher
MATCH (parent)-[:HAS_COMPONENT]->(c:Component)
WITH parent, c.citation AS cit, collect(c.comp_id) AS ids, count(*) AS n
WHERE n > 1
RETURN parent.comp_id, cit, ids, n
```
**Hành động:** giữ bản có `_raw_text_score()` cao nhất (đã có sẵn), xoá cascade các bản còn lại.

### Rule A2 — 2 Chương/Mục có `title_text` giống hệt nhau, khác `comp_id`
Bắt các ghost mà A1 bỏ sót vì `(parent, citation)` khác nhau — dấu hiệu triplication ở cấp cao hơn dedup hiện tại xử lý.
```cypher
MATCH (c1:Component), (c2:Component)
WHERE c1.comp_id < c2.comp_id
  AND c1.level = c2.level
  AND c1.title_text = c2.title_text
  AND c1.norm_id = c2.norm_id
RETURN c1.comp_id, c2.comp_id, c1.title_text
```
**Hành động:** xem xét thủ công — có thể là ghost hoặc văn bản thực sự lặp tiêu đề (hiếm nhưng có thể xảy ra ở phụ lục).

---

## Nhóm B — Đứt gãy trình tự (Sequence integrity)

### Rule B1 — Điều nhảy số: có Điều 1, Điều 4 mà thiếu 2, 3 *(rule gốc của bạn)*
```
detected = sorted số Điều theo order_index trong 1 Norm
for i in range(len(detected)-1):
    if detected[i+1] - detected[i] > 1:
        flag(f"Thiếu Điều {detected[i]+1}..{detected[i+1]-1}")
```
**Hành động:** rescan `raw_text` của Điều liền trước gap để tìm marker bị nuốt; nếu không tìm thấy → áp dụng citation alias (xem thiết kế ở lượt trước).

### Rule B4 — Điểm nhảy chữ (a, c thiếu b)
Tương tự B1 nhưng cho chuỗi chữ cái thay vì số — dùng `ord(char) - ord('a')` để so sánh tuần tự.

---

## Nhóm C — Bất thường số lượng con (Cardinality)

### Rule C1 — Cả văn bản chỉ có 1 Điều *(rule gốc của bạn — cần thêm điều kiện)*
```cypher
MATCH (n:Norm)-[:HAS_COMPONENT*]->(d:Component {level:"DIEU"})
WITH n, count(d) AS n_dieu
WHERE n_dieu = 1
RETURN n.norm_id, n.title, n_dieu
```
**Lưu ý quan trọng:** không tự động flag mọi trường hợp — một số Quyết định/Chỉ thị hành chính **thật sự** chỉ có 1 Điều (đã thấy trong data thật, ví dụ Sắc lệnh bổ nhiệm nhân sự). Chỉ flag nếu **đồng thời** `content_html` gốc dài hơn 1 ngưỡng (ví dụ >3000 ký tự) — văn bản dài mà chỉ ra 1 Điều mới thực sự đáng ngờ.

### Rule C2 — Điều có ĐÚNG 1 Khoản con *(rule mới, đã test)*
Về logic soạn thảo: nếu cần chia Khoản, hầu như luôn có ≥2 Khoản (để phân biệt các trường hợp). Đúng 1 Khoản là dấu hiệu Khoản 2+ bị nuốt.
```cypher
MATCH (d:Component {level:"DIEU"})-[:HAS_COMPONENT]->(k:Component {level:"KHOAN"})
WITH d, count(k) AS n_khoan
WHERE n_khoan = 1
RETURN d.comp_id, d.citation, n_khoan
```
**Hành động:** rescan raw_text của Khoản duy nhất đó để tìm `"2."`, `"3."` có bị dính vào không.

### Rule C3 — Chương/Mục/Phần có 0 Điều con (rỗng)
Một Chương tồn tại nhưng không tổ chức Điều nào bên trong là vô nghĩa về mặt soạn thảo.
```cypher
MATCH (ch:Component)
WHERE ch.level IN ["CHUONG", "MUC", "PHAN"]
  AND NOT (ch)-[:HAS_COMPONENT*]->(:Component {level:"DIEU"})
RETURN ch.comp_id, ch.level, ch.citation, ch.title_text
```
**Hành động:** khả năng cao đây là ghost (title bị tách nhầm thành node) hoặc Chương thật nhưng Điều bên trong bị gán nhầm sang node khác (liên quan Rule B2).

---

## Nhóm D — Bất thường độ dài (Length anomalies)

### Rule D1 — TextUnit dài bất thường *(rule gốc của bạn — nên dùng thống kê tương đối)*
Không dùng ngưỡng cố định (ví dụ ">5000 ký tự") vì độ dài "bình thường" khác nhau giữa Bộ luật và Quyết định ngắn. Dùng **median + MAD** (median absolute deviation) so với các Component cùng cấp trong cùng Norm — đã test, phát hiện đúng outlier mà không cần biết ngưỡng tuyệt đối trước.
```python
# Chạy trong Python sau khi query text_length của mọi Component cùng level, cùng Norm
import statistics
med = statistics.median(lengths)
mad = statistics.median([abs(l - med) for l in lengths])
threshold = med + 6 * max(mad, 1)
outliers = [c for c in components if c.text_length > threshold]
```
**Hành động:** rescan raw_text tìm marker Điều/Khoản bị nuốt bên trong.

### Rule D2 — Node lá gần rỗng (dấu hiệu over-split) *(rule mới, đã test)*
Đối lập với D1 — bắt đúng kịch bản "Được đề cập trong Chương 1, nó sẽ áp dụng" bị nhận nhầm thành node Chương.
```cypher
MATCH (c:Component)-[:HAS_TEXTUNIT]->(tu:TextUnit)
WHERE size(tu.accumulated_text) < 30
  AND NOT (c)-[:HAS_COMPONENT]->(:Component)   // là node lá thật
RETURN c.comp_id, c.level, c.citation, tu.accumulated_text
```
**Hành động:** kiểm tra `is_bold`/`is_block_start` (nếu đã lưu signal từ HTML) — nếu KHÔNG có tín hiệu cấu trúc thật nào hỗ trợ, khả năng cao là false positive từ regex, nên gộp ngược vào node cha thay vì giữ làm node riêng.

---

## Nhóm E — Bất thường hình dạng cây (Tree shape)

### Rule E1 — Nhảy cấp bất hợp lệ (thiếu tầng trung gian)
Ví dụ Chương nối thẳng xuống Khoản, không qua Điều — vi phạm rank hierarchy đã định nghĩa (`CHUONG:1 → DIEU:4 → KHOAN:5`), không nên xảy ra nếu stack builder đúng, nhưng đáng để bẫy sớm phòng khi có bug mới.
```cypher
MATCH (parent:Component)-[:HAS_COMPONENT]->(child:Component)
WITH parent, child,
     CASE parent.level WHEN "CHUONG" THEN 1 WHEN "MUC" THEN 2 WHEN "DIEU" THEN 4 WHEN "KHOAN" THEN 5 END AS p_rank,
     CASE child.level  WHEN "CHUONG" THEN 1 WHEN "MUC" THEN 2 WHEN "DIEU" THEN 4 WHEN "KHOAN" THEN 5 WHEN "DIEM" THEN 6 END AS c_rank
WHERE c_rank - p_rank > 2   // nhảy quá 1 tầng liền kề là bất thường
RETURN parent.comp_id, parent.level, child.comp_id, child.level
```
**Hành động:** bug trong `parse_structure()` — ưu tiên sửa code parser, đây không phải lỗi có thể patch bằng rescan.

---

## Bảng tổng hợp — độ ưu tiên xử lý

| Rule | Loại lỗi bắt được | Sửa bằng cách nào |
|---|---|---|
| A1, A2 | Trùng lặp (ghost, triplication) | Xoá cascade |
| B1, B4, C2 | Under-split (nuốt nội dung) | Rescan raw_text → tạo Component mới |
| E1 | Bug logic parser | Sửa code, không patch data |
| C1, C3 | Cấu trúc rỗng bất thường | Rescan hoặc kiểm tra thủ công |
| D1 | Under-split (nuốt nội dung) | Rescan (warn nếu sau B1/B3/B4/C2 vẫn còn) |
| D2 | Over-split (trích dẫn bị nhận nhầm) | Gộp ngược vào cha |

**Gợi ý vận hành:** chạy toàn bộ rule này như 1 script `validate_structure.py` sau mỗi lần `--stage transform`, in ra danh sách comp_id vi phạm theo từng rule, kèm mức độ ưu tiên (E1 là bug cần sửa code ngay; còn lại có thể tích lũy rồi xử lý theo batch).
