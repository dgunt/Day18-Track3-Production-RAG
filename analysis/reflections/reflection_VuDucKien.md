# Individual Reflection — Lab 18

**Tên:** Vũ Đức Kiên-2A202600338  
**Module phụ trách:** M1 — Advanced Chunking Strategies

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** `src/m1_chunking.py` — Module 1: Advanced Chunking Strategies
- **Các hàm/class chính đã viết:**
  - `chunk_semantic()` — Tách câu bằng `underthesea.sent_tokenize`, encode bằng `SentenceTransformer`, nhóm câu theo Cosine Similarity < threshold
  - `chunk_hierarchical()` — Tạo Parent chunks (2048 chars) → tách thành Child chunks (256 chars), mỗi Child có `parent_id` trỏ về Parent
  - `chunk_structure_aware()` — Parse Markdown headers H1/H2/H3 bằng Regex, tạo chunk theo từng section, lưu tên section vào metadata
  - `compare_strategies()` — Chạy cả 4 chiến lược, thống kê và in bảng so sánh (num_chunks, avg_len, min_len, max_len)
- **Số tests pass:** 13/13 (100%)

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hierarchical Chunking (Parent-Child pattern) — index Child nhỏ để embedding chính xác, nhưng trả Parent lớn cho LLM để có đủ context. Đây là pattern được dùng trong Production RAG thực tế.
- **Điều bất ngờ nhất:** Sau khi chạy `compare_strategies()` trên dữ liệu thực tế (văn bản tài chính/pháp lý tiếng Việt), kết quả cho thấy sự khác biệt rõ rệt giữa các chiến lược:

  | Strategy | Chunks | Avg Len | Nhận xét |
  |---|---|---|---|
  | basic | 162 | 439 chars | Baseline ổn định |
  | semantic | 566 | 125 chars | ⚠️ Over-split — threshold 0.85 quá cao |
  | hierarchical | 292 | 244 chars | ✅ Tốt nhất cho dữ liệu này |
  | structure | 2 | 35,778 chars | ❌ PDF-converted MD không có `#` headers |

  Điều bất ngờ: `structure-aware` vốn rất hiệu quả với tài liệu có cấu trúc rõ ràng lại thất bại hoàn toàn khi dùng với file Markdown được convert từ PDF (không có header `#` chuẩn). Ngược lại, `hierarchical` cho kết quả ổn định nhất, đúng với lý thuyết đề xuất nó là **default recommendation** cho Production RAG.

- **Kết nối với bài giảng:** Slide về "Chunking Strategies in RAG" — đặc biệt phần trade-off giữa Precision (retrieval) và Context (LLM). Thực nghiệm xác nhận: không có chiến lược nào là tốt nhất cho mọi loại dữ liệu — phải chọn theo đặc tính corpus.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn 1:** `SentenceTransformer` crash với lỗi `Windows fatal exception: access violation` khi chạy trong pytest — do model cố gọi network (HuggingFace) trong thread của pytest trên Windows.
  - **Giải quyết:** Pre-download model trước khi chạy test; dùng `local_files_only=True` để ưu tiên cache local, tránh network call.

- **Khó khăn 2:** `chunk_semantic()` oversplit dữ liệu thực tế (566 chunks từ 2 tài liệu) do `SEMANTIC_THRESHOLD = 0.85` quá cao. Văn bản tài chính/pháp lý tiếng Việt có câu ngắn, topic thay đổi liên tục → similarity giữa câu liên tiếp thường dưới 0.85.
  - **Insight:** Threshold cần tune theo từng loại corpus. Với văn bản pháp lý, threshold 0.3–0.5 sẽ phù hợp hơn.

- **Thời gian debug:** ~25 phút tổng (15' cho crash Windows + 10' phân tích kết quả thực tế).

## 4. Nếu làm lại

1. Dùng `BAAI/bge-m3` thay vì `all-MiniLM-L6-v2` — tối ưu hơn cho tiếng Việt.
2. Thêm overlap giữa các Child chunks trong Hierarchical để tránh mất context ở ranh giới.
3. Với `structure-aware`, thêm fallback nhận diện bold text (`**...**`) hoặc dòng ALL CAPS làm "pseudo-header" cho file PDF-converted.
4. Thêm A/B test với nhiều threshold trong `compare_strategies()` để tự động tìm threshold tối ưu.

- **Module muốn thử tiếp:** M2 (Hybrid Search) — muốn hiểu cách BM25 + Dense vector kết hợp bằng RRF.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 4 |
