# Individual Reflection — Lab 18

**Tên:** Ngô Gia Bảo-2A202600385  
**Module phụ trách:** M3: Reranking

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** Module 3 - Reranking (Cross-encoder & Flashrank).
- **Các hàm/class chính đã viết:**
    - `CrossEncoderReranker`: Sử dụng model `BAAI/bge-reranker-v2-m3` để xếp hạng lại top-k kết quả.
    - `FlashrankReranker`: Một giải pháp thay thế siêu nhẹ (<5ms) khi không có GPU hoặc cần tốc độ cao.
    - `benchmark_reranker`: Hàm đo lường hiệu năng (latency) trung bình, min, max qua nhiều lần chạy.
- **Số tests pass:** 5/5 (vượt qua tất cả các test cases về tính đúng đắn, kiểu dữ liệu và thứ tự xếp hạng).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** Hiểu rõ sự khác biệt giữa **Bi-encoders** (dùng cho retrieval nhanh) và **Cross-encoders** (dùng cho reranking chính xác nhưng chậm hơn).
- **Điều bất ngờ nhất:** Model `bge-reranker-v2-m3` có khả năng hiểu ngữ nghĩa tiếng Việt cực tốt, có thể đẩy các đoạn hội thoại thực sự liên quan lên đầu ngay cả khi điểm BM25 thấp.
- **Kết nối với bài giảng:** Áp dụng kiến thức về "Two-stage Retrieval Pipeline" và "Precision at K" để tối ưu hóa context cho LLM.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Model reranker khá nặng, gây chậm ở lần load đầu tiên (hơn 60 giây) và tiêu tốn nhiều RAM.
- **Cách giải quyết:** Implement cơ chế **Lazy Loading** (chỉ load model khi thực sự gọi hàm `rerank`) và thêm **Fallback mechanism** (sử dụng word overlap hoặc Flashrank) để pipeline không bị crash khi lỗi model.
- **Thời gian debug:** Khoảng 30 phút để xử lý các lỗi dependencies của `FlagEmbedding`.

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Sẽ thử nghiệm thêm kỹ thuật **Metadata Filtering** kết hợp với Reranking để giảm số lượng documents cần rerank, từ đó tối ưu latency.
- **Module nào muốn thử tiếp:** Module 2 (Hybrid Search) để tìm hiểu cách kết hợp trọng số (weights) tối ưu giữa BM25 và Dense.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 5 |
| Code quality | 5 |
| Teamwork | 4 |
| Problem solving | 5 |
