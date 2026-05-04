# Individual Reflection — Lab 18

**Tên:** Vũ Đức Minh-2A202600459
**Module phụ trách:** M4 — RAGAS Evaluation & Failure Analysis

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** `src/m4_eval.py` — Module 4: RAGAS Evaluation
- **Các hàm/class chính đã viết hoặc hoàn thiện:**
  - `EvalResult` — dataclass lưu thông tin từng câu hỏi, câu trả lời, context, ground truth và 4 metric đánh giá.
  - `load_test_set()` — load bộ test set từ file JSON, đảm bảo dữ liệu có `question` và `ground_truth`.
  - `evaluate_ragas()` — tạo `Dataset.from_dict()`, gọi `ragas.evaluate()` với 4 metric: `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, sau đó trích xuất điểm tổng hợp và điểm theo từng câu hỏi.
  - `failure_analysis()` — lấy bottom-N câu có điểm trung bình thấp nhất, tìm metric kém nhất và map vào Diagnostic Tree để đưa ra `diagnosis` và `suggested_fix`.
  - `save_report()` — lưu kết quả đánh giá và failure analysis ra file JSON report.
- **Số tests pass:** 4/4 cho `tests/test_m4.py` (100%).

## 2. Kiến thức học được

- **Khái niệm mới nhất:** RAGAS evaluation cho hệ thống RAG. Thay vì chỉ nhìn câu trả lời đúng/sai thủ công, RAGAS tách chất lượng hệ thống thành nhiều góc nhìn: LLM có bịa không (`faithfulness`), câu trả lời có đúng trọng tâm không (`answer_relevancy`), context lấy về có liên quan không (`context_precision`) và có bị thiếu thông tin quan trọng không (`context_recall`).
- **Điều bất ngờ nhất:** Một pipeline RAG có thể trả lời nghe rất tự nhiên nhưng vẫn fail ở một metric cụ thể. Ví dụ, nếu `context_recall` thấp thì lỗi nằm nhiều ở retrieval/chunking; nếu `faithfulness` thấp thì lỗi nghiêng về hallucination của LLM. Việc tách metric giúp debug có hướng hơn thay vì chỉ nói chung chung là “answer sai”.
- **Kết nối với bài giảng:** Phần Production RAG Evaluation và Error Analysis. Module này cho thấy evaluation không chỉ là tính điểm cuối cùng, mà còn là cách tìm root cause: retrieval sai, context thiếu, context nhiễu, hay prompt/LLM sinh câu trả lời chưa đúng.

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** Output từ `result.to_pandas()` của RAGAS không luôn giữ đủ các cột gốc như `question`, `answer`, `contexts`. Ban đầu khi đọc `row.question` có thể gặp lỗi `AttributeError`.
- **Cách giải quyết:** Dùng input gốc (`questions[i]`, `answers[i]`, `contexts[i]`, `ground_truths[i]`) để tạo `EvalResult`, còn các cột metric thì lấy từ dataframe RAGAS nếu có. Cách này ổn định hơn giữa các phiên bản RAGAS khác nhau.
- **Khó khăn phụ:** RAGAS có thể phụ thuộc vào package/model/API bên ngoài, nên nếu môi trường thiếu dependency hoặc evaluate lỗi thì pipeline dễ bị crash.
- **Cách xử lý:** Thêm fallback trả về kết quả rỗng với score `0.0` và vẫn tạo `per_question`, giúp test và pipeline không bị dừng đột ngột.
- **Thời gian debug:** Khoảng 20-30 phút, chủ yếu để hiểu shape output của RAGAS và sửa lỗi khi dataframe không có thuộc tính `question`.

## 4. Nếu làm lại

1. Log rõ lỗi khi `ragas.evaluate()` fail để phân biệt score thấp thật với lỗi môi trường/API.
2. Lưu thêm `answer`, `ground_truth` và một vài context liên quan trong failure report để nhóm dễ phân tích thủ công hơn.
3. Thêm ngưỡng cảnh báo cho từng metric, ví dụ `faithfulness < 0.85` hoặc `context_recall < 0.75`, để tự động highlight lỗi nghiêm trọng.
4. Chạy đánh giá so sánh giữa naive baseline và production RAG để thấy cải thiện sau khi thêm chunking, hybrid search và reranking.

- **Module muốn thử tiếp:** M2 hoặc M3, vì hai module đó ảnh hưởng trực tiếp đến `context_precision` và `context_recall`, tức là hai metric rất quan trọng trong đánh giá RAG.

## 5. Tự đánh giá

| Tiêu chí | Tự chấm (1-5) |
|----------|---------------|
| Hiểu bài giảng | 4 |
| Code quality | 4 |
| Teamwork | 4 |
| Problem solving | 4 |
