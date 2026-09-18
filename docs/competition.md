# Quy Chế Và Tuân Thủ Quy Định Cuộc Thi UIT DSC 2026 Task 2

Tài liệu này tổng hợp các quy định bắt buộc của Ban tổ chức UIT Data Science Challenge 2026 (Task 2) và cam kết tuân thủ của dự án.

---

## 1. Mục Tiêu Và Dạng Bài Thi

- **Chủ đề:** Trả lời câu hỏi pháp luật Việt Nam (Task 2 - Legal Question Answering).
- **Đầu vào (Input):** Câu hỏi pháp luật bằng tiếng Việt tự nhiên.
- **Đầu ra (Output):** Câu trả lời dạng văn xuôi tiếng Việt hoàn chỉnh, căn cứ vào dữ liệu pháp luật chính thức.
- **Nền tảng nộp bài:** Nền tảng chấm và xếp hạng chính thức do Ban tổ chức cung cấp (CodaLab / Codabench).
- **Quy cách tệp nộp:** Tệp lưu trữ nén `submission.zip` chứa duy nhất tệp `submission.json` định dạng UTF-8. Định dạng dữ liệu là một object ánh xạ mã câu hỏi `question_id` sang object câu trả lời:
  ```json
  {
    "question_id": {
      "answer": "Nội dung câu trả lời bằng tiếng Việt..."
    }
  }
  ```

---

## 2. Quy Định Về Dữ Liệu

- **Chỉ sử dụng dữ liệu chính thức:** Toàn bộ quá trình tiền xử lý, lập chỉ mục và tinh chỉnh mô hình chỉ được phép sử dụng tập dữ liệu do Ban tổ chức UIT Data Science Challenge 2026 Task 2 cung cấp.
- **Cấm thu thập dữ liệu ngoài:** Nghiêm cấm thu thập thêm dữ liệu từ Internet hoặc các nguồn dữ liệu bên ngoài.
- **Cấm tăng cường dữ liệu từ nguồn ngoài:** Không được phép đưa dữ liệu ngoài vào pipeline nhằm mục đích data augmentation.
- **Cấm gán nhãn thủ công (Manual Labeling):** Không thực hiện gán nhãn thủ công hoặc can thiệp chủ quan vào nhãn dữ liệu.

---

## 3. Quy Định Về Mô Hình Và Giới Hạn Tham Số

- **Giới hạn tham số:** Tổng số tham số của tất cả các mô hình có trọng số trong toàn bộ hệ thống dự thi phải **nhỏ hơn 4 tỷ (< 4.000.000.000)**.
- **Phạm vi tính toán tham số:** Bao gồm toàn bộ các mô hình tham gia vào quy trình:
  1. Mô hình nhúng (Dense Embedding model)
  2. Mô hình xếp hạng lại (Neural Reranker model)
  3. Mô hình sinh câu trả lời (Generator base model + LoRA parameters)
  4. Bất kỳ mô hình phụ trợ học máy nào khác.
- **Chính sách kỹ thuật nén / LoRA:** Việc sử dụng các kỹ thuật quantization (4-bit, 8-bit), LoRA hoặc offloading không làm giảm số tham số danh định được Ban tổ chức tính toán. Một mô hình gốc từ 4B trở lên vẫn không hợp lệ dù được lượng tử hóa.
- **Mô hình mã nguồn mở:** Các mô hình nguồn mở hoặc giấy phép nghiên cứu/phi thương mại được phép sử dụng nếu thuộc danh mục đăng ký với BTC.
- **Cấm sử dụng API thương mại:** Nghiêm cấm gọi API mô hình từ các dịch vụ bên ngoài (kể cả dịch vụ miễn phí hoặc phi thương mại). Hệ thống phải tải, triển khai và kiểm soát trực tiếp trên phần cứng của đội thi.

---

## 4. Công Thức Đánh Giá Chính Thức (Scoring Contract)

Ban tổ chức sử dụng hai thước đo tự động chính thức để xếp hạng:

1. **METEOR (Thước đo xếp hạng chính):**
   - Đánh giá trên token phân tách bằng khoảng trắng (whitespace tokens).
   - Sử dụng NLTK METEOR scorer chính thức với WordNet/OMW.
   - Tính trung bình số học (arithmetic macro mean) trên toàn bộ tập câu hỏi.

2. **ROUGE-L (Thước đo phụ trợ):**
   - Đánh giá trên chuỗi con chung dài nhất (LCS).
   - Sử dụng bộ đánh giá vendored ROUGE-L chuẩn hóa ký tự ASCII của BTC.
