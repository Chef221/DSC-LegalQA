# Quy chế cuộc thi UIT DSC 2026 Task 2

Tài liệu này tổng hợp các quy định bắt buộc của BTC UIT Data Science Challenge 2026 (Task 2) và cam kết tuân thủ của dự án.

---

## 1. Mục tiêu và format bài thi

- **Chủ đề:** Trả lời câu hỏi pháp luật Việt Nam (Task 2 - Legal Question Answering).
- **Input:** Câu hỏi pháp luật bằng tiếng Việt tự nhiên.
- **Output:** Câu trả lời dạng văn xuôi tiếng Việt, căn cứ vào dữ liệu pháp luật chính thức.
- **Nền tảng nộp/chấm bài:** Nền tảng nộp/chấm bài chính thức do BTC cung cấp.
- **Quy cách submission:** File zip `submission.zip` chứa duy nhất file `submission.json` (UTF-8). Format là một JSON object ánh xạ `question_id` sang answer object:
  ```json
  {
    "question_id": {
      "answer": "Nội dung câu trả lời bằng tiếng Việt..."
    }
  }
  ```

---

## 2. Quy định về dữ liệu

- **Chỉ sử dụng dữ liệu chính thức:** Tiền xử lý, indexing và fine-tuning chỉ được dùng dữ liệu do BTC UIT DSC 2026 Task 2 cung cấp.
- **Không dùng external data:** Không crawl dữ liệu từ Internet hoặc thu thập thêm từ các nguồn bên ngoài.
- **Không augmentation từ external sources:** Không đưa external data vào pipeline nhằm mục đích data augmentation.
- **Không can thiệp nhãn thủ công:** Không gán nhãn thủ công (manual labeling) hoặc can thiệp chủ quan vào dữ liệu.

---

## 3. Quy định về mô hình và parameter budget (<4B)

- **Giới hạn tham số:** Tổng số tham số của tất cả các model trong pipeline phải **nhỏ hơn 4 tỷ (< 4.000.000.000)**.
- **Phạm vi tính toán:** Tính tổng tham số của toàn bộ các model:
  1. Dense embedding model
  2. Neural reranker model
  3. Generator base model + LoRA parameters
  4. Mọi model học máy phụ trợ khác có trọng số.
- **Quy định về LoRA / Quantization:** Việc dùng quantization (4-bit, 8-bit), LoRA hay offloading không làm giảm số tham số danh định theo cách tính của BTC. Model gốc từ 4B trở lên không hợp lệ dù đã được lượng tử hóa.
- **Model mã nguồn mở:** Được phép dùng model open-source hoặc có license nghiên cứu/phi thương mại nếu thuộc danh mục đăng ký với BTC.
- **Cấm API bên ngoài:** Nghiêm cấm gọi API mô hình bên ngoài (kể cả API miễn phí). Hệ thống phải tải và chạy trực tiếp trên môi trường thi đấu / môi trường cục bộ của thí sinh.

---

## 4. Công thức đánh giá chính thức (Scoring contract)

**Official metrics (theo thông báo chính thức của BTC):**
- **METEOR:** Metric xếp hạng chính (higher is better).
- **ROUGE-L:** Metric phụ trợ (higher is better).

**Cơ chế đánh giá chi tiết (Provenance: `Scoring-Program-Task-LegalQA.zip` — SHA256: `4fac914203d325445a666c0c566530c962ba95b843e1988e4f37057c47447891`):**
1. **METEOR:**
   - Đánh giá trên whitespace tokens (`str.split()`).
   - Gọi trực tiếp NLTK `meteor_score` với WordNet/OMW.
   - Tính arithmetic macro mean trên toàn bộ tập câu hỏi.
2. **ROUGE-L:**
   - Đánh giá trên Longest Common Subsequence (LCS).
   - Sử dụng package vendored `rouge_score` đi kèm trong archive của BTC, chuẩn hóa ASCII và tính arithmetic macro mean.
