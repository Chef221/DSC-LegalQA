# Quy Định Chính Thức Cuộc Thi UIT Data Science Challenge 2026

Tài liệu này ghi nhận đầy đủ, chuẩn xác các quy định chính thức của Ban tổ chức (BTC) UIT Data Science Challenge 2026 cho **Task 2: Trả lời câu hỏi pháp luật Việt Nam (Legal Question Answering)**.

---

## 1. Đăng Ký Mô Hình Tiền Huấn Luyện (Pretrained Model Registration)

- **Nguyên tắc bắt buộc:** Chỉ các mô hình mã nguồn mở/trọng số mở đã được đăng ký và Ban tổ chức phê duyệt mới được phép sử dụng trong hệ thống thi đấu.
- **Thời hạn đăng ký:** Từ ngày **06/08/2026** đến hết ngày **18/09/2026**.
- Các đội thi có quyền cập nhật, đăng ký bổ sung danh sách mô hình thông qua biểu mẫu Google Form do BTC cung cấp trong thời gian quy định.
- **Giới hạn tham số hệ thống (< 4B):**
  - Tổng số lượng tham số của **TOÀN BỘ HỆ THỐNG** trong mỗi lượt nộp phải **DƯỚI 4 TỶ ( < 4,000,000,000 tham số)**.
  - Tổng tham số được tính bao gồm: mô hình sinh (generator), mô hình nhúng (embedding), mô hình xếp hạng lại (reranker), và bất kỳ mô hình học máy phụ trợ có tham số nào khác.
  - Mô hình chưng cất (distilled model) được phép nếu hệ thống sau chưng cất tuân thủ giới hạn dưới 4 tỷ tham số.
  - **Quy tắc bảo lưu tham số:** Các kỹ thuật tối ưu bộ nhớ như LoRA, QLoRA, Quantization (INT8/INT4/AWQ/GPTQ/GGUF) **KHÔNG ĐƯỢC TÍNH LÀ LÀM GIẢM SỐ THAM SỐ GỐC**. Mô hình có số tham số gốc từ 4B trở lên đều bị loại, bất kể có được lượng tử hóa hay nén bộ nhớ.

---

## 2. Quy Định Thực Thi Mô Hình và Cấm Sử Dụng API

- **Nghiêm cấm tuyệt đối việc sử dụng API** trong quá trình phát triển và vận hành hệ thống, bao gồm cả API thương mại (OpenAI, Anthropic, Google Gemini, v.v.) và API miễn phí/phi lợi nhuận.
- Các mô hình phải là mô hình mã nguồn mở có trọng số công khai mà thí sinh có thể tải về, cài đặt và trực tiếp kiểm soát, vận hành trên hạ tầng tính toán cục bộ/độc lập.
- Giấy phép mô hình: Chấp nhận các giấy phép mã nguồn mở (Apache-2.0, MIT) hoặc giấy phép nghiên cứu/giáo dục/phi thương mại phù hợp với quy định của cuộc thi.

---

## 3. Quy Định Dữ Liệu

- **Chỉ sử dụng dữ liệu chính thức do BTC cung cấp:**
  - `selected-contexts.zip` (Corpus 8.532 ngữ cảnh pháp luật chính thức)
  - `train.json` (7.000 mẫu câu hỏi và câu trả lời huấn luyện)
  - `warmup.json` (Tập dữ liệu khởi động)
  - `public-official.json` (1.000 câu hỏi đánh giá Public Test)
  - `private-official.json` (Tập câu hỏi Private Test bảo mật)
- **Nghiêm cấm:**
  - Không sử dụng dữ liệu pháp luật hoặc tập QA bên ngoài.
  - Không tự gán nhãn thủ công bổ sung ngoài dữ liệu BTC.
  - Không thu thập thêm dữ liệu từ Internet (crawl/scrape).
  - Không sử dụng dữ liệu tổng hợp (synthetic data), kể cả khi được sinh ra từ chính dữ liệu BTC.

---

## 4. Quy Định Nộp Bài (Submission)

- Nền tảng nộp bài chính thức: **Codabench**.
- Định dạng nộp bài: File nén `submission.zip` chứa duy nhất một file `submission.json` mã hóa UTF-8.
- Cấu trúc file: Ánh xạ từ mã câu hỏi sang câu trả lời: `{"question_id": {"answer": "Nội dung câu trả lời..."}}`.
- Đối với Private Test: Tối đa 3 lượt nộp bài mỗi ngày; kết quả cao nhất trên Private Test được sử dụng để xếp hạng chung cuộc.

---

## 5. Tái Lập Thực Nghiệm và Minh Bạch Mã Nguồn (Reproducibility)

- Top 7 đội thi xuất sắc nhất mỗi Task:
  - Phải cung cấp **Docker image + mã nguồn đầy đủ** theo giấy phép mã nguồn mở **MIT** để BTC tái lập kết quả trên Private Test.
  - Phải sẵn sàng cung cấp log huấn luyện và thông số môi trường chi tiết khi có yêu cầu.
  - Phải phản hồi và cung cấp tài liệu xác minh trong vòng **48 giờ** kể từ khi nhận được yêu cầu từ BTC.
  - Đại diện đội thi phải trình bày giải pháp tại Hội thảo Khoa học của cuộc thi.

---

## 6. Đạo Đức AI và Trách Nhiệm Dữ Liệu

- Mỗi bài nộp của giải pháp cần đi kèm:
  - **Data Statement:** Tuyên bố minh bạch về nguồn gốc và cách xử lý dữ liệu.
  - **Model Card:** Bảng mô tả chi tiết thông số, kiến trúc, giới hạn và đạo đức mô hình.
- Tôn trọng bản quyền và trích dẫn đầy đủ các thư viện, mô hình nguồn mở sử dụng.

---

## 7. Quyền Sở Hữu Trí Tuệ (IP)

- Bản quyền giải pháp thuộc về đội thi/thí sinh.
- Thí sinh cấp quyền phi độc quyền, vĩnh viễn, miễn phí cho Trường Đại học Công nghệ Thông tin (ĐHQG-HCM) và các đối tác đồng hành sử dụng giải pháp cho mục đích trưng bày phi thương mại, đào tạo và truyền thông.
- Các đội đạt giải cam kết công bố mã nguồn mở công khai trong vòng 30 ngày sau vòng chung kết.
