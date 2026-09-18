# Quá Trình Phát Triển Kỹ Thuật (Technical Evolution)

Tài liệu này ghi lại hành trình hoàn thiện kiến trúc của hệ thống LegalQA, tập trung vào các thành phần kỹ thuật thành công và được lưu giữ trong cấu trúc sản xuất cuối cùng (**vNext + P63 + P70**). Mọi nhận định kỹ thuật đều dựa trên mục tiêu kiến trúc hoặc các bằng chứng thực nghiệm đã được kiểm toán.

---

## 1. Khởi Tạo Truy Xuất Kết Hợp (Hybrid Sparse + Dense Retrieval)
- **Vấn đề ban đầu:** BM25 truyền thống đạt độ chính xác cao trên các từ khóa hành chính cụ thể (như số hiệu văn bản, thuật ngữ chuyên ngành định danh), nhưng độ nhạy giảm khi câu hỏi người dùng diễn đạt bằng ngôn ngữ đời thường. Ngược lại, Dense Retrieval đơn lẻ có xu hướng phản ánh ngữ nghĩa khái quát nhưng dễ bỏ sót các điều khoản có từ khóa định danh cụ thể.
- **Giải pháp kỹ thuật:** Thiết lập kiến trúc truy xuất song song: nhánh từ khóa sử dụng BM25Okapi ($k_1=1.5, b=0.75$) và nhánh ngữ nghĩa sử dụng mô hình nhúng `Qwen/Qwen3-Embedding-0.6B` với chỉ dẫn ngữ nghĩa `Q2`.
- **Mục tiêu kiến trúc:** Mục tiêu của kiến trúc hybrid là tăng độ bao phủ ứng viên bằng cách kết hợp tín hiệu lexical và semantic trước bước hợp nhất thứ hạng.

## 2. Hợp Nhất Thứ Hạng Tương Hỗ (Equal Reciprocal Rank Fusion - RRF)
- **Vấn đề:** Điểm số thô (raw score) của BM25 và độ tương đồng cosin của Dense Retrieval có thang đo khác biệt, không thể cộng trực tiếp mà không làm lệch phân phối điểm số.
- **Giải pháp kỹ thuật:** Áp dụng thuật toán Reciprocal Rank Fusion (RRF) với hằng số $k = 10$, trọng số đồng đều ($1:1$) và cơ chế phá vỡ thế hòa (tie-break) tất định dựa trên chunk ID.
- **Mục tiêu kiến trúc:** Tạo ra danh sách Top-100 ứng viên chuẩn hóa, không phụ thuộc vào biên độ điểm số thô của từng mô hình riêng lẻ.

## 3. Reranker Neural Trên Prefix 20
- **Vấn đề:** Các ứng viên ở đầu danh sách RRF đôi khi có sự tương đồng ngữ nghĩa rộng nhưng không giải quyết trực tiếp câu hỏi pháp luật cụ thể.
- **Giải pháp kỹ thuật:** Triển khai mô hình `Qwen/Qwen3-Reranker-0.6B` tính hiệu số logit trực tiếp trên token `yes` (ID 9693) và `no` (ID 2152) qua định dạng chat template chính thức. Nhằm cân bằng giữa độ trễ và tài nguyên tính toán, hệ thống chỉ sắp xếp lại Prefix 20 ứng viên đầu tiên, đồng thời bảo tồn nguyên vẹn thứ tự các ứng viên tail từ hạng 21 đến 100.
- **Mục tiêu kiến trúc:** Reranker được dùng để sắp xếp lại 20 ứng viên đầu theo mức độ phù hợp với câu hỏi, trong khi giữ nguyên phần tail.

## 4. Lựa Chọn Căn Cứ Học Máy (P63 Learned Evidence Selector)
- **Vấn đề (Evidence Authority Dilemma):** Lấy cố định 2 điều khoản đầu tiên (Canonical First-2) thường đưa vào các điều khoản thừa hoặc bỏ sót ngữ cảnh nếu điều khoản then chốt nằm ở vị trí thứ 3 hoặc thứ 4.
- **Khảo sát P62:** Đánh giá thực nghiệm nội bộ cho thấy việc can thiệp chọn tập căn cứ có tín hiệu cải thiện rõ ràng:
  - Incumbent METEOR (cố định First-2): **0.4453189045593393**
  - Selected METEOR (theo bộ chọn): **0.4862964382201936**
  - Mức chênh lệch ($\Delta$): **+0.04097753366085427** (phân loại kiểm định: `PASS_STRONG_POSITIVE`).
- **Triển khai P63:** Huấn luyện mô hình hồi quy `HistGradientBoostingRegressor` trên 215.147 hàng ứng viên với 37 đặc trưng hoàn toàn không tham chiếu nhãn (reference-free features).
- **Quy tắc can thiệp:** Chỉ khi mô hình dự báo mức cải thiện $\Delta \ge 	au$ (với $	au = 0.100$), hệ thống mới kích hoạt quyền ghi đè (`OVERRIDE`), ngược lại giữ nguyên phương án mặc định (`INCUMBENT`).

## 5. Đóng Gói Căn Cứ & Kiểm Soát Trả Lời (Answer Control)
- **Vấn đề:** Mô hình ngôn ngữ dễ sinh nội dung ngoài phạm vi căn cứ được cung cấp hoặc tự suy diễn quy định pháp luật.
- **Giải pháp kỹ thuật:** Đóng gói căn cứ pháp lý theo các nhãn `[E1]`, `[E2]` có cấu trúc rõ ràng, kết hợp khối hướng dẫn `[ANSWER_CONTROL]` quy định câu trả lời chỉ được xây dựng trên các thông tin có trong ngữ cảnh.

## 6. Thích Ứng Generator P70
- **Vấn đề:** Mô hình nền tảng `Qwen/Qwen3.5-2B` cần thích ứng với phong cách hành văn văn xuôi pháp luật chuẩn mực của Việt Nam theo định dạng đánh giá của cuộc thi.
- **Giải pháp kỹ thuật:** Huấn luyện Continued-LoRA (P70) trên 4.247 mẫu huấn luyện từ Folds 1–4, đảm bảo cô lập hoàn toàn không rò rỉ dữ liệu kiểm thử, áp dụng kỹ thuật mask prompt tokens khỏi tính toán loss.
- **Trạng thái khoa học:** P70 là cấu hình generator được triển khai trong hệ thống nộp bài cuối cùng. Hệ thống hoàn chỉnh sử dụng P70 đạt METEOR **0.486776583** và ROUGE-L **0.530283618** trên bảng xếp hạng chính thức. Đánh giá nội bộ dành riêng cho P70 tại thời điểm đóng hệ thống có trạng thái `INCONCLUSIVE`; quyết định triển khai được thực hiện theo `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`.

## 7. Bộ Lọc Lặp Vòng Chu Kỳ Sản Xuất (Token Suffix Loop Sanitizer)
- **Vấn đề:** Hiện tượng sinh lặp vòng vô hạn ở đuôi câu trả lời (suffix repetition loop) trong giải mã tham lam làm suy giảm độ chính xác và chất lượng văn bản.
- **Giải pháp kỹ thuật:** Tích hợp bộ lọc lặp vòng sản xuất `token_suffix_loop_sanitizer.py` (SHA256: `1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527`, 26.461 bytes) rà soát các chu kỳ token lặp và cắt tỉa chính xác tại điểm bắt đầu vòng lặp.

## 8. Kiến Trúc Hoàn Chỉnh (vNext + P63 + P70)
- Tích hợp 7 thành phần kỹ thuật trên tạo thành pipeline sản xuất hoàn chỉnh.
- Kết quả chính thức trên Codabench: **METEOR = 0.486776583**, **ROUGE-L = 0.530283618**.
