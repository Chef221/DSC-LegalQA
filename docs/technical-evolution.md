# Quá Trình Phát Triển Kỹ Thuật (Technical Evolution)

Tài liệu này ghi lại hành trình cải tiến kiến trúc của hệ thống LegalQA, tập trung vào các thành phần kỹ thuật thành công và được giữ lại trong cấu trúc sản xuất cuối cùng (**vNext + P63 + P70**).

---

## 1. Khởi Tạo Truy Xuất Kết Hợp (Hybrid Sparse + Dense Retrieval)
- **Vấn đề ban đầu:** BM25 truyền thống cho độ chính xác cao trên các từ khóa hành chính cụ thể (ví dụ: số hiệu văn bản, từ khóa chuyên ngành), nhưng mất điểm khi câu hỏi người dùng diễn đạt bằng ngôn ngữ đời thường. Ngược lại, Dense Retrieval đơn lẻ dễ bỏ sót các điều khoản có từ khóa định danh cụ thể.
- **Giải pháp kỹ thuật:** Thiết lập kiến trúc truy xuất song song: nhánh từ khóa sử dụng BM25Okapi ($k_1=1.5, b=0.75$) và nhánh ngữ nghĩa sử dụng mô hình nhúng `Qwen/Qwen3-Embedding-0.6B` với chỉ dẫn ngữ nghĩa `Q2`.
- **Kết quả:** Tăng tỷ lệ bao phủ (Recall@100) của tập ứng viên lên đáng kể trước khi bước vào các tầng sau.

## 2. Hợp Nhất Thứ Hạng Tương Hỗ (Equal Reciprocal Rank Fusion - RRF)
- **Vấn đề:** Điểm số thô (raw score) của BM25 và độ tương đồng cosin của Dense Retrieval có thang đo khác biệt, không thể cộng trực tiếp mà không làm méo mó phân phối xác suất.
- **Giải pháp kỹ thuật:** Áp dụng thuật toán Reciprocal Rank Fusion (RRF) với hằng số $k = 10$, trọng số đồng đều ($1:1$) và cơ chế phá vỡ thế hòa (tie-break) 3 cấp tất định.
- **Kết quả:** Tạo ra danh sách Top-100 ứng viên ổn định, không phụ thuộc vào biên độ điểm số thô của từng mô hình.

## 3. Reranker Neural Trên Prefix 20
- **Vấn đề:** Các ứng viên ở đầu danh sách RRF đôi khi có sự tương đồng ngữ nghĩa rộng nhưng không giải quyết chính xác câu hỏi pháp luật cụ thể.
- **Giải pháp kỹ thuật:** Triển khai mô hình `Qwen/Qwen3-Reranker-0.6B` tính hiệu số logit trực tiếp trên token `yes` (ID 9693) và `no` (ID 2152) qua định dạng chat template chính thức. Để tối ưu hóa tài nguyên phần cứng, chỉ xếp hạng lại Prefix 20 ứng viên đầu tiên, đồng thời bảo tồn nguyên vẹn thứ tự các ứng viên tail từ hạng 21 đến 100.
- **Kết quả:** Cải thiện vượt trội chất lượng của Top-5 ứng viên hàng đầu.

## 4. Khủng Hoảng Lựa Chọn Căn Cứ & Sự Ra Đời Của P63 Selector
- **Vấn đề (Evidence Authority Dilemma):** Lấy cố định 2 điều khoản đầu tiên (Canonical First-2) thường xuyên đưa vào các văn bản thừa hoặc thiếu ngữ cảnh quan trọng nếu điều khoản thực sự nằm ở vị trí thứ 3 hoặc thứ 4.
- **Khảo sát P62:** Chứng minh tín hiệu học máy có thể phân biệt được ứng viên tốt hơn hành vi mặc định.
- **Giải pháp P63:** Xây dựng bộ lựa chọn căn cứ tự động `P63` sử dụng `HistGradientBoostingRegressor` huấn luyện trên 215.147 hàng ứng viên từ 5.300 câu hỏi với 37 đặc trưng hoàn toàn không tham chiếu nhãn (reference-free features).
- **Quy tắc can thiệp:** Chỉ khi mô hình dự báo mức cải thiện $\Delta \text{METEOR} \ge \tau$ (với $\tau = 0.100$), hệ thống mới kích hoạt quyền ghi đè (`OVERRIDE`), ngược lại giữ nguyên phương án mặc định (`INCUMBENT`).
- **Kết quả:** Đạt mức tăng **+0.0410 METEOR** trên tập kiểm thử nội bộ.

## 5. Đóng Gói Căn Cứ & Kiểm Soát Trả Lời (Answer Control)
- **Vấn đề:** Mô hình sinh ngôn ngữ dễ bị phân tâm bởi các thông tin ngoài lề hoặc tự suy diễn quy định pháp luật.
- **Giải pháp kỹ thuật:** Định dạng ngữ cảnh pháp lý theo các nhãn `[E1]`, `[E2]` rõ ràng, kết hợp khối hướng dẫn `[ANSWER_CONTROL]` nghiêm ngặt yêu cầu câu trả lời chỉ dựa vào các căn cứ được cung cấp.

## 6. Thích Ứng Generator P70
- **Vấn đề:** Mô hình nền tảng `Qwen/Qwen3.5-2B` cần làm quen với phong cách viết văn xuôi pháp luật chuẩn mực của Việt Nam theo định dạng chấm điểm của cuộc thi.
- **Giải pháp kỹ thuật:** Huấn luyện Continued-LoRA (P70) trên 4.247 mẫu huấn luyện từ các Fold 1–4, đảm bảo tách biệt tuyệt đối không rò rỉ dữ liệu kiểm thử, áp dụng kỹ thuật mask prompt tokens khỏi tính toán mất mát.
- **Trạng thái khoa học:** P70 là cấu hình generator được triển khai trong hệ thống nộp bài cuối cùng. Hệ thống hoàn chỉnh sử dụng P70 đạt METEOR 0.486776583 và ROUGE-L 0.530283618 trên bảng xếp hạng chính thức. Đánh giá nội bộ dành riêng cho P70 tại thời điểm đóng hệ thống có trạng thái `INCONCLUSIVE`; quyết định triển khai được thực hiện theo `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`.

## 7. Bộ Lọc Lặp Vòng Chu Kỳ Sản Xuất (Token Suffix Loop Sanitizer)
- **Vấn đề:** Hiện tượng sinh lặp vô hạn ở đuôi câu trả lời (suffix repetition loop) trong giải mã tham lam là nguyên nhân phổ biến làm sụt giảm nghiêm trọng điểm METEOR và ROUGE-L.
- **Giải pháp kỹ thuật:** Tích hợp bộ lọc lặp vòng sản xuất `token_suffix_loop_sanitizer.py` (SHA256: `1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527`, 26 KB) rà soát các chu kỳ token lặp và cắt tỉa chính xác tại điểm bắt đầu vòng lặp.

## 8. Kiến Trúc Hoàn Chỉnh (vNext + P63 + P70)
- Tích hợp liên hoàn toàn bộ 7 cải tiến trên tạo nên hệ thống cuối cùng, đạt thành tích cao nhất của toàn dự án trên Leaderboard chính thức.
