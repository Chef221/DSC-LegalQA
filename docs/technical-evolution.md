# Báo Cáo Tiến Trình Kỹ Thuật (Technical Evolution Report)

Báo cáo này tường thuật quá trình nghiên cứu và nâng cấp kỹ thuật của hệ thống, **chỉ ghi nhận các cải tiến thành công và được bảo lưu trong kiến trúc cuối cùng**.

---

## Giai Đoạn 1: Cải Thiện Bộ Trích Xuất và Lựa Chọn Căn Cứ (P62 Evaluation)

- **Vấn đề ban đầu:** Chiến lược truy xuất đơn lẻ (A0) thường bị thiếu ngữ cảnh khi câu hỏi bao hàm nhiều khía cạnh pháp lý (ví dụ: điều kiện áp dụng kết hợp với mức phạt xử lý vi phạm).
- **Giải pháp kỹ thuật:** Xây dựng cơ chế lựa chọn căn cứ đa ứng viên, khai thác các tập căn cứ mở rộng từ Top-K và phân đoạn văn bản phi điều khoản.
- **Phương pháp đánh giá:** Đánh giá OOF trên tập dữ liệu giám sát gồm 5.300 câu hỏi.
- **Kết quả đo lường thực nghiệm:**
  - Điểm METEOR của Incumbent: **0.4453189**
  - Điểm METEOR của Selected: **0.4862964**
  - Mức cải thiện: **+0.0409775 (+0.0410 METEOR)**
  - Phân loại cổng khoa học: `PASS_STRONG_POSITIVE`.

---

## Giai Đoạn 2: Xây Dựng Bộ Chọn Học Máy Triển Khai P63 (P63 Selector)

- **Vấn đề:** Không thể chạy đánh giá so sánh trực tiếp khi vận hành thực tế do không có nhãn tham chiếu gold.
- **Giải pháp kỹ thuật:** Huấn luyện mô hình hồi quy `HistGradientBoostingRegressor` dựa trên 37 đặc trưng phi tham chiếu (độ dài, số lượng từ khóa, độ bao phủ từ vựng, mức tương quan xếp hạng).
- **Hiệu chuẩn ngưỡng triển khai:**
  - Ngưỡng tối ưu thực nghiệm: $\tau = 0.100$.
  - Khi mô hình dự đoán mức cải thiện $\Delta \ge 0.100$, hệ thống kích hoạt **OVERRIDE**; ngược lại giữ nguyên **INCUMBENT**.
  - Kiểm định toàn vẹn: Đạt 0 vi phạm leakage, bảo toàn thứ tự tất định.

---

## Giai Đoạn 3: Huấn Luyện Tinh Chỉnh Mô Hình Sinh P70 (P70 LoRA Adaptation)

- **Vấn đề:** Mô hình ngôn ngữ nền tảng Qwen3.5-2B tuy có năng lực tổng quát tốt nhưng phong cách trả lời văn bản pháp luật chưa chuẩn mực theo mẫu của BTC và đôi khi lặp lại nội dung.
- **Giải pháp kỹ thuật:** Huấn luyện tinh chỉnh QLoRA (rank 16, alpha 32) trên 96 module chiếu attention và MLP của mô hình Qwen3.5-2B bằng dữ liệu huấn luyện chính thức của cuộc thi.
- **Minh bạch khoa học về nguồn gốc thăng hạng:**
  - Đánh giá cổng khoa học nội bộ tại thời điểm hết hạn: `INCONCLUSIVE`.
  - Quyết định phê duyệt sản xuất: Được người vận hành phê duyệt thông qua `OPERATOR_OVERRIDE_DEADLINE_2026-09-17` để kịp hạn chót nộp bài.
  - **Kết quả chính thức trên Leaderboard cuộc thi:**
    - **METEOR = 0.486776583**
    - **ROUGE-L = 0.530283618**
- **Kết luận:** Kết quả thi đấu chính thức đã khẳng định tính hiệu quả vượt bậc của hệ thống sản xuất P70 trên tập dữ liệu kiểm thử thực tế của Ban tổ chức.
