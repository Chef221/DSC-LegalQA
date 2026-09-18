# Kiến Trúc Hệ Thống Chi Tiết (DSC-LegalQA System Architecture)

Hệ thống **DSC-LegalQA** triển khai kiến trúc lai đa giai đoạn (Multi-stage Hybrid Retrieval-Augmented Generation) tối ưu hóa đặc thù cho pháp luật Việt Nam.

![Kiến Trúc Hệ Thống](diagrams/architecture.png)

---

## Các Khối Thành Phần Chính

### 1. Chuẩn Hóa và Tiền Xử Lý (Text Normalization)
- Chuẩn hóa văn bản đầu vào theo chuẩn Unicode NFC.
- Bảo toàn dấu tiếng Việt, dấu câu pháp lý, số ký hiệu văn bản, các từ khóa phủ định và từ chỉ điều kiện ngoại lệ (`không`, `chưa`, `trừ`, `chỉ khi`, `trong trường hợp`).
- Phân đoạn văn bản pháp luật thành các khối điều luật (1 Điều = 1 chunk).

### 2. Truy Xuất Kép Lai (Hybrid Sparse & Dense Retrieval)
- **Nhánh Sparse (BM25Okapi):** Nắm bắt chính xác các từ khóa pháp lý đặc trưng, số hiệu nghị định, điều luật cụ thể.
- **Nhánh Dense Vector (`Qwen3-Embedding-0.6B`):** Nắm bắt ngữ nghĩa câu hỏi, các khái niệm đồng nghĩa và ngữ cảnh pháp lý mở rộng.
- Cả hai nhánh độc lập truy xuất Top 100 ứng viên từ kho 8.532 ngữ cảnh.

### 3. Hợp Nhất Hạng Đối Xứng (Equal Reciprocal Rank Fusion - RRF)
- Sử dụng công thức RRF đối xứng với hằng số $k=10$:
  $$RRF\_Score(d) = \frac{1}{10 + rank_{BM25}(d)} + \frac{1}{10 + rank_{Dense}(d)}$$
- Tạo ra danh sách Top 100 ứng viên hợp nhất có tính đa dạng cao và triệt tiêu sai số thiên lệch của từng nhánh.

### 4. Xếp Hạng Lại Nơ-ron (Neural Cross-Encoder Reranking)
- Sử dụng mô hình `Qwen3-Reranker-0.6B` đánh giá tương quan sâu giữa câu hỏi và ứng viên.
- Rerank trên **Prefix 20** ứng viên hàng đầu; bảo toàn thứ tự nguyên vẹn của phần đuôi ứng viên (ranks 21-100) để giữ tính đa dạng.

### 5. Bộ Chọn Căn Cứ Học Máy P63 (P63 Learned Evidence Selector)
- Trích xuất bộ 37 đặc trưng phi tham chiếu (reference-free features) mô tả cấu trúc cặp, độ tương đồng từ vựng, mật độ từ khóa pháp lý và vị trí thứ hạng.
- Sử dụng mô hình `HistGradientBoostingRegressor` dự đoán mức gia tăng điểm METEOR ($\Delta$).
- Quy tắc định tuyến:
  - Nếu $\Delta \ge 0.100$: Áp dụng quyết định **OVERRIDE** (kết hợp cặp điều luật hoặc fallback mở rộng).
  - Nếu $\Delta < 0.100$: Giữ nguyên quyết định **INCUMBENT (A0)** (điều luật đơn lẻ xếp đầu).

### 6. Đóng Gói Căn Cứ và Kiểm Soát Sinh (Evidence Packing & Answer-Control)
- Đóng gói căn cứ pháp lý theo mẫu chuẩn `[E1] ... [E2] ...`.
- Tiền tố câu nhắc với khung kiểm soát `[ANSWER_CONTROL]` nghiêm ngặt yêu cầu mô hình chỉ trả lời dựa trên căn cứ được cung cấp, không suy đoán.

### 7. Sinh Câu Trả Lời và Bộ Lọc Vòng Lặp (Grounded Generation & Duplicate Guard)
- Mô hình nền tảng: `Qwen/Qwen3.5-2B` kết hợp với trọng số tinh chỉnh **P70 LoRA** (96 module LoRA).
- Cơ chế giải mã: Greedy decoding tất định (`do_sample=false`, `num_beams=1`, `max_new_tokens=1536`).
- **Production Duplicate Guard:** Thuật toán phát hiện chu kỳ tuần hoàn ở đuôi chuỗi token để cắt bỏ triệt để các vòng lặp lặp lại, giữ lại duy nhất một chu kỳ hợp lệ.
