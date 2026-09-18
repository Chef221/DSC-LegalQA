# Dataset UIT DSC 2026 Task 2

Tài liệu này mô tả cấu trúc dữ liệu theo đặc tả chính thức của BTC và thống kê quan sát từ snapshot dữ liệu của dự án.

---

## 1. Đặc tả dữ liệu chính thức từ BTC (Official specification)

Dữ liệu do BTC UIT Data Science Challenge 2026 Task 2 công bố bao gồm:

1. **`selected-contexts.zip`:** File nén chứa context corpus chính thức.
   - Mỗi context lưu trong một file JSON với schema:
     ```json
     {
       "id": 1,
       "name": "ten_van_ban_slug",
       "link": "https://thuvienphapluat.vn/...",
       "passage": "Toàn văn hoặc trích đoạn điều khoản pháp luật..."
     }
     ```
   - Trường `id`: ID gốc dạng số của context.
   - Trường `name`: Tên định danh (slug) của văn bản.
   - Trường `link`: Provenance URL đối soát nguồn, không crawl.
   - Trường `passage`: Nội dung văn bản chứa cấu trúc Điều, Khoản, Điểm.

2. **Files câu hỏi & nhãn:**
   - `train.json`: 7.000 mẫu câu hỏi kèm answer tham chiếu.
   - `warmup.json`: File mẫu warm-up (50 mẫu).
   - `public-official.json`: 1.000 câu hỏi Public Test (question-only, không có nhãn answer hay evidence).
   - `private-official.json`: Private Test dùng để xếp hạng chung cuộc.

---

## 2. Thống kê quan sát từ snapshot dữ liệu

*(Observed in the official dataset snapshot used by this project)*

Kiểm toán dữ liệu snapshot ghi nhận:

- **Context corpus (`selected-contexts.zip`):** 8.532 files JSON.
  - Canonical archive SHA256: `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`.
  - Có 20 passage rỗng và 1.125 context thiếu name; hệ thống giữ nguyên hiện trạng, không drop và không bịa dữ liệu.
- **Train set (`train.json`):** 7.000 records có answer label.
- **Warm-up (`warmup.json`):** 50 records.
- **Public Test (`public-official.json`):** 1.000 câu hỏi.

> **Chính sách bảo mật dữ liệu:** Toàn bộ dữ liệu thô (`*.json`, `*.zip`) và thư mục giải nén đều được cấu hình trong `.gitignore` và không commit lên GitHub theo quy định của BTC.
