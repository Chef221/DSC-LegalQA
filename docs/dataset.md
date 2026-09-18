# Dữ Liệu Cuộc Thi UIT DSC 2026 Task 2

Tài liệu này mô tả cấu trúc dữ liệu theo đặc tả chính thức của Ban tổ chức và các đặc điểm quan sát được trong snapshot dữ liệu dự án.

---

## 1. Đặc Tả Dữ Liệu Chính Thức Của Ban Tổ Chức (Official Specification)

Theo thông báo chính thức của Ban tổ chức UIT Data Science Challenge 2026 Task 2, bộ dữ liệu bao gồm:

1. **`selected-contexts.zip`:** Tệp nén chứa các ngữ cảnh pháp luật chính thức.
   - Mỗi ngữ cảnh được lưu trong một tệp JSON với cấu trúc:
     ```json
     {
       "id": 1,
       "name": "ten_van_ban_slug",
       "link": "https://thuvienphapluat.vn/...",
       "passage": "Toàn văn hoặc trích đoạn điều khoản pháp luật..."
     }
     ```
   - Trường `id`: Mã định danh số của ngữ cảnh.
   - Trường `name`: Tên định danh (slug) của văn bản.
   - Trường `link`: Nguồn gốc xuất xứ (provenance), chỉ dùng đối soát, cấm thu thập (crawl).
   - Trường `passage`: Nội dung văn bản pháp luật chứa tiêu đề Điều, Khoản, Điểm.

2. **Các tệp câu hỏi & nhãn:**
   - `train.json`: Tệp dữ liệu câu hỏi và câu trả lời tham chiếu có nhãn.
   - `warmup.json`: Tệp mẫu dữ liệu giai đoạn khởi động (warm-up).
   - `public-official.json`: Tập câu hỏi công khai (Public Test) chỉ gồm câu hỏi, không có nhãn câu trả lời hoặc căn cứ.
   - `private-official.json`: Tập dữ liệu kiểm tra bảo mật (Private Test) dùng để xếp hạng chung cuộc.

---

## 2. Thống Kê Quan Sát Được Trong Snapshot Dữ Liệu Dự Án

*(Observed in the official dataset snapshot used by this project)*

Trong quá trình tiếp nhận và kiểm toán dữ liệu snapshot từ Ban tổ chức, dự án ghi nhận các số lượng thực tế sau:

- **Số lượng ngữ cảnh (Contexts):** 8.532 tệp context trong `selected-contexts.zip`.
  - Canonical archive hash SHA256: `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`.
  - Có 20 passage rỗng và 1.125 context thiếu name; hệ thống giữ nguyên vẹn không tự ý loại bỏ hay phát minh thêm dữ liệu.
- **Tập huấn luyện (`train.json`):** 7.000 bản ghi câu hỏi và câu trả lời.
- **Tập khởi động (`warmup.json`):** 50 bản ghi.
- **Tập kiểm tra công khai (`public-official.json`):** 1.000 bản ghi câu hỏi pháp luật.

> **Chính Sách Bảo Mật Dữ Liệu:** Toàn bộ các tệp dữ liệu thô (`*.json`, `*.zip`) và dữ liệu giải nén của cuộc thi đều được cấu hình trong `.gitignore` và tuyệt đối **không được đẩy lên GitHub** nhằm tuân thủ quy chế bảo mật của Ban tổ chức.
