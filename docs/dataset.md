# Quy Chuẩn Dữ Liệu và Nhiệm Vụ (Task & Dataset Specification)

## 1. Định Nghĩa Bài Toán (Task Definition)

- **Cuộc thi:** UIT Data Science Challenge 2026.
- **Bài toán:** Task 2 — Trả lời câu hỏi pháp luật Việt Nam (Legal Question Answering - LegalQA).
- **Đầu vào (Input):** Một câu hỏi pháp lý bằng tiếng Việt tự nhiên (ví dụ về tranh chấp lao động, bảo hiểm xã hội, tố tụng dân sự, giao thông, hành chính).
- **Đầu ra (Output):** Câu trả lời dạng văn xuôi bằng tiếng Việt chuẩn mực, có căn cứ và trích dẫn điều luật chính xác dựa trên ngữ cảnh pháp lý được cung cấp.

---

## 2. Cấu Trúc Tập Dữ Liệu Chính Thức Do Ban Tổ Chức Cung Cấp

Ban tổ chức cung cấp 5 tệp dữ liệu chính thống:

| Tên File | Quy Mô | Cấu Trúc Schema | Mô Tả Vai Trò |
|---|---:|---|---|
| `selected-contexts.zip` | 8.532 file | `{id, name, link, passage}` | Toàn bộ kho ngữ cảnh văn bản pháp luật chính thức |
| `train.json` | 7.000 mẫu | `qid -> {question, answer}` | Tập dữ liệu câu hỏi và câu trả lời tham chiếu để huấn luyện |
| `warmup.json` | 50 mẫu | `qid -> {question, answer}` | Tập dữ liệu mẫu giai đoạn khởi động |
| `public-official.json` | 1.000 câu | `qid -> {question}` | Tập đánh giá công khai giai đoạn Public Test |
| `private-official.json` | Bí mật | `qid -> {question}` | Tập đánh giá bảo mật chung cuộc Private Test |

### Chi Tiết Cấu Trúc Ngữ Cảnh Pháp Luật (`context_*.json`)

```json
{
  "id": 1234,
  "name": "nghi-dinh-145-2020-nd-cp-huong-dan-bo-luat-lao-dong",
  "link": "https://vanbanphapluat.example.gov.vn/...",
  "passage": "Điều 98. Tiền lương làm thêm giờ, làm việc vào ban đêm..."
}
```

- `id`: Định danh ngữ cảnh (số nguyên).
- `name`: Tên slug của văn bản pháp lý.
- `link`: Nguồn gốc xuất xứ của văn bản (chỉ dùng định danh, nghiêm cấm crawl dữ liệu).
- `passage`: Nội dung điều luật/văn bản pháp luật (chứa tiêu đề Điều, Khoản, Điểm và nội dung chi tiết).

---

## 3. Chính Sách Bảo Mật và Không Đưa Dữ Liệu Thô Lên Git (Fail-Closed Data Policy)

Nhằm tuân thủ bản quyền dữ liệu và thể lệ cuộc thi:
1. **Kho lưu trữ này KHÔNG chứa bất kỳ tệp dữ liệu thô nào của cuộc thi.**
2. Các tệp `train.json`, `warmup.json`, `public-official.json`, `selected-contexts.zip` đã được đưa vào `.gitignore`.
3. Người sử dụng muốn tái lập kết quả cần chủ động đặt các tệp dữ liệu hợp lệ vào thư mục `data/` theo hướng dẫn tại [data/README.md](../data/README.md).
