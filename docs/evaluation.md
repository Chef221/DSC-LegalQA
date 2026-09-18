# Báo Cáo Đánh Giá và Điểm Số Chính Thức (Evaluation Report)

## 1. Kết Quả Chính Thức Trên Bảng Xếp Hạng Cuộc Thi (Official Leaderboard)

Dưới đây là kết quả thi đấu chính thức của hệ thống **P70** được ghi nhận trên nền tảng chấm thi Codabench của Ban tổ chức UIT Data Science Challenge 2026 Task 2:

| Chỉ Số Đánh Giá | Điểm Số Chính Thức | Vai Trò Chỉ Số | Phương Pháp Tính |
|---|---:|:---:|---|
| **METEOR** | **0.486776583** | **Chỉ số xếp hạng chính (Primary)** | NLTK Meteor trên chuỗi từ phân tách khoảng trắng |
| **ROUGE-L** | **0.530283618** | **Chỉ số phụ (Secondary)** | ROUGE-L F1 (LCS) mã hóa ASCII |

Mã SHA256 của file nộp bài chính thức sinh ra kết quả trên:
`ac6b796794cf3c422889620753743fa248d9ada8dd18d681f36d2e38784e1056`

---

## 2. Kết Quả Nghiên Cứu Nội Bộ (Internal Held-out Benchmarks)

Bảng dưới đây ghi nhận kết quả đánh giá thực nghiệm nội bộ trên tập tách biệt 500 câu hỏi (Held-out 500):

| Cấu Hình Thử Nghiệm | METEOR Nội Bộ | ROUGE-L Nội Bộ | Ghi Chú Kỹ Thuật |
|---|---:|---:|---|
| BM25 Baseline | 0.3812 | 0.4120 | Truy xuất từ khóa đơn lẻ |
| Hybrid RRF (BM25 + Qwen3-Embed) | 0.4150 | 0.4560 | Hợp nhất đối xứng k=10 |
| Hybrid + Neural Reranker (Prefix 20) | 0.4453 | 0.4890 | Rerank với Qwen3-Reranker |
| **P63 Learned Selector (tau=0.100)** | **0.4863** | **0.5280** | **Lựa chọn căn cứ học máy (+0.0410 METEOR)** |
