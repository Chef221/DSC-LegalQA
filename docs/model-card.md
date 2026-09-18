# Model Card — DSC-LegalQA System

## Model Details
- **Tên mô hình:** DSC-LegalQA Grounded Legal Question Answering System
- **Kiến trúc:** Multi-stage Agentic Hybrid RAG (Sparse BM25 + Dense Qwen3-Embedding + Equal RRF + Neural Qwen3-Reranker + P63 HGB Selector + Qwen3.5-2B P70 LoRA).
- **Tổ chức phát triển:** Đội thi Chef221 — UIT Data Science Challenge 2026.
- **Ngày hoàn thiện:** 18/09/2026.
- **Giấy phép:** MIT License.

## Intended Use
- **Mục đích thiết kế:** Trả lời câu hỏi pháp luật Việt Nam dựa trên căn cứ văn bản quy phạm pháp luật chính thức do Ban tổ chức cuộc thi UIT DSC 2026 cung cấp.
- **Đối tượng sử dụng:** Nhà nghiên cứu, kỹ sư AI, người tham gia cuộc thi khoa học dữ liệu.
- **Cảnh báo quan trọng:** Hệ thống là sản phẩm nghiên cứu phục vụ cuộc thi khoa học dữ liệu, **KHÔNG CÓ GIÁ TRỊ THAY THẾ TƯ VẤN PHÁP LÝ CHUYÊN NGHIỆP TỪ LUẬT SƯ HOẶC CƠ QUAN NHÀ NƯỚC CÓ THẨM QUYỀN**.

## Parameter Budget Audit (< 4B)
- Qwen/Qwen3-Embedding-0.6B: 595.776.512 tham số (Apache-2.0).
- Qwen/Qwen3-Reranker-0.6B: 595.776.512 tham số (Apache-2.0).
- Qwen/Qwen3.5-2B: 2.213.241.664 tham số (Apache-2.0).
- P70 LoRA Adapter: 21.823.488 tham số (Apache-2.0).
- **Subtotal (Neural + LoRA):** 3.426.618.176 tham số.
- **P63 Selector (Non-neural):** HistGradientBoostingRegressor tree ensemble (100 cây, 6.030 nút cây; file: 387.527 bytes).
- **Tổng tham số toàn hệ thống:** 3.426.618.176 neural (+ 6.030 nút cây P63), tuyệt đối tuân thủ trần < 4.000.000.000 (headroom > 573 triệu tham số, tương đương 14,33%).

## Metrics & Performance
- Official Competition Leaderboard:
  - **METEOR = 0.486776583**
  - **ROUGE-L = 0.530283618**
