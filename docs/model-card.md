# Model Card — DSC-LegalQA

## Model details
- **Model name:** DSC-LegalQA Grounded Legal Question Answering System
- **Pipeline:** Multi-stage Hybrid RAG (BM25 + Qwen3-Embedding-0.6B + Equal RRF + Qwen3-Reranker-0.6B Prefix20 + P63 HGB Selector + Qwen3.5-2B P70 LoRA).
- **Tác giả:** Đội thi Chef221 — UIT Data Science Challenge 2026.
- **Ngày hoàn thiện:** 18/09/2026.
- **License:** MIT License.

## Intended use
- **Mục đích:** Trả lời câu hỏi pháp luật Việt Nam dựa trên căn cứ văn bản quy phạm pháp luật do BTC UIT DSC 2026 cung cấp.
- **Đối tượng sử dụng:** Nhà nghiên cứu, kỹ sư ML/NLP, thí sinh tham gia cuộc thi khoa học dữ liệu.
- **Lưu ý:** Hệ thống là sản phẩm nghiên cứu trong khuôn khổ cuộc thi, **không có giá trị thay thế tư vấn pháp lý chuyên nghiệp từ luật sư hoặc cơ quan có thẩm quyền**.

## Parameter budget (<4B)
- `Qwen/Qwen3-Embedding-0.6B`: 595.776.512 params (Apache-2.0).
- `Qwen/Qwen3-Reranker-0.6B`: 595.776.512 params (Apache-2.0).
- `Qwen/Qwen3.5-2B`: 2.213.241.664 params (Apache-2.0).
- `P70 LoRA Adapter`: 21.823.488 params (Apache-2.0).
- **Subtotal (Neural + LoRA):** 3.426.618.176 params.
- **P63 Selector (Non-neural):** HistGradientBoostingRegressor tree ensemble (100 trees, 6.030 tree nodes; artifact: 387.527 bytes).
- **Tổng parameter count:** 3.426.618.176 neural (+ 6.030 tree nodes P63), nằm dưới giới hạn < 4B của BTC (headroom > 573M params, ~14,33%).

## Evaluation & metrics
- Official leaderboard (Public Test):
  - **METEOR = 0.486776583**
  - **ROUGE-L = 0.530283618**
