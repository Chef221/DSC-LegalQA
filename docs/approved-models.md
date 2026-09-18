# Danh mục model được BTC phê duyệt (Approved models)

> **Lưu ý:** Tài liệu này tổng hợp danh mục các model tiền huấn luyện và công cụ NLP được BTC UIT Data Science Challenge 2026 công nhận và cho phép đăng ký/sử dụng cho Task 2. Việc một model xuất hiện trong danh mục này biểu thị tính hợp lệ về nguồn gốc; khi triển khai, toàn bộ hệ thống phải đáp ứng điều kiện: **tổng số tham số của tất cả model trong pipeline phải dưới 4 tỷ (< 4.000.000.000)**.

---

## 1. Model sử dụng trong production pipeline (vNext + P63 + P70)

Bảng dưới đây liệt kê các model cấu thành nên pipeline sản xuất **FROZEN_P3_G2_VNEXT_P63_P70** đạt METEOR = **0.486776583** và ROUGE-L = **0.530283618** trên official leaderboard:

| Role | Model / HF ID | Pinned revision | Parameter count | License | Trạng thái duyệt | Nguồn |
|---|---|---|---:|---|:---:|---|
| **Dense Retriever** | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| **Neural Reranker** | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| **Generator Base** | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2.213.241.664 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **Generator LoRA** | `P70 Adapter` (Continued-LoRA) | `193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c` | 21.823.488 | Apache-2.0 / Team | **ĐÃ DUYỆT** | Train trên data BTC |

### Parameter budget toàn hệ thống

$$\sum \text{Params} = 595.776.512 + 595.776.512 + 2.213.241.664 + 21.823.488 = 3.426.618.176 \approx 3,43\text{B}$$

- **Giới hạn BTC:** $< 4.000.000.000$ params.
- **Headroom:** $+573.381.824$ params (margin $14,33\%$).
- **Trạng thái:** **PASS**.
- *Ghi chú về P63 Evidence Selector:* P63 selector là non-neural tree ensemble (`HistGradientBoostingRegressor`) gồm 100 cây và 6.030 tree nodes trên 37 features (dung lượng file: 387.527 bytes). Cấu trúc nút cây không quy đổi tương đương trọng số tensor neural. Ngay cả khi cộng tượng trưng toàn bộ 6.030 tree nodes, tổng độ phức tạp của pipeline là 3.426.624.206 tham số, vẫn nằm dưới giới hạn 4B của BTC.

---

## 2. Toàn bộ danh mục model & tài nguyên NLP được BTC phê duyệt

> **Nguồn danh mục:** Tổng hợp các dòng model và tài nguyên ngôn ngữ từ thông báo của BTC và danh mục đăng ký hợp lệ.

### Nhóm A: Pretrained / Instruction models (< 4B)

| STT | Model / Resource | Phân loại | License | Nguồn tham chiếu |
|---:|---|---|---|---|
| 1 | `Qwen/Qwen3.5-2B` | Generator (Khuyên dùng) | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B) |
| 2 | `Qwen/Qwen3-1.7B` | Generator | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen3-1.7B) |
| 3 | `Qwen/Qwen2.5-1.5B-Instruct` | Generator | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| 4 | `Qwen/Qwen2.5-0.5B-Instruct` | Generator | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) |
| 5 | `ntphuc149/ViLegalQwen3-1.7B-Base` | Generator (Legal Pretrained) | Apache-2.0 | [HuggingFace](https://huggingface.co/ntphuc149/ViLegalQwen3-1.7B-Base) |
| 6 | `ntphuc149/ViLegalQwen2.5-1.5B-Base` | Generator (Legal Pretrained) | Apache-2.0 | [HuggingFace](https://huggingface.co/ntphuc149/ViLegalQwen2.5-1.5B-Base) |
| 7 | `AITeamVN/Vi-Qwen2-1.5B-RAG` | Generator (Vietnamese RAG) | Apache-2.0 | [HuggingFace](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG) |
| 8 | `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | Generator (GRPO Finetuned) | Apache-2.0 | [HuggingFace](https://huggingface.co/thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2) |
| 9 | `google/gemma-3-1b-it` | Generator | Gemma Terms of Use | [HuggingFace](https://huggingface.co/google/gemma-3-1b-it) |
| 10 | `google/gemma-2-2b-it` | Generator | Gemma Terms of Use | [HuggingFace](https://huggingface.co/google/gemma-2-2b-it) |
| 11 | `VietAI/vit5-large` | Generator (Seq2Seq) | Apache-2.0 | [HuggingFace](https://huggingface.co/VietAI/vit5-large) |
| 12 | `VietAI/vit5-base` | Generator (Seq2Seq) | Apache-2.0 | [HuggingFace](https://huggingface.co/VietAI/vit5-base) |
| 13 | `VietAI/gpt-neo-1.3B-vietnamese-news` | Generator | MIT | [HuggingFace](https://huggingface.co/VietAI/gpt-neo-1.3B-vietnamese-news) |
| 14 | `VoVanPhuc/gpt2-vietnamese-small` | Generator | MIT | [HuggingFace](https://huggingface.co/VoVanPhuc/gpt2-vietnamese-small) |
| 15 | `vilm/vinallama-2.7b` | Generator | Apache-2.0 | [HuggingFace](https://huggingface.co/vilm/vinallama-2.7b) |
| 16 | `vilm/vinallama-2.7b-chat` | Generator | Apache-2.0 | [HuggingFace](https://huggingface.co/vilm/vinallama-2.7b-chat) |
| 17 | `bkai-foundation-models/vietnamese-bi-encoder` | Dense Embedding | MIT | [HuggingFace](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) |
| 18 | `Qwen/Qwen3-Embedding-0.6B` | Dense Embedding | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| 19 | `BAAI/bge-m3` | Dense / Multi-Function Embedding | MIT | [HuggingFace](https://huggingface.co/BAAI/bge-m3) |
| 20 | `AITeamVN/Vietnamese_Embedding` | Dense Embedding | MIT | [HuggingFace](https://huggingface.co/AITeamVN/Vietnamese_Embedding) |
| 21 | `jinaai/jina-embeddings-v3` | Dense Embedding | CC-BY-NC-4.0 | [HuggingFace](https://huggingface.co/jinaai/jina-embeddings-v3) |
| 22 | `dangvantuan/vietnamese-embedding` | Dense Embedding | Apache-2.0 | [HuggingFace](https://huggingface.co/dangvantuan/vietnamese-embedding) |
| 23 | `vinai/phobert-base-v2` | Encoder / Embedding | MIT | [HuggingFace](https://huggingface.co/vinai/phobert-base-v2) |
| 24 | `vinai/phobert-large` | Encoder / Embedding | MIT | [HuggingFace](https://huggingface.co/vinai/phobert-large) |
| 25 | `Qwen/Qwen3-Reranker-0.6B` | Neural Reranker | Apache-2.0 | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| 26 | `BAAI/bge-reranker-v2-m3` | Neural Reranker | MIT | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| 27 | `BAAI/bge-reranker-large` | Neural Reranker | MIT | [HuggingFace](https://huggingface.co/BAAI/bge-reranker-large) |
| 28 | `jinaai/jina-reranker-v2-base-multilingual` | Neural Reranker | Apache-2.0 | [HuggingFace](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) |
| 29 | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural Reranker | Apache-2.0 | [HuggingFace](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) |

### Nhóm B: Công cụ và thư viện NLP tiếng Việt (Rule-based / Non-parametric)

Các thư viện tách từ, POS tagging tiếng Việt được phép dùng trong tiền xử lý và tokenization cho BM25 (không tính vào parameter budget neural):

| STT | Tool / Thư viện | Vai trò | License | Nguồn |
|---:|---|---|---|---|
| 30 | `underthesea` | Thư viện tách từ & POS tag tiếng Việt | GPL-3.0 | [GitHub](https://github.com/undertheseanlp/underthesea) |
| 31 | `VnCoreNLP` | Bộ công cụ phân tích ngôn ngữ tiếng Việt | MIT | [GitHub](https://github.com/vncorenlp/VnCoreNLP) |
| 32 | `pyvi` | Thư viện tách từ tiếng Việt | MIT | [PyPI](https://pypi.org/project/pyvi/) |
| 33 | `RDRSegmenter` | Thuật toán tách từ dựa trên cây quyết định | GPL | [GitHub](https://github.com/vncorenlp/VnCoreNLP) |
