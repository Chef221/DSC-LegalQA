# Danh Sách Mô Hình Được Ban Tổ Chức Phê Duyệt (Approved Models)

> **Lưu ý nguyên tắc:** Tài liệu này tổng hợp danh mục các mô hình tiền huấn luyện và công cụ xử lý ngôn ngữ tự nhiên (NLP) được Ban tổ chức UIT Data Science Challenge 2026 công nhận và cho phép đăng ký/sử dụng cho Task 2. Việc một mô hình xuất hiện trong danh mục này biểu thị tính hợp lệ về mặt nguồn gốc; khi triển khai thực tế, toàn bộ hệ thống phải thỏa mãn điều kiện tiên quyết: **tổng số tham số của tất cả các mô hình có trọng số trong hệ thống phải dưới 4 tỷ (< 4.000.000.000)**.

---

## 1. Các Mô Hình Được Sử Dụng Bởi Hệ Thống Sản Xuất Cuối Cùng (vNext + P63 + P70)

Bảng dưới đây liệt kê các mô hình chính thức cấu thành nên hệ thống sản xuất **FROZEN_P3_G2_VNEXT_P63_P70** đạt kết quả METEOR = **0.486776583** và ROUGE-L = **0.530283618** trên Leaderboard chính thức:

| Vai Trò | Tên Mô Hình / HF ID | Pinned Revision | Số Tham Số Xác Minh | Giấy Phép | Duyệt BTC | Nguồn / URL |
|---|---|---|---:|---|:---:|---|
| **Dense Retriever** | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| **Neural Reranker** | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| **Generator Base** | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2.213.241.664 | Apache-2.0 | **ĐÃ DUYỆT** | [HuggingFace](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **Generator LoRA** | `P70 Adapter` (Continued-LoRA) | `193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c` | 21.823.488 | Apache-2.0 / Team | **ĐÃ DUYỆT** | Huấn luyện từ dữ liệu BTC |

### Bảng Kê Ngân Sách Tham Số Toàn Hệ Thống

$$\sum \text{Params} = 595.776.512 + 595.776.512 + 2.213.241.664 + 21.823.488 = 3.426.618.176 \approx 3,43\text{B}$$

- **Giới hạn quy định BTC:** $< 4.000.000.000$ tham số.
- **Biên độ an toàn (Headroom):** $+573.381.824$ tham số ($14,33\%$ margin).
- **Trạng thái tuân thủ:** **PASS**.
- *Báo cáo minh bạch về P63 Evidence Selector:* Mô hình lựa chọn căn cứ P63 là một tập hợp cây quyết định phi nơ-ron (`HistGradientBoostingRegressor`) gồm 100 cây và 6.030 nút cây (tree nodes) trên 37 đặc trưng (kích thước file pickle: 387.527 bytes, không sử dụng mạng nơ-ron). Các nút cây/ngưỡng chia không có quy ước quy đổi 1:1 tương đương với trọng số tensor mạng nơ-ron. Ngay cả khi tính tượng trưng toàn bộ 6.030 nút cây như các tham số học độc lập, tổng độ phức tạp của toàn hệ thống vẫn là 3.426.624.206 tham số, tuyệt đối nằm dưới ngưỡng 4 tỷ.

---

## 2. Toàn Bộ Danh Mục Mô Hình & Tài Nguyên NLP Được BTC Phê Duyệt

> **Ghi chú về nguồn dữ liệu:** Danh sách dưới đây tổng hợp đầy đủ các dòng mô hình và tài nguyên ngôn ngữ từ văn bản công bố của Ban tổ chức cũng như danh mục đăng ký hợp lệ. Các mục trùng lặp nguyên văn đã được gom nhóm hiển thị và chú thích rõ ràng.

### Nhóm A: Pretrained / Instruction Models (< 4B)

| STT | Tên Mô Hình / Resource | Phân Loại | Giấy Phép / Quyền Sử Dụng | Nguồn Tham Chiếu |
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

### Nhóm B: Công Cụ & Thư Viện NLP Tiếng Việt Hợp Lệ (Rule-Based / Non-Parametric)

Các công cụ dưới đây là thư viện tách từ, phân tích ngữ pháp tiếng Việt được phép sử dụng trong giai đoạn tiền xử lý và tách từ cho BM25 (không tính vào ngân sách tham số neural):

| STT | Tên Công Cụ / Thư Viện | Vai Trò | Giấy Phép | Nguồn |
|---:|---|---|---|---|
| 30 | `underthesea` | Thư viện tách từ & POS tag tiếng Việt | GPL-3.0 | [GitHub](https://github.com/undertheseanlp/underthesea) |
| 31 | `VnCoreNLP` | Bộ công cụ phân tích ngôn ngữ tiếng Việt | MIT | [GitHub](https://github.com/vncorenlp/VnCoreNLP) |
| 32 | `pyvi` | Thư viện tách từ tiếng Việt | MIT | [PyPI](https://pypi.org/project/pyvi/) |
| 33 | `RDRSegmenter` | Thuật toán tách từ dựa trên cây quyết định | GPL | [GitHub](https://github.com/vncorenlp/VnCoreNLP) |