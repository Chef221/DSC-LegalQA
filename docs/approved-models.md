# Danh Sách Mô Hình Được Ban Tổ Chức Phê Duyệt (Approved Models)

> **Lưu ý:** Tài liệu này ghi nhận danh sách các mô hình tiền huấn luyện và tài nguyên NLP được Ban tổ chức UIT Data Science Challenge 2026 công bố phê duyệt cho Task 2. Việc một mô hình nằm trong danh sách không đồng nghĩa toàn bộ các mô hình này được kích hoạt đồng thời; hệ thống thực tế phải tuân thủ nghiêm ngặt ràng buộc tổng số tham số < 4 tỷ.

---

## 1. Bảng Mô Hình Được Sử Dụng Bởi Hệ Thống Sản Xuất P70

Dưới đây là các mô hình chính thức cấu thành nên hệ thống sản xuất **FROZEN_P3_G2_VNEXT_P63_P70** đạt kết quả chính thức trên Leaderboard cuộc thi:

| Vai Trò | Tên Mô Hình / Hugging Face ID | Pinned Revision | Số Tham Số | Giấy Phép | Duyệt BTC | Nguồn / URL |
|---|---|---|---:|---|:---:|---|
| **Dense Retriever** | `Qwen/Qwen3-Embedding-0.6B` | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HF Repo](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| **Neural Reranker** | `Qwen/Qwen3-Reranker-0.6B` | `e61197ed45024b0ed8a2d74b80b4d909f1255473` | 595.776.512 | Apache-2.0 | **ĐÃ DUYỆT** | [HF Repo](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B) |
| **Generator Base** | `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2.213.241.664 | Apache-2.0 | **ĐÃ DUYỆT** | [HF Repo](https://huggingface.co/Qwen/Qwen3.5-2B) |
| **Generator LoRA** | `p2_epoch1_qlora` (P70 Adapter) | `2b396cf119e72dac316bc06cd9c781d9895fbefd` | 21.823.488 | Apache-2.0 / Team | **ĐÃ DUYỆT** | Huấn luyện từ dữ liệu BTC |

### Tổng Hợp Ngân Sách Tham Số Hệ Thống

$$\sum \text{Params} = 595.776.512 + 595.776.512 + 2.213.241.664 + 21.823.488 = 3.426.618.176 \text{ tham số} \approx 3,43\text{B}$$

- **Giới hạn quy định:** $< 4.000.000.000$ tham số.
- **Biên độ an toàn (Headroom):** $+573.381.824$ tham số ($14,33\%$).
- **Trạng thái tuân thủ:** **PASS**.

---

## 2. Danh Sách Mô Hình Ứng Viên Khác Được BTC Chấp Thuận

Các mô hình dưới đây thuộc danh mục khảo sát và đăng ký hợp lệ theo quy chế cuộc thi:

| STT | Tên Mô Hình | Vai Trò Khảo Sát | Số Tham Số | Giấy Phép | Nguồn |
|---:|---|---|---:|---|---|
| 1 | `ntphuc149/ViLegalQwen3-1.7B-Base` | Generator | 1,72B | Apache-2.0 | [HF](https://huggingface.co/ntphuc149/ViLegalQwen3-1.7B-Base) |
| 2 | `ntphuc149/ViLegalQwen2.5-1.5B-Base` | Generator | 1,54B | Apache-2.0 | [HF](https://huggingface.co/ntphuc149/ViLegalQwen2.5-1.5B-Base) |
| 3 | `AITeamVN/Vi-Qwen2-1.5B-RAG` | Generator | 1,54B | Apache-2.0 | [HF](https://huggingface.co/AITeamVN/Vi-Qwen2-1.5B-RAG) |
| 4 | `VietAI/vit5-large` | Generator | 754M | Apache-2.0 | [HF](https://huggingface.co/VietAI/vit5-large) |
| 5 | `VietAI/vit5-base` | Generator | 226M | Apache-2.0 | [HF](https://huggingface.co/VietAI/vit5-base) |
| 6 | `thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2` | Generator | 1,72B | Apache-2.0 | [HF](https://huggingface.co/thangvip/qwen3-1.7b-vietnamese-legal-grpo-phase-2) |
| 7 | `Qwen/Qwen3-1.7B` | Generator | 1,72B | Apache-2.0 | [HF](https://huggingface.co/Qwen/Qwen3-1.7B) |
| 8 | `Qwen/Qwen2.5-1.5B-Instruct` | Generator | 1,54B | Apache-2.0 | [HF](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| 9 | `google/gemma-3-1b-it` | Generator | 1,00B | Gemma Terms | [HF](https://huggingface.co/google/gemma-3-1b-it) |
| 10 | `BAAI/bge-m3` | Embedding | 567M | MIT | [HF](https://huggingface.co/BAAI/bge-m3) |
| 11 | `AITeamVN/Vietnamese_Embedding` | Embedding | 278M | MIT | [HF](https://huggingface.co/AITeamVN/Vietnamese_Embedding) |
| 12 | `BAAI/bge-reranker-v2-m3` | Reranker | 567M | MIT | [HF](https://huggingface.co/BAAI/bge-reranker-v2-m3) |
| 13 | `jinaai/jina-embeddings-v3` | Embedding | 570M | CC-BY-NC-4.0 | [HF](https://huggingface.co/jinaai/jina-embeddings-v3) |
| 14 | `jinaai/jina-reranker-v2-base-multilingual` | Reranker | 278M | Apache-2.0 | [HF](https://huggingface.co/jinaai/jina-reranker-v2-base-multilingual) |
