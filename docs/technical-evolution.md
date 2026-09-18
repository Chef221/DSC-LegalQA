# Quá trình phát triển kỹ thuật (Technical Evolution)

Tài liệu này ghi lại quá trình hoàn thiện pipeline qua các giai đoạn thực nghiệm, tập trung vào những cải tiến thành công và được giữ lại trong kiến trúc cuối cùng (**`vNext + P63 + P70`**).

---

## 1. Hybrid Retrieval
- **Vấn đề ban đầu:** BM25 thuần túy cho độ khớp cao trên từ khóa hành chính (số hiệu văn bản, thuật ngữ cố định), nhưng giảm hiệu quả khi câu hỏi diễn đạt theo văn phong tự nhiên. Ngược lại, Dense Retrieval đơn lẻ nắm bắt ngữ nghĩa khái quát tốt nhưng dễ bỏ sót các điều khoản có từ khóa định danh cụ thể.
- **Giải pháp:** Chạy song song 2 nhánh retrieval: BM25Okapi ($k_1=1.5, b=0.75$) và `Qwen/Qwen3-Embedding-0.6B` với legal instruction $Q2$.
- **Mục tiêu:** Tăng candidate coverage bằng cách kết hợp cả lexical matching và semantic embedding trước bước merge thứ hạng.

## 2. Equal RRF
- **Vấn đề:** Điểm BM25 và cosine similarity của dense retriever nằm trên hai thang đo khác nhau, không thể cộng trực tiếp.
- **Giải pháp:** Dùng Reciprocal Rank Fusion (RRF) với hằng số $k = 10$, trọng số đồng đều ($1.0 : 1.0$) và tie-break rule dựa trên chunk ID.
- **Mục tiêu:** Tạo danh sách Top-100 candidates ổn định, độc lập với biên độ raw score của từng model.

## 3. Prefix20 reranking
- **Vấn đề:** Một số candidates ở đầu danh sách RRF có độ tương đồng ngữ nghĩa chung nhưng chưa giải quyết chính xác câu hỏi pháp lý cụ thể.
- **Giải pháp:** Dùng `Qwen/Qwen3-Reranker-0.6B` tính logit difference giữa token `yes` (ID 9693) và `no` (ID 2152) trên chat template. Rerank cho 20 candidates đầu tiên (Prefix20); tail 21–100 được giữ nguyên thứ tự.
- **Mục tiêu:** Tối ưu ranking cho nhóm candidates đầu bảng mà không tốn compute cho toàn bộ 100 items.

## 4. P63 Evidence Selector
- **Vấn đề (Evidence Authority Dilemma):** Lấy cố định 2 điều khoản đầu tiên (Incumbent First-2) có thể thừa văn bản hoặc thiếu căn cứ quan trọng nếu điều khoản then chốt nằm ở vị trí 3 hoặc 4.
- **Khảo sát P62:** Đánh giá thực nghiệm nội bộ chứng minh việc can thiệp chọn candidate set đem lại cải thiện rõ ràng:
  - Incumbent METEOR (cố định First-2): **0.4453189045593393**
  - Selected METEOR (theo selector): **0.4862964382201936**
  - Chênh lệch ($\Delta$): **+0.04097753366085427** (`PASS_STRONG_POSITIVE`).
- **Triển khai P63:** Huấn luyện `HistGradientBoostingRegressor` trên 215.147 hàng candidate với 37 reference-free features.
- **Quy tắc can thiệp:** Khi model dự báo mức cải thiện $\Delta \ge 0.100$ ($\tau = 0.100$), hệ thống kích hoạt `OVERRIDE` sang candidate điểm cao nhất; ngược lại giữ `INCUMBENT` mặc định.

## 5. Evidence packing và Answer Control
- **Vấn đề:** Model sinh có thể bị hallucination hoặc diễn giải ngoài căn cứ pháp luật được cung cấp.
- **Giải pháp:** Format các chunks được chọn theo cấu trúc `[E1]`, `[E2]`, kết hợp block chỉ thị `[ANSWER_CONTROL]` yêu cầu câu trả lời chỉ dựa vào các căn cứ này.

## 6. P70 Continued-LoRA
- **Vấn đề:** Base model `Qwen/Qwen3.5-2B` cần làm quen với văn phong văn xuôi pháp luật Việt Nam theo tiêu chí chấm điểm của Task 2.
- **Giải pháp:** Fine-tune Continued-LoRA (P70) trên 4.247 mẫu từ Folds 1–4, tách biệt hoàn toàn với heldout split, áp dụng loss masking trên prompt tokens.
- **Trạng thái khoa học:** P70 là generator configuration được triển khai trong submission cuối cùng. Toàn bộ pipeline dùng P70 đạt METEOR **0.486776583** và ROUGE-L **0.530283618** trên official leaderboard. Đánh giá nội bộ dành riêng cho P70 tại thời điểm đóng hệ thống có trạng thái `INCONCLUSIVE`; quyết định triển khai được thực hiện theo `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`.

## 7. Production duplicate guard
- **Vấn đề:** Greedy decoding đôi khi gặp hiện tượng suffix repetition loops ở đuôi câu trả lời, làm tụt điểm METEOR và ROUGE-L.
- **Giải pháp:** Tích hợp `token_suffix_loop_sanitizer.py` (SHA256: `1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527`, 26 KB) để phát hiện chu kỳ token lặp và cắt tỉa đúng vị trí bắt đầu loop.

## 8. Final production pipeline
- Tích hợp liên hoàn 7 thành phần trên tạo nên pipeline hoàn chỉnh `vNext + P63 + P70`.
- Kết quả official leaderboard: **METEOR = 0.486776583**, **ROUGE-L = 0.530283618**.
