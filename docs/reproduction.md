# Hướng Dẫn Tái Lập Thực Nghiệm (Reproduction Guide)

Tài liệu này cung cấp các quy trình tái lập hệ thống sản xuất **FROZEN_P3_G2_VNEXT_P63_P70** cho bài toán UIT Data Science Challenge 2026 Task 2.

---

## 1. Xác Minh Môi Trường Thực Thi & Kiểm Tra Tạo Tác (Preflight Checks)

Trước khi thực hiện bất kỳ quy trình nào, hãy chạy hai công cụ xác minh tất định sau:

```bash
# 1. Xác minh tính toàn vẹn của các tệp mô hình và mã nguồn đóng băng
python scripts/verify_artifacts.py

# 2. Xác minh môi trường tái lập chuẩn (Strict Mode - bắt buộc cho P70 production)
python scripts/verify_environment.py --strict

# (Tùy chọn) Kiểm tra thông tin tương thích cho máy phát triển
python scripts/verify_environment.py
```

---

## 2. Quy Trình Tái Lập Suy Luận Sản Xuất (Production Inference Reproduction)

Để chạy lại toàn bộ quy trình suy luận từ câu hỏi đầu vào đến tệp kết quả `submission.json`:

### Bước 1: Chuẩn bị dữ liệu cuộc thi
Đặt tệp câu hỏi (ví dụ `public-official.json`) và tệp ngữ cảnh pháp lý đã giải nén hoặc tệp chunks đã xử lý:
```text
data/
  ├── public-official.json
  └── chunks.jsonl
```

### Bước 2: Chạy bộ suy luận hoàn chỉnh
```bash
python scripts/run_inference.py \
    --input_questions data/public-official.json \
    --chunks_path data/chunks.jsonl \
    --output_submission submission.json \
    --device cuda
```

Hệ thống sẽ thực hiện theo thứ tự:
1. Chuẩn hóa câu hỏi tiếng Việt (NFC + chuẩn hóa khoảng trắng).
2. Lập chỉ mục & truy xuất BM25 (Top 100) song song với Qwen3 Dense Embedding (Q2 instruction, Top 100).
3. Hợp nhất thứ hạng tương hỗ Equal RRF ($k=10$) chọn Top 100 ứng viên.
4. Rerank Prefix 20 ứng viên bằng `Qwen/Qwen3-Reranker-0.6B` (hiệu logit yes/no), giữ nguyên tail 21–100.
5. Tạo tập ứng viên và trích xuất 37 đặc trưng không tham chiếu cho P63 Selector.
6. Mô hình `HistGradientBoostingRegressor` đưa ra quyết định `INCUMBENT` hoặc `OVERRIDE` dựa trên $\tau = 0.100$.
7. Đóng gói căn cứ pháp luật theo chuẩn `[E1]`, `[E2]` kết hợp khối `[ANSWER_CONTROL]`.
8. Sinh câu trả lời qua `Qwen/Qwen3.5-2B` + P70 LoRA với giải mã tham lam (`max_new_tokens=1536`).
9. Bộ lọc lặp vòng `token_suffix_loop_sanitizer` xử lý các chu kỳ lặp nếu có.
10. Xuất tệp `submission.json` chuẩn cấu trúc Ban tổ chức yêu cầu.

---

## 3. Quy Trình Tái Lập Huấn Luyện (Training Reproduction)

### A. Huấn luyện lại bộ lựa chọn căn cứ P63
```bash
python scripts/train_selector.py \
    --features_path data/p63_training_features.npz \
    --output_dir artifacts/p63_selector
```
*Lưu ý:* Tập đặc trưng `p63_training_features.npz` chứa 215.147 hàng ứng viên từ 5.300 câu hỏi huấn luyện. Nếu tệp không tồn tại, script sẽ dừng có kiểm soát (`fail-closed`), tuyệt đối không tự ý sinh dữ liệu ngẫu nhiên.

### B. Huấn luyện lại Continued-LoRA P70
```bash
python scripts/train_p70_lora.py \
    --train_file data/P70_TRAIN_ROWS.jsonl \
    --heldout_file data/P70_HELDOUT500_ROWS.jsonl \
    --p2_adapter_dir artifacts/p2_adapter \
    --output_dir artifacts/p70_trained_adapter \
    --epochs 1 \
    --lr 1e-4 \
    --grad_accum 16
```
