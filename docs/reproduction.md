# Reproduction guide

Tài liệu hướng dẫn tái lập các bước thực nghiệm của production pipeline **`vNext + P63 + P70`** (lineage: `FROZEN_P3_G2_VNEXT_P63_P70`) cho UIT Data Science Challenge 2026 — Task 2.

---

## 1. Environment preflight & artifact check

Trước khi chạy inference hoặc training, cần kiểm tra tính toàn vẹn của artifacts và môi trường thực thi:

```bash
# 1. Verify SHA256 và kích thước của 5 production artifacts
python scripts/verify_artifacts.py

# 2. Verify môi trường runtime chuẩn (Strict Mode - bắt buộc cho P70 reproduction)
python scripts/verify_environment.py --strict

# (Tùy chọn) Kiểm tra tương thích cơ bản cho máy dev
python scripts/verify_environment.py
```

Strict mode (`--strict`) đối soát các pinned versions chính thức:
- `transformers==5.17.0`
- `peft==0.20.0`
- `accelerate==1.15.0`
- `bitsandbytes==0.50.2`

---

## 2. Production inference

Để chạy full pipeline từ input question đến file `submission.json`:

### Bước 1: Chuẩn bị dữ liệu
Đặt file câu hỏi và legal chunks vào thư mục `data/`:
```text
data/
  ├── public-official.json    # File câu hỏi JSON/JSONL
  └── chunks.jsonl            # Corpus chunks đã lập chỉ mục
```

### Bước 2: Chạy inference
```bash
python scripts/run_inference.py \
    --input_questions data/public-official.json \
    --chunks_path data/chunks.jsonl \
    --output_submission submission.json \
    --device cuda
```

Thứ tự xử lý của pipeline:
1. **Query normalization:** Unicode NFC + whitespace collapse.
2. **Hybrid Retrieval:** BM25 (Top-100) + Qwen3 Dense Embedding (Q2 prompt, Top-100).
3. **Equal RRF:** Hợp nhất 2 ranking lists ($k=10$) lấy Top-100 candidates.
4. **Prefix20 reranking:** Qwen3-Reranker-0.6B tính yes/no logit difference trên 20 candidates đầu; giữ nguyên tail 21–100.
5. **Candidate materialization:** Tạo candidate sets và trích 37 reference-free features.
6. **P63 selector:** HistGradientBoostingRegressor quyết định `INCUMBENT` hoặc `OVERRIDE` dựa trên threshold $\tau = 0.100$.
7. **Evidence packing:** Đóng gói context với format `[E1]`, `[E2]`... kèm thẻ `[ANSWER_CONTROL]`.
8. **Generator:** Qwen3.5-2B + P70 LoRA sinh câu trả lời bằng greedy decoding (`max_new_tokens=1536`).
9. **Duplicate guard:** `token_suffix_loop_sanitizer` xử lý suffix repetition loops nếu có.
10. **Submission builder:** Format output thành `submission.json` chuẩn schema nộp bài của BTC.

---

## 3. Training reproduction

### A. Huấn luyện P63 Selector
```bash
python scripts/train_selector.py \
    --features_path data/p63_training_features.npz \
    --output_dir artifacts/p63_selector
```
*Ghi chú:* Script yêu cầu file ma trận đặc trưng `p63_training_features.npz` (215.147 rows từ 5.300 training questions). Nếu thiếu file, script sẽ fail-closed (báo lỗi dừng lại) chứ không tự sinh random data.

### B. Huấn luyện P70 Continued-LoRA
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
*Ghi chú:* 4.247 training samples lấy từ Folds 1–4, tách biệt hoàn toàn với heldout split 500 câu hỏi để tránh data leakage.
