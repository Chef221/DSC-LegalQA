# Evaluation & benchmark results

Tài liệu này tổng hợp kết quả đánh giá trên official leaderboard UIT Data Science Challenge 2026 và các thực nghiệm nội bộ (internal evaluation).

---

## 1. Kết quả official leaderboard

Final production pipeline (Lineage: **FROZEN_P3_G2_VNEXT_P63_P70**) ghi nhận kết quả trên nền tảng chấm chính thức của BTC:

| Metric | Score | Vai trò | Cách tính |
|---|---:|:---:|---|
| **METEOR** | **0.486776583** | Metric xếp hạng chính | NLTK METEOR trên whitespace tokens (WordNet/OMW) |
| **ROUGE-L** | **0.530283618** | Metric phụ | Vendored ASCII-tokenized ROUGE-L (LCS macro mean) |

- **Submission hash:** `ac6b796794cf3c422889620753743fa248d9ada8dd18d681f36d2e38784e1056`
- **Số lượng câu hỏi:** 1.000 câu hỏi (Public Test).
- **Trạng thái:** Hoàn thành 1.000/1.000 câu, 0 lỗi format, khớp 100% submission schema.

---

## 2. P63 Evidence Selector validation

Mô hình **P63 Evidence Selector** (`HistGradientBoostingRegressor`) với 37 reference-free features và decision threshold $\tau = 0.100$ được kiểm chứng trên tập disjoint heldout:

| Config | Mean METEOR | Delta (\(\Delta\)) | Gate status |
|---|---:|---:|:---:|
| **Incumbent Action (Top-2 mặc định)** | 0.4453189045593393 | — | Baseline |
| **P63 Selected Action (\(\tau = 0.100\))** | **0.4862964382201936** | **+0.04097753366085427** | **PASS_STRONG_POSITIVE** |

- **Artifact provenance:** `scratch/p63_unseen_query_deployment_staging_r4/P63_DEPLOYMENT_MODEL_METADATA.json`
- **Training data:** 215.147 candidate rows từ 5.300 câu hỏi train.
- **Heldout test set:** 500 câu hỏi được phân bổ bằng SHA256(qid) để tránh leakage vào tập fit selector.
- **Kết luận:** P63 selector tăng **+0.0410 METEOR** so với baseline cố định Top-2 candidates.

---

## 3. Trạng thái khoa học của P70 generator

- **Internal gate status tại thời điểm code freeze:** `INCONCLUSIVE` (thời điểm trước deadline không kịp chạy đủ chu kỳ A/B test có kiểm soát trên toàn bộ 1.000 câu hỏi).
- **Quyết định triển khai:** Đưa vào production theo quyết định vận hành `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`.
- **Kết quả thực tế:** Pipeline tích hợp P70 đạt điểm số cao nhất của dự án trên official leaderboard (METEOR 0.4868, ROUGE-L 0.5303).
