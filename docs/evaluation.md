# Kết Quả Đánh Giá & Bằng Chứng Thực Nghiệm

Tài liệu này ghi nhận kết quả đánh giá chính thức trên Leaderboard cuộc thi UIT Data Science Challenge 2026 và các kết quả thực nghiệm nội bộ có đầy đủ bằng chứng kiểm toán (provenance).

---

## 1. Kết Quả Bảng Xếp Hạng Chính Thức (Official Leaderboard Result)

Hệ thống nộp bài hoàn chỉnh của đội thi (Lineage: **FROZEN_P3_G2_VNEXT_P63_P70**) đã được ghi nhận trên nền tảng chấm chính thức với kết quả:

| Chỉ Số Đánh Giá | Điểm Số Chính Thức | Vai Trò Xếp Hạng | Cơ Chế Tính Điểm |
|---|---:|:---:|---|
| **METEOR** | **0.486776583** | **Chỉ số xếp hạng chính** | NLTK METEOR trên whitespace tokens (WordNet/OMW) |
| **ROUGE-L** | **0.530283618** | **Chỉ số xếp hạng phụ** | Vendored ASCII-tokenized ROUGE-L (LCS macro mean) |

- **Mã định danh bản nộp:** `ac6b796794cf3c422889620753743fa248d9ada8dd18d681f36d2e38784e1056`
- **Số lượng câu hỏi đánh giá:** 1.000 câu hỏi (Public Test).
- **Trạng thái thực thi:** Hoàn thành 1.000/1.000 câu, 0 lỗi định dạng, 100% tuân thủ cấu trúc submission.

---

## 2. Đánh Giá Thành Phần Lựa Chọn Căn Cứ P63 (P63 Selector Validation)

Mô hình lựa chọn căn cứ pháp lý **P63** (`HistGradientBoostingRegressor`) với 37 đặc trưng không tham chiếu (reference-free) và ngưỡng can thiệp $\tau = 0.100$ được kiểm chứng trên tập dữ liệu kiểm thử nội bộ tách biệt (disjoint heldout):

| Cấu Hình Lựa Chọn | METEOR Trung Bình | Delta (\(\Delta\)) | Phân Loại Cổng Khoa Học |
|---|---:|---:|:---:|
| **Incumbent Action (Top-2 Mặc Định)** | 0.4453189045593393 | — | Baseline |
| **P63 Selected Action (\(\tau = 0.100\))** | **0.4862964382201936** | **+0.04097753366085427** | **PASS_STRONG_POSITIVE** |

- **Bằng chứng thực nghiệm:** `scratch/p63_unseen_query_deployment_staging_r4/P63_DEPLOYMENT_MODEL_METADATA.json`
- **Số hàng ứng viên huấn luyện:** 215.147 hàng từ 5.300 câu hỏi huấn luyện.
- **Tập kiểm thử:** Heldout 500 câu hỏi được phân bổ tất định bằng hàm băm SHA256(qid), hoàn toàn không rò rỉ vào tập fit selector.
- **Kết luận:** Cơ chế P63 mang lại mức tăng thực nghiệm nhảy vọt **+0.0410 METEOR** so với việc chỉ lấy cố định 2 văn bản xếp hạng đầu tiên.

---

## 3. Trạng Thái Khoa Học Của P70 Generator

- **Trạng thái cổng nội bộ tại thời điểm đóng hệ thống:** `INCONCLUSIVE` (thời điểm đóng sổ trước hạn chót nộp bài không đủ chu kỳ A/B test kiểm soát trên toàn bộ 1.000 câu hỏi).
- **Quyết định triển khai:** Được đưa vào sản xuất theo quyết định vận hành `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`.
- **Hiệu quả thực tế:** Hệ thống hoàn chỉnh tích hợp P70 đã xác lập điểm số cao nhất của dự án trên Leaderboard chính thức (METEOR 0.4868, ROUGE-L 0.5303).
