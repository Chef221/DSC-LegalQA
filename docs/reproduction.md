# Hướng Dẫn Tái Lập Thực Nghiệm (Reproduction Guide)

Hệ thống hỗ trợ 2 quy trình tái lập thực nghiệm rõ ràng:

---

## Quy Trình 1: Tái Lập Nhanh Bằng Trọng Số Đóng Băng (Quick Reproduction)

Mục tiêu: Chạy pipeline suy luận sản xuất hoàn chỉnh bằng cách sử dụng các trọng số và mô hình đã đóng băng đi kèm repo.

### Bước 1: Chuẩn bị môi trường
```bash
git clone https://github.com/Chef221/DSC-LegalQA.git
cd DSC-LegalQA
pip install -e .
```

### Bước 2: Cung cấp dữ liệu cuộc thi
Đặt file ngữ cảnh chính thức vào `data/selected-contexts/` theo hướng dẫn tại `data/README.md`.

### Bước 3: Chuẩn bị phân đoạn điều luật
```bash
python scripts/prepare_data.py --data_dir data/selected-contexts/ --output_path data/chunks.jsonl
```

### Bước 4: Chạy suy luận sinh câu trả lời
```bash
python scripts/run_inference.py --input_questions examples/sample_question.json --output_submission submission.json
```

### Bước 5: Đóng gói bài nộp
```bash
python scripts/build_submission.py --input_json submission.json --output_zip submission.zip
```

---

## Quy Trình 2: Huấn Luyện Lại Từ Đầu (Full Reproduction)

Mục tiêu: Huấn luyện lại bộ chọn căn cứ P63 và mô hình sinh LoRA P70 từ dữ liệu huấn luyện chính thức của cuộc thi.

### 1. Huấn luyện lại bộ chọn P63:
```bash
python scripts/train_selector.py --output_dir artifacts/p63_selector/
```

### 2. Huấn luyện lại mô hình sinh P70 LoRA:
```bash
python scripts/train_p70_lora.py --epochs 1 --batch_size 1 --output_dir artifacts/p70_adapter_retrained/
```
