# Hướng Dẫn Cung Cấp Dữ Liệu Thi Đấu (Official Data Placement)

Theo quy định bảo mật và bản quyền của Ban tổ chức UIT Data Science Challenge 2026, **kho lưu trữ công khai này KHÔNG chứa dữ liệu thô của cuộc thi**.

Thành viên muốn tái lập thực nghiệm cần tải dữ liệu chính thức qua kênh công bố của Ban tổ chức và đặt vào thư mục `data/` theo cấu trúc sau:

```text
data/
├── README.md
├── train.json                  # 7.000 mẫu huấn luyện có nhãn câu trả lời
├── warmup.json                 # Tập khởi động (warm-up)
├── public-official.json        # 1.000 câu hỏi đánh giá Public Test
├── selected-contexts.zip       # Corpus pháp luật gồm 8.532 ngữ cảnh
└── selected-contexts/          # Thư mục giải nén từ selected-contexts.zip
    ├── context_0.json
    ├── context_1.json
    └── ...
```

### Kiểm tra tính toàn vẹn (Canonical SHA256)

- `selected-contexts.zip`: `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`

Sau khi đặt dữ liệu, chạy lệnh chuẩn bị:
```bash
python scripts/prepare_data.py --data_dir data/
```
