# Cung cấp dữ liệu thi đấu (Data placement)

Theo quy định của BTC UIT Data Science Challenge 2026, **repo công khai này không chứa dữ liệu thô của cuộc thi**.

Để tái lập thực nghiệm, tải dữ liệu chính thức từ BTC và đặt vào thư mục `data/` theo cấu trúc sau:

```text
data/
├── README.md
├── train.json                  # 7.000 mẫu train có nhãn câu trả lời
├── warmup.json                 # File khởi động (50 mẫu)
├── public-official.json        # 1.000 câu hỏi đánh giá Public Test
├── selected-contexts.zip       # Context corpus gồm 8.532 ngữ cảnh
└── selected-contexts/          # Thư mục giải nén từ selected-contexts.zip
    ├── context_0.json
    ├── context_1.json
    └── ...
```

### Kiểm tra tính toàn vẹn (Canonical SHA256)

- `selected-contexts.zip`: `9a4441b4537ceb646b15359f470a1da0904e6c92a61e8c4c376c19e17dec395e`

Sau khi đặt dữ liệu, chạy script chuẩn bị:
```bash
python scripts/prepare_data.py --data_dir data/
```
