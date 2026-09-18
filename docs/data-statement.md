# Data Statement — DSC-LegalQA

## 1. Nguồn dữ liệu
Hệ thống tuân thủ quy định của UIT Data Science Challenge 2026:
- Chỉ sử dụng dữ liệu chính thức do BTC cung cấp.
- Không dùng external data, không crawl web, không tự gán nhãn thủ công.
- Không sử dụng synthetic data.

## 2. Đặc điểm ngôn ngữ & văn bản
- Ngôn ngữ: Tiếng Việt pháp luật.
- Phạm vi văn bản: Các văn bản quy phạm pháp luật Việt Nam (Bộ luật, Luật, Nghị định, Thông tư) trong corpus `selected-contexts.zip` (8.532 contexts).

## 3. Chính sách dữ liệu của repo
- Theo quy chế cuộc thi, repo công khai không chứa dữ liệu thô của BTC.
- Repo chỉ cung cấp source code, config, pipeline scripts và learned weights/adapters. Người dùng tải dữ liệu trực tiếp từ BTC và đặt vào thư mục `data/` theo hướng dẫn tại `data/README.md`.
