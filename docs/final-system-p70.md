# Hệ Thống Sản Xuất Chung Cuộc P70 (Final P70 Production System)

Tài liệu này công bố chi tiết kỹ thuật và nguồn gốc xác thực của hệ thống thi đấu chính thức **FROZEN_P3_G2_VNEXT_P63_P70**.

---

## 1. Thông Tin Nhận Diện Trọng Số và Mã Băm Xác Thực

Mọi thành phần trong hệ thống sản xuất P70 đều có mã băm toàn vẹn SHA256 được kiểm toán độc lập:

| Thành Phần | Tên File / Định Danh | Mã Băm SHA256 | Kích Thước |
|---|---|---|---:|
| **P70 LoRA Adapter Weights** | `adapter_model.safetensors` | `193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c` | 87.322.136 bytes |
| **P70 LoRA Adapter Config** | `adapter_config.json` | `2761928c3f17aa894bd2e7793db0d00a2e7c1b0821a7d864dc7bf411dc603f37` | 1.148 bytes |
| **P63 Selector Model** | `P63_DEPLOYMENT_MODEL.pkl` | `1654bf0184f6be7f2bc8a715e4eba5f6d47ebc280f10481fac7a1b09bc424c38` | 387.527 bytes |
| **P63 Selector Metadata** | `P63_DEPLOYMENT_MODEL_METADATA.json` | `ef59ed7efe1a47edf76b417dd53660eb15bfb52fc2a7bc314bb3dd58ca0e27ac` | 778 bytes |
| **Production Duplicate Guard** | `token_suffix_loop_sanitizer.py` | `1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527` | 26.461 bytes |
| **Official Final Submission** | `submission_original_p70.json` | `ac6b796794cf3c422889620753743fa248d9ada8dd18d681f36d2e38784e1056` | 2.231.844 bytes |

---

## 2. Cấu Hình Sinh Greedy Tất Định (Deterministic Generation Settings)

```python
DECODE_CONFIG = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 1536,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "use_cache": True,
}
```

---

## 3. Bản Đồ Tham Số và Tính Hợp Lệ Quy Chế (< 4B)

- Qwen3-Embedding-0.6B: .776.512$
- Qwen3-Reranker-0.6B: .776.512$
- Qwen3.5-2B: .213.241.664$
- P70 LoRA Adapter: .823.488$
- **Subtotal (Neural + LoRA):** .426.618.176$
- **P63 Selector (Non-neural):** $ cây quyết định (.030$ nút cây; file pickle: .527$ bytes).
- **Tổng toàn hệ thống:** .426.618.176$ neural ($+ 6.030$ nút cây P63) $< 4.000.000.000$ (Dư dôi an toàn hơn $ triệu tham số, tương đương ,33\%$).

