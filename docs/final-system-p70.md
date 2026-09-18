# Final P70 Production System

Tài liệu này ghi lại thông số kỹ thuật và provenance của production pipeline chính thức **`vNext + P63 + P70`** (lineage: `FROZEN_P3_G2_VNEXT_P63_P70`).

---

## 1. Provenance và artifact hashes

Tất cả artifacts trong production pipeline P70 được quản lý qua `artifacts/MANIFEST.json` và verify bằng SHA256:

| Artifact | File / Identifier | SHA256 | Bytes |
|---|---|---|---:|
| **P70 LoRA Adapter Weights** | `adapter_model.safetensors` | `193913014b49e9d1d431a44778903842cb8c98678fdf32bd1f3a45e6c020887c` | 87.322.136 |
| **P70 LoRA Adapter Config** | `adapter_config.json` | `2761928c3f17aa894bd2e7793db0d00a2e7c1b0821a7d864dc7bf411dc603f37` | 1.148 |
| **P63 Selector Model** | `P63_DEPLOYMENT_MODEL.pkl` | `1654bf0184f6be7f2bc8a715e4eba5f6d47ebc280f10481fac7a1b09bc424c38` | 387.527 |
| **P63 Selector Metadata** | `P63_DEPLOYMENT_MODEL_METADATA.json` | `ef59ed7efe1a47edf76b417dd53660eb15bfb52fc2a7bc314bb3dd58ca0e27ac` | 778 |
| **Production Duplicate Guard** | `token_suffix_loop_sanitizer.py` | `1d58bb1cac5bef635e39500959b13e77bbd1da3d7c18ec82edeeac5e09713527` | 26.461 |
| **Official Final Submission** | `submission_original_p70.json` (provenance) | `ac6b796794cf3c422889620753743fa248d9ada8dd18d681f36d2e38784e1056` | 2.231.844 |

---

## 2. Config giải mã (Greedy Decoding)

```python
BASELINE_DECODE = {
    "do_sample": False,
    "num_beams": 1,
    "max_new_tokens": 1536,
    "eos_token_id": 248046,
    "pad_token_id": 248044,
    "use_cache": True,
}
```

---

## 3. Parameter budget (<4B)

- `Qwen/Qwen3-Embedding-0.6B`: 595.776.512
- `Qwen/Qwen3-Reranker-0.6B`: 595.776.512
- `Qwen/Qwen3.5-2B`: 2.213.241.664
- `P70 LoRA Adapter`: 21.823.488
- **Subtotal (Neural + LoRA):** **3.426.618.176** tham số
- **P63 Selector (Non-neural):** 100 cây quyết định (6.030 nút cây; file pickle: 387.527 bytes)
- **Tổng toàn hệ thống:** **3.426.618.176** neural (+ 6.030 nút cây P63) < 4.000.000.000 (headroom > 573 triệu tham số, tương đương 14.33%).

---

## 4. Scientific status & deployment decision

Đánh giá nội bộ tại thời điểm đóng hệ thống cho cấu hình generator P70 ghi nhận trạng thái:
- **Scientific Gate:** `INCONCLUSIVE`
- **Promotion Basis:** `OPERATOR_OVERRIDE_DEADLINE_2026-09-17`

Cấu hình P70 được đưa vào final submission trước deadline và đạt kết quả xếp hạng chính thức trên leaderboard của BTC:
- **METEOR:** `0.486776583`
- **ROUGE-L:** `0.530283618`
---

## 5. Runtime environment & historical pins

Historical production pins (được đối soát khi chạy `verify_environment.py --strict`):
- `transformers==5.17.0`
- `peft==0.20.0`
- `accelerate==1.15.0`
- `bitsandbytes==0.50.2`
