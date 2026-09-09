# Nhật ký thay đổi (Changelog)

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](CHANGELOG.md)

## v1.0.0

Bản phát hành công khai đầu tiên của **Mage-Flow-Turbo-Native-Inference**, bộ công cụ suy luận và triển khai native di động cho Mage-Flow-Turbo.

- Model manifest JSON với xác minh SHA-256 cho từng thành phần canonical.
- Runtime manager di động để resolve, build và xác minh `stable-diffusion.cpp` (`sd-cli`) đã pin cho Linux CPU và NVIDIA CUDA.
- Backend/parameter placement tường minh gồm `cpu`, `cuda0`, chuỗi placement đã kiểm tra, `max_vram`, `split_mode` và `auto_fit` cho các thử nghiệm ngoài release gate.
- CLI `mageflow-native`: `doctor`, `verify`, `generate`, `serve`, và `runtime build --backend cpu|cuda`.
- REST API loopback với endpoint health, readiness, info, generation và artifact ổn định.
- Kaggle là adapter mỏng dưới `integrations/kaggle/`; core chung không có hard dependency `/kaggle/*`.
- Harness qualification CPU/CUDA với evidence đã sanitize và có thể clean-room verify.
- CI không chứa model weights, audit public surface, invariant history và release-contract checks.
- Evidence exact-head CPU/CUDA, notebook đã chạy, PNG acceptance, release provenance và SHA-256 checksums được phát hành cùng GitHub Release.

Hợp đồng kỹ thuật đông cứng:

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
canonical request     = 512x512, seed 42, 4 steps, CFG 1.0, 4 threads
```
