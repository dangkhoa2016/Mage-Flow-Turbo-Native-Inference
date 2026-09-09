# Ghi chú phát hành Mage-Flow-Turbo-Native-Inference v1.0.0

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](RELEASE-NOTES-v1.0.0.md)

## Tổng quan

Mage-Flow-Turbo-Native-Inference là **bộ công cụ suy luận và triển khai native di động cho Mage-Flow-Turbo**. Python đảm nhận cấu hình, xác minh, điều phối CLI/REST, lifecycle và thu thập evidence; việc suy luận thực tế do runtime native `stable-diffusion.cpp` (`sd-cli`) thực hiện. Dự án không định nghĩa, huấn luyện hoặc fine-tune trọng số mô hình.

v1.0.0 là bản phát hành công khai đầu tiên của dòng này.

## Mục tiêu qualification của release

Các target dưới đây bắt buộc phải vượt qua exact-head qualification trước khi GitHub Release v1.0.0 được phát hành:

- **Linux x86-64 CPU** — canonical native CPU target.
- **NVIDIA CUDA `cuda0`** — canonical single-GPU CUDA target.
- **Notebook Kaggle CPU** — CPU adapter/reference target.
- **Notebook Kaggle T4/T4x2** — CUDA adapter/reference target dùng physical GPU 0.

Multi-GPU inference, Vulkan, Metal, ROCm, SYCL và Windows nằm ngoài phạm vi qualification của v1.0.0.

## Điểm nổi bật

- Model manifest JSON với xác minh SHA-256 fail-closed cho mọi thành phần canonical.
- Runtime manager di động cho `stable-diffusion.cpp` `sd-cli` đã pin trên CPU và CUDA.
- Backend/parameter placement tường minh và placement qualification xác định.
- CLI `mageflow-native` cho diagnostics, verification, generation, serving và runtime lifecycle.
- REST API loopback với endpoint health/readiness/info/generation/artifact ổn định.
- Kaggle integration được cô lập trong `integrations/kaggle/`; generic core không có hard dependency vào Kaggle path.
- Harness qualification CPU/CUDA với telemetry RAM/VRAM có cấu trúc và exact argv capture.
- Evidence archive đã sanitize, clean-room verify được, kèm manifest và sidecar checksum.
- CI không chứa model weights, audit public surface, verify public-history invariant và release-contract audit.

## Hợp đồng kỹ thuật đông cứng

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE-only SHA256       = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
canonical size        = 512x512
seed / steps / CFG    = 42 / 4 / 1.0
threads               = 4
```

## Evidence và provenance

Sau khi toàn bộ exact-head gate bắt buộc PASS, artifact qualification CPU/CUDA được phát hành dưới dạng GitHub Release assets. Bộ asset gồm evidence archive và sidecar, notebook đã chạy, PNG acceptance, release provenance và `SHA256SUMS` bao phủ mọi custom release asset. Không có model weights.

Wall time, peak RAM/VRAM, runtime binary SHA-256, acceptance PNG hash, kết quả PASS/FAIL và GitHub Actions run ID được ghi trong release assets và GitHub Release body thay vì source documentation cần thay đổi sau qualification.
