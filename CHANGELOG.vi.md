# Nhật ký thay đổi (Changelog)

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](CHANGELOG.md)

## v1.0.0 — đang chờ publish

Bản phát hành công khai đầu tiên dự kiến của **Mage-Flow-Turbo-Native-Inference** — bộ công cụ suy luận native di động. Publication vẫn bị chặn cho tới khi exact public head vượt qua qualification CPU/CUDA và evidence đã được xác minh.

- Core chung với model manifest JSON và xác minh SHA-256 từng thành phần chuẩn.
- Runtime manager di động: giải quyết, build và xác minh `stable-diffusion.cpp` (`sd-cli`) đã pin cho Linux CPU và NVIDIA CUDA.
- Backend/placement cấu hình được (`cpu`, `cuda0`, chuỗi placement đã kiểm tra, `max_vram`, `split_mode`, `auto_fit`).
- CLI `mageflow-native`: `doctor`, `verify`, `generate`, `serve`, `runtime build --backend cpu|cuda`.
- REST API loopback với các endpoint ổn định (`/healthz`, `/readyz`, `/v1/info`, `/v1/images/generate`, `/v1/artifacts/<png>`).
- Kaggle trở thành adapter mỏng (`integrations/kaggle/`); core chung không có phụ thuộc cứng `/kaggle/*`.
- Bằng chứng portability layout cục bộ và harness qualification CPU/CUDA với đóng gói evidence đã khử nhạy cảm.
- CI không chứa trọng số mô hình và audit publication surface (Actions v7).

```
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
qualification        = pending: Linux CPU, NVIDIA CUDA (cuda0)
steps / CFG / threads = 4 / 1.0 / 4
```
