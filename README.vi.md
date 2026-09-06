# Mage-Flow-Turbo-Native-Inference

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![Native Runtime](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/releases/tag/v1.0.0)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

**Bộ công cụ suy luận native di động cho Mage-Flow-Turbo.** Dự án không huấn luyện hoặc sửa trọng số model. Python đảm nhận cấu hình, xác minh, điều phối CLI/REST, lifecycle và evidence; suy luận thực tế do runtime native `stable-diffusion.cpp` (`sd-cli`) thực hiện.

```text
manifest → xác minh SHA-256 → runtime sd-cli đã pin
        → Mage-Flow-Turbo DiT Q8_0
        → Qwen3-VL-4B text encoder Q4_K_M
        → VAE riêng
        → Linux CPU hoặc NVIDIA CUDA cuda0
        → PNG + evidence có cấu trúc
```

## Reference stack chính xác

| Vai trò | Artifact chính xác | Định dạng / lượng tử hóa |
|---|---|---|
| Diffusion model | `Mage-Flow-Turbo-DiT-Q8_0.gguf` | GGUF Q8_0 |
| Text encoder | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | GGUF Q4_K_M |
| VAE | `diffusion_pytorch_model.safetensors` | SafeTensors |
| Runtime native | `stable-diffusion.cpp` `sd-cli` | commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba` |

## Phạm vi qualification v1.0.0

| Môi trường | Backend | Vai trò |
|---|---|---|
| Linux x86-64 | CPU | release target bắt buộc |
| Linux + NVIDIA GPU | CUDA `cuda0` | release target bắt buộc |
| Kaggle Accelerator=None | CPU tự chọn | production/evidence integration target |
| Kaggle NVIDIA T4/T4x2 | CUDA `cuda0` tự chọn trên physical GPU 0 | production/evidence integration target |

Production notebook Kaggle chỉ dùng **prebuilt runtime**: Accelerator=None chọn CPU; T4/T4x2 chọn CUDA0 slot 0; P100, TPU và accelerator khác fail closed. Không fallback CPU từ GPU không hỗ trợ và không source-build fallback.

## Benchmark fresh CPU ↔ T4 canonical

Hai evidence session authoritative dùng cùng notebook source, source HEAD, model inputs, prompt, seed, steps, CFG, threads và thứ tự matrix.

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

**Headline:** ở 1024×1024, native generation trên T4 mất **12,420 giây** so với **939,371 giây** trên CPU — nhanh hơn **75.63×** khi so cùng resolution.

Xem [evidence benchmark đầy đủ](docs/BENCHMARKS-v1.0.0.vi.md) để biết notebook/runtime hash, methodology và quy tắc diễn giải.

## Vì sao dùng native inference?

Diffusion, text conditioning và VAE decoding do `sd-cli` thực hiện; không có vòng lặp inference PyTorch/Transformers trong dự án. Python xác minh model, dựng subprocess argv tường minh với `shell=False`, giám sát native process và ghi evidence.

## Xác minh model stack

```bash
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

## Linux CPU cục bộ

Generic CLI vẫn hỗ trợ build runtime cục bộ khi chủ động phát triển ngoài production Kaggle notebook:

```bash
python -m pip install -e .
mageflow-native runtime build --backend cpu
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

## NVIDIA CUDA

```bash
python -m pip install -e .
mageflow-native runtime build --backend cuda
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json --backend cuda0
```

## REST API

Service tham chiếu bind vào `127.0.0.1` theo mặc định.

```text
GET  /healthz
GET  /readyz
GET  /v1/info
POST /v1/images/generate
GET  /v1/artifacts/<png>
```

## Kaggle

Notebook public [notebooks/kaggle-production-demo.ipynb](notebooks/kaggle-production-demo.ipynb) tự detect accelerator Kaggle được hỗ trợ. Mặc định public là `RUN_MODE="experiment"` và `RUN_FAIR_COMPARISON_BENCHMARK=False`; maintainer có thể bật one-shot `evidence` mode để chạy matrix đóng băng `512 → 640 → 768 → 1024`. Xem [docs/kaggle.vi.md](docs/kaggle.vi.md).

## Tái lập và evidence

Canonical request:

```text
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

Output CPU và CUDA có thể khác byte-for-byte do backend số học khác nhau. Evidence ghi exact source/runtime/model identity, backend, prompt, dimensions, timing class, memory telemetry và PNG SHA-256.

## Tài liệu

- [Kiến trúc](docs/architecture.md)
- [Model stack](docs/model-stack.md)
- [Linux cục bộ](docs/local-linux.md)
- [CUDA](docs/cuda.md)
- [Kaggle](docs/kaggle.vi.md)
- [Benchmark canonical](docs/BENCHMARKS-v1.0.0.vi.md)
- [REST API](docs/REST-API.md)
- [Kiểm thử](docs/TESTING.vi.md)
- [Xử lý sự cố](docs/TROUBLESHOOTING.vi.md)
- [Đóng góp](.github/CONTRIBUTING.md)
- [Chính sách bảo mật](.github/SECURITY.md)

## Giấy phép

MIT License. Copyright © 2026 Đăng Khoa <i.am@dangkhoa.dev>.
