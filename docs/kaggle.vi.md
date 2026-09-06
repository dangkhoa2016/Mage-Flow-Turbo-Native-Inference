# Production notebook Kaggle

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

Kaggle là môi trường adapter/reference đã được kiểm thử. Production notebook public tái sử dụng **prebuilt native runtime** đã xác minh và không compile `stable-diffusion.cpp`.

## Accelerator policy

Notebook detect hardware trước runtime/model discovery:

- `Accelerator=None` → `cpu`
- NVIDIA T4 → `cuda0`
- NVIDIA T4x2 → `cuda0`, chỉ physical slot 0 với `CUDA_VISIBLE_DEVICES=0`
- P100, TPU/v5e-8, mixed GPU và mọi accelerator khác → hard FAIL

Không có silent CPU fallback khi gắn accelerator không hỗ trợ. Nếu exact prebuilt runtime cho backend đã detect bị thiếu hoặc sai hash, notebook FAIL; không build source.

## Mặc định public

```python
RUN_MODE = "experiment"
RESOLUTION_PRESET = "auto"
RUN_FAIR_COMPARISON_BENCHMARK = False
```

`auto` chọn 512×512 trên CPU và 1024×1024 trên CUDA0. Experiment mode tự tạo run label duy nhất và output directory riêng cho mỗi Run All.

## Canonical evidence mode

Maintainer thu authoritative evidence dùng fresh Kaggle session với:

```python
RUN_MODE = "evidence"
RUN_FAIR_COMPARISON_BENCHMARK = True
```

Matrix one-shot:

```text
512 → 640 → 768 → 1024
```

Evidence guard được ghi ngay trước real generation đầu tiên. Evidence transaction thứ hai cho cùng backend yêu cầu fresh Kaggle session.

## Input bắt buộc

Attach ba canonical model input và **chỉ runtime dataset khớp accelerator**:

- CPU: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime`
- T4/T4x2: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime`

Exact runtime binary SHA-256:

- CPU: `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`
- CUDA T4: `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`

## Canonical evidence đã đo

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

Ở 1024×1024, native-generation speedup đo được là **75.63×**. Xem [BENCHMARKS-v1.0.0.vi.md](BENCHMARKS-v1.0.0.vi.md) để biết đầy đủ evidence identity và methodology.
