# Kaggle production notebook

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

Kaggle is a tested adapter/reference environment. The public production notebook reuses verified **prebuilt native runtimes** and does not compile `stable-diffusion.cpp`.

## Accelerator policy

The notebook detects hardware before runtime or model discovery:

- `Accelerator=None` → `cpu`
- NVIDIA T4 → `cuda0`
- NVIDIA T4x2 → `cuda0`, physical slot 0 only with `CUDA_VISIBLE_DEVICES=0`
- P100, TPU/v5e-8, mixed GPUs and every other unsupported accelerator → hard FAIL

There is no silent CPU fallback from an attached unsupported accelerator. If the exact prebuilt runtime for the detected backend is missing or hash-mismatched, the notebook fails; it does not build from source.

## Public defaults

```python
RUN_MODE = "experiment"
RESOLUTION_PRESET = "auto"
RUN_FAIR_COMPARISON_BENCHMARK = False
```

`auto` resolves to 512×512 on CPU and 1024×1024 on CUDA0. Experiment mode allocates a unique automatic run label and isolated output directory for every Run All.

## Canonical evidence mode

Maintainers collecting authoritative evidence use a fresh Kaggle session with:

```python
RUN_MODE = "evidence"
RUN_FAIR_COMPARISON_BENCHMARK = True
```

The one-shot matrix is:

```text
512 → 640 → 768 → 1024
```

The evidence guard is written immediately before the first real generation. A second evidence transaction for the same backend requires a fresh Kaggle session.

## Required inputs

Attach the three canonical model inputs plus **only the runtime dataset matching the accelerator**:

- CPU: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime`
- T4/T4x2: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime`

Exact runtime binary SHA-256:

- CPU: `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`
- CUDA T4: `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`

## Measured canonical evidence

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

At 1024×1024 the measured native-generation speedup is **75.63×**. See [BENCHMARKS-v1.0.0.md](BENCHMARKS-v1.0.0.md) for the full evidence identity and methodology.
