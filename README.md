# Mage-Flow-Turbo-Native-Inference

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![Native Runtime](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/releases/tag/v1.0.0)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

A **portable native inference and deployment stack for Mage-Flow-Turbo**. The project does not train or modify model weights. Python provides configuration, validation, CLI/REST orchestration, lifecycle control and evidence collection; model inference is executed by the native `stable-diffusion.cpp` `sd-cli` runtime.

```text
manifest → SHA-256 verification → pinned sd-cli runtime
        → Mage-Flow-Turbo DiT Q8_0
        → Qwen3-VL-4B text encoder Q4_K_M
        → dedicated VAE
        → Linux CPU or NVIDIA CUDA cuda0
        → PNG artifact + structured evidence
```

## Exact reference stack

| Role | Exact artifact | Format / quantization |
|---|---|---|
| Diffusion model | `Mage-Flow-Turbo-DiT-Q8_0.gguf` | GGUF Q8_0 |
| Text encoder | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | GGUF Q4_K_M |
| VAE | `diffusion_pytorch_model.safetensors` | SafeTensors |
| Native runtime | `stable-diffusion.cpp` `sd-cli` | pinned commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba` |

Frozen SHA-256 identities are enforced before real inference. The repository contains no model weights.

## v1.0.0 qualification scope

| Environment | Backend | Qualification role |
|---|---|---|
| Linux x86-64 | CPU | required release target |
| Linux + NVIDIA GPU | CUDA `cuda0` | required release target |
| Kaggle Accelerator=None | auto-selected CPU | production/evidence integration target |
| Kaggle NVIDIA T4/T4x2 | auto-selected CUDA `cuda0` on physical GPU 0 | production/evidence integration target |

The public Kaggle production notebook is **prebuilt-runtime only**: Accelerator=None selects CPU; T4/T4x2 selects CUDA0 slot 0; P100, TPU and other unsupported accelerators fail closed. There is no CPU fallback from an attached unsupported GPU and no source-build fallback.

## Canonical fresh CPU ↔ T4 benchmark

Both authoritative evidence sessions used the same notebook source, source HEAD, model inputs, prompt, seed, steps, CFG, threads and matrix order.

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

**Headline:** at 1024×1024, T4 native generation completed in **12.420 s** versus **939.371 s** on CPU — a **75.63×** same-resolution native-generation speedup.

See [the complete benchmark evidence](docs/BENCHMARKS-v1.0.0.md) for notebook/runtime hashes, methodology and interpretation rules.

## Why native inference?

The diffusion step, text conditioning and VAE decoding are executed by `sd-cli`; there is no PyTorch/Transformers inference loop in the project. Python validates model identity, builds explicit subprocess arguments with `shell=False`, monitors the native process and records evidence.

## Verify the model stack

```bash
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

## Local Linux CPU

The generic CLI still supports building a local runtime when deliberately developing outside the production Kaggle notebook:

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

Release qualification uses deterministic placement (`cpu` or `cuda0`), never inference auto-splitting.

## REST API

The reference service binds to `127.0.0.1` by default.

```text
GET  /healthz
GET  /readyz
GET  /v1/info
POST /v1/images/generate
GET  /v1/artifacts/<png>
```

## Kaggle integration

The public notebook [notebooks/kaggle-production-demo.ipynb](notebooks/kaggle-production-demo.ipynb) automatically detects the supported Kaggle accelerator. Public defaults are `RUN_MODE="experiment"` and `RUN_FAIR_COMPARISON_BENCHMARK=False`; maintainers can opt into one-shot `evidence` mode for the frozen `512 → 640 → 768 → 1024` matrix. See [docs/kaggle.md](docs/kaggle.md).

## Reproducibility and evidence

The canonical request is:

```text
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

CPU and CUDA outputs may legitimately differ byte-for-byte across numerical backends. Evidence records exact source/runtime/model identity, backend, prompt, dimensions, timing class, memory telemetry and PNG SHA-256.

## Documentation

- [Architecture](docs/architecture.md)
- [Model stack](docs/model-stack.md)
- [Local Linux](docs/local-linux.md)
- [CUDA](docs/cuda.md)
- [Kaggle](docs/kaggle.md)
- [Canonical benchmarks](docs/BENCHMARKS-v1.0.0.md)
- [REST API](docs/REST-API.md)
- [Testing](docs/TESTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing](.github/CONTRIBUTING.md)
- [Security policy](.github/SECURITY.md)

## License

MIT License. Copyright © 2026 Đăng Khoa <i.am@dangkhoa.dev>.
