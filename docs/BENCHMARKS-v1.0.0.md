# v1.0.0 Canonical CPU ↔ T4 Benchmark Evidence

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARKS-v1.0.0.vi.md)

This document records the authoritative fresh-vs-fresh Kaggle benchmark pair for **Mage-Flow-Turbo-Native-Inference v1.0.0**. Both executed notebooks use identical cell source, the same immutable source head, the same model stack and frozen generation recipe, and the same resolution order. Only the hardware/backend runtime differs.

## Headline

At **1024×1024**, native generation completed in **12.420 s on NVIDIA T4** versus **939.371 s on CPU**, a **75.63× same-resolution native-generation speedup**.

The wrapper/wall comparison at the same resolution is **33.047 s vs 962.737 s**, or **29.13×**.

## Frozen contract

- Release: `v1.0.0`
- Qualified source HEAD: `b9042c743aa925042349af3cf6fcf37dc455af6e`
- stable-diffusion.cpp: `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`
- ggml: `e20c3a14aa70ee84ca58499814206dd08d8026bc`
- Prompt: `A small red fox sitting in a quiet green forest, natural light, detailed photography.`
- Seed: `42`
- Steps: `4`
- CFG: `1.0`
- Threads: `4`
- Matrix order: `512 → 640 → 768 → 1024`
- CPU accelerator: Kaggle `None` → `cpu`
- GPU accelerator: Kaggle T4x2 → `cuda0`, physical slot 0 only (`CUDA_VISIBLE_DEVICES=0`)
- Source build: disabled for the production notebook

## Results

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

Aggregate native matrix time was **1984.901 s on CPU** versus **38.418 s on T4**, a **51.67×** aggregate native speedup.

## Evidence identity

- CPU executed notebook SHA-256: `13f8f01a2b42432bb6f67055fb049a33aa3d78a5c6931750b8bf3dc748a4a718`
- T4x2 executed notebook SHA-256: `be3983f403d2ed7db9977f130db2983c9c8a6e924b089abcc65e7b86022c37b5`
- CPU `sd-cli` SHA-256: `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`
- CUDA `sd-cli` SHA-256: `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`
- Executed notebook source parity: **PASS** (33/33 cells have identical source)
- CPU 512 canonical PNG SHA-256: `c67f3aa4c475f33f5fcecb58392b0a21d1cd82d4d545d5f6e48f59e6a585d819`
- T4 1024 showcase PNG SHA-256: `2c82bdb7cd68746c113eea0f95593d4860b3a201eb3869d0c384221b39b0e49e`

## Interpretation rules

Use **native vs native** or **wall vs wall**, never native vs wrapper. Performance claims must compare the **same resolution**. The notebook's 12-second 1024 reviewer target is informational rather than an acceptance SLA; the measured 12.420 s is retained exactly and the benchmark remains PASS.
