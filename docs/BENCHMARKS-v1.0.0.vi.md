# Evidence benchmark CPU ↔ T4 canonical cho v1.0.0

> 🌐 Language / Ngôn ngữ: [English](BENCHMARKS-v1.0.0.md) | **Tiếng Việt**

Tài liệu này ghi lại cặp benchmark Kaggle fresh-vs-fresh authoritative cho **Mage-Flow-Turbo-Native-Inference v1.0.0**. Hai executed notebook có source cell giống hệt nhau, cùng immutable source HEAD, cùng model stack, cùng frozen generation recipe và cùng thứ tự resolution; chỉ hardware/backend runtime khác nhau.

## Kết quả nổi bật

Ở **1024×1024**, native generation hoàn thành trong **12,420 giây trên NVIDIA T4** so với **939,371 giây trên CPU**, tương đương **nhanh hơn 75.63×** khi so sánh cùng resolution.

So sánh wrapper/wall cùng resolution là **33,047 giây so với 962,737 giây**, tương đương **29.13×**.

## Contract đã đóng băng

- Release: `v1.0.0`
- Qualified source HEAD: `b9042c743aa925042349af3cf6fcf37dc455af6e`
- stable-diffusion.cpp: `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`
- ggml: `e20c3a14aa70ee84ca58499814206dd08d8026bc`
- Prompt: `A small red fox sitting in a quiet green forest, natural light, detailed photography.`
- Seed: `42`
- Steps: `4`
- CFG: `1.0`
- Threads: `4`
- Thứ tự matrix: `512 → 640 → 768 → 1024`
- CPU: Kaggle Accelerator `None` → `cpu`
- GPU: Kaggle T4x2 → `cuda0`, chỉ physical slot 0 (`CUDA_VISIBLE_DEVICES=0`)
- Source build: tắt trong production notebook

## Kết quả

| Resolution | CPU native | T4 native | Native speedup | CPU wall | T4 wall | Wall speedup | T4 GPU peak |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s | **28.43×** | 238.072 s | 27.209 s | **8.75×** | 7,930 MiB |
| 640×640 | 338.014 s | 8.698 s | **38.86×** | 363.299 s | 29.040 s | **12.51×** | 8,356 MiB |
| 768×768 | 491.700 s | 9.710 s | **50.64×** | 513.961 s | 30.072 s | **17.09×** | 8,702 MiB |
| 1024×1024 | 939.371 s | 12.420 s | **75.63×** | 962.737 s | 33.047 s | **29.13×** | 9,316 MiB |

Tổng native matrix là **1984.901 giây trên CPU** so với **38.418 giây trên T4**, tương đương **51.67×**.

## Danh tính evidence

- CPU executed notebook SHA-256: `13f8f01a2b42432bb6f67055fb049a33aa3d78a5c6931750b8bf3dc748a4a718`
- T4x2 executed notebook SHA-256: `be3983f403d2ed7db9977f130db2983c9c8a6e924b089abcc65e7b86022c37b5`
- CPU `sd-cli` SHA-256: `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`
- CUDA `sd-cli` SHA-256: `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`
- Source parity của hai executed notebook: **PASS** (33/33 cell có source giống nhau)
- CPU 512 canonical PNG SHA-256: `c67f3aa4c475f33f5fcecb58392b0a21d1cd82d4d545d5f6e48f59e6a585d819`
- T4 1024 showcase PNG SHA-256: `2c82bdb7cd68746c113eea0f95593d4860b3a201eb3869d0c384221b39b0e49e`

## Quy tắc diễn giải

Chỉ so **native với native** hoặc **wall với wall**, không so native với wrapper. Mọi claim hiệu năng phải so **cùng resolution**. Target 12 giây ở 1024 trong notebook chỉ là reviewer target, không phải acceptance SLA; số đo 12,420 giây được giữ nguyên và benchmark vẫn PASS.
