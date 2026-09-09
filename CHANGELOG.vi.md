# Nhật ký thay đổi (Changelog)

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](CHANGELOG.md)

## v1.0.0

Bản phát hành công khai đầu tiên của **Mage-Flow-Turbo-Native-Inference**, bộ công cụ suy luận và triển khai native di động cho Mage-Flow-Turbo.

- Native inference được thực thi bằng `stable-diffusion.cpp` `sd-cli` đã pin; không có Hugging Face Transformers/PyTorch inference backend.
- Hai release model profiles:
  - `q8-reference` — Mage-Flow-Turbo GGUF Q8_0 diffusion, Qwen3-VL-4B GGUF Q4_K_M text encoder dùng chung và VAE SafeTensors; canonical/default.
  - `bf16-safetensors` — Mage-Flow-Turbo BF16 SafeTensors diffusion, Qwen3-VL-4B GGUF Q4_K_M text encoder dùng chung và VAE SafeTensors; supported alternative.
- Cả hai profile đều hỗ trợ explicit qualification backend `cpu` và `cuda0`.
- Strict four-cell qualification matrix: Q8/CPU, BF16/CPU, Q8/T4 CUDA0 và BF16/T4 CUDA0.
- Mỗi cell dùng một fresh Kaggle session, chỉ attach đúng một Mage diffusion family, cùng final source HEAD/TREE và cùng ordered canonical matrix `512 → 640 → 768 → 1024`.
- GPU qualification là strict single-T4 kể cả khi host là T4x2: `CUDA_VISIBLE_DEVICES=0`, effective backend `cuda0`, không `cuda1`, không multi-GPU split, không `auto-fit`, không CPU inference fallback.
- Release qualification chỉ dùng prebuilt runtime; source build vẫn dành cho deliberate development ngoài evidence session.
- JSON model manifest khóa SHA-256 identity cho từng model component được chọn.
- Evidence ghi source HEAD/TREE, profile/backend identity, runtime commit/binary SHA, model identities, canonical request, elapsed time, memory/RSS, CUDA peak VRAM và PNG integrity.
- Evidence packaging fail nếu có model weights hoặc known secret patterns, có internal SHA-256 manifest và exact `v1.0.0` archive basenames.
- Strict offline four-cell comparator chỉ tính ratio sau khi comparability gates pass; deterministic renderer sinh Markdown benchmark summary cho GitHub Release sau source freeze.
- Fresh measured 2×2 numbers được publish trong checksum-protected GitHub Release assets/body thay vì paste lại vào source sau qualification.
- Loopback REST API, CLI tooling, Kaggle adapter, publication-surface audit, public-history invariants và release-contract checks vẫn là một phần của release.

Các technical identities đóng băng trước final qualification:

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
DiT BF16 SHA256       = 6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE SHA256            = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
CPU sd-cli SHA256     = 7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c
CUDA sd-cli SHA256    = 3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0
canonical request     = seed 42, 4 steps, CFG 1.0, 4 threads
resolution matrix     = 512, 640, 768, 1024
```
