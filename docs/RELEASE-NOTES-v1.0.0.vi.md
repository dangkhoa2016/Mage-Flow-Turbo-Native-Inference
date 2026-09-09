# Ghi chú phát hành Mage-Flow-Turbo-Native-Inference v1.0.0

> 🌐 Ngôn ngữ / Language: **Tiếng Việt** | [English](RELEASE-NOTES-v1.0.0.md)

## Tổng quan

Mage-Flow-Turbo-Native-Inference là **bộ công cụ suy luận và triển khai native di động cho Mage-Flow-Turbo**. Python đảm nhận cấu hình, xác minh identity, điều phối CLI/REST, lifecycle, telemetry và thu thập evidence; việc thực thi model thật sự do native runtime `stable-diffusion.cpp` `sd-cli` thực hiện.

v1.0.0 là bản phát hành công khai đầu tiên của dòng này. Release hỗ trợ hai representation của Mage-Flow-Turbo diffusion nhưng cùng một execution engine native:

| Profile | Diffusion representation | Backends | Vai trò |
|---|---|---|---|
| `q8-reference` | GGUF Q8_0 | `cpu`, `cuda0` | canonical/default |
| `bf16-safetensors` | BF16 SafeTensors | `cpu`, `cuda0` | supported alternative |

Dự án **không** có Hugging Face Transformers/PyTorch inference backend. Cụm `PyTorch/Transformers` chỉ mô tả nguồn/layout phân phối artifact BF16 khi xuất hiện trong model provenance hoặc Kaggle mirror path.

## Strict v1.0.0 qualification matrix

Release public đầu tiên yêu cầu fresh exact-head evidence cho bốn cell:

1. `q8-reference` / CPU
2. `bf16-safetensors` / CPU
3. `q8-reference` / NVIDIA T4 `cuda0`
4. `bf16-safetensors` / NVIDIA T4 `cuda0`

Mỗi cell dùng một fresh Kaggle authority session riêng, chỉ attach đúng một Mage diffusion family, cùng final source HEAD/TREE và cùng ordered resolution matrix:

```text
512x512 → 640x640 → 768x768 → 1024x1024
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

Generation không được âm thầm retry bằng backend hoặc placement khác. Nếu resolution phía sau gặp giới hạn platform/runtime thật, evidence sẽ ghi lại giới hạn đó và combined report để các ratio không có đủ dữ liệu ở trạng thái không xác định.

## Policy CPU

- Kaggle `Accelerator=None`.
- Backend đúng `cpu`.
- Release qualification chỉ dùng prebuilt CPU `sd-cli`.
- Không CUDA fallback.
- Ghi host-memory và `sd-cli` RSS telemetry.
- BF16 CPU giữ visible-RAM và `MemAvailable` headroom gate tường minh.

## Policy T4/T4x2

- Physical host phải là NVIDIA T4 hoặc T4x2.
- Release qualification chỉ dùng physical GPU slot 0.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- `CUDA_VISIBLE_DEVICES=0`.
- Effective inference backend đúng `cuda0`.
- Không `cuda1`, không multi-GPU split, không `auto-fit`, không CPU inference fallback.
- Chỉ dùng prebuilt CUDA runtime.
- CUDA record thành công phải có GPU-memory telemetry dương.

Do đó T4x2 chỉ được chấp nhận như host configuration; v1.0.0 vẫn là strict single-T4 qualification contract.

## Điểm nổi bật

- JSON model manifest với fail-closed SHA-256 verification.
- Hai supported release profiles dùng cùng native `sd-cli` execution path.
- Backend-specific prebuilt CPU/CUDA runtime identities được đóng băng trước qualification.
- Ordered four-resolution matrix harness hỗ trợ `cpu|cuda0` và `q8-reference|bf16-safetensors`.
- Evidence ghi source HEAD và TREE, profile/backend identity, runtime identity, model identities, canonical request, elapsed time, RAM/RSS, CUDA VRAM và PNG integrity.
- Strict four-cell offline comparator không tính ratio nếu comparability gates fail.
- Deterministic Markdown renderer sinh release-facing benchmark table sau source freeze.
- Evidence packager dùng exact v1.0.0 naming, reject model weights, scan known secrets, kiểm tra tar path safety, internal SHA-256 manifest và external sidecar checksum.
- CLI `mageflow-native` và loopback REST API vẫn khả dụng cho local use/deployment.
- Kaggle-specific behavior vẫn nằm trong `integrations/kaggle/`; generic core không có hard `/kaggle/*` dependency.
- CI enforce source/publication/history/release contracts mà không commit model weights.

## Frozen technical identities

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
DiT Q8 SHA256         = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
DiT BF16 SHA256       = 6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
Qwen Q4_K_M SHA256    = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE SHA256            = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
CPU sd-cli SHA256     = 7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c
CUDA sd-cli SHA256    = 3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0
resolution matrix     = 512, 640, 768, 1024
seed / steps / CFG    = 42 / 4 / 1.0
threads               = 4
```

## Evidence và publication

Fresh measured values chỉ được tạo sau final source freeze. Frozen comparator kiểm tra bốn accepted aggregate evidence files và frozen renderer sinh release-facing benchmark Markdown. Nhờ vậy không cần sửa README/source docs sau qualification và không làm mất exact-head authority.

GitHub Release publish checksum-protected assets cho bốn qualification cells cùng combined comparison JSON/Markdown summary và final provenance/verification material. Không có model weights.

Measured latency, RSS/VRAM, PNG hashes, cell status và recorded platform limits được chấp nhận sẽ authoritative trong GitHub Release assets/body; source documentation định nghĩa immutable protocol và interpretation rules.

## Ngoài phạm vi qualification v1.0.0

- multi-GPU `cuda0&cuda1` qualification;
- P100, TPU, Vulkan, Metal, ROCm hoặc SYCL qualification;
- Hugging Face Transformers hoặc PyTorch inference;
- model training/fine-tuning;
- automatic model-family conversion;
- tuyên bố BF16 luôn có chất lượng cao hơn Q8.
