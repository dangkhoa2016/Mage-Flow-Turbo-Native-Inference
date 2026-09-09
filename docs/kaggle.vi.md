# Production notebook Kaggle

> 🌐 Language / Ngôn ngữ: [English](kaggle.md) | **Tiếng Việt**

Notebook public là đường reproduction/demo. Fresh public Q8/T4 notebook smoke
là supplemental reproduction evidence và không thay thế retained strict 2×2
benchmark authority.

## Onboarding Kaggle chính xác: Input → Add Input

1. Trong Kaggle editor, mở **Session options → Accelerator**. Chọn `None` cho
   CPU, hoặc NVIDIA T4/T4x2 cho CUDA. Không dùng P100, TPU hay accelerator khác.
2. Mở **Input → Add Input**. Các runtime dưới đây là **Kaggle Datasets**; chỉ
   attach đúng một dataset khớp accelerator đã chọn:

   | Accelerator | Dataset slug |
   |---|---|
   | `None` / CPU | `dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime` |
   | T4 hoặc T4x2 | `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime` |

3. Trong **Input → Add Input**, chọn **Kaggle Models** (không phải Datasets)
   cho model inputs. Mage-Flow là
   `dangkhoa2016/mage-flow-community-mage-flow-turbo`; Qwen là
   `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf`.

Với `q8-reference`, attach Mage-Flow **GGUF / q8-0** và **PyTorch / vae-only**,
cùng Qwen **GGUF / q4-k-m**. Với `bf16-safetensors`, chỉ attach Mage-Flow
**PyTorch / default** cùng Qwen **GGUF / q4-k-m**. Mage-Flow **PyTorch /
default** đã có cả BF16 diffusion model và VAE: không thêm **GGUF / q8-0** hoặc
**PyTorch / vae-only** vào BF16 session. Không được trộn Q8 và BF16 Mage-Flow
attachment families trong normal session.

### Checklist recommended fresh BF16 T4x2 reproduction

Trước khi chạy, đặt configuration cell thành:

```python
RUN_MODE = "experiment"
MODEL_PROFILE = "bf16-safetensors"
RESOLUTION_PRESET = "auto"
RUN_FAIR_COMPARISON_BENCHMARK = False

ALLOW_SOURCE_BUILD = False

ENABLE_REST_DEMO = True
RUN_REST_GENERATION = True
ENABLE_QUICK_TUNNEL = False
I_UNDERSTAND_QUICK_TUNNEL_IS_PUBLIC = False
```

Trạng thái cuối của **Input → Add Input** phải là:

- ATTACH Dataset: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime`
- ATTACH Model: `dangkhoa2016/mage-flow-community-mage-flow-turbo` →
  **PyTorch / default**
- ATTACH Model: `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf` →
  **GGUF / q4-k-m**
- DO NOT ATTACH Mage-Flow **GGUF / q8-0**, Mage-Flow **PyTorch / vae-only**,
  hoặc portable CPU runtime dataset.

Trên T4x2, marker mong đợi là `ACCELERATOR_DETECTED=nvidia-t4x2`,
`ACCELERATOR_POLICY=PASS`, `BACKEND_AUTO_SELECTED=cuda0`, và
`GPU1_NOT_USED=PASS`. Khi checklist hoàn tất, chọn **Run → Run All**.

## Policy accelerator và runtime

- `Accelerator=None` chọn backend `cpu` và prebuilt CPU runtime.
- NVIDIA T4 hoặc T4x2 chọn backend `cuda0` và prebuilt CUDA T4 runtime. CUDA
  authority chỉ dùng physical slot 0 với `CUDA_DEVICE_ORDER=PCI_BUS_ID` và
  `CUDA_VISIBLE_DEVICES=0`.
- P100, TPU, mixed GPU và accelerator khác đều fail trước discovery.

Prebuilt CPU và CUDA T4 runtime là **Kaggle Dataset**. Chỉ attach một runtime
dataset khớp accelerator; notebook không build `stable-diffusion.cpp` từ source.

## Model profile

Mage-Flow, Qwen và VAE là **Kaggle Model** inputs. Chọn đúng một profile, độc
lập với backend CPU/T4 được detect tự động.

| Profile | Model attachments bắt buộc |
|---|---|
| `q8-reference` | Mage-Flow `GGUF / q8-0`, Qwen `GGUF / q4-k-m`, và Mage-Flow `PyTorch / vae-only` |
| `bf16-safetensors` | Mage-Flow `PyTorch / default` và Qwen `GGUF / q4-k-m`; default đã gồm VAE nên không attach `PyTorch / vae-only` riêng |

Normal manifest discovery fail closed nếu attach cả hai Mage-Flow diffusion
families. Mixed-family tooling chỉ dành cho research có kiểm soát và không
được dùng cho fresh release qualification cell.

## Mặc định public

```python
RUN_MODE = "experiment"
MODEL_PROFILE = "q8-reference"
RESOLUTION_PRESET = "auto"
RUN_FAIR_COMPARISON_BENCHMARK = False
```

Mỗi experiment **Run All** có local state riêng. Reviewer có thể opt in matrix,
và xem `qualification_matrix`, nhưng đây là inspection surface chứ không phải
hướng dẫn rerun retained four-cell authority chỉ vì public documentation đổi.

## Retained qualification và public reproduction

Canonical measurements vẫn gắn với measured benchmark evidence source HEAD/TREE
được ghi trong retained evidence artifacts. Final publication source chứa public
notebook/documentation/contract-test corrections. Checksum-protected
qualification-equivalence manifest bridge các identity chỉ khi
qualification-critical Git objects byte-identical; measured evidence provenance
không bị rewrite.

1. `q8-reference` / `cpu`
2. `bf16-safetensors` / `cpu`
3. `q8-reference` / `cuda0`
4. `bf16-safetensors` / `cuda0`

Retained records dùng recorded source identity, profile đã chọn, pinned runtime
theo backend và matrix `512 → 640 → 768 → 1024`. Q8/CPU giữ documented
same-session full-reset recovery exception; equivalence không relabel run này
thành fresh-session evidence. BF16 CPU giữ 27 GiB / 3 GiB safety gates; gate
này không áp dụng cho BF16 CUDA. Canonical measurements và evidence digest chỉ
được publish trong GitHub Release assets/body có checksum.
