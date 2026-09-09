# BF16 SafeTensors Release Profile và nghiên cứu CPU lịch sử

> 🌐 Language / Ngôn ngữ: [English](BF16-HIGH-MEMORY-CPU.md) | **Tiếng Việt**

`bf16-safetensors` là profile BF16 được hỗ trợ cho Mage-Flow-Turbo-Native-Inference v1.0.0. Profile này dùng cùng native execution engine `stable-diffusion.cpp` `sd-cli` như `q8-reference`; nó **không** bổ sung Hugging Face Transformers hoặc PyTorch inference backend.

Tên file tài liệu này được giữ lại để không làm gãy các link hiện có. Release contract CPU-only cũ đã bị supersede bởi strict v1.0.0 2×2 profile/backend matrix.

## Contract profile

`bf16-safetensors` dùng Mage-Flow-Turbo `PyTorch / default` mirror làm source distribution cho BF16 transformer và VAE:

```text
mage-flow-community-mage-flow-turbo / PyTorch / default
```

Các component bắt buộc:

- Mage-Flow-Turbo transformer: BF16 SafeTensors, SHA-256 `6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d`.
- Qwen3-VL-4B text encoder: GGUF `Q4_K_M`, SHA-256 `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a`.
- Mage VAE: SafeTensors, SHA-256 `34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0`.
- Native runtime: pinned `stable-diffusion.cpp` commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`.
- Supported release backends: `cpu`, `cuda0`.

`PyTorch / default` chỉ mô tả upstream/mirror artifact layout. Inference vẫn chạy bằng native `sd-cli`.

## Policy CPU

BF16 CPU qualification giữ các high-memory safety gates bảo thủ đã được thiết lập từ nghiên cứu trước:

- visible host RAM tối thiểu: 27 GiB;
- `MemAvailable` headroom quan sát trong canonical run thành công: tối thiểu 3 GiB;
- backend đúng `cpu`;
- prebuilt CPU runtime SHA-256 `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`;
- không accelerator fallback.

Đây là release-safety gates cho qualification workflow đã test, không phải tuyên bố phổ quát rằng mọi BF16 use case đều cần đúng 27 GiB.

## Policy CUDA0

BF16 CUDA qualification là strict single-T4 path:

- Kaggle T4 hoặc T4x2 host;
- chỉ physical GPU slot 0;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- effective backend `cuda0`;
- không `cuda1`, multi-GPU split, `auto-fit`, hoặc CPU inference fallback;
- prebuilt CUDA runtime SHA-256 `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`;
- generation thành công phải có VRAM telemetry dương.

CPU-only gate 27 GiB / 3 GiB không được dùng để reject CUDA backend. Host memory vẫn được ghi, nhưng CUDA feasibility được quyết định bởi native CUDA execution tường minh, artifact hợp lệ và VRAM evidence.

## Strict release matrix

Cho cả CPU và CUDA0, authority matrix chạy theo thứ tự:

```text
512x512 → 640x640 → 768x768 → 1024x1024
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

CPU command:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu \
  --profile bf16-safetensors \
  --input-root /kaggle/input \
  --work-root /kaggle/working/v1-bf16-cpu \
  --repo-dir "$PWD" \
  --resolutions 512,640,768,1024
```

CUDA command trong fresh T4/T4x2 session sau khi set `CUDA_VISIBLE_DEVICES=0`:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cuda0 \
  --profile bf16-safetensors \
  --input-root /kaggle/input \
  --work-root /kaggle/working/v1-bf16-t4 \
  --repo-dir "$PWD" \
  --resolutions 512,640,768,1024
```

Harness chạy tuần tự và fail-fast. Canonical 512 là feasibility gate đầu tiên. Nếu resolution phía sau gặp giới hạn resource/runtime thật, evidence được giữ nguyên; qualification path không được đổi sang CPU offload, multi-GPU, auto-fit hoặc backend khác chỉ để ép kết quả thành PASS.

## Semantics của release evidence

Mỗi matrix ghi source HEAD/TREE, exact model/runtime identities, profile/backend, canonical request, elapsed time, host memory/RSS, CUDA peak VRAM khi áp dụng, PNG identity và failure classification tường minh.

Bốn release cells được independent review rồi mới đưa vào frozen four-cell comparator. Ratio chỉ được tính khi các record liên quan thành công và tất cả identity/comparability gates đều PASS.

Q8 vẫn là canonical/default profile. Việc hỗ trợ BF16 như một release profile không có nghĩa dự án tuyên bố BF16 luôn có chất lượng cao hơn hoặc hiệu quả hơn Q8.

## Historical pre-release CPU visual research

Trước strict 2×2 redesign, dự án đã có một same-host paired 768×768 CPU visual study giữa Q8 và BF16:

- 10 prompt × 2 representations = 20 runs;
- 4 steps, CFG 1.0, 4 CPU threads;
- 20/20 runs thành công;
- Q8 mean elapsed ≈ 640,0 s/hình;
- BF16 mean elapsed ≈ 1013,9 s/hình;
- Q8 peak `sd-cli` RSS ≈ 8,89 GB;
- BF16 peak `sd-cli` RSS ≈ 12,54 GB;
- blind visual result: BF16 4 win, Q8 3 win, 3 tie, chỉ một BF16 win rõ ràng về mặt vật chất.

Nghiên cứu đó vẫn hữu ích cho quality-oriented research nhưng **không** phải final v1.0.0 strict 2×2 performance authority, không dùng final redesigned source HEAD và không được thay thế bất kỳ fresh release qualification cell nào.

Fresh final performance numbers chỉ được tạo sau new source freeze và được publish trong checksum-protected GitHub Release assets/body thay vì sửa ngược vào source documentation.
