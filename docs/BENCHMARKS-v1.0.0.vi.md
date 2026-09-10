# Strict v1.0.0 2×2 Native Benchmark Contract

> 🌐 Language / Ngôn ngữ: [English](BENCHMARKS-v1.0.0.md) | **Tiếng Việt**

Tài liệu này định nghĩa benchmark và interpretation contract cho release public đầu tiên **Mage-Flow-Turbo-Native-Inference v1.0.0**. Retained measurements vẫn gắn với measured benchmark evidence source HEAD/TREE đã ghi trong checksum-protected GitHub Release evidence. Final publication source chỉ được bridge bằng qualification-equivalence manifest khi các qualification-critical Git objects là byte-identical.

## Bốn release cells

| Cell | Profile | Backend | Authority environment |
|---|---|---|---|
| Q8 CPU | `q8-reference` | `cpu` | retained measured evidence; documented same-session full-reset recovery |
| BF16 CPU | `bf16-safetensors` | `cpu` | retained measured evidence |
| Q8 T4 | `q8-reference` | `cuda0` | retained measured evidence, chỉ physical slot 0 |
| BF16 T4 | `bf16-safetensors` | `cuda0` | retained measured evidence, chỉ physical slot 0 |

Các retained cells dùng source HEAD/TREE đã ghi trong evidence và cùng canonical request. Final publication source chỉ có thể khác ở public-demo, documentation và contract-test surfaces được phép; equivalence manifest không rewrite evidence provenance.

## Frozen common protocol

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
prompt               = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed                 = 42
steps                = 4
CFG                  = 1.0
threads              = 4
resolution order     = 512 → 640 → 768 → 1024
generation count     = đúng 1 lần mỗi resolution
```

Model/runtime identities:

```text
Q8 diffusion SHA256   = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
BF16 diffusion SHA256 = 6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
Qwen SHA256           = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE SHA256            = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
CPU sd-cli SHA256     = 7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c
CUDA sd-cli SHA256    = 3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0
```

Cả hai profile inference bằng native `stable-diffusion.cpp` `sd-cli`. BF16 SafeTensors support không phải Hugging Face Transformers inference backend.

## Backend fairness

### CPU

- Kaggle `Accelerator=None`;
- backend đúng `cpu`;
- chỉ dùng prebuilt CPU runtime;
- không CUDA fallback;
- ghi host memory và peak `sd-cli` RSS;
- BF16 CPU giữ release-safety RAM/headroom gates.

### T4 / T4x2

- physical host GPUs phải là NVIDIA T4;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- inference backend đúng `cuda0`;
- không `cuda1`, multi-GPU split, `auto-fit`, hoặc CPU inference fallback;
- chỉ dùng prebuilt CUDA runtime;
- successful record phải có positive GPU peak telemetry.

T4x2 chỉ được chấp nhận như host configuration; v1.0.0 qualification là single-T4 comparison.

## Fail-fast và partial-result semantics

Matrix bắt đầu ở 512 và chạy tuần tự. Setup hoặc 512 fail thì không chạy resolution sau. Giới hạn resource/runtime thật ở resolution phía sau sẽ dừng matrix và được giữ nguyên trong evidence.

Identity mismatch, mixed-family attachment, source mismatch, fallback, thiếu required telemetry, corrupted evidence hoặc policy violation là hard failure. Chúng không được reclassify thành platform limit có thể chấp nhận.

Later-resolution limit chỉ có thể được independent review và phân loại `PARTIAL_WITH_RECORDED_LIMIT` khi canonical 512 đã PASS. Release decision cho partial cell là gate riêng.

## Strict comparison rules

Frozen four-cell comparator chỉ tính performance sau khi verify:

- cùng measured evidence source HEAD/TREE ở cả bốn cells;
- cùng pinned upstream runtime commit;
- backend-specific runtime SHA-256;
- exact Q8/BF16 diffusion identities;
- cùng Qwen và VAE identities;
- cùng ordered resolution list;
- cùng canonical prompt, seed, steps, CFG và threads;
- CPU cells thực sự dùng CPU;
- CUDA cells thực sự dùng `cuda0` với `CUDA_VISIBLE_DEVICES=0`;
- successful CUDA records có positive GPU peak telemetry.

Ở mỗi resolution có đủ successful records, comparator báo:

1. Q8 CPU → T4 speedup = `Q8 CPU elapsed / Q8 T4 elapsed`.
2. BF16 CPU → T4 speedup = `BF16 CPU elapsed / BF16 T4 elapsed`.
3. CPU BF16/Q8 latency ratio = `BF16 CPU elapsed / Q8 CPU elapsed`.
4. T4 BF16/Q8 latency ratio = `BF16 T4 elapsed / Q8 T4 elapsed`.
5. Chênh peak RSS trên CPU.
6. Chênh peak VRAM trên T4.

Nếu record cần thiết thiếu hoặc fail, ratio tương ứng là `N/A`/`null`; không suy diễn giá trị.

## Release-facing outputs

Release mang đúng các retained checksum-protected assets sau:

- `mage-flow-turbo-native-inference-v1.0.0-q8-reference-cpu-same-session-recovery-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-q8-reference-cuda0-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cpu-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cuda0-evidence.tar.gz`
- `mage-flow-v1.0.0-strict-2x2-comparison.json`
- `mage-flow-v1.0.0-strict-2x2-release-summary.md`
- `SHA256SUMS.txt`

Frozen comparator tạo JSON và frozen renderer tạo summary. GitHub Release body
được lấy từ generated summary; source documentation không được sửa sau
qualification chỉ để paste số liệu động.

## Historical pre-redesign measurements

Trước strict 2×2 redesign, dự án đã hoàn tất một Q8-only fresh CPU/T4 matrix trên pre-release source state cũ:

| Resolution | Historical Q8 CPU | Historical Q8 T4 |
|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s |
| 640×640 | 338.014 s | 8.698 s |
| 768×768 | 491.700 s | 9.710 s |
| 1024×1024 | 939.371 s | 12.420 s |

Những số này vẫn hữu ích cho pre-release research/regression context nhưng không thay thế retained strict 2×2 authority hoặc recorded evidence provenance của nó.

Historical T4 1024 PNG SHA-256 `2c82bdb7cd68746c113eea0f95593d4860b3a201eb3869d0c384221b39b0e49e` cũng chỉ là historical evidence, không phải reference hash cho retained 512 hoặc BF16 artifact.
