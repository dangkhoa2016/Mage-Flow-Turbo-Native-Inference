# Mage-Flow-Turbo-Native-Inference

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![Native Runtime](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/releases/tag/v1.0.0)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: [English](README.md) | **Tiếng Việt**

**Bộ công cụ suy luận và triển khai native di động cho Mage-Flow-Turbo.** Repository không huấn luyện hoặc sửa trọng số model. Python đảm nhận cấu hình, xác minh danh tính model/runtime, điều phối CLI/REST, lifecycle, telemetry và evidence; phần thực thi model thật sự đi qua runtime native `stable-diffusion.cpp` `sd-cli`.

Tên dự án mô tả **execution stack**, không mô tả một quantization hoặc serialization duy nhất. v1.0.0 hỗ trợ profile Q8 GGUF canonical và profile BF16 SafeTensors, cả hai dùng cùng native runtime.

## Các profile model của v1.0.0

| Profile | Mage-Flow diffusion | Text encoder | VAE | Native runtime | CPU | CUDA `cuda0` | Vai trò |
|---|---|---|---|---|---:|---:|---|
| `q8-reference` | GGUF `Q8_0` | Qwen3-VL-4B GGUF `Q4_K_M` | SafeTensors | `stable-diffusion.cpp` `sd-cli` đã pin | có | có | canonical/default |
| `bf16-safetensors` | BF16 SafeTensors | Qwen3-VL-4B GGUF `Q4_K_M` | SafeTensors | `stable-diffusion.cpp` `sd-cli` đã pin | có | có | profile thay thế được hỗ trợ |

Dự án **không** cung cấp Hugging Face Transformers inference backend và **không** chạy vòng lặp suy luận bằng PyTorch/Transformers. Cụm `PyTorch/Transformers` trong provenance hoặc đường dẫn model mirror chỉ nói về nguồn/layout phân phối artifact BF16 SafeTensors, không phải framework dùng để inference.

Dự án cũng không phải “GGUF-only”: ngay profile Q8 canonical vẫn dùng VAE SafeTensors.

## Danh tính model/runtime đóng băng

| Vai trò | Artifact / danh tính | Định dạng |
|---|---|---|
| Q8 diffusion | `Mage-Flow-Turbo-DiT-Q8_0.gguf` | GGUF Q8_0 |
| BF16 diffusion | `diffusion_pytorch_model.safetensors` từ Mage-Flow `PyTorch / default` | BF16 SafeTensors |
| Text encoder | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | GGUF Q4_K_M |
| VAE | `diffusion_pytorch_model.safetensors` | SafeTensors |
| Native runtime | `stable-diffusion.cpp` `sd-cli` | commit đã pin `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba` |

Exact SHA-256 được kiểm tra trước real inference. Git repository không chứa model weights.

## Strict 2×2 qualification matrix cho v1.0.0

Release public đầu tiên được qualification bằng bốn fresh cell:

| Profile | Kaggle CPU | Kaggle T4/T4x2 `cuda0` |
|---|---:|---:|
| `q8-reference` | fresh exact-head matrix | fresh exact-head matrix |
| `bf16-safetensors` | fresh exact-head matrix | fresh exact-head matrix |

Mỗi cell dùng cùng frozen source HEAD/TREE và cùng canonical protocol:

```text
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
matrix  = 512 → 640 → 768 → 1024
```

Mỗi resolution được generate đúng một lần trong authority session. Matrix chạy tuần tự và fail-fast. Nếu một resolution phía sau gặp giới hạn platform/runtime thật, evidence sẽ ghi nhận giới hạn đó thay vì âm thầm đổi placement hoặc bịa ratio.

### Policy CPU

- Kaggle `Accelerator=None`;
- backend chính xác `cpu`;
- chỉ dùng prebuilt CPU `sd-cli`;
- không CUDA fallback;
- ghi host-memory và process-RSS telemetry;
- BF16 CPU giữ các gate RAM/headroom riêng.

### Policy T4/T4x2

- host phải là T4 hoặc T4x2;
- release qualification chỉ dùng physical GPU slot 0;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- effective inference backend là `cuda0`;
- không `cuda1`, không multi-GPU split, không `auto-fit`, không CPU inference fallback;
- chỉ dùng prebuilt CUDA runtime;
- generation CUDA thành công phải có VRAM telemetry dương.

Vì vậy T4x2 chỉ là host configuration; v1.0.0 vẫn là strict single-T4 benchmark.

## Chính sách publish fresh benchmark

Fresh strict 2×2 measurements được tạo **sau final source freeze**. Comparator đã đóng băng sẽ kiểm tra source HEAD/TREE, canonical request, model identities, runtime commit và runtime SHA theo backend trước khi tính ratio. Renderer đã đóng băng sau đó sinh bảng Markdown dùng cho GitHub Release.

Số liệu cuối cùng được publish thành GitHub Release assets/body có checksum thay vì paste ngược vào source sau qualification. Cách này tránh làm mất exact-head authority chỉ vì sửa README sau benchmark.

Xem [benchmark contract](docs/BENCHMARKS-v1.0.0.vi.md).

## Vì sao dùng native inference?

Diffusion, text conditioning và VAE decoding do `sd-cli` thực hiện. Python xác minh identity, dựng subprocess argument tường minh với `shell=False`, giám sát native process, kiểm tra PNG và ghi structured evidence.

Hai profile dùng chung một execution engine, nhờ đó so sánh Q8/BF16 × CPU/CUDA dễ audit hơn đáng kể.

## Xác minh Q8 reference stack

```bash
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

## Phát triển local Linux

Generic CLI có thể build local runtime khi chủ động phát triển ngoài release qualification:

```bash
python -m pip install -e .
mageflow-native runtime build --backend cpu
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

Riêng release qualification là **prebuilt-runtime only**.

## Phát triển NVIDIA CUDA

```bash
python -m pip install -e .
mageflow-native runtime build --backend cuda
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json --backend cuda0
```

Release qualification dùng placement `cuda0` xác định trước, không dùng automatic splitting.

## REST API

Reference service bind vào `127.0.0.1` theo mặc định.

```text
GET  /healthz
GET  /readyz
GET  /v1/info
POST /v1/images/generate
GET  /v1/artifacts/<png>
```

## Kaggle integration

Notebook public [notebooks/kaggle-production-demo.ipynb](notebooks/kaggle-production-demo.ipynb) detect accelerator Kaggle được hỗ trợ. Với release qualification, dùng dedicated matrix harness và exact prebuilt runtime/profile inputs thay vì dựa vào notebook defaults. Xem [docs/kaggle.vi.md](docs/kaggle.vi.md).

### Policy attach model

Trong một qualification/inference session thông thường, chỉ attach đúng một Mage-Flow-Turbo diffusion family:

- `q8-reference` — Mage-Flow `GGUF / q8-0`, Qwen GGUF dùng chung và VAE-only SafeTensors;
- `bf16-safetensors` — Mage-Flow `PyTorch / default` BF16 transformer/VAE cùng Qwen GGUF dùng chung.

Không attach đồng thời hai Mage diffusion families trong authority session thông thường. Mixed-family attachment chỉ dành cho research tooling có kiểm soát và không thuộc bốn fresh release qualification cells.

## Historical BF16 CPU visual research

Trước redesign strict 2×2, dự án đã thực hiện một paired visual study 768×768 trên CPU cùng host để so Q8 và BF16. Nghiên cứu lịch sử đó vẫn hữu ích cho hướng chất lượng, nhưng không phải final v1.0.0 2×2 performance authority và không quyết định default profile. Q8 vẫn là canonical/default.

Xem [BF16 SafeTensors background và qualification policy](docs/BF16-HIGH-MEMORY-CPU.vi.md).

## Tái lập và evidence

Output CPU và CUDA có thể khác byte-for-byte do numerical backend khác nhau. Release evidence ghi:

- exact source HEAD và TREE;
- profile/backend identity;
- tên/format/SHA-256 của các model component;
- pinned native runtime commit và binary SHA-256;
- canonical request và resolution;
- elapsed time;
- host memory / process RSS;
- CUDA peak VRAM khi áp dụng;
- PNG filename, dimensions, byte count và SHA-256;
- failure classification tường minh khi matrix dừng.

Evidence archives có checksum, internal manifest, và fail nếu chứa model weights hoặc known secret patterns.

## Tài liệu

- [Kiến trúc](docs/architecture.md)
- [Model stack](docs/model-stack.md)
- [Linux cục bộ](docs/local-linux.md)
- [CUDA](docs/cuda.md)
- [Kaggle](docs/kaggle.vi.md)
- [Strict v1.0.0 benchmark contract](docs/BENCHMARKS-v1.0.0.vi.md)
- [BF16 SafeTensors profile](docs/BF16-HIGH-MEMORY-CPU.vi.md)
- [REST API](docs/REST-API.md)
- [Kiểm thử](docs/TESTING.vi.md)
- [Xử lý sự cố](docs/TROUBLESHOOTING.vi.md)
- [Đóng góp](.github/CONTRIBUTING.md)
- [Chính sách bảo mật](.github/SECURITY.md)

## Giấy phép

MIT License. Copyright © 2026 Đăng Khoa <i.am@dangkhoa.dev>.
