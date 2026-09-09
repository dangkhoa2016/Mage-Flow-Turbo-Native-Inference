# Mage-Flow-Turbo-Native-Inference

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![Native Runtime](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/native-runtime.yml)
[![Release](https://img.shields.io/github/v/release/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/releases/tag/v1.0.0)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](README.vi.md)

A **portable native inference and deployment stack for Mage-Flow-Turbo**. The repository does not train or modify model weights. Python provides configuration, model/runtime identity verification, CLI/REST orchestration, lifecycle control, telemetry and evidence collection; the actual model execution path is the native `stable-diffusion.cpp` `sd-cli` runtime.

The project name describes the execution stack, not a single quantization or serialization format. v1.0.0 supports a canonical Q8 GGUF profile and a BF16 SafeTensors profile, both through the same native runtime.

## v1.0.0 model profiles

| Profile | Mage-Flow diffusion | Text encoder | VAE | Native runtime | CPU | CUDA `cuda0` | Role |
|---|---|---|---|---|---:|---:|---|
| `q8-reference` | GGUF `Q8_0` | Qwen3-VL-4B GGUF `Q4_K_M` | SafeTensors | pinned `stable-diffusion.cpp` `sd-cli` | yes | yes | canonical/default |
| `bf16-safetensors` | BF16 SafeTensors | Qwen3-VL-4B GGUF `Q4_K_M` | SafeTensors | pinned `stable-diffusion.cpp` `sd-cli` | yes | yes | supported alternative |

The project does **not** provide a Hugging Face Transformers inference backend and does **not** run a PyTorch/Transformers inference loop. `PyTorch/Transformers` wording in provenance or model-mirror paths refers to the source artifact/distribution layout of the BF16 SafeTensors weights, not the execution framework.

The project is also intentionally not described as GGUF-only: even the canonical Q8 profile uses a SafeTensors VAE.

## Frozen model/runtime identities

| Role | Artifact / identity | Format |
|---|---|---|
| Q8 diffusion | `Mage-Flow-Turbo-DiT-Q8_0.gguf` | GGUF Q8_0 |
| BF16 diffusion | `diffusion_pytorch_model.safetensors` from Mage-Flow `PyTorch / default` | BF16 SafeTensors |
| Text encoder | `Qwen3VL-4B-Instruct-Q4_K_M.gguf` | GGUF Q4_K_M |
| VAE | `diffusion_pytorch_model.safetensors` | SafeTensors |
| Native runtime | `stable-diffusion.cpp` `sd-cli` | pinned commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba` |

Exact SHA-256 identities are verified before real inference. The Git repository contains no model weights.

## Strict 2×2 v1.0.0 qualification matrix

The first public release is qualified across four fresh cells:

| Profile | Kaggle CPU | Kaggle T4/T4x2 `cuda0` |
|---|---:|---:|
| `q8-reference` | fresh exact-head matrix | fresh exact-head matrix |
| `bf16-safetensors` | fresh exact-head matrix | fresh exact-head matrix |

Every cell uses the same frozen source HEAD/TREE and the same canonical protocol:

```text
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
matrix  = 512 → 640 → 768 → 1024
```

Each resolution is generated exactly once per authority session. The matrix is sequential and fail-fast. If a genuine later-resolution platform/runtime limit occurs, evidence records it rather than silently changing placement or inventing a ratio.

### CPU policy

- Kaggle `Accelerator=None`;
- backend exactly `cpu`;
- prebuilt CPU `sd-cli` only;
- no CUDA fallback;
- host-memory and process-RSS telemetry;
- the BF16 CPU profile retains its explicit high-memory/headroom safety gates.

### T4/T4x2 policy

- host must be T4 or T4x2;
- release qualification uses only physical GPU slot 0;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- effective inference backend `cuda0`;
- no `cuda1`, no multi-GPU split, no `auto-fit`, no CPU inference fallback;
- prebuilt CUDA runtime only;
- successful CUDA generations require positive VRAM telemetry.

A T4x2 host is therefore allowed as a host configuration, but v1.0.0 qualification remains a strict single-T4 benchmark.

## Fresh benchmark publication policy

Fresh strict 2×2 measurements are generated **after the final source freeze**. The frozen comparator verifies source HEAD/TREE, canonical request, model identities, runtime commit and backend-specific runtime SHA values before calculating ratios. A frozen renderer then produces the release-facing Markdown table.

The final measured numbers are published as checksum-protected GitHub Release assets/body rather than pasted back into source after qualification. This avoids invalidating exact-head evidence by editing README after the benchmark has run.

See [the benchmark contract](docs/BENCHMARKS-v1.0.0.md).

## Why native inference?

Diffusion execution, text conditioning and VAE decoding are performed by `sd-cli`. Python validates identities, constructs explicit subprocess arguments with `shell=False`, monitors the native process, validates PNG artifacts and records structured evidence.

This architecture gives both model profiles one common runtime path, which makes the Q8/BF16 × CPU/CUDA comparison substantially easier to audit.

## Verify the Q8 reference stack

```bash
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

## Local Linux development

The generic CLI may build a local runtime when deliberately developing outside release qualification:

```bash
python -m pip install -e .
mageflow-native runtime build --backend cpu
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json
mageflow-native verify --manifest configs/mage-flow-turbo-q8-reference.json
```

Release qualification itself is **prebuilt-runtime only**.

## NVIDIA CUDA development

```bash
python -m pip install -e .
mageflow-native runtime build --backend cuda
mageflow-native doctor --manifest configs/mage-flow-turbo-q8-reference.json --backend cuda0
```

Release qualification uses deterministic `cuda0` placement rather than automatic splitting.

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

The public notebook [notebooks/kaggle-production-demo.ipynb](notebooks/kaggle-production-demo.ipynb) detects supported Kaggle accelerators. For release qualification, use the dedicated matrix harness and exact prebuilt runtime/profile inputs rather than relying on notebook defaults. See [docs/kaggle.md](docs/kaggle.md).

### Model-attachment policy

For a normal qualification/inference session, attach exactly one Mage-Flow-Turbo diffusion family:

- `q8-reference` — Mage-Flow `GGUF / q8-0`, shared Qwen GGUF and VAE-only SafeTensors;
- `bf16-safetensors` — Mage-Flow `PyTorch / default` BF16 transformer/VAE plus the shared Qwen GGUF text encoder.

Do not attach both Mage diffusion families in an ordinary authority session. Mixed-family attachment is reserved for explicitly controlled research tooling and is not part of the four fresh release qualification cells.

## Historical BF16 CPU visual research

Before the strict 2×2 release redesign, a same-host paired 768×768 CPU visual study compared the Q8 and BF16 representations. That historical study remains useful as quality-oriented research, but it is not the final v1.0.0 2×2 performance authority and does not determine the default profile. Q8 remains the canonical/default profile.

See [BF16 SafeTensors background and qualification policy](docs/BF16-SAFETENSORS.md).

## Reproducibility and evidence

CPU and CUDA outputs may legitimately differ byte-for-byte across numerical backends. Release evidence records:

- exact source HEAD and TREE;
- profile/backend identity;
- model component names/formats/SHA-256 values;
- pinned native runtime commit and binary SHA-256;
- canonical request and resolution;
- elapsed time;
- host memory / process RSS;
- CUDA peak VRAM when applicable;
- PNG filename, dimensions, byte count and SHA-256;
- explicit failure classification when a matrix stops.

Evidence archives are checksum-protected, contain internal manifests, and reject model weights and known secret patterns.

## Documentation

- [Architecture](docs/architecture.md)
- [Model stack](docs/model-stack.md)
- [Local Linux](docs/local-linux.md)
- [CUDA](docs/cuda.md)
- [Kaggle](docs/kaggle.md)
- [Strict v1.0.0 benchmark contract](docs/BENCHMARKS-v1.0.0.md)
- [BF16 SafeTensors profile](docs/BF16-SAFETENSORS.md)
- [REST API](docs/REST-API.md)
- [Testing](docs/TESTING.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Contributing](.github/CONTRIBUTING.md)
- [Security policy](.github/SECURITY.md)

## License

MIT License. Copyright © 2026 Đăng Khoa <i.am@dangkhoa.dev>.
