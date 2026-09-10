# Mage-Flow-Turbo-Native-Inference v1.0.0 Release Notes

[![CI](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml/badge.svg)](https://github.com/dangkhoa2016/Mage-Flow-Turbo-Native-Inference/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/dangkhoa2016/Mage-Flow-Turbo-Native-Inference)](../LICENSE)

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](RELEASE-NOTES-v1.0.0.vi.md)

## Overview

Mage-Flow-Turbo-Native-Inference is a **portable native inference and deployment stack for Mage-Flow-Turbo**. Python provides configuration, identity verification, CLI/REST orchestration, lifecycle control, telemetry and evidence collection; actual model execution is performed by the native `stable-diffusion.cpp` `sd-cli` runtime.

v1.0.0 is the first public release of this lineage. It supports two Mage-Flow-Turbo diffusion representations through one native execution engine:

| Profile | Diffusion representation | Backends | Role |
|---|---|---|---|
| `q8-reference` | GGUF Q8_0 | `cpu`, `cuda0` | canonical/default |
| `bf16-safetensors` | BF16 SafeTensors | `cpu`, `cuda0` | supported alternative |

The project does **not** provide a Hugging Face Transformers/PyTorch inference backend. `PyTorch/Transformers` wording refers only to the BF16 source artifact/distribution layout when it appears in model provenance or Kaggle mirror paths.

## Strict v1.0.0 qualification matrix

The retained strict 2×2 measurements are bound to the measured benchmark evidence source recorded in their checksum-protected artifacts. The final publication source is bridged by a qualification-equivalence manifest when qualification-critical Git objects are byte-identical.

1. `q8-reference` / CPU
2. `bf16-safetensors` / CPU
3. `q8-reference` / NVIDIA T4 `cuda0`
4. `bf16-safetensors` / NVIDIA T4 `cuda0`

Each retained record uses one selected Mage diffusion family, its recorded measured-evidence source HEAD/TREE and the same ordered resolution matrix. Q8/CPU retains its documented same-session full-reset recovery exception; it is not relabeled as a fresh session.

```text
512x512 → 640x640 → 768x768 → 1024x1024
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

A generation is never silently retried under a different backend or placement. If a later resolution hits a genuine platform/runtime limit, the evidence records that limit and the combined report leaves unavailable ratios unset.

## CPU policy

- Kaggle `Accelerator=None`.
- Backend exactly `cpu`.
- Prebuilt CPU `sd-cli` only for release qualification.
- No CUDA fallback.
- Host-memory and `sd-cli` RSS telemetry are recorded.
- BF16 CPU retains an explicit visible-RAM and `MemAvailable` headroom gate.

## Kaggle public reproduction inputs

Use **Session options → Accelerator**, then **Input → Add Input**. Runtime
entries are Kaggle Datasets: use
`dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime` for CPU or
`dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime` for T4/T4x2.
Mage-Flow and Qwen are Kaggle Models:
`dangkhoa2016/mage-flow-community-mage-flow-turbo` and
`dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf`.

Q8 uses Mage-Flow `GGUF / q8-0` plus `PyTorch / vae-only`, and Qwen `GGUF /
q4-k-m`. BF16 uses Mage-Flow `PyTorch / default` and Qwen `GGUF / q4-k-m`;
`PyTorch / default` already contains the BF16 diffusion model and VAE. The
recommended BF16 T4x2 input state attaches only the CUDA runtime, Mage-Flow
`PyTorch / default`, and Qwen `GGUF / q4-k-m`; do not attach the CPU runtime,
Mage-Flow `GGUF / q8-0`, or Mage-Flow `PyTorch / vae-only`. Follow the
[detailed Kaggle procedure](kaggle.md), then select **Run → Run All**.

## T4/T4x2 policy

- Physical host must be NVIDIA T4 or T4x2.
- Release qualification uses physical GPU slot 0 only.
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- `CUDA_VISIBLE_DEVICES=0`.
- Effective inference backend exactly `cuda0`.
- No `cuda1`, no multi-GPU split, no `auto-fit`, no CPU inference fallback.
- Prebuilt CUDA runtime only.
- Successful CUDA records require positive GPU-memory telemetry.

T4x2 is therefore accepted only as a host configuration; v1.0.0 remains a strict single-T4 qualification contract.

## Highlights

- JSON model manifests with fail-closed SHA-256 verification.
- Two supported release profiles using one native `sd-cli` execution path.
- Backend-specific prebuilt CPU/CUDA runtime identities frozen before qualification.
- Ordered four-resolution matrix harness supporting `cpu|cuda0` and `q8-reference|bf16-safetensors`.
- Evidence records source HEAD and TREE, profile/backend identity, runtime identity, model identities, canonical request, elapsed time, RAM/RSS, CUDA VRAM and PNG integrity.
- Strict four-cell offline comparator that refuses to compute ratios when comparability gates fail.
- Deterministic Markdown renderer for release-facing benchmark tables after source freeze.
- Evidence packager with exact v1.0.0 naming, model-weight rejection, known-secret scanning, tar path-safety validation, internal SHA-256 manifest and external sidecar checksum.
- `mageflow-native` CLI and loopback REST API remain available for local use/deployment.
- Kaggle-specific behavior remains isolated to `integrations/kaggle/`; the generic core has no hard `/kaggle/*` dependency.
- CI enforces source/publication/history/release contracts without committing model weights.

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

## Evidence and publication

The frozen comparator checks the four retained aggregate evidence files and the frozen renderer emits the release-facing benchmark Markdown. The checksum-protected qualification-equivalence manifest proves that the final publication source preserves byte-identical qualification-critical Git objects; it bridges provenance without rewriting the measured evidence source.

The GitHub Release publishes checksum-protected assets for all four qualification cells plus the combined comparison JSON/Markdown summary and final provenance/verification material. No model weights are included.

Measured latency, RSS/VRAM, PNG hashes, cell status and any accepted recorded platform limits are therefore authoritative in the GitHub Release assets/body, while source documentation defines the immutable protocol and interpretation rules. A corrected public Q8/T4 notebook smoke is supplemental reproduction, not replacement strict evidence.

## Outside v1.0.0 qualification scope

- multi-GPU `cuda0&cuda1` qualification;
- P100, TPU, Vulkan, Metal, ROCm or SYCL qualification;
- Hugging Face Transformers or PyTorch inference;
- model training/fine-tuning;
- automatic model-family conversion;
- claims that BF16 is universally higher quality than Q8.
