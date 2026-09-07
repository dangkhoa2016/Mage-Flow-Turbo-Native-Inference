# BF16 High-Memory CPU Qualification

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BF16-HIGH-MEMORY-CPU.vi.md)

This document defines the opt-in BF16 CPU qualification track for Mage-Flow-Turbo-Native-Inference. It does not replace the frozen `q8-reference` profile and does not change the v1.0.0 CPU ↔ T4 benchmark contract. The BF16 profile is experimental qualification infrastructure and is **not release-qualified** until a real CPU run produces evidence.

## Profile contract

`bf16-high-memory-cpu` uses a single Kaggle model mirror for both the transformer and the VAE:

```text
mage-flow-community-mage-flow-turbo / PyTorch / default
```

which provides both:

- `transformer/diffusion_pytorch_model.safetensors`
- `vae/diffusion_pytorch_model.safetensors`

No separate `pytorch/vae-only` input is required for the BF16 profile.

- Mage-Flow-Turbo transformer: official BF16 SafeTensors from the single `PyTorch / default` mirror, SHA-256 `6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d`.
- Qwen3-VL-4B text encoder: existing `Q4_K_M` GGUF reference artifact.
- Mage VAE: existing SafeTensors reference artifact from the same `PyTorch / default` mirror.
- Backend: `cpu` only.
- Minimum visible RAM: 27 GiB.
- Required observed runtime headroom: at least 3 GiB `MemAvailable` during the canonical generation.

The profile fails closed. CUDA, unsupported accelerators, insufficient RAM, missing telemetry, identity mismatches, and missing model artifacts do not fall back to Q8.

## Canonical first gate

Run exactly one 512×512 generation before any larger matrix:

```bash
python -m integrations.kaggle.qualification \
  --backend cpu \
  --profile bf16-high-memory-cpu \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-bf16-qualification \
  --repo-dir "$PWD"
```

The canonical request remains seed 42, 4 steps, CFG 1.0, 4 threads and the frozen fox prompt. Evidence records exact model identities, runtime identity, total RAM, pre-run available RAM, minimum available RAM, peak `sd-cli` RSS, elapsed time and PNG identity.

Gate 1 512 has run successfully on a fresh CPU-only Kaggle session at source HEAD `ee9119e8831558353dd514ef41fe867808e327b9`: model/runtime verification passed, CPU backend, artifact is a valid 512×512 PNG, minimum observed `MemAvailable` stayed at or above 3 GiB.

## Resolution matrix qualification

The matrix command resolves models, verifies model artifacts and verifies the runtime **once per matrix process**, then runs the frozen canonical request sequentially at 512 → 640 → 768 → 1024:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu \
  --profile bf16-high-memory-cpu \
  --input-root /kaggle/input \
  --work-root /kaggle/working/mageflow-bf16-matrix \
  --repo-dir "$PWD"
```

The matrix is qualification evidence for feasibility, memory behavior, latency and artifact correctness; it is **not release qualification** and performs no visual-quality comparison. Only width and height change between runs; prompt, seed, steps, CFG and threads are frozen.

The matrix requires an explicit prebuilt CPU runtime via `MAGE_CPU_PREBUILT_SD_CLI` and **no source build**. Setup runs once: RAM probe, profile preflight, manifest build, manifest verification, prebuilt `sd-cli` resolution, runtime identity and binary SHA verification. Setup timing telemetry is recorded separately so artifact-verification overhead, runtime-verification overhead and inference latency can be distinguished.

Fail-fast RAM policy: the BF16 minimum visible RAM of 27 GiB and the 3 GiB minimum observed `MemAvailable` are retained. If headroom falls below 3 GiB, the current resolution is recorded as failed, partial evidence is written and the matrix stops immediately; no later resolution runs. The 640/768/1024 matrix must produce real evidence before any cross-resolution timing or memory claims are published.

## Acceptance

The 512×512 gate is accepted only when model and runtime verification pass, `sd-cli` exits successfully, the PNG is valid, CPU is the selected backend, and minimum observed available memory remains at or above 3 GiB.

Only after this gate passes should a fresh CPU session run the same 512 → 640 → 768 → 1024 matrix for direct Q8 versus BF16 comparison. Until real BF16 evidence exists, this profile is experimental qualification infrastructure and must not be described as release-qualified.
