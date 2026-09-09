# BF16 SafeTensors Release Profile and Historical CPU Research

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BF16-HIGH-MEMORY-CPU.vi.md)

`bf16-safetensors` is the supported BF16 model profile for Mage-Flow-Turbo-Native-Inference v1.0.0. It uses the same native `stable-diffusion.cpp` `sd-cli` execution engine as `q8-reference`; it does **not** introduce a Hugging Face Transformers or PyTorch inference backend.

The filename of this document is retained for link stability. The former CPU-only release contract is superseded by the strict v1.0.0 2×2 profile/backend matrix.

## Profile contract

`bf16-safetensors` uses the Mage-Flow-Turbo `PyTorch / default` mirror as the source distribution for the BF16 transformer and VAE:

```text
mage-flow-community-mage-flow-turbo / PyTorch / default
```

Required components:

- Mage-Flow-Turbo transformer: BF16 SafeTensors, SHA-256 `6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d`.
- Qwen3-VL-4B text encoder: GGUF `Q4_K_M`, SHA-256 `66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a`.
- Mage VAE: SafeTensors, SHA-256 `34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0`.
- Native runtime: pinned `stable-diffusion.cpp` commit `6b3edaaf32cc19e5bb2d819c788bd557eddc8eba`.
- Supported release backends: `cpu`, `cuda0`.

`PyTorch / default` describes the upstream/mirror artifact layout only. Inference still runs through native `sd-cli`.

## CPU policy

BF16 CPU qualification keeps the conservative high-memory safety gates established by earlier research:

- minimum visible host RAM: 27 GiB;
- required observed `MemAvailable` headroom during a successful canonical run: at least 3 GiB;
- backend exactly `cpu`;
- prebuilt CPU runtime SHA-256 `7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c`;
- no accelerator fallback.

These are release-safety gates for the tested qualification workflow, not a universal claim that every BF16 use case inherently needs exactly 27 GiB.

## CUDA0 policy

BF16 CUDA qualification is a strict single-T4 path:

- Kaggle T4 or T4x2 host;
- physical GPU slot 0 only;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- effective backend `cuda0`;
- no `cuda1`, multi-GPU split, `auto-fit`, or CPU inference fallback;
- prebuilt CUDA runtime SHA-256 `3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0`;
- successful generations require positive VRAM telemetry.

The CPU-only 27 GiB / 3 GiB headroom gate is not reused as a CUDA backend rejection. Host memory is still recorded, but CUDA feasibility is determined by the explicit native CUDA execution, artifact validity and VRAM evidence.

## Strict release matrix

For both CPU and CUDA0, the ordered authority matrix is:

```text
512x512 → 640x640 → 768x768 → 1024x1024
prompt  = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed    = 42
steps   = 4
CFG     = 1.0
threads = 4
```

Run with the dedicated matrix harness:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cpu \
  --profile bf16-safetensors \
  --input-root /kaggle/input \
  --work-root /kaggle/working/v1-bf16-cpu \
  --repo-dir "$PWD" \
  --resolutions 512,640,768,1024
```

or, in a fresh T4/T4x2 session with `CUDA_VISIBLE_DEVICES=0`:

```bash
python -m integrations.kaggle.qualification_matrix \
  --backend cuda0 \
  --profile bf16-safetensors \
  --input-root /kaggle/input \
  --work-root /kaggle/working/v1-bf16-t4 \
  --repo-dir "$PWD" \
  --resolutions 512,640,768,1024
```

The harness is sequential and fail-fast. Canonical 512 is the first feasibility gate. If a later resolution hits a genuine runtime/resource limit, that limit is preserved in evidence; the qualification path must not switch to CPU offload, multi-GPU, auto-fit or another backend merely to force a pass.

## Release evidence semantics

Each matrix records source HEAD/TREE, exact model/runtime identities, profile/backend, canonical request, elapsed time, host memory/RSS, CUDA peak VRAM when applicable, PNG identity and explicit failure classification.

The four release cells are independently reviewed and then compared by the frozen four-cell comparator. Ratios are calculated only when the relevant records are individually successful and all identity/comparability gates pass.

Q8 remains the canonical/default profile. Supporting BF16 as a release profile does not assert that BF16 is universally higher quality or more efficient.

## Historical pre-release CPU visual research

Before the strict 2×2 redesign, a same-host paired 768×768 CPU visual study compared Q8 and BF16:

- 10 prompts × 2 representations = 20 runs;
- 4 steps, CFG 1.0, 4 CPU threads;
- 20/20 runs succeeded;
- Q8 mean elapsed ≈ 640.0 s/image;
- BF16 mean elapsed ≈ 1013.9 s/image;
- Q8 peak `sd-cli` RSS ≈ 8.89 GB;
- BF16 peak `sd-cli` RSS ≈ 12.54 GB;
- blind visual result: BF16 4 wins, Q8 3 wins, 3 ties, with only one materially clear BF16 win.

That study remains historical quality-oriented research. It is **not** the final v1.0.0 strict 2×2 performance authority, does not use the final redesigned source HEAD, and must not be substituted for any of the four fresh release qualification cells.

Fresh final performance numbers are generated only after the new source freeze and are published in checksum-protected GitHub Release assets/body rather than edited back into source documentation.
