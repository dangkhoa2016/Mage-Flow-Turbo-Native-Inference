# v1.0.0 Strict 2×2 Native Benchmark Contract

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](BENCHMARKS-v1.0.0.vi.md)

This document defines the benchmark and interpretation contract for the first public **Mage-Flow-Turbo-Native-Inference v1.0.0** release. Retained measurements remain bound to the measured benchmark evidence source HEAD/TREE recorded in their checksum-protected GitHub Release evidence. The final publication source is bridged by a qualification-equivalence manifest only when qualification-critical Git objects are byte-identical.

## Four release cells

| Cell | Profile | Backend | Authority environment |
|---|---|---|---|
| Q8 CPU | `q8-reference` | `cpu` | retained measured evidence; documented same-session full-reset recovery |
| BF16 CPU | `bf16-safetensors` | `cpu` | retained measured evidence |
| Q8 T4 | `q8-reference` | `cuda0` | retained measured evidence, physical slot 0 only |
| BF16 T4 | `bf16-safetensors` | `cuda0` | retained measured evidence, physical slot 0 only |

The retained cells use the source HEAD/TREE recorded in their evidence and the same canonical request. The final publication source may differ only on allowed public-demo, documentation, and contract-test surfaces; the equivalence manifest does not rewrite evidence provenance.

## Frozen common protocol

```text
stable-diffusion.cpp = 6b3edaaf32cc19e5bb2d819c788bd557eddc8eba
prompt               = A small red fox sitting in a quiet green forest, natural light, detailed photography.
seed                 = 42
steps                = 4
CFG                  = 1.0
threads              = 4
resolution order     = 512 → 640 → 768 → 1024
generation count     = exactly 1 per resolution
```

Model identities:

```text
Q8 diffusion SHA256   = 4c3dafc143ee64121692b6b63563a4f5288bf6183c4870e1d65f1566519ba7f0
BF16 diffusion SHA256 = 6df47df3d7efc9ebdad075b87b3e9e4f74d09dca672d592271788f0ee27ab97d
Qwen SHA256           = 66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a
VAE SHA256            = 34e076dc1e8a15321e1e07be5111d59cf16dd10b804b7c7e20b4de29013427e0
CPU sd-cli SHA256     = 7539d90b99eaf2b6279eec4f9006a68ae53e87bfe0c9c325ff3f329220468a5c
CUDA sd-cli SHA256    = 3fae6c1991ad0ac764c36495f688817c8a3d295d7651369bf74b7fd33743c3d0
```

Both profiles execute through native `stable-diffusion.cpp` `sd-cli`. BF16 SafeTensors support is not a Hugging Face Transformers inference backend.

## Backend fairness

### CPU

- Kaggle `Accelerator=None`;
- backend exactly `cpu`;
- prebuilt CPU runtime only;
- no CUDA fallback;
- record host memory and peak `sd-cli` RSS;
- BF16 CPU keeps its release-safety RAM/headroom gates.

### T4 / T4x2

- physical host GPUs must be NVIDIA T4;
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`;
- `CUDA_VISIBLE_DEVICES=0`;
- inference backend exactly `cuda0`;
- no `cuda1`, multi-GPU split, `auto-fit`, or CPU inference fallback;
- prebuilt CUDA runtime only;
- successful records require positive GPU peak telemetry.

A T4x2 host is allowed only as a host configuration; v1.0.0 qualification is a single-T4 comparison.

## Fail-fast and partial-result semantics

The matrix starts at 512 and runs sequentially. Setup or 512 failure prevents later resolutions. A genuine later-resolution resource/runtime limit stops the remaining matrix and is preserved in evidence.

Identity mismatches, mixed-family attachment, source mismatch, fallback, missing required telemetry, corrupted evidence or policy violations are hard failures. They are never reclassified as an acceptable platform limit.

A later-resolution limit may be independently classified as `PARTIAL_WITH_RECORDED_LIMIT` only after the canonical 512 gate has passed and the evidence has been independently reviewed. The release decision for such a partial cell is a separate gate.

## Strict comparison rules

The frozen four-cell comparator refuses performance calculations unless it verifies:

- same measured evidence source HEAD and TREE across all four cells;
- same pinned upstream runtime commit;
- backend-specific runtime SHA-256 values;
- exact Q8/BF16 diffusion identities;
- same Qwen and VAE identities;
- same ordered resolution list;
- same canonical prompt, seed, steps, CFG and threads;
- CPU cells actually use CPU;
- CUDA cells actually use `cuda0` with `CUDA_VISIBLE_DEVICES=0`;
- successful CUDA records have positive GPU peak telemetry.

For each resolution where the required underlying records passed, the comparator reports:

1. Q8 CPU → T4 speedup = `Q8 CPU elapsed / Q8 T4 elapsed`.
2. BF16 CPU → T4 speedup = `BF16 CPU elapsed / BF16 T4 elapsed`.
3. CPU BF16/Q8 latency ratio = `BF16 CPU elapsed / Q8 CPU elapsed`.
4. T4 BF16/Q8 latency ratio = `BF16 T4 elapsed / Q8 T4 elapsed`.
5. CPU peak-RSS difference between BF16 and Q8.
6. T4 peak-VRAM difference between BF16 and Q8.

If a required record is missing or failed, the associated ratio is `N/A`/`null`; no inferred value is published.

## Release-facing outputs

The release carries exactly these retained checksum-protected assets:

- `mage-flow-turbo-native-inference-v1.0.0-q8-reference-cpu-same-session-recovery-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-q8-reference-cuda0-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cpu-evidence.tar.gz`
- `mage-flow-turbo-native-inference-v1.0.0-bf16-safetensors-cuda0-evidence.tar.gz`
- `mage-flow-v1.0.0-strict-2x2-comparison.json`
- `mage-flow-v1.0.0-strict-2x2-release-summary.md`
- `SHA256SUMS.txt`

The frozen comparator produces the JSON and the frozen renderer produces the
summary. The GitHub Release body is derived from that generated summary; source
documentation is not modified after qualification to paste dynamic numbers.

## Historical pre-redesign measurements

Before the strict 2×2 release redesign, the project completed a Q8-only fresh CPU/T4 matrix at an earlier pre-release source state. The historical measured native-generation times were:

| Resolution | Historical Q8 CPU | Historical Q8 T4 |
|---:|---:|---:|
| 512×512 | 215.816 s | 7.590 s |
| 640×640 | 338.014 s | 8.698 s |
| 768×768 | 491.700 s | 9.710 s |
| 1024×1024 | 939.371 s | 12.420 s |

Those values remain useful as pre-release research and regression context, but they are not a replacement for the retained strict 2×2 authority or its recorded evidence provenance.

The historical T4 1024 PNG SHA-256 `2c82bdb7cd68746c113eea0f95593d4860b3a201eb3869d0c384221b39b0e49e` likewise remains historical evidence only and is not a reference hash for a retained 512 or BF16 artifact.
