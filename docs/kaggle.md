# Kaggle production notebook

> 🌐 Language / Ngôn ngữ: **English** | [Tiếng Việt](kaggle.vi.md)

The public notebook is a reproduction/demo path. A fresh public Q8/T4 notebook
smoke is supplemental reproduction evidence and never replaces the retained
strict 2×2 benchmark authority.

## Exact Kaggle onboarding: Input → Add Input

1. In the Kaggle editor, open **Session options → Accelerator**. Select
   `None` for CPU, or NVIDIA T4/T4x2 for CUDA. Do not use P100, TPU, or another
   accelerator.
2. Open **Input → Add Input**. The runtime entries below are **Kaggle
   Datasets**; attach exactly one that matches the selected accelerator:

   | Accelerator | Dataset slug |
   |---|---|
   | `None` / CPU | `dangkhoa2016/stable-diffusion-cpp-6b3edaa-portable-cpu-runtime` |
   | T4 or T4x2 | `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime` |

3. In **Input → Add Input**, select **Kaggle Models** (not Datasets) for the
   model inputs. Mage-Flow is
   `dangkhoa2016/mage-flow-community-mage-flow-turbo`; Qwen is
   `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf`.

For `q8-reference`, attach Mage-Flow **GGUF / q8-0** and **PyTorch /
vae-only**, plus Qwen **GGUF / q4-k-m**. For `bf16-safetensors`, attach only
Mage-Flow **PyTorch / default** plus Qwen **GGUF / q4-k-m**. Mage-Flow
**PyTorch / default** already contains the BF16 diffusion model and VAE: do
not add **GGUF / q8-0** or **PyTorch / vae-only** to a BF16 session. Never mix
the Q8 and BF16 Mage-Flow attachment families in a normal session.

### Recommended fresh BF16 T4x2 reproduction checklist

Before running, set the configuration cell to:

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

The final **Input → Add Input** state must be:

- ATTACH Dataset: `dangkhoa2016/stable-diffusion-cpp-6b3edaa-cuda-t4-runtime`
- ATTACH Model: `dangkhoa2016/mage-flow-community-mage-flow-turbo` →
  **PyTorch / default**
- ATTACH Model: `dangkhoa2016/qwen-qwen3-vl-4b-instruct-gguf` →
  **GGUF / q4-k-m**
- DO NOT ATTACH Mage-Flow **GGUF / q8-0**, Mage-Flow **PyTorch / vae-only**,
  or the portable CPU runtime dataset.

On T4x2, the expected markers are `ACCELERATOR_DETECTED=nvidia-t4x2`,
`ACCELERATOR_POLICY=PASS`, `BACKEND_AUTO_SELECTED=cuda0`, and
`GPU1_NOT_USED=PASS`. Once the checklist is satisfied, choose **Run → Run All**.

## Accelerator and runtime policy

- `Accelerator=None` selects backend `cpu` and the prebuilt CPU runtime.
- NVIDIA T4 or T4x2 selects backend `cuda0` and the prebuilt CUDA T4 runtime.
  CUDA authority uses physical slot 0 only with `CUDA_DEVICE_ORDER=PCI_BUS_ID`
  and `CUDA_VISIBLE_DEVICES=0`.
- P100, TPU, mixed GPUs, and all other accelerators fail before discovery.

The prebuilt CPU and CUDA T4 runtimes are **Kaggle Datasets**. Attach exactly
one runtime dataset matching the selected accelerator; the notebook never
builds `stable-diffusion.cpp` from source.

## Model profiles

Mage-Flow, Qwen, and VAE inputs are **Kaggle Models**. Select exactly one
profile independently of the automatic CPU/T4 backend selection.

| Profile | Required model attachments |
|---|---|
| `q8-reference` | Mage-Flow `GGUF / q8-0`, Qwen `GGUF / q4-k-m`, and Mage-Flow `PyTorch / vae-only` |
| `bf16-safetensors` | Mage-Flow `PyTorch / default` and Qwen `GGUF / q4-k-m`; the default variation already includes the VAE, so do not attach separate `PyTorch / vae-only` |

Normal manifest discovery fails closed if both Mage-Flow diffusion families are
attached. Mixed-family tooling is controlled research only and is not allowed
for any fresh release qualification cell.

## Public defaults

```python
RUN_MODE = "experiment"
MODEL_PROFILE = "q8-reference"
RESOLUTION_PRESET = "auto"
RUN_FAIR_COMPARISON_BENCHMARK = False
```

Each **Run All** experiment has isolated local state. A reviewer may opt into
the `qualification_matrix` view, but it is an inspection surface, not an
instruction to rerun the retained four-cell authority merely because public
documentation changed.

## Retained qualification and public reproduction

The canonical measurements remain bound to the measured benchmark evidence
source HEAD/TREE recorded inside the retained evidence artifacts. The final
publication source contains public notebook/documentation/contract-test
corrections. A checksum-protected qualification-equivalence manifest bridges
the identities only when qualification-critical Git objects are byte-identical;
the measured evidence provenance is not rewritten.

1. `q8-reference` / `cpu`
2. `bf16-safetensors` / `cpu`
3. `q8-reference` / `cuda0`
4. `bf16-safetensors` / `cuda0`

The retained records use their recorded source identity, selected profile,
pinned runtime, and ordered `512 → 640 → 768 → 1024` matrix. Q8/CPU retains
its documented same-session full-reset recovery exception; equivalence does not
relabel it as fresh-session evidence. BF16 CPU retains its 27 GiB / 3 GiB
safety gates; those gates do not apply to BF16 CUDA. Canonical measurements and
evidence digests are published only in checksum-protected GitHub Release assets
and body.
