# Dev Log — vision-encoder-comparison

## 2026-04-21

### Goal
Measure the representational similarity between the vision encoders of
`Qwen/Qwen3.6-35B-A3B` and `google/gemma-4-26B-A4B-it` using **linear CKA**,
layer by layer. The configs look suspiciously alike (same hidden=1152,
depth=27, heads=16, patch=16, intermediate=4304, SigLIP2-so400m-class shape) —
so the interesting question is how much each encoder has *drifted* during
multimodal fine-tuning.

### Environment
- OS: Windows 11, bash via Git Bash
- GPU: NVIDIA RTX 4060 (8 GB VRAM), driver 572.16, CUDA 12.8
- Python: 3.12 (uv-managed venv)
- Package manager: `uv` 0.8.6

### Dependencies
| Package | Version |
| --- | --- |
| torch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| transformers | 5.5.4 |
| accelerate | 1.13.x |
| huggingface_hub | 1.11.0 |
| safetensors | 0.7.0 |
| datasets | latest |

Install:
```bash
uv init --python 3.12 --no-readme
uv add torch torchvision --index https://download.pytorch.org/whl/cu128
uv add transformers accelerate huggingface_hub safetensors datasets numpy pillow matplotlib
```

### Constraint hit: disk + VRAM
First try of naive `snapshot_download` with `allow_patterns` filtering on
`visual.*` shards asked for **49.9 GB** for Gemma's `model-00001-of-00002.safetensors`
(the shard contains the vision tower *plus* a chunk of the LM). C: had only 41 GB
free. Aborted mid-download; partial 39 GB cleanup brought the drive back to 44 GB free,
still tight. Full checkpoints for both models total ~130 GB — infeasible here.

### Pivot: byte-range safetensors download
Safetensors files begin with `[u64 header_size] [json header] [tensor data]`,
where the header carries `data_offsets` per tensor. I wrote
`download_vision.py` to:

1. `hf_hub_download` the small `model.safetensors.index.json` and identify which
   shard(s) hold tensors whose names contain `visual.` / `vision_tower.`.
2. For each such shard: HTTP-range GET the 8-byte header size, then the JSON
   header, identify byte ranges of vision tensors only, stream those via
   `Range: bytes=start-end`, and re-pack into a new, compact safetensors file
   with contiguous offsets and a re-computed weight_map index.

Final download size:
- Qwen: **893 MB** (333 tensors from 2 shards — patch embedder + blocks + merger)
- Gemma: **1.14 GB** (355 tensors from 1 shard)

Both fit comfortably on disk and load well within the 8 GB VRAM budget in bf16.

### Loading standalone vision towers
`transformers 5.5.4` exposes:
- `Qwen3_5MoeVisionModel` (model_type `qwen3_5_moe`)
- `Gemma4VisionModel` (model_type `gemma4_vision`)

Both accept a nested `vision_config` via `_from_config(cfg.vision_config)`. Our
saved tensors still have the full-model prefixes (`model.visual.` for Qwen,
`model.vision_tower.` for Gemma) — stripping those makes `load_state_dict`
succeed with `missing=0, unexpected=0` on both. Sanity:

| Model | params | depth | hidden | patch | loaded from disk in bf16 |
| --- | --- | --- | --- | --- | --- |
| Qwen3_5MoeVisionModel | ~675 M | 27 | 1152 | 16 | ~0.9 GB |
| Gemma4VisionModel     | ~570 M | 27 | 1152 | 16 | ~1.1 GB |

### Forward signatures & hidden-state capture

The two models diverge here:

- **Qwen** `forward(hidden_states, grid_thw) -> Tensor`. The processor produces
  pre-patched `pixel_values` of shape `[num_patches, patch_dim]` (packed, native
  resolution) and `image_grid_thw`. No `output_hidden_states` support — return
  is a raw tensor.
- **Gemma** `forward(pixel_values, pixel_position_ids) -> BaseModelOutput`.
  The processor returns `[B, T, patch_dim]` with explicit position ids.

To compare layer-by-layer, I registered **forward hooks** on each entry of
`model.blocks` (both exist at the same path in these two architectures), caching
outputs per image. For pooling to `[D]` I mean over the token axis — standard
CKA protocol. Activations kept in bf16 on GPU during forward, cast to fp32 only
for the CKA computation itself.

### Data
CIFAR-10 test split (first 64 images), upscaled to 224×224 with bicubic
interpolation. Picked because: no HF auth gate, tiny download, enough domain
variety for CKA to stabilize (diagonal values move <0.01 beyond N=32 in spot
checks). Not a benchmark, a probe.

### Results

`results/cka_diagonal.csv` (Qwen layer *i* vs Gemma layer *i*):

```
 layer  CKA
   0   0.786
   1   0.752
   2   0.696
   3   0.676
   4   0.677
   5   0.721
   6   0.782
   7   0.764
   8   0.747
   9   0.690
  10   0.648
  11   0.618
  12   0.602
  13   0.617
  14   0.623
  15   0.627
  16   0.627
  17   0.611
  18   0.650
  19   0.626
  20   0.663
  21   0.647
  22   0.677
  23   0.654
  24   0.685
  25   0.727
  26   0.178
```

**Summary:**
- Diagonal mean: **0.658**, min **0.178** (layer 26), max **0.786** (layer 0).
- Early layers (0–8) sit in the **0.68–0.79** band — strong shared perceptual
  features, consistent with a common SigLIP2-class initialization.
- Mid layers (9–25) cluster around **0.60–0.73** — moderate task-driven drift
  from the two separate multimodal fine-tunes.
- Layer 26 (final): **0.178**. Sharp drop. This is the layer that feeds the
  projector / merger, which is where the two architectures actually diverge —
  Qwen compresses 2×2 spatial neighborhoods and projects to 2048, Gemma
  avg-pools (kernel 3) into a variable-budget soft-token stream. The
  interface-shaping objective looks very different, and that shows up cleanly
  at the top.

**Off-diagonal observation** (`best_gemma_for_each_qwen_layer` in `summary.json`):
Qwen layers 10–25 all have their *highest* CKA with **Gemma layer 26**. Put
differently: Qwen's deep/late representations look most like Gemma's final
pre-projector layer. A plausible reading is that Qwen's later blocks already
carry the "compressed, merger-ready" shape that Gemma only arrives at at its
last layer — Qwen has specialized earlier.

### Artifacts
- `download_vision.py`      — byte-range vision-only downloader
- `cka.py`                  — instantiation, hook-based collection, CKA + plots
- `results/cka_diagonal.csv`
- `results/cka_matrix.npy`  — 27 × 27 layer-pair CKA
- `results/cka_heatmap.png`
- `results/summary.json`

### What I'd do next
- **Scale N**: 64 CIFAR images is enough for stable CKA but not for strong
  claims. Repeat on 512 ImageNet val images once HF auth is sorted.
- **Domain sweep**: re-run on documents (DocVQA sample), charts (ChartQA), and
  photographs — the OCR-heavy fine-tuning both models advertise should pull CKA
  apart more on textual imagery than on natural images.
- **Weight-space check**: since the two encoders have matching shapes,
  directly diff corresponding tensors (cosine similarity of flattened weights)
  to see whether they share a literal initialization or just an architecture.
- **Pre-projector vs post-projector**: add hooks on `merger` / `avg_pool` to
  measure whether alignment collapses there too.
- **Linear-probe transferability**: freeze each encoder, train a 1-layer
  classifier on ImageNet-1k features, compare — the real test of "are these
  substitutable?".
