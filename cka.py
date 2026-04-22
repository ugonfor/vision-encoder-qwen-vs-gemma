"""Compute linear CKA between Qwen3.6 and Gemma-4 vision encoders.

Each vision tower is instantiated standalone (weights from download_vision.py,
which pulled only vision tensors via safetensors byte-range reads). We run both
on a shared image set, collect per-layer hidden states, then compute linear CKA.
"""

from __future__ import annotations

import csv
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file
from transformers import (
    AutoConfig,
    AutoProcessor,
    Gemma4VisionModel,
    Qwen3_5MoeVisionModel,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
N_IMAGES = 64
IMG_SIZE = 224  # CIFAR-10 is 32x32; upscale so both ViTs get meaningful token grids
WEIGHTS_DIR = Path("weights")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# -------------------- model loading --------------------

def load_vision_model(name: str, model_cls, prefix_candidates: tuple[str, ...]):
    d = WEIGHTS_DIR / name
    cfg = AutoConfig.from_pretrained(str(d))
    vision_cfg = cfg.vision_config
    print(f"[{name}] instantiating {model_cls.__name__}")
    try:
        model = model_cls._from_config(vision_cfg, dtype=DTYPE)
    except TypeError:
        model = model_cls._from_config(vision_cfg, torch_dtype=DTYPE)

    state: dict[str, torch.Tensor] = {}
    for shard in sorted(glob.glob(str(d / "vision-model-*.safetensors"))):
        state.update(load_file(shard))
    prefix = next((p for p in prefix_candidates if any(k.startswith(p) for k in state)), None)
    assert prefix is not None, f"No prefix match in {list(state)[:3]}"
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[{name}] prefix='{prefix}', missing={len(missing)}, unexpected={len(unexpected)}")
    if missing[:3]: print(f"[{name}] first missing: {missing[:3]}")

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[{name}] params={n_params:.1f}M  blocks={len(model.blocks) if hasattr(model,'blocks') else len(model.vision_model.encoder.layers) if hasattr(model,'vision_model') else '?'}")
    model = model.to(DEVICE).eval()

    proc = AutoProcessor.from_pretrained(str(d)).image_processor
    return model, proc


# -------------------- CKA --------------------

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X.float() - X.float().mean(0, keepdim=True)
    Y = Y.float() - Y.float().mean(0, keepdim=True)
    num = (X.T @ Y).pow(2).sum()
    den = (X.T @ X).pow(2).sum().sqrt() * (Y.T @ Y).pow(2).sum().sqrt()
    return (num / den).item()


# -------------------- image set --------------------

def load_images(n: int, size: int) -> list[Image.Image]:
    from datasets import load_dataset
    ds = load_dataset("cifar10", split="test", streaming=True)
    out = []
    for i, ex in enumerate(ds):
        if i == n: break
        img = (ex.get("img") or ex.get("image")).convert("RGB").resize((size, size), Image.BICUBIC)
        out.append(img)
    return out


# -------------------- feature collection --------------------

def _get_block_list(model) -> torch.nn.ModuleList:
    if hasattr(model, "blocks"):
        return model.blocks
    # Gemma: look for a ModuleList of transformer layers
    if hasattr(model, "vision_model") and hasattr(model.vision_model, "encoder"):
        return model.vision_model.encoder.layers
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return model.encoder.layers
    # Fallback: find the first ModuleList child
    for m in model.modules():
        if isinstance(m, torch.nn.ModuleList) and len(m) > 5:
            return m
    raise RuntimeError("could not locate transformer block list")


@torch.no_grad()
def collect_features_with_hooks(name: str, model, processor, images, forward_fn):
    blocks = _get_block_list(model)
    n_layers = len(blocks)
    cache: list[torch.Tensor] = [None] * n_layers
    handles = []

    def make_hook(i):
        def hook(_m, _inp, out):
            # Output could be Tensor or tuple (hidden_states, ...); take the tensor
            t = out[0] if isinstance(out, tuple) else out
            cache[i] = t.detach()
        return hook

    for i, blk in enumerate(blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))

    feats_per_layer: list[list[torch.Tensor]] = [[] for _ in range(n_layers)]

    try:
        for img in images:
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            if "pixel_values" in inputs:
                inputs["pixel_values"] = inputs["pixel_values"].to(DTYPE)
            forward_fn(model, inputs)
            for i in range(n_layers):
                t = cache[i]
                # Normalize shape to [tokens, D]
                if t.dim() == 3:   # [B, T, D]
                    pooled = t[0].float().mean(0)
                elif t.dim() == 2: # [T, D]  (packed Qwen ViT)
                    pooled = t.float().mean(0)
                else:
                    raise RuntimeError(f"unexpected block output shape {tuple(t.shape)}")
                feats_per_layer[i].append(pooled.cpu())
    finally:
        for h in handles:
            h.remove()

    return [torch.stack(v) for v in feats_per_layer]


def qwen_forward(model, inputs):
    # Qwen ViT: forward(hidden_states, grid_thw)
    return model(hidden_states=inputs["pixel_values"], grid_thw=inputs["image_grid_thw"])


def gemma_forward(model, inputs):
    return model(pixel_values=inputs["pixel_values"],
                 pixel_position_ids=inputs["image_position_ids"])


# -------------------- main --------------------

def main() -> None:
    print(f"device={DEVICE} dtype={DTYPE} n_images={N_IMAGES} size={IMG_SIZE}")
    t0 = time.time()
    qwen, qwen_proc   = load_vision_model("qwen",  Qwen3_5MoeVisionModel,
        prefix_candidates=("model.visual.", "visual.", "vision_tower.", "vision_model."))
    gemma, gemma_proc = load_vision_model("gemma", Gemma4VisionModel,
        prefix_candidates=("model.vision_tower.", "vision_tower.", "visual.", "vision_model."))
    print(f"[load] {time.time()-t0:.1f}s")

    images = load_images(N_IMAGES, IMG_SIZE)
    print(f"[data] {len(images)} images @ {IMG_SIZE}x{IMG_SIZE}")

    t0 = time.time()
    Xq = collect_features_with_hooks("qwen",  qwen,  qwen_proc,  images, qwen_forward)
    torch.cuda.empty_cache()
    print(f"[qwen] {len(Xq)} layers, D={Xq[0].shape[-1]} in {time.time()-t0:.1f}s")
    t0 = time.time()
    Xg = collect_features_with_hooks("gemma", gemma, gemma_proc, images, gemma_forward)
    print(f"[gemma] {len(Xg)} layers, D={Xg[0].shape[-1]} in {time.time()-t0:.1f}s")

    L = min(len(Xq), len(Xg))
    with open(RESULTS_DIR / "cka_diagonal.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["layer", "cka_same_index"])
        print("  layer  CKA(qwen_i, gemma_i)")
        for i in range(L):
            v = linear_cka(Xq[i], Xg[i])
            w.writerow([i, f"{v:.4f}"])
            print(f"  {i:>5}   {v:.3f}")

    M = np.zeros((len(Xq), len(Xg)), dtype=np.float32)
    for i in range(len(Xq)):
        for j in range(len(Xg)):
            M[i, j] = linear_cka(Xq[i], Xg[j])
    np.save(RESULTS_DIR / "cka_matrix.npy", M)
    print(f"[save] cka_matrix.npy {M.shape}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(M, origin="lower", vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xlabel("Gemma-4 layer"); ax.set_ylabel("Qwen-3.6 layer")
        ax.set_title(f"Linear CKA — vision towers (N={N_IMAGES}, CIFAR-10@{IMG_SIZE})")
        fig.colorbar(im, ax=ax, label="CKA")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "cka_heatmap.png", dpi=150)
        print("[save] cka_heatmap.png")
    except Exception as e:
        print(f"[plot] skipped: {e}")

    summary = {
        "n_images": N_IMAGES,
        "image_size": IMG_SIZE,
        "dataset": "cifar10-test",
        "qwen_layers": len(Xq),
        "gemma_layers": len(Xg),
        "qwen_hidden": list(Xq[0].shape[1:]),
        "gemma_hidden": list(Xg[0].shape[1:]),
        "cka_diagonal_mean": float(np.mean([M[i, i] for i in range(L)])),
        "cka_diagonal_min":  float(np.min([M[i, i] for i in range(L)])),
        "cka_diagonal_max":  float(np.max([M[i, i] for i in range(L)])),
        "best_gemma_for_each_qwen_layer": [int(np.argmax(M[i])) for i in range(len(Xq))],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
