"""Download only the vision-tower tensors via safetensors byte-range reads.

Full shards for these models are 5–50 GB; vision encoders are ~0.5–0.7 GB.
Safetensors files are laid out as:
    [8 bytes: header_size u64 LE] [header_size bytes: JSON header] [tensor bytes...]
The header maps tensor_name -> {dtype, shape, data_offsets:[start,end]} where
offsets are relative to the start of the data region (right after the header).

We:
  1. Fetch model.safetensors.index.json to find which shards hold vision tensors.
  2. For each shard, HTTP-range GET the 8-byte header size, then the header JSON.
  3. Filter to vision tensor names, range-GET each tensor's byte slice.
  4. Write a compact local safetensors file containing only those tensors.
Also download the config / preprocessor / tokenizer files needed to instantiate
the model.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download, hf_hub_url

MODELS = {
    "qwen":  {"repo": "Qwen/Qwen3.6-35B-A3B",      "key_patterns": ("visual.", "vision_tower.", "vision_model.")},
    "gemma": {"repo": "google/gemma-4-26B-A4B-it", "key_patterns": ("vision_tower.", "visual.", "vision_model.")},
}

CONFIG_FILES = [
    "config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
]


def _range_get(url: str, start: int, end: int) -> bytes:
    """Inclusive byte range GET."""
    r = requests.get(url, headers={"Range": f"bytes={start}-{end}"}, stream=True, timeout=120)
    r.raise_for_status()
    return r.content


def fetch_shard_header(url: str) -> tuple[dict, int]:
    """Returns (header_dict, data_region_start_offset)."""
    size_bytes = _range_get(url, 0, 7)
    (header_size,) = struct.unpack("<Q", size_bytes)
    header_bytes = _range_get(url, 8, 8 + header_size - 1)
    header = json.loads(header_bytes)
    return header, 8 + header_size


def build_vision_safetensors(
    url: str,
    header: dict,
    data_offset: int,
    keep: list[str],
    out_path: Path,
) -> None:
    """Assemble a new safetensors file containing only `keep` tensors.

    We re-pack offsets to be contiguous in the output file.
    """
    new_header: dict = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]

    cursor = 0
    # Preserve deterministic order
    for name in keep:
        meta = header[name]
        old_start, old_end = meta["data_offsets"]
        size = old_end - old_start
        new_header[name] = {
            "dtype": meta["dtype"],
            "shape": meta["shape"],
            "data_offsets": [cursor, cursor + size],
        }
        cursor += size

    header_json = json.dumps(new_header, separators=(",", ":")).encode("utf-8")
    # Safetensors allows any length; no alignment required, but common practice
    # pads to 8. We skip padding to keep it simple.
    header_len = len(header_json)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<Q", header_len))
        f.write(header_json)
        for name in keep:
            meta = header[name]
            old_start, old_end = meta["data_offsets"]
            abs_start = data_offset + old_start
            abs_end = data_offset + old_end - 1
            # Stream in ~64MB chunks to keep memory low
            CHUNK = 64 * 1024 * 1024
            pos = abs_start
            while pos <= abs_end:
                end = min(pos + CHUNK - 1, abs_end)
                f.write(_range_get(url, pos, end))
                pos = end + 1
    print(f"    wrote {out_path.name}: {len(keep)} tensors, {cursor / 1e6:.1f} MB data")


def download_configs(repo: str, local_dir: Path) -> None:
    for fname in CONFIG_FILES:
        try:
            hf_hub_download(repo_id=repo, filename=fname, local_dir=str(local_dir))
        except Exception as e:
            # Some files may not exist on every repo
            if "404" in str(e) or "EntryNotFound" in type(e).__name__:
                continue
            # Re-raise on unexpected errors
            print(f"    warn {fname}: {e}")


def main() -> None:
    out_root = Path("weights")
    out_root.mkdir(exist_ok=True)

    for name, cfg in MODELS.items():
        repo = cfg["repo"]
        patterns = cfg["key_patterns"]
        print(f"\n=== {name}  ({repo}) ===")
        local_dir = out_root / name
        local_dir.mkdir(exist_ok=True)

        download_configs(repo, local_dir)

        # Load shard index
        idx_path = hf_hub_download(
            repo_id=repo,
            filename="model.safetensors.index.json",
            local_dir=str(local_dir),
        )
        weight_map: dict[str, str] = json.loads(Path(idx_path).read_text())["weight_map"]
        vision_keys = [k for k in weight_map if any(p in k for p in patterns)]
        vision_shards = sorted({weight_map[k] for k in vision_keys})
        print(f"  {len(vision_keys)} vision tensors across {len(vision_shards)} shard(s): {vision_shards}")

        new_weight_map: dict[str, str] = {}
        for shard in vision_shards:
            url = hf_hub_url(repo_id=repo, filename=shard)
            print(f"  reading header of {shard} ...")
            header, data_off = fetch_shard_header(url)
            keep = [k for k in vision_keys if weight_map[k] == shard]
            # Keep order stable (by key)
            keep.sort()
            out_shard = local_dir / f"vision-{shard}"
            build_vision_safetensors(url, header, data_off, keep, out_shard)
            for k in keep:
                new_weight_map[k] = out_shard.name

        # Write a trimmed index so HF loader can find the tensors.
        trimmed_idx = {
            "metadata": {"total_size": sum(
                (out_root / name / v).stat().st_size for v in set(new_weight_map.values())
            )},
            "weight_map": new_weight_map,
        }
        (local_dir / "vision.safetensors.index.json").write_text(
            json.dumps(trimmed_idx, indent=2)
        )
        print(f"  index -> vision.safetensors.index.json")


if __name__ == "__main__":
    main()
