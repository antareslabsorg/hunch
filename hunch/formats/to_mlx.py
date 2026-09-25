"""Convert the Hunch checkpoint to MLX: f16, plus 8-bit and 4-bit affine quantisation (group size 64, mlx-lm's defaults).

    python -m hunch.formats.to_mlx --checkpoint checkpoints/.../model.safetensors [--out hunch/formats/out] [--levels f16,8bit,4bit] [--dtype float16]

The backbone goes into mlx-lm's `Qwen3Model` -- the backbone class, not the `Model` wrapper that adds an LM head -- so
nothing is invented for a head that does not exist. The readout norm and head are stored next to it in fp32 under
`readout_norm.weight` and `head.weight`, never quantised, and applied in fp32 by `hunch_mlx.py`. `config.json`
carries the Qwen3 hyper-parameters under `model_type: "hunch_scorer"` (so `mlx_lm.load` refuses the folder instead of
wrapping a chat model around it), the `quantization` block mlx-lm uses, and a `hunch` block with the temperature.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
from huggingface_hub import hf_hub_download
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
from mlx_lm.utils import quantize_model
from safetensors import safe_open

from .common import BACKBONE, NAME, OUT, load_readout, metadata, sha256_file, verify_source
from .to_gguf import record

LEVELS = {"f16": None, "8bit": 8, "4bit": 4}
GROUP_SIZE = 64


def load_backbone(ckpt: Path, cfg: dict, dtype) -> Qwen3Model:
    model = Qwen3Model(ModelArgs.from_dict(cfg))
    with safe_open(str(ckpt), framework="np") as f:
        model.load_weights([(k.removeprefix("backbone."), mx.array(f.get_tensor(k)).astype(dtype))
                            for k in f.keys() if k.startswith("backbone.")], strict=True)
    mx.eval(model.parameters())
    return model


def convert(ckpt: Path, cfg: dict, level: str, dst: Path, dtype_name: str, source_sha: str) -> None:
    model = load_backbone(ckpt, cfg, getattr(mx, dtype_name))  # fresh per level: nn.quantize rewrites the model in place
    config = {k: v for k, v in cfg.items() if k != "architectures"}
    if LEVELS[level]:
        model, config = quantize_model(model, config, GROUP_SIZE, LEVELS[level])
    norm_w, head_w = load_readout(ckpt)
    weights = dict(tree_flatten(model.parameters()))
    weights["readout_norm.weight"], weights["head.weight"] = mx.array(norm_w), mx.array(head_w)  # fp32, never quantised
    dst.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(dst / "model.safetensors"), weights, metadata={"format": "mlx"})
    config.update(model_type="hunch_scorer",
                  hunch={**metadata(source_sha), "backbone_model_type": cfg["model_type"], "dtype": dtype_name, "level": level})
    (dst / "config.json").write_text(json.dumps(config, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model.safetensors of the release checkpoint")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"], help="backbone dtype (also the activation dtype)")
    ap.add_argument("--force", action="store_true", help="rebuild artifacts that already exist")
    args = ap.parse_args()

    levels = args.levels.split(",")
    bad = set(levels) - set(LEVELS)
    if bad:
        raise SystemExit(f"unknown levels {sorted(bad)}; known: {list(LEVELS)}")
    ckpt, out = Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = verify_source(ckpt)
    cfg = json.load(open(hf_hub_download(BACKBONE, "config.json")))

    for level in levels:
        dst = out / f"{NAME}-mlx-{level}"
        if (dst / "config.json").exists() and not args.force:
            print(f"{dst} exists; skipping (use --force to rebuild)")
        else:
            t0 = time.time()
            convert(ckpt, cfg, level, dst, args.dtype, source_sha)
            print(f"wrote {dst} in {time.time() - t0:.0f}s")
        f = dst / "model.safetensors"
        entry = {"file": f"{dst.name}/model.safetensors", "format": "mlx", "level": level, "dtype": args.dtype,
                 "quantization": json.loads((dst / "config.json").read_text()).get("quantization"),
                 "bytes": f.stat().st_size, "sha256": sha256_file(f), "source_sha256": source_sha}
        record(out / "manifest.json", entry)
        print(f"artifact {entry['file']}\n  bytes  {entry['bytes']}\n  sha256 {entry['sha256']}", flush=True)


if __name__ == "__main__":
    main()
