"""Run an MLX Hunch artifact: backbone in MLX (f16 or quantised), readout and per-question softmax in fp32.

    from hunch.formats.hunch_mlx import load
    h = load("hunch-0.6b-preview-mlx-f16")
    out = h.predict(json.load(open("examples/request.json"))["states"])      # the embedded release T applied by default

    python -m hunch.formats.hunch_mlx hunch-0.6b-preview-mlx-f16 examples/request.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import torch
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model

from ..fast import FastHunch
from .common import Backbone, HunchOnDevice, readout


class MLXBackbone(Backbone):
    """`activations="float32"` (default) keeps the residual stream, norms and attention in fp32 and only the stored
    weights in f16 / 4-bit / 8-bit -- the same precision class as llama.cpp, which computes in f32 and rounds matmul
    inputs. `activations="float16"` runs everything in f16 (mlx-lm's default behaviour): about twice as fast, and
    measurably further from the fp32 reference (FORMATS.md)."""

    def __init__(self, path: str | Path, activations: str = "float32"):
        d = Path(path)
        cfg = json.loads((d / "config.json").read_text())
        if cfg.get("model_type") != "hunch_scorer" or "hunch" not in cfg:
            raise ValueError(f"{d}: not a Hunch MLX artifact (model_type={cfg.get('model_type')!r})")
        self.meta = cfg["hunch"]
        self.model = Qwen3Model(ModelArgs.from_dict({**cfg, "model_type": self.meta["backbone_model_type"]}))
        q = cfg.get("quantization")
        if q:
            nn.quantize(self.model, group_size=q["group_size"], bits=q["bits"], mode=q.get("mode", "affine"))
        weights = mx.load(str(d / "model.safetensors"))
        self.norm_w = torch.from_numpy(np.array(weights.pop("readout_norm.weight"), dtype=np.float32))
        self.head_w = torch.from_numpy(np.array(weights.pop("head.weight"), dtype=np.float32))
        self.model.load_weights(list(weights.items()), strict=True)
        if activations == "float32":
            # Only the small tensors move to f32: norm weights and quantisation scales/biases. The big f16 or packed weights stay as
            # stored and are promoted inside each matmul, so resident memory stays at the artifact's size, not an fp32 copy of it.
            self.model.apply(lambda a: a.astype(mx.float32) if a.dtype in (mx.float16, mx.bfloat16) and a.ndim == 1 else a)
            for _, m in self.model.named_modules():
                if isinstance(m, (nn.QuantizedLinear, nn.QuantizedEmbedding)):
                    m.scales, m.biases = m.scales.astype(mx.float32), m.biases.astype(mx.float32)
        elif activations != "float16":
            raise ValueError(f"activations must be float32 or float16, got {activations!r}")
        self.act = getattr(mx, activations)
        mx.eval(self.model.parameters())
        mx.set_cache_limit(256 << 20)  # return freed Metal buffers above 256 MB instead of hoarding them
        self.settings = {"dtype": self.meta.get("dtype"), "activations": activations, "quantization": q, "mlx": mx.__version__}

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, readout_idx: torch.Tensor) -> torch.Tensor:
        # [P, L, D] after the backbone's final norm. The mask is causal, so right padding never reaches a readout token.
        x = mx.array(input_ids.numpy())
        h = self.model(x, input_embeddings=self.model.embed_tokens(x).astype(self.act))  # the residual stream runs in self.act
        hr = h[mx.arange(h.shape[0]), mx.array(readout_idx.numpy())].astype(mx.float32)
        mx.eval(hr)
        return torch.from_numpy(np.array(hr))

    def packed(self, ids: list[int], pos: list[int], mask: torch.Tensor, ro: list[int]) -> torch.Tensor:
        """One state's paths packed as in `hunch.fast.pack_state`: the backbone once, each token at its own path's position and
        seeing only its own path (the layers of mlx-lm's Qwen3 block, with the mask and positions given); readout logits, fp32."""
        m, L, P = self.model, len(ids), mx.array(pos)
        M = mx.array(mask.numpy())[None, None]
        # mx.fast.rope takes one offset per batch row, so each token is its own row: exactly the rotation for its position
        rope = lambda r, t: mx.fast.rope(t.transpose(2, 1, 0, 3), r.dims, traditional=r.traditional, base=r.base, scale=r.scale,
                                         offset=P).transpose(2, 1, 0, 3)
        h = m.embed_tokens(mx.array([ids])).astype(self.act)
        for layer in m.layers:
            a, y = layer.self_attn, layer.input_layernorm(h)
            if type(a.rope) is not nn.RoPE:
                raise NotImplementedError(f"rope {type(a.rope).__name__}: only the default RoPE of Qwen3-0.6B/1.7B is packed")
            q = a.q_norm(a.q_proj(y).reshape(1, L, a.n_heads, -1)).transpose(0, 2, 1, 3)
            k = a.k_norm(a.k_proj(y).reshape(1, L, a.n_kv_heads, -1)).transpose(0, 2, 1, 3)
            v = a.v_proj(y).reshape(1, L, a.n_kv_heads, -1).transpose(0, 2, 1, 3)
            o = mx.fast.scaled_dot_product_attention(rope(a.rope, q), rope(a.rope, k), v, scale=a.scale, mask=M)
            h = h + a.o_proj(o.transpose(0, 2, 1, 3).reshape(1, L, -1))
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        hr = m.norm(h)[0, mx.array(ro)].astype(mx.float32)
        mx.eval(hr)
        return readout(torch.from_numpy(np.array(hr)), self.norm_w, self.head_w)


class FastHunchOnDevice(FastHunch, HunchOnDevice):
    """`hunch.fast.FastHunch` over an MLX build: the contract of HunchOnDevice, each state read once per request."""

    def _scores(self, ids, pos, mask, ro) -> torch.Tensor:
        return self.scorer.packed(ids, pos, mask, ro)

    predict = FastHunch.decide


def load(path: str | Path, temperature: float | None = None, token_budget: int = 16384, activations: str = "float32",
         fast: bool = False) -> HunchOnDevice:
    """Same contract as hunch.infer.Hunch.load; `temperature=None` uses the T embedded in config.json (the release temperature).
    `fast=True` is shared-prefix execution (`hunch/fast.py`): the same answers, each state read once per request."""
    return (FastHunchOnDevice if fast else HunchOnDevice)(MLXBackbone(path, activations), temperature, token_budget)


if __name__ == "__main__":
    request = sys.argv[2] if len(sys.argv) > 2 else "examples/request.json"
    print(json.dumps(load(sys.argv[1]).predict(json.load(open(request))["states"]), indent=1))
