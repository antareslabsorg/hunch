"""Shared pieces of the on-device formats.

The readout (fp32 RMSNorm + scalar head at the last token of each candidate path) is the same three lines as
`hunch/model.py::DecisionScorer.forward`; both format wrappers run the backbone in whatever precision the artifact
has and apply the readout below in fp32. Everything an artifact says about itself (temperature, serialization
version, source checkpoint) is defined here once and embedded by both converters.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

from ..infer import Hunch
from ..serialize import SERIALIZATION_VERSION

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "hunch" / "formats" / "out"
# The release checkpoint of each size: the file the converters pin, the in-family temperature every artifact embeds
# (fitted on the disjoint calibration split; docs/results/launch/derived.json) and the archived held-out read the
# equivalence test compares against. HUNCH_BACKBONE picks the size (the readout shape is read from the checkpoint, so only
# the tokenizer/config source changes); HUNCH_ARCHIVE and HUNCH_TEMPERATURE override the defaults for another scorer.
RELEASE = {
    "Qwen/Qwen3-0.6B": {"run": "w16-06-v3dgp10-1200-s55", "bytes": 2_384_244_232, "temperature": 0.8706,
                        "sha256": "75d678f36ce35cfcb8fc85f96a719bad77638af5b21b3ebabeea83e360ccf2a6",
                        "archive": "docs/results/day6/w16-06-v3dgp10-1200-s55_heldout2000.json"},
    "Qwen/Qwen3-1.7B": {"run": "g2-17-v4t", "bytes": 6_882_352_784, "temperature": 0.8706,
                        "sha256": "e2fa7f0ab23e3f1d1eadf0e630ba2b453e41c4bd13de5febca45cb4d9a572d66",
                        "archive": "docs/results/day7/v1/g2-17-v4t_heldout2000.json"},
}
BACKBONE = os.environ.get("HUNCH_BACKBONE", "Qwen/Qwen3-1.7B")
NAME = "hunch-" + BACKBONE.split("-")[-1].lower() + "-preview"  # named after the backbone so sizes cannot overwrite each other
_REL = RELEASE.get(BACKBONE, {})
# In family; out of family use 1.0. A size with no release has no temperature until one is given (metadata() refuses).
TEMPERATURE = float(os.environ.get("HUNCH_TEMPERATURE", _REL.get("temperature", "nan")))
READOUT_NORM_EPS = 1e-6
SOURCE_SHA256, SOURCE_BYTES = _REL.get("sha256"), _REL.get("bytes")
# The reference predictions to compare against: the size's release checkpoint's archived held-out read, unless
# HUNCH_ARCHIVE points at another run's _heldout2000.json, so a scorer is always compared against its OWN predictions.
_arch = Path(os.environ.get("HUNCH_ARCHIVE", REPO / _REL.get("archive", "docs/results/NO-RELEASE-FOR-THIS-BACKBONE.json")))
ARCHIVE = _arch if _arch.is_absolute() else (REPO / _arch)  # a relative override must still resolve inside the repo
DATA = REPO / "data/sprint_v3d"
OPEN_SPLITS = ("heldout", "dev", "calibration")  # `test` is sealed until 2026-09-25; nothing in this package may read it


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_source(path: str | Path) -> str:
    """sha256 the source checkpoint, print it, and refuse a mismatch: a truncated download converts fine and scores garbage.

    The pin applies to the release checkpoint only. Converting any other scorer (a 0.6B seed, a soup) is a
    deliberate act, so it must be named as one with HUNCH_ALLOW_UNPINNED=1; the sha is still computed and
    printed, so the artifact is still traceable to exactly one file.
    """
    path = Path(path)
    size, sha = path.stat().st_size, sha256_file(path)
    print(f"source {path}\n  bytes  {size}\n  sha256 {sha}", flush=True)
    if size != SOURCE_BYTES or sha != SOURCE_SHA256:
        if os.environ.get("HUNCH_ALLOW_UNPINNED") != "1":
            raise SystemExit(f"REFUSED: not the release checkpoint (expected {SOURCE_BYTES} bytes, sha256 "
                             f"{SOURCE_SHA256}). Set HUNCH_ALLOW_UNPINNED=1 to convert a different scorer on purpose.")
        print(f"  UNPINNED: not the release checkpoint; converting on purpose (HUNCH_ALLOW_UNPINNED=1)", flush=True)
    return sha


def load_readout(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """The two tensors outside the backbone: norm.weight [D] and head.weight [1, D] -> [D], fp32 (D = 1024 at 0.6B, 2048 at 1.7B)."""
    with safe_open(str(path), framework="np") as f:
        return f.get_tensor("norm.weight").astype(np.float32), f.get_tensor("head.weight").reshape(-1).astype(np.float32)


def readout(h: torch.Tensor, norm_w: torch.Tensor, head_w: torch.Tensor, eps: float = READOUT_NORM_EPS) -> torch.Tensor:
    """fp32 RMSNorm + scalar head on gathered readout vectors [P, D] -> logits [P]. DecisionScorer.forward, lines 82-83."""
    h = h.float()
    h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps) * norm_w.float()
    return F.linear(h_norm, head_w.float().reshape(1, -1)).squeeze(-1)


def metadata(source_sha: str) -> dict:
    """What every artifact embeds about itself. Floats stay floats so the GGUF writer types them as F32."""
    if not np.isfinite(TEMPERATURE):
        raise SystemExit(f"no release temperature for {BACKBONE}: set HUNCH_TEMPERATURE to the scorer's fitted value")
    return {
        "name": NAME,
        "model_type": "scorer",
        "backbone": BACKBONE,
        "serialization": SERIALIZATION_VERSION,
        "temperature": TEMPERATURE,
        "temperature_out_of_family": 1.0,
        "readout": "last token of each candidate path -> fp32 RMSNorm(eps=1e-6) -> Linear(hidden->1, no bias) -> softmax per question",
        "readout_norm_eps": READOUT_NORM_EPS,
        "source_checkpoint_sha256": source_sha,
        "license": "apache-2.0",
    }


def split_path(split: str) -> Path:
    if split not in OPEN_SPLITS:
        raise ValueError(f"split {split!r} is not readable here; allowed: {OPEN_SPLITS}")
    return DATA / f"{split}.jsonl"


class Backbone:
    """A backbone in some on-device format. Subclasses implement `hidden`; `__call__` has DecisionScorer's signature so
    `hunch.infer.Hunch` drives it unchanged."""

    meta: dict
    norm_w: torch.Tensor
    head_w: torch.Tensor

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, readout_idx: torch.Tensor) -> torch.Tensor:
        """Backbone last hidden state (after the backbone's own final norm) at the readout token: [P, D] fp32."""
        raise NotImplementedError

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, readout_idx: torch.Tensor) -> torch.Tensor:
        return readout(self.hidden(input_ids, attention_mask, readout_idx), self.norm_w, self.head_w)


class HunchOnDevice(Hunch):
    """`hunch.infer.Hunch` over an on-device backbone: same request/response contract, the embedded release T by default."""

    def __init__(self, backbone: Backbone, temperature: float | None = None, token_budget: int = 16384):
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(backbone.meta["backbone"])
        T = float(backbone.meta["temperature"]) if temperature is None else float(temperature)
        super().__init__(backbone, tok, T, torch.device("cpu"), None, token_budget)

    predict = Hunch.decide
