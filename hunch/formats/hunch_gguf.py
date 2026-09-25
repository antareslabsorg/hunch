"""Run a GGUF Hunch artifact with llama.cpp: backbone in embedding mode, readout and per-question softmax in fp32.

    from hunch.formats.hunch_gguf import load
    h = load("hunch-0.6b-preview.f16.gguf")
    out = h.predict(json.load(open("examples/request.json"))["states"])      # the embedded release T applied by default

    python -m hunch.formats.hunch_gguf hunch-0.6b-preview.f16.gguf examples/request.json

llama.cpp's embedding output with pooling NONE is the hidden state *after* `output_norm`: every graph builder sets
`res->t_embd` to the normed tensor, immediately before the `output` matmul. Because this GGUF's `output_norm` is the
backbone's own final norm, that vector is the reference's `last_hidden_state`, and Hunch's fp32 readout norm is applied
exactly once, here. The equivalence test confirms this empirically against the PyTorch hidden state before and after
the backbone norm (`design_a` in the equivalence report).
"""
from __future__ import annotations

import ctypes
import json
import os
import sys
from pathlib import Path

import gguf
import llama_cpp as L
import numpy as np
import torch

from .common import Backbone, HunchOnDevice

MAX_SEQ = 256  # LLAMA_MAX_SEQ: the most sequences one llama.cpp context can hold

_quiet = L.llama_log_callback(lambda level, text, user: None)  # ctypes callbacks must outlive their registration


def read_metadata(path: str | Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """The hunch.* keys: scalars and strings as a dict, plus the fp32 readout norm and head arrays."""
    meta, arrays = {}, {}
    for key, field in gguf.GGUFReader(str(path)).fields.items():
        if key.startswith("hunch."):
            v = field.contents()
            (arrays if isinstance(v, list) else meta)[key.removeprefix("hunch.")] = v
    if not meta or "readout_norm.weight" not in arrays or "head.weight" not in arrays:
        raise ValueError(f"{path}: no hunch.* metadata; not a Hunch artifact")
    return meta, np.asarray(arrays["readout_norm.weight"], dtype=np.float32), np.asarray(arrays["head.weight"], dtype=np.float32)


class GGUFBackbone(Backbone):
    def __init__(self, path: str | Path, n_ctx: int = 16384, n_ubatch: int = 512, n_gpu_layers: int = -1,
                 n_threads: int | None = None, flash_attn: bool = False, verbose: bool = False):
        self.path = str(path)
        self.meta, norm_w, head_w = read_metadata(path)
        self.norm_w, self.head_w = torch.from_numpy(norm_w), torch.from_numpy(head_w)
        if not verbose:
            L.llama_log_set(_quiet, ctypes.c_void_p(0))
        L.llama_backend_init()
        mp = L.llama_model_default_params()
        mp.n_gpu_layers = 999 if n_gpu_layers < 0 else n_gpu_layers
        self.model = L.llama_model_load_from_file(self.path.encode(), mp)
        if not self.model:
            raise RuntimeError(f"llama.cpp could not load {path}")
        cp = L.llama_context_default_params()
        cp.n_ctx = cp.n_batch = n_ctx
        cp.n_ubatch = n_ubatch
        cp.n_seq_max = MAX_SEQ
        cp.kv_unified = True  # one pool of n_ctx KV cells shared by a batch's sequences; per-sequence streams would get n_ctx/256 each
        cp.embeddings = True
        cp.pooling_type = L.LLAMA_POOLING_TYPE_NONE
        cp.flash_attn_type = L.LLAMA_FLASH_ATTN_TYPE_ENABLED if flash_attn else L.LLAMA_FLASH_ATTN_TYPE_DISABLED
        cp.op_offload = n_gpu_layers != 0  # with 0 layers offloaded, llama.cpp would still stream big matmuls to a GPU; 0 means CPU arithmetic
        cp.n_threads = cp.n_threads_batch = n_threads or os.cpu_count() or 4
        self.ctx = L.llama_init_from_model(self.model, cp)
        if not self.ctx:
            raise RuntimeError(f"llama.cpp could not create a context of {n_ctx} tokens")
        self.n_ctx, self.n_embd = n_ctx, L.llama_model_n_embd(self.model)
        self.vocab = L.llama_model_get_vocab(self.model)
        self.batch = L.llama_batch_init(n_ctx, 0, MAX_SEQ)
        self.settings = {"n_ctx": n_ctx, "n_ubatch": n_ubatch, "n_gpu_layers": mp.n_gpu_layers, "flash_attn": flash_attn,
                         "kv_unified": True, "type_k": int(cp.type_k), "type_v": int(cp.type_v), "llama_cpp_python": L.__version__}

    def close(self) -> None:
        """Free the context and model explicitly: ggml-metal asserts at process exit if residency sets are still registered."""
        if getattr(self, "batch", None) is not None:
            L.llama_batch_free(self.batch); self.batch = None
        if getattr(self, "ctx", None):
            L.llama_free(self.ctx); self.ctx = None
        if getattr(self, "model", None):
            L.llama_model_free(self.model); self.model = None

    __del__ = close

    def hidden(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, readout_idx: torch.Tensor) -> torch.Tensor:
        lengths = attention_mask.sum(-1).tolist()
        if sum(lengths) > self.n_ctx:
            raise ValueError(f"{sum(lengths)} tokens in one batch > n_ctx={self.n_ctx}; lower the token budget")
        ids = input_ids.tolist()
        out = np.empty((len(lengths), self.n_embd), dtype=np.float32)
        for start in range(0, len(lengths), MAX_SEQ):
            stop = min(start + MAX_SEQ, len(lengths))
            out[start:stop] = self._decode([ids[r][: lengths[r]] for r in range(start, stop)])
        return torch.from_numpy(out)

    def _decode(self, seqs: list[list[int]]) -> np.ndarray:
        """One llama_decode over up to MAX_SEQ independent sequences (each from position 0); the post-output_norm vector of each last token."""
        b, n = self.batch, sum(map(len, seqs))
        tok = np.ctypeslib.as_array(b.token, shape=(self.n_ctx,))
        pos = np.ctypeslib.as_array(b.pos, shape=(self.n_ctx,))
        nsq = np.ctypeslib.as_array(b.n_seq_id, shape=(self.n_ctx,))
        flag = np.ctypeslib.as_array(b.logits, shape=(self.n_ctx,))
        tok[:n] = np.concatenate([np.asarray(s, dtype=np.int32) for s in seqs])
        pos[:n] = np.concatenate([np.arange(len(s), dtype=np.int32) for s in seqs])
        nsq[:n] = 1
        flag[:n] = 0
        last, i = [], 0
        for s, seq in enumerate(seqs):
            for j in range(len(seq)):
                b.seq_id[i + j][0] = s
            i += len(seq)
            last.append(i - 1)
            flag[i - 1] = 1  # only the readout token produces an output
        b.n_tokens = n
        L.llama_memory_clear(L.llama_get_memory(self.ctx), False)
        rc = L.llama_decode(self.ctx, b)
        if rc != 0:
            raise RuntimeError(f"llama_decode returned {rc} for {len(seqs)} sequences / {n} tokens")
        out = np.empty((len(seqs), self.n_embd), dtype=np.float32)
        for s, i in enumerate(last):
            ptr = L.llama_get_embeddings_ith(self.ctx, i)
            if not ptr:
                raise RuntimeError(f"llama.cpp returned no embedding for batch token {i}")
            out[s] = np.ctypeslib.as_array(ptr, shape=(self.n_embd,))
        return out

    def tokenize(self, text: str) -> list[int]:
        """llama.cpp's own tokenizer on this GGUF, no special tokens: for parity checks against the HF tokenizer."""
        raw = text.encode("utf-8")
        buf = (L.llama_token * (len(raw) + 8))()
        n = L.llama_tokenize(self.vocab, raw, len(raw), buf, len(buf), False, False)
        if n < 0:
            buf = (L.llama_token * -n)()
            n = L.llama_tokenize(self.vocab, raw, len(raw), buf, -n, False, False)
        return list(buf[:n])


def load(path: str | Path, temperature: float | None = None, token_budget: int = 16384, **kw) -> HunchOnDevice:
    """Same contract as hunch.infer.Hunch.load; `temperature=None` uses the T embedded in the file (the release temperature)."""
    return HunchOnDevice(GGUFBackbone(path, n_ctx=token_budget, **kw), temperature, token_budget)


if __name__ == "__main__":
    request = sys.argv[2] if len(sys.argv) > 2 else "examples/request.json"
    h = load(sys.argv[1])
    print(json.dumps(h.predict(json.load(open(request))["states"]), indent=1))
    h.scorer.close()
