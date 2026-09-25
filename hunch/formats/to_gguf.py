"""Convert the Hunch checkpoint to GGUF: f16 written here, Q8_0 and Q4_K_M by llama.cpp's own quantizer.

    python -m hunch.formats.to_gguf --checkpoint checkpoints/.../model.safetensors [--out hunch/formats/out] [--levels f16,Q8_0,Q4_K_M]

The design used here. The file holds the *backbone only*, as a standard `qwen3` GGUF whose `output_norm` is the
backbone's own final norm (`backbone.norm.weight`). The runtime (`hunch_gguf.py`) runs llama.cpp in embedding mode
with pooling NONE, which returns that post-norm hidden state -- the reference's `last_hidden_state` -- at the readout
token, and applies Hunch's fp32 readout norm, head and per-question softmax itself. The two readout tensors, the
temperature and the serialization version travel inside the file as `hunch.*` metadata (llama.cpp ignores keys it
does not know), so the artifact is self-describing and survives `llama_model_quantize` unchanged.

Why not design (a) (`norm.weight -> output_norm`, `head.weight -> output`, vocab 1): the reference applies TWO norms
in sequence -- Qwen3's final norm inside the backbone, then Hunch's fp32 norm -- and they do not fold into one:
RMSNorm(RMSNorm(x)*w1)*w2 == RMSNorm(x*w1)*w2, which is not an RMSNorm of x. llama.cpp's graph has a single
`output_norm`, so (a) would silently drop `backbone.norm` and score the wrong vector; the equivalence test measures how
wrong (`design_a` in its report). llama.cpp also refuses a file whose tensor count differs from the architecture's, so
the readout tensors cannot ride along as extra tensors -- metadata arrays are the honest place for them.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

import gguf
import numpy as np
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers import AutoTokenizer

from .common import BACKBONE, NAME, OUT, load_readout, metadata, sha256_file, verify_source

LEVELS = {"f16": None, "Q8_0": "LLAMA_FTYPE_MOSTLY_Q8_0", "Q4_K_M": "LLAMA_FTYPE_MOSTLY_Q4_K_M",
          "Q6_K": "LLAMA_FTYPE_MOSTLY_Q6_K", "Q5_K_M": "LLAMA_FTYPE_MOSTLY_Q5_K_M"}  # the last two are extras, not in the default --levels
DEFAULT_LEVELS = "f16,Q8_0,Q4_K_M"

RENAME = {  # checkpoint (HF Qwen3 names under `backbone.`) -> GGUF qwen3 names; no q/k permutation for this arch
    "embed_tokens": "token_embd", "input_layernorm": "attn_norm", "post_attention_layernorm": "ffn_norm",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v", "self_attn.o_proj": "attn_output",
    "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}


def gguf_name(key: str) -> str:
    """'backbone.layers.3.mlp.up_proj.weight' -> 'blk.3.ffn_up.weight'; 'backbone.norm.weight' -> 'output_norm.weight'."""
    body = key.removeprefix("backbone.").removesuffix(".weight")
    if body == "norm":
        return "output_norm.weight"
    if body.startswith("layers."):
        _, i, rest = body.split(".", 2)
        return f"blk.{i}.{RENAME[rest]}.weight"
    return f"{RENAME[body]}.weight"


def add_tokenizer(w: gguf.GGUFWriter, tok, cfg: dict) -> None:
    """The Qwen3 byte-level BPE vocabulary laid out as llama.cpp expects a `gpt2`-type vocab (pre-tokenizer `qwen2`)."""
    reverse = {i: t for t, i in tok.get_vocab().items()}
    added = tok.get_added_vocab()
    tokens, types = [], []
    for i in range(cfg["vocab_size"]):  # the embedding table has vocab_size rows; ids past the tokenizer's last are padding
        t = reverse.get(i)
        if t is None:
            tokens.append(f"[PAD{i}]"); types.append(gguf.TokenType.UNUSED)
        elif t in added:
            tokens.append(t); types.append(gguf.TokenType.CONTROL if tok.added_tokens_decoder[i].special else gguf.TokenType.USER_DEFINED)
        else:
            tokens.append(t); types.append(gguf.TokenType.NORMAL)
    merges = json.load(open(hf_hub_download(BACKBONE, "tokenizer.json")))["model"]["merges"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("qwen2")
    w.add_token_list(tokens)
    w.add_token_types(types)
    w.add_token_merges([" ".join(m) if isinstance(m, list) else m for m in merges])
    w.add_eos_token_id(tok.eos_token_id)
    w.add_pad_token_id(tok.pad_token_id)
    if cfg.get("bos_token_id") is not None:
        w.add_bos_token_id(cfg["bos_token_id"])
    w.add_add_bos_token(False)  # the serialization adds no special tokens (hunch/serialize.py: add_special_tokens=False)


def write_f16(ckpt: Path, out: Path, source_sha: str) -> None:
    cfg = json.load(open(hf_hub_download(BACKBONE, "config.json")))  # the raw file: transformers' config object renames fields across major versions
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    norm_w, head_w = load_readout(ckpt)
    w = gguf.GGUFWriter(str(out), "qwen3")
    w.add_name(NAME)
    w.add_type("model")
    w.add_description(f"Hunch decision scorer: {BACKBONE.split('/')[-1]} backbone with no LM head. NOT a causal LM: score with "
                      "hunch.formats.hunch_gguf; llama-cli would generate nonsense from the tied embeddings.")
    w.add_license("apache-2.0")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(cfg["num_hidden_layers"])
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(cfg["num_attention_heads"])
    w.add_head_count_kv(cfg["num_key_value_heads"])
    w.add_key_length(cfg["head_dim"])
    w.add_value_length(cfg["head_dim"])
    w.add_rope_freq_base(cfg["rope_theta"])
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_vocab_size(cfg["vocab_size"])
    add_tokenizer(w, tok, cfg)
    for k, v in metadata(source_sha).items():
        (w.add_float64 if isinstance(v, float) else w.add_string)(f"hunch.{k}", v)  # float64 so T reads back exactly
    w.add_array("hunch.readout_norm.weight", norm_w.tolist())
    w.add_array("hunch.head.weight", head_w.tolist())

    with safe_open(str(ckpt), framework="np") as f:
        keys = sorted(f.keys())
        outside = {k for k in keys if not k.startswith("backbone.")}
        if outside != {"norm.weight", "head.weight"}:
            raise SystemExit(f"unexpected tensors outside the backbone: {sorted(outside)}")
        for k in [k for k in keys if k not in outside]:
            t = f.get_tensor(k)
            # 2-D weights to f16; 1-D norms stay f32, as llama.cpp's own converter does for an f16 file
            w.add_tensor(gguf_name(k), t.astype(np.float16) if t.ndim == 2 else t.astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=True)
    w.close()


def quantize(src: Path, dst: Path, level: str) -> None:
    import llama_cpp as L

    p = L.llama_model_quantize_default_params()
    p.ftype = getattr(L, LEVELS[level])
    p.nthread = os.cpu_count() or 4
    rc = L.llama_model_quantize(str(src).encode(), str(dst).encode(), ctypes.byref(p))
    if rc != 0:
        raise RuntimeError(f"llama_model_quantize -> {level} returned {rc}")


def record(manifest: Path, entry: dict) -> None:
    m = json.loads(manifest.read_text()) if manifest.exists() else {}
    m[entry["file"]] = entry
    manifest.write_text(json.dumps(m, indent=1, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="model.safetensors of the release checkpoint")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--levels", default=DEFAULT_LEVELS, help=f"comma-separated subset of {list(LEVELS)}")
    ap.add_argument("--force", action="store_true", help="rebuild artifacts that already exist")
    args = ap.parse_args()

    levels = args.levels.split(",")
    bad = set(levels) - set(LEVELS)
    if bad:
        raise SystemExit(f"unknown levels {sorted(bad)}; known: {list(LEVELS)}")
    ckpt, out = Path(args.checkpoint), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_sha = verify_source(ckpt)

    f16 = out / f"{NAME}.f16.gguf"
    for level in ["f16"] + [l for l in levels if l != "f16"]:  # every quantised level is made from the f16 file
        dst = out / f"{NAME}.{level}.gguf"
        if dst.exists() and not args.force:
            print(f"{dst} exists; skipping (use --force to rebuild)")
        else:
            t0 = time.time()
            if level == "f16":
                write_f16(ckpt, dst, source_sha)
            else:
                quantize(f16, dst, level)
            print(f"wrote {dst} in {time.time() - t0:.0f}s")
        if level in levels:
            entry = {"file": dst.name, "format": "gguf", "level": level, "bytes": dst.stat().st_size, "sha256": sha256_file(dst), "source_sha256": source_sha}
            record(out / "manifest.json", entry)
            print(f"artifact {dst.name}\n  bytes  {entry['bytes']}\n  sha256 {entry['sha256']}", flush=True)


if __name__ == "__main__":
    main()
