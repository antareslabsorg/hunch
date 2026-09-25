"""The flat decision scorer: a causal decoder backbone + scalar readout, one logit per candidate path.

Reference implementation for the sprint. Every candidate path is a full sequence; the shared-prefix
staged forward (design notes, unpublished) is validated against this later.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChoiceSetHead(nn.Module):
    """Permutation-equivariant residual candidate-set head (design notes, unpublished).
    Zero-initialized so that at step 0 it is an exact identity with the flat scorer.
    """
    def __init__(self, hidden_size: int, set_dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(hidden_size + 1, set_dim),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=set_dim, num_heads=num_heads, batch_first=True)
        self.ln = nn.LayerNorm(set_dim)
        self.psi = nn.Linear(set_dim, 1, bias=False)
        nn.init.zeros_(self.psi.weight)

    def forward(self, h: torch.Tensor, group_sizes: list[int], is_choice: list[bool] | None = None) -> torch.Tensor:
        """h: [P, d] normalized candidate readout vectors. Returns delta [P] residual logits."""
        P, d = h.shape
        Q = len(group_sizes)
        Kmax = max(group_sizes)
        device = h.device

        h_packed = h.new_zeros((Q, Kmax, d))
        mask = torch.ones((Q, Kmax), dtype=torch.bool, device=device)
        log_k = torch.zeros((Q, Kmax, 1), dtype=h.dtype, device=device)

        offset = 0
        for i, k in enumerate(group_sizes):
            h_packed[i, :k] = h[offset : offset + k]
            mask[i, :k] = False
            log_k[i, :k] = torch.log(torch.tensor(float(k), device=device, dtype=h.dtype))
            offset += k

        feat = torch.cat([h_packed, log_k], dim=-1)
        u = self.phi(feat)
        m, _ = self.attn(u, u, u, key_padding_mask=mask)
        delta = self.psi(self.ln(u + m)).squeeze(-1)

        delta_flat = h.new_zeros(P)
        offset = 0
        for i, k in enumerate(group_sizes):
            if is_choice is None or is_choice[i]:
                delta_flat[offset : offset + k] = delta[i, :k]
            offset += k
        return delta_flat


class DecisionScorer(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, use_set_head: bool = False):
        super().__init__()
        self.backbone = backbone
        self.norm = nn.RMSNorm(hidden_size, eps=1e-6)
        self.head = nn.Linear(hidden_size, 1, bias=False)  # a shared bias cancels inside the per-question softmax
        nn.init.normal_(self.head.weight, std=0.02)
        self.set_head = ChoiceSetHead(hidden_size) if use_set_head else None

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        readout_idx: torch.Tensor,
        group_sizes: list[int] | None = None,
        is_choice: list[bool] | None = None,
    ) -> torch.Tensor:
        """input_ids/attention_mask: [P, L] right-padded; readout_idx: [P] index of the readout token. Returns fp32 logits [P]."""
        hidden = self.backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        h = hidden[torch.arange(hidden.shape[0], device=hidden.device), readout_idx].float()
        # readout, norm and logits in fp32 regardless of autocast or FSDP mixed precision (which hands us fp16 weight views)
        with torch.autocast(device_type=h.device.type, enabled=False):
            h_norm = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.norm.eps) * self.norm.weight.float()
            z = F.linear(h_norm, self.head.weight.float()).squeeze(-1)
            if self.set_head is not None and group_sizes is not None:
                z = z + self.set_head(h_norm, group_sizes, is_choice)
            return z



def _disable_cudnn_sdpa_on_unsupported_arch() -> None:
    """Fall back off cuDNN's attention backend on a GPU this torch build has no kernels for.

    The B300 reports compute capability (10, 3) -- sm_103. torch 2.11.0+cu128 ships
    ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120'], with no sm_103, and cuDNN then raises
    "No valid execution plans built" on the first attention call. Training died before step 1 on a
    card costing $7.85/h. Flash and mem-efficient SDPA both work on that card; only the cuDNN
    backend is missing, so switching it off is enough.

    Deliberately conditional. Turning cuDNN SDPA off everywhere would change the attention kernel on
    the RTX PRO 6000 (sm_120) runs too, and those are the runs any B300 result gets compared against.
    Different SDPA backends agree mathematically but not bit-for-bit, so this fires only where the
    arch is genuinely unsupported -- and where it fires, it says so, because a run whose attention
    backend differs from its comparators has to be able to prove which one it used.
    """
    if not torch.cuda.is_available():
        return
    major, minor = torch.cuda.get_device_capability()
    if f"sm_{major}{minor}" in torch.cuda.get_arch_list():
        return
    import sys
    torch.backends.cuda.enable_cudnn_sdp(False)
    print(f"SDPA_BACKEND cudnn_disabled arch=sm_{major}{minor} "
          f"reason=absent_from_torch_arch_list torch={torch.__version__} "
          f"device={torch.cuda.get_device_name()}", file=sys.stderr, flush=True)


_disable_cudnn_sdpa_on_unsupported_arch()


def load_scorer(
    model_name: str,
    dtype: torch.dtype = torch.float32,
    attn_implementation: str = "sdpa",
    revision: str | None = None,
    use_set_head: bool = False,
):
    """Backbone without LM head (AutoModel) + new head. Returns (scorer, tokenizer)."""
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(model_name, revision=revision, dtype=dtype, attn_implementation=attn_implementation)
    backbone.config.use_cache = False
    return DecisionScorer(backbone, backbone.config.hidden_size, use_set_head=use_set_head), tokenizer


def pad_paths(paths: list[list[int]], pad_id: int, device: torch.device | str = "cpu"):
    """Right-pad a list of token paths. Returns (input_ids [P,L], attention_mask [P,L], readout_idx [P])."""
    lengths = torch.tensor([len(p) for p in paths], dtype=torch.long)
    width = int(lengths.max())
    ids = torch.full((len(paths), width), pad_id, dtype=torch.long)
    for i, p in enumerate(paths):
        ids[i, : len(p)] = torch.tensor(p, dtype=torch.long)
    mask = torch.arange(width)[None, :] < lengths[:, None]
    return ids.to(device), mask.to(device), (lengths - 1).to(device)


def group_logits(z: torch.Tensor, group_sizes: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter flat path logits [P] into [Q, Kmax] with a validity mask; invalid slots are -inf for softmax."""
    kmax = max(group_sizes)
    out = z.new_full((len(group_sizes), kmax), float("-inf"))
    valid = torch.zeros((len(group_sizes), kmax), dtype=torch.bool, device=z.device)
    offset = 0
    for i, k in enumerate(group_sizes):
        out[i, :k] = z[offset : offset + k]
        valid[i, :k] = True
        offset += k
    assert offset == z.shape[0], "group sizes do not cover all paths"
    return out, valid


def question_probs(z_grouped: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    p = F.softmax(z_grouped.float(), dim=-1)
    return p.masked_fill(~valid, 0.0)


if __name__ == "__main__":  # note: one runnable check on a tiny random backbone (CPU)
    try:
        from transformers import Qwen3Config as Cfg, Qwen3Model as Mdl
    except ImportError:
        try:
            from transformers import Qwen2Config as Cfg, Qwen2Model as Mdl
        except ImportError:
            from transformers import LlamaConfig as Cfg, LlamaModel as Mdl

    cfg = Cfg(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
              num_key_value_heads=2, head_dim=8, max_position_embeddings=64, use_cache=False, pad_token_id=0)
    cfg._attn_implementation = "eager"
    model = DecisionScorer(Mdl(cfg), 32, use_set_head=True).eval()
    paths = [[5, 6, 7, 9], [5, 6, 7, 10, 11], [5, 6, 8, 12]]
    ids, mask, ro = pad_paths(paths, pad_id=0)
    with torch.no_grad():
        z = model(ids, mask, ro)
        # padding invariance: the same path alone must give the same logit as inside the padded batch
        z0 = model(*pad_paths(paths[:1], pad_id=0))
    assert z.shape == (3,) and torch.allclose(z[0], z0[0], atol=1e-5), (z, z0)
    zg, valid = group_logits(z, [2, 1])
    p = question_probs(zg, valid)
    assert torch.allclose(p.sum(-1), torch.ones(2)) and p[1, 1] == 0
    print("model ok", z.tolist())
