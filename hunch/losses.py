"""Per-question proper losses (design notes, unpublished). All in fp32, over valid candidates only, one value per question."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
    assert valid.dtype == torch.bool and z.shape == target.shape == valid.shape
    assert bool((valid.sum(-1) >= 2).all()), "every question needs at least two valid candidates"
    assert bool(torch.isfinite(z[valid]).all()), "non-finite logits"
    assert bool((target[~valid] == 0).all()), "target mass on invalid slots"
    assert torch.allclose(target.sum(-1), torch.ones_like(target[:, 0]), atol=1e-5), "targets must sum to 1"


def log_probs(z: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(z.float().masked_fill(~valid, float("-inf")), dim=-1)


def soft_cross_entropy(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """-sum_k t_k log p_k per question. Same gradient as KL(t || p)."""
    _check(z, target, valid)
    logp = log_probs(z, valid).masked_fill(~valid, 0.0)  # avoid 0 * -inf
    return -(target.float() * logp).sum(-1)


def forward_kl(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """KL(t || p) per question, for fidelity reporting (not a training loss; identical gradient to soft CE)."""
    t = target.float()
    ce = soft_cross_entropy(z, t, valid)
    ent = -(torch.where(t > 0, t * t.clamp_min(torch.finfo(t.dtype).tiny).log(), torch.zeros_like(t))).sum(-1)
    return ce - ent


def brier(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """sum_k (p_k - t_k)^2 per question (vector Brier; binary case is 2x the scalar Bernoulli Brier)."""
    _check(z, target, valid)
    p = log_probs(z, valid).exp().masked_fill(~valid, 0.0)
    return ((p - target.float()) ** 2).sum(-1)


def crps_ordinal(z: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """sum_k (F_p(k) - F_t(k))^2 over ordered levels; candidates must be in level order and left-packed."""
    _check(z, target, valid)
    p = log_probs(z, valid).exp().masked_fill(~valid, 0.0)
    return ((p.cumsum(-1) - target.float().cumsum(-1)) ** 2).masked_fill(~valid, 0.0).sum(-1)


if __name__ == "__main__":  # note: gradient check against the analytic p - t
    z = torch.tensor([[0.2, -0.7, 1.5, 0.0], [-0.1, 0.4, 0.0, 0.0]], requires_grad=True)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    t = torch.tensor([[0.1, 0.2, 0.7, 0.0], [0.5, 0.5, 0.0, 0.0]])
    soft_cross_entropy(z, t, valid).sum().backward()
    p = log_probs(z.detach(), valid).exp().masked_fill(~valid, 0.0)
    assert torch.allclose(z.grad, (p - t).masked_fill(~valid, 0.0), atol=1e-6)
    assert forward_kl(z.detach(), t, valid).min() >= -1e-6
    b = brier(z.detach(), t, valid)
    assert b.shape == (2,) and (b >= 0).all()
    print("losses ok")
