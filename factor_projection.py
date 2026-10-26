"""Transport a retained LoRA factor to a newly synchronized A basis.

The operation solves, block by block,

    min_B || B A_new - B_old A_old ||_F^2

with a small Tikhonov term.  It is a local state transport step: it sends no
extra tensor and does not change the server aggregation protocol.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import torch


def project_B_to_new_A(
    old_A: torch.Tensor,
    old_B: torch.Tensor,
    new_A: torch.Tensor,
    *,
    ridge: float = 1e-6,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Return the least-squares transported B and residual diagnostics."""

    if old_A.ndim != 2 or old_B.ndim != 2 or new_A.ndim != 2:
        raise ValueError("LoRA factors must be matrices")
    rank = int(new_A.shape[0])
    if tuple(old_A.shape) != tuple(new_A.shape):
        raise ValueError(f"A shape mismatch: old={tuple(old_A.shape)}, new={tuple(new_A.shape)}")
    if tuple(old_B.shape[1:]) != (rank,) or int(old_B.shape[0]) == 0:
        raise ValueError(f"B shape mismatch: B={tuple(old_B.shape)}, rank={rank}")
    if float(ridge) < 0.0:
        raise ValueError("ridge must be non-negative")

    with torch.no_grad():
        work_dtype = torch.float32
        old_a = old_A.detach().to(dtype=work_dtype)
        old_b = old_B.detach().to(dtype=work_dtype)
        new_a = new_A.detach().to(dtype=work_dtype)
        old_update = old_b @ old_a
        gram = new_a @ new_a.T
        eye = torch.eye(rank, device=gram.device, dtype=gram.dtype)
        rhs = old_update @ new_a.T
        system = gram + float(ridge) * eye
        try:
            transported = torch.linalg.solve(system, rhs.T).T
        except RuntimeError:
            transported = rhs @ torch.linalg.pinv(system)

        before = torch.linalg.norm(old_b @ new_a - old_update).item()
        after = torch.linalg.norm(transported @ new_a - old_update).item()
        old_norm = max(torch.linalg.norm(old_update).item(), 1e-12)
        diagnostics = {
            "residual_before": float(before),
            "residual_after": float(after),
            "relative_reduction": float((before - after) / old_norm),
            "old_update_norm": float(old_norm),
        }
        return transported.to(device=old_B.device, dtype=old_B.dtype), diagnostics


def project_factor_dicts(
    old_A: Dict[str, torch.Tensor],
    old_B: Dict[str, torch.Tensor],
    new_A: Dict[str, torch.Tensor],
    new_B: Dict[str, torch.Tensor],
    target_layers: Iterable[str],
    *,
    ridge: float = 1e-6,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
    """Transport B for target blocks while preserving all other B tensors."""

    projected = {name: value.clone() for name, value in new_B.items()}
    diagnostics: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(target_layers)):
        if name not in old_A or name not in old_B or name not in new_A or name not in new_B:
            continue
        projected[name], diagnostics[name] = project_B_to_new_A(
            old_A[name], old_B[name], new_A[name], ridge=ridge
        )
    return projected, diagnostics
