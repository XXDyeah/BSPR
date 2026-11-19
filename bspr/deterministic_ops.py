"""Differentiable selections that avoid CUDA atomic scatter-add backward.

Older PyTorch releases implement the backward pass of ``gather`` and
``index_select`` with CUDA scatter/index-add kernels.  Strict deterministic
execution rejects those kernels even when the indices happen to be unique.
The small BSPR Probe selections can instead be written as dense selector
operations, whose backward graph contains only elementwise reductions or a
matrix multiplication.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def select_class_scores_without_scatter(
    scores: torch.Tensor,
    class_indices: torch.Tensor,
) -> torch.Tensor:
    """Select one score per row without a differentiable ``gather``.

    ``class_indices`` must already be valid.  ``torch.where`` is used rather
    than multiplication so a non-selected NaN cannot leak through ``0 * NaN``.
    """

    if scores.dim() != 2:
        raise ValueError("scores must have shape [batch, classes]")
    if class_indices.dim() != 1 or class_indices.size(0) != scores.size(0):
        raise ValueError("class_indices must have shape [batch]")
    if class_indices.device != scores.device:
        raise ValueError("class_indices must share the score device")
    if scores.size(1) == 0:
        raise ValueError("scores must contain at least one class")

    indices = class_indices.detach().long()
    class_axis = torch.arange(
        scores.size(1), dtype=torch.long, device=scores.device
    ).view(1, -1)
    selector = indices.view(-1, 1).eq(class_axis)
    selected = torch.where(selector, scores, torch.zeros_like(scores))
    return selected.sum(dim=1)


def select_rows_without_scatter(
    values: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Select matrix rows through a dense selector instead of ``index_select``.

    BSPR calls this only for a unique subset of one mini-batch.
    The selector matrix therefore has negligible size and its matrix-product
    backward avoids the atomic index accumulation used by older CUDA builds.
    """

    if values.dim() != 2:
        raise ValueError("values must have shape [rows, features]")
    if row_indices.dim() != 1:
        raise ValueError("row_indices must have shape [selected_rows]")
    if row_indices.device != values.device:
        raise ValueError("row_indices must share the value device")

    indices = row_indices.detach().long()
    row_axis = torch.arange(
        values.size(0), dtype=torch.long, device=values.device
    ).view(1, -1)
    selector = indices.view(-1, 1).eq(row_axis).to(dtype=values.dtype)
    return torch.mm(selector, values)


def margin_ranking_loss_without_scatter(
    angular_scores: torch.Tensor,
    targets: torch.Tensor,
    num_old_classes: int,
    distance: float,
    top_k: int,
    coefficient: float,
) -> torch.Tensor:
    """Compute the original LUCIR MR objective without scatter backward."""

    true_scores = select_class_scores_without_scatter(angular_scores, targets)
    novel_scores = angular_scores[:, int(num_old_classes):].topk(
        int(top_k), dim=1
    )[0]
    per_pair = F.relu(
        float(distance) - true_scores.view(-1, 1) + novel_scores
    )
    old_mask = targets.lt(int(num_old_classes)).to(dtype=angular_scores.dtype)
    denominator = (old_mask.sum() * float(top_k)).clamp(min=1.0)
    return float(coefficient) * (
        per_pair * old_mask.view(-1, 1)
    ).sum() / denominator
