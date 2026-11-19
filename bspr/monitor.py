"""Scale-free old-new monitoring used by BSPR."""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import torch


class BatchPairwiseEstimate(NamedTuple):
    values: torch.Tensor
    observed: torch.Tensor
    counts: torch.Tensor


def _validate_scores(
    angular_scores: torch.Tensor,
    targets: torch.Tensor,
    num_old_classes: int,
    num_new_classes: Optional[int],
) -> int:
    if angular_scores.dim() != 2:
        raise ValueError("angular_scores must have shape [batch, classes]")
    if targets.dim() != 1 or targets.size(0) != angular_scores.size(0):
        raise ValueError("targets must have shape [batch]")
    if targets.device != angular_scores.device:
        raise ValueError("targets and angular_scores must share a device")
    if num_old_classes < 0 or num_old_classes > angular_scores.size(1):
        raise ValueError("num_old_classes is outside the score dimension")
    available = angular_scores.size(1) - num_old_classes
    resolved = available if num_new_classes is None else int(num_new_classes)
    if resolved < 0 or resolved > available:
        raise ValueError("num_new_classes is outside the score dimension")
    return resolved


def pairwise_old_side_response(
    margins: torch.Tensor, temperature: float = 0.20
) -> torch.Tensor:
    """Return ``sigmoid(margin / tau_b)``."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.sigmoid(margins / float(temperature))


def restorative_response(
    margins: torch.Tensor, temperature: float = 0.20
) -> torch.Tensor:
    """Return ``Q = min(1, 2 * (1 - sigmoid(margin/tau_b)))``."""

    return (2.0 * (1.0 - pairwise_old_side_response(
        margins, temperature
    ))).clamp(min=0.0, max=1.0)


def _empty_estimate(reference: torch.Tensor, shape) -> BatchPairwiseEstimate:
    return BatchPairwiseEstimate(
        reference.new_zeros(shape),
        torch.zeros(shape, dtype=torch.bool, device=reference.device),
        torch.zeros(shape, dtype=torch.long, device=reference.device),
    )


def compute_batch_pressure(
    angular_scores: torch.Tensor,
    targets: torch.Tensor,
    num_old_classes: int,
    num_new_classes: Optional[int] = None,
    temperature: float = 0.20,
) -> BatchPairwiseEstimate:
    """Average new-sample pressure for every old-new class pair."""

    num_new = _validate_scores(
        angular_scores, targets, num_old_classes, num_new_classes
    )
    shape = (num_old_classes, num_new)
    if num_old_classes == 0 or num_new == 0:
        return _empty_estimate(angular_scores, shape)
    new_mask = targets.ge(num_old_classes) & targets.lt(num_old_classes + num_new)
    selected = angular_scores[new_mask]
    local_targets = targets[new_mask].long() - num_old_classes
    if selected.size(0) == 0:
        return _empty_estimate(angular_scores, shape)

    target_scores = selected[:, num_old_classes:num_old_classes + num_new]
    target_scores = target_scores.gather(1, local_targets.view(-1, 1))
    margins = selected[:, :num_old_classes] - target_scores
    responses = pairwise_old_side_response(margins, temperature)
    membership = local_targets.view(-1, 1).eq(
        torch.arange(num_new, device=angular_scores.device).view(1, -1)
    ).to(dtype=responses.dtype)
    counts_by_new = membership.sum(dim=0)
    sums = torch.mm(responses.t(), membership)
    values = sums / counts_by_new.clamp(min=1.0).view(1, -1)
    counts = counts_by_new.to(dtype=torch.long).view(1, -1).expand(
        num_old_classes, -1
    )
    return BatchPairwiseEstimate(values, counts.gt(0), counts)


def compute_batch_restoration(
    angular_scores: torch.Tensor,
    targets: torch.Tensor,
    num_old_classes: int,
    num_new_classes: Optional[int] = None,
    temperature: float = 0.20,
) -> BatchPairwiseEstimate:
    """Average replay response for every old-new class pair."""

    num_new = _validate_scores(
        angular_scores, targets, num_old_classes, num_new_classes
    )
    shape = (num_old_classes, num_new)
    if num_old_classes == 0 or num_new == 0:
        return _empty_estimate(angular_scores, shape)
    old_mask = targets.ge(0) & targets.lt(num_old_classes)
    selected = angular_scores[old_mask]
    old_targets = targets[old_mask].long()
    if selected.size(0) == 0:
        return _empty_estimate(angular_scores, shape)

    true_scores = selected[:, :num_old_classes].gather(
        1, old_targets.view(-1, 1)
    )
    new_scores = selected[:, num_old_classes:num_old_classes + num_new]
    responses = restorative_response(true_scores - new_scores, temperature)
    membership = old_targets.view(-1, 1).eq(
        torch.arange(num_old_classes, device=angular_scores.device).view(1, -1)
    ).to(dtype=responses.dtype)
    counts_by_old = membership.sum(dim=0)
    sums = torch.mm(membership.t(), responses)
    values = sums / counts_by_old.clamp(min=1.0).view(-1, 1)
    counts = counts_by_old.to(dtype=torch.long).view(-1, 1).expand(-1, num_new)
    return BatchPairwiseEstimate(values, counts.gt(0), counts)


def update_masked_ema(
    previous: torch.Tensor,
    initialized: torch.Tensor,
    batch_values: torch.Tensor,
    observed: torch.Tensor,
    beta: float = 0.95,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Commit observed cells; initialize a first observation directly."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if previous.shape != batch_values.shape:
        raise ValueError("previous and batch_values must have equal shape")
    if initialized.shape != previous.shape or observed.shape != previous.shape:
        raise ValueError("EMA masks must have the statistic shape")
    first = observed & (~initialized)
    repeated = observed & initialized
    ema = float(beta) * previous + (1.0 - float(beta)) * batch_values
    updated = torch.where(first, batch_values, torch.where(repeated, ema, previous))
    return updated, initialized | observed


def compute_blindness(pressure: torch.Tensor, restoration: torch.Tensor) -> torch.Tensor:
    """Return ``Gamma = E * (1 - R)``."""

    if pressure.shape != restoration.shape:
        raise ValueError("pressure and restoration must have equal shape")
    return pressure.clamp(0.0, 1.0) * (1.0 - restoration.clamp(0.0, 1.0))


def supported_pairs(
    pressure_initialized: torch.Tensor,
    restoration_initialized: torch.Tensor,
    pressure_counts: torch.Tensor,
    restoration_counts: torch.Tensor,
    min_pressure_observations: int,
    min_restoration_observations: int,
) -> torch.Tensor:
    """Return pairs whose pressure and replay support are both sufficient."""

    shape = pressure_initialized.shape
    if any(tensor.shape != shape for tensor in (
        restoration_initialized, pressure_counts, restoration_counts
    )):
        raise ValueError("pairwise support tensors must have equal shape")
    return (
        pressure_initialized
        & restoration_initialized
        & pressure_counts.ge(int(min_pressure_observations))
        & restoration_counts.ge(int(min_restoration_observations))
    )


def normalize_blindness(blindness: torch.Tensor, smoothing: float = 0.10) -> torch.Tensor:
    if smoothing <= 0:
        raise ValueError("smoothing must be positive")
    bounded = blindness.clamp(0.0, 1.0)
    return bounded / (bounded + float(smoothing))


def compute_local_priorities(
    old_angular_scores: torch.Tensor,
    old_targets: torch.Tensor,
    blindness: torch.Tensor,
    num_old_classes: int,
    num_new_classes: Optional[int] = None,
    temperature: float = 0.20,
) -> torch.Tensor:
    """Return ``Pi_ij = Gamma[c_i,j] * (1 - Q_cij(u_i))``."""

    num_new = _validate_scores(
        old_angular_scores, old_targets, num_old_classes, num_new_classes
    )
    if blindness.shape != (num_old_classes, num_new):
        raise ValueError("blindness must have shape [C_old, C_new]")
    if old_angular_scores.size(0) == 0 or num_new == 0:
        return old_angular_scores.new_zeros((old_angular_scores.size(0), num_new))
    true_scores = old_angular_scores[:, :num_old_classes].gather(
        1, old_targets.long().view(-1, 1)
    )
    new_scores = old_angular_scores[:, num_old_classes:num_old_classes + num_new]
    margins = true_scores - new_scores
    response = restorative_response(margins, temperature)
    return blindness.index_select(0, old_targets.long()) * (1.0 - response)
