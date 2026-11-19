"""Closed-form geodesic probes and the straight-through BSPR loss."""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from .deterministic_ops import (
    select_class_scores_without_scatter,
    select_rows_without_scatter,
)
from .monitor import (
    compute_local_priorities,
    normalize_blindness,
)


class ProbeOutput(NamedTuple):
    loss: torch.Tensor
    accepted: torch.Tensor


def _strict_native_correct(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Require the true-class score to be strictly above every other score."""

    if logits.dim() != 2 or targets.dim() != 1:
        raise ValueError("invalid logits or target shape")
    class_ids = torch.arange(logits.size(1), device=logits.device).view(1, -1)
    other = class_ids.ne(targets.long().view(-1, 1))
    negative_infinity = logits.new_full(logits.shape, float("-inf"))
    other_scores = torch.where(other, logits, negative_infinity)
    true_scores = logits.gather(1, targets.long().view(-1, 1)).view(-1)
    return true_scores.gt(other_scores.max(dim=1)[0])


def straight_through_probe(
    normalized_features: torch.Tensor,
    detached_targets: torch.Tensor,
) -> torch.Tensor:
    """Return ``Norm(u + stop_gradient(tilde_u - u))``."""

    if normalized_features.shape != detached_targets.shape:
        raise ValueError("straight-through probe tensors must have equal shape")
    routed = normalized_features + (
        detached_targets - normalized_features
    ).detach()
    return F.normalize(routed, p=2, dim=1, eps=1e-12)


def construct_probe_loss(
    features: torch.Tensor,
    native_logits: torch.Tensor,
    angular_scores: torch.Tensor,
    targets: torch.Tensor,
    classifier_head,
    blindness: torch.Tensor,
    pair_supported: torch.Tensor,
    class_lambdas: torch.Tensor,
    num_old_classes: int,
    batch_size: int,
    pair_temperature: float,
    trigger_base: float,
    trigger_floor: float,
    priority_smoothing: float,
    min_margin: float,
    max_movement: float,
    max_angle: float,
    restoration_increment: float,
    probe_temperature: float,
    weight_floor: float,
) -> ProbeOutput:
    """Construct candidates, apply native validation, and compute Eq. (18).

    Every supported competitor above its class threshold is tested.  Selecting
    the highest-priority valid competitor is equivalent to trying competitors
    in descending priority and stopping at the first successful native check.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    num_new_classes = int(angular_scores.size(1) - num_old_classes)
    old_mask = targets.lt(num_old_classes)
    old_indices = torch.nonzero(old_mask, as_tuple=False).view(-1)
    old_count = int(old_indices.numel())
    graph_zero = features.sum() * 0.0
    accepted_zero = features.new_zeros(())
    if old_count == 0 or num_new_classes <= 0:
        return ProbeOutput(graph_zero, accepted_zero)

    old_features = select_rows_without_scatter(features, old_indices)
    old_unit = F.normalize(old_features, p=2, dim=1, eps=1e-12)
    old_targets = targets.index_select(0, old_indices).long()
    old_logits = native_logits.index_select(0, old_indices)
    old_angles = angular_scores.index_select(0, old_indices)
    with torch.no_grad():
        priorities = compute_local_priorities(
            old_angles.detach(),
            old_targets,
            blindness.detach(),
            num_old_classes,
            num_new_classes,
            pair_temperature,
        )
        supported = pair_supported.index_select(0, old_targets)
        lambdas = class_lambdas.index_select(0, old_targets).clamp(min=0.0)
        thresholds = torch.max(
            priorities.new_full((old_count,), float(trigger_floor)),
            float(trigger_base) / (1.0 + lambdas),
        )
        above_threshold = priorities.ge(thresholds.view(-1, 1))
        raw_correct = _strict_native_correct(old_logits.detach(), old_targets)
        feature_valid = old_features.detach().norm(p=2, dim=1).gt(1e-12)

        weights = classifier_head.normalized_weight_matrix().detach()
        q_old = weights.index_select(0, old_targets)
        new_ids = torch.arange(
            num_old_classes,
            num_old_classes + num_new_classes,
            device=features.device,
            dtype=torch.long,
        )
        q_new = weights.index_select(0, new_ids)
        old_weight_valid = q_old.norm(p=2, dim=1).gt(1e-12)
        new_weight_valid = q_new.norm(p=2, dim=1).gt(1e-12)
        pair_gradient = q_old.unsqueeze(1) - q_new.unsqueeze(0)
        unit_detached = old_unit.detach()
        raw_margins = (
            pair_gradient * unit_detached.unsqueeze(1)
        ).sum(dim=2)
        tangent = pair_gradient - raw_margins.unsqueeze(2) * unit_detached.unsqueeze(1)
        tangent_norm = tangent.norm(p=2, dim=2)
        directions = -tangent / tangent_norm.clamp(min=1e-12).unsqueeze(2)
        derivative = (pair_gradient * directions).sum(dim=2)

        positive_margin = raw_margins.gt(float(min_margin))
        tangent_valid = tangent_norm.gt(1e-12) & derivative.lt(-1e-12)
        requested = supported & above_threshold
        candidate = (
            requested
            & raw_correct.view(-1, 1)
            & feature_valid.view(-1, 1)
            & old_weight_valid.view(-1, 1)
            & new_weight_valid.view(1, -1)
            & positive_margin
            & tangent_valid
        )

        normalized_priority = normalize_blindness(priorities, priority_smoothing)
        movement = float(max_movement) * normalized_priority / (
            1.0 + lambdas.view(-1, 1)
        )
        target_margins = float(min_margin) + (
            1.0 - movement
        ) * (raw_margins - float(min_margin))
        zeta = torch.sqrt(raw_margins.pow(2) + derivative.pow(2)).clamp(min=1e-12)
        theta_zero = torch.acos((raw_margins / zeta).clamp(-1.0, 1.0))
        roots = torch.acos((target_margins / zeta).clamp(-1.0, 1.0)) - theta_zero
        alphas = roots.clamp(min=0.0, max=float(max_angle))
        candidates = (
            torch.cos(alphas).unsqueeze(2) * unit_detached.unsqueeze(1)
            + torch.sin(alphas).unsqueeze(2) * directions
        )
        candidates = F.normalize(candidates, p=2, dim=2, eps=1e-12)
        achieved_margins = (pair_gradient * candidates).sum(dim=2)

        flat_candidate = candidate.view(-1)
        candidate_indices = torch.nonzero(flat_candidate, as_tuple=False).view(-1)
        full_valid = torch.zeros_like(flat_candidate)
        if candidate_indices.numel():
            candidate_features = candidates.view(-1, candidates.size(2)).index_select(
                0, candidate_indices
            )
            candidate_old_targets = old_targets.view(-1, 1).expand(
                -1, num_new_classes
            ).reshape(-1).index_select(0, candidate_indices)
            candidate_logits = classifier_head(candidate_features)
            retained = _strict_native_correct(candidate_logits, candidate_old_targets)
            full_valid.index_copy_(0, candidate_indices, retained)
        full_valid = full_valid.view(old_count, num_new_classes)
        accepted_pairs = candidate & full_valid
        ranked = torch.where(
            accepted_pairs, priorities, priorities.new_full(priorities.shape, -1.0)
        )
        selected_priority, selected_local = ranked.max(dim=1)
        selected = selected_priority.ge(0.0)
        selected_rows = torch.nonzero(selected, as_tuple=False).view(-1)

    accepted = selected.to(features.dtype).sum()
    if selected_rows.numel() == 0:
        return ProbeOutput(graph_zero, accepted)

    selected_targets = old_targets.index_select(0, selected_rows)
    selected_local_targets = selected_local.index_select(0, selected_rows)
    selected_new_targets = selected_local_targets + num_old_classes
    selected_candidates = candidates[selected_rows, selected_local_targets].detach()
    selected_unit = select_rows_without_scatter(old_unit, selected_rows)
    probes = straight_through_probe(selected_unit, selected_candidates)
    probe_angles = classifier_head.angular_forward_normalized(probes)
    probe_old_scores = select_class_scores_without_scatter(
        probe_angles, selected_targets
    )
    probe_new_scores = select_class_scores_without_scatter(
        probe_angles, selected_new_targets
    )
    probe_margins = probe_old_scores - probe_new_scores

    selected_raw_margin = raw_margins[selected_rows, selected_local_targets].detach()
    selected_achieved_margin = achieved_margins[
        selected_rows, selected_local_targets
    ].detach()
    restoration_targets = torch.min(
        selected_raw_margin,
        selected_achieved_margin + float(restoration_increment),
    ).detach()
    selected_normalized_priority = normalized_priority[
        selected_rows, selected_local_targets
    ].detach()
    base_weights = float(weight_floor) + selected_normalized_priority
    normalized_base_weights = base_weights / base_weights.mean().clamp(min=1e-12)
    selected_lambdas = lambdas.index_select(0, selected_rows)
    probe_weights = normalized_base_weights * (1.0 + selected_lambdas)
    per_probe_loss = float(probe_temperature) * F.softplus(
        (restoration_targets - probe_margins) / float(probe_temperature)
    )
    loss = (probe_weights * per_probe_loss).sum() / float(batch_size)
    return ProbeOutput(loss, accepted)
