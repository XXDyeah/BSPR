"""Half-Brier risk budget and delayed dual controller for BSPR."""

from __future__ import annotations

import contextlib
import random
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .deterministic_ops import select_class_scores_without_scatter


def half_brier_per_sample(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return half-Brier risk in ``[0, 1]`` for each sample."""

    if logits.dim() != 2:
        raise ValueError("logits must have shape [batch, classes]")
    if targets.dim() != 1 or targets.size(0) != logits.size(0):
        raise ValueError("targets must have shape [batch]")
    probabilities = F.softmax(logits, dim=1)
    true_probability = select_class_scores_without_scatter(
        probabilities, targets.long()
    )
    squared_norm = probabilities.pow(2).sum(dim=1)
    return 0.5 * (squared_norm + 1.0 - 2.0 * true_probability)


def compute_class_risk_estimates(
    logits: torch.Tensor,
    targets: torch.Tensor,
    old_class_ids: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one mean half-Brier estimate per old class and observation mask."""

    if old_class_ids.dim() != 1:
        raise ValueError("old_class_ids must be one-dimensional")
    risks = half_brier_per_sample(logits, targets)
    if old_class_ids.numel() == 0:
        return risks.new_zeros((0,)), torch.zeros(
            0, dtype=torch.bool, device=risks.device
        )
    membership = targets.long().view(-1, 1).eq(old_class_ids.view(1, -1))
    membership_float = membership.to(dtype=risks.dtype)
    counts = membership_float.sum(dim=0)
    sums = torch.mm(risks.view(1, -1), membership_float).view(-1)
    values = sums / counts.clamp(min=1.0)
    return values, counts.gt(0)


def class_balanced_risk_loss(
    risk_values: torch.Tensor,
    observed: torch.Tensor,
    class_lambdas: torch.Tensor,
) -> torch.Tensor:
    """Return the mean ``lambda_c * r_c`` over represented old classes."""

    if risk_values.shape != observed.shape or risk_values.shape != class_lambdas.shape:
        raise ValueError("risk tensors must have equal shape")
    active = observed.to(dtype=risk_values.dtype)
    denominator = active.sum().clamp(min=1.0)
    return (risk_values * class_lambdas * active).sum() / denominator


def update_risk_ema(
    previous: torch.Tensor,
    batch_values: torch.Tensor,
    observed: torch.Tensor,
    beta: float,
    initialized: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Update observed initialized classes and hold absent classes fixed."""

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if previous.shape != batch_values.shape or observed.shape != previous.shape:
        raise ValueError("risk EMA tensors must have equal shape")
    active = observed if initialized is None else observed & initialized
    candidate = float(beta) * previous + (1.0 - float(beta)) * batch_values
    return torch.where(active, candidate, previous)


def update_dual_variables(
    class_lambdas: torch.Tensor,
    risk_ema: torch.Tensor,
    budgets: torch.Tensor,
    observed: torch.Tensor,
    step_size: float,
    maximum: float,
) -> torch.Tensor:
    """Apply the projected post-update dual controller."""

    if any(tensor.shape != class_lambdas.shape for tensor in (
        risk_ema, budgets, observed
    )):
        raise ValueError("dual-controller tensors must have equal shape")
    proposal = (class_lambdas + float(step_size) * (risk_ema - budgets)).clamp(
        min=0.0, max=float(maximum)
    )
    return torch.where(observed, proposal, class_lambdas)


@contextlib.contextmanager
def _isolated_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        random.seed(int(seed))
        np.random.seed(int(seed) % (2 ** 32))
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def evaluate_reference_risks(
    model,
    raw_uint8_images: np.ndarray,
    mapped_labels: np.ndarray,
    train_transform,
    device,
    num_views: int = 4,
    batch_size: int = 128,
    seed: int = 0,
) -> Dict[int, float]:
    """Evaluate fixed first-stage references on retained stochastic views."""

    images = np.asarray(raw_uint8_images)
    labels = np.asarray(mapped_labels, dtype=np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("reference images and labels must have equal length")
    if num_views <= 0 or batch_size <= 0:
        raise ValueError("num_views and batch_size must be positive")
    if labels.size == 0:
        return {}

    device = torch.device(device)
    was_training = bool(model.training)
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    model.eval()
    try:
        with _isolated_rng(seed), torch.no_grad():
            for _ in range(int(num_views)):
                for start in range(0, labels.size, int(batch_size)):
                    stop = min(start + int(batch_size), labels.size)
                    tensors = [
                        train_transform(Image.fromarray(images[index]))
                        for index in range(start, stop)
                    ]
                    inputs = torch.stack(tensors, dim=0).to(device)
                    target = torch.from_numpy(labels[start:stop]).long().to(device)
                    logits = model(inputs)
                    risks = half_brier_per_sample(logits, target).detach().cpu().numpy()
                    for class_id, value in zip(labels[start:stop], risks):
                        key = int(class_id)
                        sums[key] = sums.get(key, 0.0) + float(value)
                        counts[key] = counts.get(key, 0) + 1
    finally:
        model.train(was_training)
    return {class_id: sums[class_id] / counts[class_id] for class_id in sums}
