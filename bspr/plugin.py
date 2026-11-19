"""Single-forward, single-backward BSPR integration for LUCIR."""

from __future__ import annotations

from typing import Dict, NamedTuple

import torch

from .controller import (
    class_balanced_risk_loss,
    compute_class_risk_estimates,
    update_dual_variables,
    update_risk_ema,
)
from .monitor import (
    compute_batch_pressure,
    compute_batch_restoration,
    compute_blindness,
    supported_pairs,
    update_masked_ema,
)
from .probe import construct_probe_loss


class BSPRStepOutput(NamedTuple):
    probe_loss: torch.Tensor
    risk_loss: torch.Tensor
    old_fraction: torch.Tensor
    delayed_statistics: Dict[str, torch.Tensor]
    accepted: torch.Tensor


class BSPRPlugin(object):
    """Compute the BSPR losses and commit detached state after optimization."""

    def __init__(self, config, state, classifier_head):
        self.config = config
        self.state = state
        self.head = classifier_head

    def compute_losses(
        self,
        features: torch.Tensor,
        native_logits: torch.Tensor,
        angular_scores: torch.Tensor,
        targets: torch.Tensor,
        num_old_classes: int,
    ) -> BSPRStepOutput:
        config = self.config
        state = self.state
        num_new_classes = int(angular_scores.size(1) - num_old_classes)
        if tuple(state.E.shape) != (num_old_classes, num_new_classes):
            raise ValueError("BSPR state does not match the current classifier")

        batch_size = int(targets.size(0))
        old_count = int(targets.lt(num_old_classes).to(dtype=torch.long).sum().item())
        old_fraction = native_logits.new_tensor(float(old_count) / float(batch_size))

        pressure = compute_batch_pressure(
            angular_scores.detach(),
            targets.detach(),
            num_old_classes,
            num_new_classes,
            config.pair_temperature,
        )
        restoration = compute_batch_restoration(
            angular_scores.detach(),
            targets.detach(),
            num_old_classes,
            num_new_classes,
            config.pair_temperature,
        )
        delayed = {
            "pressure_values": pressure.values.detach(),
            "pressure_observed": pressure.observed.detach(),
            "pressure_counts": pressure.counts.detach(),
            "restoration_values": restoration.values.detach(),
            "restoration_observed": restoration.observed.detach(),
            "restoration_counts": restoration.counts.detach(),
        }

        risk_values, risk_observed = compute_class_risk_estimates(
            native_logits,
            targets,
            state.old_class_ids,
        )
        initialized = state.risk_initialized.index_select(0, state.old_class_ids)
        risk_observed = risk_observed & initialized
        risk_loss = class_balanced_risk_loss(
            risk_values,
            risk_observed,
            state.dual.index_select(0, state.old_class_ids).detach(),
        )
        delayed["risk_values"] = risk_values.detach()
        delayed["risk_observed"] = risk_observed.detach()

        blindness = compute_blindness(state.E, state.R).detach()
        pair_supported = supported_pairs(
            state.E_initialized,
            state.R_initialized,
            state.n_E,
            state.n_R,
            config.min_new_observations,
            config.min_old_observations,
        )
        probe = construct_probe_loss(
            features=features,
            native_logits=native_logits,
            angular_scores=angular_scores,
            targets=targets,
            classifier_head=self.head,
            blindness=blindness,
            pair_supported=pair_supported,
            class_lambdas=state.dual.detach(),
            num_old_classes=num_old_classes,
            batch_size=batch_size,
            pair_temperature=config.pair_temperature,
            trigger_base=config.trigger_base,
            trigger_floor=config.trigger_floor,
            priority_smoothing=config.blindness_norm,
            min_margin=config.min_margin,
            max_movement=config.max_movement,
            max_angle=config.max_angle,
            restoration_increment=config.restoration_increment,
            probe_temperature=config.probe_temperature,
            weight_floor=config.probe_weight_floor,
        )
        return BSPRStepOutput(
            probe.loss,
            risk_loss,
            old_fraction,
            delayed,
            probe.accepted,
        )

    def commit_post_step_statistics(self, stats: Dict[str, torch.Tensor]) -> None:
        config = self.config
        state = self.state
        required = (
            "pressure_values",
            "pressure_observed",
            "pressure_counts",
            "restoration_values",
            "restoration_observed",
            "restoration_counts",
            "risk_values",
            "risk_observed",
        )
        missing = [name for name in required if name not in stats]
        if missing:
            raise ValueError("missing BSPR statistics: {}".format(missing))

        with torch.no_grad():
            state.E, state.E_initialized = update_masked_ema(
                state.E,
                state.E_initialized,
                stats["pressure_values"],
                stats["pressure_observed"],
                config.ema_pressure,
            )
            state.R, state.R_initialized = update_masked_ema(
                state.R,
                state.R_initialized,
                stats["restoration_values"],
                stats["restoration_observed"],
                config.ema_restoration,
            )
            state.n_E.add_(stats["pressure_counts"].to(dtype=torch.long))
            state.n_R.add_(stats["restoration_counts"].to(dtype=torch.long))

            ids = state.old_class_ids
            current = state.risk_ema.index_select(0, ids)
            initialized = state.risk_initialized.index_select(0, ids)
            observed = stats["risk_observed"].bool() & initialized
            updated = update_risk_ema(
                current,
                stats["risk_values"],
                observed,
                config.ema_restoration,
                initialized,
            )
            state.risk_ema.index_copy_(0, ids, updated)
            updated_dual = update_dual_variables(
                state.dual.index_select(0, ids),
                updated,
                state.rho.index_select(0, ids),
                observed,
                config.dual_step,
                config.dual_max,
            )
            state.dual.index_copy_(0, ids, updated_dual)
            state.stage_step += 1
            state.global_step += 1
