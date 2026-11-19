"""Command-line configuration for the paper BSPR method."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class BSPRConfig:
    """Resolved optimization settings for BSPR."""

    enabled: bool = False
    allow_state_rebuild: bool = False

    ema_pressure: float = 0.90
    ema_restoration: float = 0.95
    pair_temperature: float = 0.20
    min_new_observations: int = 32
    min_old_observations: int = 8
    trigger_base: float = 0.10
    trigger_floor: float = 0.05
    blindness_norm: float = 0.10
    max_angle: float = 0.35
    min_margin: float = 0.05
    max_movement: float = 0.75
    restoration_increment: float = 0.10
    probe_temperature: float = 0.10
    probe_weight_floor: float = 0.05
    probe_coefficient: float = 1.0
    risk_budget_increment: float = 0.10
    dual_step: float = 0.02
    dual_max: float = 1.0
    risk_coefficient: float = 1.0
    reference_views: int = 8

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def algorithm_dict(self) -> Dict[str, Any]:
        result = self.to_dict()
        for name in ("enabled", "allow_state_rebuild"):
            result.pop(name, None)
        return result

    def stable_hash(self) -> str:
        payload = json.dumps(
            self.algorithm_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def add_bspr_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the paper method and its hyperparameters."""

    modules = parser.add_argument_group("BSPR modules")
    modules.add_argument("--bspr", action="store_true", help="enable full BSPR")
    modules.add_argument("--bspr_allow_state_rebuild", action="store_true")

    hyper = parser.add_argument_group("BSPR hyperparameters")
    hyper.add_argument("--bspr_ema_pressure", type=float, default=0.90)
    hyper.add_argument("--bspr_ema_restoration", type=float, default=0.95)
    hyper.add_argument("--bspr_pair_temperature", type=float, default=0.20)
    hyper.add_argument("--bspr_min_new_observations", type=int, default=32)
    hyper.add_argument("--bspr_min_old_observations", type=int, default=8)
    hyper.add_argument("--bspr_trigger_base", type=float, default=0.10)
    hyper.add_argument("--bspr_trigger_floor", type=float, default=0.05)
    hyper.add_argument("--bspr_blindness_norm", type=float, default=0.10)
    hyper.add_argument("--bspr_max_angle", type=float, default=0.35)
    hyper.add_argument("--bspr_min_margin", type=float, default=0.05)
    hyper.add_argument("--bspr_max_movement", type=float, default=0.75)
    hyper.add_argument("--bspr_restoration_increment", type=float, default=0.10)
    hyper.add_argument("--bspr_probe_temperature", type=float, default=0.10)
    hyper.add_argument("--bspr_probe_weight_floor", type=float, default=0.05)
    hyper.add_argument("--bspr_probe_coefficient", type=float, default=1.0)
    hyper.add_argument("--bspr_risk_budget_increment", type=float, default=0.10)
    hyper.add_argument("--bspr_dual_step", type=float, default=0.02)
    hyper.add_argument("--bspr_dual_max", type=float, default=1.0)
    hyper.add_argument("--bspr_risk_coefficient", type=float, default=1.0)
    hyper.add_argument("--bspr_reference_views", type=int, default=8)


def _positive(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value <= 0:
        raise ValueError("{} must be positive, got {}".format(name, value))


def _non_negative(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError("{} must be non-negative, got {}".format(name, value))


def _unit_interval(name: str, value: float, include_one: bool = True) -> None:
    upper_ok = value <= 1 if include_one else value < 1
    if not math.isfinite(float(value)) or value < 0 or not upper_ok:
        bracket = "[0, 1]" if include_one else "[0, 1)"
        raise ValueError("{} must be in {}, got {}".format(name, bracket, value))


def resolve_bspr_config(args: argparse.Namespace) -> BSPRConfig:
    """Resolve and validate all BSPR arguments."""

    for name in ("bspr_ema_pressure", "bspr_ema_restoration"):
        _unit_interval(name, float(getattr(args, name)), include_one=False)
    for name in (
        "bspr_pair_temperature",
        "bspr_blindness_norm",
        "bspr_probe_temperature",
        "bspr_dual_step",
        "bspr_dual_max",
    ):
        _positive(name, float(getattr(args, name)))
    for name in (
        "bspr_min_new_observations",
        "bspr_min_old_observations",
        "bspr_reference_views",
    ):
        _positive(name, int(getattr(args, name)))
    for name in (
        "bspr_trigger_base",
        "bspr_trigger_floor",
        "bspr_max_angle",
        "bspr_min_margin",
        "bspr_max_movement",
        "bspr_restoration_increment",
        "bspr_probe_weight_floor",
        "bspr_probe_coefficient",
        "bspr_risk_budget_increment",
        "bspr_risk_coefficient",
    ):
        _non_negative(name, float(getattr(args, name)))
    if args.bspr_trigger_floor > args.bspr_trigger_base:
        raise ValueError("--bspr_trigger_floor must not exceed --bspr_trigger_base")
    if not 0.0 <= args.bspr_trigger_floor <= args.bspr_trigger_base <= 1.0:
        raise ValueError("BSPR trigger thresholds must satisfy 0 <= floor <= base <= 1")
    if args.bspr_trigger_base <= 0.0:
        raise ValueError("--bspr_trigger_base must be positive")
    if not 0.0 < args.bspr_max_angle <= math.pi / 2.0:
        raise ValueError("--bspr_max_angle must be in (0, pi/2]")
    if not 0.0 < args.bspr_max_movement <= 1.0:
        raise ValueError("--bspr_max_movement must be in (0, 1]")
    if args.bspr_min_margin <= 0.0:
        raise ValueError("--bspr_min_margin must be positive")
    if args.bspr_probe_weight_floor <= 0.0:
        raise ValueError("--bspr_probe_weight_floor must be positive")
    if args.bspr_restoration_increment <= 0.0:
        raise ValueError("--bspr_restoration_increment must be positive")

    return BSPRConfig(
        enabled=bool(args.bspr),
        allow_state_rebuild=bool(args.bspr_allow_state_rebuild),
        ema_pressure=args.bspr_ema_pressure,
        ema_restoration=args.bspr_ema_restoration,
        pair_temperature=args.bspr_pair_temperature,
        min_new_observations=args.bspr_min_new_observations,
        min_old_observations=args.bspr_min_old_observations,
        trigger_base=args.bspr_trigger_base,
        trigger_floor=args.bspr_trigger_floor,
        blindness_norm=args.bspr_blindness_norm,
        max_angle=args.bspr_max_angle,
        min_margin=args.bspr_min_margin,
        max_movement=args.bspr_max_movement,
        restoration_increment=args.bspr_restoration_increment,
        probe_temperature=args.bspr_probe_temperature,
        probe_weight_floor=args.bspr_probe_weight_floor,
        probe_coefficient=args.bspr_probe_coefficient,
        risk_budget_increment=args.bspr_risk_budget_increment,
        dual_step=args.bspr_dual_step,
        dual_max=args.bspr_dual_max,
        risk_coefficient=args.bspr_risk_coefficient,
        reference_views=args.bspr_reference_views,
    )
