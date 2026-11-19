"""Stage-boundary companion checkpoint helpers for BSPR."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .config import BSPRConfig
from .state import BSPRState


def companion_filename(prefix: str, run: int, iteration: int) -> str:
    """Return the companion filename paired with one completed LUCIR stage."""

    stage_filename = "run_{}_iteration_{}_bspr_state.pth".format(run, iteration)
    if not prefix:
        return stage_filename
    return "{}_{}".format(prefix, stage_filename)


def save_bspr_checkpoint(
    path: str,
    state: BSPRState,
    config: BSPRConfig,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Atomically save detached state, resolved config, and stage metadata."""

    directory = os.path.dirname(os.path.abspath(path))
    if directory and not os.path.isdir(directory):
        os.makedirs(directory)
    payload = {
        "format": "bspr_companion",
        "format_version": 1,
        "state": state.detach_for_checkpoint(),
        "config": config.to_dict(),
        "metadata": {} if metadata is None else dict(metadata),
    }
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_bspr_checkpoint(
    path: str,
    state: BSPRState,
    map_location: Optional[Union[str, torch.device]] = None,
    strict_config: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a companion checkpoint into ``state`` and return config/metadata."""

    if not os.path.isfile(path):
        raise FileNotFoundError("BSPR companion state not found: {}".format(path))
    payload = torch.load(path, map_location=map_location)
    if not isinstance(payload, dict) or payload.get("format") != "bspr_companion":
        raise ValueError("not a BSPR companion checkpoint: {}".format(path))
    if int(payload.get("format_version", -1)) != 1:
        raise ValueError("unsupported BSPR companion checkpoint version")
    if "state" not in payload:
        raise ValueError("BSPR companion checkpoint has no state")
    state.load_state_dict(payload["state"], strict_config=strict_config)
    config = payload.get("config", {})
    metadata = payload.get("metadata", {})
    if not isinstance(config, dict) or not isinstance(metadata, dict):
        raise ValueError("BSPR companion config and metadata must be dictionaries")
    return config, metadata


def validate_bspr_checkpoint_identity(
    state: BSPRState,
    metadata: Dict[str, Any],
    expected_run: int,
    expected_iteration: int,
    expected_model_checkpoint: str,
    num_seen_classes: int,
    num_new_classes: int,
    incremental_stage: bool = True,
) -> None:
    """Reject a companion copied from the wrong run, stage, or model.

    Shape validation internal to :class:`BSPRState` is necessary but cannot
    detect a well-formed companion with the wrong filename.  This identity
    check binds the dynamic state to the LUCIR checkpoint being resumed.
    """

    expected = {
        "run": int(expected_run),
        "iteration": int(expected_iteration),
        "model_checkpoint": os.path.basename(expected_model_checkpoint),
    }
    for name, value in expected.items():
        if name not in metadata:
            raise ValueError("BSPR companion metadata is missing {!r}".format(name))
        observed = metadata[name]
        if name in ("run", "iteration"):
            try:
                observed = int(observed)
            except (TypeError, ValueError):
                pass
        else:
            observed = os.path.basename(str(observed))
        if observed != value:
            raise ValueError(
                "BSPR companion {} mismatch: observed {!r}, expected {!r}".format(
                    name, observed, value
                )
            )

    if int(state.stage_index) != int(expected_iteration):
        raise ValueError(
            "BSPR companion stage_index {} does not match iteration {}".format(
                state.stage_index, expected_iteration
            )
        )

    if not incremental_stage:
        if state.old_class_ids.numel() or state.new_class_ids.numel():
            raise ValueError("base-stage BSPR companion must not contain pairwise ids")
        return

    num_seen_classes = int(num_seen_classes)
    num_new_classes = int(num_new_classes)
    num_old_classes = num_seen_classes - num_new_classes
    if num_old_classes < 1 or num_new_classes < 1:
        raise ValueError("invalid old/new class counts for BSPR resume validation")
    expected_old = torch.arange(
        num_old_classes, dtype=torch.long, device=state.old_class_ids.device
    )
    expected_new = torch.arange(
        num_old_classes,
        num_seen_classes,
        dtype=torch.long,
        device=state.new_class_ids.device,
    )
    if not torch.equal(state.old_class_ids, expected_old):
        raise ValueError("BSPR companion old class ids do not match the loaded model")
    if not torch.equal(state.new_class_ids, expected_new):
        raise ValueError("BSPR companion new class ids do not match the loaded model")
    expected_pair_shape = (num_old_classes, num_new_classes)
    for name in ("E", "R", "n_E", "n_R", "E_initialized", "R_initialized"):
        if tuple(getattr(state, name).shape) != expected_pair_shape:
            raise ValueError(
                "BSPR companion {} shape does not match the loaded model".format(name)
            )
    if int(state.rho.numel()) < num_seen_classes:
        raise ValueError("BSPR companion classwise state is smaller than the loaded model")
