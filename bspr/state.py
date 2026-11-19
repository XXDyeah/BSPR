"""Serializable post-update state for the paper BSPR method."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Union

import torch


TensorLikeIds = Union[torch.Tensor, Iterable[int]]


class BSPRState(object):
    """Pairwise monitor state and persistent classwise risk controller state."""

    VERSION = 2

    def __init__(self, config_hash: str = "", device: Optional[torch.device] = None):
        self.version = self.VERSION
        self.config_hash = str(config_hash)
        self.stage_index = -1
        self.stage_step = 0
        self.global_step = 0
        self.state_rebuilt = False
        self._device = torch.device("cpu") if device is None else torch.device(device)

        self.old_class_ids = torch.empty(0, dtype=torch.long, device=self._device)
        self.new_class_ids = torch.empty(0, dtype=torch.long, device=self._device)
        self.E = torch.empty(0, 0, dtype=torch.float32, device=self._device)
        self.R = torch.empty(0, 0, dtype=torch.float32, device=self._device)
        self.n_E = torch.empty(0, 0, dtype=torch.long, device=self._device)
        self.n_R = torch.empty(0, 0, dtype=torch.long, device=self._device)
        self.E_initialized = torch.empty(0, 0, dtype=torch.bool, device=self._device)
        self.R_initialized = torch.empty(0, 0, dtype=torch.bool, device=self._device)

        self.reference_risk = torch.empty(0, dtype=torch.float32, device=self._device)
        self.rho = torch.empty(0, dtype=torch.float32, device=self._device)
        self.risk_ema = torch.empty(0, dtype=torch.float32, device=self._device)
        self.dual = torch.empty(0, dtype=torch.float32, device=self._device)
        self.risk_initialized = torch.empty(0, dtype=torch.bool, device=self._device)

    @property
    def device(self) -> torch.device:
        return self._device

    def _ids(self, values: TensorLikeIds) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            result = values.detach().to(device=self._device, dtype=torch.long)
        else:
            result = torch.tensor(list(values), dtype=torch.long, device=self._device)
        if result.dim() != 1:
            raise ValueError("class ids must be one-dimensional")
        if result.numel() and torch.unique(result).numel() != result.numel():
            raise ValueError("class ids must not contain duplicates")
        return result

    def ensure_class_capacity(self, size: int) -> None:
        size = int(size)
        if size < 0:
            raise ValueError("class capacity must be non-negative")
        current = int(self.rho.numel())
        if size <= current:
            return
        extra = size - current
        float_zeros = torch.zeros(extra, dtype=torch.float32, device=self._device)
        bool_zeros = torch.zeros(extra, dtype=torch.bool, device=self._device)
        self.reference_risk = torch.cat((self.reference_risk, float_zeros.clone()))
        self.rho = torch.cat((self.rho, float_zeros.clone()))
        self.risk_ema = torch.cat((self.risk_ema, float_zeros.clone()))
        self.dual = torch.cat((self.dual, float_zeros.clone()))
        self.risk_initialized = torch.cat((self.risk_initialized, bool_zeros))

    def begin_stage(
        self,
        stage_index: int,
        old_class_ids: TensorLikeIds,
        new_class_ids: TensorLikeIds,
    ) -> None:
        """Reset only the stage-local old-by-new monitor matrices."""

        old_ids = self._ids(old_class_ids)
        new_ids = self._ids(new_class_ids)
        if old_ids.numel() and new_ids.numel():
            if bool(old_ids.view(-1, 1).eq(new_ids.view(1, -1)).any().item()):
                raise ValueError("old and new class ids must be disjoint")
        max_id = -1
        if old_ids.numel():
            max_id = max(max_id, int(old_ids.max().item()))
        if new_ids.numel():
            max_id = max(max_id, int(new_ids.max().item()))
        self.ensure_class_capacity(max_id + 1)

        self.stage_index = int(stage_index)
        self.stage_step = 0
        self.old_class_ids = old_ids
        self.new_class_ids = new_ids
        shape = (int(old_ids.numel()), int(new_ids.numel()))
        self.E = torch.zeros(shape, dtype=torch.float32, device=self._device)
        self.R = torch.zeros(shape, dtype=torch.float32, device=self._device)
        self.n_E = torch.zeros(shape, dtype=torch.long, device=self._device)
        self.n_R = torch.zeros(shape, dtype=torch.long, device=self._device)
        self.E_initialized = torch.zeros(shape, dtype=torch.bool, device=self._device)
        self.R_initialized = torch.zeros(shape, dtype=torch.bool, device=self._device)

    def set_class_references(
        self,
        class_references: Dict[int, float],
        budget_increment: float,
    ) -> Dict[str, int]:
        """Record each class reference once and initialize its controller.

        The budget is ``min(1, reference + increment)``.  Repeated calls for a
        class are ignored so later exemplar or model changes cannot redefine
        the fixed first-stage reference.
        """

        if budget_increment < 0:
            raise ValueError("risk budget increment must be non-negative")
        if not class_references:
            return {"initialized": 0, "already_initialized": 0}
        max_id = max(int(class_id) for class_id in class_references)
        self.ensure_class_capacity(max_id + 1)
        initialized_count = 0
        existing_count = 0
        with torch.no_grad():
            for raw_class_id, raw_reference in class_references.items():
                class_id = int(raw_class_id)
                reference = float(raw_reference)
                if not 0.0 <= reference <= 1.0:
                    raise ValueError(
                        "half-Brier reference for class {} is outside [0, 1]".format(
                            class_id
                        )
                    )
                if bool(self.risk_initialized[class_id].item()):
                    existing_count += 1
                    continue
                self.reference_risk[class_id] = reference
                self.rho[class_id] = min(1.0, reference + float(budget_increment))
                self.risk_ema[class_id] = reference
                self.dual[class_id] = 0.0
                self.risk_initialized[class_id] = True
                initialized_count += 1
        return {
            "initialized": initialized_count,
            "already_initialized": existing_count,
        }

    def to(self, device: Union[str, torch.device]) -> "BSPRState":
        self._device = torch.device(device)
        for name in self._tensor_names():
            setattr(self, name, getattr(self, name).detach().to(self._device))
        return self

    @staticmethod
    def _tensor_names():
        return (
            "old_class_ids",
            "new_class_ids",
            "E",
            "R",
            "n_E",
            "n_R",
            "E_initialized",
            "R_initialized",
            "reference_risk",
            "rho",
            "risk_ema",
            "dual",
            "risk_initialized",
        )

    def _tensor_state(self, cpu: bool) -> Dict[str, torch.Tensor]:
        result = {}
        for name in self._tensor_names():
            value = getattr(self, name).detach().clone()
            result[name] = value.cpu() if cpu else value
        return result

    def state_dict(self) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "config_hash": self.config_hash,
            "stage_index": self.stage_index,
            "stage_step": self.stage_step,
            "global_step": self.global_step,
            "state_rebuilt": self.state_rebuilt,
        }
        result.update(self._tensor_state(cpu=False))
        return result

    def detach_for_checkpoint(self) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "config_hash": self.config_hash,
            "stage_index": self.stage_index,
            "stage_step": self.stage_step,
            "global_step": self.global_step,
            "state_rebuilt": self.state_rebuilt,
        }
        result.update(self._tensor_state(cpu=True))
        return result

    def load_state_dict(
        self,
        state_dict: Dict[str, Any],
        strict_config: bool = True,
    ) -> None:
        version = int(state_dict.get("version", -1))
        if version != self.VERSION:
            raise ValueError(
                "unsupported BSPR state version {}; expected {}".format(
                    version, self.VERSION
                )
            )
        loaded_hash = str(state_dict.get("config_hash", ""))
        if (
            strict_config
            and self.config_hash
            and loaded_hash
            and loaded_hash != self.config_hash
        ):
            raise ValueError("BSPR companion state configuration hash does not match")
        missing = [name for name in self._tensor_names() if name not in state_dict]
        if missing:
            raise ValueError("BSPR state is missing tensors: {}".format(missing))

        self.version = version
        if loaded_hash:
            self.config_hash = loaded_hash
        self.stage_index = int(state_dict.get("stage_index", -1))
        self.stage_step = int(state_dict.get("stage_step", 0))
        self.global_step = int(state_dict.get("global_step", 0))
        self.state_rebuilt = bool(state_dict.get("state_rebuilt", False))
        long_names = {"old_class_ids", "new_class_ids", "n_E", "n_R"}
        bool_names = {"E_initialized", "R_initialized", "risk_initialized"}
        for name in self._tensor_names():
            value = state_dict[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError("BSPR state field {} is not a tensor".format(name))
            dtype = (
                torch.long if name in long_names
                else torch.bool if name in bool_names
                else torch.float32
            )
            setattr(
                self,
                name,
                value.detach().clone().to(device=self._device, dtype=dtype),
            )
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        pair_shape = (int(self.old_class_ids.numel()), int(self.new_class_ids.numel()))
        for name in ("E", "R", "n_E", "n_R", "E_initialized", "R_initialized"):
            if tuple(getattr(self, name).shape) != pair_shape:
                raise ValueError("BSPR pairwise state has inconsistent shape")
        capacity = int(self.rho.numel())
        for name in (
            "reference_risk",
            "risk_ema",
            "dual",
            "risk_initialized",
        ):
            if int(getattr(self, name).numel()) != capacity:
                raise ValueError("BSPR classwise state has inconsistent capacity")
        if self.old_class_ids.numel() and (
            torch.unique(self.old_class_ids).numel() != self.old_class_ids.numel()
        ):
            raise ValueError("BSPR old class ids contain duplicates")
        if self.new_class_ids.numel() and (
            torch.unique(self.new_class_ids).numel() != self.new_class_ids.numel()
        ):
            raise ValueError("BSPR new class ids contain duplicates")
        if self.old_class_ids.numel() and self.new_class_ids.numel() and bool(
            self.old_class_ids.view(-1, 1).eq(
                self.new_class_ids.view(1, -1)
            ).any().item()
        ):
            raise ValueError("BSPR old and new class ids overlap")
        for name in ("E", "R", "reference_risk", "rho", "risk_ema"):
            value = getattr(self, name)
            if not bool(torch.isfinite(value).all().item()):
                raise ValueError("BSPR {} contains non-finite values".format(name))
            if bool(((value < 0.0) | (value > 1.0)).any().item()):
                raise ValueError("BSPR {} is outside [0, 1]".format(name))
        if bool((self.dual < 0.0).any().item()) or not bool(
            torch.isfinite(self.dual).all().item()
        ):
            raise ValueError("BSPR dual state must be finite and non-negative")
        if bool((self.n_E < 0).any().item()) or bool((self.n_R < 0).any().item()):
            raise ValueError("BSPR observation counts must be non-negative")
