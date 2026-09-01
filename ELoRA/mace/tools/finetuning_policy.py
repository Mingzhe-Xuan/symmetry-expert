"""Explicit Goal-0 fine-tuning configuration and parameter accounting.

The upstream branch toggled parameters from string matches inside every training
step.  This module makes the policy a validated, serializable object and applies
it once before optimizer construction.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import torch


UPDATE_MODES = ("dense", "elora_clean", "elora_paper")
SCOPES = ("readout", "tail_1", "no_first", "full")
ROUTERS = (
    "shared",
    "crystal_system",
    "point_group",
    "space_group",
    "learned",
    "random_control",
)


@dataclass(frozen=True)
class FineTuningConfig:
    update_mode: str = "elora_clean"
    scope: str = "full"
    router: str = "shared"
    rank: int = 16
    alpha: float = 16.0
    num_experts: int = 1
    expert_map: Mapping[str, int] = field(default_factory=dict)
    train_size: Optional[int] = None
    seed: int = 123
    split_manifest: Optional[str] = None

    def validate(self) -> "FineTuningConfig":
        if self.update_mode not in UPDATE_MODES:
            raise ValueError("update_mode must be one of " + repr(UPDATE_MODES))
        if self.scope not in SCOPES:
            raise ValueError("scope must be one of " + repr(SCOPES))
        if self.router not in ROUTERS:
            raise ValueError("router must be one of " + repr(ROUTERS))
        if self.rank <= 0:
            raise ValueError("rank must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.train_size is not None and self.train_size <= 0:
            raise ValueError("train_size must be positive when supplied")
        if self.router == "shared" and self.num_experts != 1:
            raise ValueError("shared routing requires num_experts=1")
        if self.router not in ("shared", "learned"):
            values = sorted(set(int(value) for value in self.expert_map.values()))
            if values != list(range(self.num_experts)):
                raise ValueError(
                    "expert_map values must cover contiguous ids [0, num_experts)"
                )
        return self

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["expert_map"] = dict(sorted(self.expert_map.items()))
        return result


def load_expert_map(path: Optional[str]) -> Dict[str, int]:
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("expert_map must be a JSON object")
    return {str(key): int(value) for key, value in payload.items()}


def file_sha256(path: Optional[str]) -> Optional[str]:
    if path is None:
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scope_indices(model: torch.nn.Module, scope: str) -> Dict[str, Set[int]]:
    num_interactions = len(getattr(model, "interactions", []))
    num_products = len(getattr(model, "products", []))
    if scope == "readout":
        interaction_indices: Set[int] = set()
        product_indices: Set[int] = set()
    elif scope == "tail_1":
        interaction_indices = {num_interactions - 1} if num_interactions else set()
        product_indices = {num_products - 1} if num_products else set()
    elif scope == "no_first":
        interaction_indices = set(range(1, num_interactions))
        product_indices = set(range(1, num_products))
    else:
        interaction_indices = set(range(num_interactions))
        product_indices = set(range(num_products))
    return {"interactions": interaction_indices, "products": product_indices}


def _indexed_prefix(name: str, prefix: str, indices: Set[int]) -> bool:
    for index in indices:
        if name.startswith("{}.{}.".format(prefix, index)):
            return True
    return False


def configure_finetuning(
    model: torch.nn.Module, config: FineTuningConfig
) -> List[Dict[str, Any]]:
    """Configure expert banks and apply the gradient whitelist once."""
    config.validate()
    selected = _scope_indices(model, config.scope)

    for product_index, product in enumerate(getattr(model, "products", [])):
        contractions = getattr(product, "symmetric_contractions", None)
        if contractions is None:
            continue
        # Unselected blocks keep a disabled zero adapter. Shared-Dense updates
        # original parameters directly and therefore must not also allocate or
        # train a delta bank for the same weight.
        active = product_index in selected["products"]
        uses_expert_bank = active and not (
            config.update_mode == "dense" and config.router == "shared"
        )
        contractions.configure_adapters(
            config.update_mode if uses_expert_bank else "elora_clean",
            config.rank,
            config.alpha,
            config.num_experts if uses_expert_bank else 1,
        )
        if not uses_expert_bank:
            contractions.disable_adapters()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    for name, parameter in model.named_parameters():
        is_readout = name.startswith("readouts.") or name.startswith("scale_shift.")
        is_router = name.startswith("expert_router.")
        in_interaction = _indexed_prefix(
            name, "interactions", selected["interactions"]
        )
        in_product = _indexed_prefix(name, "products", selected["products"])
        is_adapter = any(
            marker in name
            for marker in ("lora_A_bank", "lora_B_bank", "expert_delta_bank")
        )

        trainable = False
        if is_router:
            trainable = config.router == "learned"
        elif is_readout:
            # Every registered scope includes the dense task readouts.  The
            # paper-compatible mode adds the historical radial/contraction
            # updates; it does not change this common scope definition.
            trainable = True
        elif config.update_mode == "dense" and config.router == "shared":
            trainable = (in_interaction or in_product) and not is_adapter
        elif is_adapter and in_product:
            trainable = parameter.numel() > 0
        elif config.update_mode == "elora_paper":
            trainable = (
                name.startswith("radial_embedding.")
                or (
                    in_product
                    and ".symmetric_contractions." in name
                    and ".weights_max" not in name
                    and not name.endswith("expert_delta_bank")
                )
            )
        parameter.requires_grad_(trainable)

    manifest = build_parameter_manifest(model, config)
    if not any(row["trainable"] for row in manifest):
        raise ValueError("fine-tuning policy selected no trainable parameters")
    return manifest


def build_parameter_manifest(
    model: torch.nn.Module, config: FineTuningConfig
) -> List[Dict[str, Any]]:
    selected = _scope_indices(model, config.scope)
    rows: List[Dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        owner = "shared"
        if any(
            marker in name
            for marker in ("lora_A_bank", "lora_B_bank", "expert_delta_bank")
        ):
            owner = "expert_delta"
        elif name.startswith("readouts."):
            owner = "readout"
        scope = "outside"
        if name.startswith("readouts.") or name.startswith("scale_shift."):
            scope = "readout"
        elif _indexed_prefix(name, "products", selected["products"]):
            scope = config.scope
        elif _indexed_prefix(name, "interactions", selected["interactions"]):
            scope = config.scope
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "scope": scope,
                "owner": owner,
                "trainable": bool(parameter.requires_grad),
                "numel": int(parameter.numel()),
            }
        )
    return rows


def optimizer_parameter_names(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> List[str]:
    name_by_id = {id(parameter): name for name, parameter in model.named_parameters()}
    names: List[str] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            names.append(name_by_id.get(id(parameter), "<unregistered>"))
    return sorted(names)


def parameter_statistics(
    model: torch.nn.Module, optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    parameters = list(model.parameters())
    unique = {id(parameter): parameter for parameter in parameters}
    trainable = [parameter for parameter in unique.values() if parameter.requires_grad]
    nonzero_grad = [
        parameter
        for parameter in unique.values()
        if parameter.grad is not None and bool(torch.count_nonzero(parameter.grad))
    ]
    stats: Dict[str, Any] = {
        "parameter_objects": len(parameters),
        "unique_parameter_objects": len(unique),
        "stored_parameters": sum(parameter.numel() for parameter in unique.values()),
        "stored_bytes": sum(
            parameter.numel() * parameter.element_size()
            for parameter in unique.values()
        ),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "nonzero_gradient_parameters": sum(
            parameter.numel() for parameter in nonzero_grad
        ),
    }
    if optimizer is not None:
        optimizer_parameters = {
            id(parameter): parameter
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        stats["optimizer_parameters"] = sum(
            parameter.numel() for parameter in optimizer_parameters.values()
        )
        stats["optimizer_parameter_names"] = optimizer_parameter_names(model, optimizer)
    return stats


def readiness_metadata(
    config: FineTuningConfig,
    code_version: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "config": config.to_dict(),
        "expert_map": dict(sorted(config.expert_map.items())),
        "split_manifest": config.split_manifest,
        "split_manifest_sha256": file_sha256(config.split_manifest),
        "code_version": code_version,
        "seed": config.seed,
    }


def write_json(path: str, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
