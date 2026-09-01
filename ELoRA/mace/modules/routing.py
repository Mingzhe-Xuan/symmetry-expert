"""Configuration-level routers for expert delta banks."""

from typing import Dict, Mapping, Optional, Tuple

import torch


class ExpertRouter(torch.nn.Module):
    """Route whole configurations, never individual atoms.

    Deterministic and random-control labels are frozen into the data manifest and
    supplied as contiguous ``label_ids``.  Learned routing accepts invariant
    configuration features only; the caller is responsible for recording the
    feature definition in the run configuration.
    """

    def __init__(
        self,
        router_type: str,
        num_experts: int,
        invariant_dim: int = 0,
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        allowed = (
            "shared",
            "crystal_system",
            "point_group",
            "space_group",
            "learned",
            "random_control",
        )
        if router_type not in allowed:
            raise ValueError("unsupported router_type: " + router_type)
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if router_type == "shared" and num_experts != 1:
            raise ValueError("shared router requires exactly one expert")
        if router_type == "learned" and invariant_dim <= 0:
            raise ValueError("learned router requires invariant_dim > 0")
        self.router_type = router_type
        self.num_experts = num_experts
        self.invariant_dim = invariant_dim
        if router_type == "learned":
            self.network = torch.nn.Sequential(
                torch.nn.Linear(invariant_dim, hidden_dim),
                torch.nn.SiLU(),
                torch.nn.Linear(hidden_dim, num_experts),
            )
        else:
            self.network = torch.nn.Identity()

    def forward(
        self,
        num_graphs: int,
        label_ids: Optional[torch.Tensor] = None,
        invariant_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.router_type == "shared":
            ids = torch.zeros(
                num_graphs,
                dtype=torch.long,
                device=(
                    invariant_features.device
                    if invariant_features is not None
                    else label_ids.device
                    if label_ids is not None
                    else None
                ),
            )
            logits = torch.zeros(
                num_graphs, 1, dtype=torch.get_default_dtype(), device=ids.device
            )
            return ids, logits, torch.ones_like(logits)
        if self.router_type == "learned":
            if invariant_features is None:
                raise ValueError("learned router requires invariant_features")
            if invariant_features.shape != (num_graphs, self.invariant_dim):
                raise ValueError("invalid invariant_features shape")
            logits = self.network(invariant_features)
            probabilities = torch.softmax(logits, dim=-1)
            ids = torch.argmax(probabilities, dim=-1)
            hard = torch.nn.functional.one_hot(ids, self.num_experts).to(
                dtype=probabilities.dtype
            )
            straight_through = hard + probabilities - probabilities.detach()
            return ids, logits, straight_through
        if label_ids is None:
            raise ValueError("deterministic router requires label_ids")
        ids = label_ids.reshape(-1).to(dtype=torch.long)
        if ids.numel() != num_graphs:
            raise ValueError("label_ids must contain one id per graph")
        if bool(torch.any(ids < 0)) or bool(torch.any(ids >= self.num_experts)):
            raise ValueError("label id is outside the frozen expert map")
        logits = torch.nn.functional.one_hot(ids, self.num_experts).to(
            dtype=torch.get_default_dtype()
        )
        return ids, logits, torch.nn.functional.one_hot(
            ids, self.num_experts
        ).to(dtype=torch.get_default_dtype())

    @staticmethod
    def load_balance_loss(logits: torch.Tensor) -> torch.Tensor:
        probabilities = torch.softmax(logits, dim=-1)
        mean_probability = probabilities.mean(dim=0)
        target = torch.full_like(mean_probability, 1.0 / mean_probability.numel())
        return torch.mean((mean_probability - target) ** 2)

    @staticmethod
    def diagnostics(logits: torch.Tensor) -> Dict[str, float]:
        probabilities = torch.softmax(logits, dim=-1)
        assignments = torch.argmax(probabilities, dim=-1)
        counts = torch.bincount(assignments, minlength=logits.shape[-1]).float()
        usage = counts / max(1, assignments.numel())
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
        return {
            "entropy": float(entropy.mean().detach().cpu()),
            "collapse_rate": float(usage.max().detach().cpu()),
            "unused_experts": float(torch.count_nonzero(counts == 0).detach().cpu()),
        }


def encode_expert_labels(
    labels: torch.Tensor, expert_map: Mapping[int, int]
) -> torch.Tensor:
    """Apply a frozen categorical map and fail on unseen benchmark labels."""
    result = torch.empty_like(labels, dtype=torch.long)
    for source, target in expert_map.items():
        result[labels == int(source)] = int(target)
    known = torch.zeros_like(labels, dtype=torch.bool)
    for source in expert_map:
        known |= labels == int(source)
    if not bool(torch.all(known)):
        raise ValueError("encountered a category absent from the frozen expert_map")
    return result
