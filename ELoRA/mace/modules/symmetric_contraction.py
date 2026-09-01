###########################################################################################
# Implementation of the symmetric contraction algorithm presented in the MACE paper
# (Batatia et al, MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields , Eq.10 and 11)
# Authors: Ilyes Batatia
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import Dict, Optional, Union

import opt_einsum_fx
import torch
import torch.fx
from e3nn import o3
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode

from mace.tools.cg import U_matrix_real

BATCH_EXAMPLE = 10
ALPHABET = ["w", "x", "v", "n", "z", "r", "t", "y", "u", "o", "p", "s"]


@compile_mode("script")
class SymmetricContraction(CodeGenMixin, torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irreps_out: o3.Irreps,
        correlation: Union[int, Dict[str, int]],
        irrep_normalization: str = "component",
        path_normalization: str = "element",
        internal_weights: Optional[bool] = None,
        shared_weights: Optional[bool] = None,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        if irrep_normalization is None:
            irrep_normalization = "component"

        if path_normalization is None:
            path_normalization = "element"

        assert irrep_normalization in ["component", "norm", "none"]
        assert path_normalization in ["element", "path", "none"]

        self.irreps_in = o3.Irreps(irreps_in)
        self.irreps_out = o3.Irreps(irreps_out)

        del irreps_in, irreps_out

        if not isinstance(correlation, tuple):
            corr = correlation
            correlation = {}
            for irrep_out in self.irreps_out:
                correlation[irrep_out] = corr

        assert shared_weights or not internal_weights

        if internal_weights is None:
            internal_weights = True

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        del internal_weights, shared_weights

        self.contractions = torch.nn.ModuleList()
        self.routing_enabled = True
        for irrep_out in self.irreps_out:
            self.contractions.append(
                Contraction(
                    irreps_in=self.irreps_in,
                    irrep_out=o3.Irreps(str(irrep_out.ir)),
                    correlation=correlation[irrep_out],
                    internal_weights=self.internal_weights,
                    num_elements=num_elements,
                    weights=self.shared_weights,
                )
            )

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        outs = [contraction(x, y, None) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)

    def forward_with_experts(
        self, x: torch.Tensor, y: torch.Tensor, expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Apply graph-level expert deltas after ids have been expanded to nodes."""
        if not self.routing_enabled:
            return self.forward(x, y)
        outs = [contraction(x, y, expert_ids) for contraction in self.contractions]
        return torch.cat(outs, dim=-1)

    def forward_with_expert_weights(
        self, x: torch.Tensor, y: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        if not self.routing_enabled:
            return self.forward(x, y)
        outs = [
            contraction.forward_weighted(x, y, expert_weights)
            for contraction in self.contractions
        ]
        return torch.cat(outs, dim=-1)

    def configure_adapters(
        self,
        update_mode: str,
        rank: int,
        alpha: float,
        num_experts: int,
    ) -> None:
        self.routing_enabled = True
        for contraction in self.contractions:
            contraction.configure_adapter(update_mode, rank, alpha, num_experts)

    def disable_adapters(self) -> None:
        self.routing_enabled = False
        for contraction in self.contractions:
            contraction.disable_adapter()


@compile_mode("script")
class Contraction(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        irrep_out: o3.Irreps,
        correlation: int,
        internal_weights: bool = True,
        num_elements: Optional[int] = None,
        weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()

        self.num_features = irreps_in.count((0, 1))
        self.coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
        self.correlation = correlation
        dtype = torch.get_default_dtype()
        for nu in range(1, correlation + 1):
            U_matrix = U_matrix_real(
                irreps_in=self.coupling_irreps,
                irreps_out=irrep_out,
                correlation=nu,
                dtype=dtype,
            )[-1]
            self.register_buffer(f"U_matrix_{nu}", U_matrix)

        # Tensor contraction equations
        self.contractions_weighting = torch.nn.ModuleList()
        self.contractions_features = torch.nn.ModuleList()

        # Create weight for product basis
        self.weights = torch.nn.ParameterList([])

        for i in range(correlation, 0, -1):
            # Shapes definying
            num_params = self.U_tensors(i).size()[-1]
            num_equivariance = 2 * irrep_out.lmax + 1
            num_ell = self.U_tensors(i).size()[-2]

            if i == correlation:
                parse_subscript_main = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                    + ["ik,ekc,bci,be -> bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1) - 1)]
                )
                graph_module_main = torch.fx.symbolic_trace(
                    lambda x, y, w, z: torch.einsum(
                        "".join(parse_subscript_main), x, y, w, z
                    )
                )

                # Optimizing the contractions
                self.graph_opt_main = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_main,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights_max = w
                # Adapter parameters are created lazily by configure_adapter.
                # Keeping them absent from the default state_dict preserves strict
                # compatibility with upstream foundation checkpoints.
                self.adapter_kind = "none"
                self.alpha = 1.0
                self.r = 1
                self.num_experts = 1
                self.merged = False
                self.lora_A_bank = torch.nn.Parameter(
                    self.weights_max.new_empty((0,)), requires_grad=False
                )
                self.lora_B_bank = torch.nn.Parameter(
                    self.weights_max.new_empty((0,)), requires_grad=False
                )
                self.expert_delta_bank = torch.nn.Parameter(
                    self.weights_max.new_empty((0,)), requires_grad=False
                )
            else:
                # Generate optimized contractions equations
                parse_subscript_weighting = (
                    [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                    + ["k,ekc,be->bc"]
                    + [ALPHABET[j] for j in range(i + min(irrep_out.lmax, 1))]
                )
                parse_subscript_features = (
                    ["bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                    + ["i,bci->bc"]
                    + [ALPHABET[j] for j in range(i - 1 + min(irrep_out.lmax, 1))]
                )

                # Symbolic tracing of contractions
                graph_module_weighting = torch.fx.symbolic_trace(
                    lambda x, y, z: torch.einsum(
                        "".join(parse_subscript_weighting), x, y, z
                    )
                )
                graph_module_features = torch.fx.symbolic_trace(
                    lambda x, y: torch.einsum("".join(parse_subscript_features), x, y)
                )

                # Optimizing the contractions
                graph_opt_weighting = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_weighting,
                    example_inputs=(
                        torch.randn(
                            [num_equivariance] + [num_ell] * i + [num_params]
                        ).squeeze(0),
                        torch.randn((num_elements, num_params, self.num_features)),
                        torch.randn((BATCH_EXAMPLE, num_elements)),
                    ),
                )
                graph_opt_features = opt_einsum_fx.optimize_einsums_full(
                    model=graph_module_features,
                    example_inputs=(
                        torch.randn(
                            [BATCH_EXAMPLE, self.num_features, num_equivariance]
                            + [num_ell] * i
                        ).squeeze(2),
                        torch.randn((BATCH_EXAMPLE, self.num_features, num_ell)),
                    ),
                )
                self.contractions_weighting.append(graph_opt_weighting)
                self.contractions_features.append(graph_opt_features)
                # Parameters for the product basis
                w = torch.nn.Parameter(
                    torch.randn((num_elements, num_params, self.num_features))
                    / num_params
                )
                self.weights.append(w)
        if not internal_weights:
            self.weights = weights[:-1]
            self.weights_max = weights[-1]

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        # Upstream foundation checkpoints predate Goal-0 adapter banks. Empty
        # disabled banks are structural metadata, so inject only those absent
        # keys before delegating to PyTorch's strict loader.
        for name in ("lora_A_bank", "lora_B_bank", "expert_delta_bank"):
            key = prefix + name
            if key not in state_dict:
                state_dict[key] = getattr(self, name)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _adapter_delta(self, expert_idx: int) -> torch.Tensor:
        if self.merged:
            return torch.zeros_like(self.weights_max)
        if self.adapter_kind == "elora":
            return (
                self.alpha
                / float(self.r)
                * torch.matmul(
                    self.lora_A_bank[expert_idx], self.lora_B_bank[expert_idx]
                )
            )
        if self.adapter_kind == "dense":
            return self.expert_delta_bank[expert_idx]
        return torch.zeros_like(self.weights_max)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        expert_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if expert_ids is None:
            out = self.graph_opt_main(
                self.U_tensors(self.correlation),
                self.weights_max + self._adapter_delta(0),
                x,
                y,
            )
        else:
            if expert_ids.ndim != 1 or expert_ids.shape[0] != x.shape[0]:
                raise ValueError("expert_ids must contain one id per input row")
            if expert_ids.dtype != torch.long:
                expert_ids = expert_ids.to(dtype=torch.long)
            if bool(torch.any(expert_ids < 0)) or bool(
                torch.any(expert_ids >= self.num_experts)
            ):
                raise ValueError("expert id is outside the configured expert bank")
            out = self.graph_opt_main(
                self.U_tensors(self.correlation), self.weights_max, x, y
            )
            # The shared contraction runs once. Only each active delta branch is
            # grouped, then scattered back into the original node order.
            for expert_idx in range(self.num_experts):
                indices = torch.nonzero(expert_ids == expert_idx).flatten()
                if indices.numel() == 0:
                    continue
                expert_out = self.graph_opt_main(
                    self.U_tensors(self.correlation),
                    self._adapter_delta(expert_idx),
                    x.index_select(0, indices),
                    y.index_select(0, indices),
                )
                out.index_add_(0, indices, expert_out)
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1),
                weight,
                y,
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)

        return out.view(out.shape[0], -1)

    def forward_weighted(
        self, x: torch.Tensor, y: torch.Tensor, expert_weights: torch.Tensor
    ) -> torch.Tensor:
        if expert_weights.shape != (x.shape[0], self.num_experts):
            raise ValueError("expert_weights must have shape [rows, num_experts]")
        out = self.graph_opt_main(
            self.U_tensors(self.correlation), self.weights_max, x, y
        )
        for expert_idx in range(self.num_experts):
            delta_out = self.graph_opt_main(
                self.U_tensors(self.correlation),
                self._adapter_delta(expert_idx),
                x,
                y,
            )
            row_weights = expert_weights[:, expert_idx]
            while row_weights.ndim < delta_out.ndim:
                row_weights = row_weights.unsqueeze(-1)
            out = out + delta_out * row_weights
        for i, (weight, contract_weights, contract_features) in enumerate(
            zip(self.weights, self.contractions_weighting, self.contractions_features)
        ):
            c_tensor = contract_weights(
                self.U_tensors(self.correlation - i - 1), weight, y
            )
            c_tensor = c_tensor + out
            out = contract_features(c_tensor, x)
        return out.view(out.shape[0], -1)

    def U_tensors(self, nu: int):
        return dict(self.named_buffers())[f"U_matrix_{nu}"]

    def configure_adapter(
        self, update_mode: str, rank: int, alpha: float, num_experts: int
    ) -> None:
        """Configure a fresh expert bank without replacing the shared weight."""
        if update_mode not in ("dense", "elora_clean", "elora_paper"):
            raise ValueError("unsupported update_mode: " + update_mode)
        if rank <= 0:
            raise ValueError("rank must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.merged:
            self.unmerge_adapter()

        if (
            update_mode in ("elora_clean", "elora_paper")
            and self.adapter_kind == "elora"
            and self.r == rank
            and self.alpha == alpha
            and self.num_experts == num_experts
            and self.lora_A_bank is not None
            and self.lora_A_bank.numel() > 0
        ):
            # The constructor already created this exact zero-delta bank. Keep it
            # to preserve initialization/RNG compatibility with historical runs.
            return

        self.r = int(rank)
        self.alpha = float(alpha)
        self.num_experts = int(num_experts)
        num_elements, num_params, num_features = self.weights_max.shape
        if update_mode == "dense":
            self.adapter_kind = "dense"
            self.expert_delta_bank = torch.nn.Parameter(
                self.weights_max.new_zeros(
                    (num_experts, num_elements, num_params, num_features)
                )
            )
            self.lora_A_bank = torch.nn.Parameter(
                self.weights_max.new_empty((0,)), requires_grad=False
            )
            self.lora_B_bank = torch.nn.Parameter(
                self.weights_max.new_empty((0,)), requires_grad=False
            )
        else:
            self.adapter_kind = "elora"
            self.lora_A_bank = torch.nn.Parameter(
                torch.randn(
                    num_experts,
                    num_elements,
                    num_params,
                    rank,
                    dtype=self.weights_max.dtype,
                    device=self.weights_max.device,
                )
                / num_params
            )
            self.lora_B_bank = torch.nn.Parameter(
                self.weights_max.new_zeros((num_experts, rank, num_features))
            )
            self.expert_delta_bank = torch.nn.Parameter(
                self.weights_max.new_empty((0,)), requires_grad=False
            )

    def disable_adapter(self) -> None:
        if self.merged:
            self.unmerge_adapter()
        self.adapter_kind = "none"
        self.num_experts = 1
        self.lora_A_bank = torch.nn.Parameter(
            self.weights_max.new_empty((0,)), requires_grad=False
        )
        self.lora_B_bank = torch.nn.Parameter(
            self.weights_max.new_empty((0,)), requires_grad=False
        )
        self.expert_delta_bank = torch.nn.Parameter(
            self.weights_max.new_empty((0,)), requires_grad=False
        )

    def merge_adapter(self) -> None:
        """Merge a single expert reversibly; multi-expert inference stays unmerged."""
        if self.num_experts != 1:
            raise RuntimeError("cannot merge a multi-expert parameter bank")
        if not self.merged:
            with torch.no_grad():
                self.weights_max.add_(self._adapter_delta(0))
            self.merged = True

    def unmerge_adapter(self) -> None:
        if self.merged:
            self.merged = False
            with torch.no_grad():
                self.weights_max.sub_(self._adapter_delta(0))

    def merge_LoRA(self) -> None:
        """Backward-compatible alias used by the historical save path."""
        self.merge_adapter()


def upgrade_legacy_adapter_state(model: torch.nn.Module) -> torch.nn.Module:
    """Add disabled adapter state to models pickled before Goal-0 adapters."""
    for module in model.modules():
        if isinstance(module, SymmetricContraction) and not hasattr(
            module, "routing_enabled"
        ):
            module.routing_enabled = False
        if not isinstance(module, Contraction):
            continue
        if not hasattr(module, "adapter_kind"):
            module.adapter_kind = "none"
        if not hasattr(module, "alpha"):
            module.alpha = 1.0
        if not hasattr(module, "r"):
            module.r = 1
        if not hasattr(module, "num_experts"):
            module.num_experts = 1
        if not hasattr(module, "merged"):
            module.merged = False
        for name in ("lora_A_bank", "lora_B_bank", "expert_delta_bank"):
            if name not in module._parameters:
                module.register_parameter(
                    name,
                    torch.nn.Parameter(
                        module.weights_max.new_empty((0,)), requires_grad=False
                    ),
                )
    return model
