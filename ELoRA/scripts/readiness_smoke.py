"""Short CPU/GPU forward-backward-checkpoint-evaluation readiness smoke."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import torch

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mace-readiness-matplotlib")
)
import mace  # initializes the narrow e3nn/PyTorch safe-global compatibility
from e3nn import o3

from mace.modules.symmetric_contraction import SymmetricContraction

UPDATE_MODES = ("dense", "elora_clean", "elora_paper")


def _updated_bank(module: SymmetricContraction, update_mode: str):
    bank_name = "expert_delta_bank" if update_mode == "dense" else "lora_B_bank"
    return bank_name, [getattr(item, bank_name) for item in module.contractions]


def run(device: torch.device, update_mode: str = "elora_clean") -> dict:
    torch.manual_seed(7)
    module = SymmetricContraction(
        irreps_in=o3.Irreps("4x0e + 4x1o"),
        irreps_out=o3.Irreps("4x0e + 4x1o"),
        correlation=2,
        num_elements=2,
    ).to(device)
    module.configure_adapters(update_mode, rank=2, alpha=2.0, num_experts=2)
    bank_name, updated_banks = _updated_bank(module, update_mode)
    banks_before = [bank.detach().clone() for bank in updated_banks]
    optimizer = torch.optim.Adam(
        [parameter for parameter in module.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    features = torch.randn(8, 4, 4, device=device)
    attrs = torch.nn.functional.one_hot(torch.arange(8, device=device) % 2, 2).to(
        features.dtype
    )
    expert_ids = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1], device=device)
    optimizer.zero_grad(set_to_none=True)
    output = module.forward_with_experts(features, attrs, expert_ids)
    if not bool(torch.isfinite(output).all()):
        raise RuntimeError(f"{update_mode} forward produced non-finite values")
    loss = output.square().mean()
    loss.backward()
    gradient_nonzero_by_expert = []
    for expert_idx in range(2):
        expert_gradients = [
            None if bank.grad is None else bank.grad[expert_idx]
            for bank in updated_banks
        ]
        if any(gradient is None for gradient in expert_gradients):
            raise RuntimeError(f"{update_mode} expert {expert_idx} has no gradient")
        if not all(
            bool(torch.isfinite(gradient).all()) for gradient in expert_gradients
        ):
            raise RuntimeError(
                f"{update_mode} expert {expert_idx} has non-finite gradients"
            )
        nonzero = sum(
            int(torch.count_nonzero(gradient).detach().cpu())
            for gradient in expert_gradients
        )
        if nonzero == 0:
            raise RuntimeError(f"{update_mode} expert {expert_idx} gradient is zero")
        gradient_nonzero_by_expert.append(nonzero)
    optimizer.step()
    parameter_delta_norm_by_expert = []
    for expert_idx in range(2):
        delta_norm = sum(
            float((bank.detach()[expert_idx] - before[expert_idx]).norm().cpu())
            for bank, before in zip(updated_banks, banks_before)
        )
        if not delta_norm > 0.0:
            raise RuntimeError(f"{update_mode} expert {expert_idx} did not update")
        parameter_delta_norm_by_expert.append(delta_norm)

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save(
            {
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "update_mode": update_mode,
                    "rank": 2,
                    "alpha": 2.0,
                    "num_experts": 2,
                    "seed": 7,
                },
            },
            checkpoint,
        )
        restored = SymmetricContraction(
            irreps_in=o3.Irreps("4x0e + 4x1o"),
            irreps_out=o3.Irreps("4x0e + 4x1o"),
            correlation=2,
            num_elements=2,
        ).to(device)
        restored.configure_adapters(update_mode, rank=2, alpha=2.0, num_experts=2)
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        restored.load_state_dict(payload["model"])
        restored.eval()
        with torch.no_grad():
            expected = module.forward_with_experts(features, attrs, expert_ids)
            actual = restored.forward_with_experts(features, attrs, expert_ids)
        if not torch.allclose(expected, actual, atol=1e-6, rtol=1e-6):
            raise RuntimeError("checkpoint restore changed smoke output")

    result = {
        "device": str(device),
        "update_mode": update_mode,
        "updated_bank": bank_name,
        "num_experts": 2,
        "loss": float(loss.detach().cpu()),
        "output_shape": list(output.shape),
        "gradient_nonzero_by_expert": gradient_nonzero_by_expert,
        "parameter_delta_norm_by_expert": parameter_delta_norm_by_expert,
        "checkpoint_restore": True,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        capability = torch.cuda.get_device_capability(device)
        kernel = torch.randn(256, 256, device=device) @ torch.randn(
            256, 256, device=device
        )
        torch.cuda.synchronize(device)
        result.update(
            {
                "torch_cuda": torch.version.cuda,
                "device_name": properties.name,
                "compute_capability": "{}.{}".format(*capability),
                "sm_120_supported": "sm_120" in torch.cuda.get_arch_list(),
                "cuda_kernel_sum": float(kernel.sum().cpu()),
            }
        )
        if capability != (12, 0):
            raise RuntimeError(
                "Goal 0 expects the Guqq RTX 5090 compute capability 12.0"
            )
        if not result["sm_120_supported"]:
            raise RuntimeError("installed PyTorch build does not include sm_120")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--update-mode", choices=UPDATE_MODES, default="elora_clean"
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(torch.device(args.device), args.update_mode)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
