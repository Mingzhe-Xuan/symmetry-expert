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


def run(device: torch.device) -> dict:
    torch.manual_seed(7)
    module = SymmetricContraction(
        irreps_in=o3.Irreps("4x0e + 4x1o"),
        irreps_out=o3.Irreps("4x0e + 4x1o"),
        correlation=2,
        num_elements=2,
    ).to(device)
    module.configure_adapters("elora_clean", rank=2, alpha=2.0, num_experts=2)
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
    loss = output.square().mean()
    loss.backward()
    optimizer.step()

    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save(
            {
                "model": module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "update_mode": "elora_clean",
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
        restored.configure_adapters(
            "elora_clean", rank=2, alpha=2.0, num_experts=2
        )
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
        "loss": float(loss.detach().cpu()),
        "output_shape": list(output.shape),
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
            raise RuntimeError("Goal 0 expects the Guqq RTX 5090 compute capability 12.0")
        if not result["sm_120_supported"]:
            raise RuntimeError("installed PyTorch build does not include sm_120")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    result = run(torch.device(args.device))
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
