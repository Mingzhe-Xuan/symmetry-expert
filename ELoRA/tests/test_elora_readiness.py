import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import mace  # initializes the narrow e3nn/PyTorch safe-global compatibility
from e3nn import o3

from mace import data, modules, tools
from mace.cli.run_train import _prepare_xyz_routing
from mace.modules.routing import ExpertRouter
from mace.modules.symmetric_contraction import (
    SymmetricContraction,
    upgrade_legacy_adapter_state,
)
from mace.readiness.dataset_statistics import analyze_dataset, classification_level
from mace.tools.checkpoint import CheckpointBuilder, CheckpointState
from mace.tools.finetuning_policy import (
    FineTuningConfig,
    configure_finetuning,
    parameter_statistics,
    readiness_metadata,
)
from mace.tools.train import _temporarily_disable_parameter_gradients, evaluate


def _contraction(num_experts=2, update_mode="elora_clean"):
    module = SymmetricContraction(
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        irreps_out=o3.Irreps("2x0e + 2x1o"),
        correlation=2,
        num_elements=2,
    )
    module.configure_adapters(update_mode, rank=2, alpha=2.0, num_experts=num_experts)
    return module


def _inputs(rows=6):
    generator = torch.Generator().manual_seed(4)
    return torch.randn(rows, 2, 4, generator=generator), torch.nn.functional.one_hot(
        torch.arange(rows) % 2, 2
    ).to(torch.get_default_dtype())


def test_zero_delta_and_mixed_expert_consistency():
    module = _contraction()
    x, attrs = _inputs()
    shared = module(x, attrs)
    ids = torch.tensor([0, 1, 0, 1, 1, 0])
    mixed = module.forward_with_experts(x, attrs, ids)
    assert torch.allclose(shared, mixed)

    for contraction in module.contractions:
        with torch.no_grad():
            contraction.lora_B_bank.copy_(
                torch.randn_like(contraction.lora_B_bank) * 0.1
            )
    mixed = module.forward_with_experts(x, attrs, ids)
    expected = torch.empty_like(mixed)
    for expert in (0, 1):
        indices = torch.nonzero(ids == expert).flatten()
        expected.index_copy_(
            0,
            indices,
            module.forward_with_experts(
                x.index_select(0, indices),
                attrs.index_select(0, indices),
                torch.full((indices.numel(),), expert, dtype=torch.long),
            ),
        )
    assert torch.allclose(mixed, expected, atol=1e-6, rtol=1e-6)


def test_unconfigured_and_legacy_adapter_state_are_foundation_compatible():
    module = SymmetricContraction(
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        irreps_out=o3.Irreps("2x0e + 2x1o"),
        correlation=2,
        num_elements=2,
    )
    legacy_state = {
        name: value
        for name, value in module.state_dict().items()
        if "_bank" not in name
    }
    clone = SymmetricContraction(
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        irreps_out=o3.Irreps("2x0e + 2x1o"),
        correlation=2,
        num_elements=2,
    )
    clone.load_state_dict(legacy_state, strict=True)
    x, attrs = _inputs()
    expected = clone(x, attrs)

    for contraction in clone.contractions:
        for name in (
            "adapter_kind",
            "alpha",
            "r",
            "num_experts",
            "merged",
            "lora_A_bank",
            "lora_B_bank",
            "expert_delta_bank",
        ):
            delattr(contraction, name)
    delattr(clone, "routing_enabled")
    upgrade_legacy_adapter_state(clone)
    assert torch.allclose(expected, clone(x, attrs))
    clone.configure_adapters("elora_clean", rank=2, alpha=2.0, num_experts=2)
    ids = torch.tensor([0, 1, 0, 1, 1, 0])
    assert torch.allclose(expected, clone.forward_with_experts(x, attrs, ids))


def test_expert_gradient_isolation_and_shared_uniqueness():
    module = _contraction()
    x, attrs = _inputs(4)
    ids = torch.zeros(4, dtype=torch.long)
    module.forward_with_experts(x, attrs, ids).sum().backward()
    for contraction in module.contractions:
        assert contraction.lora_B_bank.grad is not None
        assert torch.count_nonzero(contraction.lora_B_bank.grad[0]) > 0
        assert torch.count_nonzero(contraction.lora_B_bank.grad[1]) == 0
        base_ids = [
            id(parameter)
            for name, parameter in contraction.named_parameters()
            if name == "weights_max"
        ]
        assert len(base_ids) == len(set(base_ids)) == 1


def test_merge_unmerge_is_reversible_and_multi_expert_is_explicitly_disabled():
    module = _contraction(num_experts=1)
    x, attrs = _inputs()
    for contraction in module.contractions:
        with torch.no_grad():
            contraction.lora_B_bank.normal_()
    before = module(x, attrs)
    original = [item.weights_max.detach().clone() for item in module.contractions]
    for contraction in module.contractions:
        contraction.merge_adapter()
    assert torch.allclose(before, module(x, attrs), atol=1e-6, rtol=1e-6)
    for contraction in module.contractions:
        contraction.unmerge_adapter()
    assert torch.allclose(before, module(x, attrs), atol=1e-6, rtol=1e-6)
    for contraction, weight in zip(module.contractions, original):
        assert torch.allclose(contraction.weights_max, weight)

    multi = _contraction(num_experts=2)
    with pytest.raises(RuntimeError, match="multi-expert"):
        multi.contractions[0].merge_adapter()


def test_dense_expert_bank_isolation():
    module = _contraction(num_experts=2, update_mode="dense")
    x, attrs = _inputs(4)
    ids = torch.zeros(4, dtype=torch.long)
    module.forward_with_experts(x, attrs, ids).sum().backward()
    for contraction in module.contractions:
        assert contraction.expert_delta_bank.grad is not None
        assert torch.count_nonzero(contraction.expert_delta_bank.grad[0]) > 0
        assert torch.count_nonzero(contraction.expert_delta_bank.grad[1]) == 0


def test_scalar_vector_equivariance_with_expert_delta():
    module = _contraction(num_experts=2)
    x, attrs = _inputs(4)
    ids = torch.tensor([0, 1, 0, 1])
    for contraction in module.contractions:
        with torch.no_grad():
            contraction.lora_B_bank.normal_(std=0.1)
    rotation = o3.rand_matrix()
    coupling = o3.Irreps("0e + 1o").D_from_matrix(rotation)
    output_irreps = o3.Irreps("2x0e + 2x1o")
    output_rotation = output_irreps.D_from_matrix(rotation)
    rotated_x = torch.einsum("nfi,ji->nfj", x, coupling)
    actual = module.forward_with_experts(rotated_x, attrs, ids)
    expected = module.forward_with_experts(x, attrs, ids) @ output_rotation.T
    assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_mace_expert_energy_invariance_and_force_equivariance():
    """Exercise routing through a real MACE energy/force forward pass."""
    table = tools.AtomicNumberTable([1, 8])
    model = modules.MACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([0.0, 0.0]),
        avg_num_neighbors=3.0,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
    )
    configure_finetuning(
        model,
        FineTuningConfig(
            update_mode="elora_clean",
            scope="full",
            router="crystal_system",
            rank=2,
            alpha=2.0,
            num_experts=2,
            expert_map={"cubic": 0, "hexagonal": 1},
        ),
    )
    for product in model.products:
        for contraction in product.symmetric_contractions.contractions:
            with torch.no_grad():
                contraction.lora_B_bank.normal_(std=0.05)

    positions = np.array([[0.0, -1.0, 0.2], [0.9, 0.1, 0.0], [-0.2, 0.8, 0.1]])
    rotation = np.array(
        [[0.5, -np.sqrt(3.0) / 2.0, 0.0], [np.sqrt(3.0) / 2.0, 0.5, 0.0], [0.0, 0.0, 1.0]]
    )
    configurations = [
        data.Configuration(
            atomic_numbers=np.array([8, 1, 1]),
            positions=current,
            expert_id=1,
            router_features=np.array([3.0, 2.0, 1.0]),
        )
        for current in (positions, positions @ rotation.T)
    ]
    dataset = [
        data.AtomicData.from_config(config, z_table=table, cutoff=4.0)
        for config in configurations
    ]
    batch = next(
        iter(
            tools.torch_geometric.dataloader.DataLoader(
                dataset=dataset, batch_size=2, shuffle=False
            )
        )
    )
    output = model(batch.to_dict(), training=True)
    assert torch.allclose(output["energy"][0], output["energy"][1], atol=2e-6, rtol=2e-6)
    expected_forces = output["forces"][:3] @ torch.as_tensor(
        rotation.T, dtype=output["forces"].dtype
    )
    assert torch.allclose(output["forces"][3:], expected_forces, atol=2e-5, rtol=2e-5)

    # Graph features retain their graph dimension through PyG collation, and a
    # learned top-1 router participates in a real MACE backward pass.
    assert batch.router_features.shape == (2, 3)
    model.expert_router = ExpertRouter("learned", 2, invariant_dim=3)
    learned = model(batch.to_dict(), training=True)
    assert learned["router_logits"].shape == (2, 2)
    learned["energy"].sum().backward()
    assert all(parameter.grad is not None for parameter in model.expert_router.parameters())


def test_learned_router_top1_and_invariant_input_gradient():
    router = ExpertRouter("learned", 3, invariant_dim=4)
    features = torch.tensor(
        [[2.0, 3.0, 4.0, 5.0], [7.0, 11.0, 13.0, 17.0]], requires_grad=True
    )
    ids, logits, weights = router(2, invariant_features=features)
    assert ids.shape == (2,)
    assert torch.allclose(weights.detach().sum(-1), torch.ones(2))
    # A rigid transform or atom permutation does not change precomputed scalar
    # invariants, hence routing is deterministic under those transformations.
    ids_again, _, _ = router(2, invariant_features=features.clone())
    assert torch.equal(ids, ids_again)
    (weights * torch.arange(3, dtype=weights.dtype)).sum().backward()
    assert any(parameter.grad is not None for parameter in router.parameters())


@pytest.mark.parametrize(
    ("router_name", "label_field", "expert_map"),
    [
        ("crystal_system", "crystal_system", {"cubic": 0, "hexagonal": 1}),
        ("random_control", "random_control", {"group-a": 0, "group-b": 1}),
    ],
)
def test_frozen_hard_router_is_rigid_transform_and_permutation_invariant(
    router_name, label_field, expert_map
):
    positions = np.array(
        [[0.0, 0.1, 0.2], [0.8, -0.2, 0.0], [-0.1, 0.7, 0.3]]
    )
    rotation = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    permutation = np.array([2, 0, 1])
    transformed = (
        positions,
        positions @ rotation.T + np.array([1.5, -0.5, 2.0]),
        positions[permutation],
    )
    configurations = []
    for index, current_positions in enumerate(transformed):
        atomic_numbers = np.array([8, 1, 1])
        if index == 2:
            atomic_numbers = atomic_numbers[permutation]
        item = data.Configuration(
            atomic_numbers=atomic_numbers,
            positions=current_positions,
        )
        setattr(item, label_field, next(iter(expert_map)))
        configurations.append(item)
    collections = SimpleNamespace(train=configurations, valid=[], tests=[])
    config = FineTuningConfig(
        router=router_name,
        num_experts=2,
        expert_map=expert_map,
    ).validate()
    _prepare_xyz_routing(collections, config)
    label_ids = torch.tensor([item.expert_id for item in configurations])
    router = ExpertRouter(router_name, num_experts=2)
    routed_ids, _, _ = router(len(configurations), label_ids=label_ids)
    assert torch.equal(routed_ids, torch.zeros(3, dtype=torch.long))


def test_evaluation_restores_exact_gradient_policy_even_on_error():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 1, bias=False),
    )
    parameters = list(model.parameters())
    parameters[0].requires_grad_(False)
    expected = [parameter.requires_grad for parameter in parameters]
    with _temporarily_disable_parameter_gradients(model):
        assert not any(parameter.requires_grad for parameter in parameters)
    assert [parameter.requires_grad for parameter in parameters] == expected

    class _Batch:
        def to(self, _device):
            return self

        @staticmethod
        def to_dict():
            return {}

    class _FailingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frozen = torch.nn.Parameter(torch.ones(1), requires_grad=False)
            self.trainable = torch.nn.Parameter(torch.ones(1), requires_grad=True)

        def forward(self, *_args, **_kwargs):
            raise RuntimeError("intentional evaluation failure")

    failing = _FailingModel()
    failing_expected = [parameter.requires_grad for parameter in failing.parameters()]
    with pytest.raises(RuntimeError, match="intentional evaluation failure"):
        evaluate(
            failing,
            loss_fn=torch.nn.Identity(),
            data_loader=[_Batch()],
            output_args={"forces": False, "virials": False, "stress": False},
            device=torch.device("cpu"),
        )
    assert [parameter.requires_grad for parameter in failing.parameters()] == failing_expected


class _ConfigurableContractions(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights_max = torch.nn.Parameter(torch.ones(2, 2))
        self.lora_A_bank = torch.nn.Parameter(torch.zeros(1, 2, 1))
        self.lora_B_bank = torch.nn.Parameter(torch.zeros(1, 1, 2))
        self.expert_delta_bank = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def configure_adapters(self, update_mode, rank, alpha, num_experts):
        if update_mode == "dense":
            self.expert_delta_bank = torch.nn.Parameter(torch.zeros(num_experts, 2, 2))
            self.lora_A_bank = torch.nn.Parameter(torch.empty(0), requires_grad=False)
            self.lora_B_bank = torch.nn.Parameter(torch.empty(0), requires_grad=False)
        else:
            self.lora_A_bank = torch.nn.Parameter(torch.zeros(num_experts, 2, rank))
            self.lora_B_bank = torch.nn.Parameter(torch.zeros(num_experts, rank, 2))
            self.expert_delta_bank = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def disable_adapters(self):
        return None


class _Product(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.symmetric_contractions = _ConfigurableContractions()
        self.linear = torch.nn.Linear(2, 2, bias=False)


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.node_embedding = torch.nn.Linear(2, 2, bias=False)
        self.radial_embedding = torch.nn.Linear(2, 2, bias=False)
        self.interactions = torch.nn.ModuleList(
            [torch.nn.Linear(2, 2, bias=False), torch.nn.Linear(2, 2, bias=False)]
        )
        self.products = torch.nn.ModuleList([_Product(), _Product()])
        self.readouts = torch.nn.ModuleList([torch.nn.Linear(2, 1, bias=False)])


@pytest.mark.parametrize("scope", ["readout", "tail_1", "no_first", "full"])
@pytest.mark.parametrize("update_mode", ["dense", "elora_clean", "elora_paper"])
def test_scope_update_policy_and_parameter_manifest(scope, update_mode):
    model = _TinyModel()
    config = FineTuningConfig(update_mode=update_mode, scope=scope)
    manifest = configure_finetuning(model, config)
    assert manifest
    assert all(
        set(row) == {"name", "shape", "scope", "owner", "trainable", "numel"}
        for row in manifest
    )
    trainable = {row["name"] for row in manifest if row["trainable"]}
    assert any(name.startswith("readouts.") for name in trainable)
    if scope == "readout":
        assert not any("lora_" in name or "expert_delta" in name for name in trainable)
    if update_mode == "dense":
        assert not any("lora_" in name or "expert_delta" in name for name in trainable)
    optimizer = torch.optim.Adam(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    stats = parameter_statistics(model, optimizer)
    assert stats["optimizer_parameters"] == stats["trainable_parameters"]


def test_config_and_checkpoint_metadata_round_trip(tmp_path):
    split = tmp_path / "split.json"
    split.write_text('{"train_order": ["a"]}\n', encoding="utf-8")
    config = FineTuningConfig(split_manifest=str(split)).validate()
    metadata = readiness_metadata(config, code_version="abc123")
    model = torch.nn.Linear(2, 1)
    model.readiness_metadata = metadata
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    checkpoint = CheckpointBuilder.create_checkpoint(
        CheckpointState(model, optimizer, scheduler)
    )
    restored = torch.nn.Linear(2, 1)
    restored.readiness_metadata = metadata
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        restored_optimizer, gamma=0.9
    )
    CheckpointBuilder.load_checkpoint(
        CheckpointState(restored, restored_optimizer, restored_scheduler),
        checkpoint,
        strict=True,
    )
    assert restored.readiness_metadata == metadata
    assert torch.allclose(model(torch.ones(1, 2)), restored(torch.ones(1, 2)))


def _records():
    records = []
    for class_index, (crystal, point, space) in enumerate(
        [("cubic", "m-3m", "225"), ("hexagonal", "6/mmm", "194")]
    ):
        for index in range(6):
            records.append(
                {
                    "structure_id": "s{}-{}".format(class_index, index),
                    "parent_id": "p{}-{}".format(class_index, index // 2),
                    "fingerprint": "f{}-{}".format(class_index, index),
                    "crystal_system": crystal,
                    "point_group": point,
                    "space_group": space,
                    "atomic_numbers": [14, 8],
                    "num_atoms": 2 + index,
                    "energy_per_atom": -1.0 - class_index - index / 10,
                    "force_norm": 0.1 + index / 100,
                    "composition": "SiO",
                    "prototype": "proto{}".format(index % 2),
                    "source": "synthetic",
                    "valid_labels": True,
                }
            )
    return records


def test_dataset_statistics_outputs_and_group_split(tmp_path):
    result = analyze_dataset(
        _records(), str(tmp_path), minimum_class_size=3, fallback_threshold=2000, seed=7
    )
    assert result["classification_decision"]["primary_level"] == "space_group"
    required = {
        "dataset_summary.md",
        "class_counts.csv",
        "removed_classes.csv",
        "split_counts.csv",
        "dataset_manifest.json",
        "classification_decision.json",
    }
    assert required <= {path.name for path in tmp_path.iterdir()}
    figures = {path.name for path in (tmp_path / "figures").iterdir()}
    assert {
        "crystal_system_counts.png",
        "point_group_counts.png",
        "space_group_counts.png",
        "space_group_long_tail_log.png",
        "screening_before_after.png",
        "split_counts_stacked.png",
        "primary_property_boxplots.png",
        "primary_composition_heatmap.png",
        "parent_vs_configuration_counts.png",
    } <= figures
    manifest = json.loads((tmp_path / "dataset_manifest.json").read_text("utf-8"))
    parent_splits = {}
    for record in manifest["records"]:
        parent = record["parent_id"]
        parent_splits.setdefault(parent, set()).add(record["split"])
    assert all(len(splits) == 1 for splits in parent_splits.values())
    assert len(manifest["train_order"]) == sum(
        record["split"] == "train" for record in manifest["records"]
    )
    with open(tmp_path / "split_counts.csv", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            assert int(row["train"]) + int(row["valid"]) + int(row["test"]) == int(
                row["total"]
            )
    with open(tmp_path / "class_counts.csv", encoding="utf-8") as stream:
        class_rows = list(csv.DictReader(stream))
    required_columns = {
        "unique_proportion",
        "retained_proportion",
        "train",
        "valid",
        "test",
        "atoms_median",
        "atoms_q1",
        "atoms_q3",
        "energy_per_atom_mean",
        "energy_per_atom_std",
        "force_norm_mean",
        "force_norm_std",
    }
    assert required_columns <= set(class_rows[0])
    for level in ("crystal_system", "point_group", "space_group"):
        level_rows = [row for row in class_rows if row["category_level"] == level]
        assert sum(float(row["unique_proportion"]) for row in level_rows) == pytest.approx(1.0)
        assert sum(float(row["retained_proportion"]) for row in level_rows) == pytest.approx(1.0)
        for row in level_rows:
            assert sum(int(row[split]) for split in ("train", "valid", "test")) == int(
                row["retained_configurations"]
            )
    summary = (tmp_path / "dataset_summary.md").read_text("utf-8")
    assert "Element occurrence proportions" in summary
    assert "Data-source counts" in summary


def test_classification_gate_boundaries_and_pre_removal_source():
    assert classification_level(2000) == "space_group"
    assert classification_level(2001) == "crystal_system"


def test_crystal_fallback_restarts_from_pre_space_group_collection(tmp_path):
    records = []
    for crystal_index, crystal in enumerate(("cubic", "hexagonal")):
        for space_offset in range(2):
            for item in range(3):
                identifier = "{}-{}-{}".format(crystal_index, space_offset, item)
                records.append(
                    {
                        "structure_id": identifier,
                        "parent_id": "p-" + identifier,
                        "fingerprint": "f-" + identifier,
                        "crystal_system": crystal,
                        "point_group": "pg{}".format(crystal_index),
                        "space_group": "sg{}{}".format(crystal_index, space_offset),
                        "atomic_numbers": [14],
                        "num_atoms": 1,
                        "energy_per_atom": -1.0,
                        "force_norm": 0.1,
                        "composition": "Si",
                        "valid_labels": True,
                    }
                )
    # Every SG has 3 < 4 and is removed by the SG audit (12 removed > 10),
    # but each crystal has 6 >= 4 and must be recovered from the pre-SG set.
    result = analyze_dataset(
        records,
        str(tmp_path),
        minimum_class_size=4,
        fallback_threshold=10,
        seed=3,
    )
    assert result["classification_decision"]["primary_level"] == "crystal_system"
    assert result["retained_count"] == 12


def test_invalid_expert_map_is_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        FineTuningConfig(
            router="space_group", num_experts=2, expert_map={"225": 1}
        ).validate()
