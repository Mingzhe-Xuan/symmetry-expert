"""Mandatory, reproducible dataset audit before any Goal-1 training."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple


REQUIRED_FIELDS = (
    "structure_id",
    "parent_id",
    "fingerprint",
    "crystal_system",
    "point_group",
    "space_group",
    "atomic_numbers",
    "num_atoms",
    "energy_per_atom",
    "force_norm",
    "composition",
    "valid_labels",
)


def classification_level(removed_space_group_structures: int, fallback: int = 2000) -> str:
    """The boundary is intentionally strict: 2000 stays at space group."""
    return "crystal_system" if removed_space_group_structures > fallback else "space_group"


def _canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate(records: Sequence[Mapping[str, Any]]) -> None:
    ids = set()
    for index, record in enumerate(records):
        missing = [field for field in REQUIRED_FIELDS if field not in record]
        if missing:
            raise ValueError("record {} missing fields {}".format(index, missing))
        structure_id = str(record["structure_id"])
        if structure_id in ids:
            raise ValueError("duplicate structure_id: " + structure_id)
        ids.add(structure_id)
        if not isinstance(record["atomic_numbers"], list):
            raise ValueError("atomic_numbers must be a list")


def _deduplicate(
    records: Sequence[Mapping[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[str]]:
    result: List[Dict[str, Any]] = []
    removed: List[str] = []
    seen = set()
    for source in records:
        record = dict(source)
        key = (str(record["fingerprint"]), str(record["composition"]))
        if key in seen:
            removed.append(str(record["structure_id"]))
            continue
        seen.add(key)
        result.append(record)
    return result, removed


def _counts(records: Sequence[Mapping[str, Any]], field: str) -> Counter:
    return Counter(str(record[field]) for record in records)


def _group_split(
    records: Sequence[MutableMapping[str, Any]], primary: str, seed: int
) -> Dict[str, str]:
    parents_by_class: Dict[str, List[str]] = defaultdict(list)
    parent_class: Dict[str, str] = {}
    for record in records:
        parent = str(record["parent_id"])
        category = str(record[primary])
        if parent in parent_class and parent_class[parent] != category:
            raise ValueError("parent_id crosses primary categories: " + parent)
        parent_class[parent] = category
    for parent, category in parent_class.items():
        parents_by_class[category].append(parent)

    assignment: Dict[str, str] = {}
    rng = random.Random(seed)
    for category in sorted(parents_by_class):
        parents = sorted(parents_by_class[category])
        rng.shuffle(parents)
        count = len(parents)
        if count < 3:
            raise ValueError("each retained category needs at least three parents")
        valid_count = max(1, int(round(count * 0.15)))
        test_count = max(1, int(round(count * 0.15)))
        if valid_count + test_count >= count:
            valid_count = test_count = 1
        train_count = count - valid_count - test_count
        for parent in parents[:train_count]:
            assignment[parent] = "train"
        for parent in parents[train_count : train_count + valid_count]:
            assignment[parent] = "valid"
        for parent in parents[train_count + valid_count :]:
            assignment[parent] = "test"
    return assignment


def _entropy_metrics(counts: Counter) -> Dict[str, float]:
    total = sum(counts.values())
    probabilities = [value / total for value in counts.values()] if total else []
    entropy = -sum(value * math.log(value) for value in probabilities if value)
    return {
        "imbalance_ratio": (
            max(counts.values()) / min(counts.values()) if counts else 0.0
        ),
        "entropy": entropy,
        "effective_classes": math.exp(entropy) if probabilities else 0.0,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def _describe(values: Sequence[float]) -> Dict[str, float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {}
    index = lambda fraction: ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]
    average = mean(ordered)
    variance = mean([(value - average) ** 2 for value in ordered])
    return {
        "mean": average,
        "std": math.sqrt(variance),
        "median": median(ordered),
        "q1": index(0.25),
        "q3": index(0.75),
        "min": ordered[0],
        "max": ordered[-1],
    }


def _figures(
    records: Sequence[Mapping[str, Any]],
    figures: Path,
    before_records: Sequence[Mapping[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    figures.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records)
    for field in ("crystal_system", "point_group", "space_group"):
        counts = frame[field].astype(str).value_counts().sort_index()
        axis = counts.plot(kind="bar", figsize=(10, 4), title=field.replace("_", " "))
        axis.set_ylabel("unique structures")
        plt.tight_layout()
        plt.savefig(figures / (field + "_counts.png"), dpi=150)
        plt.close()
    counts = frame["space_group"].astype(str).value_counts().sort_values(ascending=False)
    axis = counts.plot(kind="bar", logy=True, figsize=(12, 4), title="Space-group long tail")
    axis.set_ylabel("unique structures (log)")
    plt.tight_layout()
    plt.savefig(figures / "space_group_long_tail_log.png", dpi=150)
    plt.close()

    before = Counter(str(record["space_group"]) for record in before_records)
    after = Counter(str(record["space_group"]) for record in records)
    categories = sorted(before)
    comparison = pd.DataFrame(
        {
            "before": [before[category] for category in categories],
            "after": [after[category] for category in categories],
        },
        index=categories,
    )
    comparison.plot(kind="bar", figsize=(12, 4), title="Screening before and after")
    plt.tight_layout()
    plt.savefig(figures / "screening_before_after.png", dpi=150)
    plt.close()

    primary = str(frame["primary_level"].iloc[0])
    split_table = pd.crosstab(frame[primary].astype(str), frame["split"])
    split_table.plot(kind="bar", stacked=True, figsize=(10, 4), title="Split counts")
    plt.tight_layout()
    plt.savefig(figures / "split_counts_stacked.png", dpi=150)
    plt.close()

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, field in zip(axes, ("num_atoms", "energy_per_atom", "force_norm")):
        frame.boxplot(column=field, by=primary, ax=axis, rot=90)
        axis.set_title(field)
    figure.suptitle("Primary-class property distributions")
    figure.tight_layout()
    figure.savefig(figures / "primary_property_boxplots.png", dpi=150)
    plt.close(figure)

    composition = pd.crosstab(frame[primary].astype(str), frame["composition"].astype(str))
    figure, axis = plt.subplots(figsize=(max(8, composition.shape[1] * 0.3), 5))
    image = axis.imshow(composition.values, aspect="auto")
    axis.set_yticks(range(len(composition.index)), composition.index)
    axis.set_xticks(range(len(composition.columns)), composition.columns, rotation=90)
    axis.set_title("Primary class x composition")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figures / "primary_composition_heatmap.png", dpi=150)
    plt.close(figure)

    parent_counts = frame.groupby(primary)["parent_id"].nunique()
    configuration_counts = frame.groupby(primary)["structure_id"].count()
    comparison = pd.DataFrame(
        {"parents": parent_counts, "configurations": configuration_counts}
    )
    comparison.plot(kind="bar", figsize=(10, 4), title="Parents vs configurations")
    plt.tight_layout()
    plt.savefig(figures / "parent_vs_configuration_counts.png", dpi=150)
    plt.close()


def analyze_dataset(
    records: Sequence[Mapping[str, Any]],
    output_dir: str,
    minimum_class_size: int = 100,
    fallback_threshold: int = 2000,
    seed: int = 123,
) -> Dict[str, Any]:
    """Audit, classify, split and freeze a small metadata dataset."""
    _validate(records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    valid = [dict(record) for record in records if bool(record["valid_labels"])]
    unique, duplicate_ids = _deduplicate(valid)

    sg_counts = _counts(unique, "space_group")
    removed_sg = {key: value for key, value in sg_counts.items() if value < minimum_class_size}
    removed_sg_structures = sum(removed_sg.values())
    primary = classification_level(removed_sg_structures, fallback_threshold)
    source_for_primary = unique
    primary_counts = _counts(source_for_primary, primary)
    removed_primary = {
        key: value for key, value in primary_counts.items() if value < minimum_class_size
    }
    retained = [
        dict(record)
        for record in source_for_primary
        if str(record[primary]) not in removed_primary
    ]
    if not retained:
        raise ValueError("class filtering removed the entire dataset")

    decision = {
        "decision_stage": "before_split",
        "minimum_class_size": minimum_class_size,
        "fallback_threshold": fallback_threshold,
        "removed_space_group_unique_structures": removed_sg_structures,
        "comparison": "strictly_greater_than",
        "primary_level": primary,
        "space_group_removed_classes": removed_sg,
        "primary_removed_classes": removed_primary,
        "source_collection": "valid_deduplicated_before_space_group_removal",
    }
    decision["sha256"] = _canonical_hash(decision)
    with open(output / "classification_decision.json", "w", encoding="utf-8") as stream:
        json.dump(decision, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    parent_assignment = _group_split(retained, primary, seed)
    for record in retained:
        record["split"] = parent_assignment[str(record["parent_id"])]
        record["primary_level"] = primary

    class_rows: List[Dict[str, Any]] = []
    for level in ("crystal_system", "point_group", "space_group"):
        counts = _counts(unique, level)
        for category in sorted(counts):
            subset = [record for record in unique if str(record[level]) == category]
            retained_subset = [
                record for record in retained if str(record[level]) == category
            ]
            split_counts = Counter(
                str(record["split"]) for record in retained_subset
            )
            atoms = _describe([record["num_atoms"] for record in subset])
            energies = _describe([record["energy_per_atom"] for record in subset])
            forces = _describe([record["force_norm"] for record in subset])
            class_rows.append(
                {
                    "category_level": level,
                    "category_id": category,
                    "crystal_system": ";".join(
                        sorted({str(record["crystal_system"]) for record in subset})
                    ),
                    "point_group": ";".join(
                        sorted({str(record["point_group"]) for record in subset})
                    ),
                    "space_group": ";".join(
                        sorted({str(record["space_group"]) for record in subset})
                    ),
                    "raw_configurations": sum(
                        str(record[level]) == category for record in records
                    ),
                    "unique_structures": len(subset),
                    "unique_proportion": len(subset) / len(unique),
                    "unique_parents": len({str(record["parent_id"]) for record in subset}),
                    "valid_labels": len(subset),
                    "removed": not bool(retained_subset),
                    "retained_configurations": len(retained_subset),
                    "retained_proportion": (
                        len(retained_subset) / len(retained) if retained else 0.0
                    ),
                    "train": split_counts["train"],
                    "valid": split_counts["valid"],
                    "test": split_counts["test"],
                    "compositions": len({str(record["composition"]) for record in subset}),
                    "prototypes": len({str(record.get("prototype", "")) for record in subset}),
                    "atoms_median": atoms["median"],
                    "atoms_q1": atoms["q1"],
                    "atoms_q3": atoms["q3"],
                    "energy_per_atom_mean": energies["mean"],
                    "energy_per_atom_std": energies["std"],
                    "force_norm_mean": forces["mean"],
                    "force_norm_std": forces["std"],
                }
            )
    _write_csv(
        output / "class_counts.csv", class_rows, list(class_rows[0].keys())
    )

    removed_rows = [
        {"category_level": "space_group", "category_id": key, "unique_structures": value}
        for key, value in sorted(removed_sg.items())
    ]
    if primary == "crystal_system":
        removed_rows.extend(
            {"category_level": primary, "category_id": key, "unique_structures": value}
            for key, value in sorted(removed_primary.items())
        )
    _write_csv(
        output / "removed_classes.csv",
        removed_rows,
        ("category_level", "category_id", "unique_structures"),
    )

    split_rows: List[Dict[str, Any]] = []
    for category in sorted(_counts(retained, primary)):
        row: Dict[str, Any] = {"category_level": primary, "category_id": category}
        for split in ("train", "valid", "test"):
            row[split] = sum(
                str(record[primary]) == category and record["split"] == split
                for record in retained
            )
        row["total"] = row["train"] + row["valid"] + row["test"]
        split_rows.append(row)
    _write_csv(
        output / "split_counts.csv",
        split_rows,
        ("category_level", "category_id", "train", "valid", "test", "total"),
    )

    train_by_class: Dict[str, List[str]] = defaultdict(list)
    for record in retained:
        if record["split"] == "train":
            train_by_class[str(record[primary])].append(str(record["structure_id"]))
    for values in train_by_class.values():
        values.sort()
        random.Random(seed).shuffle(values)
    train_order: List[str] = []
    offset = 0
    while True:
        added = False
        for category in sorted(train_by_class):
            if offset < len(train_by_class[category]):
                train_order.append(train_by_class[category][offset])
                added = True
        if not added:
            break
        offset += 1

    manifest = {
        "schema_version": 1,
        "deduplication_rule": "(fingerprint, composition), keep first input record",
        "duplicate_structure_ids": duplicate_ids,
        "classification_decision_sha256": decision["sha256"],
        "seed": seed,
        "primary_level": primary,
        "train_order": train_order,
        "records": [
            {
                key: record.get(key)
                for key in (
                    "structure_id",
                    "parent_id",
                    "fingerprint",
                    "crystal_system",
                    "point_group",
                    "space_group",
                    "composition",
                    "source",
                    "prototype",
                    "split",
                )
            }
            for record in sorted(retained, key=lambda item: str(item["structure_id"]))
        ],
    }
    manifest["sha256"] = _canonical_hash(manifest)
    with open(output / "dataset_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")

    metrics = _entropy_metrics(_counts(retained, primary))
    elements = Counter(
        int(element) for record in retained for element in record["atomic_numbers"]
    )
    element_total = sum(elements.values())
    element_proportions = {
        key: value / element_total for key, value in sorted(elements.items())
    }
    sources = Counter(str(record.get("source", "unknown")) for record in retained)
    cross = Counter(
        (
            str(record["crystal_system"]),
            str(record["point_group"]),
            str(record["space_group"]),
        )
        for record in retained
    )
    summary_lines = [
        "# Dataset summary",
        "",
        "- Raw records: {}".format(len(records)),
        "- Valid-label records: {}".format(len(valid)),
        "- Deduplicated structures: {}".format(len(unique)),
        "- Retained structures: {}".format(len(retained)),
        "- Unique parents: {}".format(len({str(item['parent_id']) for item in retained})),
        "- Unique compositions: {}".format(
            len({str(item["composition"]) for item in retained})
        ),
        "- Unique prototypes: {}".format(
            len({str(item.get("prototype", "")) for item in retained})
        ),
        "- Deduplication: `(fingerprint, composition)`, keeping the first record.",
        "- Primary classification: `{}` (decision made before split).".format(primary),
        "- Removed space-group structures: {}".format(removed_sg_structures),
        "- Imbalance ratio: {:.6g}".format(metrics["imbalance_ratio"]),
        "- Class entropy: {:.6g}".format(metrics["entropy"]),
        "- Effective classes: {:.6g}".format(metrics["effective_classes"]),
        "- Element occurrence counts: `{}`".format(dict(sorted(elements.items()))),
        "- Element occurrence proportions: `{}`".format(element_proportions),
        "- Data-source counts: `{}`".format(dict(sorted(sources.items()))),
        "",
        "## Numeric distributions",
        "",
        "- Atoms/structure: `{}`".format(_describe([item["num_atoms"] for item in retained])),
        "- Energy/atom: `{}`".format(_describe([item["energy_per_atom"] for item in retained])),
        "- Force norm: `{}`".format(_describe([item["force_norm"] for item in retained])),
        "",
        "## Crystal-system - point-group - space-group intersections",
        "",
        "| crystal system | point group | space group | structures |",
        "|---|---|---:|---:|",
    ]
    summary_lines.extend(
        "| {} | {} | {} | {} |".format(*key, value)
        for key, value in sorted(cross.items())
    )
    summary_lines.extend(
        [
            "",
            "All retained classes satisfy `n_g >= {}`. Split counts and removals are in the companion CSV files.".format(
                minimum_class_size
            ),
        ]
    )
    (output / "dataset_summary.md").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    _figures(retained, output / "figures", unique)
    return {
        "classification_decision": decision,
        "dataset_manifest": manifest,
        "retained_count": len(retained),
    }


def _load_records(path: str) -> List[Dict[str, Any]]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("input JSON must be an array")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output_dir")
    parser.add_argument("--minimum-class-size", type=int, default=100)
    parser.add_argument("--fallback-threshold", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    analyze_dataset(
        _load_records(args.input),
        args.output_dir,
        minimum_class_size=args.minimum_class_size,
        fallback_threshold=args.fallback_threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
