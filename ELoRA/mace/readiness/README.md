# Readiness utilities

This module implements the mandatory pre-training dataset audit for Goal 0.

Input is a JSON array (or JSON Lines file) with one record per deduplicatable
structure. Required fields are `structure_id`, `parent_id`, `fingerprint`,
`crystal_system`, `point_group`, `space_group`, `atomic_numbers`, `num_atoms`,
`energy_per_atom`, `force_norm`, `composition`, and `valid_labels`. Optional
`source` and `prototype` fields enrich the report.

`python -m mace.readiness.dataset_statistics INPUT OUTPUT_DIR` writes the frozen
classification decision, split/data manifests, CSV audits, Markdown summary,
and figures. Classification is decided before group splitting. The command does
not train a model or download data.
