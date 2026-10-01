# Experiment data

The repository publishes two small frozen inputs required by the current
experiment contract and its validation tests:

- `task_validation_v4_20260923/detector.json`: fixed public failure detector.
- `combination_learning_20260926/acceptance.json`: completed learning-baseline
  acceptance record used by experiment preflight checks.

Their bytes and paths are preserved because existing experiment receipts bind
their hashes. They are validation inputs, not new experiment results.

All other contents of `data/` are local experiment outputs and are excluded from
Git. This includes raw simulator trajectories, model checkpoints, figures,
provider request journals, API responses, and historical result packages.
Existing local runs are retained; cleaning the repository does not delete them.

See [the experiment protocol](../docs/EXPERIMENT_PROTOCOL.md) for the current
experimental setting and run commands. Large result archives should be backed
up or published separately with the corresponding manifest and checksums.
