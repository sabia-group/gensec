# Completed example run

This folder is an archived completed run used by the notebook for inspection.
It is not meant to be edited as the starting point for a fresh calculation.

Important outputs:

- `db_generated_visual.db`: generated candidate structures with molecule plus
  frame.
- `db_generated_fps.db`: FPS-ordered generated structures.
- `db_labeled_test.db`: fixed full test set.
- `db_labeled_test_easy.db`: easier diagnostic subset of the fixed test set.
- `db_train_pool.db`: FPS pool after removing fixed test structures.
- `db_labeled_global.db`: cumulative labeled database.
- `mace_dataset_test.extxyz` and `mace_dataset_test_easy.extxyz`: MACE-readable
  test files.
- `training_state.json`: training-loop checkpoint and RMSE history.
- `latest.model`: newest trained model copied out for later search.
- `training_rounds/`: per-round labeled DBs, MACE datasets, logs, checkpoints,
  and model files.

Paths stored in the state file come from the machine where this run was
originally produced, so treat them as historical metadata.
