# Training rounds

This folder contains the per-round outputs from the iterative MACE training
loop.

Each `training_round_XXX/` folder can contain:

- the FPS batch selected for that round;
- the labeled round database;
- the cumulative labeled training database;
- MACE `.extxyz` files;
- MACE logs and checkpoints;
- per-round model files.

The top-level `../training_state.json` records which rounds completed, which
FPS rows were consumed, and the full/easy test RMSE values.
