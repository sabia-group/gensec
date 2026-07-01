# Setup folder

This is the starting run folder for the tutorial.

Important files:

- `parameters.json`: main GenSec input for generation, FPS, and training.
- `MePTCDI.in`: moving molecule.
- `graphene.in`: fixed graphene frame, used as a unit cell.
- `supporting/ase_command.py`: cheap ASE calculator used during generation
  force checks.
- `supporting/aims_command.py`: FHI-aims ASE calculator used for reference
  labels.
- `supporting/trained_model_command.py`: optional helper for loading a trained
  MACE model.
- `sub.sbatch`: example cluster submission script.

Before running, edit the site-specific files. `sub.sbatch` contains cluster
resources, modules, conda environment names, and the path to `gensec.py`.
`supporting/aims_command.py` contains the FHI-aims executable/species setup,
unless you override it with environment variables. The atomic energies in
`parameters.json` should also be recomputed with the same reference settings
used for labeling.

The small generated databases and `good_luck.xyz`/`bad_luck.xyz` files are
included only as lightweight examples of generation outputs. A real run may
overwrite them.
