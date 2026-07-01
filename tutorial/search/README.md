# Search with the trained model

This folder is a clean search-only setup for the trained MACE model from the
completed example run.

It contains:

- `parameters.json`: GenSec input with generation, FPS, and training disabled,
  and search enabled.
- `db_generated_visual.db`: generated candidate structures to relax/search.
- `latest.model`: trained MACE model copied from `../final/latest.model`.
- `supporting/trained_model_command.py`: ASE calculator script that loads
  `latest.model` from this folder, or `GENSEC_MACE_MODEL` if that environment
  variable is set. This remains valid when GenSec enters a relaxation folder.
- `MePTCDI.in` and `graphene.in`: the molecule and fixed-frame files.
- `sub.sbatch`: scheduler script used for this example on the Ada
  cluster. Its environment, modules, resources, and GenSec path are
  cluster-specific.

Run this from inside this folder, using the usual GenSec command for the
project. The search results are written to `db_relaxed.db`,
`db_trajectories.db`, and the `search_relaxations/` directory.

On the example cluster, submit it with:

```bash
sbatch sub.sbatch parameters.json
```

If you want to use another model without copying it into this folder, set
`GENSEC_MACE_MODEL` to an absolute model path before launching GenSec.
