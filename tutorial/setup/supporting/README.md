# Supporting calculator files

These Python files define ASE calculators used by GenSec.

- `ase_command.py`: cheap calculator used during generation force checks.
- `aims_command.py`: FHI-aims calculator used for reference labeling and atomic
  reference-energy consistency.
- `trained_model_command.py`: MACE calculator helper used after training. It
  loads `latest.model` from the run folder unless `GENSEC_MACE_MODEL` is set.

GenSec imports these files from the path given by `supporting_files_folder` and
`ase_parameters_file` in `parameters.json`.

These files are examples, not portable calculator definitions. In particular,
`aims_command.py` must point to a valid FHI-aims binary and species directory
on the machine where the job runs. The example script reads `AIMS_BINARY` and
`AIMS_SPECIES_DIR` from the environment when they are set, otherwise it falls
back to the original cluster paths used for this calculation.
