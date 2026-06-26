# Supporting calculator files

These Python files define ASE calculators used by GenSec.

- `ase_command.py`: cheap calculator used during generation force checks.
- `aims_command.py`: FHI-aims calculator used for reference labeling and atomic
  reference-energy consistency.
- `trained_model_command.py`: MACE calculator helper used after training. It
  loads `latest.model` from the run folder unless `GENSEC_MACE_MODEL` is set.

GenSec imports these files from the path given by `supporting_files_folder` and
`ase_parameters_file` in `parameters.json`.
