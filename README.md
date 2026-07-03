# GenSec (Generation and Search)

GenSec is a Python toolkit for quasi-random structure generation and geometry optimization of molecules in fixed environments.
It is designed for searches around or on static frames such as cavities, atoms, defects, interfaces, and surfaces.

## What GenSec Does

- Generates molecular structures by sampling external and internal degrees of freedom.
- Supports constrained searches in 1D, 2D, and 3D fixed frames.
- Rejects invalid configurations using clash/connectivity checks.
- Runs local geometry optimization through ASE-compatible calculators including ML potential backends.
- Stores generated and relaxed structures in ASE databases for restartable workflows.
- Supports parallel search workflows sharing one structure database.

## Installation

GenSec currently runs directly from source.

1. Clone the repository:

   ```bash
   git clone https://github.com/sabia-group/gensec.git
   cd gensec
   ```

2. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install core dependencies:

   ```bash
   pip install numpy ase scipy networkx timeout-decorator
   ```

4. Add GenSec to your Python path in your shell session:

   ```bash
   export PYTHONPATH="$PYTHONPATH:$PWD"
   ```

Optional dependencies (only needed for FPS/descriptor-based selection workflows):

```bash
pip install featomic metatensor scikit-matter
```

## Quick Start

Run from the repository root.

Generate structures:

```bash
python gensec.py inputs/parameters_generate.json
```

Run relaxation/search using a run configuration:

```bash
python gensec.py inputs/run1.json
```

Inspect generated databases:

```bash
ase gui db_generated.db
ase gui db_relaxed.db
```

## Inputs and Examples

- Parameter templates are available in `inputs/`.
- Example workflows are available in `examples/`.
- Test-style minimal runs are available in `tests/`.

## ML Training Pipeline (From Scratch)

This branch uses `gensec/training.py` to train ML potentials from scratch.

- `run_training_pipeline` is the main entry point (called when `training.activate: true`),
- computes reference labels (energy/forces) for selected structures,
- builds MACE-ready `extxyz` datasets,
- runs one-shot or iterative-loop MACE training via `mace_run_train`.

This workflow is configured through the `training` block in your run parameters.

## Documentation

Project documentation source is in `docs/source/`.
If you update documentation pages, rebuild the Sphinx HTML output before deployment.

## License

GenSec is distributed under the GNU Lesser General Public License v2.1.
See `LICENSE` for the full text.
