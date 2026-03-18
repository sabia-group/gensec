# GenSec Parameter GUI (beta)

A vibe-coded PyQt5-based interactive editor for GenSec `parameters.json` files with organized, collapsible sections and intuitive controls.

## Defaults Policy

- Defaults shown in the GUI are hardcoded in the schema.
- Defaults were reviewed/updated to match code behavior on 17/03/26.

## Quick Start

```bash
# From gensec/ directory
python tools/launch_gui.py

# Or with a file directly
python tools/launch_gui.py examples/fine-tune/parameters.json
```

## Features

### 📋 Organized by Category (Collapsible)
- **Core Settings**: Project name, trials, replicas
- **Geometry Settings**: Input geometry and reference frames
- **Configuration**: Molecular orientation and positions  
- **Training (Main)**: MACE pipeline & convergence targets
- **Training (Advanced)**: Output paths, phase-2, seeds
- **MACE Arguments**: ⭐ **Full MACE CLI support** with dict editor
- **Calculator Settings**: ASE calculator constraints

### 🎨 Smart Input Widgets
- Text fields for strings
- Spinboxes for integers/floats  
- Checkboxes for booleans
- **Dict Editor** for complex params (add/remove key-value pairs with +/- buttons)
- JSON support for lists and custom structures

### ❓ Built-in Help System
- `?` button on every field with descriptions
- `*` indicator for mandatory fields
- Default value shown for reference
- Context-specific hints

### 💾 File Operations
- Open existing `parameters.json`
- Save changes to current file
- Save As to new location
- Clear status indicators

## Parameter Categories in Detail

| Category | What It Controls |
|----------|------------------|
| **Core** | Project identification (name, trial counts, replicas) |
| **Geometry** | Input structure, reference frame, periodic boundaries |
| **Configuration** | Molecular position, orientation, torsion constraints |
| **Training** | DFT labeling settings, RMSE targets, loop activation |
| **Training Adv** | File paths, test/train split, phase-2 refinement |
| **MACE Args** | ✨ Full MACE training hyperparameters (dict editor) |
| **Calculator** | ASE calculator constraints (frozen atoms, etc.) |

## MACE Arguments

The MACE Arguments section provides an interactive editor for MACE training hyperparameters. Most common parameters are already available as individual fields with sensible defaults:

- `r_max`: Cutoff radius
- `hidden_irreps`: Network architecture
- `batch_size`, `valid_batch_size`: Training batch sizes
- `max_num_epochs`, `patience`: Training convergence
- `lr`: Learning rate
- `weight_decay`: L2 regularization
- `device`, `default_dtype`: Hardware & precision
- `E0s`: Element reference energies
- And 15+ more...

**To add custom MACE parameters**: Use the "Additional MACE Args" table at the bottom of the MACE section. This lets you pass any parameter that `mace_run_train` CLI accepts that isn't already in the GUI.

### How to Add Extra Parameters
1. Expand the **MACE Arguments** section
2. Scroll to **Additional MACE Args** table
3. Click `+` button to add a new row
4. Enter the parameter name and value
5. Changes auto-save to JSON

For full MACE CLI reference, consult `mace_run_train --help`

## Installation

### Requirements
- Python 3.8+
- PyQt5

### Setup

```bash
pip install PyQt5
```

## Usage Workflow

1. **📂 Open** existing `parameters.json` or start fresh
2. **📖 Browse** sections - click arrows to expand/collapse
3. **✏️ Edit** parameters with type-appropriate widgets
4. **❓ Get Help** - click `?` button for any field
5. **👀 Review** mandatory fields (`*`) and defaults
6. **💾 Save** changes back to JSON
