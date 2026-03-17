# GenSec Parameter GUI

A PyQt5-based interactive editor for GenSec `parameters.json` files with organized, collapsible sections and intuitive controls.

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

## MACE Arguments (Dict Editor)

The MACE Arguments section uses an interactive dict editor that accepts **any parameter that `mace_run_train` CLI accepts**:

```json
{
  "r_max": 5.0,
  "hidden_irreps": "64x0e + 64x1o",
  "batch_size": 6,
  "max_num_epochs": 2000,
  "patience": 50,
  "lr": 0.001,
  "device": "cuda",
  "default_dtype": "float64",
  "E0s": {"H": -13.6, "C": -1025.3}
}
```

### How to Use the Dict Editor
1. Click on the `mace_args` field
2. Use `+` button to add new parameters
3. Enter `key` and `value` for each row
4. Click `-` to remove unwanted entries
5. Changes auto-save to JSON format

### Common MACE Parameters
- `r_max`: Cutoff radius (float, default 5.0)
- `hidden_irreps`: Network architecture (string)
- `batch_size`: Training batch size (int)
- `max_num_epochs`: Maximum epochs (int)
- `patience`: Early stopping patience (int)
- `lr`: Learning rate (float)
- `device`: GPU/CPU ("cuda" or "cpu")
- `E0s`: Element reference energies (dict)

For full MACE CLI documentation, see `mace_run_train --help`

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

## File Structure

```
tools/
├── parameter_gui/
│   ├── schema.py      # Parameter definitions & metadata
│   ├── gui.py         # PyQt5 GUI implementation
│   ├── __main__.py    # Module entry point
│   ├── __init__.py
│   └── test.py        # Verification script
├── launch_gui.py      # Python launcher
└── launch_gui.bat     # Windows batch launcher
```

## Tips & Tricks

- **Mandatory fields** marked with `*` are required for valid configs
- **Default values** shown for reference but can be overridden
- **Dict fields** (like `mace_args`) use JSON format internally
- **Collapsible sections** help reduce overwhelm—expand only what you need
- **Help system** built into every field—when stuck, click `?`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No module named 'PyQt5'" | `pip install PyQt5` |
| GUI doesn't start | Ensure Python 3.8+: `python --version` |
| JSON syntax error | Check dict fields use proper quotes: `{"key": "value"}` |
| Changes not saving | Verify file path is writable before saving |

## Advanced: Modifying the Schema

To add new parameters, edit `tools/parameter_gui/schema.py`:

```python
"your_category": {
    "label": "Display Name",
    "fields": {
        "your_param": {
            "type": "string",  # or int, float, bool, dict, list
            "default": "value",
            "mandatory": False,
            "description": "What this does",
        }
    }
}
```

The GUI automatically updates—no code changes needed!

## Notes

- The GUI is isolated in `tools/` and excluded from version control (`.gitignore`)
- All edits are reflected directly in JSON—no hidden conversions
- The dict editor for `mace_args` supports any MACE CLI parameter
- You can modify `schema.py` to add/remove parameters anytime

## Future Enhancements

- Parameter auto-validation (e.g., check RMSE targets are reasonable)
- Search/filter by parameter name or description
- Export schema as markdown reference
- Batch editing mode for multiple files
- Dark theme option
