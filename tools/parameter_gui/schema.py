"""
Parameter schema for GenSec. Defines all available parameters with types, defaults, and descriptions.
Organized by category for UI presentation.
"""

PARAMETER_SCHEMA = {
    "core": {
        "label": "Core Settings",
        "description": "Basic project identification and structure",
        "fields": {
            "name": {
                "type": "string",
                "default": "my_project",
                "mandatory": True,
                "description": "Project name identifier",
            },
            "trials": {
                "type": "int",
                "default": 1000,
                "mandatory": False,
                "description": "Number of trial attempts for geometry generation",
            },
            "success": {
                "type": "int",
                "default": 300,
                "mandatory": False,
                "description": "Target number of successful structures",
            },
            "number_of_replicas": {
                "type": "int",
                "default": 1,
                "mandatory": False,
                "description": "Number of replica runs",
            },
        },
    },
    "geometry": {
        "label": "Geometry Settings",
        "description": "Input geometry and reference frame configuration",
        "fields": {
            "geometry": {
                "type": "dict",
                "default": {"filename": "", "format": "aims"},
                "mandatory": True,
                "description": "Input geometry file and format",
                "nested_fields": {
                    "filename": {"type": "string", "description": "Path to geometry file"},
                    "format": {"type": "string", "description": "File format (aims, vasp, etc.)"},
                },
            },
            "fixed_frame": {
                "type": "dict",
                "default": {"activate": True, "filename": "", "format": "aims"},
                "mandatory": False,
                "description": "Reference frame (e.g. surface/slab)",
                "nested_fields": {
                    "activate": {"type": "bool", "description": "Enable fixed frame"},
                    "filename": {"type": "string", "description": "Path to frame file"},
                    "format": {"type": "string", "description": "File format"},
                    "is_unit_cell": {"type": "bool", "description": "Is this a unit cell?"},
                },
            },
            "mic": {
                "type": "dict",
                "default": {"activate": False, "pbc": None},
                "mandatory": False,
                "description": "Minimum image convention settings",
                "nested_fields": {
                    "activate": {"type": "bool", "description": "Enable MIC"},
                    "pbc": {"type": "list", "description": "Periodic boundary conditions"},
                },
            },
        },
    },
    "configuration": {
        "label": "Configuration Settings",
        "description": "Molecular orientation, position, and constraint settings",
        "fields": {
            "configuration": {
                "type": "dict",
                "default": {
                    "torsions": {"activate": False},
                    "orientations": {"activate": False},
                    "coms": {"activate": False},
                    "clashes": {"intramolecular": 2.0, "intermolecular": 3.5},
                },
                "mandatory": False,
                "description": "Configuration options (torsions, orientations, positions)",
            },
        },
    },
    "training": {
        "label": "Training Settings (Main)",
        "description": "Core training pipeline configuration",
        "fields": {
            "supporting_files_folder": {
                "type": "string",
                "default": "supporting",
                "mandatory": True,
                "description": "Folder containing ASE calculator scripts",
            },
            "ase_parameters_file": {
                "type": "string",
                "default": "ase_command.py",
                "mandatory": True,
                "description": "ASE calculator setup file",
            },
            "k_density": {
                "type": "float",
                "default": 30.0,
                "mandatory": False,
                "description": "K-point density for DFT calculations",
            },
            "rmse_energy_target": {
                "type": "float",
                "default": 1.0,
                "mandatory": True,
                "description": "Target RMSE for energy (meV/atom)",
            },
            "rmse_force_target": {
                "type": "float",
                "default": 10.0,
                "mandatory": True,
                "description": "Target RMSE for forces (meV/Å)",
            },
            "loop_activate": {
                "type": "bool",
                "default": False,
                "mandatory": False,
                "description": "Enable iterative training loop",
            },
            "test_set_size": {
                "type": "int",
                "default": 0,
                "mandatory": False,
                "description": "Number of structures in test set (0 = skip test set)",
            },
            "do_labeling": {
                "type": "bool",
                "default": True,
                "mandatory": False,
                "description": "Perform DFT labeling step",
            },
            "do_training": {
                "type": "bool",
                "default": True,
                "mandatory": False,
                "description": "Run MACE training",
            },
        },
    },
    "training_advanced": {
        "label": "Training Settings (Advanced)",
        "description": "Advanced training configuration and output paths",
        "fields": {
            "mace_output_name": {
                "type": "string",
                "default": "mace_training",
                "mandatory": False,
                "description": "MACE model output name",
            },
            "out_db_labeled": {
                "type": "string",
                "default": "db_labeled.db",
                "mandatory": False,
                "description": "Output labeled database path",
            },
            "out_prefix": {
                "type": "string",
                "default": "mace_dataset",
                "mandatory": False,
                "description": "Output dataset prefix",
            },
            "global_labeled_db": {
                "type": "string",
                "default": "db_labeled_global.db",
                "mandatory": False,
                "description": "Global cumulative labeled DB",
            },
            "state_file": {
                "type": "string",
                "default": "training_state.json",
                "mandatory": False,
                "description": "Training loop state file",
            },
            "test_set_seed": {
                "type": "int",
                "default": 0,
                "mandatory": False,
                "description": "Random seed for test set selection",
            },
            "fixed_test_extxyz": {
                "type": "string",
                "default": "",
                "mandatory": False,
                "description": "Path to fixed test set extxyz file",
            },
            "phase2_activate": {
                "type": "bool",
                "default": False,
                "mandatory": False,
                "description": "Enable phase-2 relax/refine",
            },
        },
    },
    "mace_args": {
        "label": "MACE Training Arguments",
        "description": "MACE hyperparameters and training settings (add custom key/value pairs)",
        "fields": {
            "mace_args": {
                "type": "dict",
                "default": {
                    "r_max": 5.0,
                    "hidden_irreps": "64x0e + 64x1o",
                    "energy_weight": 1.0,
                    "forces_weight": 100.0,
                    "batch_size": 6,
                    "max_num_epochs": 2000,
                    "patience": 50,
                    "lr": 0.001,
                    "device": "cuda",
                    "default_dtype": "float64",
                    "E0s": {},
                },
                "mandatory": False,
                "description": "MACE training arguments (customize by adding key/value pairs)",
            },
        },
    },
    "calculator": {
        "label": "Calculator Settings",
        "description": "ASE calculator constraints and options",
        "fields": {
            "calculator": {
                "type": "dict",
                "default": {
                    "constraints": {
                        "fix_atoms": [],
                    }
                },
                "mandatory": False,
                "description": "Calculator constraints and configuration",
            },
        },
    },
}


def get_schema_field(category, field_name):
    """Get a specific field from the schema."""
    if category in PARAMETER_SCHEMA:
        fields = PARAMETER_SCHEMA[category].get("fields", {})
        if field_name in fields:
            return fields[field_name]
    return None


def get_all_categories():
    """Return ordered list of category keys."""
    return list(PARAMETER_SCHEMA.keys())


def get_category_label(category):
    """Get display label for a category."""
    return PARAMETER_SCHEMA.get(category, {}).get("label", category)


def get_category_description(category):
    """Get description for a category."""
    return PARAMETER_SCHEMA.get(category, {}).get("description", "")


def flatten_schema_to_defaults():
    """Generate a default parameters dict from schema."""
    defaults = {}
    for category, cat_data in PARAMETER_SCHEMA.items():
        for field_name, field_info in cat_data.get("fields", {}).items():
            defaults[field_name] = field_info.get("default")
    return defaults
