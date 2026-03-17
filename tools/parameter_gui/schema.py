"""
Comprehensive parameter schema for GenSec.

Every top-level JSON key (other than the four flat core fields) is represented
as a 'nested_dict' field whose name matches the JSON key exactly. That way:
  - param_widgets["training"].get_value()  → {"activate": ..., "mace_args": {...}}
  - param_widgets["calculator"].get_value() → {"supporting_files_folder": ..., ...}
  - _populate_from_dict / _collect_form_data need no special routing.

Field types:
  string, int, float, bool, list  → SimpleFieldWidget
  nested_dict (nested=True)       → NestedSectionWidget (collapsible, recursive)
  mace_args                       → MaceArgsEditor (form rows + E0s table + extra table)
"""

PARAMETER_SCHEMA = {

    # ── CORE (flat top-level keys) ─────────────────────────────────────────────
    "core": {
        "label": "Core Settings",
        "description": "Project identity and run controls (what parts of the workflow execute)",
        "fields": {
            "name": {
                "type": "string",
                "mandatory": True,
                "description": "Project name identifier",
            },
            "protocol": {
                "type": "nested_dict",
                "nested": True,
                "label": "Protocol",
                "description": "Generation and search strategies",
                "default": {"check_db": False},
                "mandatory": False,
                "subfields": {
                    "generate": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Generate",
                        "subfields": {
                            "activate": {"type": "bool",   "description": "Enable generation"},
                            "method":   {"type": "string", "description": "Generation method"},
                        },
                    },
                    "search": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Search",
                        "subfields": {
                            "activate": {"type": "bool",   "description": "Enable search"},
                            "method":   {"type": "string", "description": "Search method"},
                            "folder":   {"type": "string", "description": "Output folder for search"},
                        },
                    },
                    "check_db": {"type": "bool", "default": False, "description": "Validate DB integrity after run"},
                },
            },
            "trials": {
                "type": "int",
                "default": 1000,
                "mandatory": False,
                "description": "Maximum consecutive failed placement attempts; this counter resets to 0 after each successfully generated structure",
            },
            "success": {
                "type": "int",
                "default": 1500,
                "mandatory": False,
                "description": "Target successful structures to generate",
            },
            "number_of_replicas": {
                "type": "int",
                "default": 1,
                "mandatory": False,
                "description": "Number of independent replica runs",
            },
            "fps_selection": {
                "type": "nested_dict",
                "nested": True,
                "label": "FPS Selection",
                "description": "Furthest-point sampling for diverse structure selection",
                "mandatory": False,
                "subfields": {
                    "activate": {"type": "bool", "default": False, "description": "Enable FPS"},
                    "n_select": {"type": "string", "default": "all", "description": "Number of structures to keep (or 'all')"},
                },
            },
            "training_controls": {
                "type": "nested_dict",
                "nested": True,
                "source_key": "training",
                "label": "Training Run Controls",
                "description": "Flags that decide if/which training stages execute",
                "default": {"activate": False, "loop_activate": False, "phase2_activate": False},
                "mandatory": False,
                "subfields": {
                    "activate":                {"type": "bool",   "default": False,               "description": "Enable training step"},
                    "loop_activate":           {"type": "bool",   "default": False,                "description": "Enable iterative active-learning loop"},
                    "phase2_activate":         {"type": "bool",   "default": False,                "description": "Final model refinement with addition of 50 relaxed structures"},
                },
            },
        },
    },

    # ── GEOMETRY & FRAME ───────────────────────────────────────────────────────
    "geometry_frame": {
        "label": "Geometry & Reference Frame",
        "description": "Input molecule, fixed frame, and periodic boundary settings",
        "fields": {
            "geometry": {
                "type": "nested_dict",
                "nested": True,
                "label": "Input Molecule",
                "description": "Molecule geometry file",
                "mandatory": True,
                "subfields": {
                    "filename": {"type": "string", "description": "Path to molecule geometry file"},
                    "format":   {"type": "string", "description": "File format: aims, vasp, poscar, xyz…"},
                },
            },
            "fixed_frame": {
                "type": "nested_dict",
                "nested": True,
                "label": "Fixed Frame (Surface / Slab)",
                "description": "Reference frame held fixed during placement",
                "default": {"activate": False},
                "mandatory": False,
                "subfields": {
                    "activate":    {"type": "bool",   "default": False, "description": "Enable fixed frame"},
                    "filename":    {"type": "string", "description": "Path to frame geometry file"},
                    "format":      {"type": "string", "description": "File format"},
                    "is_unit_cell":{"type": "bool",   "default": False, "description": "Is the frame a unit cell?"},
                },
            },
            "mic": {
                "type": "nested_dict",
                "nested": True,
                "label": "Minimum Image Convention (MIC)",
                "description": "Periodic boundary / MIC settings",
                "default": {"activate": False},
                "mandatory": False,
                "subfields": {
                    "activate": {"type": "bool", "default": False, "description": "Enable MIC"},
                    "pbc":      {"type": "list", "description": "Periodic boundary conditions [x, y, z]"},
                },
            },
        },
    },

    # ── SUPERCELL & UNIT CELL ──────────────────────────────────────────────────
    "supercell_unit": {
        "label": "Supercell & Unit Cell",
        "description": "Automated unit-cell and commensurate supercell finder",
        "fields": {
            "supercell_finder": {
                "type": "nested_dict",
                "nested": True,
                "label": "Supercell Finder",
                "description": "Search for commensurate supercells",
                "default": {"activate": False},
                "mandatory": False,
                "subfields": {
                    "activate":         {"type": "bool",   "default": False,  "description": "Enable supercell finder"},
                    "unit_cell_method": {"type": "string", "default": "find", "description": "find or provided"},
                    "max_attempts":     {"type": "int",    "default": 1,     "description": "Max search attempts"},
                    "m_range": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "M-Range",
                        "description": "Supercell size search range",
                        "default": {"type": "max", "max_s": 15, "max_f": 15},
                        "subfields": {
                            "type":        {"type": "string", "default": "max",          "description": "Range type: max or given_range"},
                            "max_s":       {"type": "int",    "default": 15,            "description": "Max supercell repetition (s)"},
                            "max_f":       {"type": "int",    "default": 15,            "description": "Max supercell repetition (f)"},
                            "max_range_s": {"type": "list",   "default": [10, 10],      "description": "[min, max] for s (given_range)"},
                            "max_range_f": {"type": "list",   "default": [3, 3],        "description": "[min, max] for f (given_range)"},
                        },
                    },
                    "max_area_diff":{"type": "float", "default": 0.1, "description": "Max fractional area mismatch"},
                    "Z_cell_length":{"type": "float", "default": 100, "description": "Vacuum thickness in z (Å)"},
                },
            },
            "unit_cell_finder": {
                "type": "nested_dict",
                "nested": True,
                "label": "Unit Cell Finder",
                "description": "Optimise unit cell angles and vectors",
                "default": {"min_angle": 20, "max_angle": 90, "n_steps": 36, "seperation_factor": 1.0, "displacement_version": "full"},
                "mandatory": False,
                "subfields": {
                    "min_angle":        {"type": "float", "default": 20,  "description": "Min cell angle (°)"},
                    "max_angle":        {"type": "float", "default": 90,  "description": "Max cell angle (°)"},
                    "n_steps":          {"type": "int",   "default": 36,  "description": "Number of scan steps"},
                    "seperation_factor":{"type": "float", "default": 1.0, "description": "Lattice vector separation factor"},
                    "displacement_version": {"type": "string", "default": "full", "description": "Displacement mode"},
                    "scan_first": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Scan First",
                        "description": "Coarse scan before fine optimisation",
                        "default": {"activate": False, "first_min_angle": 0, "first_max_angle": 180, "first_n_steps": 10},
                        "subfields": {
                            "activate":       {"type": "bool",  "default": False, "description": "Enable coarse scan"},
                            "first_min_angle":{"type": "float", "default": 0,     "description": "Coarse scan min angle (°)"},
                            "first_max_angle":{"type": "float", "default": 180,   "description": "Coarse scan max angle (°)"},
                            "first_n_steps":  {"type": "int",   "default": 10,    "description": "Number of coarse scan steps"},
                            "first_select_method": {"type": "string", "default": "area", "description": "Selection method for coarse scan"},
                        },
                    },
                    "adaptive": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Adaptive Search",
                        "description": "Adaptive step-size search",
                        "default": {"activate": False, "n_points": 5, "tolerance": 1e-4, "max_iterations": 10},
                        "subfields": {
                            "activate": {"type": "bool", "default": False, "description": "Enable adaptive search"},
                            "n_points": {"type": "int", "default": 5, "description": "Points per adaptive iteration"},
                            "tolerance": {"type": "float", "default": 1e-4, "description": "Adaptive tolerance"},
                            "max_iterations": {"type": "int", "default": 10, "description": "Max adaptive iterations"},
                        },
                    },
                },
            },
        },
    },

    # ── CONFIGURATION ─────────────────────────────────────────────────────────
    "configuration_section": {
        "label": "Configuration",
        "description": "Molecular torsions, orientations, positions, and clash detection",
        "fields": {
            "configuration": {
                "type": "nested_dict",
                "nested": True,
                "label": "Molecular Configuration",
                "description": "Controls how molecules are placed",
                "default": {},
                "mandatory": False,
                "subfields": {
                    "torsions": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Torsions",
                        "default": {"activate": False},
                        "subfields": {
                            "activate":          {"type": "bool",   "default": False,   "description": "Sample torsion angles"},
                            "known":             {"type": "bool",   "default": False,   "description": "Use known torsions"},
                            "same":              {"type": "bool",   "default": False,   "description": "Apply same torsions to all molecules"},
                            "list_of_tosrions":  {"type": "string", "default": "auto",  "description": "auto or list of torsion indices"},
                            "values":            {"type": "string", "default": "random","description": "random or fixed"},
                        },
                    },
                    "orientations": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Orientations",
                        "default": {"activate": True},
                        "subfields": {
                            "activate": {"type": "bool",   "default": True,     "description": "Sample molecular orientations"},
                            "same":     {"type": "bool",   "description": "Apply same orientation to all"},
                            "values":   {"type": "string", "description": "random or fixed"},
                        },
                    },
                    "coms": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Centre of Mass Positions",
                        "default": {"activate": False},
                        "subfields": {
                            "activate": {"type": "bool",   "default": False,       "description": "Enable COM placement"},
                            "z_values": {"type": "string", "default": "identical", "description": "identical or varied"},
                            "values":   {"type": "string", "default": "given",     "description": "COM mode when supercell finder is active"},
                            "same":     {"type": "bool",   "default": False,         "description": "Use same COM for all molecules"},
                            "x_axis":   {"type": "list",   "description": "[min, max, steps]"},
                            "y_axis":   {"type": "list",   "description": "[min, max, steps]"},
                            "z_axis":   {"type": "list",   "description": "[min, max, steps]"},
                        },
                    },
                    "clashes": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Clash Detection",
                        "default": {"intramolecular": 2.0, "with_fixed_frame": 1.5},
                        "subfields": {
                            "intramolecular":  {"type": "float", "default": 2.0, "description": "Min intramolecular distance (Å)"},
                            "with_fixed_frame":{"type": "float", "default": 1.5, "description": "Min distance to fixed frame (Å)"},
                        },
                    },
                    "check_forces": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Force Check",
                        "default": {"activate": False},
                        "subfields": {
                            "activate":  {"type": "bool",  "default": False, "description": "Run quick force check after placement"},
                            "max_force": {"type": "float", "default": 0.02,  "description": "Max allowed force (eV/Å)"},
                            "max_time":  {"type": "int",   "default": 10,    "description": "Max calculator time (s)"},
                        },
                    },
                    "adsorption": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Adsorption Placement",
                        "default": {"activate": False, "method": "surface", "range": [3, 4]},
                        "subfields": {
                            "activate":  {"type": "bool",   "default": False,    "description": "Enable adsorption placement"},
                            "method":    {"type": "string", "default": "surface","description": "surface: place above surface, docking: docking mode"},
                            "range":     {"type": "list",   "default": [3, 4],    "description": "[min_dist, max_dist] from surface (Å)"},
                            "surface_z": {"type": "float",  "default": 0.0,      "description": "Surface z-level (Å)"},
                            "point":     {"type": "list",   "default": [0.0, 0.0, 0.0], "description": "Adsorption point for method='point'"},
                            "molecules": {"type": "string", "default": "all",    "description": "all or atom indices"},
                        },
                    },
                },
            },
        },
    },

    # ── CALCULATOR ────────────────────────────────────────────────────────────
    "calculator_section": {
        "label": "Calculator",
        "description": "ASE calculator used during structure relaxation",
        "fields": {
            "calculator": {
                "type": "nested_dict",
                "nested": True,
                "label": "ASE Calculator",
                "description": "Calculator and relaxation settings",
                "mandatory": False,
                "subfields": {
                    "supporting_files_folder": {"type": "string", "description": "Folder with calculator scripts"},
                    "ase_parameters_file":     {"type": "string", "description": "Calculator setup file (e.g. ase_command.py)"},
                    "algorithm":               {"type": "string", "description": "Optimiser: FIRE, BFGS, etc."},
                    "optimize":                {"type": "string", "description": "generate or relax"},
                    "steps":                   {"type": "int",    "description": "Max relaxation steps"},
                    "fmax":                    {"type": "float",  "description": "Force convergence threshold (eV/Å)"},
                    "constraints": {
                        "type": "nested_dict",
                        "nested": True,
                        "label": "Atomic Constraints",
                        "default": {"fix_atoms": []},
                        "subfields": {
                            "fix_atoms": {"type": "list", "default": [], "description": "Atom indices to fix (e.g. [-10, -9] = last two)"},
                        },
                    },
                },
            },
        },
    },

    # ── TRAINING SETTINGS (non-run-control) ─────────────────────────────────
    "training_section": {
        "label": "Training Settings",
        "description": "Training-specific configuration",
        "fields": {
            "training": {
                "type": "nested_dict",
                "nested": True,
                "label": "Training Parameters",
                "description": "DFT labeling and MACE training parameters",
                "default": {},
                "mandatory": False,
                "subfields": {
                    "supporting_files_folder": {"type": "string", "description": "Folder with DFT calculator scripts"},
                    "ase_parameters_file":     {"type": "string", "description": "DFT calculator setup file"},
                    "k_density":               {"type": "float",  "default": 30.0,                 "description": "K-point density for DFT"},
                    "test_set_size":           {"type": "int",    "default": 0,                    "description": "Test set size (0 = skip)"},
                    "phase2_fps_n_select":     {"type": "int",    "default": 50,                   "description": "Phase2 N structures (FPS selection)"},
                    "mace_output_name":        {"type": "string", "default": "mlip-output",        "description": "MACE model output name prefix"},
                    "state_file":              {"type": "string", "default": "training_state.json", "description": "Loop state tracking file"},
                    "global_labeled_db":       {"type": "string", "default": "db_labeled_global.db","description": "Cumulative labelled structures DB"},
                    "test_set_extxyz":         {"type": "string", "default": "mace_dataset_test.extxyz", "description": "Test-set extxyz filename"},
                    "test_set_easy_extxyz":    {"type": "string", "default": "mace_dataset_test_easy.extxyz", "description": "Easy test-set extxyz filename"},
                    "test_set_seed":           {"type": "int",    "default": 0,                    "description": "Random seed for test-set splitting"},
                    "loop_use_external_labeled_db": {"type": "bool", "default": False,            "description": "Use an external labeled DB for loop rounds"},
                    "do_labeling":             {"type": "bool",   "default": True,                 "description": "Run DFT labeling stage"},
                    "do_training":             {"type": "bool",   "default": True,                 "description": "Run MACE training stage"},
                    "mace_args": {
                        "type": "mace_args",
                        "default": {},
                        "mandatory": False,
                        "description": "MACE hyperparameters. Overrides mace_run_train defaults. Add extras in 'Additional MACE Args'.",
                        "known_fields": {
                            "atomic_numbers":    {"type": "list",   "description": "Atomic numbers in dataset (e.g. [1, 6, 7, 8])"},
                            "r_max":             {"type": "float",  "default": 5.0,                  "description": "Cutoff radius (Å)"},
                            "hidden_irreps":     {"type": "string", "default": "64x0e + 64x1o",      "description": "MACE irreps string"},
                            "energy_weight":     {"type": "float",  "default": 1.0,                  "description": "Energy loss weight"},
                            "forces_weight":     {"type": "float",  "default": 100.0,                "description": "Forces loss weight"},
                            "stress_weight":     {"type": "float",  "default": 0.0,                  "description": "Stress loss weight"},
                            "scaling":           {"type": "string", "default": "rms_forces_scaling", "description": "Loss/data scaling mode"},
                            "swa":               {"type": "bool",   "default": True,                 "description": "Enable SWA flag"},
                            "swa_energy_weight": {"type": "float",  "default": 100.0,                "description": "SWA energy loss weight"},
                            "swa_forces_weight": {"type": "float",  "default": 1.0,                  "description": "SWA forces loss weight"},
                            "swa_stress_weight": {"type": "float",  "default": 0.0,                  "description": "SWA stress loss weight"},
                            "batch_size":        {"type": "int",    "default": 6,                    "description": "Training batch size"},
                            "valid_batch_size":  {"type": "int",    "default": 6,                    "description": "Validation batch size"},
                            "valid_fraction":    {"type": "float",  "default": 0.1,                  "description": "Validation split fraction"},
                            "max_num_epochs":    {"type": "int",    "default": 2000,                 "description": "Maximum training epochs"},
                            "patience":          {"type": "int",    "default": 50,                   "description": "Early-stopping patience"},
                            "weight_decay":      {"type": "float",  "default": 5e-05,                "description": "L2 regularisation"},
                            "ema":               {"type": "bool",   "default": True,                 "description": "Enable EMA flag"},
                            "ema_decay":         {"type": "float",  "default": 0.999,                "description": "EMA decay"},
                            "lr":                {"type": "float",  "default": 0.001,                "description": "Learning rate"},
                            "amsgrad":           {"type": "bool",   "default": True,                 "description": "Enable AMSGrad flag"},
                            "device":            {"type": "string", "default": "cuda",              "description": "cuda or cpu"},
                            "default_dtype":     {"type": "string", "default": "float64",           "description": "float32 or float64"},
                            "save_cpu":          {"type": "bool",   "default": True,                 "description": "Enable save_cpu flag"},
                        },
                    },
                },
            },
        },
    },

}


# ── UTILITY FUNCTIONS ────────────────────────────────────────────────────────

def get_all_categories():
    return list(PARAMETER_SCHEMA.keys())

def get_category_label(category):
    return PARAMETER_SCHEMA.get(category, {}).get("label", category)

def get_category_description(category):
    return PARAMETER_SCHEMA.get(category, {}).get("description", "")

def get_schema_field(category, field_name):
    fields = PARAMETER_SCHEMA.get(category, {}).get("fields", {})
    return fields.get(field_name)

def is_nested_field(field_info):
    return field_info.get("type") == "nested_dict" and field_info.get("nested", False)

def is_mace_args_field(field_info):
    return field_info.get("type") == "mace_args"

