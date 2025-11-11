import os
import math
import ase.db
from ase.io import write, read
import numpy as np
import subprocess
import shutil
from gensec.relaxation import load_source

def _k_grid_from_cell(atoms, reference_atoms, k_density):
    """Compute k-grid by comparing cell lengths to reference cell."""
    cell = np.asarray(atoms.get_cell())
    ref_cell = np.asarray(reference_atoms.get_cell())
    
    if cell.size == 0 or ref_cell.size == 0:
        return (1, 1, 1)
        
    # Get cell lengths 
    lengths = np.linalg.norm(cell[:3], axis=1)
    ref_lengths = np.linalg.norm(ref_cell[:3], axis=1)
    
    # Scale k-points inversely with cell size compared to reference
    kx = max(1, int(np.ceil(k_density * ref_lengths[0] / lengths[0])))
    ky = max(1, int(np.ceil(k_density * ref_lengths[1] / lengths[1]))) 
    kz = 1  # Surface system
    
    return (kx, ky, kz)

def compute_labels_on_db(parameters, db_in_path, db_out_path="db_labeled.db"):
    """Compute single-point energies+forces for structures in db_in_path."""
    ft = parameters["fine_tuning"]
    folder = ft["supporting_files_folder"]
    ase_file = ft["ase_parameters_file"]
    k_density = float(ft.get("k_density", 30.0))
    fullpath = os.path.join(os.getcwd(), folder, ase_file)
    
    ref_file = parameters["fixed_frame"]["filename"] 
    ref_format = parameters["fixed_frame"]["format"]
    ref_atoms = read(ref_file, format=ref_format)

    db_in = ase.db.connect(db_in_path)
    if os.path.exists(db_out_path):
        os.remove(db_out_path)
    db_out = ase.db.connect(db_out_path)

    skipped = 0
    written = 0

    for row in db_in.select():
        print(f"[fine_tune] Processing structure id={row.id}")
        atoms = row.toatoms()
        extras = dict(row.data) if hasattr(row, "data") else {}
        k_grid = _k_grid_from_cell(atoms, ref_atoms, k_density)
        
        # Get the calculator and update its k_grid parameter
        module = load_source(ase_file, fullpath)
        calc = module.calculator
        calc.parameters.update({'k_grid': list(k_grid)})
        atoms.set_calculator(calc)
        
        try:
            E = float(atoms.get_potential_energy())
            F = np.asarray(atoms.get_forces(), dtype=float)
            print(f"[fine_tune] Successfully computed E/F for structure {row.id}")
        except Exception as e:
            print(f"[fine_tune] Skipping row {row.id}: calculation failed: {e}")
            skipped += 1
            continue

        # Store in data dict
        extras = dict(row.data) if hasattr(row, "data") else {}
        extras["REF_energy"] = float(E)
        extras["REF_forces"] = F.flatten().tolist()  # Flatten to 1D list
        
        db_out.write(atoms, data=extras)
        written += 1

    print(f"[fine_tune] Labeling finished: wrote {written} structures, skipped {skipped} structures.")
    return db_out_path

def prepare_mace_extxyz(parameters, db_in_path, out_prefix="mace_dataset"):
    """Convert labeled DB into MACE extxyz files (train/val/test)."""
    ft = parameters["fine_tuning"]
    split = ft.get("split_ratio", [0.8, 0.1, 0.1])
    split_ratio = (float(split[0]), float(split[1]), float(split[2]))

    db = ase.db.connect(db_in_path)
    n_total = db.count()
    if n_total == 0:
        raise ValueError("Input DB is empty: " + db_in_path)

    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])

    paths = {
        "train": f"{out_prefix}_train.extxyz",
        "val": f"{out_prefix}_val.extxyz", 
        "test": f"{out_prefix}_test.extxyz"
    }
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

    for i, row in enumerate(db.select()):
        atoms = row.toatoms()
        atoms.calc = None

        if hasattr(row, 'data') and row.data:
            if "REF_energy" in row.data:
                atoms.info["energy"] = float(row.data["REF_energy"])
            if "REF_forces" in row.data:
                # Reshape flattened forces back to (N, 3)
                forces_flat = np.array(row.data["REF_forces"])
                n_atoms = len(atoms)
                atoms.arrays["forces"] = forces_flat.reshape(n_atoms, 3) 
        
        if i < n_train:
            write(paths["train"], atoms, format="extxyz", append=True)
        elif i < n_train + n_val:
            write(paths["val"], atoms, format="extxyz", append=True)
        else:
            write(paths["test"], atoms, format="extxyz", append=True)

    return paths


def run_mace_training(parameters, train_xyz, valid_xyz=None, test_xyz=None):
    """Launch MACE training on the prepared dataset (overrideable via fine_tuning.mace_args)."""
    ft = parameters["fine_tuning"]
    foundation_model = ft.get("foundation_model")
    name = ft.get("mace_name", parameters.get("name", "mace_finetune"))
    user_args = ft.get("mace_args", {}) or {}

    mace_exe = shutil.which("mace_run_train")
    if not mace_exe:
        raise RuntimeError("mace_run_train not found in PATH")
    print(f"[fine_tune] Using mace_run_train from: {mace_exe}")

    # Default args
    base_args = [
        ("name", name),
        ("train_file", train_xyz),
        ("foundation_model", foundation_model),
        ("energy_key", "energy"),
        ("forces_key", "forces"),
        ("energy_weight", 1.0),
        ("forces_weight", 100.0),
        ("multiheads_finetuning", False),
        ("E0s", "average"),
        ("scaling", "rms_forces_scaling"),
        ("swa", None),
        ("start_swa", 60),
        ("swa_energy_weight", 100.0),
        ("swa_forces_weight", 1.0),
        ("batch_size", 2),
        ("valid_batch_size", 6),
        ("max_num_epochs", 100),
        ("ema", None),
        ("ema_decay", 0.9999),
        ("lr", 0.001),
        ("amsgrad", None),
        ("default_dtype", "float64"),
        ("device", "cuda"),
        ("save_cpu", None),
        ("seed", 0),
    ]
    if valid_xyz:
        base_args.append(("valid_file", valid_xyz))
    if test_xyz:
        base_args.append(("test_file", test_xyz))

    # overwrite existing keys, append new ones
    merged = {k: v for k, v in base_args}

    for k, v in user_args.items():
        #trying to handle all cases of value
        if v is None:
            merged[k] = None
            continue
        if isinstance(v, bool):
            merged[k] = None if v else "False"
            continue
        if isinstance(v, (list, tuple)):
            merged[k] = ",".join(str(x) for x in v)
            continue
        merged[k] = v
 
    cmd = [mace_exe]
    for k, v in merged.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    print("[fine_tune] running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def run_full_pipeline(parameters, fps_db_path):
    """Run complete fine-tuning pipeline: labeling -> dataset prep -> optional training."""
    if "fine_tuning" not in parameters:
        raise ValueError("No fine_tuning block in parameters")

    ft = parameters["fine_tuning"]
    do_labeling = ft.get("do_labeling", True)
    do_training = ft.get("do_training", True)

    if do_labeling:
        print("[fine_tune] Starting labeling...")
        db_labeled = compute_labels_on_db(parameters, fps_db_path, db_out_path=ft.get("out_db_labeled", "db_labeled.db"))
    else:
        db_labeled = ft.get("out_db_labeled", "db_labeled.db")
        if not os.path.exists(db_labeled):
            raise ValueError(f"Labeling disabled but labeled DB not found: {db_labeled}")

    print("[fine_tune] Preparing MACE extxyz dataset...")
    datasets = prepare_mace_extxyz(parameters, db_labeled, out_prefix=ft.get("out_prefix", "mace_dataset"))

    if do_training:
        print("[fine_tune] Launching MACE training...")
        run_mace_training(parameters, datasets["train"], datasets.get("val"), datasets.get("test"))
    else:
        print("[fine_tune] Training disabled; dataset ready:", datasets)
