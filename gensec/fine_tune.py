import os
import json
import glob
import re
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
    split = ft.get("split_ratio", [0.9, 0.1])
    if len(split) < 2:
        raise ValueError("fine_tuning.split_ratio must have at least two entries (train, val)")
    if len(split) > 2:
        print("[fine_tune] Warning: split_ratio has more than two entries; only train/val are used in loop mode.")
    split_ratio = (float(split[0]), float(split[1]))

    db = ase.db.connect(db_in_path)
    n_total = db.count()
    if n_total == 0:
        raise ValueError("Input DB is empty: " + db_in_path)

    n_train = int(n_total * split_ratio[0])
    n_val = int(n_total * split_ratio[1])
    n_train = min(n_train, n_total)
    n_val = min(max(n_val, 0), n_total - n_train)

    paths = {
        "train": f"{out_prefix}_train.extxyz",
        "val": f"{out_prefix}_val.extxyz",
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
        else:
            write(paths["val"], atoms, format="extxyz", append=True)
    return paths

def run_mace_training(parameters, train_xyz, valid_xyz=None, test_xyz=None, workdir=None, foundation_override=None, run_name=None):
    """Launch MACE training on the prepared dataset (overrideable via fine_tuning.mace_args)."""
    ft = parameters["fine_tuning"]
    foundation_model = foundation_override or ft.get("foundation_model")
    if foundation_model is None:
        raise ValueError("fine_tuning.foundation_model must be provided for MACE training")

    name = run_name or ft.get("mace_output_name", parameters.get("name", "mace_finetune"))
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
    subprocess.run(cmd, check=True, cwd=workdir)

    result = {
        "run_name": name,
        "workdir": workdir or os.getcwd(),
        "log_dir": os.path.join(workdir or os.getcwd(), "log"),
        "command": cmd,
    }
    # Locate model: try stagetwo first (with SWA), then regular model
    model = _locate_model(result["workdir"], name)
    if model:
        result["model"] = model
    return result


def _locate_model(workdir, name):
    """Locate model: {name}_stagetwo.model or {name}.model (no compiled)."""
    print(f"[fine_tune] _locate_model: workdir={workdir}, name={name}")
    if not os.path.isdir(workdir):
        print(f"[fine_tune] workdir is not a directory!")
        return None
    
    # Try stagetwo first (with SWA)
    stagetwo_path = os.path.join(workdir, f"{name}_stagetwo.model")
    print(f"[fine_tune] Checking: {stagetwo_path}")
    if os.path.exists(stagetwo_path) and not stagetwo_path.endswith("_compiled.model"):
        return stagetwo_path
    
    # Fall back to regular model (no SWA) - explicitly NOT the compiled version
    regular_path = os.path.join(workdir, f"{name}.model")
    print(f"[fine_tune] Checking: {regular_path}")
    if os.path.exists(regular_path) and not regular_path.endswith("_compiled.model"):
        print(f"[fine_tune] Found model: {regular_path}")
        return regular_path
    
    print(f"[fine_tune] No model found")
    return None


def _parse_mace_test_rmse(log_dir):
    """Parse the latest MACE log to extract TEST RMSE energy/force values. Copilot generated."""
    if not os.path.isdir(log_dir):
        raise FileNotFoundError(f"MACE log directory not found: {log_dir}")

    log_candidates = [
        os.path.join(log_dir, entry)
        for entry in os.listdir(log_dir)
        if os.path.isfile(os.path.join(log_dir, entry))
    ]
    if not log_candidates:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    latest_log = max(log_candidates, key=os.path.getmtime)
    with open(latest_log, "r", encoding="utf-8", errors="ignore") as handle:
        content = handle.read()

    blocks = content.split("Error-table on TEST:")
    if len(blocks) < 2:
        raise ValueError("Unable to locate 'Error-table on TEST' in MACE log output")

    last_block = blocks[-1]
    match = re.search(r"\|\s*Default_Default\s*\|\s*([0-9eE+\-.]+)\s*\|\s*([0-9eE+\-.]+)", last_block)
    if not match:
        raise ValueError("Failed to parse Default_Default row from MACE TEST table")

    energy_rmse = float(match.group(1))
    force_rmse = float(match.group(2))
    return energy_rmse, force_rmse


def _load_loop_state(state_path, foundation_model, initial_index=0):
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state.setdefault("history", [])
        state.setdefault("reserved_fps", initial_index)
        if state.get("next_fps_index", initial_index) < initial_index:
            state["next_fps_index"] = initial_index
        return state

    if foundation_model is None:
        raise ValueError("fine_tuning.foundation_model is required to initialise the loop state")

    return {
        "next_fps_index": initial_index,
        "round_index": 1,
        "last_model_path": foundation_model,
        "history": [],
        "status": "initialized",
        "reserved_fps": initial_index,
    }


def _save_loop_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _extract_fps_batch(fps_db_path, start_index, batch_size, out_db_path, round_index=None, extra_metadata=None):
    db_in = ase.db.connect(fps_db_path)
    total = db_in.count()
    if start_index >= total:
        return 0, total

    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    db_out = ase.db.connect(out_db_path)

    written = 0
    for idx, row in enumerate(db_in.select()):
        if idx < start_index:
            continue
        if written >= batch_size:
            break
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        data.setdefault("fps_row_id", row.id)
        if round_index is not None:
            data["loop_round"] = round_index
        if extra_metadata:
            data.update(extra_metadata)
        db_out.write(atoms, data=data)
        written += 1

    return written, total


def _append_labeled_round(round_db_path, global_db_path):
    src = ase.db.connect(round_db_path)
    dest = ase.db.connect(global_db_path)
    for row in src.select():
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        dest.write(atoms, data=data)


def _write_extxyz_from_db(db_path, out_path):
    db = ase.db.connect(db_path)
    if os.path.exists(out_path):
        os.remove(out_path)

    for row in db.select():
        atoms = row.toatoms()
        atoms.calc = None
        if hasattr(row, "data") and row.data:
            if "REF_energy" in row.data:
                atoms.info["energy"] = float(row.data["REF_energy"])
            if "REF_forces" in row.data:
                forces_flat = np.array(row.data["REF_forces"])
                atoms.arrays["forces"] = forces_flat.reshape(len(atoms), 3)
        write(out_path, atoms, format="extxyz", append=True)


def _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db):
    if test_size <= 0:
        return None

    test_subset_db = ft.get("test_subset_db", "db_test_subset.db")
    test_labeled_db = ft.get("test_set_db", "db_labeled_test.db")
    test_extxyz = ft.get("test_set_extxyz", "mace_dataset_test.extxyz")

    if os.path.exists(test_extxyz) and os.path.exists(test_labeled_db):
        existing = ase.db.connect(test_labeled_db).count()
        return {"extxyz": test_extxyz, "reserved_count": existing}

    print(f"[fine_tune] Preparing fixed test set with first {test_size} FPS structures.")
    written, total = _extract_fps_batch(
        fps_db_path,
        start_index=0,
        batch_size=test_size,
        out_db_path=test_subset_db,
        round_index=None,
        extra_metadata={"subset": "test"},
    )
    if written < test_size:
        raise ValueError(
            f"Requested test_set_size={test_size} but FPS database has only {total} entries."
        )

    compute_labels_on_db(parameters, test_subset_db, db_out_path=test_labeled_db)
    _write_extxyz_from_db(test_labeled_db, test_extxyz)
    _append_labeled_round(test_labeled_db, global_labeled_db)

    return {"extxyz": test_extxyz, "reserved_count": written}


def run_finetune_loop(parameters, fps_db_path):
    ft = parameters["fine_tuning"]
    energy_target = float(ft["rmse_energy_target"])
    force_target = float(ft["rmse_force_target"])
    batch_size = int(ft.get("fps_batch_size", 10))
    if batch_size <= 0:
        raise ValueError("fine_tuning.fps_batch_size must be positive")

    state_path = ft.get("state_file", "fine_tune_state.json")
    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    foundation_model = ft.get("foundation_model")
    test_size = int(ft.get("test_set_size", 0))
    fixed_test_info = None
    reserved_count = 0

    if test_size > 0:
        fixed_test_info = _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db)
        reserved_count = fixed_test_info["reserved_count"]
    elif ft.get("fixed_test_extxyz"):
        fixed_test_info = {"extxyz": ft["fixed_test_extxyz"], "reserved_count": 0}

    fixed_test_extxyz = fixed_test_info["extxyz"] if fixed_test_info else None

    state = _load_loop_state(state_path, foundation_model, initial_index=reserved_count)
    state_reserved = state.get("reserved_fps", reserved_count)
    if state_reserved < reserved_count:
        state_reserved = reserved_count
    state["reserved_fps"] = state_reserved
    reserved = state_reserved

    next_index = max(state.get("next_fps_index", reserved), reserved)
    round_index = state.get("round_index", 1)
    current_model = state.get("last_model_path", foundation_model)

    fps_db = ase.db.connect(fps_db_path)
    total_fps = fps_db.count()
    if total_fps == 0:
        raise ValueError(f"FPS database is empty: {fps_db_path}")

    while next_index < total_fps:
        round_dir = os.path.join(os.getcwd(), f"fine_tune_round_{round_index:03d}")
        os.makedirs(round_dir, exist_ok=True)
        batch_db_path = os.path.join(round_dir, f"fps_batch_round_{round_index:03d}.db")
        batch_written, _ = _extract_fps_batch(
            fps_db_path,
            next_index,
            batch_size,
            batch_db_path,
            round_index=round_index,
        )

        if batch_written == 0:
            print("[fine_tune] FPS database exhausted before reaching RMSE targets.")
            break

        print(f"[fine_tune] Round {round_index}: labeling {batch_written} FPS structures starting at index {next_index}.")
        round_labeled_db = os.path.join(round_dir, f"db_labeled_round_{round_index:03d}.db")
        compute_labels_on_db(parameters, batch_db_path, db_out_path=round_labeled_db)
        _append_labeled_round(round_labeled_db, global_labeled_db)

        dataset_prefix = os.path.join(round_dir, "mace_dataset")
        datasets = prepare_mace_extxyz(parameters, round_labeled_db, out_prefix=dataset_prefix)

        default_name = ft.get("mace_output_name", parameters.get("name", "mace_finetune"))
        run_name = f"{default_name}_round{round_index:03d}"
        test_xyz = fixed_test_extxyz or datasets.get("test")

        result = run_mace_training(
            parameters,
            datasets["train"],
            datasets.get("val"),
            test_xyz,
            workdir=round_dir,
            foundation_override=current_model,
            run_name=run_name,
        )

        energy_rmse, force_rmse = _parse_mace_test_rmse(result["log_dir"])
        print(f"[fine_tune] Round {round_index} TEST RMSE: energy={energy_rmse:.3f} meV/atom, force={force_rmse:.3f} meV/Å")

        stage_model = result.get("model", current_model)
        history_entry = {
            "round": round_index,
            "fps_start_index": next_index,
            "fps_count": batch_written,
            "energy_rmse_mev_per_atom": energy_rmse,
            "force_rmse_mev_per_a": force_rmse,
            "round_dir": round_dir,
            "model_path": stage_model,
        }
        state.setdefault("history", []).append(history_entry)
        state.update({
            "last_model_path": stage_model,
            "next_fps_index": next_index + batch_written,
            "round_index": round_index + 1,
            "status": "running",
            "reserved_fps": reserved,
        })
        _save_loop_state(state_path, state)

        if energy_rmse <= energy_target and force_rmse <= force_target:
            print(f"[fine_tune] RMSE targets reached in round {round_index}.")
            state["status"] = "converged"
            state["reserved_fps"] = reserved
            _save_loop_state(state_path, state)
            return

        next_index += batch_written
        round_index += 1
        current_model = stage_model

    print("[fine_tune] Loop stopped because FPS pool was exhausted before convergence.")
    state["status"] = "exhausted"
    state["reserved_fps"] = reserved
    _save_loop_state(state_path, state)


def run_full_pipeline(parameters, fps_db_path):
    """Run complete fine-tuning pipeline: labeling -> dataset prep -> optional training."""
    if "fine_tuning" not in parameters:
        raise ValueError("No fine_tuning block in parameters")

    ft = parameters["fine_tuning"]
    if ft.get("loop_activate"):
        print("[fine_tune] Starting iterative fine-tuning loop...")
        run_finetune_loop(parameters, fps_db_path)
        return # if loop is activated, we do not proceed with single-shot fine-tuning
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
