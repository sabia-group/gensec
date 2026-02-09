import os
import json
import random
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
    """Convert labeled DB into MACE extxyz files (train/val)."""
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
        ("scaling", "rms_forces_scaling"),
        ("swa", None),
        ("start_swa", 60),
        ("swa_energy_weight", 100.0),
        ("swa_forces_weight", 1.0),
        ("batch_size", 2),
        ("valid_batch_size", 6),
        ("max_num_epochs", 100),
        ("ema", None),
        ("ema_decay", 0.999),
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
        if isinstance(v, dict):
            parts = []
            for key, val in v.items():
                try:
                    key_out = int(key)
                except (TypeError, ValueError):
                    key_out = key
                parts.append(f"{key_out}: {val}")
            merged[k] = "{" + ", ".join(parts) + "}"
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
        "log_dir": os.path.join(workdir or os.getcwd(), "logs"),
        "command": cmd,
    }
    return result


def _parse_mace_test_rmse(log_dir):
    """Grab TEST RMSE numbers from the newest log file in log_dir."""
    log_files = sorted(
        (os.path.join(log_dir, name) for name in os.listdir(log_dir)),
        key=os.path.getmtime,
    )
    if not log_files:
        raise FileNotFoundError(f"No log files in {log_dir}")

    latest_log = log_files[-1]
    energy_rmse = force_rmse = None
    with open(latest_log, "r") as handle:
        for line in handle:
            if "Default_Default" not in line:
                continue
            parts = [segment.strip() for segment in line.split("|") if segment.strip()]
            if len(parts) >= 3:
                energy_rmse = float(parts[1])
                force_rmse = float(parts[2])
                # Keep the last TEST row so we capture stage-two metrics if present.

    if energy_rmse is None or force_rmse is None:
        raise ValueError("Could not parse TEST RMSE from MACE log")
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


def _build_train_pool_db(source_db_path, exclude_ids, out_db_path):
    src = ase.db.connect(source_db_path)
    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    dest = ase.db.connect(out_db_path)

    for row in src.select():
        if row.id in exclude_ids:
            continue
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        data.setdefault("subset", "train")
        data["source_row_id"] = row.id
        dest.write(atoms, data=data)

    return dest.count()


def _write_cumulative_train_db(global_db_path, out_db_path):
    src = ase.db.connect(global_db_path)
    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    dest = ase.db.connect(out_db_path)

    for row in src.select():
        data = dict(row.data) if row.data else {}
        if data.get("subset") == "test":
            continue
        atoms = row.toatoms()
        dest.write(atoms, data=data)


def _copy_labeled_subset_db(source_db_path, dest_db_path, start_index=0, limit=None, subset_label=None, extra_metadata=None):
    src = ase.db.connect(source_db_path)
    if os.path.exists(dest_db_path):
        os.remove(dest_db_path)
    dest = ase.db.connect(dest_db_path)

    copied = 0
    for idx, row in enumerate(src.select()):
        if idx < start_index:
            continue
        if limit is not None and copied >= limit:
            break
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        if subset_label:
            data["subset"] = subset_label
        if extra_metadata:
            data.update(extra_metadata)
        dest.write(atoms, data=data)
        copied += 1
    return copied


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


def _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db, labeled_source_db=None):
    if test_size <= 0:
        return None

    test_subset_db = ft.get("test_subset_db", "db_test_subset.db")
    test_labeled_db = ft.get("test_set_db", "db_labeled_test.db")
    test_extxyz = ft.get("test_set_extxyz", "mace_dataset_test.extxyz")
    train_pool_db = ft.get("train_pool_db", "db_train_pool.db")
    
    # Convert to absolute path so it works from any cwd
    if not os.path.isabs(test_extxyz):
        test_extxyz = os.path.abspath(test_extxyz)
    if not os.path.isabs(test_labeled_db):
        test_labeled_db = os.path.abspath(test_labeled_db)

    if os.path.exists(test_extxyz) and os.path.exists(test_labeled_db):
        existing = ase.db.connect(test_labeled_db).count()
        print(f"[fine_tune] Warining: Using existing fixed test set: {test_labeled_db} ({existing} structures). Test set creation skipped.")
        if not os.path.exists(global_labeled_db):
            _append_labeled_round(test_labeled_db, global_labeled_db)
            print(f"[fine_tune] Initialized global_labeled_db from existing test set.")

        if not os.path.exists(train_pool_db):
            source_db = labeled_source_db or fps_db_path
            test_ids = []
            for row in ase.db.connect(test_labeled_db).select():
                data = dict(row.data) if row.data else {}
                if "source_row_id" in data:
                    test_ids.append(int(data["source_row_id"]))
            if test_ids:
                _build_train_pool_db(source_db, set(test_ids), train_pool_db)
            else:
                print("[fine_tune] Warning: existing test set has no source_row_id metadata; train pool will include all structures.")
                _build_train_pool_db(source_db, set(), train_pool_db)

        return {"extxyz": test_extxyz, "train_pool_db": train_pool_db}

    if labeled_source_db:
        print(f"[fine_tune] Preparing fixed test set using {test_size} random entries from external labeled DB: {labeled_source_db}.")
        source_conn = ase.db.connect(labeled_source_db)
        total_available = source_conn.count()
        if total_available < test_size:
            raise ValueError(
                f"Requested test_set_size={test_size} but external labeled DB has only {total_available} entries."
            )
        ids = [row.id for row in source_conn.select()]
        chosen_ids = set(random.sample(ids, test_size))

        if os.path.exists(test_subset_db):
            os.remove(test_subset_db)
        test_db = ase.db.connect(test_subset_db)
        for row in source_conn.select():
            if row.id not in chosen_ids:
                continue
            atoms = row.toatoms()
            data = dict(row.data) if row.data else {}
            data["subset"] = "test"
            data["source_row_id"] = row.id
            test_db.write(atoms, data=data)

        written = test_db.count()
        _write_extxyz_from_db(test_labeled_db, test_extxyz)
        if os.path.abspath(labeled_source_db) != os.path.abspath(global_labeled_db):
            _append_labeled_round(test_labeled_db, global_labeled_db)
        _build_train_pool_db(labeled_source_db, chosen_ids, train_pool_db)
    else:
        print(f"[fine_tune] Preparing fixed test set with {test_size} random FPS structures (preserving pool).")
        fps_conn = ase.db.connect(fps_db_path)
        total = fps_conn.count()
        if total < test_size:
            raise ValueError(
                f"Requested test_set_size={test_size} but FPS database has only {total} entries."
            )

        ids = [row.id for row in fps_conn.select()]
        chosen_ids = set(random.sample(ids, test_size))

        if os.path.exists(test_subset_db):
            os.remove(test_subset_db)
        test_db = ase.db.connect(test_subset_db)

        for row in fps_conn.select():
            if row.id not in chosen_ids:
                continue
            atoms = row.toatoms()
            data = dict(row.data) if row.data else {}
            data["subset"] = "test"
            data["source_row_id"] = row.id
            test_db.write(atoms, data=data)

        compute_labels_on_db(parameters, test_subset_db, db_out_path=test_labeled_db)
        _write_extxyz_from_db(test_labeled_db, test_extxyz)
        _append_labeled_round(test_labeled_db, global_labeled_db)
        if os.path.exists(test_subset_db):
            os.remove(test_subset_db)

        _build_train_pool_db(fps_db_path, chosen_ids, train_pool_db)

    return {"extxyz": test_extxyz, "train_pool_db": train_pool_db}


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
    loop_use_external = bool(ft.get("loop_use_external_labeled_db", False))
    external_labeled_db = None
    fixed_test_info = None
    pool_db_path = fps_db_path

    if loop_use_external:
        if test_size <= 0:
            raise ValueError("loop_use_external_labeled_db requires fine_tuning.test_set_size > 0")
        external_labeled_db = ft.get("external_labeled_db") or global_labeled_db
        if not external_labeled_db:
            raise ValueError("loop_use_external_labeled_db is True but external_labeled_db path is not set")
        if not os.path.exists(external_labeled_db):
            raise FileNotFoundError(f"External labeled DB not found: {external_labeled_db}")
        if ft.get("fixed_test_extxyz"):
            print("[fine_tune] Warning: fixed_test_extxyz is ignored when loop_use_external_labeled_db=True; rebuilding from the external DB.")

        fixed_test_info = _ensure_fixed_test_set(
            parameters,
            fps_db_path,
            test_size,
            ft,
            global_labeled_db,
            labeled_source_db=external_labeled_db,
        )
        pool_db_path = fixed_test_info.get("train_pool_db", external_labeled_db)
    else:
        if test_size > 0:
            fixed_test_info = _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db)
            pool_db_path = fixed_test_info.get("train_pool_db", fps_db_path)
        elif ft.get("fixed_test_extxyz"):
            fixed_test_info = {"extxyz": ft["fixed_test_extxyz"]}

    fixed_test_extxyz = fixed_test_info["extxyz"] if fixed_test_info else None

    state = _load_loop_state(state_path, foundation_model, initial_index=0)
    next_index = state.get("next_fps_index", 0)
    round_index = state.get("round_index", 1)

    if loop_use_external:
        pool_conn = ase.db.connect(external_labeled_db)
        total_pool = pool_conn.count()
        if total_pool == 0:
            raise ValueError(f"External labeled DB is empty: {external_labeled_db}")
    else:
        fps_db = ase.db.connect(pool_db_path)
        total_pool = fps_db.count()
        if total_pool == 0:
            raise ValueError(f"FPS database is empty: {pool_db_path}")

    while next_index < total_pool:
        round_dir = os.path.join(os.getcwd(), f"fine_tune_round_{round_index:03d}")
        os.makedirs(round_dir, exist_ok=True)
        round_labeled_db = os.path.join(round_dir, f"db_labeled_round_{round_index:03d}.db")
        cumulative_db = os.path.join(round_dir, f"db_labeled_cumulative_{round_index:03d}.db")
        dataset_prefix = os.path.join(round_dir, "mace_dataset")
        dataset_train_path = f"{dataset_prefix}_train.extxyz"
        
        # Check if this round was already prepared (e.g., previous run crashed during training)
        round_already_prepared = (
            os.path.exists(cumulative_db) and
            ase.db.connect(cumulative_db).count() > 0 and
            os.path.exists(dataset_train_path)
        )
        
        if round_already_prepared:
            print(f"[fine_tune] Round {round_index}: detected existing labeled DB and datasets, skipping labeling/prep.")
            batch_written = ase.db.connect(round_labeled_db).count() if os.path.exists(round_labeled_db) else 0
            datasets = {
                "train": dataset_train_path,
                "val": f"{dataset_prefix}_val.extxyz" if os.path.exists(f"{dataset_prefix}_val.extxyz") else None,
            }
        elif loop_use_external:
            batch_written = _copy_labeled_subset_db(
                pool_db_path,
                round_labeled_db,
                start_index=next_index,
                limit=batch_size,
                subset_label="train",
                extra_metadata={"loop_round": round_index},
            )
            if batch_written == 0:
                print("[fine_tune] External labeled DB exhausted before reaching RMSE targets.")
                break
            print(f"[fine_tune] Round {round_index}: reusing {batch_written} pre-labeled structures starting at index {next_index}.")
            
            _write_cumulative_train_db(global_labeled_db, cumulative_db)
            datasets = prepare_mace_extxyz(parameters, cumulative_db, out_prefix=dataset_prefix)
        else:
            batch_db_path = os.path.join(round_dir, f"fps_batch_round_{round_index:03d}.db")
            batch_written, _ = _extract_fps_batch(
                pool_db_path,
                next_index,
                batch_size,
                batch_db_path,
                round_index=round_index,
            )

            if batch_written == 0:
                print("[fine_tune] FPS database exhausted before reaching RMSE targets.")
                break

            print(f"[fine_tune] Round {round_index}: labeling {batch_written} FPS structures starting at index {next_index}.")
            compute_labels_on_db(parameters, batch_db_path, db_out_path=round_labeled_db)
            _append_labeled_round(round_labeled_db, global_labeled_db)
            
            _write_cumulative_train_db(global_labeled_db, cumulative_db)
            datasets = prepare_mace_extxyz(parameters, cumulative_db, out_prefix=dataset_prefix)

        default_name = ft.get("mace_output_name", parameters.get("name", "mace_finetune"))
        run_name = f"{default_name}_round{round_index:03d}"

        train_xyz = datasets["train"]
        if not os.path.isabs(train_xyz):
            train_xyz = os.path.abspath(train_xyz)

        val_xyz = datasets.get("val")
        if val_xyz and not os.path.isabs(val_xyz):
            val_xyz = os.path.abspath(val_xyz)

        test_xyz = fixed_test_extxyz or datasets.get("test")
        if test_xyz and not os.path.isabs(test_xyz):
            test_xyz = os.path.abspath(test_xyz)

        result = run_mace_training(
            parameters,
            train_xyz,
            val_xyz,
            test_xyz,
            workdir=round_dir,
            foundation_override=foundation_model,
            run_name=run_name,
        )

        energy_rmse, force_rmse = _parse_mace_test_rmse(result["log_dir"])
        print(f"[fine_tune] Round {round_index} TEST RMSE: energy={energy_rmse:.3f} meV/atom, force={force_rmse:.3f} meV/Å")

        history_entry = {
            "round": round_index,
            "fps_start_index": next_index,
            "fps_count": batch_written,
            "energy_rmse_mev_per_atom": energy_rmse,
            "force_rmse_mev_per_a": force_rmse,
            "round_dir": round_dir,
        }
        state.setdefault("history", []).append(history_entry)
        state.update({
            "next_fps_index": next_index + batch_written,
            "round_index": round_index + 1,
            "status": "running",
            "reserved_fps": 0,
        })
        _save_loop_state(state_path, state)

        if energy_rmse <= energy_target and force_rmse <= force_target:
            print(f"[fine_tune] RMSE targets reached in round {round_index}.")
            state["status"] = "converged"
            state["reserved_fps"] = 0
            _save_loop_state(state_path, state)
            return

        next_index += batch_written
        round_index += 1

    pool_name = "external labeled" if loop_use_external else "FPS"
    print(f"[fine_tune] Loop stopped because the {pool_name} pool was exhausted before convergence.")
    state["status"] = "exhausted"
    state["reserved_fps"] = 0
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

    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    test_size = int(ft.get("test_set_size", 0))
    fixed_test_info = None
    reserved_count = 0

    if test_size > 0:
        fixed_test_info = _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db)
        reserved_count = fixed_test_info["reserved_count"]
    elif ft.get("fixed_test_extxyz"):
        extxyz_path = ft["fixed_test_extxyz"]
        if not os.path.exists(extxyz_path):
            raise FileNotFoundError(f"Configured fixed_test_extxyz not found: {extxyz_path}")
        fixed_test_info = {"extxyz": extxyz_path, "reserved_count": 0}

    fixed_test_extxyz = fixed_test_info["extxyz"] if fixed_test_info else None

    do_labeling = ft.get("do_labeling", True)
    do_training = ft.get("do_training", True)

    if do_labeling:
        print("[fine_tune] Starting labeling...")
        source_fps_db = fps_db_path
        if reserved_count > 0:
            fps_conn = ase.db.connect(fps_db_path)
            total_entries = fps_conn.count()
            remaining = total_entries - reserved_count
            if remaining <= 0:
                raise ValueError("No FPS structures left for training after reserving the fixed test set")
            train_subset_db = ft.get("train_subset_db", "db_train_subset.db")
            _extract_fps_batch(
                fps_db_path,
                start_index=reserved_count,
                batch_size=remaining,
                out_db_path=train_subset_db,
                round_index=None,
                extra_metadata={"subset": "train"},
            )
            source_fps_db = train_subset_db

        db_labeled = compute_labels_on_db(
            parameters,
            source_fps_db,
            db_out_path=ft.get("out_db_labeled", "db_labeled.db"),
        )
        if os.path.abspath(db_labeled) != os.path.abspath(global_labeled_db):
            _append_labeled_round(db_labeled, global_labeled_db)
    else:
        db_labeled = ft.get("out_db_labeled", "db_labeled.db")
        if not os.path.exists(db_labeled):
            raise ValueError(f"Labeling disabled but labeled DB not found: {db_labeled}")
        if reserved_count > 0:
            print("[fine_tune] Warning: fixed test set reserved but labeling was skipped; ensure the labeled DB excludes those structures.")

    print("[fine_tune] Preparing MACE extxyz dataset...")
    datasets = prepare_mace_extxyz(parameters, db_labeled, out_prefix=ft.get("out_prefix", "mace_dataset"))

    if do_training:
        print("[fine_tune] Launching MACE training...")
        test_xyz = fixed_test_extxyz or datasets.get("test")
        if not test_xyz:
            print("[fine_tune] Warning: no test set provided to MACE; TEST metrics will be unavailable.")
        run_mace_training(parameters, datasets["train"], datasets.get("val"), test_xyz)
    else:
        print("[fine_tune] Training disabled; dataset ready:", datasets)
