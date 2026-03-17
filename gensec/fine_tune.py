import os
import json
import random
import math
import importlib
import ase.db
from ase.io import write, read
import numpy as np
import subprocess
import shutil
from ase.constraints import FixAtoms
from ase.optimize import FIRE
from gensec.relaxation import load_source
from gensec.fps_selection import select_structures_fps

TEST_SET_FORCE_BINS = 5
TEST_SET_EASY_DROP_HARDEST_BINS = 2

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
    ft = parameters["training"]
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
        print(f"[training] Processing structure id={row.id}")
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
            print(f"[training] Successfully computed E/F for structure {row.id}")
        except Exception as e:
            print(f"[training] Skipping row {row.id}: calculation failed: {e}")
            skipped += 1
            continue

        # Store in data dict
        extras = dict(row.data) if hasattr(row, "data") else {}
        extras["REF_energy"] = float(E)
        extras["REF_forces"] = F.flatten().tolist()  # Flatten to 1D list
        
        db_out.write(atoms, data=extras)
        written += 1

    print(f"[training] Labeling finished: wrote {written} structures, skipped {skipped} structures.")
    return db_out_path


def prepare_mace_extxyz(parameters, db_in_path, out_prefix="mace_dataset"):
    """Convert labeled DB into a single MACE extxyz training file.

    Validation splitting is delegated to MACE via valid_fraction.
    """
    db = ase.db.connect(db_in_path)
    n_total = db.count()
    if n_total == 0:
        raise ValueError("Input DB is empty: " + db_in_path)

    paths = {
        "train": f"{out_prefix}_train.extxyz",
    }
    for p in paths.values():
        if os.path.exists(p):
            os.remove(p)

    for row in db.select():
        atoms = row.toatoms()
        atoms.calc = None

        if hasattr(row, 'data') and row.data:
            if "REF_energy" in row.data:
                atoms.info["REF_energy"] = float(row.data["REF_energy"])
            if "REF_forces" in row.data:
                # Reshape flattened forces back to (N, 3)
                forces_flat = np.array(row.data["REF_forces"])
                n_atoms = len(atoms)
                atoms.arrays["REF_forces"] = forces_flat.reshape(n_atoms, 3)

        write(paths["train"], atoms, format="extxyz", append=True)
    return paths


def run_mace_training(parameters, train_xyz, valid_xyz=None, test_xyz=None, workdir=None, run_name=None):
    """Launch MACE training on the prepared dataset (overrideable via training.mace_args)."""
    ft = parameters["training"]

    name = run_name or ft.get("mace_output_name", parameters.get("name", "mace_training"))
    user_args = ft.get("mace_args", {}) or {}

    mace_exe = shutil.which("mace_run_train")
    if not mace_exe:
        raise RuntimeError("mace_run_train not found in PATH")
    print(f"[training] Using mace_run_train from: {mace_exe}")

    # Default args
    base_args = [
        ("name", name),
        ("energy_key", "REF_energy"),
        ("forces_key", "REF_forces"),
        ("r_max", 5.0),
        ("hidden_irreps", "64x0e + 64x1o"),
        ("energy_weight", 1.0),
        ("forces_weight", 100.0),
        ("stress_weight", 0.0),
        ("scaling", "rms_forces_scaling"),
        ("swa", None),
        ("swa_energy_weight", 100.0),
        ("swa_forces_weight", 1.0),
        ("swa_stress_weight", 0.0),
        ("batch_size", 6),
        ("valid_batch_size", 6),
        ("valid_fraction", 0.1),
        ("max_num_epochs", 2000),
        ("patience", 50),
        ("weight_decay", 5e-05),
        ("ema", None),
        ("ema_decay", 0.999),
        ("lr", 0.001),
        ("amsgrad", None),
        ("default_dtype", "float64"),
        ("device", "cuda"),
        ("save_cpu", None),
        ("train_file", train_xyz),
    ]

    if valid_xyz:
        base_args = [(k, v) for k, v in base_args if k != "valid_fraction"]
        base_args.append(("valid_file", valid_xyz))
    if test_xyz:
        base_args.append(("test_file", test_xyz))

    # overwrite existing keys, append new ones
    merged = {k: v for k, v in base_args}

    for k, v in user_args.items():
        # trying to handle all cases of value
        if v is None:
            merged[k] = None
            continue
        if isinstance(v, bool):
            if k == "swa":
                if v:
                    merged[k] = None
                else:
                    merged.pop(k, None)
                continue
            if k == "ema":
                if v:
                    merged[k] = None
                else:
                    merged.pop(k, None)
                continue
            merged[k] = None if v else "False"
            continue
        if isinstance(v, dict):
            if k == "E0s":
                parts = []
                for key, val in v.items():
                    try:
                        key_out = int(key)
                    except (TypeError, ValueError):
                        key_out = key
                    parts.append(f"{key_out}: {val}")
                merged[k] = "{" + ", ".join(parts) + "}"
            else:
                merged[k] = repr(v)
            continue
        if isinstance(v, (list, tuple)):
            if k == "atomic_numbers":
                merged[k] = "[" + ",".join(str(int(x)) for x in v) + "]"
            else:
                merged[k] = ",".join(str(x) for x in v)
            continue
        merged[k] = v

    if "E0s" not in merged or merged["E0s"] in (None, "", "{}"):
        print(
            "[training] Suggestion: `training.mace_args.E0s` is not set. "
            "Training usually works better when element reference energies (E0s) are provided."
        )
 
    cmd = [mace_exe]
    for k, v in merged.items():
        if v is None:
            cmd.append(f"--{k}")
        else:
            cmd.append(f"--{k}={v}")

    print("[training] running:", " ".join(cmd))
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
    expected_config = "default_default"
    with open(latest_log, "r") as handle:
        for line in handle:
            line_lower = line.lower()
            if "default_" not in line_lower:
                continue
            parts = [segment.strip() for segment in line.split("|") if segment.strip()]
            if len(parts) >= 3:
                config_type = parts[0].lower()
                if config_type != expected_config:
                    continue
                energy_rmse = float(parts[1])
                force_rmse = float(parts[2])
                # Keep the last TEST row so we capture stage-two metrics if present.

    if energy_rmse is None or force_rmse is None:
        raise ValueError("Could not parse TEST RMSE from MACE log")
    return energy_rmse, force_rmse


def _load_loop_state(state_path, initial_index=0):
    if os.path.exists(state_path):
        with open(state_path, "r", encoding="utf-8") as handle:
            state = json.load(handle)
        state.setdefault("history", [])
        if state.get("next_fps_index", initial_index) < initial_index:
            state["next_fps_index"] = initial_index
        return state

    return {
        "next_fps_index": initial_index,
        "round_index": 1,
        "history": [],
        "status": "initialized",
    }


def _round_batch_size(round_index):
    """Hard-coded naive loop schedule for newly labeled structures per round."""
    schedule = [60, 50, 40, 30, 25]
    if round_index <= len(schedule):
        return schedule[round_index - 1]
    return 20


def _save_loop_state(state_path, state):
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _round_metric(value, decimals=3):
    return round(float(value), int(decimals))


def _compact_eval_metrics(metrics, decimals=3):
    if not metrics:
        return None
    return {
        "energy_rmse_mev_per_atom": _round_metric(metrics["energy_rmse_mev_per_atom"], decimals),
        "force_rmse_mev_per_a": _round_metric(metrics["force_rmse_mev_per_a"], decimals),
    }


def _find_newest_model_file(search_dir, preferred_name=None):
    candidates = []
    preferred = []
    for root, _, files in os.walk(search_dir):
        for fname in files:
            if not fname.endswith(".model"):
                continue
            path = os.path.join(root, fname)
            if preferred_name and preferred_name in fname:
                preferred.append(path)
            else:
                candidates.append(path)

    source = preferred if preferred else candidates
    if not source:
        return None
    return max(source, key=os.path.getmtime)


def _build_phase2_calculator(model_path, ft):
    try:
        mace_calculators = importlib.import_module("mace.calculators")
        MACECalculator = getattr(mace_calculators, "MACECalculator")
    except Exception as exc:
        raise RuntimeError("Phase-2 relaxation requires mace Python package with MACECalculator.") from exc

    device = str(ft.get("phase2_device", ft.get("mace_args", {}).get("device", "cuda")))
    dtype = str(ft.get("phase2_dtype", ft.get("mace_args", {}).get("default_dtype", "float64")))
    kwargs = {
        "device": device,
        "default_dtype": dtype,
    }

    try:
        return MACECalculator(model_paths=model_path, **kwargs)
    except TypeError:
        return MACECalculator(model_path=model_path, **kwargs)


def _relax_db_with_model(parameters, source_db_path, model_path, out_db_path, fmax=0.01, steps=1000):
    ft = parameters["training"]
    calc = _build_phase2_calculator(model_path, ft)
    src = ase.db.connect(source_db_path)
    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    out = ase.db.connect(out_db_path)

    fix_atoms = (
        parameters.get("calculator", {})
        .get("constraints", {})
        .get("fix_atoms", [])
    )

    written = 0
    skipped = 0
    for row in src.select():
        atoms = row.toatoms()
        if fix_atoms:
            atoms.set_constraint(FixAtoms(indices=list(fix_atoms)))
        atoms.calc = calc

        try:
            opt = FIRE(atoms, logfile=None)
            opt.run(fmax=float(fmax), steps=int(steps))
            relaxed_energy = float(atoms.get_potential_energy())
            max_force = float(np.linalg.norm(atoms.get_forces(), axis=1).max())
        except Exception as exc:
            print(f"[training] Phase-2 relax skipped row {row.id}: {exc}")
            skipped += 1
            continue

        data = dict(row.data) if row.data else {}
        data["phase2_relaxed_energy"] = relaxed_energy
        data["phase2_relaxed_fmax"] = max_force
        data["phase2_relaxed_model"] = os.path.abspath(model_path)
        data["phase2_source_row_id"] = row.id
        out.write(atoms, data=data)
        written += 1

    print(f"[training] Phase-2 relax finished: wrote {written}, skipped {skipped}")
    return out_db_path


def _select_low_energy_db(relaxed_db_path, out_db_path, fraction=1.0 / 3.0, min_count=50):
    db = ase.db.connect(relaxed_db_path)
    rows = []
    for row in db.select():
        data = dict(row.data) if row.data else {}
        if "phase2_relaxed_energy" in data:
            rows.append((float(data["phase2_relaxed_energy"]), row.id))

    if not rows:
        raise ValueError("No phase2_relaxed_energy entries found in relaxed DB.")

    rows.sort(key=lambda item: item[0])
    total = len(rows)
    n_keep = max(int(math.ceil(total * float(fraction))), int(min_count))
    n_keep = min(n_keep, total)
    selected_ids = {row_id for _, row_id in rows[:n_keep]}

    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    out = ase.db.connect(out_db_path)
    for row in db.select():
        if row.id not in selected_ids:
            continue
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        data["phase2_subset"] = "low_energy"
        out.write(atoms, data=data)

    print(f"[training] Phase-2 low-energy subset: kept {n_keep}/{total} structures")
    return out_db_path


def _fps_select_db(source_db_path, out_db_path, n_select=50):
    db = ase.db.connect(source_db_path)
    rows = list(db.select())
    if not rows:
        raise ValueError(f"Input DB for phase-2 FPS is empty: {source_db_path}")

    frames = [row.toatoms() for row in rows]
    selected_idx = select_structures_fps(frames, n_select=min(int(n_select), len(frames)))
    selected_ids = {rows[idx].id for idx in selected_idx}

    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    out = ase.db.connect(out_db_path)
    for row in rows:
        if row.id not in selected_ids:
            continue
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        data["phase2_subset"] = "low_energy_fps"
        out.write(atoms, data=data)

    print(f"[training] Phase-2 FPS subset: selected {len(selected_ids)} structures")
    return out_db_path


def run_phase2_relax_refine(parameters, fps_db_path):
    ft = parameters["training"]
    if not ft.get("loop_activate", False):
        print("[training] Phase-2 requested without loop_activate; skipping phase-2.")
        return

    phase2_dir = os.path.abspath(ft.get("phase2_folder", "training_relax"))
    os.makedirs(phase2_dir, exist_ok=True)

    state_path = ft.get("state_file", "training_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Phase-2 requires phase-1 state file: {state_path}")
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("phase2", {}).get("status") == "completed":
        print("[training] Phase-2 already completed; skipping.")
        return

    latest_model = os.path.abspath("latest.model")
    phase1_model = os.path.abspath("phase1.model")
    if os.path.exists(latest_model):
        model_path = latest_model
    elif os.path.exists(phase1_model):
        model_path = phase1_model
    else:
        raise FileNotFoundError("No model found for phase-2 relax. Expected latest.model or phase1.model in the run directory.")

    print(f"[training] Phase-2 using model: {model_path}")

    source_db = os.path.abspath(fps_db_path)
    if not os.path.exists(source_db):
        raise FileNotFoundError(f"Phase-2 source DB not found: {source_db}")

    relax_db = os.path.join(phase2_dir, "db_relax_all.db")
    lowe_db = os.path.join(phase2_dir, "db_relax_lowE.db")
    lowe_fps_db = os.path.join(phase2_dir, "db_relax_lowE_fps.db")
    lowe_fps_label_db = os.path.join(phase2_dir, "db_relax_lowE_fps_label.db")
    augmented_db = os.path.join(phase2_dir, "db_labeled_augmented.db")

    _relax_db_with_model(
        parameters,
        source_db,
        model_path,
        relax_db,
        fmax=float(ft.get("phase2_relax_fmax", 0.01)),
        steps=int(ft.get("phase2_relax_steps", 20)),
    )
    _select_low_energy_db(
        relax_db,
        lowe_db,
        fraction=float(ft.get("phase2_low_energy_fraction", 1.0 / 3.0)),
        min_count=int(ft.get("phase2_low_energy_min", 50)),
    )
    _fps_select_db(
        lowe_db,
        lowe_fps_db,
        n_select=int(ft.get("phase2_fps_n_select", 50)),
    )

    compute_labels_on_db(parameters, lowe_fps_db, db_out_path=lowe_fps_label_db)

    if not state.get("history"):
        raise ValueError("Phase-2 requires at least one completed phase-1 round in state history.")
    last_round = int(state["history"][-1].get("round"))
    phase1_round_dir = os.path.abspath(f"training_round_{last_round:03d}")
    phase1_db = os.path.join(phase1_round_dir, f"db_labeled_cumulative_{last_round:03d}.db")
    if not os.path.exists(phase1_db):
        raise FileNotFoundError(f"Could not find phase-1 cumulative DB: {phase1_db}")

    shutil.copy2(phase1_db, augmented_db)
    _append_labeled_round(lowe_fps_label_db, augmented_db)

    datasets = prepare_mace_extxyz(parameters, augmented_db, out_prefix=os.path.join(phase2_dir, "mace_dataset_augmented"))

    test_xyz = None
    if ft.get("fixed_test_extxyz"):
        test_xyz = ft["fixed_test_extxyz"]
    else:
        test_extxyz = ft.get("test_set_extxyz", "mace_dataset_test.extxyz")
        if os.path.exists(test_extxyz):
            test_xyz = test_extxyz

    phase2_result = run_mace_training(
        parameters,
        datasets["train"],
        datasets.get("val"),
        test_xyz,
        workdir=phase2_dir,
        run_name=f"{ft.get('mace_output_name', parameters.get('name', 'mace_training'))}_phase2",
    )

    phase2_model_src = _find_newest_model_file(phase2_dir)
    if phase2_model_src:
        phase2_latest_model = os.path.abspath("latest.model")
        shutil.copy2(phase2_model_src, phase2_latest_model)
        print(f"[training] Updated latest model after phase-2: {phase2_latest_model}")
        phase2_final_model = os.path.abspath("final.model")
        shutil.copy2(phase2_model_src, phase2_final_model)
        print(f"[training] Saved final model after phase-2: {phase2_final_model}")

    state["phase2"] = {
        "status": "completed",
        "phase2_dir": phase2_dir,
        "source_db": source_db,
        "relax_db": relax_db,
        "low_energy_db": lowe_db,
        "low_energy_fps_db": lowe_fps_db,
        "low_energy_fps_label_db": lowe_fps_label_db,
        "augmented_db": augmented_db,
        "phase2_model_input": model_path,
        "phase2_training": phase2_result,
    }
    _save_loop_state(state_path, state)


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
                atoms.info["REF_energy"] = float(row.data["REF_energy"])
            if "REF_forces" in row.data:
                forces_flat = np.array(row.data["REF_forces"])
                atoms.arrays["REF_forces"] = forces_flat.reshape(len(atoms), 3)
        write(out_path, atoms, format="extxyz", append=True)


def _get_row_force_marker(row):
    data = dict(row.data) if row.data else {}
    if "REF_forces" in data:
        ref_forces = np.asarray(data["REF_forces"], dtype=float).reshape(-1, 3)
        if ref_forces.size:
            return float(np.linalg.norm(ref_forces, axis=1).max())

    row_fmax = getattr(row, "fmax", None)
    if row_fmax is not None:
        try:
            row_fmax = float(row_fmax)
            if np.isfinite(row_fmax):
                return row_fmax
        except (TypeError, ValueError):
            pass

    try:
        atoms = row.toatoms()
        if getattr(atoms, "calc", None) is not None:
            forces = np.asarray(atoms.get_forces(), dtype=float)
            if forces.size:
                return float(np.linalg.norm(forces, axis=1).max())
    except Exception:
        pass

    return None


def _score_rows_by_force_marker(rows):
    scored = []
    for row in rows:
        marker = _get_row_force_marker(row)
        if marker is None:
            continue
        scored.append((marker, row.id))
    return sorted(scored, key=lambda item: item[0])


def _split_scored_rows_into_bins(scored_rows, n_bins):
    if not scored_rows:
        return []

    n_bins = max(1, min(int(n_bins), len(scored_rows)))
    edges = np.linspace(0, len(scored_rows), n_bins + 1, dtype=int)
    bins = []
    for idx in range(n_bins):
        start = int(edges[idx])
        end = int(edges[idx + 1])
        chunk = scored_rows[start:end]
        if chunk:
            bins.append(chunk)
    return bins


def _pick_test_ids_from_force_bins(scored_rows, test_size, sampler, n_bins=5):
    if len(scored_rows) < test_size:
        raise ValueError(
            f"Requested test_set_size={test_size} but only {len(scored_rows)} labeled entries have REF_forces."
        )

    bins = _split_scored_rows_into_bins(scored_rows, n_bins)
    bin_pools = [[row_id for _, row_id in chunk] for chunk in bins]
    selected_by_bin = {idx: [] for idx in range(len(bin_pools))}
    remaining = int(test_size)

    while remaining > 0:
        progress = False
        for idx, pool in enumerate(bin_pools):
            if remaining <= 0:
                break
            if not pool:
                continue
            chosen_id = sampler.choice(pool)
            pool.remove(chosen_id)
            selected_by_bin[idx].append(chosen_id)
            remaining -= 1
            progress = True
        if not progress:
            break

    if remaining != 0:
        raise ValueError("Could not complete stratified test-set sampling from force bins.")

    marker_by_id = {row_id: marker for marker, row_id in scored_rows}
    return {
        "selected_ids": {row_id for ids in selected_by_bin.values() for row_id in ids},
        "selected_by_bin": selected_by_bin,
        "marker_by_id": marker_by_id,
        "n_bins": len(bin_pools),
    }


def _split_test_db_easy(labeled_test_db, easy_db_path, n_bins=5, drop_hardest_bins=2):
    src = ase.db.connect(labeled_test_db)
    rows = list(src.select())
    if not rows:
        raise ValueError(f"Labeled test DB is empty: {labeled_test_db}")

    metadata_bin_ids = {}
    for row in rows:
        data = dict(row.data) if row.data else {}
        if "test_bin_index" in data:
            metadata_bin_ids[row.id] = int(data["test_bin_index"])

    if metadata_bin_ids and len(metadata_bin_ids) == len(rows):
        n_bins = max(metadata_bin_ids.values()) + 1
        n_drop_hardest = max(0, min(int(drop_hardest_bins), max(0, n_bins - 1)))
        hardest_start = max(0, n_bins - n_drop_hardest)
        easy_ids = {row_id for row_id, bin_idx in metadata_bin_ids.items() if bin_idx < hardest_start}
    else:
        scored = _score_rows_by_force_marker(rows)
        if not scored:
            raise ValueError(f"No REF_forces found in labeled test DB: {labeled_test_db}")

        bins = _split_scored_rows_into_bins(scored, n_bins)
        n_bins = len(bins)
        n_drop_hardest = max(0, min(int(drop_hardest_bins), max(0, n_bins - 1)))
        hardest_start = max(0, n_bins - n_drop_hardest)
        easy_ids = {
            row_id
            for bin_idx, chunk in enumerate(bins)
            if bin_idx < hardest_start
            for _, row_id in chunk
        }

    split_meta = {
        "mode": "abs_force_bins",
        "n_bins": n_bins,
        "drop_hardest_bins": n_drop_hardest,
    }

    if not easy_ids:
        scored = _score_rows_by_force_marker(rows)
        keep = max(1, len(scored) - max(1, len(scored) // 3))
        easy_ids = {row_id for _, row_id in scored[:keep]}

    if os.path.exists(easy_db_path):
        os.remove(easy_db_path)

    easy_db = ase.db.connect(easy_db_path)
    for row in rows:
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        if row.id in easy_ids:
            easy_data = dict(data)
            easy_data["subset"] = "test_easy"
            easy_db.write(atoms, data=easy_data)

    print(f"[training] Test split (force bins): full={len(rows)}, easy={len(easy_ids)}")
    return {
        "easy_db": easy_db_path,
        "full_count": len(rows),
        "easy_count": len(easy_ids),
        **split_meta,
    }


def _evaluate_model_on_labeled_db(parameters, model_path, labeled_db_path):
    ft = parameters["training"]
    calc = _build_phase2_calculator(model_path, ft)

    db = ase.db.connect(labeled_db_path)
    energy_err = []
    force_err = []
    n_atoms_total = 0
    n_structures = 0

    for row in db.select():
        data = dict(row.data) if row.data else {}
        if "REF_energy" not in data or "REF_forces" not in data:
            continue

        atoms = row.toatoms()
        atoms.calc = calc
        pred_energy = float(atoms.get_potential_energy())
        pred_forces = np.asarray(atoms.get_forces(), dtype=float)

        ref_energy = float(data["REF_energy"])
        ref_forces = np.asarray(data["REF_forces"], dtype=float).reshape(len(atoms), 3)

        energy_err.append((pred_energy - ref_energy) / len(atoms))
        force_err.append((pred_forces - ref_forces).reshape(-1))
        n_atoms_total += len(atoms)
        n_structures += 1

    if n_structures == 0:
        raise ValueError(f"No labeled structures with REF_energy/REF_forces in {labeled_db_path}")

    energy_err = np.asarray(energy_err, dtype=float)
    force_err = np.concatenate(force_err)

    return {
        "db_path": os.path.abspath(labeled_db_path),
        "n_structures": int(n_structures),
        "n_atoms": int(n_atoms_total),
        "energy_rmse_mev_per_atom": float(np.sqrt(np.mean(energy_err ** 2)) * 1000.0),
        "force_rmse_mev_per_a": float(np.sqrt(np.mean(force_err ** 2)) * 1000.0),
    }


def _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db, labeled_source_db=None):
    if test_size <= 0:
        return None

    test_subset_db = ft.get("test_subset_db", "db_test_subset.db")
    test_labeled_db = ft.get("test_set_db", "db_labeled_test.db")
    test_extxyz = ft.get("test_set_extxyz", "mace_dataset_test.extxyz")
    test_easy_db = ft.get("test_set_easy_db", "db_labeled_test_easy.db")
    test_easy_extxyz = ft.get("test_set_easy_extxyz", "mace_dataset_test_easy.extxyz")
    train_pool_db = ft.get("train_pool_db", "db_train_pool.db")
    test_seed = ft.get("test_set_seed", 0)
    test_force_bins = TEST_SET_FORCE_BINS
    test_easy_drop_hardest_bins = TEST_SET_EASY_DROP_HARDEST_BINS
    
    # Convert to absolute path so it works from any cwd
    if not os.path.isabs(test_extxyz):
        test_extxyz = os.path.abspath(test_extxyz)
    if not os.path.isabs(test_labeled_db):
        test_labeled_db = os.path.abspath(test_labeled_db)
    if not os.path.isabs(test_easy_db):
        test_easy_db = os.path.abspath(test_easy_db)
    if not os.path.isabs(test_easy_extxyz):
        test_easy_extxyz = os.path.abspath(test_easy_extxyz)
    if not os.path.isabs(train_pool_db):
        train_pool_db = os.path.abspath(train_pool_db)

    sampler = random if test_seed is None else random.Random(int(test_seed))

    if os.path.exists(test_labeled_db):
        existing = ase.db.connect(test_labeled_db).count()
        if os.path.exists(test_extxyz):
            print(f"[training] Warining: Using existing fixed test set: {test_labeled_db} ({existing} structures). Test set creation skipped.")
        else:
            print(f"[training] Rebuilding missing test extxyz from existing labeled test DB: {test_labeled_db} ({existing} structures).")
            _write_extxyz_from_db(test_labeled_db, test_extxyz)
        if not os.path.exists(global_labeled_db):
            _append_labeled_round(test_labeled_db, global_labeled_db)
            print(f"[training] Initialized global_labeled_db from existing test set.")

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
                print("[training] Warning: existing test set has no source_row_id metadata; train pool will include all structures.")
                _build_train_pool_db(source_db, set(), train_pool_db)

        split_info = _split_test_db_easy(
            test_labeled_db,
            test_easy_db,
            n_bins=test_force_bins,
            drop_hardest_bins=test_easy_drop_hardest_bins,
        )
        _write_extxyz_from_db(test_easy_db, test_easy_extxyz)

        return {
            "extxyz": test_extxyz,
            "all_db": test_labeled_db,
            "all_extxyz": test_extxyz,
            "easy_db": test_easy_db,
            "easy_extxyz": test_easy_extxyz,
            "train_pool_db": train_pool_db,
            "split": split_info,
        }

    if labeled_source_db:
        print(f"[training] Preparing fixed test set using stratified force-bin sampling from external labeled DB: {labeled_source_db}.")
        source_conn = ase.db.connect(labeled_source_db)
        rows = list(source_conn.select())
        scored = _score_rows_by_force_marker(rows)
        total_available = len(scored)
        if total_available < test_size:
            raise ValueError(
                f"Requested test_set_size={test_size} but external labeled DB has only {total_available} labeled entries with REF_forces."
            )
        pick_info = _pick_test_ids_from_force_bins(scored, test_size, sampler, n_bins=test_force_bins)
        chosen_ids = pick_info["selected_ids"]
        selected_bin_by_id = {
            row_id: bin_idx
            for bin_idx, ids in pick_info["selected_by_bin"].items()
            for row_id in ids
        }

        if os.path.exists(test_labeled_db):
            os.remove(test_labeled_db)
        test_db = ase.db.connect(test_labeled_db)
        for row in rows:
            if row.id not in chosen_ids:
                continue
            atoms = row.toatoms()
            data = dict(row.data) if row.data else {}
            data["subset"] = "test"
            data["source_row_id"] = row.id
            data["test_bin_index"] = int(selected_bin_by_id[row.id])
            data["test_force_marker"] = float(pick_info["marker_by_id"][row.id])
            data["test_split_strategy"] = "force_bins"
            test_db.write(atoms, data=data)

        written = test_db.count()
        if written != test_size:
            raise ValueError(f"Expected {test_size} test structures, got {written} from external labeled DB")
        _write_extxyz_from_db(test_labeled_db, test_extxyz)
        split_info = _split_test_db_easy(
            test_labeled_db,
            test_easy_db,
            n_bins=test_force_bins,
            drop_hardest_bins=test_easy_drop_hardest_bins,
        )
        _write_extxyz_from_db(test_easy_db, test_easy_extxyz)
        if os.path.abspath(labeled_source_db) != os.path.abspath(global_labeled_db):
            _append_labeled_round(test_labeled_db, global_labeled_db)
        _build_train_pool_db(labeled_source_db, chosen_ids, train_pool_db)
    else:
        print(f"[training] Preparing fixed test set using stratified force-bin sampling from FPS pool: {fps_db_path}.")
        fps_conn = ase.db.connect(fps_db_path)
        rows = list(fps_conn.select())
        scored = _score_rows_by_force_marker(rows)
        total = len(scored)
        if total < test_size:
            raise ValueError(
                f"Requested test_set_size={test_size} but FPS database has only {total} entries with usable force data."
            )
        pick_info = _pick_test_ids_from_force_bins(scored, test_size, sampler, n_bins=test_force_bins)
        chosen_ids = pick_info["selected_ids"]
        selected_bin_by_id = {
            row_id: bin_idx
            for bin_idx, ids in pick_info["selected_by_bin"].items()
            for row_id in ids
        }

        if os.path.exists(test_subset_db):
            os.remove(test_subset_db)
        test_db = ase.db.connect(test_subset_db)

        for row in rows:
            if row.id not in chosen_ids:
                continue
            atoms = row.toatoms()
            data = dict(row.data) if row.data else {}
            data["subset"] = "test"
            data["source_row_id"] = row.id
            data["test_bin_index"] = int(selected_bin_by_id[row.id])
            data["test_force_marker"] = float(pick_info["marker_by_id"][row.id])
            data["test_split_strategy"] = "force_bins"
            test_db.write(atoms, data=data)

        compute_labels_on_db(parameters, test_subset_db, db_out_path=test_labeled_db)
        _write_extxyz_from_db(test_labeled_db, test_extxyz)
        split_info = _split_test_db_easy(
            test_labeled_db,
            test_easy_db,
            n_bins=test_force_bins,
            drop_hardest_bins=test_easy_drop_hardest_bins,
        )
        _write_extxyz_from_db(test_easy_db, test_easy_extxyz)
        _append_labeled_round(test_labeled_db, global_labeled_db)
        if os.path.exists(test_subset_db):
            os.remove(test_subset_db)

        _build_train_pool_db(fps_db_path, chosen_ids, train_pool_db)

    return {
        "extxyz": test_extxyz,
        "all_db": test_labeled_db,
        "all_extxyz": test_extxyz,
        "easy_db": test_easy_db,
        "easy_extxyz": test_easy_extxyz,
        "train_pool_db": train_pool_db,
        "split": split_info,
    }


def run_training_loop(parameters, fps_db_path):
    ft = parameters["training"]
    energy_target = float(ft["rmse_energy_target"])
    force_target = float(ft["rmse_force_target"])

    state_path = ft.get("state_file", "training_state.json")
    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    test_size = int(ft.get("test_set_size", 0))
    loop_use_external = bool(ft.get("loop_use_external_labeled_db", False))
    external_labeled_db = None
    fixed_test_info = None
    pool_db_path = fps_db_path

    if loop_use_external:
        if test_size <= 0:
            raise ValueError("loop_use_external_labeled_db requires training.test_set_size > 0")
        external_labeled_db = ft.get("external_labeled_db") or global_labeled_db
        if not external_labeled_db:
            raise ValueError("loop_use_external_labeled_db is True but external_labeled_db path is not set")
        if not os.path.exists(external_labeled_db):
            raise FileNotFoundError(f"External labeled DB not found: {external_labeled_db}")
        if ft.get("fixed_test_extxyz"):
            print("[training] Warning: fixed_test_extxyz is ignored when loop_use_external_labeled_db=True; rebuilding from the external DB.")

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

    fixed_test_db = fixed_test_info.get("all_db") if fixed_test_info else None
    fixed_test_easy_db = fixed_test_info.get("easy_db") if fixed_test_info else None

    state = _load_loop_state(state_path, initial_index=0)
    if state.get("status") == "converged":
        print("[training] Phase-1 already converged; skipping loop.")
        return
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
        batch_size = _round_batch_size(round_index)
        round_dir = os.path.join(os.getcwd(), f"training_round_{round_index:03d}")
        os.makedirs(round_dir, exist_ok=True)
        round_labeled_db = os.path.join(round_dir, f"db_labeled_round_{round_index:03d}.db")
        cumulative_db = os.path.join(round_dir, f"db_labeled_cumulative_{round_index:03d}.db")
        dataset_prefix = os.path.join(round_dir, "mace_dataset")
        dataset_train_path = f"{dataset_prefix}_train.extxyz"
        
        # Check if this round already has labeled cumulative data.
        # If extxyz files are missing, regenerate them from cumulative DB to avoid relabeling.
        round_has_cumulative = (
            os.path.exists(cumulative_db) and
            ase.db.connect(cumulative_db).count() > 0
        )
        
        if round_has_cumulative:
            if os.path.exists(dataset_train_path):
                print(f"[training] Round {round_index}: detected existing labeled DB and training dataset, skipping labeling/prep.")
                datasets = {
                    "train": dataset_train_path,
                }
            else:
                print(f"[training] Round {round_index}: rebuilding missing training extxyz from cumulative labeled DB.")
                datasets = prepare_mace_extxyz(parameters, cumulative_db, out_prefix=dataset_prefix)
            batch_written = ase.db.connect(round_labeled_db).count() if os.path.exists(round_labeled_db) else 0
            if batch_written == 0:
                print(f"[training] Warning: {round_labeled_db} missing or empty; next_fps_index may not advance for this resumed round.")
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
                print("[training] External labeled DB exhausted before reaching RMSE targets.")
                break
            print(f"[training] Round {round_index}: reusing {batch_written} pre-labeled structures starting at index {next_index}.")
            
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
                print("[training] FPS database exhausted before reaching RMSE targets.")
                break

            print(f"[training] Round {round_index}: labeling {batch_written} FPS structures starting at index {next_index}.")
            compute_labels_on_db(parameters, batch_db_path, db_out_path=round_labeled_db)
            _append_labeled_round(round_labeled_db, global_labeled_db)
            
            _write_cumulative_train_db(global_labeled_db, cumulative_db)
            datasets = prepare_mace_extxyz(parameters, cumulative_db, out_prefix=dataset_prefix)

        default_name = ft.get("mace_output_name", parameters.get("name", "mace_training"))
        run_name = f"{default_name}_round{round_index:03d}"

        train_xyz = datasets["train"]
        if not os.path.isabs(train_xyz):
            train_xyz = os.path.abspath(train_xyz)

        val_xyz = datasets.get("val")
        if val_xyz and not os.path.isabs(val_xyz):
            val_xyz = os.path.abspath(val_xyz)

        result = run_mace_training(
            parameters,
            train_xyz,
            val_xyz,
            None,
            workdir=round_dir,
            run_name=run_name,
        )
        phase1_model_src = _find_newest_model_file(result["workdir"], preferred_name=run_name)
        if phase1_model_src:
            latest_model = os.path.abspath("latest.model")
            shutil.copy2(phase1_model_src, latest_model)
            print(f"[training] Updated latest phase-1 model: {latest_model}")
        else:
            print(f"[training] Warning: no .model artifact found in {result['workdir']}")

        if not fixed_test_db:
            raise ValueError("Test DB is required for loop convergence evaluation.")
        if not phase1_model_src:
            raise ValueError("No trained model artifact found; cannot evaluate test/easy RMSE.")

        test_metrics = _evaluate_model_on_labeled_db(parameters, phase1_model_src, fixed_test_db)
        easy_metrics = None
        if fixed_test_easy_db and os.path.exists(fixed_test_easy_db):
            easy_metrics = _evaluate_model_on_labeled_db(parameters, phase1_model_src, fixed_test_easy_db)

        energy_rmse = test_metrics["energy_rmse_mev_per_atom"]
        force_rmse = test_metrics["force_rmse_mev_per_a"]
        print(
            f"[training] Round {round_index} TEST RMSE: "
            f"energy={energy_rmse:.3f} meV/atom, force={force_rmse:.3f} meV/Å"
        )
        if easy_metrics:
            print(
                f"[training] Round {round_index} EASY TEST RMSE: "
                f"energy={easy_metrics['energy_rmse_mev_per_atom']:.3f} meV/atom, "
                f"force={easy_metrics['force_rmse_mev_per_a']:.3f} meV/Å"
            )

        metric_decimals = int(ft.get("state_metric_decimals", 3))
        compact_test_metrics = _compact_eval_metrics(test_metrics, decimals=metric_decimals)
        compact_easy_metrics = _compact_eval_metrics(easy_metrics, decimals=metric_decimals)

        history_entry = {
            "round": round_index,
            "fps_start_index": next_index,
            "fps_count": batch_written,
            "test_rmse": compact_test_metrics,
            "easy_test_rmse": compact_easy_metrics,
            "round_dir": round_dir,
            "phase1_model": phase1_model_src,
        }
        state.setdefault("history", []).append(history_entry)
        state.update({
            "next_fps_index": next_index + batch_written,
            "round_index": round_index + 1,
            "status": "running",
        })
        _save_loop_state(state_path, state)

        if energy_rmse <= energy_target and force_rmse <= force_target:
            print(f"[training] RMSE targets reached in round {round_index}.")
            if phase1_model_src:
                phase1_model = os.path.abspath("phase1.model")
                shutil.copy2(phase1_model_src, phase1_model)
                print(f"[training] Saved converged phase-1 model: {phase1_model}")
            state["status"] = "converged"
            _save_loop_state(state_path, state)
            return

        next_index += batch_written
        round_index += 1

    pool_name = "external labeled" if loop_use_external else "FPS"
    print(f"[training] Loop stopped because the {pool_name} pool was exhausted before convergence.")
    state["status"] = "exhausted"
    _save_loop_state(state_path, state)


def run_training_pipeline(parameters, fps_db_path):
    """Run complete training pipeline: labeling -> dataset prep -> optional training."""
    if "training" not in parameters:
        raise ValueError("No training block in parameters")

    ft = parameters["training"]
    if ft.get("loop_activate"):
        print("[training] Starting iterative training loop...")
        run_training_loop(parameters, fps_db_path)
        if ft.get("phase2_activate", False):
            print("[training] Starting phase-2 relax/refine...")
            run_phase2_relax_refine(parameters, fps_db_path)
        return # if loop is activated, we do not proceed with single-shot training

    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    test_size = int(ft.get("test_set_size", 0))
    fixed_test_info = None
    source_fps_db = fps_db_path

    if test_size > 0:
        fixed_test_info = _ensure_fixed_test_set(parameters, fps_db_path, test_size, ft, global_labeled_db)
        source_fps_db = fixed_test_info.get("train_pool_db", fps_db_path)
    elif ft.get("fixed_test_extxyz"):
        extxyz_path = ft["fixed_test_extxyz"]
        if not os.path.exists(extxyz_path):
            raise FileNotFoundError(f"Configured fixed_test_extxyz not found: {extxyz_path}")
        fixed_test_info = {"extxyz": extxyz_path}

    fixed_test_extxyz = fixed_test_info["extxyz"] if fixed_test_info else None

    do_labeling = ft.get("do_labeling", True)
    do_training = ft.get("do_training", True)

    if do_labeling:
        print("[training] Starting labeling...")
        if not os.path.exists(source_fps_db):
            raise FileNotFoundError(f"Training source DB not found: {source_fps_db}")
        train_total = ase.db.connect(source_fps_db).count()
        if train_total <= 0:
            raise ValueError("No structures available for training after test-set selection")

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

    print("[training] Preparing MACE extxyz dataset...")
    datasets = prepare_mace_extxyz(parameters, db_labeled, out_prefix=ft.get("out_prefix", "mace_dataset"))

    if do_training:
        print("[training] Launching MACE training...")
        test_xyz = fixed_test_extxyz or datasets.get("test")
        if not test_xyz:
            print("[training] Warning: no test set provided to MACE; TEST metrics will be unavailable.")
        run_mace_training(parameters, datasets["train"], datasets.get("val"), test_xyz)
    else:
        print("[training] Training disabled; dataset ready:", datasets)
