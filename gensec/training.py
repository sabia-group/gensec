import json
import os
import shutil
import subprocess

import ase.db
import numpy as np
from ase.io import read, write

from gensec.relaxation import load_source
from gensec.training_tools import (
    _append_labeled_round,
    _compact_eval_metrics,
    _copy_labeled_subset_db,
    _ensure_fixed_test_set,
    _evaluate_model_on_labeled_db,
    _extract_fps_batch,
    _filter_highest_force_bins,
    _find_newest_model_file,
    _fps_select_db,
    _load_loop_state,
    _relax_db_with_model,
    _round_batch_size,
    _save_loop_state,
    _select_low_energy_db,
    _write_cumulative_train_db,
)

__all__ = [
    "compute_labels_on_db",
    "prepare_mace_extxyz",
    "run_mace_training",
    "run_one_shot_training",
    "run_phase2_relax_refine",
    "run_training_loop",
    "run_training_pipeline",
]


def _k_grid_from_cell(atoms, reference_atoms, k_density):
    """Choose a simple k-point grid from the structure cell.

    The calculator file used for labeling is reused for structures with
    different cell sizes. This helper scales the k-grid from a reference cell
    so larger cells get fewer k-points and smaller cells get more.

    Args:
        atoms: Structure that will be labeled.
        reference_atoms: Structure read from the fixed-frame input file.
        k_density: Multiplicative density factor from ``training.k_density``.

    Returns:
        Tuple ``(kx, ky, kz)``. The current workflow keeps ``kz = 1`` because
        these systems are treated as slab-like/surface-like cells.
    """

    cell = np.asarray(atoms.get_cell())
    ref_cell = np.asarray(reference_atoms.get_cell())

    if cell.size == 0 or ref_cell.size == 0:
        return (1, 1, 1)

    lengths = np.linalg.norm(cell[:3], axis=1)
    ref_lengths = np.linalg.norm(ref_cell[:3], axis=1)

    kx = max(1, int(np.ceil(k_density * ref_lengths[0] / lengths[0])))
    ky = max(1, int(np.ceil(k_density * ref_lengths[1] / lengths[1])))
    kz = 1

    return (kx, ky, kz)


def _db_has_rows(path):
    return os.path.exists(path) and ase.db.connect(path).count() > 0


def _resume_key_for_row(row):
    data = dict(row.data) if row.data else {}
    for key in ("phase2_source_row_id", "source_row_id", "fps_row_id"):
        if key in data:
            return key, data[key]
    return "row_id", row.id


def compute_labels_on_db(parameters, db_in_path, db_out_path="db_labeled.db", resume_existing=False):
    """Compute reference energies and forces for all structures in an ASE DB.

    This is the expensive labeling step. For every input row, it loads the ASE
    calculator configured in ``training.supporting_files_folder`` /
    ``training.ase_parameters_file``, updates its k-grid for the current cell,
    then stores ``REF_energy`` and flattened ``REF_forces`` in ``row.data`` of
    the output DB.

    Args:
        parameters: Full GenSec parameters dictionary.
        db_in_path: ASE database containing unlabeled candidate structures.
        db_out_path: ASE database to create with successfully labeled rows.
        resume_existing: If True, keep an existing output DB and skip input
            rows already present there. The default keeps the historical
            behavior and recreates the output DB from scratch.

    Returns:
        Path to ``db_out_path``.

    Notes:
        Failed single-point calculations are skipped, not fatal for the whole
        batch. That is intentional for generated structures where a few bad
        geometries can still slip through.
    """

    ft = parameters["training"]
    folder = ft["supporting_files_folder"]
    ase_file = ft["ase_parameters_file"]
    k_density = float(ft.get("k_density", 30.0))
    fullpath = os.path.join(os.getcwd(), folder, ase_file)

    ref_file = parameters["fixed_frame"]["filename"]
    ref_format = parameters["fixed_frame"]["format"]
    ref_atoms = read(ref_file, format=ref_format)

    db_in = ase.db.connect(db_in_path)
    if os.path.exists(db_out_path) and not resume_existing:
        os.remove(db_out_path)
    db_out = ase.db.connect(db_out_path)
    existing_keys = set()
    if resume_existing and os.path.exists(db_out_path):
        existing_keys = {
            _resume_key_for_row(row)
            for row in db_out.select()
            if row.data and "REF_energy" in row.data and "REF_forces" in row.data
        }
        if existing_keys:
            print(f"[training] Resuming labels from {db_out_path}: found {len(existing_keys)} existing structures.")

    skipped = 0
    written = 0
    reused = 0

    for row in db_in.select():
        row_key = _resume_key_for_row(row)
        if row_key in existing_keys:
            print(f"[training] Reusing existing label for structure id={row.id}")
            reused += 1
            continue

        print(f"[training] Processing structure id={row.id}")
        atoms = row.toatoms()
        k_grid = _k_grid_from_cell(atoms, ref_atoms, k_density)

        # Reload the user calculator for each row so calculations do not carry
        # stale state between structures.
        module = load_source(ase_file, fullpath)
        calc = module.calculator
        calc.parameters.update({"k_grid": list(k_grid)})
        atoms.set_calculator(calc)

        try:
            energy = float(atoms.get_potential_energy())
            forces = np.asarray(atoms.get_forces(), dtype=float)
            print(f"[training] Successfully computed E/F for structure {row.id}")
        except Exception as exc:
            print(f"[training] Skipping row {row.id}: calculation failed: {exc}")
            skipped += 1
            continue

        data = dict(row.data) if hasattr(row, "data") else {}
        data["REF_energy"] = energy
        data["REF_forces"] = forces.flatten().tolist()

        db_out.write(atoms, data=data)
        written += 1

    print(f"[training] Labeling finished: wrote {written} structures, reused {reused}, skipped {skipped} structures.")
    return db_out_path


def prepare_mace_extxyz(parameters, db_in_path, out_prefix="mace_dataset"):
    """Convert a labeled ASE DB into a MACE-readable extxyz dataset.

    MACE reads labels from atom metadata, not directly from ASE DB row data.
    This function copies each labeled row to ``<out_prefix>_train.extxyz`` and
    moves ``REF_energy`` and ``REF_forces`` to the places MACE expects.

    Args:
        parameters: Full GenSec parameters dictionary. Kept in the signature so
            this function matches the other training-stage functions.
        db_in_path: ASE DB containing rows labeled by ``compute_labels_on_db``.
        out_prefix: Prefix used for the generated extxyz file.

    Returns:
        Dict of dataset paths. Currently only ``{"train": ...}`` is produced.

    Raises:
        ValueError: If the input DB is empty.
    """

    db = ase.db.connect(db_in_path)
    if db.count() == 0:
        raise ValueError("Input DB is empty: " + db_in_path)

    paths = {
        "train": f"{out_prefix}_train.extxyz",
    }
    for path in paths.values():
        if os.path.exists(path):
            os.remove(path)

    for row in db.select():
        atoms = row.toatoms()
        atoms.calc = None

        # The labels live in ASE row.data; MACE wants them on the Atoms object.
        if hasattr(row, "data") and row.data:
            if "REF_energy" in row.data:
                atoms.info["REF_energy"] = float(row.data["REF_energy"])
            if "REF_forces" in row.data:
                forces_flat = np.array(row.data["REF_forces"])
                atoms.arrays["REF_forces"] = forces_flat.reshape(len(atoms), 3)

        write(paths["train"], atoms, format="extxyz", append=True)
    return paths


def run_mace_training(parameters, train_xyz, valid_xyz=None, test_xyz=None, workdir=None, run_name=None):
    """Launch ``mace_run_train`` with GenSec defaults plus user overrides.

    This function does not train in Python directly. It builds the command-line
    call to MACE, starting from conservative defaults and then applying
    ``training.mace_args`` from the input file. This keeps the protocol code
    small while still allowing advanced MACE options to pass through.

    Args:
        parameters: Full GenSec parameters dictionary.
        train_xyz: Training extxyz file.
        valid_xyz: Optional validation extxyz file. If present, MACE's internal
            ``valid_fraction`` is removed and this explicit file is used.
        test_xyz: Optional fixed test extxyz file passed to MACE.
        workdir: Directory where MACE should run and write logs/models.
        run_name: Optional MACE output name. Defaults to
            ``training.mace_output_name``.

    Returns:
        Dict with the run name, working directory, expected log directory, and
        exact command list used.

    Raises:
        RuntimeError: If ``mace_run_train`` is not available in PATH.
        subprocess.CalledProcessError: If MACE exits with a non-zero status.
    """

    ft = parameters["training"]

    # MACE runs inside ``workdir``. Resolve dataset paths from the directory in
    # which GenSec was started so relative paths do not silently change meaning.
    train_xyz = os.path.abspath(train_xyz)
    if valid_xyz:
        valid_xyz = os.path.abspath(valid_xyz)
    if test_xyz:
        test_xyz = os.path.abspath(test_xyz)

    name = run_name or ft.get("mace_output_name", "mlip-output")
    user_args = ft.get("mace_args", {}) or {}

    mace_exe = shutil.which("mace_run_train")
    if not mace_exe:
        raise RuntimeError("mace_run_train not found in PATH")
    print(f"[training] Using mace_run_train from: {mace_exe}")

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
        ("weight_decay", 5e-07),
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
        base_args = [(key, value) for key, value in base_args if key != "valid_fraction"]
        base_args.append(("valid_file", valid_xyz))
    if test_xyz:
        base_args.append(("test_file", test_xyz))

    # Start from defaults, then let the input file override or remove options.
    merged = {key: value for key, value in base_args}

    for key, value in user_args.items():
        if value is None:
            merged[key] = None
            continue
        if isinstance(value, bool):
            if key in {"swa", "ema"}:
                if value:
                    merged[key] = None
                else:
                    merged.pop(key, None)
                continue
            merged[key] = None if value else "False"
            continue
        if isinstance(value, dict):
            if key == "E0s":
                # MACE expects E0s as a CLI string like "{1: -13.6, 8: -75.0}".
                parts = []
                for element, energy in value.items():
                    try:
                        element_out = int(element)
                    except (TypeError, ValueError):
                        element_out = element
                    parts.append(f"{element_out}: {energy}")
                merged[key] = "{" + ", ".join(parts) + "}"
            else:
                merged[key] = repr(value)
            continue
        if isinstance(value, (list, tuple)):
            if key == "atomic_numbers":
                merged[key] = "[" + ",".join(str(int(item)) for item in value) + "]"
            else:
                merged[key] = ",".join(str(item) for item in value)
            continue
        merged[key] = value

    if "E0s" not in merged or merged["E0s"] in (None, "", "{}"):
        print(
            "[training] Suggestion: `training.mace_args.E0s` is not set. "
            "Training usually works better when element reference energies (E0s) are provided."
        )

    cmd = [mace_exe]
    for key, value in merged.items():
        if value is None:
            # Boolean CLI flags such as --swa are represented by None.
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")

    print("[training] running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=workdir)

    return {
        "run_name": name,
        "workdir": workdir or os.getcwd(),
        "log_dir": os.path.join(workdir or os.getcwd(), "logs"),
        "command": cmd,
    }


def run_phase2_relax_refine(parameters, fps_db_path):
    """Run the optional phase-2 model-guided refinement step.

    Phase 1 trains from the FPS pool. Phase 2 then uses the best/latest phase-1
    model to relax the full FPS pool cheaply, keeps the low-energy candidates,
    FPS-selects a smaller diverse subset, labels that subset with the reference
    calculator, appends it to the phase-1 training data, and trains once more.

    Args:
        parameters: Full GenSec parameters dictionary.
        fps_db_path: Original FPS-selected candidate database used by phase 1.

    Returns:
        None. Outputs are written to ``training.phase2_folder`` and model copies
        are written as ``latest.model`` / ``final.model`` in the run directory.

    Raises:
        FileNotFoundError: If the phase-1 state/model/source DB is missing.
        ValueError: If phase 1 has no completed round to augment.
    """

    ft = parameters["training"]
    if not ft.get("loop_activate", False):
        print("[training] Phase-2 requested without loop_activate; skipping phase-2.")
        return

    phase2_dir = os.path.abspath(ft.get("phase2_folder", "training_relax"))
    os.makedirs(phase2_dir, exist_ok=True)
    fps_db_path = _filter_highest_force_bins(
        fps_db_path,
        ft.get("force_filtered_db", "db_force_filtered.db"),
        ft.get("exclude_highest_force_bins", 0),
    )

    state_path = ft.get("state_file", "training_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"Phase-2 requires phase-1 state file: {state_path}")
    with open(state_path, "r", encoding="utf-8") as handle:
        state = json.load(handle)
    if state.get("phase2", {}).get("status") == "completed":
        print("[training] Phase-2 already completed; skipping.")
        return

    # Phase 2 starts from the best model we have from phase 1.
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

    # Relax broadly, filter by low predicted energy, then restore diversity
    # before paying for reference labels.
    if _db_has_rows(relax_db):
        print(f"[training] Phase-2 resume: reusing relaxed DB {relax_db}")
    else:
        _relax_db_with_model(
            parameters,
            source_db,
            model_path,
            relax_db,
            fmax=float(ft.get("phase2_relax_fmax", 0.01)),
            steps=int(ft.get("phase2_relax_steps", 20)),
        )

    if _db_has_rows(lowe_db):
        print(f"[training] Phase-2 resume: reusing low-energy DB {lowe_db}")
    else:
        _select_low_energy_db(
            relax_db,
            lowe_db,
            fraction=float(ft.get("phase2_low_energy_fraction", 1.0 / 3.0)),
            min_count=int(ft.get("phase2_low_energy_min", 50)),
        )

    if _db_has_rows(lowe_fps_db):
        print(f"[training] Phase-2 resume: reusing FPS DB {lowe_fps_db}")
    else:
        _fps_select_db(
            lowe_db,
            lowe_fps_db,
            n_select=int(ft.get("phase2_fps_n_select", 50)),
        )

    compute_labels_on_db(
        parameters,
        lowe_fps_db,
        db_out_path=lowe_fps_label_db,
        resume_existing=True,
    )

    # Augment the last phase-1 cumulative labeled set, then train one final
    # model on phase-1 labels plus the new phase-2 labels.
    if not state.get("history"):
        raise ValueError("Phase-2 requires at least one completed phase-1 round in state history.")
    last_round = int(state["history"][-1].get("round"))
    phase1_round_dir = os.path.abspath(f"training_round_{last_round:03d}")
    phase1_db = os.path.join(phase1_round_dir, f"db_labeled_cumulative_{last_round:03d}.db")
    if not os.path.exists(phase1_db):
        raise FileNotFoundError(f"Could not find phase-1 cumulative DB: {phase1_db}")

    shutil.copy2(phase1_db, augmented_db)
    _append_labeled_round(lowe_fps_label_db, augmented_db)

    datasets = prepare_mace_extxyz(
        parameters,
        augmented_db,
        out_prefix=os.path.join(phase2_dir, "mace_dataset_augmented"),
    )

    test_xyz = None
    if ft.get("fixed_test_extxyz"):
        test_xyz = ft["fixed_test_extxyz"]
    else:
        # Reuse the fixed test set from phase 1 when one was generated.
        test_extxyz = ft.get("test_set_extxyz", "mace_dataset_test.extxyz")
        if os.path.exists(test_extxyz):
            test_xyz = test_extxyz

    phase2_result = run_mace_training(
        parameters,
        datasets["train"],
        datasets.get("val"),
        test_xyz,
        workdir=phase2_dir,
        run_name=f"{ft.get('mace_output_name', 'mlip-output')}_phase2",
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


def run_training_loop(parameters, fps_db_path):
    """Run the iterative active-learning-style MACE training loop.

    Each round takes the next batch from the training pool, labels it if needed,
    appends it to the cumulative labeled DB, trains a MACE model, and evaluates
    the model on the fixed full/easy test sets. The loop stops when both RMSE
    targets are reached or when the pool is exhausted.

    Args:
        parameters: Full GenSec parameters dictionary.
        fps_db_path: FPS-selected DB from the generation stage. This is either
            the actual training pool or the source from which a test set is
            removed before training starts.

    Returns:
        None. Progress is saved in ``training.state_file`` so the loop can be
        resumed round by round.

    Raises:
        ValueError: For missing test configuration, empty pools, missing model
        artifacts, or impossible external-labeled-db settings.
        FileNotFoundError: If an external labeled DB is requested but missing.
    """

    ft = parameters["training"]
    energy_target = float(ft["rmse_energy_target"])
    force_target = float(ft["rmse_force_target"])

    state_path = ft.get("state_file", "training_state.json")
    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    test_size = int(ft.get("test_set_size", 0))
    loop_use_external = bool(ft.get("loop_use_external_labeled_db", False))
    exclude_force_bins = ft.get("exclude_highest_force_bins", 0)
    filtered_db = ft.get("force_filtered_db", "db_force_filtered.db")
    fixed_test_info = None
    pool_db_path = fps_db_path

    # External mode means the expensive REF labels already exist. We still build
    # the same fixed test/train split so convergence is measured the same way.
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

        external_labeled_db = _filter_highest_force_bins(
            external_labeled_db, filtered_db, exclude_force_bins
        )

        fixed_test_info = _ensure_fixed_test_set(
            parameters,
            fps_db_path,
            test_size,
            ft,
            global_labeled_db,
            labeled_source_db=external_labeled_db,
            label_func=compute_labels_on_db,
        )
        pool_db_path = fixed_test_info.get("train_pool_db", external_labeled_db)
    else:
        # Normal mode: create the fixed test set first, label it, and remove
        # those structures from the pool used for later training batches.
        fps_db_path = _filter_highest_force_bins(
            fps_db_path, filtered_db, exclude_force_bins
        )
        if test_size > 0:
            fixed_test_info = _ensure_fixed_test_set(
                parameters,
                fps_db_path,
                test_size,
                ft,
                global_labeled_db,
                label_func=compute_labels_on_db,
            )
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
        pool_conn = ase.db.connect(pool_db_path)
        total_pool = pool_conn.count()
        if total_pool == 0:
            raise ValueError(f"External labeled DB is empty: {external_labeled_db}")
    else:
        fps_db = ase.db.connect(pool_db_path)
        total_pool = fps_db.count()
        if total_pool == 0:
            raise ValueError(f"FPS database is empty: {pool_db_path}")

    while next_index < total_pool:
        # One directory per round keeps logs, datasets, and model artifacts easy
        # to inspect without overwriting earlier attempts.
        batch_size = _round_batch_size(round_index)
        round_dir = os.path.join(os.getcwd(), f"training_round_{round_index:03d}")
        os.makedirs(round_dir, exist_ok=True)
        round_labeled_db = os.path.join(round_dir, f"db_labeled_round_{round_index:03d}.db")
        cumulative_db = os.path.join(round_dir, f"db_labeled_cumulative_{round_index:03d}.db")
        dataset_prefix = os.path.join(round_dir, "mace_dataset")
        dataset_train_path = f"{dataset_prefix}_train.extxyz"

        round_has_cumulative = (
            os.path.exists(cumulative_db) and
            ase.db.connect(cumulative_db).count() > 0
        )

        if round_has_cumulative:
            # Resume path: if the labeled DB already exists, avoid relabeling
            # and just rebuild missing extxyz files if necessary.
            if os.path.exists(dataset_train_path):
                print(f"[training] Round {round_index}: detected existing labeled DB and training dataset, skipping labeling/prep.")
                datasets = {"train": dataset_train_path}
            else:
                print(f"[training] Round {round_index}: rebuilding missing training extxyz from cumulative labeled DB.")
                datasets = prepare_mace_extxyz(parameters, cumulative_db, out_prefix=dataset_prefix)
            batch_written = ase.db.connect(round_labeled_db).count() if os.path.exists(round_labeled_db) else 0
            if batch_written == 0:
                print(f"[training] Warning: {round_labeled_db} missing or empty; next_fps_index may not advance for this resumed round.")
        elif loop_use_external:
            # External labeled DB mode only copies rows; no reference calculator
            # is called inside the loop.
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
            # Normal mode labels the next FPS slice with the reference calculator.
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

        default_name = ft.get("mace_output_name", "mlip-output")
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

        # Convergence is judged on our fixed labeled DBs, not on a changing MACE
        # validation split.
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
            f"energy={energy_rmse:.3f} meV/atom, force={force_rmse:.3f} meV/A"
        )
        if easy_metrics:
            print(
                f"[training] Round {round_index} EASY TEST RMSE: "
                f"energy={easy_metrics['energy_rmse_mev_per_atom']:.3f} meV/atom, "
                f"force={easy_metrics['force_rmse_mev_per_a']:.3f} meV/A"
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
            # Keep a stable phase-1 model path for phase 2 and for users.
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


def run_one_shot_training(parameters, fps_db_path):
    """Run the non-iterative training path once.

    This is the simpler mode: optionally carve out a fixed test set, label the
    remaining structures, write the MACE extxyz dataset, and launch one MACE
    training job. There is no phase-2 refinement here; phase 2 belongs to the
    iterative loop workflow.

    Args:
        parameters: Full GenSec parameters dictionary.
        fps_db_path: FPS-selected database from generation.

    Returns:
        None. The function writes the labeled DB, extxyz dataset, and optional
        MACE training outputs to disk.

    Raises:
        ValueError: If the selected training set is empty, or labeling is
        disabled without an existing labeled DB.
        FileNotFoundError: If a configured fixed test file or training DB is
        missing.
    """

    ft = parameters["training"]
    global_labeled_db = ft.get("global_labeled_db", "db_labeled_global.db")
    test_size = int(ft.get("test_set_size", 0))
    fixed_test_info = None
    source_fps_db = _filter_highest_force_bins(
        fps_db_path,
        ft.get("force_filtered_db", "db_force_filtered.db"),
        ft.get("exclude_highest_force_bins", 0),
    )

    # In one-shot mode we can still carve out a fixed test set first, so the
    # training data does not include those structures.
    if test_size > 0:
        fixed_test_info = _ensure_fixed_test_set(
            parameters,
            source_fps_db,
            test_size,
            ft,
            global_labeled_db,
            label_func=compute_labels_on_db,
        )
        source_fps_db = fixed_test_info.get("train_pool_db", source_fps_db)
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
        # Useful when labels were prepared in an earlier run and we only want to
        # regenerate extxyz or rerun MACE with different hyperparameters.
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


def run_training_pipeline(parameters, fps_db_path):
    """Run the configured training workflow from the protocol entry point.

    This is the public entry function called by ``protocols.py``. It decides
    whether to run the iterative loop or the simpler one-shot training path.
    Phase 2 is only available after the iterative loop, because it depends on a
    phase-1 loop state and model.

    Args:
        parameters: Full GenSec parameters dictionary.
        fps_db_path: FPS-selected database from generation. In the current
            protocol this is normally ``db_generated_fps.db``.

    Returns:
        None. The function writes DBs, extxyz datasets, training logs, and MACE
        model artifacts to disk.

    Raises:
        ValueError: If the ``training`` block is missing or the selected mode
        has invalid inputs.
        FileNotFoundError: If a configured input file or DB is missing.
    """
    if "training" not in parameters:
        raise ValueError("No training block in parameters")

    ft = parameters["training"]
    if ft.get("loop_activate"):
        # Iterative mode owns its own labeling, dataset prep, and convergence
        # evaluation round by round.
        print("[training] Starting iterative training loop...")
        run_training_loop(parameters, fps_db_path)
        if ft.get("phase2_activate", False):
            print("[training] Starting phase-2 relax/refine...")
            run_phase2_relax_refine(parameters, fps_db_path)
        return

    if ft.get("phase2_activate", False):
        print("[training] Warning: phase2_activate is ignored when loop_activate is False.")
    run_one_shot_training(parameters, fps_db_path)
