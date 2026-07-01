import os
import json
import random
import math
import importlib
import ase.db
from ase.io import write
import numpy as np
from ase.constraints import FixAtoms
from ase.optimize import FIRE

TEST_SET_FORCE_BINS = 5
TEST_SET_EASY_DROP_HARDEST_BINS = 2                                                                       

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
    """Read the active-learning loop checkpoint, or make the first one.

    This is what lets a stopped run continue from the next FPS row instead of
    starting the training batches again from zero.
    """
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
    """Write the small JSON checkpoint used by the training loop."""
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2)


def _round_metric(value, decimals=3):
    """Round metrics before saving them, just to keep state files readable."""
    return round(float(value), int(decimals))


def _compact_eval_metrics(metrics, decimals=3):
    """Keep only the two RMSE numbers we usually want to see in the loop state."""
    if not metrics:
        return None
    return {
        "energy_rmse_mev_per_atom": _round_metric(metrics["energy_rmse_mev_per_atom"], decimals),
        "force_rmse_mev_per_a": _round_metric(metrics["force_rmse_mev_per_a"], decimals),
    }


def _find_newest_model_file(search_dir, preferred_name=None):
    """Find the newest MACE ``.model`` file under a training output directory.

    If a preferred name is given, matching files win over other model files.
    """
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
    """Create a MACE calculator from one trained model file.

    This is used for phase-2 relaxation and for our own easy/full test-set
    evaluation, not for the MACE training command itself.
    """
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
    """Relax every structure in a DB with the current MACE model.

    The output DB keeps only successfully relaxed structures and stores the
    relaxed energy/fmax in row.data so the next helpers can sort and filter it.
    """
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
    """Keep the low-energy part of a phase-2 relaxed database.

    This is the cheap exploitation step: after relaxing with the model, keep the
    structures the model thinks are most promising before doing FPS again.
    """
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
    """Run FPS on a database and write only the selected structures.

    Used in phase 2 after the low-energy filter, so the next expensive labels
    are not just many near-identical low-energy structures.
    """
    from gensec.fps_selection import select_structures_fps

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


def _extract_fps_batch(fps_db_path, start_index, batch_size, out_db_path, round_index=None, extra_metadata=None):
    """Copy the next slice of the FPS pool into a temporary DB for labeling.

    The loop advances by row position in the FPS DB, not by ASE row id, because
    row ids can be recreated when databases are rewritten.
    """
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
    """Append one newly labeled round to the cumulative labeled database."""
    src = ase.db.connect(round_db_path)
    dest = ase.db.connect(global_db_path)
    for row in src.select():
        atoms = row.toatoms()
        data = dict(row.data) if row.data else {}
        dest.write(atoms, data=data)


def _build_train_pool_db(source_db_path, exclude_ids, out_db_path):
    """Build the pool used for training batches, excluding fixed test rows.

    The excluded ids are ids in the source database. We store source_row_id so
    we can still trace where a selected/labeled row originally came from.
    """
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
    """Write the training DB from all labeled rows, skipping rows tagged as test."""
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
    """Copy a simple labeled slice from one DB to another.

    This is mostly a convenience helper for creating round/subset databases
    while preserving REF_energy and REF_forces in row.data.
    """
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
    """Convert a labeled ASE DB into the extxyz file MACE expects.

    REF_energy goes into atoms.info and REF_forces into atoms.arrays, which is
    the format the MACE command-line training code reads.
    """
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
    """Return one force-size number for a row, if we can find one.

    We prefer REF_forces because that is the real label. Older/generated rows
    may only have fmax metadata or forces attached to the calculator, so those
    are kept as fallbacks.
    """
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
    """Make a sorted ``(force_marker, row_id)`` list for rows with force info."""
    scored = []
    for row in rows:
        marker = _get_row_force_marker(row)
        if marker is None:
            continue
        scored.append((marker, row.id))
    return sorted(scored, key=lambda item: item[0])


def _filter_highest_force_bins(source_db_path, out_db_path, exclude_highest_bins, n_bins=TEST_SET_FORCE_BINS):
    """Remove the hardest force bins before making the test and training sets.

    Rows keep their original database order, which also preserves an existing
    FPS ordering. Every row must have a force marker: mixing scored and unscored
    structures would make the requested cutoff ambiguous.
    """
    original_value = exclude_highest_bins
    try:
        exclude_highest_bins = int(original_value)
    except (TypeError, ValueError) as exc:
        raise ValueError("training.exclude_highest_force_bins must be 0, 1, or 2.") from exc
    if exclude_highest_bins not in (0, 1, 2) or str(original_value).strip() not in {"0", "1", "2"}:
        raise ValueError("training.exclude_highest_force_bins must be 0, 1, or 2.")
    if exclude_highest_bins == 0:
        return source_db_path

    rows = list(ase.db.connect(source_db_path).select())
    scored = _score_rows_by_force_marker(rows)
    if len(scored) != len(rows):
        raise ValueError(
            "training.exclude_highest_force_bins requires force markers for every candidate "
            f"structure, but found them for {len(scored)}/{len(rows)} rows in {source_db_path}. "
            "Enable force checking during generation, or set "
            "training.exclude_highest_force_bins to 0 to continue without force filtering."
        )

    bins = _split_scored_rows_into_bins(scored, n_bins)
    if exclude_highest_bins >= len(bins):
        raise ValueError(
            f"Cannot remove {exclude_highest_bins} force bins from a database that produced "
            f"only {len(bins)} non-empty bins."
        )
    excluded_ids = {row_id for chunk in bins[-exclude_highest_bins:] for _, row_id in chunk}
    bin_by_id = {
        row_id: bin_index
        for bin_index, chunk in enumerate(bins)
        for _, row_id in chunk
    }

    if os.path.exists(out_db_path):
        os.remove(out_db_path)
    dest = ase.db.connect(out_db_path)
    for row in rows:
        if row.id in excluded_ids:
            continue
        data = dict(row.data) if row.data else {}
        data["force_filter_bin"] = int(bin_by_id[row.id])
        data["force_filter_excluded_highest_bins"] = exclude_highest_bins
        dest.write(row.toatoms(), data=data)

    print(
        f"[training] Force-domain filter: retained {dest.count()}/{len(rows)} structures "
        f"after removing the highest {exclude_highest_bins}/{len(bins)} force bins."
    )
    return out_db_path


def _split_scored_rows_into_bins(scored_rows, n_bins):
    """Split force-sorted rows into equal-count bins from easy to hard."""
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
    """Pick a fixed test set spread across low/mid/high force structures.

    This is the stratified option: it avoids a test set made only of very easy
    low-force structures, while still being reproducible through the sampler.
    """
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


def _pick_random_test_ids(rows, test_size, sampler):
    """Pick a fixed test set randomly when force-bin splitting is not possible."""
    if len(rows) < test_size:
        raise ValueError(
            f"Requested test_set_size={test_size} but source database has only {len(rows)} entries."
        )
    return set(sampler.sample([row.id for row in rows], int(test_size)))


def _split_test_db_easy(labeled_test_db, easy_db_path, n_bins=5, drop_hardest_bins=2):
    """Create the 'easy' test DB by dropping the hardest force bins.

    The full test set stays untouched. This just makes a second evaluation set
    that answers: how well does the model do away from the hardest geometries?
    """
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
    """Evaluate one trained MACE model on a labeled DB and return RMSE numbers.

    This is our own evaluation pass for the fixed full/easy test DBs, separate
    from whatever MACE prints during training.
    """
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


def _ensure_fixed_test_set(
    parameters,
    fps_db_path,
    test_size,
    ft,
    global_labeled_db,
    labeled_source_db=None,
    label_func=None,
):
    """Create or reuse the fixed test set used across all training rounds.

    If the source already has labels/force markers, we try a force-bin split.
    If not, we make a seeded random split first and label it afterwards. In both
    cases the chosen test rows are removed from the training pool.
    """
    if label_func is None:
        raise ValueError("label_func is required to create a new fixed test set.")

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
        excluded_force_bins = int(ft.get("exclude_highest_force_bins", 0))
        if excluded_force_bins:
            existing_rows = list(ase.db.connect(test_labeled_db).select())
            filter_matches = all(
                int((dict(row.data) if row.data else {}).get("force_filter_excluded_highest_bins", -1))
                == excluded_force_bins
                for row in existing_rows
            )
            if not filter_matches:
                raise ValueError(
                    "The existing fixed test set was created with a different force-domain filter. "
                    "Start a fresh training dataset by removing the existing fixed test and train-pool "
                    "artifacts before using training.exclude_highest_force_bins."
                )
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
        print(f"[training] Preparing fixed test set from FPS pool: {fps_db_path}.")
        fps_conn = ase.db.connect(fps_db_path)
        rows = list(fps_conn.select())
        scored = _score_rows_by_force_marker(rows)
        if len(scored) >= test_size:
            print("[training] Using force-bin sampling for fixed test set.")
            pick_info = _pick_test_ids_from_force_bins(scored, test_size, sampler, n_bins=test_force_bins)
            chosen_ids = pick_info["selected_ids"]
            selected_bin_by_id = {
                row_id: bin_idx
                for bin_idx, ids in pick_info["selected_by_bin"].items()
                for row_id in ids
            }
            test_split_strategy = "force_bins"
        else:
            print(
                "[training] FPS pool has insufficient force markers for force-bin sampling "
                f"({len(scored)}/{test_size}); using seeded random fixed test split."
            )
            chosen_ids = _pick_random_test_ids(rows, test_size, sampler)
            selected_bin_by_id = {}
            pick_info = {"marker_by_id": {}}
            test_split_strategy = "random"

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
            if test_split_strategy == "force_bins":
                data["test_bin_index"] = int(selected_bin_by_id[row.id])
                data["test_force_marker"] = float(pick_info["marker_by_id"][row.id])
            data["test_split_strategy"] = test_split_strategy
            test_db.write(atoms, data=data)

        label_func(parameters, test_subset_db, db_out_path=test_labeled_db)
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
