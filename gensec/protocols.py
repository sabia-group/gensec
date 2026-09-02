"""Create different search protocols
"""

import ase.db
import os
import numpy as np
from ase.io import  write
from gensec.structure import Structure, Fixed_frame
from gensec.modules import all_right, merge_together, run_with_timeout_decorator, return_inf
from gensec.outputs import Directories
from gensec.relaxation import Calculator
from gensec.check_input import Check_input, _is_active
from gensec.supercell_finder import Supercell_finder
from gensec.training import run_training_pipeline
from ase.io.trajectory import Trajectory

from gensec.unit_cell_finder import Unit_cell_finder, gen_base_sheet

# TODO: Add a permanent log containing at least all print outputs

# TODO: There might be a potential for speedups in the SC and UC finder by replacing loops with numpy operations and larger numpy arrays.


def _is_active_config(value):
    """Compatibility helper for legacy bool flags and dict-style activate flags."""
    return _is_active(value)


def _db_has_trajectory(database, trajectory_id):
    """Return True when a search DB already contains this generated row id."""
    if trajectory_id is None:
        return False
    return any(database.select(trajectory=trajectory_id, limit=1))



class Protocol:

    """Summary

    Attributes:
        success (TYPE): Description
        trials (int): Description
    """

    def init(self):
        """Summary

        Args:
            parameters (TYPE): Description
        """
        pass

    @staticmethod
    def db_setup(name: str, cleanup_stale_files: bool = False):
        db_path = name if name.endswith(".db") else name + ".db"
        db_base = db_path[:-3]

        if not os.path.exists(db_path):
            open(db_path, "a").close()

        # In multi-writer mode, lock/journal files may be valid and must not be removed.
        if cleanup_stale_files:
            for stale_file in (db_path + "-journal", db_base + ".lock", db_path + ".lock"):
                try:
                    os.remove(stale_file)
                except FileNotFoundError:
                    pass


    def run(self, parameters):
        """Summary

        Args:
            parameters (TYPE): Description
        """
        parameters = Check_input(parameters)
        generate_active = _is_active_config(parameters.get("protocol", {}).get("generate", False))
        search_active = _is_active_config(parameters.get("protocol", {}).get("search", False))

        if generate_active:
            # connect to the database and start creating structures there
            print("Start generating of the structures")

            self.db_setup("db_generated")
            db_generated = ase.db.connect("db_generated.db")

            self.db_setup("db_generated_frames")
            db_generated_frames = ase.db.connect("db_generated_frames.db")

            self.db_setup("db_relaxed")
            db_relaxed = ase.db.connect("db_relaxed.db")

            self.db_setup("db_trajectories")
            db_trajectories = ase.db.connect("db_trajectories.db")

            self.db_setup("db_generated_visual")
            db_generated_visual = ase.db.connect("db_generated_visual.db")

            self.trials = 0
            self.success = db_generated.count()
            print("Generated structures", db_generated.count())

            # TODO: Implement a checkpoint for the supercell finder so if there is already a database we can use the same supercell
            # TODO: Does still need to be here? Makes things more complicated.But still interesting for cases where you see the cells on images and dont know the exact orientation.
            if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "inputfile":
                supercell_finder = Supercell_finder(parameters)
                supercell_finder.run()
                structure = Structure(parameters, supercell_finder)
                fixed_frame = Fixed_frame(parameters, supercell_finder.S_atoms)
                fixed_frame_sheet = Fixed_frame(parameters, supercell_finder.S_atoms)

            elif parameters["supercell_finder"]["activate"]:
                structure = Structure(parameters)
                fixed_frame = Fixed_frame(parameters)
                base_sheet = gen_base_sheet(structure.atoms, fixed_frame.fixed_frame, num_mol = parameters["number_of_replicas"])
                fixed_frame_sheet = Fixed_frame(parameters, base_sheet)
                supercell_finder = Supercell_finder(parameters, set_unit_cells = False)

            else:
                structure = Structure(parameters)
                fixed_frame = Fixed_frame(parameters)
                # Check if we have a base sheet we want to work with or if we need to create one from a unit cell. Only supposed to be used to chcek for clashes with the configured structure.
                if parameters["fixed_frame"]["activate"]:
                    if not parameters["fixed_frame"]["is_unit_cell"]:
                        fixed_frame_sheet = Fixed_frame(parameters)
                    else:
                        base_sheet = gen_base_sheet(structure.atoms, fixed_frame.fixed_frame, num_mol = parameters["number_of_replicas"])
                        fixed_frame_sheet = Fixed_frame(parameters, base_sheet)

            dirs = Directories(parameters)

            if parameters["configuration"]["check_forces"]["activate"]:
                calculator = Calculator(parameters)

            if "definite" in parameters["configuration"]:
                definite = parameters["configuration"]["definite"]["activate"]
            else:
                definite = False

            while self.success < parameters["success"] and self.trials < parameters["trials"]:
                print(self.trials, self.success)
                # Generate the vector in internal degrees of freedom
                if definite:
                    _, conf = structure.create_configuration(parameters, self.success)
                else:
                    _, conf = structure.create_configuration(parameters)
                # Apply the configuration to structure
                structure.apply_conf(conf)
                if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "find":
                    oriented_mol = structure.atoms_object()

                # Check if that structure is sensible
                is_good = True
                if all_right(structure, fixed_frame_sheet):
                    # Check if it is in database
                    in_db = False
                    if parameters["protocol"]["check_db"]:
                        in_db = True

                        # Currently not recommended to use database check, significantly increases search or makes it even impossible if done wrong
                        if not structure.find_in_database(conf, db_generated, parameters):
                            if not structure.find_in_database(conf, db_relaxed, parameters ):
                                if not structure.find_in_database(conf, db_trajectories, parameters):
                                    in_db = False

                    if not in_db:
                        if parameters["supercell_finder"]["activate"] and parameters["supercell_finder"]["unit_cell_method"] == "find":
                            oriented_mol_with_cell, _, _ = Unit_cell_finder(oriented_mol, parameters = parameters)
                            supercell_finder.set_unit_cell('find', oriented_mol_with_cell)    # TODO: Input parameters (dont restrict to standard ones in definition of the function) and add all to check input
                            try:
                                supercell_finder.run()
                                conf = structure.get_configuration(supercell_finder.F_atoms)
                                fixed_frame_temp = Fixed_frame(parameters, supercell_finder.S_atoms)
                                structure_temp = Structure(parameters, supercell_finder)
                                if all_right(structure_temp, fixed_frame_temp):
                                    if parameters["configuration"]["check_forces"]["activate"] == True:
                                        if "max_atoms" in parameters["supercell_finder"] and len(supercell_finder.joined_atoms) > parameters["supercell_finder"]["max_atoms"]:
                                            print("Too many atoms in the supercell")
                                            print(len(supercell_finder.joined_atoms))
                                            is_good = False

                                        if is_good:
                                            supercell_finder.joined_atoms.calc = calculator.calculator
                                            if not run_with_timeout_decorator(lambda: (supercell_finder.joined_atoms.get_forces() ** 2).sum(axis=1).max(), return_inf,
                                                                                        timeout = parameters["configuration"]["check_forces"]["max_time"]) > parameters["configuration"]["check_forces"]["max_force"] ** 2:
                                                db_generated.write(supercell_finder.F_atoms, **conf)
                                                db_generated_frames.write(supercell_finder.S_atoms, **conf)
                                                db_generated_visual.write(supercell_finder.joined_atoms, **conf)
                                                write("good_luck.xyz",supercell_finder.joined_atoms,format="extxyz")

                                            else:
                                                print("Forces too large")
                                                is_good = False

                                    else:
                                        db_generated.write(supercell_finder.F_atoms, **conf)
                                        db_generated_frames.write(supercell_finder.S_atoms, **conf)
                                        db_generated_visual.write(supercell_finder.joined_atoms, **conf)
                                        write("good_luck.xyz",supercell_finder.joined_atoms,format="extxyz")

                                else:
                                    is_good = False
                            except:
                                print("Supercell finder failed")
                                is_good = False
                        else:
                            merged = merge_together(structure, fixed_frame)
                            if parameters["configuration"]["check_forces"]["activate"]:
                                merged.calc = calculator.calculator
                                if not run_with_timeout_decorator(lambda: (merged.get_forces() ** 2).sum(axis=1).max(), return_inf,
                                                                                timeout = parameters["configuration"]["check_forces"]["max_time"]) > parameters["configuration"]["check_forces"]["max_force"] ** 2:
                                    db_generated.write(structure.atoms_object(), **conf)
                                    db_generated_visual.write(merged,**conf)
                                    write("good_luck.xyz",merged,format="extxyz")

                                else:
                                    print("Forces too large")
                                    is_good = False

                            else:
                                db_generated.write(structure.atoms_object(), **conf)
                                db_generated_visual.write(merged,**conf)
                                write("good_luck.xyz",merged,format="extxyz")

                    else:
                        is_good = False

                else:
                    is_good = False

                if is_good:
                    self.trials = 0
                    self.success = db_generated.count()
                    print("Good", conf)
                    print("Generated structures:", self.success)
                else:
                    if parameters["supercell_finder"]["activate"] and hasattr(supercell_finder, "joined_atoms"):
                        write("bad_luck.xyz",supercell_finder.joined_atoms,format="extxyz")
                    else:
                        write("bad_luck.xyz",merge_together(structure, fixed_frame_sheet),format="extxyz")
                    self.trials += 1

        training_source_db = "db_generated_visual.db"

        fps_selection = parameters.get("fps_selection", {})
        if _is_active_config(fps_selection.get("activate", False)):
            from gensec.fps_selection import run_fps_selection

            training_source_db = run_fps_selection(parameters)

        if "training" in parameters and parameters["training"].get("activate", False):
            print("training.activate True -> running training pipeline")
            run_training_pipeline(parameters, training_source_db)


        if search_active:

            self.db_setup("db_relaxed")
            db_relaxed = ase.db.connect("db_relaxed.db")

            self.db_setup("db_generated_visual")
            db_generated_visual = ase.db.connect("db_generated_visual.db")

            self.db_setup("db_trajectories")
            db_trajectories = ase.db.connect("db_trajectories.db")
            generated_count = db_generated_visual.count()
            if generated_count == 0:
                raise ValueError("No structures in the database. Please generate some first.")

            success_target = parameters["success"]
            if isinstance(success_target, str):
                if success_target.lower() != "all":
                    raise ValueError("Search success must be an integer or 'all'.")
                success_target = generated_count
            else:
                success_target = int(success_target)

            name = parameters["name"]

            print("Start relaxing structures")
            self.success = db_relaxed.count()
            print("Relaxed structures", db_relaxed.count())
            print("Search target structures", success_target)
            structure = Structure(parameters)
            fixed_frame = Fixed_frame(parameters)
            dirs = Directories(parameters)
            calculator = Calculator(parameters)
            search_folder = parameters["protocol"]["search"]["folder"]
            original_cwd = os.getcwd()
            if not os.path.exists(search_folder):
                os.mkdir(search_folder)
            # Perform optimizations in the folder specified in parameters file
            print("Changing Directory!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            os.chdir(search_folder)
            try:
                # Finish unfinished calculations
                calculator.finish_relaxation(
                    structure, fixed_frame, parameters, calculator
                )

                while self.success < success_target:
                    self.success = db_relaxed.count()
                    made_progress = False
                    skipped_existing = 0

                    for num, row in enumerate(db_generated_visual.select()):
                        if self.success >= success_target:
                            break

                        traj_id = row.unique_id if row.unique_id is not None else str(row.id)
                        dirs.dir_num = row.id

                        if parameters["protocol"]["check_db"]:
                            if _db_has_trajectory(db_relaxed, traj_id) or _db_has_trajectory(db_trajectories, traj_id):
                                skipped_existing += 1
                                continue

                        print("Relaxing generated row", row.id)
                        row_atoms = row.toatoms().copy()
                        dirs.create_directory(parameters)
                        dirs.save_to_directory(row_atoms, parameters)
                        calculator.simple_relax(row_atoms, parameters, dirs.current_dir(parameters))

                        calculator.finished(dirs.current_dir(parameters))
                        traj = Trajectory(os.path.join(dirs.current_dir(parameters), "trajectory_{}.traj".format(name)))
                        print("Structure relaxed")

                        if parameters.get("save_trajectories", False):
                            e_min = 100000
                            for i, step in enumerate(traj):
                                full_conf = structure.get_configuration(step)
                                db_trajectories.write(step, **full_conf, trajectory=traj_id)
                                e_min_temp = step._calc.results["energy"]
                                if e_min_temp < e_min:
                                    e_min = e_min_temp
                                    arg_emin = i
                        else:
                            energies = [step._calc.results["energy"] for step in traj]
                            arg_emin = int(np.argmin(energies))

                        full_conf = structure.get_configuration(traj[arg_emin])
                        db_relaxed.write(traj[arg_emin], **full_conf, trajectory=traj_id, step=arg_emin)

                        self.success = db_relaxed.count()
                        made_progress = True

                    if not made_progress and self.success < success_target:
                        raise RuntimeError(
                            "Search could not make progress: no new generated structures were relaxed. "
                            f"Relaxed={self.success}, target={success_target}, "
                            f"skipped_existing={skipped_existing}, generated={db_generated_visual.count()}. "
                            "Lower `success`, add more generated structures, or inspect existing search DBs."
                        )
            finally:
                os.chdir(original_cwd)

            print("Finished relaxations.")
