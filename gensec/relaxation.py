"""Summary
"""
from gensec.optimize import BFGS_mod
from gensec.optimize import TRM_BFGS
from gensec.optimize import TRM_BFGS_IPI
from gensec.optimize import BFGSLineSearch_mod
from gensec.optimize import LBFGS_Linesearch_mod
from ase.constraints import FixAtoms
from ase.io import write
import os
import imp
import numpy as np
from ase.io import read
from ase.optimize.precon.neighbors import estimate_nearest_neighbour_distance
import gensec.precon as precon
import shutil
import pickle
from ase.optimize.precon import Exp
from ase.io.trajectory import Trajectory


class Calculator:

    """Creates ASE calculator for performing of the geometry optimizations

    Attributes:
        calculator (TYPE): ASE calculator
    """

    def __init__(self, parameters):
        """Loads the calculator from the
        specified ase_parameters file.

        Args:
            parameters {JSON} : Parameters from file
        """
        folder = parameters["calculator"]["supporting_files_folder"]
        ase_file_name = parameters["calculator"]["ase_parameters_file"]
        full_path = os.path.join(os.getcwd(), folder, ase_file_name)
        self.calculator = imp.load_source(ase_file_name, full_path).calculator

    def finished(self, directory):
        """Mark, that calculation finished successfully

        Write file "finished" if the geometry optimiztion is finished

        Arguments:
            directory {str} -- Directory, where the calculation was carried out
        """

        f = open(os.path.join(directory, "finished"), "w")
        f.write("Calculation was finished")
        f.close()

    def set_constrains(self, atoms, parameters):
        """Setting the constrains for geometry optimization

        For now only freezing of the atoms within
        specified values of z-coordinate is implemented.

        Args:
            atoms {Atoms}: ASE Atoms  object
            parameters {JSON} : Parameters from file
        """
        z = parameters["calculator"]["constraints"]["z-coord"]
        c = FixAtoms(
            indices=[atom.index for atom in atoms if atom.position[2] <= z[-1]]
        )
        atoms.set_constraint(c)

    def estimate_mu(self, structure, fixed_frame, parameters):
        """Estimate scaling parameter mu for
        Expoential preconditioner scheme.
        For more implementation detail see Packwood et. al:
        A universal preconditioner for simulating condensed phase materials,
        J. Chem. Phys. 144, 164109 (2016).
        https://aip.scitation.org/doi/full/10.1063/1.4947024

        and

        https://wiki.fysik.dtu.dk/ase/ase/optimize.html

        First reads the parameters file and checks if the
        estimation of mu is necessary. Then Estimates mu
        with default parameters of r_cut=2*r_NN, where r_NN
        is estimated nearest neighbour distance. Parameter
        A=3.0 set to default value as was mentioned in the
        paper.

        Args:
            structure {GenSec structure}: structure object
            fixed_frame {GenSec fixed frame}: fixed frame object
            parameters {JSON} : Parameters from file

        Returns:
            float: Scaling parameter mu
        """
        # Figure out for which atoms Exp is applicapble
        precons_parameters = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["precon"],
            "fixed_frame": parameters["calculator"]["preconditioner"][
                "fixed_frame"
            ]["precon"],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"][
                "precon"
            ],
            "mol-fixed_frame": parameters["calculator"]["preconditioner"][
                "mol-fixed_frame"
            ]["precon"],
        }
        precons_parameters_init = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["initial"],
            "fixed_frame": parameters["calculator"]["preconditioner"][
                "fixed_frame"
            ]["initial"],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"][
                "initial"
            ],
            "mol-fixed_frame": parameters["calculator"]["preconditioner"][
                "mol-fixed_frame"
            ]["initial"],
        }
        precons_parameters_update = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["update"],
            "fixed_frame": parameters["calculator"]["preconditioner"][
                "fixed_frame"
            ]["update"],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"][
                "update"
            ],
            "mol-fixed_frame": parameters["calculator"]["preconditioner"][
                "mol-fixed_frame"
            ]["update"],
        }
        need_for_exp = False
        for i in range(len(list(precons_parameters.values()))):
            if list(precons_parameters.values())[i] == "Exp":
                if (
                    list(precons_parameters_init.values())[i]
                    or list(precons_parameters_update.values())[i]
                ):
                    need_for_exp = True
        mu = 1.0
        if need_for_exp:
            if len(structure.molecules) > 1:
                a0 = structure.molecules[0].copy()
                for i in range(1, len(structure.molecules)):
                    a0 += structure.molecules[i]
            else:
                a0 = structure.molecules[0]
            if hasattr(fixed_frame, "fixed_frame"):
                all_atoms = a0 + fixed_frame.fixed_frame
            else:
                all_atoms = a0
            atoms = all_atoms.copy()
            atoms.set_calculator(self.calculator)
            self.set_constrains(atoms, parameters)
            r_NN = estimate_nearest_neighbour_distance(atoms)
            try:
                mu = Exp(r_cut=2.0 * r_NN, A=3.0).estimate_mu(atoms)[0]
            except:
                print("Something is wrong!")
        return mu

    def relax(self, structure, fixed_frame, parameters, directory):
        """Perform geometry optimization with specified ASE
        calculator.

        Merge the fixed frame and structure object into the
        Atoms object that enters then the geometry optimization
        routine.

        Args:
            structure {GenSec structure}: structure object
            fixed_frame {GenSec fixed frame}: fixed frame object
            parameters {JSON} : Parameters from file
            directory {str} -- Directory, where the calculation was carried out
        """
        if len(structure.molecules) > 1:
            a0 = structure.molecules[0].copy()
            for i in range(1, len(structure.molecules)):
                a0 += structure.molecules[i]
        else:
            a0 = structure.molecules[0]

        if hasattr(fixed_frame, "fixed_frame"):
            all_atoms = a0 + fixed_frame.fixed_frame
        else:
            all_atoms = a0

        # Preconditioner part
        name = parameters["name"]
        atoms = all_atoms.copy()
        self.set_constrains(atoms, parameters)
        atoms.set_calculator(self.calculator)
        if parameters["calculator"]["preconditioner"]["rmsd_update"][
            "activate"
        ]:
            rmsd_threshhold = parameters["calculator"]["preconditioner"][
                "rmsd_update"
            ]["value"]
        else:
            rmsd_threshhold = 100000000000
        if not hasattr(structure, "mu"):
            structure.mu = 1
        if not hasattr(structure, "A"):
            structure.A = 3
        H0 = np.eye(3 * len(atoms)) * 70
        H0_init = precon.preconditioned_hessian(
            structure, fixed_frame, parameters, atoms, H0, task="initial"
        )
        if parameters["calculator"]["algorithm"] == "bfgs":
            opt = BFGS_mod(
                atoms,
                trajectory=os.path.join(
                    directory, "trajectory_{}.traj".format(name)
                ),
                maxstep=0.004,
                initial=a0,
                molindixes=list(range(len(a0))),
                rmsd_dev=rmsd_threshhold,
                structure=structure,
                fixed_frame=fixed_frame,
                parameters=parameters,
                H0=H0_init,
                mu=structure.mu,
                A=structure.A,
                logfile=os.path.join(directory, "logfile.log"),
                restart=os.path.join(directory, "qn.pckl"),
            )
        if parameters["calculator"]["algorithm"] == "bfgs_linesearch":
            opt = BFGSLineSearch_mod(
                atoms,
                trajectory=os.path.join(
                    directory, "trajectory_{}.traj".format(name)
                ),
                initial=a0,
                molindixes=list(range(len(a0))),
                rmsd_dev=rmsd_threshhold,
                maxstep=0.2,
                structure=structure,
                fixed_frame=fixed_frame,
                parameters=parameters,
                H0=H0_init,
                mu=structure.mu,
                A=structure.A,
                logfile=os.path.join(directory, "logfile.log"),
                restart=os.path.join(directory, "qn.pckl"),
                c1=0.23,
                c2=0.46,
                alpha=1.0,
                stpmax=50.0,
                force_consistent=True,
            )
        if parameters["calculator"]["algorithm"] == "lbfgs":
            opt = LBFGS_Linesearch_mod(
                atoms,
                trajectory=os.path.join(
                    directory, "trajectory_{}.traj".format(name)
                ),
                initial=a0,
                molindixes=list(range(len(a0))),
                rmsd_dev=rmsd_threshhold,
                maxstep=0.2,
                structure=structure,
                fixed_frame=fixed_frame,
                parameters=parameters,
                H0_init=H0_init,
                mu=structure.mu,
                A=structure.A,
                logfile=os.path.join(directory, "logfile.log"),
                restart=os.path.join(directory, "qn.pckl"),
                force_consistent=False,
            )
        if parameters["calculator"]["algorithm"] == "trm_nocedal":
            opt = TRM_BFGS(
                atoms,
                trajectory=os.path.join(
                    directory, "trajectory_{}.traj".format(name)
                ),
                maxstep=0.2,
                initial=a0,
                molindixes=list(range(len(a0))),
                rmsd_dev=rmsd_threshhold,
                structure=structure,
                fixed_frame=fixed_frame,
                parameters=parameters,
                H0=H0_init,
                mu=structure.mu,
                A=structure.A,
                logfile=os.path.join(directory, "logfile.log"),
                restart=os.path.join(directory, "qn.pckl"),
            )

        if parameters["calculator"]["algorithm"] == "trm":
            opt = TRM_BFGS_IPI(
                atoms,
                trajectory=os.path.join(
                    directory, "trajectory_{}.traj".format(name)
                ),
                maxstep=0.2,
                initial=a0,
                molindixes=list(range(len(a0))),
                rmsd_dev=rmsd_threshhold,
                structure=structure,
                fixed_frame=fixed_frame,
                parameters=parameters,
                H0=H0_init,
                mu=structure.mu,
                A=structure.A,
                logfile=os.path.join(directory, "logfile.log"),
                restart=os.path.join(directory, "qn.pckl"),
            )

        opt.run(fmax=parameters["calculator"]["fmax"], steps=3000)
        write(
            os.path.join(directory, "final_configuration_{}.in".format(name)),
            atoms,
            format="aims",
        )
        try:
            calculator.close()
        except:
            pass

    def finish_relaxation(self, structure, fixed_frame, parameters, calculator):
        """Finishes unfinished calculation

        Reads the output in logfile and compares to the convergence criteria in
        parameters.json file. If no "finished" reads the trajectory file and
        relax the structure.

        Arguments:
            structure {GenSec structure}: structure object
            fixed_frame {GenSec fixed frame}: fixed frame object
            parameters {JSON} : Parameters from file
            directory {str} -- Directory, where the calculation was carried out
            calculator (ASE Calculator): Calculator for performing relaxation
        """

        def unfinished_directories(working_dir):
            """Return the directories ith unfinished calculations

            Goes throug directories and create list with directories
            where there is no file "finished" in the directory

            Arguments:
                working_dir {str} -- The working directory that is specified
                                    in the parameters file

            Returns:
                [list] -- Directories with unfinished calculations
            """

            dirs = os.listdir(working_dir)
            to_finish = []
            for i in dirs:
                d = os.path.join(working_dir, i)
                if os.path.isdir(d):
                    if not "finished" in os.listdir(d):
                        to_finish.append(d)

            return to_finish

        def find_traj(directory):
            """Check the state of previous calculation

            If there are no trajectory file then calculation was
            interrupted in the very beginning. If there are
            trajectories with "history" in the name then
            the calculation was previously
            restarted.

            Arguments:
                directory {str} -- the directory with unfinished calculation

            Returns:
                [trajectory name] -- returns the trajectory file with the last
                step that have been made.
            """

            for output in os.listdir(directory):
                if (
                    "trajectory" in output
                    and ".traj" in output
                    and "history" not in output
                ):
                    return output
            else:
                return None

        def analyze_traj(directory, traj):
            """Analyze the found trajectory

            1. If the trajectory is None then the calculation didn't started and resumming
            of the calculation should start from the created initial molecular geometry.
            2. If the trajectory file found, we calculate it's size and:
                2.1 If the size is zero and history trajectory file is found -
                rename the last trajectory file to trajectory and perform
                calculation from the this renamed traectory
                2.2 If no history files found, perform calculation
                from initial molecular geometry.
                2.3 If trajectory file found and it's size is not 0 -
                perform restart from this trajectory

            Arguments:
                directory {str} -- the directory with unfinished calculation
                traj {trajectory name} -- name of trajectory file

            Returns:
                bool --True, if the calculation should be performed from the found trajectory file
                       False, if no trajectory file found or only trajectories with "history"
                       found and calculation performed from initialy generated molecular geometry.
            """

            history_trajs = [i for i in os.listdir(directory) if "history" in i]

            if traj is None:
                # No trajectory file found
                if len(history_trajs) > 0:
                    # the history files found - rename the last trajectory file to trajectory
                    # and perform calculation from the this renamed traectory.
                    traj = "trajectory_{}.traj".format(parameters["name"])
                    name_history_traj = "{:05d}_history_{}".format(
                        len(history_trajs), traj
                    )
                    os.rename(
                        os.path.join(directory, name_history_traj),
                        os.path.join(directory, traj),
                    )
                    return True, traj
                else:
                    return False, traj
            else:
                # trajectory file found, let's calculate it's size
                size = os.path.getsize(os.path.join(directory, traj))
                if size == 0:
                    # The size is zero, let's take a look if there are "history" trajectories
                    if len(history_trajs) > 0:
                        # the history files found - rename the last trajectory file to trajectory
                        # and perform calculation from the this renamed traectory.
                        name_history_traj = "{:05d}_history_{}".format(
                            len(history_trajs), traj
                        )
                        os.rename(
                            os.path.join(directory, name_history_traj),
                            os.path.join(directory, traj),
                        )
                        return True, traj
                    else:
                        # No history files found, perform calculation from initial molecular geometry.
                        return False, traj
                else:
                    # trajectory file found and it's size is not 0, perform restart from this trajectory
                    return True, traj

        def send_traj_to_history(directory, traj):
            """Send the trajectory file to history

            Rename the trajectory file in the folder:
                The trajectory file will be renamed to be the last history file.
                The calculation will rewrite the trajectory file but the last
                step will be taken from the just renamed last history trajectory

            Arguments:
                directory {str} -- the directory with unfinished calculation
                traj {trajectory name} -- name of trajectory file
            """

            t = os.path.join(directory, "{}".format(traj))
            history_trajs = [i for i in os.listdir(directory) if "history" in i]
            name_history_traj = "{:05d}_history_{}".format(
                len(history_trajs) + 1, traj
            )
            shutil.copyfile(t, os.path.join(directory, name_history_traj))

        def concatenate_trajs(directory, traj):
            """Concantenate trajectories

            Merge all the history and trajectory files
            in one full trajectory of relaxation

            Arguments:
                directory {str} -- the directory with unfinished calculation
                traj {trajectory name} -- name of trajectory file

            """

            history_trajs = [i for i in os.listdir(directory) if "history" in i]
            temp_traj = Trajectory(os.path.join(directory, "temp.traj"), "a")
            for t in history_trajs:
                tt = Trajectory(os.path.join(directory, t))
                # Merge history trajectories without clashed last steps in each
                # Since the first step and the last step in the trajectories are the same
                for at in tt[:-1]:
                    temp_traj.write(at)
                tt.close()
            last_traj = Trajectory(os.path.join(directory, traj))
            for at in last_traj:
                temp_traj.write(at)
            last_traj.close()
            temp_traj.close()
            os.rename(
                os.path.join(directory, "temp.traj"),
                os.path.join(directory, traj),
            )
            # Cleaning up
            for i in history_trajs:
                os.remove(os.path.join(directory, i))

        def check_restart_file(directory):
            """Check fo rrestart file

            If the restart Hessian file is corrupted it will delete it

            Arguments:
                directory {str} -- the directory with unfinished calculation
            """

            restart_file = os.path.join(directory, "qn.pckl")
            try:
                if os.path.isfile(restart_file):
                    with open(restart_file, "rb") as fd:
                        print(restart_file)
                        l = pickle.load(fd)
                else:
                    pass
            except:
                os.remove(restart_file)

        # Working directory
        # print("Entering the directory")
        working_dir = os.getcwd()
        to_finish_dirs = unfinished_directories(working_dir)
        for d in to_finish_dirs:
            check_restart_file(d)
            traj = find_traj(d)
            found, traj = analyze_traj(d, traj)
            if found:
                # perform restart from this trajectory
                # print("Calculation will be performed from trajectory", traj)
                # Send the trajectory file to history
                send_traj_to_history(d, traj)
                t = Trajectory(os.path.join(d, traj))
                atoms = t[-1].copy()
                # apply positions from atoms aobject to structure and fixed frame
                structure.set_structure_positions(atoms)
                if fixed_frame is not None:
                    fixed_frame.set_fixed_frame_positions(structure, atoms)

                calculator.relax(structure, fixed_frame, parameters, d)
                concatenate_trajs(d, traj)
                calculator.finished(d)
                try:
                    calculator.close()
                except:
                    pass
            else:
                # Perform restart from initial geometry generated in the folder
                # print("Calculation will be performed from molecular geometry")
                foldername = os.path.basename(os.path.normpath(d))
                structure_file = os.path.join(d, foldername + ".in")
                print(structure_file)
                # Clean up
                for i in os.listdir(d):
                    if os.path.join(d, i) != structure_file:
                        os.remove(os.path.join(d, i))
                atoms = read(os.path.join(structure_file), format="aims")
                # apply positions from atoms aobject to structure and fixed frame
                structure.set_structure_positions(atoms)
                if fixed_frame is not None:
                    fixed_frame.set_fixed_frame_positions(structure, atoms)

                calculator.relax(structure, fixed_frame, parameters, d)
                concatenate_trajs(d, traj)
                calculator.finished(d)
                try:
                    calculator.close()
                except:
                    pass
