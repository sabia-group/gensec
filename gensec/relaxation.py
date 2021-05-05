from gensec.optimize import BFGS_mod
from gensec.optimize import TRM_BFGS
from gensec.optimize import TRM_BFGS_IPI
from gensec.optimize import BFGSLineSearch_mod
from gensec.optimize import LBFGS_Linesearch_mod
from gensec.optimize import PreconLBFGS_mod
from ase.optimize import BFGSLineSearch
from ase.optimize import BFGS
from ase.optimize import LBFGS
from gensec.defaults import defaults
from ase.constraints import FixAtoms
from ase.io import write
import os
import sys
import imp
import numpy as np
from ase.io import read
from gensec.neighbors import estimate_nearest_neighbour_distance
from ase.utils import longsum
import gensec.precon as precon
import random
from subprocess import Popen
import shutil
import pickle

from ase.optimize.precon import Exp

from ase.io.trajectory import Trajectory


class Calculator:
    def __init__(self, parameters):

        folder = parameters["calculator"]["supporting_files_folder"]
        ase_file_name = parameters["calculator"]["ase_parameters_file"]
        full_path = os.path.join(os.getcwd(), folder, ase_file_name)
        self.calculator = imp.load_source(ase_file_name, full_path).calculator

    def finished(self, directory):
        """ Mark, that calculation finished successfully
        
        Write file "finished" if the 
        
        Arguments:
            directory {str} -- Directory, where the calculation was carried out
        """

        f = open(os.path.join(directory, "finished"), "w")
        f.write("Calculation was finished")
        f.close()

    def set_constrains(self, atoms, parameters):
        z = parameters["calculator"]["constraints"]["z-coord"]
        c = FixAtoms(
            indices=[atom.index for atom in atoms if atom.position[2] <= z[-1]]
        )
        atoms.set_constraint(c)

    def estimate_mu(self, structure, fixed_frame, parameters):

        # Figure out for which atoms Exp is applicapble
        precons_parameters = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["precon"],
            "fixed_frame": parameters["calculator"]["preconditioner"]["fixed_frame"][
                "precon"
            ],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"]["precon"],
            "mol-fixed_frame": parameters["calculator"]["preconditioner"][
                "mol-fixed_frame"
            ]["precon"],
        }
        precons_parameters_init = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["initial"],
            "fixed_frame": parameters["calculator"]["preconditioner"]["fixed_frame"][
                "initial"
            ],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"]["initial"],
            "mol-fixed_frame": parameters["calculator"]["preconditioner"][
                "mol-fixed_frame"
            ]["initial"],
        }
        precons_parameters_update = {
            "mol": parameters["calculator"]["preconditioner"]["mol"]["update"],
            "fixed_frame": parameters["calculator"]["preconditioner"]["fixed_frame"][
                "update"
            ],
            "mol-mol": parameters["calculator"]["preconditioner"]["mol-mol"]["update"],
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

            # hessian_indices = []
            # for i in range(len(structure.molecules)):
            #     hessian_indices.append(["mol{}".format(i) for k in range(len(structure.molecules[i]))])
            # if hasattr(fixed_frame, "fixed_frame"):
            #     hessian_indices.append(["fixed_frame" for k in range(len(fixed_frame.fixed_frame))])
            # hessian_indices = sum(hessian_indices, [])

            # inds = []

            # for j in range(len(all_atoms)):
            #     # print(j, hessian_indices[j])
            #     if "mol" == hessian_indices[j] and precons_parameters["mol"]=="Exp":
            #         inds.append(j)
            #     elif "fixed_frame" == hessian_indices[j] and precons_parameters["fixed_frame"]=="Exp":
            #         inds.append(j)
            #     elif "mol-mol" == hessian_indices[j] and precons_parameters["mol-mol"]=="Exp":
            #         inds.append(j)
            #     if "mol-fixed_frame" == hessian_indices[j] and precons_parameters["mol-fixed_frame"]=="Exp":
            #         inds.append(j)

            # for i in range(len(all_atoms)):
            #     for j in range(len(all_atoms)):
            #         if hessian_indices[i] == hessian_indices[j]:
            #             if "fixed_frame" in hessian_indices[j] and precons_parameters["fixed_frame"]:
            #                 inds.append(j)
            #             elif "mol" in hessian_indices[j] and precons_parameters["mol"]=="Exp":
            #                 inds.append(j)
            #         else:
            #             if "fixed_frame" not in [hessian_indices[i], hessian_indices[j]] and precons_parameters["mol-mol"]=="Exp":
            #                 inds.append(j)
            #             elif precons_parameters["mol-fixed_frame"]=="Exp":
            #                 inds.append(j)

            # atoms = all_atoms[[atom.index for atom in all_atoms if atom.index in list(set(inds))]].copy()
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
        # write(os.path.join(directory, "initial_configuration_{}.in".format(name)), atoms, format="aims" )
        if parameters["calculator"]["preconditioner"]["rmsd_update"]["activate"]:
            rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"][
                "value"
            ]
        else:
            rmsd_threshhold = 100000000000
        if not hasattr(structure, "mu"):
            structure.mu = 1
        if not hasattr(structure, "A"):
            structure.A = 1
        H0 = np.eye(3 * len(atoms)) * 70
        H0_init = precon.preconditioned_hessian(
            structure, fixed_frame, parameters, atoms, H0, task="initial"
        )
        if parameters["calculator"]["algorithm"] == "bfgs":
            opt = BFGS_mod(
                atoms,
                trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
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
                trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
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
                trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
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
                trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
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
                trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
                maxstep=0.15,
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
        # np.savetxt(os.path.join(directory, "hes_{}_final.hes".format(name)), opt.H)
        try:
            calculator.close()
        except:
            pass

    def finish_relaxation(self, structure, fixed_frame, parameters, calculator):
        """ Finishes unfinished calculation
        
        Reads the output in logfile and compares to the convergence criteria in
        parameters.json file. If no "finished" reads the trajectory file and 
        relax the structure.
        
        Arguments:
            structure {[type]} -- Structure object
            fixed_frame {[type]} -- Fixed frame object
            parameters {[type]} -- parameters file 
            directory {[type]} -- the directory with unfinished calculation
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
            
            If there are no trajectory file then calculation was interrupted in the very beginning.
            If there are trajectories with "history" in the name then the calculation was previously 
            restarted.
            
            Arguments:
                directory {str} -- the directory with unfinished calculation
            
            Returns:
                [trajectory name] -- returns the trajectory file with the last step that have been made.
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
            2. If the 
            
            Arguments:
                directory {str} -- the directory with unfinished calculation
                traj {trajectory name} -- name of trajectory file
            
            Returns:
                bool --True, if the calculation should be performed from the found trajectory file
                       False, if no trajectory file found or only trajectories with "history" 
                       found and calculation performed from initialy generated molecular geometry.
            """

            history_trajs = [i for i in os.listdir(directory) if "history" in i]

            if traj == None:
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
            name_history_traj = "{:05d}_history_{}".format(len(history_trajs) + 1, traj)
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
                os.path.join(directory, "temp.traj"), os.path.join(directory, traj)
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

        # if os.path.basename(os.path.normpath(directory)) != format(0, "010d"):
        #     if not "finished" in os.listdir(directory) and not "known" in os.listdir(directory):
        #         traj = find_traj(directory)
        #         if analyze_traj(directory, traj):
        #             if len(structure.molecules) > 1:
        #                 molsize = len(structure.molecules[0])*len(structure.molecules)
        #             else:
        #                 molsize = len(structure.molecules[0])
        #             if parameters["calculator"]["preconditioner"]["rmsd_update"]["activate"]:
        #                 rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]["value"]
        #             else:
        #                 rmsd_threshhold = 100000000000

        #             name = parameters["name"]
        #             # Save the history of trajectory
        #             send_traj_to_history(name, directory)
        #             # Perform relaxation
        #             traj = os.path.join(directory, "trajectory_{}.traj".format(name))
        #             t = Trajectory(os.path.join(directory, traj))
        #             atoms = t[-1].copy()
        #             self.set_constrains(atoms, parameters)
        #             atoms.set_calculator(self.calculator)
        #             H0 = np.eye(3 * len(atoms)) * 70
        #             opt = BFGS_mod(atoms, trajectory=traj,
        #                             initial=atoms[:molsize], molindixes=list(range(molsize)), rmsd_dev=rmsd_threshhold,
        #                             structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0,
        #                             logfile=os.path.join(directory, "logfile.log"),
        #                             restart=os.path.join(directory, 'qn.pckl'))

        #             fmax = parameters["calculator"]["fmax"]
        #             opt.run(fmax=fmax, steps=1000)
        #             concatenate_trajs(name, directory)
        #             try:
        #                 calculator.close()
        #             except:
        #                 pass
        #             finished(directory)

        #         else:
        #             # Didn't perform any step - start relaxation
        #             #from initial .in  geometry.
        #             foldername = os.path.basename(os.path.normpath(directory))
        #             structure_file = os.path.join(directory, foldername+".in")
        #             for i in os.listdir(directory):
        #                 if os.path.join(directory,i)!=structure_file:
        #                     os.remove(os.path.join(directory,i))
        #             atoms = read(os.path.join(directory, foldername+".in"), format="aims")
        #             if len(structure.molecules) > 1:
        #                 molsize = len(structure.molecules[0])*len(structure.molecules)
        #             else:
        #                 molsize = len(structure.molecules[0])
        #             name = parameters["name"]
        #             self.set_constrains(atoms, parameters)
        #             atoms.set_calculator(self.calculator)
        #             traj = os.path.join(directory, "trajectory_{}.traj".format(name))
        #             if parameters["calculator"]["preconditioner"]["rmsd_update"]["activate"]:
        #                 rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]["value"]
        #             else:
        #                 rmsd_threshhold = 100000000000
        #             H0 = np.eye(3 * len(atoms)) * 70
        #             opt = BFGS_mod(atoms, trajectory=traj,
        #                             initial=atoms[:molsize], molindixes=list(range(molsize)), rmsd_dev=rmsd_threshhold,
        #                             structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0,
        #                             logfile=os.path.join(directory, "logfile.log"),
        #                             restart=os.path.join(directory, 'qn.pckl'))

        #             if not hasattr(structure, "mu"):
        #                 structure.mu = 1
        #             if not hasattr(structure, "A"):
        #                 structure.A = 1
        #             opt.H0 = precon.preconditioned_hessian(structure, fixed_frame, parameters, atoms, H0, task="initial")
        #             np.savetxt(os.path.join(directory, "hes_{}.hes".format(name)), opt.H0)
        #             fmax = parameters["calculator"]["fmax"]
        #             opt.run(fmax=fmax, steps=1000)
        #             try:
        #                 calculator.close()
        #             except:
        #                 pass
        #             finished(directory)

        # traj_ID = Trajectory(os.path.join(directory, "trajectory_ID.traj"))
        # traj_precon = Trajectory(os.path.join(directory, "trajectory_precon.traj"))
        # performace = len(traj_ID)/len(traj_precon)
        # rmsd = precon.Kabsh_rmsd(traj_ID[-1], traj_precon[-1], molindixes, removeHs=False)
        # with open(os.path.join(directory, "results.out"), "w") as res:
        # res.write("{} {}".format(round(performace, 2), round(rmsd, 1)))

    # opt.H0 = np.eye(3 * len(atoms)) * 70

    # Run with preconditioner
    # atoms = all_atoms.copy()
    # self.set_constrains(atoms, parameters)
    # atoms.set_calculator(calculator)

    # # For now, should be redone
    # if not hasattr(structure, "mu"):
    #     structure.mu = 1
    # if not hasattr(structure, "A"):
    #     structure.A = 1
    # ###

    # opt = BFGS(atoms, trajectory=os.path.join(directory, "trajectory_precon_auto.traj"),
    #             initial=a0, molindixes=list(range(len(a0))), rmsd_dev=rmsd_threshhold,
    #             structure=structure, fixed_frame=fixed_frame, parameters=parameters, mu=structure.mu, A=structure.mu)
    # opt.H0 = precon.preconditioned_hessian(structure, fixed_frame, parameters)
    # np.savetxt(os.path.join(directory, "Hes_precon_{}.hes".format(str(structure.mu))), opt.H0)
    # opt.run(fmax=1e-3, steps=1000)
    # write(os.path.join(directory, "final_configuration_precon.in"), atoms,format="aims" )

    # sys.exit(0)

    # Preconditioner also goes here


# import numpy as np
# from ase.optimize import BFGS
# from ase.calculators.lj import LennardJones
# from ase.calculators.lammpslib import LAMMPSlib
# from ase.io import read, write
# a0 = read(options.inputfile, format = options.formatfile)

# symbol = a0.get_chemical_symbols()[0]

# ABOHR = 0.52917721 # in AA
# HARTREE = 27.211383 # in eV

# #epsilon = 0.185 # in kcal * mol^(-1)
# #epsilon = 0.000294816 # in Hartree
# #epsilon = 0.008022361 # in eV

# #sigma = R0_vdW[symbol]
# epsilon = 0.0103
# sigma = 3.4

# lammps_header=[
#                 "dimension     3",
#                 "boundary      p p p",
#                 "atom_style    atomic",
#                 "units         metal",
#                 #"neighbor      2.0  bin",
#                 'atom_modify     map array',
#                 ]
# lammps_cmds = [
#                 'pair_style lj/cut 15.0',
#                 'pair_coeff * *  {} {}'.format(epsilon ,sigma),
#                 #'pair_coeff 1 1  0.238 3.405',
#                 #'fix         1 all nve',
#                 ]
# #atom_types={'Ar':1}
# lammps = LAMMPSlib(lmpcmds=lammps_cmds,
#             #atom_types=atom_types,
#             lammps_header=lammps_header,
#             log_file='LOG.log',
#             keep_alive=True)


# #calculator = LennardJones()
# calculator = lammps
# atoms = a0.copy()
# atoms.set_calculator(calculator)

# opt = BFGS(atoms, trajectory=options.outputfile, initial=a0, molindixes=list(range(len(a0))), rmsd_dev=0.1)
# #opt = BFGS(atoms, trajectory=options.outputfile, initial=a0, rmsd_dev=0.1)
# #opt = BFGS(atoms)
# opt.H0 = pd.read_csv(options.hessian, sep='\s+', header=None)
# opt.run(fmax=1e-3, steps=500)
# #xyz_out = options.outputfile.replace(".traj", ".xyz").replace("Trajectories_vdW", "xyz_vdW")
# #write(xyz_out, atoms, format="xyz")
