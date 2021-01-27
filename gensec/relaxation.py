
from gensec.optimize import BFGS_mod
from gensec.optimize import BFGSLineSearch_mod
from gensec.defaults import defaults
from ase.constraints import FixAtoms
from ase.io import write
import os
import sys
import imp
import numpy as np
from ase.io import read
from gensec.neighbors import estimate_nearest_neighbour_distance

import gensec.precon as precon
import random
from subprocess import Popen
import shutil

from ase.io.trajectory import Trajectory

class Calculator:
    def __init__(self, parameters):

        folder = parameters["calculator"]["supporting_files_folder"]
        ase_file_name = parameters["calculator"]["ase_parameters_file"]
        full_path = os.path.join(os.getcwd(), folder, ase_file_name)
        self.calculator = imp.load_source(ase_file_name, full_path).calculator


    def set_constrains(self, atoms, parameters):
        z = parameters["calculator"]["constraints"]["z-coord"]
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2]<=z[-1]])
        atoms.set_constraint(c)

    def estimate_mu(self, structure, fixed_frame, parameters):

        # Figure out for which atoms Exp is applicapble
        precons_parameters = {
            "mol" : parameters["calculator"]["preconditioner"]["mol"]["precon"],
            "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"]["precon"], 
            "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"]["precon"],
            "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol-fixed_frame"]["precon"]
        }
        precons_parameters_init = {
            "mol" : parameters["calculator"]["preconditioner"]["mol"]["initial"],
            "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"]["initial"], 
            "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"]["initial"],
            "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol-fixed_frame"]["initial"]
        }
        precons_parameters_update = {
            "mol" : parameters["calculator"]["preconditioner"]["mol"]["update"],
            "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"]["update"], 
            "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"]["update"],
            "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol-fixed_frame"]["update"]
        }
        need_for_exp = False 
        for i in range(len(list(precons_parameters.values()))):
            if list(precons_parameters.values())[i] == "Exp":
                if list(precons_parameters_init.values())[i] or list(precons_parameters_update.values())[i]:
                    need_for_exp = True

        if need_for_exp:
            if len(structure.molecules) > 1:
                print("more than one")
                a0 = structure.molecules[0].copy()
                for i in range(1, len(structure.molecules)):
                    a0+=structure.molecules[i]
            else:
                a0 = structure.molecules[0]
            if hasattr(fixed_frame, "fixed_frame"):
                all_atoms = a0 + fixed_frame.fixed_frame        
            else:
                all_atoms = a0

            hessian_indices = []
            for i in range(len(structure.molecules)):
                hessian_indices.append(["mol{}".format(i) for k in range(len(structure.molecules[i]))])
            if hasattr(fixed_frame, "fixed_frame"):
                hessian_indices.append(["fixed_frame" for k in range(len(fixed_frame.fixed_frame))])
            hessian_indices = sum(hessian_indices, [])

            inds = []

            for j in range(len(all_atoms)):
                # print(j, hessian_indices[j])
                if "mol" == hessian_indices[j] and precons_parameters["mol"]=="Exp":
                    inds.append(j)
                elif "fixed_frame" == hessian_indices[j] and precons_parameters["fixed_frame"]=="Exp":
                    inds.append(j)
                elif "mol-mol" == hessian_indices[j] and precons_parameters["mol-mol"]=="Exp":
                    inds.append(j)
                if "mol-fixed_frame" == hessian_indices[j] and precons_parameters["mol-fixed_frame"]=="Exp":
                    inds.append(j)


            for i in range(len(all_atoms)):
                for j in range(len(all_atoms)):
                    if hessian_indices[i] == hessian_indices[j]:
                        if "fixed_frame" in hessian_indices[j] and precons_parameters["fixed_frame"]:
                            inds.append(j)
                        elif "mol" in hessian_indices[j] and precons_parameters["mol"]=="Exp":
                            inds.append(j)
                    else:
                        if "fixed_frame" not in [hessian_indices[i], hessian_indices[j]] and precons_parameters["mol-mol"]=="Exp":
                            inds.append(j)
                        elif precons_parameters["mol-fixed_frame"]=="Exp":               
                            inds.append(j)

            atoms = all_atoms[[atom.index for atom in all_atoms if atom.index in list(set(inds))]].copy()
            atoms.set_calculator(self.calculator)
            self.set_constrains(atoms, parameters)
            # Step 1: get energies and forces for initial configuration
            forces_initial = atoms.get_forces()
            coords_initial = atoms.get_positions()
            # # Step 2: make displacements
            # lengths of lattice vectors
            cell_vectors = atoms.get_cell()
            Lx = np.linalg.norm(cell_vectors[0])
            Ly = np.linalg.norm(cell_vectors[1])
            Lz = np.linalg.norm(cell_vectors[2])
            # r_nn - estimation for nearest neighbours
            r_nn = estimate_nearest_neighbour_distance(atoms)
            # Exponential preconditioner with mu=1 and A=1
            P = precon.ExpHessian_P(atoms, mu=1, A=1)
            # Create matrix M=0.01*r_nn*I
            M = r_nn*0.01
            # Displacements
            v = []
            for coord in coords_initial:
                v.append([M*np.sin(coord[0]/Lx), 
                                      M*np.sin(coord[1]/Ly), 
                                      M*np.sin(coord[2]/Lz)])
            v = np.array(v)
            # apply displacements
            atoms.set_positions(coords_initial + v)
            forces_after = atoms.get_forces()
            # forces differences
            force_diff = (forces_after - forces_initial).sum(axis=1)
            v = v.sum(axis=1)
            A = np.dot(v.T, force_diff)
            B = np.dot(v.T, np.dot(P, v))
            mu = A / B
            # calculator.close()
        else:
            mu = 1.0
        return mu

    def relax(self, structure, fixed_frame, parameters, directory, known):

        if len(structure.molecules) > 1:
            a0 = structure.molecules[0].copy()
            for i in range(1, len(structure.molecules)):
                a0+=structure.molecules[i]
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
            rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]["value"]
        else:
            rmsd_threshhold = 100000000000    
        if not hasattr(structure, "mu"):
            structure.mu = 1
        if not hasattr(structure, "A"):
            structure.A = 1        
        H0 = np.eye(3 * len(atoms)) * 70
        H0_init= precon.preconditioned_hessian(structure, fixed_frame, parameters, atoms, H0, task="initial")
        opt = BFGS_mod(atoms, trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)), maxstep=0.04, 
                            initial=a0, molindixes=list(range(len(a0))), rmsd_dev=rmsd_threshhold, 
                            structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0_init,
                            mu=structure.mu, A=structure.A, logfile=os.path.join(directory, "logfile.log"),
                            restart=os.path.join(directory, 'qn.pckl'))  

        # opt = BFGSLineSearch_mod(atoms, trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)), 
        #                     initial=a0, molindixes=list(range(len(a0))), rmsd_dev=rmsd_threshhold, 
        #                     structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0_init,
        #                     mu=structure.mu, A=structure.A, logfile=os.path.join(directory, "logfile.log"),
        #                     restart=os.path.join(directory, 'qn.pckl')) 


        # opt.H0 = H0_init        
        # np.savetxt(os.path.join(directory, "hes_{}.hes".format(name)), opt.H0)
        fmax = parameters["calculator"]["fmax"]
        opt.run(fmax=fmax, steps=10000)
        write(os.path.join(directory, "final_configuration_{}.in".format(name)), atoms,format="aims" )
        # np.savetxt(os.path.join(directory, "hes_{}_final.hes".format(name)), opt.H)
        try:
            calculator.close()
        except:
            pass

    def finish_relaxation(self, structure, fixed_frame, parameters, directory):
        """ Finishes unfinished calculation
        
        Reads the output in logfile and compares to the convergence criteria in
        parameters.json file. If no "finished" reads the trajectory file and 
        relax the structure.
        
        Arguments:
            structure {[type]} -- [description]
            fixed_frame {[type]} -- [description]
            parameters {[type]} -- [description]
            directory {[type]} -- [description]
        """

        def find_traj(directory):        
            for output in os.listdir(directory):
                if "trajectory" in output and ".traj" in output and "history" not in output:
                    return output
            else:
                return None

        def send_traj_to_history(name, directory):
            traj = os.path.join(directory, "trajectory_{}.traj".format(name))
            history_trajs = [i for i in os.listdir(directory) if "history" in i]
            name_history_traj = "{:05d}_history_trajectory_{}.traj".format(len(history_trajs)+1, name)
            shutil.copyfile(traj, os.path.join(directory, name_history_traj))

        def concatenate_trajs(name, directory):
            traj = "trajectory_{}.traj".format(name)
            trajs = [i for i in os.listdir(directory) if "history" in i]
            history_trajs = " ".join(sorted(trajs))
            temp_traj = "temp.traj"
            os.system("cd {} && ase gui {} {} -o {}".format(directory, history_trajs, traj, temp_traj))
            os.rename(os.path.join(directory, temp_traj), os.path.join(directory, traj))
            # Cleaning up
            for i in trajs:
                os.remove(os.path.join(directory, i))

        def finished(directory):
            f = open(os.path.join(directory, "finished"), "w")
            f.write("Calculation was finished")
            f.close()

        def perform_from_last(directory, traj):

            if traj == None:
                return False
            else:
                size = os.path.getsize(os.path.join(directory, traj))
                if size == 0:
                    history_trajs = [i for i in os.listdir(directory) if "history" in i]
                    if len(history_trajs)>0:
                        name_history_traj = "{:05d}_history_{}".format(len(history_trajs), traj)
                        os.rename(os.path.join(directory, name_history_traj), 
                                  os.path.join(directory, traj))
                        return True
                    else:
                        return False
                else:
                    return True

        if os.path.basename(os.path.normpath(directory)) != format(0, "010d"):
            if not "finished" in os.listdir(directory) and not "known" in os.listdir(directory):
                traj = find_traj(directory)
                if perform_from_last(directory, traj): 
                    if len(structure.molecules) > 1:
                        molsize = len(structure.molecules[0])*len(structure.molecules)
                    else:
                        molsize = len(structure.molecules[0])
                    if parameters["calculator"]["preconditioner"]["rmsd_update"]["activate"]:  
                        rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]["value"]
                    else:
                        rmsd_threshhold = 100000000000

                    name = parameters["name"]
                    # Save the history of trajectory
                    send_traj_to_history(name, directory)
                    # Perform relaxation
                    traj = os.path.join(directory, "trajectory_{}.traj".format(name))
                    t = Trajectory(os.path.join(directory, traj))
                    atoms = t[-1].copy()
                    self.set_constrains(atoms, parameters)
                    atoms.set_calculator(self.calculator)
                    H0 = np.eye(3 * len(atoms)) * 70
                    opt = BFGS_mod(atoms, trajectory=traj, 
                                    initial=atoms[:molsize], molindixes=list(range(molsize)), rmsd_dev=rmsd_threshhold, 
                                    structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0,
                                    logfile=os.path.join(directory, "logfile.log"), 
                                    restart=os.path.join(directory, 'qn.pckl')) 

                    fmax = parameters["calculator"]["fmax"]
                    opt.run(fmax=fmax, steps=1000)
                    concatenate_trajs(name, directory)
                    try:
                        calculator.close()
                    except:
                        pass
                    finished(directory)

                else:
                    # Didn't perform any step - start relaxation 
                    #from initial .in  geometry.
                    foldername = os.path.basename(os.path.normpath(directory))
                    structure_file = os.path.join(directory, foldername+".in")
                    for i in os.listdir(directory):
                        if os.path.join(directory,i)!=structure_file:
                            os.remove(os.path.join(directory,i))
                    atoms = read(os.path.join(directory, foldername+".in"), format="aims")
                    if len(structure.molecules) > 1:
                        molsize = len(structure.molecules[0])*len(structure.molecules)
                    else:
                        molsize = len(structure.molecules[0])
                    name = parameters["name"]
                    self.set_constrains(atoms, parameters)  
                    atoms.set_calculator(self.calculator)
                    traj = os.path.join(directory, "trajectory_{}.traj".format(name))
                    if parameters["calculator"]["preconditioner"]["rmsd_update"]["activate"]:  
                        rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]["value"]
                    else:
                        rmsd_threshhold = 100000000000
                    H0 = np.eye(3 * len(atoms)) * 70    
                    opt = BFGS_mod(atoms, trajectory=traj, 
                                    initial=atoms[:molsize], molindixes=list(range(molsize)), rmsd_dev=rmsd_threshhold, 
                                    structure=structure, fixed_frame=fixed_frame, parameters=parameters, H0=H0,
                                    logfile=os.path.join(directory, "logfile.log"), 
                                    restart=os.path.join(directory, 'qn.pckl'))   

                    if not hasattr(structure, "mu"):
                        structure.mu = 1
                    if not hasattr(structure, "A"):
                        structure.A = 1
                    opt.H0 = precon.preconditioned_hessian(structure, fixed_frame, parameters, atoms, H0, task="initial")
                    np.savetxt(os.path.join(directory, "hes_{}.hes".format(name)), opt.H0)
                    fmax = parameters["calculator"]["fmax"]
                    opt.run(fmax=fmax, steps=1000)
                    try:
                        calculator.close()
                    except:
                        pass
                    finished(directory)

        
        #traj_ID = Trajectory(os.path.join(directory, "trajectory_ID.traj"))
        #traj_precon = Trajectory(os.path.join(directory, "trajectory_precon.traj"))
        #performace = len(traj_ID)/len(traj_precon)    
        #rmsd = precon.Kabsh_rmsd(traj_ID[-1], traj_precon[-1], molindixes, removeHs=False)
        #with open(os.path.join(directory, "results.out"), "w") as res:
            #res.write("{} {}".format(round(performace, 2), round(rmsd, 1)))
        
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
