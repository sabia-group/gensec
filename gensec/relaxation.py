
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from ase.io import write
import os
import sys
import imp
import numpy as np
from ase.io.trajectory import Trajectory
from ase.optimize.precon.neighbors import estimate_nearest_neighbour_distance

import gensec.precon as precon
import random
from subprocess import Popen


class Calculator:
    def __init__(self, parameters):
        pass


    def load_calculator(self, parameters):
        folder = parameters["calculator"]["supporting_files_folder"]
        ase_file_name = parameters["calculator"]["ase_parameters_file"]
        full_path = os.path.join(os.getcwd(), folder, ase_file_name)
        ase_file = imp.load_source(ase_file_name, full_path)
        return ase_file

    def set_constrains(self, atoms, parameters):
        z = parameters["calculator"]["constraints"]["z-coord"]
        c = FixAtoms(indices=[atom.index for atom in atoms if atom.position[2]<=z[-1]])
        atoms.set_constraint(c)

    def estimate_mu(self, structure, fixed_frame, parameters):

        # Figure out for which atoms Exp is applicapble
        precons_parameters = {
            "mol" : parameters["calculator"]["preconditioner"]["mol"],
            "fixed_frame" : parameters["calculator"]["preconditioner"]["fixed_frame"], 
            "mol-mol" : parameters["calculator"]["preconditioner"]["mol-mol"],
            "mol-fixed_frame" : parameters["calculator"]["preconditioner"]["mol"]
        }


        if "Exp" in parameters["calculator"]["preconditioner"].values():
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
            calculator = self.load_calculator(parameters).calculator  
            atoms = all_atoms.copy()
            atoms.set_calculator(calculator)
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
            calculator.close()


            # if precons_parameters["mol"]=="Exp":
            #     if len(structure.molecules) > 1:
            #         a0 = structure.molecules[0].copy()
            #         for i in range(1, len(structure.molecules)):
            #             a0+=structure.molecules[i]
            #     else:
            #         a0 = structure.molecules[0]
                
            #     if precons_parameters["fixed_frame"]=="Exp":
            #         all_atoms = a0 + fixed_frame.fixed_frame
            #     else:
            #         all_atoms = a0
            # else:
            #     if precons_parameters["fixed_frame"]=="Exp":
            #         all_atoms = fixed_frame.fixed_frame
            #     else:
            #         print("For nothing estimate mu")
            #         sys.exit(0)
        else:
            mu = 1
        return mu

    def relax(self, structure, fixed_frame, parameters, directory):

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

        symbol = all_atoms.get_chemical_symbols()[0]   
        molindixes = list(range(len(a0)))
        # Preconditioner part 
        name = parameters["name"]
        calculator = self.load_calculator(parameters).calculator 
        atoms = all_atoms.copy()
        self.set_constrains(atoms, parameters)    
        atoms.set_calculator(calculator)
        write(os.path.join(directory, "initial_configuration_{}.in".format(name)), atoms,format="aims" )
        rmsd_threshhold = parameters["calculator"]["preconditioner"]["rmsd_update"]       
        opt = BFGS(atoms, trajectory=os.path.join(directory, "trajectory_{}.traj".format(name)),
                    initial=a0, molindixes=list(range(len(a0))), rmsd_dev=10000, 
                    structure=structure, fixed_frame=fixed_frame, parameters=parameters)
        # For now, should be redone
        if not hasattr(structure, "mu"):
            structure.mu = 1
        if not hasattr(structure, "A"):
            structure.A = 1
        opt.H0 = precon.preconditioned_hessian(structure, fixed_frame, parameters)
        np.savetxt(os.path.join(directory, "hes_{}.hes".format(name)), opt.H0)
        opt.run(fmax=1e-2, steps=1000)
        write(os.path.join(directory, "final_configuration_{}.in".format(name)), atoms,format="aims" )
        calculator.close()
        
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
