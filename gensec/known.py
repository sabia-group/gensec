import os, sys
import numpy as np 
from gensec.modules import *
from ase.io.trajectory import Trajectory
from ase.io import read, write
import shutil

class Known:
    """ Handling of the known structures
    
    Class for keeping of already known structures 
    in order to avoid repetative calculations
    and generate unique structures.
    """
    def __init__(self, structure, parameters):

        if len(structure.molecules) > 1:
            torsions = np.empty_like(np.array([0 for i in structure.list_of_torsions]))
            # quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            # value_com = np.array([0, 0, 0])
            # known_one = np.hstack((torsions, quaternion, value_com))
            # known = np.hstack((torsions, quaternion, value_com)) 
            for i in range(len(structure.molecules) - 1):
                known = np.concatenate((torsions, torsions), axis=0)
        else:
            known = np.array([0 for i in structure.list_of_torsions])
            # known = []
            # quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            # value_com = np.array([0, 0, 0])
            # known = np.hstack((torsions, quaternion, value_com))        
        self.known = known 
        self.torsional_diff_degree = 90
        self.criteria = "strict"
        self.dir = parameters["calculator"]["known_folder"]

        if len(os.path.split(self.dir)[0]) == 0:
            self.dir = os.path.join(os.getcwd(), self.dir)
            # Means taht known folder in the same directory with os.getcwd()
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)
            else:
                pass
        else:
            # Means taht known folder specified with full path
            if not os.path.exists(self.dir):
                os.mkdir(self.dir)

        self.names = os.listdir(self.dir)



    def add_to_known(self, vector):
        self.known = np.vstack((self.known, vector))

    @staticmethod
    def minimal_angle(x, y):
        rad = min((2 * np.pi) - abs(np.deg2rad(x) - np.deg2rad(y)), 
                                abs(np.deg2rad(x) - np.deg2rad(y)))
        return np.rad2deg(rad)

    def torsional_diff(self, point, vector, criteria, t):
        
        diff_angles = [self.minimal_angle(x, y) for x,y in zip(vector, point)]
        # Checking if structures are similar depending on
        # torsional differences of angles:
        if criteria == "strict":
            if any(a < t for a in diff_angles):
                similar = True 
            else:
                similar = False

        elif criteria == "loose":
            if all(a < t for a in diff_angles):
                similar = True 
            else:
                similar = False
        return similar
        # return []

    def find_in_known(self, vector, criteria, t):
        found = False
        if type(self.known[0]) == np.ndarray:
            for point in self.known[1:]:
                if self.torsional_diff(point, vector, criteria=criteria, t=t):
                    found = True
                    break
        return found                   
        

    def get_known(self):
        for vec in self.known:
            print(vec)

    def get_len(self):
        return len(self.known)

    def find_traj(self, directory):
        for output in os.listdir(directory):
            if "trajectory" in output and ".traj" in outputs:
                return outputs
        else:
            return None

    def check_calculated(self, dirs, parameters):

        calculated_dir = os.getcwd()
        num_run = parameters["calculator"]["optimize"].split("_")[-1]
        if dirs.dir_num > 0:
            for i in range(1, dirs.dir_num+1):
                traj_name = "{:010d}".format(i)
                calculated_names = [z.split("_")[0] for z in os.listdir(self.dir)]
                traj = self.find_traj(os.path.join(calculated_dir, traj_name))
                if traj is not None:
                    t = Trajectory(os.path.join(calculated_dir, traj_name, traj))
                    if len(t) != calculated_names.count(str(traj_name)):
                        for k in range(len(t)):
                            n = os.path.join(self.dir, "{:010d}_{}_{}.in".format(i, k, num_run))
                            write(n, t[k], format="aims")

    def send_traj_to_known_folder(self, dirs, parameters):

        calculated_dir = os.getcwd()
        num_run = parameters["calculator"]["optimize"].split("_")[-1]

        # for i in range(1, dirs.dir_num+1):
        traj_name = "{:010d}".format(dirs.dir_num)
        calculated_names = [z.split("_")[0] for z in os.listdir(self.dir)]
        traj = self.find_traj(os.path.join(calculated_dir, traj_name))
        if traj is not None:
            t = Trajectory(os.path.join(calculated_dir, traj_name, traj))
            if len(t) != calculated_names.count(str(traj_name)):
                for k in range(len(t)):
                    n = os.path.join(self.dir, "{:010d}_{}_{}.in".format(dirs.dir_num, k, num_run))
                    write(n, t[k], format="aims")




    def analyze_calculated(self, structure, fixed_frame, parameters):

        t = structure.list_of_torsions

        if parameters["calculator"]["optimize"] == "generate":
            # Check if structures in the known:
            for m in os.listdir(self.dir):
                configuration = read(os.path.join(self.dir, m), format="aims")
                template = merge_together(structure, fixed_frame)
                template.set_positions(configuration.get_positions())
                # print(template.get_positions())
                for i in range(len(structure.molecules)):
                    len_mol = len(structure.molecules[i])
                    coords = template.get_positions()[i*len_mol:i*len_mol+len_mol, :]
                    structure.molecules[i].set_positions(coords)
                    torsions = []
                    for torsion in t:
                        torsions.append(structure.molecules[i].get_dihedral(
                                                        a1=torsion[0],
                                                        a2=torsion[1],
                                                        a3=torsion[2],
                                                        a4=torsion[3]))
                    self.add_to_known(torsions)

            # Go through generated structures:
            dir = os.getcwd()
            if len(os.path.split(dir)[0]) == 0:
                dir = os.path.join(os.getcwd(), dir)
            for m in list(filter(os.path.isdir, os.listdir(dir))):
                configuration = read(os.path.join(dir, m, m+".in"), format="aims")
                template = merge_together(structure, fixed_frame)
                template.set_positions(configuration.get_positions())
                for i in range(len(structure.molecules)):
                    len_mol = len(structure.molecules[i])
                    coords = template.get_positions()[i*len_mol:i*len_mol+len_mol, :]
                    structure.molecules[i].set_positions(coords)
                    torsions = []
                    for torsion in t:
                        torsions.append(structure.molecules[i].get_dihedral(
                                                        a1=torsion[0],
                                                        a2=torsion[1],
                                                        a3=torsion[2],
                                                        a4=torsion[3]))
                    self.add_to_known(torsions)

        else:
            t = structure.list_of_torsions
            for m in os.listdir(self.dir):
                configuration = read(os.path.join(self.dir, m), format="aims")
                template = merge_together(structure, fixed_frame)
                template.set_positions(configuration.get_positions())
                # print(template.get_positions())
                for i in range(len(structure.molecules)):
                    len_mol = len(structure.molecules[i])
                    coords = template.get_positions()[i*len_mol:i*len_mol+len_mol, :]
                    structure.molecules[i].set_positions(coords)
                    torsions = []
                    for torsion in t:
                        torsions.append(structure.molecules[i].get_dihedral(
                                                        a1=torsion[0],
                                                        a2=torsion[1],
                                                        a3=torsion[2],
                                                        a4=torsion[3]))
                    self.add_to_known(torsions)

    def add_to_known_traj(self, structure, fixed_frame, current_dir):
        t = structure.list_of_torsions
        traj = Trajectory(os.path.join(current_dir, self.find_traj(current_dir)))
        for m in range(len(traj)):
            configuration = traj[m]
            template = merge_together(structure, fixed_frame)
            template.set_positions(configuration.get_positions())
            # print(template.get_positions())
            for i in range(len(structure.molecules)):
                len_mol = len(structure.molecules[i])
                coords = template.get_positions()[i*len_mol:i*len_mol+len_mol, :]
                structure.molecules[i].set_positions(coords)
                torsions = []
                for torsion in t:
                    torsions.append(structure.molecules[i].get_dihedral(
                                                    a1=torsion[0],
                                                    a2=torsion[1],
                                                    a3=torsion[2],
                                                    a4=torsion[3]))
                self.add_to_known(torsions)

    def update_known(self, list_a, list_b, structure, fixed_frame):

        if len(list_a) > len(list_b):
            smaller = set(list_b)
            bigger = set(list_a)
        else:
            smaller = set(list_a)
            bigger = set(list_b)            

        diff = [item for item in bigger if item not in smaller]
        if len(diff) > 0:
            t = structure.list_of_torsions
            for m in diff:
                configuration = read(os.path.join(self.dir, m), format="aims")
                template = merge_together(structure, fixed_frame)
                template.set_positions(configuration.get_positions())
                # print(template.get_positions())
                for i in range(len(structure.molecules)):
                    len_mol = len(structure.molecules[i])
                    coords = template.get_positions()[i*len_mol:i*len_mol+len_mol, :]
                    structure.molecules[i].set_positions(coords)
                    torsions = []
                    for torsion in t:
                        torsions.append(structure.molecules[i].get_dihedral(
                                                        a1=torsion[0],
                                                        a2=torsion[1],
                                                        a3=torsion[2],
                                                        a4=torsion[3]))
                    self.add_to_known(torsions)
        self.names = bigger