import os
import numpy as np 
from gensec.modules import *
from ase.io.trajectory import Trajectory
from ase.io import read, write
import shutil

class Blacklist:

    def __init__(self, structure):

        if len(structure.molecules) > 1:
            torsions = np.array([0 for i in structure.list_of_torsions])
            # quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            # value_com = np.array([0, 0, 0])
            # blacklist_one = np.hstack((torsions, quaternion, value_com))
            # blacklist = np.hstack((torsions, quaternion, value_com)) 
            for i in range(len(structure.molecules) - 1):
                blacklist = np.concatenate((torsions, torsions), axis=0)
        else:
            blacklist = np.array([0 for i in structure.list_of_torsions])
            # blacklist = []
            # quaternion = produce_quaternion(0, np.array([0, 0, 1]))
            # value_com = np.array([0, 0, 0])
            # blacklist = np.hstack((torsions, quaternion, value_com))        
        self.blacklist = blacklist 
        self.dir = os.path.join(os.getcwd(), "blacklist")
        self.torsional_diff_degree = 120
        self.criteria = "strict"



    def add_to_blacklist(self, vector):
        self.blacklist = np.vstack((self.blacklist, vector))

    @staticmethod
    def minimal_angle(x, y):
        rad = min((2 * np.pi) - abs(np.deg2rad(x) - np.deg2rad(y)), 
                                abs(np.deg2rad(x) - np.deg2rad(y)))
        return np.rad2deg(rad)

    def torsional_diff(self, point, vector, criteria):
        
        diff_angles = [self.minimal_angle(x, y) for x,y in zip(vector, point)]
        # Checking if structures are similar depending on
        # torsional differences of angles:
        if criteria == "strict":
            if any(a < self.torsional_diff_degree for a in diff_angles):
                similar = True 
            else:
                similar = False

        elif criteria == "loose":
            if all(a < self.torsional_diff_degree for a in diff_angles):
                similar = True 
            else:
                similar = False
        return similar
        # return []

    def find_in_blacklist(self, vector, criteria):
        found = False
        if type(self.blacklist[0]) == np.ndarray:
            for point in self.blacklist[1:]:
                if self.torsional_diff(point, vector, criteria):
                    found = True
                    return found                   
        

    def get_blacklist(self):
        for vec in self.blacklist:
            print(vec)

    def get_len(self):
        return len(self.blacklist)

    def find_traj(self, directory):
        for outputs in os.listdir(directory):
            if "trajectory" in outputs and ".traj" in outputs:
                return outputs
        else:
            return None

    def check_calculated(self, dirs, parameters):

        calculated_dir = os.path.join(os.getcwd(), "search")
        if dirs.dir_num > 0:
            for i in range(1, dirs.dir_num+1):
                traj_name = "{:010d}".format(i)
                calculated_names = [z.split("_")[0] for z in os.listdir(self.dir)]
                traj = self.find_traj(os.path.join(calculated_dir, traj_name))
                if traj is not None:
                    t = Trajectory(os.path.join(calculated_dir, traj_name, traj))
                    if len(t) != calculated_names.count(str(traj_name)):
                        for k in range(len(t)):
                            n = os.path.join(self.dir, "{:010d}_{}.in".format(i, k))
                            write(n, t[k], format="aims")

    def analyze_calculated(self, structure, fixed_frame, parameters):

        t = structure.list_of_torsions

        if parameters["calculator"]["optimize"] == "generate":
            # Check if structures in the blacklist:
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
                    self.add_to_blacklist(torsions)

            # Go through generated structures:
            dir = os.path.join(os.getcwd(), "generate")
            for m in os.listdir(dir):
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
                    self.add_to_blacklist(torsions)

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
                    self.add_to_blacklist(torsions)

    def add_to_blacklist_traj(self, structure, fixed_frame, current_dir):
        t = structure.list_of_torsions
        print(os.path.join(current_dir, self.find_traj(current_dir)))
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
                self.add_to_blacklist(torsions)

