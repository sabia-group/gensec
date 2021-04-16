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

        if not any(parameters["configuration"][i]["known"] for i in parameters["configuration"]):
            print("Keeping history is disabled")
            sys.exit(0)

        # full_vector = []
        t = []  # torsions
        o = []  # orientations
        c = []  # centres of masses  
        for i in range(len(structure.molecules)):
            tt = np.empty(shape = len(structure.list_of_torsions))
            oo = np.array([0,0,0,0])
            cc = np.array([0, 0, 0])

            t.append(tt) 
            o.append(oo) 
            c.append(cc) 

        torsions = np.hstack(t)
        orientations = np.hstack(o)
        coms = np.hstack(c)

        self.torsions = torsions
        self.orientations = orientations
        self.coms = coms

        self.torsional_diff_degree = 20
        self.criteria = "any"
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


    def add_to_known_torsions(self, t):
        self.torsions = np.vstack((self.torsions, t))
 
    def add_to_known_orientations(self, o):
        self.orientations = np.vstack((self.orientations, o))
 
    def add_to_known_coms(self, c):
        self.coms = np.vstack((self.coms, c))

    def add_to_known(self, t, o, c):
        self.add_to_known_torsions(t)
        self.add_to_known_orientations(o)
        self.add_to_known_coms(c)

    @staticmethod
    def minimal_angle(x, y):
        rad = min((2 * np.pi) - abs(np.deg2rad(x) - np.deg2rad(y)), 
                                abs(np.deg2rad(x) - np.deg2rad(y)))
        return np.rad2deg(rad)

    def torsional_diff(self, point, vector, criteria, t):
        
        diff_angles = [self.minimal_angle(x, y) for x,y in zip(vector, point)]
        # Checking if structures are similar depending on
        # torsional differences of angles:
        if criteria == "any":
            # If even one torsion angle is less than specified angle 
            # Structures are considered similar
            if any(a < t for a in diff_angles):
                similar = True 
            else:
                similar = False

        elif criteria == "all":
            # If all torsion angles areany less than specified angle 
            # Structures are considered similar
            if all(a < t for a in diff_angles):
                similar = True 
            else:
                similar = False
        return similar

    def angle_between(vec1, vec2):
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    def orientational_diff(self, point, vector):
        similar = False
        main_vec_angle = angle_between(point[1:], vector[1:])
        if main_vec_angle < 30:
            ratotion_angle = self.minimal_angle(point[0], vector[0])
            if ratotion_angle < 30:
                similar = True
        return similar

    def coms_diff(self, point, vector):
        similar = False
        if np.linalg.norm(point - vector) < 1:
            similar = True
        return similar



    def find_in_known(self, coords, parameters, structure, fixed_frame, criteria, t):
        found = False # For now permutations are not implemented for several molecules.

        # Goes through all the vectors except first one that was generated for template.

        tt, oo, cc = self.get_internal_vector(coords, structure, fixed_frame, parameters)

        # Goes first throug torsions
        if parameters["configuration"]["torsions"]["known"]:
            if len(self.torsions.shape) > 1:
                for point in range(1, len(self.torsions)):
                    if self.torsional_diff(self.torsions[point], tt, criteria=criteria, t=t):
                        found = True
                        index = point
                        break
            else:
                pass
            # if found in torsions then goes through orientation ofr 
            # the same point as was found for torsions
            if found:
                print("Found in torsions")
                if parameters["configuration"]["orientations"]["known"]:
                    print("Check if the orientations are also the same")
                    if self.orientational_diff(self.orientations[point], oo):
                        print("Found in orientations")
                        pass
                    else:
                        print("Even though the torsions are the same the orientations are different")
                        found = False
            if found:
                # If torsions are the same and orientations are the same, check for coms
                # If checking for coms is activated.
                if parameters["configuration"]["coms"]["known"]:
                    if self.coms_diff(self.coms[point], cc):
                        print("Also found in the coms")
                        pass
                    else:
                        print("Structure is unique")
                        found = False
            else:
                print("Structure is unique")
                pass

        # If need to check only through orientations
        else:
            print("Torsions are disabled")
            # Goes first throug orientations
            if parameters["configuration"]["orientations"]["known"]:
                print("going through orientations")
                if len(self.orientations.shape) > 1:
                    for point in range(1, len(self.orientations)):
                        if self.orientational_diff(self.orientations[point], oo):
                            found = True
                            index = point
                            break
                else:
                    print("This is the first structure")
                    pass

                if found:
                    # if found in torsions then goes through coms
                    if parameters["configuration"]["coms"]["known"]:
                        if self.coms_diff(self.coms[point], cc):
                            print("Found in coms")
                            pass
                        else:
                            print("Structure not found in coms")
                            found = False
            else:
                # Only check for positions is activated:
                # goes through positions and check coms:
                if parameters["configuration"]["coms"]["known"]:
                    print("going through coms")
                    if len(self.coms.shape) > 1:
                        for point in range(1, len(self.coms)):
                            if self.coms_diff(self.coms[point], cc):
                                found = True
                                index = point
                                break
                    else:
                        print("This is the first structure")
                        pass                
        return found                   
        

    def get_known(self):
        for vec in self.known:
            pass
            # print(vec)

    def get_len(self):
        return len(self.known)

    def find_traj(self, directory):
        for output in os.listdir(directory):
            if "trajectory" in output and ".traj" in output:
                return output
        else:
            return None

    def check_calculated(self, dirs, parameters):

        calculated_dir = os.getcwd()
        num_run = parameters["calculator"]["optimize"].split("_")[-1]
        if dirs.dir_num > 0:
            # first_dir = 
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
        print(parameters["calculator"]["optimize"].split("_")[-1])
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


    def get_internal_vector(self, configuration, structure, fixed_frame, parameters):
        # get the internal vector
        # template = merge_together(structure, fixed_frame)
        # template = merge_together(structure, fixed_frame)
        # template.set_positions(configuration.get_positions())
        full_vector = {}
        t = []
        o = []
        c = []
        for i in range(len(structure.molecules)):
            len_mol = len(structure.molecules[i])
            coords = configuration.get_positions()[i*len_mol:i*len_mol+len_mol, :]
            structure.molecules[i].set_positions(coords)
            orientation = measure_quaternion(structure.molecules[i], 0, -1)
            com = structure.molecules[i].get_center_of_mass()
            torsions = []
            for torsion in structure.list_of_torsions:
                torsions.append(structure.molecules[i].get_dihedral(
                                                a0=torsion[0],
                                                a1=torsion[1],
                                                a2=torsion[2], 
                                                a3=torsion[3]))
            t.append(torsions)
            o.append(orientation)
            c.append(com)
        return np.hstack(np.array(t)), np.hstack(np.array(o)), np.hstack(np.array(c))


    def analyze_calculated(self, structure, fixed_frame, parameters):

        if parameters["calculator"]["optimize"] == "generate":
            # Check if structures in the known:
            for m in os.listdir(self.dir):
                configuration = read(os.path.join(self.dir, m), format="aims")
                t, o, c = self.get_internal_vector(configuration, structure, fixed_frame, parameters)
                self.add_to_known_torsions(t)
                self.add_to_known_orientations(o)
                self.add_to_known_coms(c)
            # Go through generated structures:
            dir = os.getcwd()
            if len(os.path.split(dir)[0]) == 0:
                dir = os.path.join(os.getcwd(), dir)
            for m in list(filter(os.path.isdir, os.listdir(dir))):
                configuration = read(os.path.join(dir, m, m+".in"), format="aims")
                t, o, c = self.get_internal_vector(configuration, structure, fixed_frame, parameters)
                self.add_to_known_torsions(t)
                self.add_to_known_orientations(o)
                self.add_to_known_coms(c)

        else:
            # Go through working directory
            for m in os.listdir(self.dir):
                configuration = read(os.path.join(self.dir, m), format="aims")
                t, o, c = self.get_internal_vector(configuration, structure, fixed_frame, parameters)
                self.add_to_known_torsions(t)
                self.add_to_known_orientations(o)
                self.add_to_known_coms(c)

    def add_to_known_traj(self, structure, fixed_frame, current_dir):
        t = structure.list_of_torsions
        traj = Trajectory(os.path.join(current_dir, self.find_traj(current_dir)))
        for m in range(len(traj)):
            configuration = traj[m]
            t, o, c = self.get_internal_vector(configuration, structure, fixed_frame, parameters)
            self.add_to_known_torsions(t)
            self.add_to_known_orientations(o)
            self.add_to_known_coms(c)

    def update_known(self, list_a, list_b, structure, fixed_frame, parameters):

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
                t, o, c = self.get_internal_vector(configuration, structure, fixed_frame, parameters)
                self.add_to_known_torsions(t)
                self.add_to_known_orientations(o)
                self.add_to_known_coms(c)
        self.names = bigger

