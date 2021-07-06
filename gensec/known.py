"""Summary
"""
import os
import sys
import numpy as np
from gensec.modules import *


class Known:
    """Handling of the known structures

    Class for keeping of already known structures
    in order to avoid repetative calculations
    and generate unique structures.

    Attributes:
        coms (TYPE): Description
        criteria (str): Description
        orientations (TYPE): Description
        torsional_diff_degree (int): Description
        torsions (TYPE): Description
    """

    def __init__(self, structure, parameters):
        """Summary

        Args:
            structure (TYPE): Description
            parameters (TYPE): Description
        """
        if not any(
            parameters["configuration"][i]["known"]
            for i in parameters["configuration"]
        ):
            print("Keeping history is disabled")
            sys.exit(0)

        # full_vector = []
        t = []  # torsions
        o = []  # orientations
        c = []  # centres of masses
        for i in range(len(structure.molecules)):
            tt = np.empty(shape=len(structure.list_of_torsions))
            oo = np.array([0, 0, 0, 0])
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
        # self.dir = parameters["calculator"]["known_folder"] # Old mmodule need to delete after revision

        # if len(os.path.split(self.dir)[0]) == 0:
        #     self.dir = os.path.join(os.getcwd(), self.dir)
        #     # Means taht known folder in the same directory with os.getcwd()
        #     if not os.path.exists(self.dir):
        #         os.mkdir(self.dir)
        #     else:
        #         pass
        # else:
        #     # Means taht known folder specified with full path
        #     if not os.path.exists(self.dir):
        #         os.mkdir(self.dir)

        # self.names = os.listdir(self.dir)

    def add_to_known_torsions(self, t):
        """Summary

        Args:
            t (TYPE): Description
        """
        self.torsions = np.vstack((self.torsions, t))

    def add_to_known_orientations(self, o):
        """Summary

        Args:
            o (TYPE): Description
        """
        self.orientations = np.vstack((self.orientations, o))

    def add_to_known_coms(self, c):
        """Summary

        Args:
            c (TYPE): Description
        """
        self.coms = np.vstack((self.coms, c))

    def add_to_known(self, t, o, c):
        """Summary

        Args:
            t (TYPE): Description
            o (TYPE): Description
            c (TYPE): Description
        """
        self.add_to_known_torsions(t)
        self.add_to_known_orientations(o)
        self.add_to_known_coms(c)

    @staticmethod
    def minimal_angle(x, y):
        """Summary

        Args:
            x (TYPE): Description
            y (TYPE): Description

        Returns:
            TYPE: Description
        """
        rad = min(
            (2 * np.pi) - abs(np.deg2rad(x) - np.deg2rad(y)),
            abs(np.deg2rad(x) - np.deg2rad(y)),
        )
        return np.rad2deg(rad)

    def torsional_diff(self, point, vector, criteria, t):
        """Summary

        Args:
            point (TYPE): Description
            vector (TYPE): Description
            criteria (TYPE): Description
            t (TYPE): Description

        Returns:
            TYPE: Description
        """
        diff_angles = [self.minimal_angle(x, y) for x, y in zip(vector, point)]
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
        """Summary

        Args:
            vec1 (TYPE): Description
            vec2 (TYPE): Description

        Returns:
            TYPE: Description
        """
        return np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))

    def orientational_diff(self, point, vector):
        """Summary

        Args:
            point (TYPE): Description
            vector (TYPE): Description

        Returns:
            TYPE: Description
        """
        similar = False
        main_vec_angle = angle_between(point[1:], vector[1:])
        if main_vec_angle < 30:
            ratotion_angle = self.minimal_angle(point[0], vector[0])
            if ratotion_angle < 30:
                similar = True
        return similar

    def coms_diff(self, point, vector):
        """Summary

        Args:
            point (TYPE): Description
            vector (TYPE): Description

        Returns:
            TYPE: Description
        """
        similar = False
        if np.linalg.norm(point - vector) < 0.5:
            similar = True
        return similar

    def find_in_known(
        self, coords, parameters, structure, fixed_frame, criteria, t
    ):
        """Summary

        Args:
            coords (TYPE): Description
            parameters (TYPE): Description
            structure (TYPE): Description
            fixed_frame (TYPE): Description
            criteria (TYPE): Description
            t (TYPE): Description
        """
        found = False  # For now permutations are not implemented for several molecules.

        # Goes through all the vectors except first one that was generated for template.

        tt, oo, cc = self.get_internal_vector(
            coords, structure, fixed_frame, parameters
        )

        # Goes first throug torsions
        if parameters["configuration"]["torsions"]["known"]:
            if len(self.torsions.shape) > 1:
                for point in range(1, len(self.torsions)):
                    if self.torsional_diff(
                        self.torsions[point], tt, criteria=criteria, t=t
                    ):
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
                        print(
                            "Even though the torsions are the same the orientations are different"
                        )
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
                        if self.orientational_diff(
                            self.orientations[point], oo
                        ):
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
        """Summary"""
        for vec in self.known:
            pass
            # print(vec)

    def get_len(self):
        """Summary

        Returns:
            TYPE: Description
        """
        return len(self.known)

    def find_traj(self, directory):
        """Summary

        Args:
            directory (TYPE): Description

        Returns:
            TYPE: Description
        """
        for output in os.listdir(directory):
            if "trajectory" in output and ".traj" in output:
                return output
        else:
            return None

    def get_internal_vector(
        self, configuration, structure, fixed_frame, parameters
    ):
        """Summary

        Args:
            configuration (TYPE): Description
            structure (TYPE): Description
            fixed_frame (TYPE): Description
            parameters (TYPE): Description
        """
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
            coords = configuration.get_positions()[
                i * len_mol : i * len_mol + len_mol, :
            ]
            structure.molecules[i].set_positions(coords)
            orientation = measure_quaternion(structure.molecules[i], 0, -1)
            com = structure.molecules[i].get_center_of_mass()
            torsions = []
            for torsion in structure.list_of_torsions:
                torsions.append(
                    structure.molecules[i].get_dihedral(
                        a0=torsion[0],
                        a1=torsion[1],
                        a2=torsion[2],
                        a3=torsion[3],
                    )
                )
            t.append(torsions)
            o.append(orientation)
            c.append(com)
        return (
            np.hstack(np.array(t)),
            np.hstack(np.array(o)),
            np.hstack(np.array(c)),
        )
