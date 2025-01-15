"""Summary
"""
from gensec.modules import *
from ase.io import read, write
from random import random, randint, uniform, choice, sample
import itertools


class Structure:

    """
    Creates the structure object with multiple molecules.

    This class represents a structure composed of multiple molecules. It includes methods for creating the structure based on parameters from a file, creating a configuration for the structure, and creating a torsion and orientation for the structure. The structure object includes attributes for the atoms in the structure, the full and isolated connectivity matrices, a list of torsions, a minimum image convention (MIC) flag, a list of molecules, a mu parameter for the exponential preconditioner, and periodic boundary conditions (PBC).
    """

    def __init__(self, parameters):
        """
        Initializes the structure object.

        This function takes a dictionary of parameters and initializes the structure object based on these parameters. It reads the atoms from a file, creates the full and isolated connectivity matrices, detects rotatable torsions and cycles if activated in the parameters, sets the periodic boundary conditions (PBC) if the minimum image convention (MIC) is activated in the parameters, creates copies of the atoms for each replica, and sets the intramolecular and fixed frame clash parameters. If adsorption is activated in the parameters, it sets the adsorption range and point.

        Args:
            parameters (dict): A dictionary of parameters for initializing the structure object.
        """
        self.parameters = parameters

        self.atoms = read(
            parameters["geometry"][0], format=parameters["geometry"][1]
        )
        self.connectivity_matrix_full = create_connectivity_matrix(
            self.atoms, bothways=True
        )
        self.connectivity_matrix_isolated = create_connectivity_matrix(
            self.atoms, bothways=False
        )

        if parameters["configuration"]["torsions"]["activate"]:
            if (
                parameters["configuration"]["torsions"]["list_of_tosrions"]
                == "auto"
            ):
                self.list_of_torsions = detect_rotatble(
                    self.connectivity_matrix_isolated, self.atoms
                )
                self.cycles = detect_cycles(self.connectivity_matrix_full)
                self.list_of_torsions = exclude_rotatable_from_cycles(
                    self.list_of_torsions, self.cycles
                )

            else:
                self.list_of_torsions = parameters["configuration"]["torsions"][
                    "list_of_tosrions"
                ]

        if parameters["mic"]["activate"] == True:
            self.pbc = parameters["mic"]["pbc"]
            self.mic = True
            self.atoms.set_cell(self.pbc)
            self.atoms.set_pbc(True)
            self.mu = None  # Parameter mu for exponential preconditioner
        self.molecules = [
            self.atoms.copy() for i in range(parameters["number_of_replicas"])
        ]
        self.clashes_intramolecular = parameters["configuration"]["clashes"][
            "intramolecular"
        ]
        self.clashes_with_fixed_frame = parameters["configuration"]["clashes"][
            "with_fixed_frame"
        ]
        if parameters["configuration"]["adsorption"]["activate"]:
            self.adsorption = True
            self.adsorption_range = parameters["configuration"]["adsorption"][
                "range"
            ]
            self.adsorption_point = parameters["configuration"]["adsorption"][
                "point"
            ]
        #if parameters["configuration"]["adsorption_surface"]["activate"]:
            #self.adsorption_surface = True
            #self.adsorption_surface_range = parameters["configuration"][
                #"adsorption_surface"
            #]["range"]
            #self.adsorption_surface_Z = parameters["configuration"][
                #"adsorption_surface"
            #]["surface"]
            #self.adsorption_surface_mols = parameters["configuration"][
                #"adsorption_surface"
            #]["molecules"]
        # if len(self.cycles) > 0:
        # for i in range(len(self.cycles)):
        # self.cycles[i] = make_canonical_pyranosering(self.atoms, self.cycles[i])

    def create_configuration(self, parameters):
        """
        Creates a configuration for the structure.

        This function takes a dictionary of parameters and creates a configuration for the structure based on these parameters. The details of how the configuration is created are not specified in the provided code excerpt.

        Args:
            parameters (dict): A dictionary of parameters for creating the configuration.

        Returns:
            TYPE: The type of the return value is not specified in the provided code excerpt.
        """

        def make_torsion(self, parameters, label):
            """
            Creates a torsion for the structure.

            This function takes a dictionary of parameters and a label, and creates a torsion for the structure based on these parameters. If the torsion values are set to "random" in the parameters, it generates random torsion angles for each torsion in the list of torsions. It returns the torsion angles and a dictionary mapping each torsion to its angle.

            Args:
                parameters (dict): A dictionary of parameters for creating the torsion.
                label (TYPE): A label for the torsion.

            Returns:
                tuple: A tuple containing a numpy array of torsion angles and a dictionary mapping each torsion to its angle.
            """
            if parameters["configuration"]["torsions"]["values"] == "random":
                torsions = np.array(
                    [randint(0, 360) for i in self.list_of_torsions]
                )
            t = {
                "m{}t{}".format(label, i): torsions[i]
                for i in range(len(torsions))
            }
            return torsions, t

        def make_orientation(self, parameters, label):
            """
            Creates an orientation for the structure.

            This function takes a dictionary of parameters and a label, and creates an orientation for the structure based on these parameters. The details of how the orientation is created are not specified in the provided code excerpt.

            Args:
                parameters (dict): A dictionary of parameters for creating the orientation.
                label (TYPE): A label for the orientation.

            Returns:
                TYPE: The type of the return value is not specified in the provided code excerpt.
            """
            if not parameters["configuration"]["orientations"]["activate"]:
                quaternion = [0, 0, 0, 1]
                q = {
                    "m{}q{}".format(label, i): quaternion[i]
                    for i in range(len(quaternion))
                }
                return quaternion, q

            if (
                parameters["configuration"]["orientations"]["values"]
                == "random"
            ):
                quaternion = produce_quaternion(
                    randint(0, 360), np.array([random(), random(), random()])
                )

            elif (
                parameters["configuration"]["orientations"]["values"]
                == "discretized"
            ):
                # Discretizes the values for the main vector of the molecuele
                # for the angle part the number of allowed rotations
                turns = int(
                    360.0
                    // parameters["configuration"]["orientations"]["angle"]
                )

                angles = np.linspace(0, 360, num=turns + 1)
                if (
                    parameters["configuration"]["orientations"]["vector"][
                        "Type"
                    ]
                    == "exclusion"
                ):
                    exclude = np.eye(3)[choice([0, 1, 2])]
                    # print("Exclude")
                    # print(exclude)
                    x = parameters["configuration"]["orientations"]["vector"][
                        "x"
                    ]
                    y = parameters["configuration"]["orientations"]["vector"][
                        "y"
                    ]
                    z = parameters["configuration"]["orientations"]["vector"][
                        "z"
                    ]
                    quaternion = produce_quaternion(
                        choice(angles),
                        np.array(
                            [
                                choice(x) * exclude[0],
                                choice(y) * exclude[1],
                                choice(z) * exclude[2],
                            ]
                        ),
                    )
                else:
                    x = parameters["configuration"]["orientations"]["vector"][
                        "x"
                    ]
                    y = parameters["configuration"]["orientations"]["vector"][
                        "y"
                    ]
                    z = parameters["configuration"]["orientations"]["vector"][
                        "z"
                    ]
                    quaternion = produce_quaternion(
                        choice(angles),
                        np.array([choice(x), choice(y), choice(z)]),
                    )

            else:
                angle = parameters["configuration"]["orientations"]["angle"]
                x = parameters["configuration"]["orientations"]["vector"]["x"]
                y = parameters["configuration"]["orientations"]["vector"]["y"]
                z = parameters["configuration"]["orientations"]["vector"]["z"]
                quaternion = produce_quaternion(
                    randint(angle[0], angle[1]),
                    np.array(
                        [
                            uniform(x[0], x[1]),
                            uniform(y[0], y[1]),
                            uniform(z[0], z[1]),
                        ]
                    ),
                )
            q = {
                "m{}q{}".format(label, i): quaternion[i]
                for i in range(len(quaternion))
            }
            return quaternion, q

        def make_com(self, parameters, label):
            """
            Creates the center of mass (COM) for a molecule in the structure.

            This function takes a dictionary of parameters and a label, and creates the COM for a molecule in the structure based on these parameters. If the COM is not activated in the parameters, it sets the COM to the origin. If the COM values are set to "restricted" in the parameters, it generates the COM within a restricted space defined by the x, y, and z axes in the parameters. Otherwise, it generates the COM within a default space from 0 to 10 on each axis. It returns the COM and a dictionary mapping the label and index to the COM coordinates.

            Args:
                parameters (dict): A dictionary of parameters for creating the COM.
                label (int): A label for the molecule.

            Returns:
                tuple: A tuple containing a numpy array of the COM coordinates and a dictionary mapping the label and index to the COM coordinates.
            """
            if not parameters["configuration"]["coms"]["activate"]:
                com = [0, 0, 0]
                c = {"m{}c{}".format(label, i): com[i] for i in range(len(com))}
                return com, c

            if parameters["configuration"]["coms"]["values"] == "restricted":
                x = parameters["configuration"]["coms"]["x_axis"]
                y = parameters["configuration"]["coms"]["y_axis"]
                z = parameters["configuration"]["coms"]["z_axis"]

                x_space = np.linspace(
                    start=x[0], stop=x[1], num=x[2], endpoint=True
                )
                y_space = np.linspace(
                    start=y[0], stop=y[1], num=y[2], endpoint=True
                )
                z_space = np.linspace(
                    start=z[0], stop=z[1], num=z[2], endpoint=True
                )
                com = np.array(
                    [choice(x_space), choice(y_space), choice(z_space)]
                )
            else:
                com = np.array(
                    [
                        choice(np.linspace(0, 10, 10)),
                        choice(np.linspace(0, 10, 10)),
                        choice(np.linspace(0, 10, 10)),
                    ]
                )
            c = {"m{}c{}".format(label, i): com[i] for i in range(len(com))}
            return com, c

        if parameters["configuration"]["torsions"]["activate"]:
            torsions, t = make_torsion(self, parameters, label=0)
        # if parameters["configuration"]["orientations"]["activate"]:
        quaternion, q = make_orientation(self, parameters, label=0)
        # else:
        # default = [0,0,0,1]
        # quaternion, q = [0,0,0,1], {"m{}q{}".format(0, i) : default[i] for i in range(len(default))}     # Initial orientation
        # if parameters["configuration"]["coms"]["activate"]:
        coms, c = make_com(self, parameters, label=0)
        # else:
        # default = [0, 0, 0]
        # coms, c = [0,0,0], {"m{}c{}".format(0, i) : default[i] for i in range(len(default))}  # put center of mass to the origin
        if not any(
            parameters["configuration"][i]["activate"]
            for i in parameters["configuration"]
        ):
            print("Nothing to sample")
            sys.exit(0)
        else:
            if parameters["configuration"]["torsions"]["activate"]:
                configuration = np.hstack((torsions, quaternion, coms))
                conf = {**t, **q, **c}
            else:
                configuration = np.hstack((quaternion, coms))
                conf = {**q, **c}

        full_conf = {}
        full_conf.update(conf)
        if len(self.molecules) > 1:
            for i in range(1, len(self.molecules)):
                if parameters["configuration"]["torsions"]["activate"]:
                    if parameters["configuration"]["torsions"]["same"]:
                        t_temp = {
                            "m{}t{}".format(i, k): t["m0t{}".format(k)]
                            for k in range(len(t))
                        }
                    else:
                        torsions, t_temp = make_torsion(
                            self, parameters, label=i
                        )

                if parameters["configuration"]["orientations"]["same"]:
                    q_temp = {
                        "m{}q{}".format(i, k): q["m0q{}".format(k)]
                        for k in range(len(q))
                    }
                else:
                    quaternion, q_temp = make_orientation(
                        self, parameters, label=i
                    )

                if parameters["configuration"]["coms"]["same"]:
                    c_temp = {
                        "m{}c{}".format(i, k): c["m0c{}".format(k)]
                        for k in range(len(c))
                    }
                else:
                    coms, c_temp = make_com(self, parameters, label=i)
                if parameters["configuration"]["torsions"]["activate"]:
                    full_conf.update(**t_temp, **q_temp, **c_temp)
                    vec = np.hstack((torsions, quaternion, coms))
                else:
                    full_conf.update(**q_temp, **c_temp)
                    vec = np.hstack((quaternion, coms))
                configuration = np.hstack((configuration, vec))
        # print(full_conf)
        return configuration, full_conf

    def extract_conf_keys_from_row(self):
        """
        Extracts the configuration keys from a row of the database.

        This function reads all keys from a row of the database and returns a list of keys that correspond to the torsional, rotational, and positional degrees of freedom of the structure object. It initializes the torsions to zeros, the quaternion to [0, 0, 0, 1], and the COM to the origin, and creates a dictionary mapping the molecule label and index to these values. It repeats this process for each molecule in the structure.

        Returns:
            list: A list of keys reflecting the configuration on internal degrees of freedom.
        """

        full_conf = {}
        if self.parameters["configuration"]["torsions"]["activate"]:
            t = np.zeros(len(self.list_of_torsions))
        q = [0, 0, 0, 1]
        c = [0, 0, 0]
        for i in range(0, len(self.molecules)):
            if self.parameters["configuration"]["torsions"]["activate"]:
                t_temp = {"m{}t{}".format(i, k): t for k in range(len(t))}
            q_temp = {"m{}q{}".format(i, k): q for k in range(len(q))}
            c_temp = {"m{}c{}".format(i, k): c for k in range(len(c))}
            if self.parameters["configuration"]["torsions"]["activate"]:
                full_conf.update(**t_temp, **q_temp, **c_temp)
            else:
                full_conf.update(**q_temp, **c_temp)

        return list(full_conf.keys())

    def read_configuration(self, atoms_positions):
        """
        Reads the configuration from atoms positions.

        This function takes an ASE Atoms object of atoms positions, and calculates the values of degrees of freedom using the template and list of torsions stored in the structure object. It sets the positions of atoms in each molecule based on the atoms positions, calculates the torsions, orientations, and COM for each molecule, and creates a dictionary mapping the molecule label and index to these values.

        Args:
            atoms_positions (ase.Atoms): An ASE Atoms object.

        Returns:
            dict: A dictionary reflecting the configuration on internal degrees of freedom.
        """

        full_conf = {}
        for ii in range(len(self.molecules)):
            atoms = self.molecules[0].get_positions()
            torsions = []
            positions = atoms_positions.get_positions()[
                ii * len(atoms) : (ii + 1) * len(atoms)
            ]
            self.molecules[ii].set_positions(positions)
            for torsion in self.list_of_torsions:
                torsions.append(
                    self.molecules[ii].get_dihedral(
                        a0=torsion[0],
                        a1=torsion[1],
                        a2=torsion[2],
                        a3=torsion[3],
                    )
                )
            orientations = measure_quaternion(
                self.molecules[ii], 0, len(atoms) - 1
            )
            com = self.molecules[ii].get_center_of_mass()
            t_temp = {
                "m{}t{}".format(ii, k): torsions[k]
                for k in range(len(torsions))
            }
            q_temp = {
                "m{}q{}".format(ii, k): orientations[k]
                for k in range(len(orientations))
            }
            c_temp = {"m{}c{}".format(ii, k): com[k] for k in range(len(com))}
            full_conf.update(**t_temp, **q_temp, **c_temp)

        return full_conf

    def apply_configuration(self, configuration):
        """Summary

        Args:
            configuration (TYPE): Description
        """
        # Old mmodule need to delete after revision
        for i in range(len(self.molecules)):
            k = -1
            for torsion in self.list_of_torsions:
                k += 1
                # +4 quaternion values and +3 COM values
                z = i * (len(self.list_of_torsions) + 4 + 3) + k
                fixed_indices = carried_atoms(
                    self.connectivity_matrix_isolated, torsion
                )
                self.molecules[i].set_dihedral(
                    angle=configuration[z],
                    a1=torsion[0],
                    a2=torsion[1],
                    a3=torsion[2],
                    a4=torsion[3],
                    indices=fixed_indices,
                )
            # Set orientation
            quaternion_set(
                self.molecules[i],
                produce_quaternion(
                    configuration[z + 1],
                    np.array(
                        [
                            configuration[z + 2],
                            configuration[z + 3],
                            configuration[z + 4],
                        ]
                    ),
                ),
                0,
                len(self.molecules[i]) - 1,
            )
            # Set center of mass
            set_centre_of_mass(
                self.molecules[i],
                np.array(
                    [
                        configuration[z + 5],
                        configuration[z + 6],
                        configuration[z + 7],
                    ]
                ),
            )

    def apply_conf(self, conf):
        """Apply confiruration to the structure object

        Getting the configuration in the form of dictionary and applies
        it to the structure object

        Arguments:
            conf (TYPE): Description
            conf {dictionary} -- Dictionary with configuration stored
        """

        for i in range(len(self.molecules)):
            mol_dict = dict(
                filter(lambda item: "m{}".format(i) in item[0], conf.items())
            )
            t_dict = dict(
                filter(
                    lambda item: "m{}t".format(i) in item[0], mol_dict.items()
                )
            )
            q_dict = dict(
                filter(
                    lambda item: "m{}q".format(i) in item[0], mol_dict.items()
                )
            )
            c_dict = dict(
                filter(
                    lambda item: "m{}c".format(i) in item[0], mol_dict.items()
                )
            )

            if self.parameters["configuration"]["torsions"]["activate"]:
                # Set torsions
                if len(self.list_of_torsions) >= 10:

                    t = 0
                    ttt = sample(
                        range(len(self.list_of_torsions)),
                        len(self.list_of_torsions),
                    )
                    # print(sampled_torsions)
                    # for t in range(len(self.list_of_torsions)):
                    while t < len(self.list_of_torsions):
                        fixed_indices = carried_atoms(
                            self.connectivity_matrix_isolated,
                            self.list_of_torsions[ttt[t]],
                        )
                        self.molecules[i].set_dihedral(
                            angle=randint(1, 360),
                            a1=self.list_of_torsions[ttt[t]][0],
                            a2=self.list_of_torsions[ttt[t]][1],
                            a3=self.list_of_torsions[ttt[t]][2],
                            a4=self.list_of_torsions[ttt[t]][3],
                            indices=fixed_indices,
                        )
                        if not internal_clashes(self):
                            print("Torsion {} is ok".format(ttt[t]))
                            t += 1
                    else:
                        pass
                # numtorsions = randint(1, 2)
                # # len(self.list_of_torsions) // 10
                # # print("Select {} of torsions".format(numtorsions))
                # torsions_to_change = sample(
                #     range(len(self.list_of_torsions)), numtorsions
                # )
                # print(torsions_to_change)
                # for t in torsions_to_change:
                #     fixed_indices = carried_atoms(
                #         self.connectivity_matrix_isolated,
                #         self.list_of_torsions[t],
                #     )
                #     self.molecules[i].set_dihedral(
                #         angle=list(t_dict.values())[t],
                #         a1=self.list_of_torsions[t][0],
                #         a2=self.list_of_torsions[t][1],
                #         a3=self.list_of_torsions[t][2],
                #         a4=self.list_of_torsions[t][3],
                #         indices=fixed_indices,
                #     )

                else:
                    for t in range(len(self.list_of_torsions)):
                        fixed_indices = carried_atoms(
                            self.connectivity_matrix_isolated,
                            self.list_of_torsions[t],
                        )
                        self.molecules[i].set_dihedral(
                            angle=list(t_dict.values())[t],
                            a1=self.list_of_torsions[t][0],
                            a2=self.list_of_torsions[t][1],
                            a3=self.list_of_torsions[t][2],
                            a4=self.list_of_torsions[t][3],
                            indices=fixed_indices,
                        )

            # Set orientation
            quaternion_set(
                self.molecules[i],
                list(q_dict.values()),
                0,
                len(self.molecules[i]) - 1,
            )
            # Set center of mass
            set_centre_of_mass(self.molecules[i], list(c_dict.values()))

    def apply_torsions(self, configuration):
        """Summary

        Args:
            configuration (TYPE): Description
        """
        # Old mmodule need to delete after revision
        for i in range(len(self.molecules)):
            k = -1
            for torsion in self.list_of_torsions:
                k += 1
                # +4 quaternion values and +3 COM values
                z = i * (len(self.list_of_torsions) + 4 + 3) + k
                fixed_indices = carried_atoms(
                    self.connectivity_matrix_isolated, torsion
                )
                self.molecules[i].set_dihedral(
                    angle=configuration[z],
                    a1=torsion[0],
                    a2=torsion[1],
                    a3=torsion[2],
                    a4=torsion[3],
                    indices=fixed_indices,
                )

    def torsions_from_conf(self, configuration):
        """Summary

        Args:
            configuration (TYPE): Description

        Returns:
            TYPE: Description
        """
        # Old mmodule need to delete after revision
        torsions = []
        for i in range(len(self.molecules)):
            k = -1
            for torsion in self.list_of_torsions:
                k += 1
                # +4 quaternion values and +3 COM values
                z = i * (len(self.list_of_torsions) + 4 + 3) + k
                torsions.append(configuration[z])
        return torsions

    def atoms_object(self):
        """Convert Structure object to ASE Atoms object

        Goes through Molecules in Structure object and join
        them to one ASE Atoms object

        Returns:
            [ASE Atoms] -- ASE Atoms object
        """

        temp = self.atoms.copy()
        # Create empty list of the appropriate type
        del temp[[atom.index for atom in self.atoms]]
        for molecule in self.molecules:
            temp += molecule
        return temp

    def atoms_object_visual(self, fixed_frame):
        """Convert Structure object to ASE Atoms object

        Goes through Molecules in Structure object and join
        them to one ASE Atoms object

        Returns:
            [ASE Atoms] -- ASE Atoms object
        """

        temp = self.atoms.copy()
        # Create empty list of the appropriate type
        del temp[[atom.index for atom in self.atoms]]
        for molecule in self.molecules:
            temp += molecule
        if hasattr(fixed_frame, 'fixed_frame'):
            temp += fixed_frame.fixed_frame
        return temp

    def set_structure_positions(self, atoms):
        """Apply the coordinates from atoms object

        Set the coordinates from atoms to structure object

        Arguments:
            atoms (TYPE): Description
            atoms {ase atoms object} -- ASE Atoms object with cordinates
        """

        for mol in range(len(self.molecules)):
            l = len(self.molecules[mol])
            mol_atoms = atoms[mol * l : (mol + 1) * l].get_positions()
            self.molecules[mol].set_positions(mol_atoms)

    def find_in_database(self, conf, database, parameters):
        """Check if the configuration is stored in database

        [description]

        Arguments:
            conf (TYPE): Description
            database (TYPE): Description
            parameters (TYPE): Description
            conf {dictionary} -- Conformation stored in dictionary
            database {ASE database} -- ASE database with other configurations
            parameters {JSON} : Parameters from file

        Returns:
            [boolean] -- True if the configuration found in database
        """

        mol_dict = dict(filter(lambda item: "t" in item[0], conf.items()))
        thresh = 5
        periodic_keys = []
        keys = []
        for i in mol_dict:
            if mol_dict[i] < thresh or mol_dict[i] > 360 - thresh:
                periodic_keys.append(i)
            else:
                keys.append(i)

        quries = []
        if len(periodic_keys) == 0:
            quries.append(
                ", ".join(
                    [
                        "{}<{}<{}".format(conf[i] - thresh, i, conf[i] + thresh)
                        for i in keys
                    ]
                )
            )
        else:
            non_periodic_query = ", ".join(
                [
                    "{}<{}<{}".format(conf[i] - thresh, i, conf[i] + thresh)
                    for i in keys
                ]
            )
            periodic_query = {}
            for i in periodic_keys:
                if conf[i] + thresh <= 360:
                    # The value is close to 360
                    periodic_query[i] = {
                        0: "{}<{}".format(i, thresh + conf[i]),
                        1: "{}>{}".format(i, 360 - thresh + conf[i]),
                    }
                else:
                    # the value is close to 0
                    periodic_query[i] = {
                        0: "{}>{}".format(i, conf[i] - thresh),
                        1: "{}<{}".format(i, conf[i] + thresh - 360),
                    }
            lst = list(itertools.product([0, 1], repeat=len(periodic_keys)))
            for i in lst:
                temp = ", ".join(
                    [periodic_query[k][z] for k, z in zip(periodic_keys, i)]
                )
                quries.append(temp + ", " + non_periodic_query)

        found = False
        for q in quries:
            # Check if from query the occurances can be found
            # If the rows are more than 1 then the structure found
            # in terms of torsion angles combination.
            rows = database.select(selection=q)
            for row in rows:
                # Need to implement check for orientations and centres of mass.
                # if parameters["configuration"]["orientations"]["known"]:
                #     quternion_dict = dict(filter(lambda item: "q" in item[0], conf.items()))
                # else:
                #     print(row.id)
                # print(row.m0t0,row.m0t1,row.m0t2)
                # print(conf)
                # print(row.m0c0, row.m0c1, row.m0c2)
                found = True
                return found
        return found


class Fixed_frame:

    """Object for fixed frame.

    Attributes:
        fixed_frame (ASE Atoms): Atoms from specified file
        mic (bool): Description
        pbc (TYPE): Description
    """

    def __init__(self, parameters):
        """
        Initializes the Fixed_frame object.

        This function takes a dictionary of parameters and initializes the Fixed_frame object based on these parameters.
        It sets the minimum image convention (MIC) flag to False by default. If the fixed frame is activated in the parameters, 
        it reads the fixed frame from a file using the ASE read function, with the filename and format specified in the parameters.
        The function does not set the periodic boundary conditions (PBC) for the fixed frame within the provided code excerpt.

        Args:
            parameters (dict): A dictionary of parameters for initializing the Fixed_frame object. 
                            It should contain a "fixed_frame" key with a dictionary value that includes "activate", "filename", and "format" keys.
        """
        # Minimum Image Convention
        self.mic = False
        if parameters["fixed_frame"]["activate"] == True:
            self.fixed_frame = read(
                parameters["fixed_frame"]["filename"],
                format=parameters["fixed_frame"]["format"],
            )

        if parameters["mic"]["activate"] == True:
            self.pbc = parameters["mic"]["pbc"]
            self.mic = True
            self.fixed_frame.set_cell(self.pbc)
            self.fixed_frame.set_pbc(True)

    def get_len(self):
        """
        Returns the number of atoms in the fixed frame of the structure.

        This function returns the number of atoms in the fixed frame of the structure. 
        The fixed frame is an ASE Atoms object, so the number of atoms is the length of this object.

        Returns:
            int: The number of atoms in the fixed frame of the structure.
        """

        return len(self.fixed_frame)

    def set_fixed_frame_positions(self, structure, atoms):
        """
        Sets the positions of atoms in the fixed frame based on another Atoms object.

        This function takes a structure and an Atoms object, and sets the positions of atoms in the fixed frame of the structure 
        based on the positions of atoms in the Atoms object. 
        It assumes that the Atoms object and the fixed frame have the same number of atoms in the same order.

        Args:
            structure (Structure): The structure object.
            atoms (ase.Atoms): An ASE Atoms object.
        """

        len_mol = len(structure.molecules)
        l = len(structure.molecules[0])
        fixed_frame_atoms = atoms[(len_mol) * l :].get_positions()
        a = self.fixed_frame.get_positions()
        self.fixed_frame.set_positions(fixed_frame_atoms)
