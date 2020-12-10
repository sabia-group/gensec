from gensec.modules import *
from ase.io import read, write
from random import random, randint, uniform, choice


class Structure:

    def __init__(self, parameters):
        self.atoms = read(parameters["geometry"][0], 
                        format=parameters["geometry"][1])
        self.connectivity_matrix_full = create_connectivity_matrix(self.atoms, bothways=True) 
        self.connectivity_matrix_isolated = create_connectivity_matrix(self.atoms, bothways=False)
        
        if parameters["configuration"]["torsions"]["list_of_tosrions"]=="auto":
            self.list_of_torsions = detect_rotatble(self.connectivity_matrix_isolated, self.atoms)
        else:
            self.list_of_torsions = parameters["configuration"]["torsions"]["list_of_tosrions"]
        if parameters["mic"]["activate"] == True:
            self.pbc = parameters["mic"]["pbc"]
            self.mic = True
            self.atoms.set_cell(self.pbc)
            self.atoms.set_pbc(True)
        self.molecules = [self.atoms.copy() for i in range(parameters["number_of_replicas"])]

        self.cycles = detect_cycles(self.connectivity_matrix_full)
        self.list_of_torsions = exclude_rotatable_from_cycles(self.list_of_torsions, self.cycles)

        if len(self.cycles) > 0:
             for i in range(len(self.cycles)):
                self.cycles[i] = make_canonical_pyranosering(self.atoms, self.cycles[i])

    def create_configuration(self, parameters):

        def make_torsion(self, parameters):
            if parameters["configuration"]["torsions"]["values"] == "random":
                torsions = np.array([randint(0, 360) 
                                    for i in self.list_of_torsions])
            return torsions


        def make_orientation(self, parameters):
            if parameters["configuration"]["orientations"]["values"] == "random":
                quaternion = produce_quaternion(
                    randint(-180, 180), 
                    np.array([random(),
                              random(),
                              random()]))
            else:
                angle = parameters["configuration"]["orientations"]["angle"] 
                x = parameters["configuration"]["orientations"]["x"] 
                y = parameters["configuration"]["orientations"]["y"] 
                z = parameters["configuration"]["orientations"]["z"] 
                quaternion = produce_quaternion(
                    randint(angle[0], angle[1]), 
                    np.array([uniform(x[0], x[1]),
                              uniform(y[0], y[1]),
                              uniform(z[0], z[1])]))             
            return quaternion

        def make_com(self, parameters):
            if parameters["configuration"]["coms"]["values"] == "restricted":
                x = parameters["configuration"]["coms"]["x_axis"] 
                y = parameters["configuration"]["coms"]["y_axis"] 
                z = parameters["configuration"]["coms"]["z_axis"] 

                x_space = np.linspace(start=x[0], stop=x[1], num=x[2], endpoint=True)
                y_space = np.linspace(start=y[0], stop=y[1], num=y[2], endpoint=True)
                z_space = np.linspace(start=z[0], stop=z[1], num=z[2], endpoint=True)
                com = np.array([choice(x_space), 
                                 choice(y_space), 
                                 choice(z_space)])
            else:
                 com = np.array([choice(np.linspace(0, 10, 10)), 
                                 choice(np.linspace(0, 10, 10)), 
                                 choice(np.linspace(0, 10, 10))])

            return com

        if parameters["configuration"]["torsions"]["activate"]:
            torsions = make_torsion(self, parameters)
        if parameters["configuration"]["orientations"]["activate"]:
            quaternion = make_orientation(self, parameters)
        else:
            quaternion = [0,0,0,1]
        if parameters["configuration"]["coms"]["activate"]:
            coms = make_com(self, parameters)
        else:
            coms = [0,0,0]       
        if not any(parameters["configuration"][i]["activate"] for i in parameters["configuration"]):
            print("Nothing to sample")
            sys.exit(0)
        else:
            configuration = np.hstack((torsions, quaternion, coms))  
            print(configuration)


        if len(self.molecules) > 1:
            for i in range(len(self.molecules) -1):
                if parameters["configuration"]["torsions"]["same"]:
                    pass
                else:
                    torsions = make_torsion(self, parameters)

                if parameters["configuration"]["orientations"]["same"]:
                    pass
                else:
                    quaternion = make_orientation(self, parameters)
                
                if parameters["configuration"]["coms"]["same"]:
                    pass
                else:
                    coms = make_com(self, parameters)   
                vec = np.hstack((torsions, quaternion, coms))
                configuration = np.hstack((configuration, vec))
   
        return configuration


    def apply_configuration(self, configuration):
    # molecules, configuration, list_of_torsions, connectivity_matrix_isolated):
        for i in range(len(self.molecules)):
            k=-1
            for torsion in self.list_of_torsions:
                k+=1
                # +4 quaternion values and +3 COM values
                z=i*(len(self.list_of_torsions)+4+3)+k
                fixed_indices = carried_atoms(
                                self.connectivity_matrix_isolated, torsion)
                self.molecules[i].set_dihedral(angle=configuration[z],
                                          a1=torsion[0],
                                          a2=torsion[1],
                                          a3=torsion[2],
                                          a4=torsion[3],
                                          indices=fixed_indices)
            # Set orientation
            quaternion_set(self.molecules[i], 
                            produce_quaternion(configuration[z+1], 
                                                np.array([configuration[z+2],
                                                        configuration[z+3],
                                                        configuration[z+4]])),
                                                0, len(self.molecules[i])-1)
            # Set center of mass
            set_centre_of_mass(self.molecules[i], np.array([configuration[z+5], 
                                                            configuration[z+6], 
                                                            configuration[z+7]]))

    def apply_torsions(self, configuration):
    # molecules, configuration, list_of_torsions, connectivity_matrix_isolated):
        for i in range(len(self.molecules)):
            k=-1
            for torsion in self.list_of_torsions:
                k+=1
                # +4 quaternion values and +3 COM values
                z=i*(len(self.list_of_torsions)+4+3)+k
                fixed_indices = carried_atoms(
                                self.connectivity_matrix_isolated, torsion)
                self.molecules[i].set_dihedral(angle=configuration[z],
                                          a1=torsion[0],
                                          a2=torsion[1],
                                          a3=torsion[2],
                                          a4=torsion[3],
                                          indices=fixed_indices)


    def torsions_from_conf(self, configuration):
        torsions = []
        for i in range(len(self.molecules)):
            k=-1
            for torsion in self.list_of_torsions:
                k+=1
                # +4 quaternion values and +3 COM values
                z=i*(len(self.list_of_torsions)+4+3)+k
                torsions.append(configuration[z])
        return torsions       

    def read_configuration(self, structure, fixed_frame, gen):
        t = structure.list_of_torsions
        configuration = read(gen, format="aims")
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
        return torsions

class Fixed_frame:

    def __init__(self, parameters):

        # Minimum Image Conventio
        self.mic = False
        if parameters["fixed_frame"]["activate"] == True:
            self.fixed_frame = read(parameters["fixed_frame"]["filename"], 
                                format=parameters["fixed_frame"]["format"])

        if parameters["mic"]["activate"] == True:
            self.pbc = parameters["mic"]["pbc"]
            self.mic = True
            self.fixed_frame.set_cell(self.pbc)
            self.fixed_frame.set_pbc(True)

    def get_len(self):
        return len(self.fixed_frame)
